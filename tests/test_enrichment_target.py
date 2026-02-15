"""Tests for the Target enrichment provider.

Tests cover:
- Order-to-transaction matching algorithm (date proximity, amount matching,
  RedCard discount, gift card skipping)
- Enrichment cache file reading and writing
- Date and price parsing helpers
- CLI ``expense enrich --source target`` command (with mocked Playwright)

No actual Target.com scraping is performed -- all browser interactions
are mocked.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from expense_tracker.cli import cli
from expense_tracker.enrichment.target import (
    AMOUNT_TOLERANCE,
    DATE_MATCH_WINDOW_DAYS,
    REDCARD_DISCOUNT_FACTOR,
    TargetLineItem,
    TargetOrder,
    _parse_price,
    _parse_target_date,
    match_orders_to_transactions,
    read_enrichment_cache,
    write_enrichment_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_order(
    order_id: str = "ORD-001",
    order_date: date = date(2026, 1, 25),
    order_total: Decimal = Decimal("127.98"),
    items: list[TargetLineItem] | None = None,
    fulfillment_type: str = "shipped",
    payment_method: str = "credit",
) -> TargetOrder:
    """Build a TargetOrder for testing."""
    if items is None:
        items = [
            TargetLineItem(name="Diapers", price=Decimal("24.99"), quantity=2),
            TargetLineItem(name="Baby Wipes", price=Decimal("6.49"), quantity=1),
        ]
    return TargetOrder(
        order_id=order_id,
        order_date=order_date,
        order_total=order_total,
        items=items,
        fulfillment_type=fulfillment_type,
        payment_method=payment_method,
    )


def _make_txn_dict(
    transaction_id: str = "abc123def456",
    txn_date: date = date(2026, 1, 25),
    amount: Decimal = Decimal("-127.98"),
    merchant: str = "TARGET 00022186",
) -> dict:
    """Build a bank transaction dict for matching tests."""
    return {
        "transaction_id": transaction_id,
        "date": txn_date.isoformat(),
        "amount": str(amount),
        "merchant": merchant,
    }


# ===========================================================================
# Matching algorithm tests
# ===========================================================================


class TestMatchOrdersToTransactions:
    """Tests for match_orders_to_transactions."""

    def test_exact_amount_same_day(self):
        """Order and transaction with same amount on same day should match."""
        order = _make_order(order_total=Decimal("50.00"))
        txn = _make_txn_dict(amount=Decimal("-50.00"), txn_date=order.order_date)

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1
        assert matches[0][0] is order
        assert matches[0][1]["transaction_id"] == txn["transaction_id"]

    def test_exact_amount_within_date_window(self):
        """Match within the default 3-day date window."""
        order = _make_order(
            order_date=date(2026, 1, 25),
            order_total=Decimal("100.00"),
        )
        txn = _make_txn_dict(
            amount=Decimal("-100.00"),
            txn_date=date(2026, 1, 27),  # 2 days later
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1

    def test_date_window_boundary_exact(self):
        """Match at exactly the boundary of the date window."""
        order = _make_order(
            order_date=date(2026, 1, 25),
            order_total=Decimal("75.00"),
        )
        txn = _make_txn_dict(
            amount=Decimal("-75.00"),
            txn_date=date(2026, 1, 28),  # Exactly 3 days
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1

    def test_outside_date_window_no_match(self):
        """No match when transaction is beyond the date window."""
        order = _make_order(
            order_date=date(2026, 1, 25),
            order_total=Decimal("75.00"),
        )
        txn = _make_txn_dict(
            amount=Decimal("-75.00"),
            txn_date=date(2026, 1, 29),  # 4 days later, beyond 3-day window
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 0

    def test_amount_mismatch_no_match(self):
        """No match when amounts differ and don't match RedCard discount."""
        order = _make_order(order_total=Decimal("100.00"))
        txn = _make_txn_dict(
            amount=Decimal("-80.00"),  # Not 100 and not 95 (RedCard)
            txn_date=order.order_date,
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 0

    def test_redcard_discount_match(self):
        """RedCard 5% discount: bank shows 95% of the order total."""
        order = _make_order(order_total=Decimal("100.00"))
        # RedCard discount: $100.00 * 0.95 = $95.00
        txn = _make_txn_dict(
            amount=Decimal("-95.00"),
            txn_date=order.order_date,
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1

    def test_redcard_discount_with_cents(self):
        """RedCard discount with rounding: $127.98 * 0.95 = $121.58."""
        order = _make_order(order_total=Decimal("127.98"))
        redcard_amount = (Decimal("127.98") * REDCARD_DISCOUNT_FACTOR).quantize(
            Decimal("0.01")
        )
        txn = _make_txn_dict(
            amount=-redcard_amount,
            txn_date=order.order_date,
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1

    def test_gift_card_order_skipped(self):
        """Orders paid with gift card should be skipped entirely."""
        order = _make_order(payment_method="gift_card")
        txn = _make_txn_dict(
            amount=-order.order_total,
            txn_date=order.order_date,
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 0

    def test_gift_card_partial_mention_skipped(self):
        """Payment method containing 'gift' should be treated as gift card."""
        order = _make_order(payment_method="Gift Card ending in 1234")
        txn = _make_txn_dict(
            amount=-order.order_total,
            txn_date=order.order_date,
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 0

    def test_empty_orders_list(self):
        """No matches when orders list is empty."""
        txn = _make_txn_dict()
        matches = match_orders_to_transactions([], [txn])
        assert matches == []

    def test_empty_transactions_list(self):
        """No matches when transactions list is empty."""
        order = _make_order()
        matches = match_orders_to_transactions([order], [])
        assert matches == []

    def test_both_empty(self):
        """No matches when both lists are empty."""
        matches = match_orders_to_transactions([], [])
        assert matches == []

    def test_one_order_matches_only_one_transaction(self):
        """An order should match at most one transaction."""
        order = _make_order(order_total=Decimal("50.00"))
        txn1 = _make_txn_dict(
            transaction_id="txn_1",
            amount=Decimal("-50.00"),
            txn_date=order.order_date,
        )
        txn2 = _make_txn_dict(
            transaction_id="txn_2",
            amount=Decimal("-50.00"),
            txn_date=order.order_date,
        )

        matches = match_orders_to_transactions([order], [txn1, txn2])
        assert len(matches) == 1

    def test_transaction_matched_only_once(self):
        """A transaction should match at most one order."""
        order1 = _make_order(
            order_id="ORD-001",
            order_total=Decimal("50.00"),
            order_date=date(2026, 1, 25),
        )
        order2 = _make_order(
            order_id="ORD-002",
            order_total=Decimal("50.00"),
            order_date=date(2026, 1, 25),
        )
        txn = _make_txn_dict(
            transaction_id="single_txn",
            amount=Decimal("-50.00"),
            txn_date=date(2026, 1, 25),
        )

        matches = match_orders_to_transactions([order1, order2], [txn])
        assert len(matches) == 1
        # The first order (by date sort) gets the match
        matched_txn_ids = {m[1]["transaction_id"] for m in matches}
        assert "single_txn" in matched_txn_ids

    def test_closest_date_preferred(self):
        """When multiple transactions could match, the closest by date wins."""
        order = _make_order(
            order_total=Decimal("50.00"),
            order_date=date(2026, 1, 25),
        )
        txn_far = _make_txn_dict(
            transaction_id="far_txn",
            amount=Decimal("-50.00"),
            txn_date=date(2026, 1, 28),  # 3 days away
        )
        txn_close = _make_txn_dict(
            transaction_id="close_txn",
            amount=Decimal("-50.00"),
            txn_date=date(2026, 1, 25),  # Same day
        )

        matches = match_orders_to_transactions([order], [txn_far, txn_close])
        assert len(matches) == 1
        assert matches[0][1]["transaction_id"] == "close_txn"

    def test_multiple_orders_multiple_transactions(self):
        """Multiple orders match to multiple distinct transactions."""
        order1 = _make_order(
            order_id="ORD-001",
            order_total=Decimal("50.00"),
            order_date=date(2026, 1, 20),
        )
        order2 = _make_order(
            order_id="ORD-002",
            order_total=Decimal("75.00"),
            order_date=date(2026, 1, 25),
        )
        txn1 = _make_txn_dict(
            transaction_id="txn_50",
            amount=Decimal("-50.00"),
            txn_date=date(2026, 1, 20),
        )
        txn2 = _make_txn_dict(
            transaction_id="txn_75",
            amount=Decimal("-75.00"),
            txn_date=date(2026, 1, 26),
        )

        matches = match_orders_to_transactions(
            [order1, order2], [txn1, txn2]
        )
        assert len(matches) == 2
        matched_order_ids = {m[0].order_id for m in matches}
        assert matched_order_ids == {"ORD-001", "ORD-002"}

    def test_custom_date_window(self):
        """Custom date window is respected."""
        order = _make_order(
            order_total=Decimal("50.00"),
            order_date=date(2026, 1, 25),
        )
        txn = _make_txn_dict(
            amount=Decimal("-50.00"),
            txn_date=date(2026, 1, 30),  # 5 days later
        )

        # Default 3-day window: no match
        matches_default = match_orders_to_transactions([order], [txn])
        assert len(matches_default) == 0

        # Custom 5-day window: match
        matches_custom = match_orders_to_transactions([order], [txn], date_window=5)
        assert len(matches_custom) == 1

    def test_date_as_date_object(self):
        """Transaction date can be a date object instead of ISO string."""
        order = _make_order(order_total=Decimal("50.00"))
        txn = {
            "transaction_id": "date_obj_txn",
            "date": order.order_date,  # date object, not string
            "amount": "-50.00",
            "merchant": "TARGET",
        }

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1

    def test_tolerance_boundary(self):
        """Amount within AMOUNT_TOLERANCE matches, just outside does not."""
        order = _make_order(order_total=Decimal("100.00"))

        # Within tolerance ($0.02)
        txn_within = _make_txn_dict(
            transaction_id="within",
            amount=Decimal("-100.02"),
            txn_date=order.order_date,
        )
        matches = match_orders_to_transactions([order], [txn_within])
        assert len(matches) == 1

        # Just outside tolerance
        txn_outside = _make_txn_dict(
            transaction_id="outside",
            amount=Decimal("-100.03"),
            txn_date=order.order_date,
        )
        matches = match_orders_to_transactions([order], [txn_outside])
        assert len(matches) == 0

    def test_transaction_before_order_date(self):
        """Transaction posting before order date (within window) still matches."""
        order = _make_order(
            order_total=Decimal("50.00"),
            order_date=date(2026, 1, 25),
        )
        txn = _make_txn_dict(
            amount=Decimal("-50.00"),
            txn_date=date(2026, 1, 23),  # 2 days before order
        )

        matches = match_orders_to_transactions([order], [txn])
        assert len(matches) == 1


# ===========================================================================
# Cache file I/O tests
# ===========================================================================


class TestWriteEnrichmentCache:
    """Tests for write_enrichment_cache."""

    def test_writes_valid_json(self, tmp_path: Path):
        """Cache file is valid JSON with expected structure."""
        order = _make_order(
            order_id="ORD-123",
            order_total=Decimal("56.47"),
            items=[
                TargetLineItem(name="Diapers", price=Decimal("24.99"), quantity=2),
                TargetLineItem(name="Baby Wipes", price=Decimal("6.49"), quantity=1),
            ],
        )
        cache_dir = tmp_path / "enrichment-cache"

        path = write_enrichment_cache(order, "txn_abc123", cache_dir)

        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source"] == "target"
        assert data["order_id"] == "ORD-123"
        assert "items" in data
        assert len(data["items"]) >= 2  # items + possible tax adjustment

    def test_creates_cache_dir_if_missing(self, tmp_path: Path):
        """Cache directory is created automatically if it doesn't exist."""
        cache_dir = tmp_path / "new" / "cache" / "dir"
        assert not cache_dir.exists()

        order = _make_order()
        write_enrichment_cache(order, "txn_abc", cache_dir)

        assert cache_dir.is_dir()

    def test_filename_matches_transaction_id(self, tmp_path: Path):
        """Cache file is named ``{transaction_id}.json``."""
        order = _make_order()
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "my_txn_id_123", cache_dir)

        assert path.name == "my_txn_id_123.json"

    def test_items_have_required_keys(self, tmp_path: Path):
        """Each item in the cache file has merchant, description, and amount."""
        order = _make_order(
            items=[TargetLineItem(name="Soap", price=Decimal("4.99"))],
            order_total=Decimal("5.33"),  # includes tax
        )
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "txn_soap", cache_dir)
        data = json.loads(path.read_text(encoding="utf-8"))

        for item in data["items"]:
            assert "merchant" in item
            assert "description" in item
            assert "amount" in item

    def test_amounts_are_negative(self, tmp_path: Path):
        """Item amounts in the cache are negative (expenses)."""
        order = _make_order(
            items=[TargetLineItem(name="Soap", price=Decimal("4.99"))],
            order_total=Decimal("4.99"),
        )
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "txn_neg", cache_dir)
        data = json.loads(path.read_text(encoding="utf-8"))

        for item in data["items"]:
            assert Decimal(item["amount"]) < 0

    def test_tax_adjustment_line_added(self, tmp_path: Path):
        """A tax/adjustments line is added when items don't sum to order total."""
        order = _make_order(
            items=[TargetLineItem(name="Shirt", price=Decimal("20.00"))],
            order_total=Decimal("21.40"),  # $1.40 tax
        )
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "txn_tax", cache_dir)
        data = json.loads(path.read_text(encoding="utf-8"))

        assert len(data["items"]) == 2
        tax_item = data["items"][-1]
        assert "Tax" in tax_item["merchant"] or "Adjustment" in tax_item["merchant"]
        assert Decimal(tax_item["amount"]) == Decimal("-1.40")

    def test_no_tax_line_when_items_sum_to_total(self, tmp_path: Path):
        """No tax line when items exactly sum to the order total."""
        order = _make_order(
            items=[
                TargetLineItem(name="Item A", price=Decimal("30.00")),
                TargetLineItem(name="Item B", price=Decimal("20.00")),
            ],
            order_total=Decimal("50.00"),
        )
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "txn_exact", cache_dir)
        data = json.loads(path.read_text(encoding="utf-8"))

        assert len(data["items"]) == 2

    def test_quantity_multiplied_in_amount(self, tmp_path: Path):
        """Item amount reflects quantity (price * quantity)."""
        order = _make_order(
            items=[TargetLineItem(name="Socks", price=Decimal("5.00"), quantity=3)],
            order_total=Decimal("15.00"),
        )
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "txn_qty", cache_dir)
        data = json.loads(path.read_text(encoding="utf-8"))

        # Socks: $5.00 * 3 = $15.00 -> -15.00
        assert Decimal(data["items"][0]["amount"]) == Decimal("-15.00")

    def test_quantity_in_description(self, tmp_path: Path):
        """Multi-quantity items show qty in description."""
        order = _make_order(
            items=[TargetLineItem(name="Socks", price=Decimal("5.00"), quantity=3)],
            order_total=Decimal("15.00"),
        )
        cache_dir = tmp_path / "cache"

        path = write_enrichment_cache(order, "txn_qdesc", cache_dir)
        data = json.loads(path.read_text(encoding="utf-8"))

        assert "qty 3" in data["items"][0]["description"]


class TestReadEnrichmentCache:
    """Tests for read_enrichment_cache."""

    def test_reads_valid_file(self, tmp_path: Path):
        """Reads and parses a valid JSON cache file."""
        cache_file = tmp_path / "txn_123.json"
        data = {"source": "target", "items": [{"merchant": "Target", "amount": "-10.00"}]}
        cache_file.write_text(json.dumps(data), encoding="utf-8")

        result = read_enrichment_cache(cache_file)
        assert result is not None
        assert result["source"] == "target"
        assert len(result["items"]) == 1

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        """Returns None when the cache file doesn't exist."""
        result = read_enrichment_cache(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path):
        """Returns None when the file contains invalid JSON."""
        cache_file = tmp_path / "bad.json"
        cache_file.write_text("not valid json {{{", encoding="utf-8")

        result = read_enrichment_cache(cache_file)
        assert result is None

    def test_roundtrip_write_read(self, tmp_path: Path):
        """Data written by write_enrichment_cache can be read back."""
        order = _make_order(
            order_id="RT-001",
            items=[TargetLineItem(name="Towel", price=Decimal("12.99"))],
            order_total=Decimal("12.99"),
        )
        cache_dir = tmp_path / "cache"
        path = write_enrichment_cache(order, "rt_txn", cache_dir)

        data = read_enrichment_cache(path)
        assert data is not None
        assert data["order_id"] == "RT-001"
        assert data["items"][0]["merchant"] == "Target - Towel"


# ===========================================================================
# Date parsing tests
# ===========================================================================


class TestParseDateHelper:
    """Tests for _parse_target_date."""

    def test_full_month_name(self):
        """Parses 'January 15, 2026'."""
        result = _parse_target_date("January 15, 2026")
        assert result == date(2026, 1, 15)

    def test_abbreviated_month(self):
        """Parses 'Jan 15, 2026'."""
        result = _parse_target_date("Jan 15, 2026")
        assert result == date(2026, 1, 15)

    def test_slash_format(self):
        """Parses '1/15/2026'."""
        result = _parse_target_date("1/15/2026")
        assert result == date(2026, 1, 15)

    def test_slash_format_zero_padded(self):
        """Parses '01/15/2026'."""
        result = _parse_target_date("01/15/2026")
        assert result == date(2026, 1, 15)

    def test_iso_format(self):
        """Parses '2026-01-15'."""
        result = _parse_target_date("2026-01-15")
        assert result == date(2026, 1, 15)

    def test_with_ordered_prefix(self):
        """Strips 'Ordered: ' prefix before parsing."""
        result = _parse_target_date("Ordered: January 15, 2026")
        assert result == date(2026, 1, 15)

    def test_with_order_placed_prefix(self):
        """Strips 'Order placed' prefix."""
        result = _parse_target_date("Order placed January 15, 2026")
        assert result == date(2026, 1, 15)

    def test_with_whitespace(self):
        """Handles leading/trailing whitespace."""
        result = _parse_target_date("  January 15, 2026  ")
        assert result == date(2026, 1, 15)

    def test_unparseable_returns_none(self):
        """Returns None for unparseable strings."""
        assert _parse_target_date("not a date") is None
        assert _parse_target_date("") is None

    def test_december(self):
        """Parses December correctly (month boundary)."""
        result = _parse_target_date("December 31, 2025")
        assert result == date(2025, 12, 31)


# ===========================================================================
# Price parsing tests
# ===========================================================================


class TestParsePriceHelper:
    """Tests for _parse_price."""

    def test_dollar_sign(self):
        """Parses '$127.98'."""
        assert _parse_price("$127.98") == Decimal("127.98")

    def test_no_dollar_sign(self):
        """Parses '127.98'."""
        assert _parse_price("127.98") == Decimal("127.98")

    def test_comma_thousands(self):
        """Parses '$1,234.56'."""
        assert _parse_price("$1,234.56") == Decimal("1234.56")

    def test_whitespace(self):
        """Handles surrounding whitespace."""
        assert _parse_price("  $42.00  ") == Decimal("42.00")

    def test_empty_string(self):
        """Returns Decimal('0') for empty string."""
        assert _parse_price("") == Decimal("0")

    def test_always_positive(self):
        """Result is always positive even if input has a minus sign."""
        result = _parse_price("-$50.00")
        assert result >= 0


# ===========================================================================
# TargetOrder model tests
# ===========================================================================


class TestTargetOrderModel:
    """Tests for TargetOrder dataclass behavior."""

    def test_has_gift_card_payment_true(self):
        """has_gift_card_payment returns True for gift card payment."""
        order = _make_order(payment_method="gift_card")
        assert order.has_gift_card_payment is True

    def test_has_gift_card_payment_mixed(self):
        """has_gift_card_payment returns True if 'gift' appears in method."""
        order = _make_order(payment_method="Gift Card + Debit")
        assert order.has_gift_card_payment is True

    def test_has_gift_card_payment_false(self):
        """has_gift_card_payment returns False for non-gift-card payment."""
        order = _make_order(payment_method="credit")
        assert order.has_gift_card_payment is False

    def test_has_gift_card_payment_empty(self):
        """has_gift_card_payment returns False for empty payment method."""
        order = _make_order(payment_method="")
        assert order.has_gift_card_payment is False


# ===========================================================================
# CLI command tests
# ===========================================================================


class TestEnrichCommand:
    """Tests for the ``expense enrich`` CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_enrich_help(self, runner: CliRunner):
        """Enrich command shows help text with expected options."""
        result = runner.invoke(cli, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "--month" in result.output
        assert "--source" in result.output
        assert "--headless" in result.output
        assert "target" in result.output.lower()

    def test_enrich_missing_month(self, runner: CliRunner):
        """Enrich without --month fails."""
        result = runner.invoke(cli, ["enrich", "--source", "target"])
        assert result.exit_code != 0

    def test_enrich_missing_source(self, runner: CliRunner):
        """Enrich without --source fails."""
        result = runner.invoke(cli, ["enrich", "--month", "2026-01"])
        assert result.exit_code != 0

    def test_enrich_invalid_source(self, runner: CliRunner):
        """Enrich with unknown source fails."""
        result = runner.invoke(
            cli, ["enrich", "--month", "2026-01", "--source", "walmart"]
        )
        assert result.exit_code != 0

    def test_enrich_invalid_month_format(self, runner: CliRunner):
        """Enrich with invalid month format fails."""
        result = runner.invoke(
            cli, ["enrich", "--month", "2026-1", "--source", "target"]
        )
        assert result.exit_code != 0

    @patch("expense_tracker.enrichment.target.enrich_target")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_enrich_target_success(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich_target: MagicMock,
        runner: CliRunner,
    ):
        """Successful target enrichment prints summary."""
        from expense_tracker.models import AppConfig, PipelineResult, Transaction

        mock_load_config.return_value = AppConfig(
            enrichment_cache_dir="enrichment-cache"
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []

        # Pipeline returns a transaction that looks like a Target purchase
        txn = Transaction(
            transaction_id="abc123",
            date=date(2026, 1, 25),
            merchant="TARGET 00022186",
            description="TARGET 00022186",
            amount=Decimal("-127.98"),
            institution="chase",
            account="Chase CC",
        )
        mock_pipeline_run.return_value = PipelineResult(transactions=[txn])

        mock_enrich_target.return_value = {
            "orders_scraped": 3,
            "orders_matched": 1,
            "cache_files_written": 1,
            "skipped_gift_card": 0,
        }

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2026-01", "--source", "target"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Target Enrichment Summary" in result.output
        assert "Orders scraped" in result.output
        assert "3" in result.output
        assert "Orders matched" in result.output
        mock_enrich_target.assert_called_once()

    @patch("expense_tracker.enrichment.target.enrich_target")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_enrich_target_import_error(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich_target: MagicMock,
        runner: CliRunner,
    ):
        """ImportError from missing playwright is shown as user-facing error."""
        from expense_tracker.models import AppConfig, PipelineResult

        mock_load_config.return_value = AppConfig(
            enrichment_cache_dir="enrichment-cache"
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []
        mock_pipeline_run.return_value = PipelineResult(transactions=[])

        mock_enrich_target.side_effect = ImportError(
            "playwright is required for Target enrichment"
        )

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2026-01", "--source", "target"],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "playwright" in result.output

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_enrich_amazon_source_accepted(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_amazon_provider_cls: MagicMock,
        runner: CliRunner,
    ):
        """Amazon source is accepted and dispatched to the Amazon provider."""
        from expense_tracker.enrichment import EnrichmentResult
        from expense_tracker.models import AppConfig, PipelineResult

        mock_load_config.return_value = AppConfig(enrichment_cache_dir="enrichment-cache")
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []
        mock_pipeline_run.return_value = PipelineResult(transactions=[])

        mock_provider = MagicMock()
        mock_provider.enrich.return_value = EnrichmentResult(
            orders_found=0, orders_matched=0, cache_files_written=0,
        )
        mock_amazon_provider_cls.return_value = mock_provider

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2026-01", "--source", "amazon"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Amazon Enrichment Summary" in result.output

    @patch("expense_tracker.enrichment.target.enrich_target")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_enrich_target_with_gift_card_skipped(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich_target: MagicMock,
        runner: CliRunner,
    ):
        """Gift card skip count is shown when > 0."""
        from expense_tracker.models import AppConfig, PipelineResult

        mock_load_config.return_value = AppConfig(
            enrichment_cache_dir="enrichment-cache"
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []
        mock_pipeline_run.return_value = PipelineResult(transactions=[])

        mock_enrich_target.return_value = {
            "orders_scraped": 5,
            "orders_matched": 2,
            "cache_files_written": 2,
            "skipped_gift_card": 1,
        }

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2026-01", "--source", "target"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Gift card" in result.output
        assert "1" in result.output


# ===========================================================================
# Integration: cache files work with pipeline enrich stage
# ===========================================================================


class TestCacheIntegrationWithPipeline:
    """Verify that cache files produced by target enrichment are compatible
    with the pipeline's _enrich stage."""

    def test_cache_file_consumed_by_pipeline_enrich(self, tmp_path: Path):
        """A cache file from write_enrichment_cache is consumed by _enrich."""
        from expense_tracker.models import AppConfig, Transaction
        from expense_tracker.pipeline import _enrich

        cache_dir = tmp_path / "enrichment-cache"
        config = AppConfig(enrichment_cache_dir="enrichment-cache")

        # Create a transaction
        txn = Transaction(
            transaction_id="integration_test_txn",
            date=date(2026, 1, 25),
            merchant="TARGET 00022186",
            description="TARGET 00022186",
            amount=Decimal("-56.47"),
            institution="chase",
            account="Chase CC",
        )

        # Write a cache file using the target enrichment module
        order = _make_order(
            order_id="INT-001",
            order_total=Decimal("56.47"),
            items=[
                TargetLineItem(name="Diapers", price=Decimal("24.99"), quantity=2),
                TargetLineItem(name="Baby Wipes", price=Decimal("6.49"), quantity=1),
            ],
        )
        write_enrichment_cache(order, txn.transaction_id, cache_dir)

        # Run the pipeline's enrich stage
        result = _enrich([txn], tmp_path, config)

        # Should have split the transaction into items + tax adjustment
        assert len(result.transactions) > 1
        assert all(t.split_from == txn.transaction_id for t in result.transactions)

        # All split amounts should be negative (expenses)
        for split_txn in result.transactions:
            assert split_txn.amount < 0

        # Sum of splits should equal original amount (within tolerance)
        split_sum = sum(t.amount for t in result.transactions)
        assert abs(split_sum - txn.amount) <= Decimal("0.01")
