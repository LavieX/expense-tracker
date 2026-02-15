"""Tests for Amazon enrichment provider.

Tests the matching algorithm, cache file I/O, the enrichment data building,
and the CLI ``expense enrich --source amazon`` command.

Does NOT test actual Amazon scraping (that requires real credentials and
a live browser).
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from expense_tracker.cli import cli
from expense_tracker.enrichment.amazon import (
    AMOUNT_TOLERANCE,
    DATE_PROXIMITY_DAYS,
    AmazonLineItem,
    AmazonOrder,
    build_enrichment_data,
    match_orders_to_transactions,
    _parse_date,
    _parse_price,
)
from expense_tracker.enrichment.cache import (
    EnrichmentData,
    EnrichmentItem,
    list_cache_files,
    read_cache_file,
    write_cache_file,
)


# ===========================================================================
# Fixtures
# ===========================================================================

FIXTURES_CONFIG_DIR = Path(__file__).parent / "fixtures" / "config"


@pytest.fixture
def sample_orders() -> list[AmazonOrder]:
    """Sample Amazon orders for testing the matching algorithm."""
    return [
        AmazonOrder(
            order_id="111-1111111-1111111",
            order_date=date(2025, 11, 5),
            order_total=Decimal("105.00"),
            items=[
                AmazonLineItem(name="Widget", price=Decimal("30.00")),
                AmazonLineItem(name="Book", price=Decimal("15.00")),
                AmazonLineItem(name="Charger", price=Decimal("60.00")),
            ],
        ),
        AmazonOrder(
            order_id="222-2222222-2222222",
            order_date=date(2025, 11, 12),
            order_total=Decimal("42.99"),
            items=[
                AmazonLineItem(name="USB Cable", price=Decimal("12.99")),
                AmazonLineItem(name="Phone Case", price=Decimal("30.00")),
            ],
        ),
        AmazonOrder(
            order_id="333-3333333-3333333",
            order_date=date(2025, 11, 20),
            order_total=Decimal("89.50"),
            items=[
                AmazonLineItem(name="Headphones", price=Decimal("89.50")),
            ],
        ),
    ]


@pytest.fixture
def sample_transactions() -> list[dict]:
    """Sample bank transactions for matching against Amazon orders."""
    return [
        {
            "transaction_id": "txn_001",
            "date": date(2025, 11, 6),
            "amount": Decimal("-105.00"),
            "merchant": "AMAZON.COM*AB1CD2EF3",
        },
        {
            "transaction_id": "txn_002",
            "date": date(2025, 11, 13),
            "amount": Decimal("-42.99"),
            "merchant": "AMZN MKTP US*ZZ0XX1YY2",
        },
        {
            "transaction_id": "txn_003",
            "date": date(2025, 11, 21),
            "amount": Decimal("-89.50"),
            "merchant": "AMAZON.COM*GH3IJ4KL5",
        },
        {
            "transaction_id": "txn_004",
            "date": date(2025, 11, 25),
            "amount": Decimal("-15.00"),
            "merchant": "SPOTIFY USA",
        },
    ]


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli_project_dir(tmp_path: Path) -> Path:
    """Temporary directory with valid config files for CLI tests."""
    project = tmp_path / "project"
    project.mkdir()

    for config_file in ("config.toml", "categories.toml", "rules.toml"):
        shutil.copy2(FIXTURES_CONFIG_DIR / config_file, project / config_file)

    (project / "input" / "chase").mkdir(parents=True)
    (project / "input" / "capital-one").mkdir(parents=True)
    (project / "input" / "elevations").mkdir(parents=True)
    (project / "output").mkdir()
    (project / "enrichment-cache").mkdir()

    return project


# ===========================================================================
# Matching algorithm tests
# ===========================================================================


class TestMatchOrders:
    """Tests for match_orders_to_transactions."""

    def test_exact_match_by_date_and_amount(
        self, sample_orders: list[AmazonOrder], sample_transactions: list[dict]
    ) -> None:
        """Orders within date window and matching amount are matched."""
        matches = match_orders_to_transactions(sample_orders, sample_transactions)
        assert len(matches) == 3

        matched_order_ids = {o.order_id for o, _ in matches}
        assert "111-1111111-1111111" in matched_order_ids
        assert "222-2222222-2222222" in matched_order_ids
        assert "333-3333333-3333333" in matched_order_ids

    def test_date_proximity_within_window(self) -> None:
        """An order 3 days before the transaction matches (within window)."""
        orders = [
            AmazonOrder(
                order_id="test-order",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Item", price=Decimal("50.00"))],
            )
        ]
        txns = [
            {
                "transaction_id": "txn_test",
                "date": date(2025, 11, 10 + DATE_PROXIMITY_DAYS),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            }
        ]
        matches = match_orders_to_transactions(orders, txns)
        assert len(matches) == 1

    def test_date_proximity_outside_window(self) -> None:
        """An order more than 3 days from the transaction does not match."""
        orders = [
            AmazonOrder(
                order_id="test-order",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Item", price=Decimal("50.00"))],
            )
        ]
        txns = [
            {
                "transaction_id": "txn_test",
                "date": date(2025, 11, 10 + DATE_PROXIMITY_DAYS + 1),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            }
        ]
        matches = match_orders_to_transactions(orders, txns)
        assert len(matches) == 0

    def test_amount_tolerance(self) -> None:
        """Orders within $0.01 of the transaction amount are matched."""
        orders = [
            AmazonOrder(
                order_id="test-order",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.01"),
                items=[AmazonLineItem(name="Item", price=Decimal("50.01"))],
            )
        ]
        txns = [
            {
                "transaction_id": "txn_test",
                "date": date(2025, 11, 10),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            }
        ]
        matches = match_orders_to_transactions(orders, txns)
        assert len(matches) == 1

    def test_amount_outside_tolerance(self) -> None:
        """Orders more than $0.01 from the transaction do not match."""
        orders = [
            AmazonOrder(
                order_id="test-order",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.02"),
                items=[AmazonLineItem(name="Item", price=Decimal("50.02"))],
            )
        ]
        txns = [
            {
                "transaction_id": "txn_test",
                "date": date(2025, 11, 10),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            }
        ]
        matches = match_orders_to_transactions(orders, txns)
        assert len(matches) == 0

    def test_ambiguous_match_skipped(self) -> None:
        """When multiple transactions could match one order, it is skipped."""
        orders = [
            AmazonOrder(
                order_id="test-order",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Item", price=Decimal("50.00"))],
            )
        ]
        txns = [
            {
                "transaction_id": "txn_1",
                "date": date(2025, 11, 10),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            },
            {
                "transaction_id": "txn_2",
                "date": date(2025, 11, 11),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            },
        ]
        matches = match_orders_to_transactions(orders, txns)
        assert len(matches) == 0

    def test_each_transaction_matched_once(self) -> None:
        """A transaction can only be matched to one order."""
        orders = [
            AmazonOrder(
                order_id="order-1",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Item A", price=Decimal("50.00"))],
            ),
            AmazonOrder(
                order_id="order-2",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Item B", price=Decimal("50.00"))],
            ),
        ]
        txns = [
            {
                "transaction_id": "txn_1",
                "date": date(2025, 11, 10),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            }
        ]
        matches = match_orders_to_transactions(orders, txns)
        # Only one order can match -- and even that is ambiguous since both
        # orders have the same amount and date, so neither matches.
        assert len(matches) == 0

    def test_empty_orders(self) -> None:
        """Empty order list produces no matches."""
        txns = [
            {
                "transaction_id": "txn_1",
                "date": date(2025, 11, 10),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON",
            }
        ]
        matches = match_orders_to_transactions([], txns)
        assert len(matches) == 0

    def test_empty_transactions(self) -> None:
        """Empty transaction list produces no matches."""
        orders = [
            AmazonOrder(
                order_id="test",
                order_date=date(2025, 11, 10),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Item", price=Decimal("50.00"))],
            )
        ]
        matches = match_orders_to_transactions(orders, [])
        assert len(matches) == 0

    def test_positive_transaction_amount_matches(self) -> None:
        """Matching uses absolute value, so positive (refund) amounts can match."""
        orders = [
            AmazonOrder(
                order_id="refund-order",
                order_date=date(2025, 11, 10),
                order_total=Decimal("25.00"),
                items=[AmazonLineItem(name="Refund Item", price=Decimal("25.00"))],
            )
        ]
        txns = [
            {
                "transaction_id": "txn_refund",
                "date": date(2025, 11, 11),
                "amount": Decimal("25.00"),
                "merchant": "AMAZON REFUND",
            }
        ]
        matches = match_orders_to_transactions(orders, txns)
        assert len(matches) == 1


# ===========================================================================
# Cache file I/O tests
# ===========================================================================


class TestCacheIO:
    """Tests for enrichment cache read/write operations."""

    def test_write_and_read_cache(self, tmp_path: Path) -> None:
        """Write a cache file and read it back."""
        cache_dir = tmp_path / "enrichment-cache"

        data = EnrichmentData(
            transaction_id="abc123def456",
            source="amazon",
            order_id="111-1111111-1111111",
            items=[
                EnrichmentItem(
                    name="Widget",
                    price=30.00,
                    quantity=1,
                    category_hint="Electronics",
                    merchant="AMAZON - Widget",
                    description="Widget",
                    amount="-30.00",
                ),
                EnrichmentItem(
                    name="Book",
                    price=15.00,
                    quantity=1,
                    merchant="AMAZON - Book",
                    description="Book",
                    amount="-15.00",
                ),
            ],
        )

        written_path = write_cache_file(cache_dir, data)
        assert written_path.is_file()
        assert written_path.name == "abc123def456.json"

        # Read back.
        result = read_cache_file(written_path)
        assert result is not None
        assert result.transaction_id == "abc123def456"
        assert result.source == "amazon"
        assert result.order_id == "111-1111111-1111111"
        assert len(result.items) == 2
        assert result.items[0].name == "Widget"
        assert result.items[0].price == 30.00
        assert result.items[1].name == "Book"
        assert result.matched_at != ""  # Should be set automatically.

    def test_write_creates_directory(self, tmp_path: Path) -> None:
        """write_cache_file creates the cache directory if it does not exist."""
        cache_dir = tmp_path / "nested" / "enrichment-cache"
        assert not cache_dir.exists()

        data = EnrichmentData(
            transaction_id="test123",
            source="amazon",
            items=[],
        )
        write_cache_file(cache_dir, data)
        assert cache_dir.is_dir()

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """read_cache_file returns None for a nonexistent file."""
        result = read_cache_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_read_invalid_json(self, tmp_path: Path) -> None:
        """read_cache_file returns None for invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json", encoding="utf-8")
        result = read_cache_file(bad_file)
        assert result is None

    def test_list_cache_files(self, tmp_path: Path) -> None:
        """list_cache_files returns sorted JSON files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "bbb.json").write_text("{}", encoding="utf-8")
        (cache_dir / "aaa.json").write_text("{}", encoding="utf-8")
        (cache_dir / "ccc.json").write_text("{}", encoding="utf-8")
        (cache_dir / "not_json.txt").write_text("", encoding="utf-8")

        files = list_cache_files(cache_dir)
        assert len(files) == 3
        assert [f.name for f in files] == ["aaa.json", "bbb.json", "ccc.json"]

    def test_list_cache_files_empty_dir(self, tmp_path: Path) -> None:
        """list_cache_files returns empty list for empty directory."""
        cache_dir = tmp_path / "empty"
        cache_dir.mkdir()
        files = list_cache_files(cache_dir)
        assert files == []

    def test_list_cache_files_nonexistent_dir(self, tmp_path: Path) -> None:
        """list_cache_files returns empty list if directory does not exist."""
        files = list_cache_files(tmp_path / "nonexistent")
        assert files == []

    def test_cache_format_compatible_with_pipeline(self, tmp_path: Path) -> None:
        """Written cache files are compatible with the pipeline's _enrich stage.

        The pipeline expects ``{"items": [{"merchant": ..., "description": ...,
        "amount": ...}]}``. Verify the written JSON matches.
        """
        cache_dir = tmp_path / "enrichment-cache"

        data = EnrichmentData(
            transaction_id="pipelinetest",
            source="amazon",
            order_id="999-9999999-9999999",
            items=[
                EnrichmentItem(
                    name="Gadget",
                    price=45.00,
                    merchant="AMAZON - Gadget",
                    description="Gadget Description",
                    amount="-45.00",
                ),
            ],
        )
        written_path = write_cache_file(cache_dir, data)

        # Read raw JSON to verify pipeline compatibility.
        raw = json.loads(written_path.read_text(encoding="utf-8"))
        assert "items" in raw
        assert len(raw["items"]) == 1
        item = raw["items"][0]
        assert "merchant" in item
        assert "description" in item
        assert "amount" in item
        assert item["merchant"] == "AMAZON - Gadget"
        assert item["amount"] == "-45.00"

    def test_overwrite_existing_cache(self, tmp_path: Path) -> None:
        """write_cache_file overwrites an existing file."""
        cache_dir = tmp_path / "enrichment-cache"

        data1 = EnrichmentData(
            transaction_id="overwrite_test",
            source="amazon",
            items=[
                EnrichmentItem(name="V1", price=10.0, merchant="V1", amount="-10.00"),
            ],
        )
        write_cache_file(cache_dir, data1)

        data2 = EnrichmentData(
            transaction_id="overwrite_test",
            source="amazon",
            items=[
                EnrichmentItem(name="V2", price=20.0, merchant="V2", amount="-20.00"),
            ],
        )
        write_cache_file(cache_dir, data2)

        result = read_cache_file(cache_dir / "overwrite_test.json")
        assert result is not None
        assert result.items[0].name == "V2"


# ===========================================================================
# Enrichment data building tests
# ===========================================================================


class TestBuildEnrichmentData:
    """Tests for build_enrichment_data."""

    def test_builds_correct_structure(self) -> None:
        """build_enrichment_data produces correctly structured EnrichmentData."""
        order = AmazonOrder(
            order_id="111-2345678-9012345",
            order_date=date(2025, 11, 5),
            order_total=Decimal("105.00"),
            items=[
                AmazonLineItem(name="Widget", price=Decimal("30.00")),
                AmazonLineItem(name="Book", price=Decimal("15.00")),
                AmazonLineItem(name="Charger", price=Decimal("60.00")),
            ],
        )

        data = build_enrichment_data(
            order=order,
            transaction_id="abc123",
            original_merchant="AMAZON.COM",
        )

        assert data.transaction_id == "abc123"
        assert data.source == "amazon"
        assert data.order_id == "111-2345678-9012345"
        assert len(data.items) == 3

    def test_item_amounts_are_negative(self) -> None:
        """Line item amounts should be negative (expenses)."""
        order = AmazonOrder(
            order_id="test-order",
            order_date=date(2025, 11, 5),
            order_total=Decimal("50.00"),
            items=[
                AmazonLineItem(name="Item", price=Decimal("50.00")),
            ],
        )

        data = build_enrichment_data(order, "txn_001", "AMAZON")
        assert data.items[0].amount == "-50.00"

    def test_item_quantity_multiplied(self) -> None:
        """Items with quantity > 1 have their price multiplied."""
        order = AmazonOrder(
            order_id="test-order",
            order_date=date(2025, 11, 5),
            order_total=Decimal("30.00"),
            items=[
                AmazonLineItem(name="Battery", price=Decimal("10.00"), quantity=3),
            ],
        )

        data = build_enrichment_data(order, "txn_001", "AMAZON")
        assert data.items[0].amount == "-30.00"

    def test_merchant_format(self) -> None:
        """Line items use the 'AMAZON - {name}' merchant format."""
        order = AmazonOrder(
            order_id="test-order",
            order_date=date(2025, 11, 5),
            order_total=Decimal("25.00"),
            items=[
                AmazonLineItem(name="Cool Gadget", price=Decimal("25.00")),
            ],
        )

        data = build_enrichment_data(order, "txn_001", "AMAZON")
        assert data.items[0].merchant == "AMAZON - Cool Gadget"

    def test_long_name_truncated_in_merchant(self) -> None:
        """Product names longer than 80 chars are truncated in the merchant field."""
        long_name = "A" * 120
        order = AmazonOrder(
            order_id="test-order",
            order_date=date(2025, 11, 5),
            order_total=Decimal("10.00"),
            items=[
                AmazonLineItem(name=long_name, price=Decimal("10.00")),
            ],
        )

        data = build_enrichment_data(order, "txn_001", "AMAZON")
        assert len(data.items[0].merchant) <= len("AMAZON - ") + 80
        # But the full name is preserved in description.
        assert data.items[0].description == long_name


# ===========================================================================
# Date and price parsing tests
# ===========================================================================


class TestDateParsing:
    """Tests for _parse_date helper."""

    def test_full_month_name(self) -> None:
        """Parses 'November 15, 2025'."""
        assert _parse_date("November 15, 2025") == date(2025, 11, 15)

    def test_abbreviated_month(self) -> None:
        """Parses 'Nov 15, 2025'."""
        assert _parse_date("Nov 15, 2025") == date(2025, 11, 15)

    def test_no_comma(self) -> None:
        """Parses 'November 15 2025' (no comma)."""
        assert _parse_date("November 15 2025") == date(2025, 11, 15)

    def test_single_digit_day(self) -> None:
        """Parses 'January 5, 2026'."""
        assert _parse_date("January 5, 2026") == date(2026, 1, 5)

    def test_unparseable_returns_none(self) -> None:
        """Unparseable strings return None."""
        assert _parse_date("not a date") is None
        assert _parse_date("") is None

    def test_whitespace_handling(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert _parse_date("  December 25, 2025  ") == date(2025, 12, 25)


class TestPriceParsing:
    """Tests for _parse_price helper."""

    def test_dollar_sign(self) -> None:
        """Parses '$30.00'."""
        assert _parse_price("$30.00") == Decimal("30.00")

    def test_no_dollar_sign(self) -> None:
        """Parses '30.00'."""
        assert _parse_price("30.00") == Decimal("30.00")

    def test_with_comma(self) -> None:
        """Parses '$1,234.56'."""
        assert _parse_price("$1,234.56") == Decimal("1234.56")

    def test_empty_string(self) -> None:
        """Empty string returns zero."""
        assert _parse_price("") == Decimal("0")

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns zero."""
        assert _parse_price("   ") == Decimal("0")


# ===========================================================================
# Provider registry tests
# ===========================================================================


class TestProviderRegistry:
    """Tests for the enrichment provider registry."""

    def test_amazon_provider_registered(self) -> None:
        """AmazonEnrichmentProvider is registered under 'amazon'."""
        from expense_tracker.enrichment import get_provider
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

        cls = get_provider("amazon")
        assert cls is AmazonEnrichmentProvider

    def test_unknown_provider_raises(self) -> None:
        """Requesting an unknown provider raises KeyError."""
        from expense_tracker.enrichment import get_provider

        with pytest.raises(KeyError, match="nonexistent"):
            get_provider("nonexistent")

    def test_amazon_provider_has_name(self) -> None:
        """AmazonEnrichmentProvider.name returns 'amazon'."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

        provider = AmazonEnrichmentProvider()
        assert provider.name == "amazon"


# ===========================================================================
# CLI command tests
# ===========================================================================


class TestEnrichCLI:
    """Tests for the ``expense enrich`` CLI command."""

    def test_enrich_help(self, runner: CliRunner) -> None:
        """expense enrich --help shows expected options."""
        result = runner.invoke(cli, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "--month" in result.output
        assert "--source" in result.output
        assert "--verbose" in result.output

    def test_enrich_missing_month(self, runner: CliRunner) -> None:
        """expense enrich without --month fails."""
        result = runner.invoke(cli, ["enrich", "--source", "amazon"])
        assert result.exit_code != 0

    def test_enrich_invalid_month(self, runner: CliRunner) -> None:
        """expense enrich with invalid month format fails."""
        result = runner.invoke(
            cli, ["enrich", "--month", "2025-13", "--source", "amazon"]
        )
        assert result.exit_code != 0
        assert "Invalid" in result.output

    def test_enrich_invalid_source(self, runner: CliRunner) -> None:
        """expense enrich with unsupported source fails."""
        result = runner.invoke(
            cli, ["enrich", "--month", "2025-11", "--source", "walmart"]
        )
        assert result.exit_code != 0

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider.enrich_multi_account")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_enrich_amazon_success(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Successful Amazon enrichment shows correct summary."""
        from expense_tracker.enrichment import EnrichmentResult
        from expense_tracker.models import (
            AccountConfig,
            AppConfig,
            MerchantRule,
            PipelineResult,
            Transaction,
        )

        mock_load_config.return_value = AppConfig(
            accounts=[
                AccountConfig(
                    name="Chase CC",
                    institution="chase",
                    parser="chase",
                    account_type="credit_card",
                    input_dir="input/chase",
                )
            ],
            enrichment_cache_dir="enrichment-cache",
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []

        mock_pipeline_run.return_value = PipelineResult(
            transactions=[
                Transaction(
                    transaction_id="txn_001",
                    date=date(2025, 11, 6),
                    merchant="AMAZON.COM",
                    description="AMAZON.COM",
                    amount=Decimal("-105.00"),
                    institution="chase",
                    account="Chase CC",
                ),
            ]
        )

        mock_enrich.return_value = EnrichmentResult(
            orders_found=5,
            orders_matched=3,
            orders_unmatched=2,
            cache_files_written=3,
            unmatched_details=[
                "Order 444-444 (2025-11-28, $19.99): Small item",
            ],
        )

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2025-11", "--source", "amazon"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Amazon Enrichment Summary" in result.output
        assert "Orders found" in result.output
        assert "5" in result.output
        assert "Orders matched" in result.output
        assert "3" in result.output
        assert "Cache files written" in result.output
        assert "Unmatched orders" in result.output

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider.enrich_multi_account")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_enrich_amazon_with_errors(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Enrichment errors are displayed to the user."""
        from expense_tracker.enrichment import EnrichmentResult
        from expense_tracker.models import (
            AccountConfig,
            AppConfig,
            PipelineResult,
        )

        mock_load_config.return_value = AppConfig(
            accounts=[
                AccountConfig(
                    name="Chase CC",
                    institution="chase",
                    parser="chase",
                    account_type="credit_card",
                    input_dir="input/chase",
                )
            ],
            enrichment_cache_dir="enrichment-cache",
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []
        mock_pipeline_run.return_value = PipelineResult(transactions=[])

        mock_enrich.return_value = EnrichmentResult(
            errors=["Amazon scraping failed: browser crash"],
        )

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2025-11", "--source", "amazon"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0  # Command itself succeeds; errors reported in output
        assert "browser crash" in result.output

    @patch("expense_tracker.config.load_config")
    def test_enrich_missing_config(
        self, mock_load_config: MagicMock, runner: CliRunner
    ) -> None:
        """Missing config.toml shows helpful error."""
        mock_load_config.side_effect = FileNotFoundError("config.toml not found")

        result = runner.invoke(
            cli,
            ["enrich", "--month", "2025-11", "--source", "amazon"],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "expense init" in result.output


# ===========================================================================
# Integration: end-to-end cache + pipeline compatibility
# ===========================================================================


class TestEnrichmentPipelineIntegration:
    """Tests that Amazon enrichment cache files work with the pipeline."""

    def test_amazon_cache_consumed_by_pipeline_enrich(self, tmp_path: Path) -> None:
        """An enrichment cache file written by the Amazon provider is
        correctly consumed by the pipeline's _enrich stage.
        """
        from expense_tracker.models import AppConfig, Transaction
        from expense_tracker.pipeline import _enrich

        # Set up project structure.
        cache_dir = tmp_path / "enrichment-cache"

        # Build an Amazon enrichment cache file.
        order = AmazonOrder(
            order_id="111-2345678-9012345",
            order_date=date(2025, 11, 5),
            order_total=Decimal("45.00"),
            items=[
                AmazonLineItem(name="Widget", price=Decimal("30.00")),
                AmazonLineItem(name="Book", price=Decimal("15.00")),
            ],
        )
        data = build_enrichment_data(order, "txn_amazon_001", "AMAZON.COM")
        write_cache_file(cache_dir, data)

        # Build a matching transaction.
        txn = Transaction(
            transaction_id="txn_amazon_001",
            date=date(2025, 11, 6),
            merchant="AMAZON.COM",
            description="AMAZON.COM",
            amount=Decimal("-45.00"),
            institution="chase",
            account="Chase CC",
        )

        config = AppConfig(enrichment_cache_dir="enrichment-cache")

        # Run the pipeline's enrich stage.
        result = _enrich([txn], tmp_path, config)

        # Should have split into 2 transactions.
        assert len(result.transactions) == 2
        assert result.warnings == []

        # Verify split details.
        assert result.transactions[0].merchant == "AMAZON - Widget"
        assert result.transactions[0].amount == Decimal("-30.00")
        assert result.transactions[0].split_from == "txn_amazon_001"

        assert result.transactions[1].merchant == "AMAZON - Book"
        assert result.transactions[1].amount == Decimal("-15.00")
        assert result.transactions[1].split_from == "txn_amazon_001"


# ===========================================================================
# Multi-account support tests
# ===========================================================================


class TestMultiAccountOrderMerging:
    """Tests for merging orders from multiple Amazon accounts before matching."""

    def test_orders_from_two_accounts_merged_and_matched(self) -> None:
        """Orders from two separate accounts are merged and matched against
        a single transaction list."""
        # Orders from account "primary".
        primary_orders = [
            AmazonOrder(
                order_id="111-1111111-1111111",
                order_date=date(2025, 11, 5),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Widget", price=Decimal("50.00"))],
                account_label="primary",
            ),
        ]
        # Orders from account "secondary".
        secondary_orders = [
            AmazonOrder(
                order_id="222-2222222-2222222",
                order_date=date(2025, 11, 12),
                order_total=Decimal("30.00"),
                items=[AmazonLineItem(name="Book", price=Decimal("30.00"))],
                account_label="secondary",
            ),
        ]

        merged = primary_orders + secondary_orders

        txns = [
            {
                "transaction_id": "txn_001",
                "date": date(2025, 11, 6),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON.COM",
            },
            {
                "transaction_id": "txn_002",
                "date": date(2025, 11, 13),
                "amount": Decimal("-30.00"),
                "merchant": "AMZN MKTP US",
            },
        ]

        matches = match_orders_to_transactions(merged, txns)
        assert len(matches) == 2

        matched_labels = {o.account_label for o, _ in matches}
        assert "primary" in matched_labels
        assert "secondary" in matched_labels

    def test_cross_account_ambiguity_handled(self) -> None:
        """When two accounts have identical orders (same amount, same date),
        they are treated as ambiguous and not matched."""
        primary_order = AmazonOrder(
            order_id="111-1111111-1111111",
            order_date=date(2025, 11, 5),
            order_total=Decimal("50.00"),
            items=[AmazonLineItem(name="Widget A", price=Decimal("50.00"))],
            account_label="primary",
        )
        secondary_order = AmazonOrder(
            order_id="222-2222222-2222222",
            order_date=date(2025, 11, 5),
            order_total=Decimal("50.00"),
            items=[AmazonLineItem(name="Widget B", price=Decimal("50.00"))],
            account_label="secondary",
        )

        txns = [
            {
                "transaction_id": "txn_001",
                "date": date(2025, 11, 5),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON.COM",
            },
        ]

        matches = match_orders_to_transactions(
            [primary_order, secondary_order], txns
        )
        # Both orders compete for the same transaction -- ambiguous.
        assert len(matches) == 0

    def test_account_label_preserved_in_enrichment_data(self) -> None:
        """The account_label is propagated to EnrichmentData when building
        cache entries from matched orders."""
        order = AmazonOrder(
            order_id="111-1111111-1111111",
            order_date=date(2025, 11, 5),
            order_total=Decimal("25.00"),
            items=[AmazonLineItem(name="Gadget", price=Decimal("25.00"))],
            account_label="secondary",
        )

        data = build_enrichment_data(
            order=order,
            transaction_id="txn_001",
            original_merchant="AMAZON.COM",
        )
        assert data.account_label == "secondary"

    def test_explicit_account_label_overrides_order_label(self) -> None:
        """When an explicit account_label is passed to build_enrichment_data,
        it takes precedence over the order's label."""
        order = AmazonOrder(
            order_id="111-1111111-1111111",
            order_date=date(2025, 11, 5),
            order_total=Decimal("25.00"),
            items=[AmazonLineItem(name="Gadget", price=Decimal("25.00"))],
            account_label="secondary",
        )

        data = build_enrichment_data(
            order=order,
            transaction_id="txn_001",
            original_merchant="AMAZON.COM",
            account_label="primary",
        )
        assert data.account_label == "primary"

    def test_account_label_persisted_in_cache(self, tmp_path: Path) -> None:
        """The account_label is written to and read back from cache files."""
        cache_dir = tmp_path / "enrichment-cache"

        data = EnrichmentData(
            transaction_id="test_label",
            source="amazon",
            order_id="111-1111111-1111111",
            account_label="primary",
            items=[
                EnrichmentItem(
                    name="Widget",
                    price=30.0,
                    merchant="AMAZON - Widget",
                    description="Widget",
                    amount="-30.00",
                ),
            ],
        )

        written_path = write_cache_file(cache_dir, data)
        result = read_cache_file(written_path)

        assert result is not None
        assert result.account_label == "primary"

    def test_account_label_absent_in_legacy_cache(self, tmp_path: Path) -> None:
        """Legacy cache files without account_label are read with empty label."""
        import json

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        legacy_data = {
            "transaction_id": "legacy_001",
            "source": "amazon",
            "order_id": "999-9999999-9999999",
            "matched_at": "2025-11-01T12:00:00",
            "items": [
                {
                    "name": "Old Item",
                    "price": 10.0,
                    "quantity": 1,
                    "merchant": "AMAZON - Old Item",
                    "description": "Old Item",
                    "amount": "-10.00",
                }
            ],
        }

        cache_file = cache_dir / "legacy_001.json"
        cache_file.write_text(json.dumps(legacy_data), encoding="utf-8")

        result = read_cache_file(cache_file)
        assert result is not None
        assert result.account_label == ""


class TestBackwardCompatibility:
    """Tests that single-account (no config sections) behavior is preserved."""

    def test_no_amazon_config_uses_single_default_account(self) -> None:
        """When no [[enrichment.amazon]] sections exist in config, the
        AppConfig has an empty amazon_accounts list."""
        from expense_tracker.models import AppConfig

        config = AppConfig()
        assert config.amazon_accounts == []

    def test_config_loading_without_enrichment_section(self, tmp_path: Path) -> None:
        """load_config with no enrichment section returns empty amazon_accounts."""
        from expense_tracker.config import load_config

        config_toml = tmp_path / "config.toml"
        config_toml.write_text(
            '[general]\noutput_dir = "output"\n'
            'enrichment_cache_dir = "enrichment-cache"\n',
            encoding="utf-8",
        )
        config = load_config(tmp_path)
        assert config.amazon_accounts == []

    def test_config_loading_with_single_amazon_account(self, tmp_path: Path) -> None:
        """load_config with one [[enrichment.amazon]] section."""
        from expense_tracker.config import load_config

        config_toml = tmp_path / "config.toml"
        config_toml.write_text(
            '[general]\noutput_dir = "output"\n'
            'enrichment_cache_dir = "enrichment-cache"\n\n'
            '[[enrichment.amazon]]\nlabel = "primary"\n',
            encoding="utf-8",
        )
        config = load_config(tmp_path)
        assert len(config.amazon_accounts) == 1
        assert config.amazon_accounts[0].label == "primary"

    def test_config_loading_with_multiple_amazon_accounts(self, tmp_path: Path) -> None:
        """load_config with two [[enrichment.amazon]] sections."""
        from expense_tracker.config import load_config

        config_toml = tmp_path / "config.toml"
        config_toml.write_text(
            '[general]\noutput_dir = "output"\n'
            'enrichment_cache_dir = "enrichment-cache"\n\n'
            '[[enrichment.amazon]]\nlabel = "primary"\n\n'
            '[[enrichment.amazon]]\nlabel = "secondary"\n',
            encoding="utf-8",
        )
        config = load_config(tmp_path)
        assert len(config.amazon_accounts) == 2
        assert config.amazon_accounts[0].label == "primary"
        assert config.amazon_accounts[1].label == "secondary"

    def test_single_enrich_delegates_to_multi_account(self) -> None:
        """The backward-compatible enrich() method delegates to
        enrich_multi_account() with a single default account."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

        provider = AmazonEnrichmentProvider()
        # Patch enrich_multi_account to capture the call.
        with patch.object(provider, "enrich_multi_account") as mock_multi:
            mock_multi.return_value = MagicMock()
            provider.enrich(month="2025-11", root=Path("/tmp/test"))
            mock_multi.assert_called_once()
            call_args = mock_multi.call_args
            accounts = call_args.kwargs.get(
                "amazon_accounts", call_args.args[2] if len(call_args.args) > 2 else None
            )
            assert len(accounts) == 1
            assert accounts[0].label == "default"


class TestPerAccountSessionStorage:
    """Tests for per-account auth state directory paths."""

    def test_default_account_uses_legacy_path(self) -> None:
        """The 'default' account uses .auth/amazon/ for backward compatibility."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

        provider = AmazonEnrichmentProvider()
        auth_dir = provider._auth_dir_for_account(Path("/project"), "default")
        assert auth_dir == Path("/project/.auth/amazon")

    def test_named_account_uses_labeled_path(self) -> None:
        """Named accounts use .auth/amazon-{label}/ paths."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

        provider = AmazonEnrichmentProvider()

        auth_dir_primary = provider._auth_dir_for_account(Path("/project"), "primary")
        assert auth_dir_primary == Path("/project/.auth/amazon-primary")

        auth_dir_secondary = provider._auth_dir_for_account(Path("/project"), "secondary")
        assert auth_dir_secondary == Path("/project/.auth/amazon-secondary")

    def test_each_account_gets_separate_directory(self, tmp_path: Path) -> None:
        """Multiple accounts create separate auth directories."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

        provider = AmazonEnrichmentProvider()

        dir1 = provider._auth_dir_for_account(tmp_path, "primary")
        dir2 = provider._auth_dir_for_account(tmp_path, "secondary")

        assert dir1 != dir2
        assert "primary" in str(dir1)
        assert "secondary" in str(dir2)


class TestMultiAccountCLISummary:
    """Tests for per-account CLI summary output with multiple accounts."""

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider.enrich_multi_account")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_multi_account_summary_output(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        """CLI shows per-account breakdown when multiple accounts are configured."""
        from expense_tracker.enrichment import (
            AccountEnrichmentStats,
            EnrichmentResult,
        )
        from expense_tracker.models import (
            AccountConfig,
            AmazonAccountConfig,
            AppConfig,
            PipelineResult,
        )

        mock_load_config.return_value = AppConfig(
            accounts=[
                AccountConfig(
                    name="Chase CC",
                    institution="chase",
                    parser="chase",
                    account_type="credit_card",
                    input_dir="input/chase",
                )
            ],
            enrichment_cache_dir="enrichment-cache",
            amazon_accounts=[
                AmazonAccountConfig(label="primary"),
                AmazonAccountConfig(label="secondary"),
            ],
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []
        mock_pipeline_run.return_value = PipelineResult(transactions=[])

        mock_enrich.return_value = EnrichmentResult(
            orders_found=23,
            orders_matched=18,
            orders_unmatched=5,
            cache_files_written=18,
            account_stats=[
                AccountEnrichmentStats(label="primary", orders_found=15, orders_matched=12),
                AccountEnrichmentStats(label="secondary", orders_found=8, orders_matched=6),
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["enrich", "--month", "2025-11", "--source", "amazon"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Amazon Enrichment Summary" in result.output
        assert 'Account "primary"' in result.output
        assert "15 orders found" in result.output
        assert "12 matched" in result.output
        assert 'Account "secondary"' in result.output
        assert "8 orders found" in result.output
        assert "6 matched" in result.output
        assert "Total: 23 orders" in result.output
        assert "18 matched" in result.output
        assert "5 unmatched" in result.output

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider.enrich_multi_account")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_single_account_summary_uses_legacy_format(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        """CLI uses the original summary format when only one account is configured."""
        from expense_tracker.enrichment import (
            AccountEnrichmentStats,
            EnrichmentResult,
        )
        from expense_tracker.models import (
            AccountConfig,
            AppConfig,
            PipelineResult,
        )

        mock_load_config.return_value = AppConfig(
            accounts=[
                AccountConfig(
                    name="Chase CC",
                    institution="chase",
                    parser="chase",
                    account_type="credit_card",
                    input_dir="input/chase",
                )
            ],
            enrichment_cache_dir="enrichment-cache",
        )
        mock_load_categories.return_value = []
        mock_load_rules.return_value = []
        mock_pipeline_run.return_value = PipelineResult(transactions=[])

        mock_enrich.return_value = EnrichmentResult(
            orders_found=5,
            orders_matched=3,
            orders_unmatched=2,
            cache_files_written=3,
            account_stats=[
                AccountEnrichmentStats(label="default", orders_found=5, orders_matched=3),
            ],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["enrich", "--month", "2025-11", "--source", "amazon"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        # Single account -- should use legacy summary format, not per-account breakdown.
        assert "Orders found" in result.output
        assert "Orders matched" in result.output
        assert 'Account "' not in result.output
        assert "Total:" not in result.output


class TestMultiAccountEnrichProvider:
    """Tests for the AmazonEnrichmentProvider.enrich_multi_account method."""

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider._scrape_orders")
    def test_enrich_multi_account_scrapes_each_account(
        self, mock_scrape: MagicMock, tmp_path: Path
    ) -> None:
        """enrich_multi_account calls _scrape_orders once per account."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider
        from expense_tracker.models import AmazonAccountConfig

        mock_scrape.return_value = []

        provider = AmazonEnrichmentProvider()
        accounts = [
            AmazonAccountConfig(label="primary"),
            AmazonAccountConfig(label="secondary"),
        ]

        provider.enrich_multi_account(
            month="2025-11",
            root=tmp_path,
            amazon_accounts=accounts,
            transactions=[],
        )

        assert mock_scrape.call_count == 2

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider._scrape_orders")
    def test_enrich_multi_account_merges_orders_before_matching(
        self, mock_scrape: MagicMock, tmp_path: Path
    ) -> None:
        """Orders from different accounts are merged and matched together."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider
        from expense_tracker.models import AmazonAccountConfig

        primary_orders = [
            AmazonOrder(
                order_id="111-1111111-1111111",
                order_date=date(2025, 11, 5),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Widget", price=Decimal("50.00"))],
            ),
        ]
        secondary_orders = [
            AmazonOrder(
                order_id="222-2222222-2222222",
                order_date=date(2025, 11, 12),
                order_total=Decimal("30.00"),
                items=[AmazonLineItem(name="Book", price=Decimal("30.00"))],
            ),
        ]

        mock_scrape.side_effect = [primary_orders, secondary_orders]

        txns = [
            {
                "transaction_id": "txn_001",
                "date": date(2025, 11, 6),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON.COM",
            },
            {
                "transaction_id": "txn_002",
                "date": date(2025, 11, 13),
                "amount": Decimal("-30.00"),
                "merchant": "AMZN MKTP US",
            },
        ]

        provider = AmazonEnrichmentProvider()
        accounts = [
            AmazonAccountConfig(label="primary"),
            AmazonAccountConfig(label="secondary"),
        ]

        result = provider.enrich_multi_account(
            month="2025-11",
            root=tmp_path,
            amazon_accounts=accounts,
            transactions=txns,
        )

        assert result.orders_found == 2
        assert result.orders_matched == 2
        assert result.orders_unmatched == 0
        assert result.cache_files_written == 2

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider._scrape_orders")
    def test_enrich_multi_account_per_account_stats(
        self, mock_scrape: MagicMock, tmp_path: Path
    ) -> None:
        """Per-account stats correctly reflect orders found and matched per account."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider
        from expense_tracker.models import AmazonAccountConfig

        primary_orders = [
            AmazonOrder(
                order_id="111-1111111-1111111",
                order_date=date(2025, 11, 5),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Widget", price=Decimal("50.00"))],
            ),
            AmazonOrder(
                order_id="111-2222222-2222222",
                order_date=date(2025, 11, 8),
                order_total=Decimal("99.99"),
                items=[AmazonLineItem(name="Keyboard", price=Decimal("99.99"))],
            ),
        ]
        secondary_orders = [
            AmazonOrder(
                order_id="222-3333333-3333333",
                order_date=date(2025, 11, 12),
                order_total=Decimal("30.00"),
                items=[AmazonLineItem(name="Book", price=Decimal("30.00"))],
            ),
        ]

        mock_scrape.side_effect = [primary_orders, secondary_orders]

        txns = [
            {
                "transaction_id": "txn_001",
                "date": date(2025, 11, 6),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON.COM",
            },
            {
                "transaction_id": "txn_002",
                "date": date(2025, 11, 13),
                "amount": Decimal("-30.00"),
                "merchant": "AMZN MKTP US",
            },
        ]

        provider = AmazonEnrichmentProvider()
        accounts = [
            AmazonAccountConfig(label="primary"),
            AmazonAccountConfig(label="secondary"),
        ]

        result = provider.enrich_multi_account(
            month="2025-11",
            root=tmp_path,
            amazon_accounts=accounts,
            transactions=txns,
        )

        assert len(result.account_stats) == 2

        primary_stat = next(s for s in result.account_stats if s.label == "primary")
        assert primary_stat.orders_found == 2
        assert primary_stat.orders_matched == 1

        secondary_stat = next(s for s in result.account_stats if s.label == "secondary")
        assert secondary_stat.orders_found == 1
        assert secondary_stat.orders_matched == 1

        assert result.orders_found == 3
        assert result.orders_matched == 2
        assert result.orders_unmatched == 1

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider._scrape_orders")
    def test_enrich_multi_account_partial_failure(
        self, mock_scrape: MagicMock, tmp_path: Path
    ) -> None:
        """If one account's scraping fails, the other account's orders
        are still processed and matched."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider
        from expense_tracker.models import AmazonAccountConfig

        good_orders = [
            AmazonOrder(
                order_id="111-1111111-1111111",
                order_date=date(2025, 11, 5),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Widget", price=Decimal("50.00"))],
            ),
        ]

        mock_scrape.side_effect = [
            good_orders,
            RuntimeError("Browser crashed"),
        ]

        txns = [
            {
                "transaction_id": "txn_001",
                "date": date(2025, 11, 6),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON.COM",
            },
        ]

        provider = AmazonEnrichmentProvider()
        accounts = [
            AmazonAccountConfig(label="primary"),
            AmazonAccountConfig(label="secondary"),
        ]

        result = provider.enrich_multi_account(
            month="2025-11",
            root=tmp_path,
            amazon_accounts=accounts,
            transactions=txns,
        )

        # The primary account's orders should still be matched.
        assert result.orders_found == 1
        assert result.orders_matched == 1

        # Errors should report the secondary account failure.
        assert len(result.errors) == 1
        assert "secondary" in result.errors[0]
        assert "Browser crashed" in result.errors[0]

        # Per-account stats should show both accounts.
        assert len(result.account_stats) == 2

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider._scrape_orders")
    def test_enrich_multi_account_cache_includes_account_label(
        self, mock_scrape: MagicMock, tmp_path: Path
    ) -> None:
        """Cache files written by multi-account enrichment include the
        account_label in their metadata."""
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider
        from expense_tracker.models import AmazonAccountConfig

        orders = [
            AmazonOrder(
                order_id="111-1111111-1111111",
                order_date=date(2025, 11, 5),
                order_total=Decimal("50.00"),
                items=[AmazonLineItem(name="Widget", price=Decimal("50.00"))],
            ),
        ]

        mock_scrape.return_value = orders

        txns = [
            {
                "transaction_id": "txn_001",
                "date": date(2025, 11, 6),
                "amount": Decimal("-50.00"),
                "merchant": "AMAZON.COM",
            },
        ]

        provider = AmazonEnrichmentProvider()
        accounts = [AmazonAccountConfig(label="primary")]

        result = provider.enrich_multi_account(
            month="2025-11",
            root=tmp_path,
            amazon_accounts=accounts,
            transactions=txns,
        )

        assert result.cache_files_written == 1

        # Read the cache file and verify account_label is present.
        cache_file = tmp_path / "enrichment-cache" / "txn_001.json"
        cache_data = read_cache_file(cache_file)
        assert cache_data is not None
        assert cache_data.account_label == "primary"
