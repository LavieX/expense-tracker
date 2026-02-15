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

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider.enrich")
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

    @patch("expense_tracker.enrichment.amazon.AmazonEnrichmentProvider.enrich")
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
