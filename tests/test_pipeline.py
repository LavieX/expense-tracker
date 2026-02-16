"""Tests for expense_tracker.pipeline — pipeline orchestration and stages."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from expense_tracker.config import load_categories, load_config, load_rules
from expense_tracker.models import (
    AccountConfig,
    AppConfig,
    MerchantRule,
    PipelineResult,
    StageResult,
    Transaction,
    generate_transaction_id,
)
from expense_tracker.pipeline import (
    _categorize,
    _deduplicate,
    _detect_transfers,
    _discover_csv_files,
    _enrich,
    _filter_month,
    _match_rules,
    _parse_stage,
    run,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES_CONFIG = FIXTURES / "config"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_txn(
    institution: str = "chase",
    account: str = "Chase CC",
    txn_date: date = date(2026, 1, 15),
    merchant: str = "TEST MERCHANT",
    description: str = "TEST MERCHANT",
    amount: Decimal = Decimal("-50.00"),
    row_ordinal: int = 0,
    category: str = "Uncategorized",
    subcategory: str = "",
    is_transfer: bool = False,
    is_return: bool = False,
    source_file: str = "",
    split_from: str = "",
) -> Transaction:
    """Build a Transaction with a deterministic ID."""
    return Transaction(
        transaction_id=generate_transaction_id(
            institution=institution,
            txn_date=txn_date,
            merchant=merchant,
            amount=amount,
            row_ordinal=row_ordinal,
        ),
        date=txn_date,
        merchant=merchant,
        description=description,
        amount=amount,
        institution=institution,
        account=account,
        category=category,
        subcategory=subcategory,
        is_transfer=is_transfer,
        is_return=is_return,
        source_file=source_file,
        split_from=split_from,
    )


# -- Chase CSV with clean data for integration tests --
_CHASE_CLEAN_CSV = """\
Transaction Date,Post Date,Description,Category,Type,Amount,Memo
01/15/2026,01/16/2026,WHOLEFDS LMT #10554,Groceries,Sale,-171.48,
01/18/2026,01/19/2026,CHIPOTLE MEXICAN GRIL,Food & Drink,Sale,-12.50,
01/20/2026,01/21/2026,SHELL OIL 574726183,Gas,Sale,-48.75,
01/25/2026,01/26/2026,TARGET        00022186,Groceries,Sale,-127.98,
02/05/2026,02/06/2026,APPLE.COM/BILL,Shopping,Sale,-42.85,
02/08/2026,02/09/2026,AMAZON.COM REFUND,Shopping,Return,25.99,
02/10/2026,02/11/2026,CHASE CREDIT CRD AUTOPAY,Payment,Payment,2150.00,
"""

# -- Elevations CSV with clean data including transfer-matching transactions --
_ELEVATIONS_CLEAN_CSV = """\
"Transaction ID","Posting Date","Effective Date","Transaction Type","Amount","Check Number","Reference Number","Description","Transaction Category","Type","Balance","Memo","Extended Description"
"row0","1/15/2026","1/15/2026","Debit","-445.00000","","1001","PRIMROSE SCHOOL TYPE: 3037741919  ID: 1470259040 CO: PRIMROSE SCHOOL","Child","ACH","10000.00","",""
"row1","1/18/2026","1/18/2026","Debit","-49.95000","","1002","LPC Nextlight TYPE: LPC BB  ID: 4846000608 CO: LPC Nextlight","Utilities","ACH","10000.00","",""
"row2","2/1/2026","2/1/2026","Credit","2254.14000","","1003","VMWARE INC TYPE: PAYROLL  ID: 9111111103 CO: VMWARE INC","Paychecks","ACH","12000.00","",""
"row3","2/5/2026","2/5/2026","Debit","-2150.00000","","1004","CHASE CREDIT CRD TYPE: AUTOPAY  ID: 4760039224 CO: CHASE CREDIT CRD","CC Payment","ACH","10000.00","",""
"row4","2/8/2026","2/8/2026","Debit","-474.90000","","1005","CAPITAL ONE TYPE: CRCARDPMT  ID: 9541719318 CO: CAPITAL ONE  NAME: LAVIE TOBEY","CC Payment","ACH","9500.00","",""
"""

# -- Capital One CSV with clean data including credit card autopay credit --
_CAPITAL_ONE_CLEAN_CSV = """\
Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit
2026-01-10,2026-01-11,4218,FIRST WATCH 0307 PAT,Dining,41.78,
2026-01-12,2026-01-13,4218,REI.COM  800-426-4840,Merchandise,160.65,
2026-02-02,2026-02-03,4218,PETCO 1234,Merchandise,45.67,
2026-02-05,2026-02-06,4218,CVS/PHARMACY #08432,Health,18.50,
2026-02-08,2026-02-09,4218,SPOTIFY USA,Entertainment,10.99,
2026-02-12,2026-02-12,4218,CAPITAL ONE AUTOPAY PYMT,Payment/Credit,,474.90
"""


@pytest.fixture
def pipeline_project_dir(tmp_path: Path) -> Path:
    """Create a temp project directory with clean CSV data for pipeline testing.

    Unlike tmp_project_dir (which uses sample CSVs that intentionally have
    >10% malformed rows for parser error testing), this fixture uses clean
    CSV files so that all three parsers succeed and the full pipeline can
    be tested end-to-end.
    """
    project = tmp_path / "pipeline-project"
    project.mkdir()

    # Copy config files from fixtures
    for config_file in ("config.toml", "categories.toml", "rules.toml"):
        shutil.copy2(FIXTURES_CONFIG / config_file, project / config_file)

    # Create input directories with clean CSVs
    chase_dir = project / "input" / "chase"
    chase_dir.mkdir(parents=True)
    (chase_dir / "Activity2026.csv").write_text(_CHASE_CLEAN_CSV, encoding="utf-8")

    cap1_dir = project / "input" / "capital-one"
    cap1_dir.mkdir(parents=True)
    (cap1_dir / "Activity2026.csv").write_text(_CAPITAL_ONE_CLEAN_CSV, encoding="utf-8")

    elev_dir = project / "input" / "elevations"
    elev_dir.mkdir(parents=True)
    (elev_dir / "Activity2026.csv").write_text(_ELEVATIONS_CLEAN_CSV, encoding="utf-8")

    # Create output and enrichment-cache directories
    (project / "output").mkdir()
    (project / "enrichment-cache").mkdir()

    return project


# ---------------------------------------------------------------------------
# CSV file discovery
# ---------------------------------------------------------------------------


class TestDiscoverCsvFiles:
    """Tests for _discover_csv_files."""

    def test_finds_csv_files(self, tmp_path: Path):
        """Discovers .csv files in the directory."""
        (tmp_path / "data.csv").write_text("header\n")
        (tmp_path / "more.csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        assert len(result) == 2
        names = [p.name for p in result]
        assert "data.csv" in names
        assert "more.csv" in names

    def test_case_insensitive_extension(self, tmp_path: Path):
        """Discovers .CSV and .Csv files."""
        (tmp_path / "upper.CSV").write_text("header\n")
        (tmp_path / "mixed.Csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        assert len(result) == 2

    def test_excludes_hidden_files(self, tmp_path: Path):
        """Files starting with . are excluded."""
        (tmp_path / ".hidden.csv").write_text("header\n")
        (tmp_path / "visible.csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "visible.csv"

    def test_excludes_temp_files(self, tmp_path: Path):
        """Files starting with ~ or _ are excluded."""
        (tmp_path / "~temp.csv").write_text("header\n")
        (tmp_path / "_scratch.csv").write_text("header\n")
        (tmp_path / "real.csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "real.csv"

    def test_non_recursive(self, tmp_path: Path):
        """Does not descend into subdirectories."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.csv").write_text("header\n")
        (tmp_path / "top.csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "top.csv"

    def test_excludes_non_csv_files(self, tmp_path: Path):
        """Non-.csv files are excluded."""
        (tmp_path / "readme.txt").write_text("text\n")
        (tmp_path / "data.tsv").write_text("header\n")
        (tmp_path / "actual.csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        assert len(result) == 1

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        """Returns empty list if directory does not exist."""
        result = _discover_csv_files(tmp_path / "nonexistent")
        assert result == []

    def test_returns_sorted(self, tmp_path: Path):
        """Results are sorted by path."""
        (tmp_path / "c.csv").write_text("header\n")
        (tmp_path / "a.csv").write_text("header\n")
        (tmp_path / "b.csv").write_text("header\n")
        result = _discover_csv_files(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.csv", "b.csv", "c.csv"]


# ---------------------------------------------------------------------------
# Parse stage
# ---------------------------------------------------------------------------


class TestParseStage:
    """Tests for _parse_stage."""

    def test_parses_all_accounts(self, pipeline_project_dir: Path):
        """Parse stage discovers and parses CSVs from all configured accounts."""
        config = load_config(pipeline_project_dir)
        result = _parse_stage(config, pipeline_project_dir)

        institutions = {txn.institution for txn in result.transactions}
        assert "chase" in institutions
        assert "capital_one" in institutions
        assert "elevations" in institutions
        assert len(result.transactions) > 0

    def test_unknown_parser_produces_error(self, pipeline_project_dir: Path):
        """An account with an unknown parser name produces an error."""
        config = load_config(pipeline_project_dir)
        config.accounts.append(
            AccountConfig(
                name="Unknown Bank",
                institution="unknown",
                parser="nonexistent",
                account_type="checking",
                input_dir="input/unknown",
            )
        )
        result = _parse_stage(config, pipeline_project_dir)
        assert any("Unknown parser" in e for e in result.errors)

    def test_empty_input_dir_warns(self, tmp_path: Path):
        """An account whose input_dir has no CSVs produces a warning."""
        empty_dir = tmp_path / "input" / "empty"
        empty_dir.mkdir(parents=True)

        config = AppConfig(
            accounts=[
                AccountConfig(
                    name="Empty",
                    institution="chase",
                    parser="chase",
                    account_type="credit_card",
                    input_dir="input/empty",
                )
            ]
        )
        result = _parse_stage(config, tmp_path)
        assert any("No CSV files found" in w for w in result.warnings)
        assert result.transactions == []

    def test_parser_warnings_propagated(self, tmp_project_dir: Path):
        """Malformed rows from parsers produce warnings in the stage result."""
        config = load_config(tmp_project_dir)
        result = _parse_stage(config, tmp_project_dir)
        # The sample CSVs have malformed rows that produce warnings/errors
        assert len(result.warnings) > 0 or len(result.errors) > 0


# ---------------------------------------------------------------------------
# Filter month
# ---------------------------------------------------------------------------


class TestFilterMonth:
    """Tests for _filter_month."""

    def test_filters_to_target_month(self):
        """Only transactions in the target month are returned."""
        txns = [
            _make_txn(txn_date=date(2026, 1, 5), row_ordinal=0),
            _make_txn(txn_date=date(2026, 1, 31), row_ordinal=1),
            _make_txn(txn_date=date(2026, 2, 1), row_ordinal=2),
            _make_txn(txn_date=date(2025, 12, 31), row_ordinal=3),
        ]
        result = _filter_month(txns, "2026-01")
        assert len(result.transactions) == 2
        assert all(t.date.month == 1 and t.date.year == 2026 for t in result.transactions)

    def test_february_boundary(self):
        """February filtering handles the 28/29-day boundary."""
        txns = [
            _make_txn(txn_date=date(2026, 2, 1), row_ordinal=0),
            _make_txn(txn_date=date(2026, 2, 28), row_ordinal=1),
            _make_txn(txn_date=date(2026, 3, 1), row_ordinal=2),
        ]
        result = _filter_month(txns, "2026-02")
        assert len(result.transactions) == 2

    def test_december_boundary(self):
        """December filtering includes the 31st."""
        txns = [
            _make_txn(txn_date=date(2025, 12, 1), row_ordinal=0),
            _make_txn(txn_date=date(2025, 12, 31), row_ordinal=1),
            _make_txn(txn_date=date(2026, 1, 1), row_ordinal=2),
        ]
        result = _filter_month(txns, "2025-12")
        assert len(result.transactions) == 2

    def test_empty_input_returns_empty(self):
        """Filtering an empty list returns an empty result."""
        result = _filter_month([], "2026-01")
        assert result.transactions == []

    def test_no_matches_returns_empty(self):
        """Returns empty if no transactions fall in the target month."""
        txns = [_make_txn(txn_date=date(2026, 3, 15))]
        result = _filter_month(txns, "2026-01")
        assert result.transactions == []


# ---------------------------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    """Tests for _deduplicate."""

    def test_removes_exact_duplicates(self):
        """Transactions with the same transaction_id are deduplicated."""
        txn = _make_txn()
        result = _deduplicate([txn, txn])
        assert len(result.transactions) == 1
        assert result.transactions[0] is txn

    def test_keeps_first_occurrence(self):
        """When duplicates exist, the first occurrence is kept."""
        txn1 = _make_txn(merchant="STORE A", amount=Decimal("-10.00"), row_ordinal=0)
        txn2 = _make_txn(merchant="STORE A", amount=Decimal("-10.00"), row_ordinal=0)
        assert txn1.transaction_id == txn2.transaction_id
        txn2_modified = replace(txn2, source_file="second_file.csv")
        result = _deduplicate([txn1, txn2_modified])
        assert len(result.transactions) == 1
        assert result.transactions[0].source_file == ""  # txn1 had empty source_file

    def test_no_duplicates_returns_all(self):
        """When there are no duplicates, all transactions pass through."""
        txns = [
            _make_txn(row_ordinal=0),
            _make_txn(row_ordinal=1),
            _make_txn(row_ordinal=2),
        ]
        result = _deduplicate(txns)
        assert len(result.transactions) == 3

    def test_warning_on_duplicates(self):
        """A warning is generated when duplicates are removed."""
        txn = _make_txn()
        result = _deduplicate([txn, txn, txn])
        assert len(result.transactions) == 1
        assert len(result.warnings) == 1
        assert "2 duplicate" in result.warnings[0]

    def test_no_warning_when_no_duplicates(self):
        """No warning when there are no duplicates."""
        txns = [_make_txn(row_ordinal=0), _make_txn(row_ordinal=1)]
        result = _deduplicate(txns)
        assert result.warnings == []

    def test_empty_input(self):
        """Empty input produces empty output."""
        result = _deduplicate([])
        assert result.transactions == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Detect transfers
# ---------------------------------------------------------------------------


class TestDetectTransfers:
    """Tests for _detect_transfers."""

    @pytest.fixture
    def transfer_config(self) -> AppConfig:
        """Config with checking and credit_card accounts."""
        return AppConfig(
            accounts=[
                AccountConfig(
                    name="Chase CC",
                    institution="chase",
                    parser="chase",
                    account_type="credit_card",
                    input_dir="input/chase",
                ),
                AccountConfig(
                    name="Elevations Checking",
                    institution="elevations",
                    parser="elevations",
                    account_type="checking",
                    input_dir="input/elevations",
                ),
                AccountConfig(
                    name="Capital One CC",
                    institution="capital_one",
                    parser="capital_one",
                    account_type="credit_card",
                    input_dir="input/capital-one",
                ),
            ],
            transfer_keywords=["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"],
            transfer_date_window=5,
        )

    def test_matches_transfer_pair(self, transfer_config: AppConfig):
        """A checking debit + CC credit with same amount within window are matched."""
        checking_debit = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="CHASE CREDIT CRD",
            description="CHASE CREDIT CRD TYPE: AUTOPAY",
            amount=Decimal("-500.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=0,
        )
        cc_credit = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="CHASE CREDIT CRD AUTOPAY",
            description="CHASE CREDIT CRD AUTOPAY",
            amount=Decimal("500.00"),
            txn_date=date(2026, 2, 7),
            row_ordinal=1,
        )
        _detect_transfers([checking_debit, cc_credit], transfer_config)
        assert checking_debit.is_transfer is True
        assert cc_credit.is_transfer is True

    def test_amount_mismatch_no_match(self, transfer_config: AppConfig):
        """Different amounts do not form a transfer pair."""
        checking_debit = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="CHASE PAYMENT",
            description="CHASE PAYMENT",
            amount=Decimal("-500.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=0,
        )
        cc_credit = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="CHASE AUTOPAY",
            description="CHASE AUTOPAY",
            amount=Decimal("499.99"),
            txn_date=date(2026, 2, 6),
            row_ordinal=1,
        )
        _detect_transfers([checking_debit, cc_credit], transfer_config)
        assert checking_debit.is_transfer is False
        assert cc_credit.is_transfer is False

    def test_outside_date_window_no_match(self, transfer_config: AppConfig):
        """Transactions outside the date window are not matched."""
        checking_debit = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="CHASE PAYMENT",
            description="CHASE PAYMENT",
            amount=Decimal("-500.00"),
            txn_date=date(2026, 2, 1),
            row_ordinal=0,
        )
        cc_credit = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="AUTOPAY PYMT",
            description="AUTOPAY PYMT",
            amount=Decimal("500.00"),
            txn_date=date(2026, 2, 10),  # 9 days apart > 5 day window
            row_ordinal=1,
        )
        _detect_transfers([checking_debit, cc_credit], transfer_config)
        assert checking_debit.is_transfer is False
        assert cc_credit.is_transfer is False

    def test_keyword_in_description(self, transfer_config: AppConfig):
        """Transfer keyword in description still triggers matching."""
        checking_debit = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="CHASE AUTOPAY",
            description="CHASE AUTOPAY TYPE: PAYMENT",
            amount=Decimal("-200.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=10,
        )
        cc_credit = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="AUTOPAY PYMT",
            description="AUTOPAY PYMT",
            amount=Decimal("200.00"),
            txn_date=date(2026, 2, 6),
            row_ordinal=11,
        )
        _detect_transfers([checking_debit, cc_credit], transfer_config)
        assert checking_debit.is_transfer is True
        assert cc_credit.is_transfer is True

    def test_no_keyword_match_no_transfer(self, transfer_config: AppConfig):
        """Checking debits without transfer keywords are not considered."""
        checking_debit = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="KING SOOPERS",
            description="KING SOOPERS GROCERY",
            amount=Decimal("-100.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=0,
        )
        cc_credit = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="REFUND",
            description="REFUND",
            amount=Decimal("100.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=1,
        )
        _detect_transfers([checking_debit, cc_credit], transfer_config)
        assert checking_debit.is_transfer is False
        assert cc_credit.is_transfer is False

    def test_non_transfer_transactions_untouched(self, transfer_config: AppConfig):
        """Regular transactions are passed through without modification."""
        grocery = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="KING SOOPERS",
            description="KING SOOPERS",
            amount=Decimal("-50.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=0,
        )
        result = _detect_transfers([grocery], transfer_config)
        assert len(result.transactions) == 1
        assert grocery.is_transfer is False

    def test_cc_credit_matched_only_once(self, transfer_config: AppConfig):
        """A CC credit is only matched to one checking debit."""
        debit1 = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="CHASE PAYMENT",
            description="CHASE PAYMENT",
            amount=Decimal("-300.00"),
            txn_date=date(2026, 2, 5),
            row_ordinal=0,
        )
        debit2 = _make_txn(
            institution="elevations",
            account="Elevations Checking",
            merchant="CHASE AUTOPAY",
            description="CHASE AUTOPAY",
            amount=Decimal("-300.00"),
            txn_date=date(2026, 2, 6),
            row_ordinal=1,
        )
        cc_credit = _make_txn(
            institution="chase",
            account="Chase CC",
            merchant="PAYMENT RECEIVED",
            description="PAYMENT RECEIVED",
            amount=Decimal("300.00"),
            txn_date=date(2026, 2, 6),
            row_ordinal=2,
        )
        _detect_transfers([debit1, debit2, cc_credit], transfer_config)
        assert cc_credit.is_transfer is True
        transfer_debits = [t for t in [debit1, debit2] if t.is_transfer]
        assert len(transfer_debits) == 1


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------


class TestEnrich:
    """Tests for _enrich."""

    @pytest.fixture
    def enrich_config(self) -> AppConfig:
        return AppConfig(enrichment_cache_dir="enrichment-cache")

    def test_no_cache_passthrough(self, tmp_path: Path, enrich_config: AppConfig):
        """Transactions without cache entries pass through unchanged."""
        (tmp_path / "enrichment-cache").mkdir()
        txn = _make_txn()
        result = _enrich([txn], tmp_path, enrich_config)
        assert len(result.transactions) == 1
        assert result.transactions[0] is txn
        assert result.warnings == []

    def test_valid_split(self, tmp_path: Path, enrich_config: AppConfig):
        """Enrichment cache with valid splits replaces the transaction."""
        cache_dir = tmp_path / "enrichment-cache"
        cache_dir.mkdir()

        txn = _make_txn(
            merchant="TARGET 00022186",
            amount=Decimal("-127.98"),
        )

        enrichment_data = {
            "items": [
                {"merchant": "TARGET - Diapers", "description": "Diapers Pack", "amount": "-65.49"},
                {
                    "merchant": "TARGET - Snacks",
                    "description": "Goldfish Crackers",
                    "amount": "-62.49",
                },
            ]
        }
        (cache_dir / f"{txn.transaction_id}.json").write_text(
            json.dumps(enrichment_data), encoding="utf-8"
        )

        result = _enrich([txn], tmp_path, enrich_config)
        assert len(result.transactions) == 2
        assert result.warnings == []

        # Check split IDs
        assert result.transactions[0].transaction_id == f"{txn.transaction_id}-1"
        assert result.transactions[1].transaction_id == f"{txn.transaction_id}-2"

        # Check split_from
        assert result.transactions[0].split_from == txn.transaction_id
        assert result.transactions[1].split_from == txn.transaction_id

        # Check merchants -- retailer prefix "TARGET - " is stripped
        # so splits get categorized by product name.
        assert result.transactions[0].merchant == "Diapers"
        assert result.transactions[1].merchant == "Snacks"

        # Check amounts
        assert result.transactions[0].amount == Decimal("-65.49")
        assert result.transactions[1].amount == Decimal("-62.49")

    def test_split_sum_mismatch_keeps_original(self, tmp_path: Path, enrich_config: AppConfig):
        """When split amounts do not sum to original, keep original and warn."""
        cache_dir = tmp_path / "enrichment-cache"
        cache_dir.mkdir()

        txn = _make_txn(
            merchant="TARGET",
            amount=Decimal("-100.00"),
        )

        enrichment_data = {
            "items": [
                {"merchant": "TARGET - Item A", "description": "Item A", "amount": "-50.00"},
                {"merchant": "TARGET - Item B", "description": "Item B", "amount": "-40.00"},
            ]
        }
        (cache_dir / f"{txn.transaction_id}.json").write_text(
            json.dumps(enrichment_data), encoding="utf-8"
        )

        result = _enrich([txn], tmp_path, enrich_config)
        assert len(result.transactions) == 1
        assert result.transactions[0] is txn  # Original kept
        assert len(result.warnings) == 1
        assert "sum to" in result.warnings[0]

    def test_split_within_tolerance(self, tmp_path: Path, enrich_config: AppConfig):
        """Split amounts within $0.01 tolerance are accepted."""
        cache_dir = tmp_path / "enrichment-cache"
        cache_dir.mkdir()

        txn = _make_txn(
            merchant="STORE",
            amount=Decimal("-100.00"),
        )

        # Splits sum to -99.99, within $0.01 of -100.00
        enrichment_data = {
            "items": [
                {"merchant": "STORE - A", "description": "A", "amount": "-66.66"},
                {"merchant": "STORE - B", "description": "B", "amount": "-33.33"},
            ]
        }
        (cache_dir / f"{txn.transaction_id}.json").write_text(
            json.dumps(enrichment_data), encoding="utf-8"
        )

        result = _enrich([txn], tmp_path, enrich_config)
        assert len(result.transactions) == 2
        assert result.warnings == []

    def test_invalid_json_warns_and_keeps_original(self, tmp_path: Path, enrich_config: AppConfig):
        """Malformed JSON produces a warning and keeps the original."""
        cache_dir = tmp_path / "enrichment-cache"
        cache_dir.mkdir()

        txn = _make_txn()
        (cache_dir / f"{txn.transaction_id}.json").write_text(
            "not valid json", encoding="utf-8"
        )

        result = _enrich([txn], tmp_path, enrich_config)
        assert len(result.transactions) == 1
        assert result.transactions[0] is txn
        assert len(result.warnings) == 1

    def test_empty_items_passthrough(self, tmp_path: Path, enrich_config: AppConfig):
        """Enrichment data with empty items list passes transaction through."""
        cache_dir = tmp_path / "enrichment-cache"
        cache_dir.mkdir()

        txn = _make_txn()
        (cache_dir / f"{txn.transaction_id}.json").write_text(
            json.dumps({"items": []}), encoding="utf-8"
        )

        result = _enrich([txn], tmp_path, enrich_config)
        assert len(result.transactions) == 1
        assert result.transactions[0] is txn

    def test_split_preserves_parent_fields(self, tmp_path: Path, enrich_config: AppConfig):
        """Split transactions inherit date, institution, account from parent."""
        cache_dir = tmp_path / "enrichment-cache"
        cache_dir.mkdir()

        txn = _make_txn(
            institution="capital_one",
            account="Cap One CC",
            txn_date=date(2026, 2, 10),
            merchant="STORE",
            amount=Decimal("-20.00"),
        )
        enrichment_data = {
            "items": [
                {"merchant": "STORE - A", "description": "A", "amount": "-10.00"},
                {"merchant": "STORE - B", "description": "B", "amount": "-10.00"},
            ]
        }
        (cache_dir / f"{txn.transaction_id}.json").write_text(
            json.dumps(enrichment_data), encoding="utf-8"
        )

        result = _enrich([txn], tmp_path, enrich_config)
        for split in result.transactions:
            assert split.date == date(2026, 2, 10)
            assert split.institution == "capital_one"
            assert split.account == "Cap One CC"


# ---------------------------------------------------------------------------
# Categorize (rule matching)
# ---------------------------------------------------------------------------


class TestCategorize:
    """Tests for _categorize and _match_rules."""

    def test_rule_matching_basic(self):
        """A matching rule assigns category and subcategory."""
        rules = [
            MerchantRule(
                pattern="KING SOOPERS", category="Food & Dining", subcategory="Groceries"
            ),
        ]
        txn = _make_txn(merchant="KING SOOPERS #123")
        _categorize([txn], rules)
        assert txn.category == "Food & Dining"
        assert txn.subcategory == "Groceries"

    def test_case_insensitive_matching(self):
        """Rule matching is case-insensitive."""
        rules = [
            MerchantRule(pattern="CHIPOTLE", category="Food & Dining", subcategory="Fast Food"),
        ]
        txn = _make_txn(merchant="Chipotle Mexican Gril")
        _categorize([txn], rules)
        assert txn.category == "Food & Dining"

    def test_longest_match_wins(self):
        """When multiple rules match, the longest pattern wins."""
        rules = [
            MerchantRule(pattern="TARGET", category="Shopping", subcategory=""),
            MerchantRule(pattern="TARGET 00022186", category="Kids", subcategory="Supplies"),
        ]
        txn = _make_txn(merchant="TARGET 00022186")
        _categorize([txn], rules)
        assert txn.category == "Kids"
        assert txn.subcategory == "Supplies"

    def test_already_categorized_skipped(self):
        """Transactions with a category other than 'Uncategorized' are skipped."""
        rules = [
            MerchantRule(pattern="STORE", category="Shopping"),
        ]
        txn = _make_txn(merchant="STORE X", category="Food & Dining", subcategory="Groceries")
        _categorize([txn], rules)
        assert txn.category == "Food & Dining"
        assert txn.subcategory == "Groceries"

    def test_no_match_stays_uncategorized(self):
        """Transactions with no matching rule remain Uncategorized."""
        rules = [
            MerchantRule(pattern="SPECIFIC STORE", category="Shopping"),
        ]
        txn = _make_txn(merchant="UNKNOWN MERCHANT")
        _categorize([txn], rules)
        assert txn.category == "Uncategorized"

    def test_match_rules_returns_none_on_no_match(self):
        """_match_rules returns None when no rule matches."""
        rules = [MerchantRule(pattern="CHIPOTLE", category="Food & Dining")]
        assert _match_rules("KING SOOPERS", rules) is None

    def test_match_rules_tie_breaks_by_order(self):
        """When patterns are the same length, first (user rule) wins."""
        rules = [
            MerchantRule(pattern="STORE", category="Shopping", source="user"),
            MerchantRule(pattern="STORE", category="Misc", source="learned"),
        ]
        match = _match_rules("STORE XYZ", rules)
        assert match is not None
        assert match.category == "Shopping"

    def test_categorize_multiple_transactions(self):
        """Categorize handles a mix of matched and unmatched transactions."""
        rules = [
            MerchantRule(
                pattern="WHOLEFDS", category="Food & Dining", subcategory="Groceries"
            ),
            MerchantRule(
                pattern="SHELL OIL", category="Transportation", subcategory="Gas/Fuel"
            ),
        ]
        txns = [
            _make_txn(merchant="WHOLEFDS LMT #10554", row_ordinal=0),
            _make_txn(merchant="SHELL OIL 574726183", row_ordinal=1),
            _make_txn(merchant="RANDOM STORE", row_ordinal=2),
        ]
        _categorize(txns, rules)
        assert txns[0].category == "Food & Dining"
        assert txns[1].category == "Transportation"
        assert txns[2].category == "Uncategorized"


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """Tests for the full pipeline.run() function."""

    def test_full_pipeline_january(self, pipeline_project_dir: Path):
        """Full pipeline for January 2026 produces valid results."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        result = run("2026-01", config, categories, rules, pipeline_project_dir)

        assert isinstance(result, PipelineResult)
        assert len(result.transactions) > 0

        # All transactions should be in January 2026
        for txn in result.transactions:
            assert txn.date.year == 2026
            assert txn.date.month == 1

        # Check that some transactions got categorized via rules
        categorized = [t for t in result.transactions if t.category != "Uncategorized"]
        assert len(categorized) > 0

    def test_full_pipeline_february(self, pipeline_project_dir: Path):
        """Full pipeline for February 2026 produces valid results."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        result = run("2026-02", config, categories, rules, pipeline_project_dir)

        assert isinstance(result, PipelineResult)
        assert len(result.transactions) > 0

        for txn in result.transactions:
            assert txn.date.year == 2026
            assert txn.date.month == 2

    def test_pipeline_no_transactions_month(self, pipeline_project_dir: Path):
        """Pipeline for a month with no data returns empty result."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        result = run("2025-06", config, categories, rules, pipeline_project_dir)

        assert isinstance(result, PipelineResult)
        assert result.transactions == []

    def test_pipeline_detects_transfers(self, pipeline_project_dir: Path):
        """Pipeline marks transfers in February data (checking->CC payments).

        The clean CSV data contains:
        - Elevations checking debit: CHASE CREDIT CRD AUTOPAY -2150.00 on 2/5
        - Chase CC credit: CHASE CREDIT CRD AUTOPAY +2150.00 on 2/10
        - Elevations checking debit: CAPITAL ONE CRCARDPMT -474.90 on 2/8
        - Capital One CC credit: CAPITAL ONE AUTOPAY PYMT +474.90 on 2/12
        """
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        result = run("2026-02", config, categories, rules, pipeline_project_dir)

        transfers = [t for t in result.transactions if t.is_transfer]
        # Should have at least 2 transfer pairs (4 transactions)
        assert len(transfers) >= 2

    def test_pipeline_deduplication_with_overlapping_files(
        self, pipeline_project_dir: Path
    ):
        """When the same CSV is copied twice, duplicates are removed."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        # First run without duplicates to get baseline count
        result_single = run(
            "2026-01", config, categories, rules, pipeline_project_dir
        )

        # Copy the Capital One CSV to create duplicates
        cap1_dir = pipeline_project_dir / "input" / "capital-one"
        shutil.copy2(
            cap1_dir / "Activity2026.csv",
            cap1_dir / "Activity2026_copy.csv",
        )

        result = run("2026-01", config, categories, rules, pipeline_project_dir)

        # Duplicates should have been removed
        assert len(result.transactions) == len(result_single.transactions)
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_pipeline_enrichment_integration(self, pipeline_project_dir: Path):
        """Pipeline splits enriched transactions from cache."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        # First run to get real transaction IDs from January
        result_pre = run("2026-01", config, categories, rules, pipeline_project_dir)
        assert len(result_pre.transactions) > 0

        # Pick an expense transaction to enrich
        target_txn = None
        for t in result_pre.transactions:
            if t.amount < 0 and not t.is_transfer:
                target_txn = t
                break
        assert target_txn is not None

        # Write enrichment cache that sums to the original amount
        cache_dir = pipeline_project_dir / "enrichment-cache"
        half = target_txn.amount / 2
        other_half = target_txn.amount - half
        enrichment_data = {
            "items": [
                {
                    "merchant": f"{target_txn.merchant} - Item A",
                    "description": "Item A",
                    "amount": str(half),
                },
                {
                    "merchant": f"{target_txn.merchant} - Item B",
                    "description": "Item B",
                    "amount": str(other_half),
                },
            ]
        }
        (cache_dir / f"{target_txn.transaction_id}.json").write_text(
            json.dumps(enrichment_data), encoding="utf-8"
        )

        # Re-run pipeline
        result_post = run("2026-01", config, categories, rules, pipeline_project_dir)

        # Should have one more transaction (1 original replaced by 2 splits)
        assert len(result_post.transactions) == len(result_pre.transactions) + 1

        # Verify split transactions exist
        splits = [
            t
            for t in result_post.transactions
            if t.split_from == target_txn.transaction_id
        ]
        assert len(splits) == 2
        assert splits[0].transaction_id == f"{target_txn.transaction_id}-1"
        assert splits[1].transaction_id == f"{target_txn.transaction_id}-2"

    def test_pipeline_accumulates_warnings_and_errors(
        self, pipeline_project_dir: Path
    ):
        """Pipeline collects warnings and errors into the result."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        result = run("2026-01", config, categories, rules, pipeline_project_dir)

        assert isinstance(result.warnings, list)
        assert isinstance(result.errors, list)

    def test_pipeline_result_type(self, pipeline_project_dir: Path):
        """run() returns a PipelineResult instance."""
        config = load_config(pipeline_project_dir)
        categories = load_categories(pipeline_project_dir)
        rules = load_rules(pipeline_project_dir)

        result = run("2026-01", config, categories, rules, pipeline_project_dir)
        assert isinstance(result, PipelineResult)
        assert hasattr(result, "transactions")
        assert hasattr(result, "warnings")
        assert hasattr(result, "errors")

    def test_pipeline_with_sample_csvs_handles_errors(self, tmp_project_dir: Path):
        """Pipeline handles the sample CSVs that have >10% malformed rows.

        The sample CSVs for Chase and Elevations intentionally have enough
        malformed rows to trigger the >10% threshold, causing those files
        to be skipped entirely with errors. Capital One's sample is under
        the threshold. The pipeline should gracefully handle this, producing
        transactions only from the files that parse successfully.
        """
        config = load_config(tmp_project_dir)
        categories = load_categories(tmp_project_dir)
        rules = load_rules(tmp_project_dir)

        result = run("2026-01", config, categories, rules, tmp_project_dir)

        assert isinstance(result, PipelineResult)
        # Should have errors from the failed parser files
        assert len(result.errors) > 0
        # Should still have transactions from Capital One (which parses OK)
        assert len(result.transactions) > 0
        institutions = {t.institution for t in result.transactions}
        assert "capital_one" in institutions
