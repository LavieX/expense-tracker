"""Tests for expense_tracker.export — CSV export and summary printer.

Covers:
- export: transfer filtering, sort order, column schema, file overwrite,
  output directory creation, and correct field serialization.
- print_summary: source counts, totals, categorization breakdown,
  top uncategorized merchants, spending by category, warnings/errors.
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

from expense_tracker.export import CSV_COLUMNS, export, print_summary
from expense_tracker.models import (
    PipelineResult,
    Transaction,
    generate_transaction_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_txn(
    institution: str = "chase",
    account: str = "Chase Credit Card",
    txn_date: date = date(2026, 1, 15),
    merchant: str = "TEST MERCHANT",
    description: str = "TEST MERCHANT",
    amount: Decimal = Decimal("-50.00"),
    row_ordinal: int = 0,
    category: str = "Uncategorized",
    subcategory: str = "",
    is_transfer: bool = False,
    is_return: bool = False,
    split_from: str = "",
    source_file: str = "",
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
        split_from=split_from,
        source_file=source_file,
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV and return (fieldnames, list-of-row-dicts)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


# ===========================================================================
# export() tests
# ===========================================================================


class TestExport:
    """Tests for the export() function."""

    def test_column_schema(self, tmp_path: Path):
        """The output CSV has exactly the fixed column schema in the correct order."""
        txns = [_make_txn()]
        result_path = export(txns, tmp_path / "output", "2026-01")
        fieldnames, _ = _read_csv(result_path)
        assert fieldnames == CSV_COLUMNS

    def test_column_schema_matches_architecture(self, tmp_path: Path):
        """The CSV_COLUMNS constant matches the architecture doc Section 4."""
        expected = [
            "transaction_id",
            "date",
            "month",
            "merchant",
            "description",
            "amount",
            "institution",
            "account",
            "category",
            "subcategory",
            "is_return",
            "is_recurring",
            "split_from",
            "source",
        ]
        assert expected == CSV_COLUMNS

        # Also verify the written file has this schema
        txns = [_make_txn()]
        result_path = export(txns, tmp_path / "output", "2026-01")
        fieldnames, _ = _read_csv(result_path)
        assert fieldnames == expected

    def test_transfers_excluded(self, tmp_path: Path):
        """Transactions with is_transfer=True are excluded from the CSV output."""
        txns = [
            _make_txn(merchant="GROCERY", amount=Decimal("-50.00"), row_ordinal=0),
            _make_txn(
                merchant="CHASE PAYMENT",
                amount=Decimal("-500.00"),
                row_ordinal=1,
                is_transfer=True,
            ),
            _make_txn(
                merchant="AUTOPAY PYMT",
                amount=Decimal("500.00"),
                row_ordinal=2,
                is_transfer=True,
            ),
            _make_txn(merchant="GAS STATION", amount=Decimal("-35.00"), row_ordinal=3),
        ]
        result_path = export(txns, tmp_path / "output", "2026-01")
        _, rows = _read_csv(result_path)
        assert len(rows) == 2
        merchants = [r["merchant"] for r in rows]
        assert "CHASE PAYMENT" not in merchants
        assert "AUTOPAY PYMT" not in merchants
        assert "GROCERY" in merchants
        assert "GAS STATION" in merchants

    def test_sort_by_date_then_institution_then_amount(self, tmp_path: Path):
        """Rows are sorted by date ascending, then institution, then amount."""
        txns = [
            # Same date, different institutions and amounts
            _make_txn(
                institution="elevations",
                txn_date=date(2026, 1, 15),
                merchant="ELEV STORE",
                amount=Decimal("-100.00"),
                row_ordinal=0,
            ),
            _make_txn(
                institution="chase",
                txn_date=date(2026, 1, 15),
                merchant="CHASE STORE",
                amount=Decimal("-200.00"),
                row_ordinal=1,
            ),
            _make_txn(
                institution="chase",
                txn_date=date(2026, 1, 15),
                merchant="CHASE SMALL",
                amount=Decimal("-50.00"),
                row_ordinal=2,
            ),
            # Earlier date
            _make_txn(
                institution="capital_one",
                txn_date=date(2026, 1, 10),
                merchant="CAP ONE STORE",
                amount=Decimal("-75.00"),
                row_ordinal=3,
            ),
            # Later date
            _make_txn(
                institution="chase",
                txn_date=date(2026, 1, 20),
                merchant="LATE STORE",
                amount=Decimal("-30.00"),
                row_ordinal=4,
            ),
        ]
        result_path = export(txns, tmp_path / "output", "2026-01")
        _, rows = _read_csv(result_path)

        # Verify sort order
        assert len(rows) == 5

        # Row 0: 2026-01-10, capital_one (earliest date)
        assert rows[0]["date"] == "2026-01-10"
        assert rows[0]["institution"] == "capital_one"

        # Row 1: 2026-01-15, chase, -200 (same date, chase < elevations, -200 < -50)
        assert rows[1]["date"] == "2026-01-15"
        assert rows[1]["institution"] == "chase"
        assert Decimal(rows[1]["amount"]) == Decimal("-200")

        # Row 2: 2026-01-15, chase, -50
        assert rows[2]["date"] == "2026-01-15"
        assert rows[2]["institution"] == "chase"
        assert Decimal(rows[2]["amount"]) == Decimal("-50")

        # Row 3: 2026-01-15, elevations
        assert rows[3]["date"] == "2026-01-15"
        assert rows[3]["institution"] == "elevations"

        # Row 4: 2026-01-20 (latest date)
        assert rows[4]["date"] == "2026-01-20"

    def test_sort_amount_ascending(self, tmp_path: Path):
        """Within same date and institution, amounts sort ascending
        (most negative first, then less negative, then positive)."""
        txns = [
            _make_txn(
                txn_date=date(2026, 1, 15),
                merchant="REFUND",
                amount=Decimal("25.00"),
                row_ordinal=0,
                is_return=True,
            ),
            _make_txn(
                txn_date=date(2026, 1, 15),
                merchant="BIG PURCHASE",
                amount=Decimal("-300.00"),
                row_ordinal=1,
            ),
            _make_txn(
                txn_date=date(2026, 1, 15),
                merchant="SMALL PURCHASE",
                amount=Decimal("-10.00"),
                row_ordinal=2,
            ),
        ]
        result_path = export(txns, tmp_path / "output", "2026-01")
        _, rows = _read_csv(result_path)

        amounts = [Decimal(r["amount"]) for r in rows]
        assert amounts == [Decimal("-300"), Decimal("-10"), Decimal("25")]

    def test_output_file_path(self, tmp_path: Path):
        """The output file is named {month}.csv in the output directory."""
        txns = [_make_txn()]
        result_path = export(txns, tmp_path / "output", "2026-01")
        assert result_path == tmp_path / "output" / "2026-01.csv"
        assert result_path.is_file()

    def test_overwrites_existing_file(self, tmp_path: Path):
        """If the output file already exists, it is overwritten."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / "2026-01.csv"
        existing.write_text("old content\n")

        txns = [_make_txn(merchant="NEW DATA")]
        result_path = export(txns, output_dir, "2026-01")
        _, rows = _read_csv(result_path)
        assert len(rows) == 1
        assert rows[0]["merchant"] == "NEW DATA"

    def test_creates_output_directory(self, tmp_path: Path):
        """The output directory is created if it does not exist."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()
        txns = [_make_txn()]
        result_path = export(txns, output_dir, "2026-01")
        assert output_dir.is_dir()
        assert result_path.is_file()

    def test_empty_transaction_list(self, tmp_path: Path):
        """An empty transaction list produces a CSV with only the header."""
        result_path = export([], tmp_path / "output", "2026-01")
        fieldnames, rows = _read_csv(result_path)
        assert fieldnames == CSV_COLUMNS
        assert rows == []

    def test_field_serialization(self, tmp_path: Path):
        """All fields are serialized correctly in the CSV output."""
        txn = _make_txn(
            institution="capital_one",
            account="Capital One Credit Card",
            txn_date=date(2026, 2, 14),
            merchant="PETCO 1234",
            description="PETCO 1234 BOULDER CO",
            amount=Decimal("-45.67"),
            row_ordinal=7,
            category="Pets",
            subcategory="Supplies",
            is_return=False,
            split_from="",
            source_file="input/capital-one/Activity2026.csv",
        )
        result_path = export([txn], tmp_path / "output", "2026-02")
        _, rows = _read_csv(result_path)
        assert len(rows) == 1
        row = rows[0]

        assert row["transaction_id"] == txn.transaction_id
        assert row["date"] == "2026-02-14"
        assert row["merchant"] == "PETCO 1234"
        assert row["description"] == "PETCO 1234 BOULDER CO"
        assert row["amount"] == "-45.67"
        assert row["institution"] == "capital_one"
        assert row["account"] == "Capital One Credit Card"
        assert row["category"] == "Pets"
        assert row["subcategory"] == "Supplies"
        assert row["is_return"] == "False"
        assert row["split_from"] == ""

    def test_is_transfer_not_in_columns(self):
        """The is_transfer field is not in the output column schema."""
        assert "is_transfer" not in CSV_COLUMNS

    def test_source_file_not_in_columns(self):
        """The source_file field is not in the output column schema."""
        assert "source_file" not in CSV_COLUMNS

    def test_return_transaction_serialized(self, tmp_path: Path):
        """A refund/return is correctly serialized with is_return=True."""
        txn = _make_txn(
            merchant="AMAZON REFUND",
            amount=Decimal("25.99"),
            is_return=True,
            row_ordinal=0,
        )
        result_path = export([txn], tmp_path / "output", "2026-01")
        _, rows = _read_csv(result_path)
        assert rows[0]["is_return"] == "True"
        assert rows[0]["amount"] == "25.99"

    def test_split_transaction_serialized(self, tmp_path: Path):
        """A split transaction has split_from populated in the output."""
        txn = Transaction(
            transaction_id="abc123def456-1",
            date=date(2026, 1, 20),
            merchant="TARGET - Diapers",
            description="Split from TARGET purchase",
            amount=Decimal("-65.49"),
            institution="chase",
            account="Chase Credit Card",
            category="Kids",
            subcategory="Supplies",
            split_from="abc123def456",
            source_file="input/chase/Activity2026.csv",
        )
        result_path = export([txn], tmp_path / "output", "2026-01")
        _, rows = _read_csv(result_path)
        assert rows[0]["split_from"] == "abc123def456"
        assert rows[0]["transaction_id"] == "abc123def456-1"

    def test_all_transfers_excluded_produces_empty(self, tmp_path: Path):
        """If all transactions are transfers, the CSV has only a header."""
        txns = [
            _make_txn(is_transfer=True, row_ordinal=0),
            _make_txn(is_transfer=True, row_ordinal=1),
        ]
        result_path = export(txns, tmp_path / "output", "2026-01")
        fieldnames, rows = _read_csv(result_path)
        assert fieldnames == CSV_COLUMNS
        assert rows == []


# ===========================================================================
# print_summary() tests
# ===========================================================================


class TestPrintSummary:
    """Tests for the print_summary() function."""

    def _make_pipeline_result(
        self,
        transactions: list[Transaction] | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> PipelineResult:
        """Build a PipelineResult with sensible defaults."""
        return PipelineResult(
            transactions=transactions or [],
            warnings=warnings or [],
            errors=errors or [],
        )

    def test_header_includes_month(self, capsys):
        """The summary header includes the target month."""
        result = self._make_pipeline_result()
        print_summary(result, "2026-01")
        output = capsys.readouterr().out
        assert "Processing Summary: 2026-01" in output

    def test_source_counts(self, capsys):
        """Source counts are printed per institution."""
        txns = [
            _make_txn(institution="chase", row_ordinal=0),
            _make_txn(institution="chase", row_ordinal=1),
            _make_txn(institution="capital_one", row_ordinal=2),
            _make_txn(institution="elevations", row_ordinal=3),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Capital One (1 txns)" in output
        assert "Chase (2 txns)" in output
        assert "Elevations (1 txns)" in output

    def test_total_and_transfer_count(self, capsys):
        """Total transactions and transfer exclusion count are shown."""
        txns = [
            _make_txn(row_ordinal=0),
            _make_txn(row_ordinal=1),
            _make_txn(row_ordinal=2, is_transfer=True),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "3 transactions" in output
        assert "1 transfers excluded" in output

    def test_categorization_breakdown(self, capsys):
        """Categorization percentage and counts are printed."""
        txns = [
            _make_txn(category="Food & Dining", row_ordinal=0),
            _make_txn(category="Transportation", row_ordinal=1),
            _make_txn(category="Uncategorized", row_ordinal=2),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Categorized: 2 / 3 (66.7%)" in output
        assert "Categorized: 2" in output
        assert "Uncategorized: 1" in output

    def test_categorization_excludes_transfers(self, capsys):
        """Transfers are not counted in categorization stats."""
        txns = [
            _make_txn(category="Food & Dining", row_ordinal=0),
            _make_txn(category="Uncategorized", is_transfer=True, row_ordinal=1),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        # Only 1 non-transfer, and it is categorized -> 100%
        assert "Categorized: 1 / 1 (100.0%)" in output

    def test_top_uncategorized_merchants(self, capsys):
        """Top uncategorized merchants are listed with count and total."""
        txns = [
            _make_txn(
                merchant="UNKNOWN STORE A",
                amount=Decimal("-30.00"),
                row_ordinal=0,
            ),
            _make_txn(
                merchant="UNKNOWN STORE A",
                amount=Decimal("-20.00"),
                row_ordinal=1,
            ),
            _make_txn(
                merchant="MYSTERY SHOP",
                amount=Decimal("-15.50"),
                row_ordinal=2,
            ),
            _make_txn(
                merchant="CATEGORIZED STORE",
                amount=Decimal("-10.00"),
                row_ordinal=3,
                category="Shopping",
            ),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Top uncategorized merchants:" in output
        assert "UNKNOWN STORE A" in output
        assert "2 txns" in output
        assert "$50.00" in output
        assert "MYSTERY SHOP" in output
        assert "1 txns" in output
        assert "$15.50" in output
        # Categorized store should NOT appear in the uncategorized list
        assert (
            "CATEGORIZED STORE"
            not in output.split("Top uncategorized merchants:")[1].split("Spending by category:")[0]
        )

    def test_spending_by_category(self, capsys):
        """Spending by category shows amounts for expense transactions."""
        txns = [
            _make_txn(
                category="Food & Dining",
                amount=Decimal("-100.00"),
                row_ordinal=0,
            ),
            _make_txn(
                category="Food & Dining",
                amount=Decimal("-50.00"),
                row_ordinal=1,
            ),
            _make_txn(
                category="Transportation",
                amount=Decimal("-75.00"),
                row_ordinal=2,
            ),
            # Refund should not count as spending
            _make_txn(
                category="Shopping",
                amount=Decimal("25.00"),
                is_return=True,
                row_ordinal=3,
            ),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Spending by category:" in output
        assert "Food & Dining:" in output
        assert "$150.00" in output
        assert "Transportation:" in output
        assert "$75.00" in output

    def test_spending_excludes_transfers(self, capsys):
        """Transfer transactions are excluded from spending by category."""
        txns = [
            _make_txn(
                category="Food & Dining",
                amount=Decimal("-100.00"),
                row_ordinal=0,
            ),
            _make_txn(
                amount=Decimal("-500.00"),
                is_transfer=True,
                row_ordinal=1,
            ),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "$100.00" in output
        assert "$500.00" not in output

    def test_warnings_printed(self, capsys):
        """Warnings are printed with a count header."""
        result = self._make_pipeline_result(
            warnings=[
                "chase/Activity2026.csv: skipped 1 malformed row (row 47)",
                "LLM: 3 transactions could not be parsed from response",
            ],
        )
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Warnings: 2" in output
        assert "skipped 1 malformed row" in output
        assert "could not be parsed" in output

    def test_errors_printed(self, capsys):
        """Errors are printed with a count header."""
        result = self._make_pipeline_result(
            errors=["Unknown parser 'bad_parser' for account 'Test'"],
        )
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Errors: 1" in output
        assert "Unknown parser" in output

    def test_no_warnings_no_section(self, capsys):
        """When there are no warnings, the Warnings section is omitted."""
        result = self._make_pipeline_result()
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Warnings:" not in output

    def test_no_errors_no_section(self, capsys):
        """When there are no errors, the Errors section is omitted."""
        result = self._make_pipeline_result()
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Errors:" not in output

    def test_empty_result(self, capsys):
        """An empty pipeline result produces a summary with zero counts."""
        result = self._make_pipeline_result()
        print_summary(result, "2026-03")
        output = capsys.readouterr().out

        assert "Processing Summary: 2026-03" in output
        assert "0 transactions" in output
        assert "0 transfers excluded" in output

    def test_enrichment_stat_shown(self, capsys):
        """Enriched (split) transactions show an enrichment line."""
        txns = [
            Transaction(
                transaction_id="parent123-1",
                date=date(2026, 1, 20),
                merchant="TARGET - Item A",
                description="Split item",
                amount=Decimal("-65.00"),
                institution="chase",
                account="Chase Credit Card",
                category="Kids",
                subcategory="Supplies",
                split_from="parent123",
            ),
            Transaction(
                transaction_id="parent123-2",
                date=date(2026, 1, 20),
                merchant="TARGET - Item B",
                description="Split item",
                amount=Decimal("-62.98"),
                institution="chase",
                account="Chase Credit Card",
                category="Shopping",
                split_from="parent123",
            ),
            _make_txn(row_ordinal=5),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Enriched: 2 split line items" in output

    def test_no_enrichment_stat(self, capsys):
        """When there are no splits, enrichment shows as 0."""
        txns = [_make_txn()]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        assert "Enriched: 0" in output

    def test_spending_categories_sorted_by_amount(self, capsys):
        """Spending categories are sorted by absolute amount, largest first."""
        txns = [
            _make_txn(
                category="Transportation",
                amount=Decimal("-50.00"),
                row_ordinal=0,
            ),
            _make_txn(
                category="Food & Dining",
                amount=Decimal("-200.00"),
                row_ordinal=1,
            ),
            _make_txn(
                category="Entertainment",
                amount=Decimal("-30.00"),
                row_ordinal=2,
            ),
        ]
        result = self._make_pipeline_result(transactions=txns)
        print_summary(result, "2026-01")
        output = capsys.readouterr().out

        # Food & Dining ($200) should come before Transportation ($50)
        # which should come before Entertainment ($30)
        food_pos = output.index("Food & Dining:")
        transport_pos = output.index("Transportation:")
        entertain_pos = output.index("Entertainment:")
        assert food_pos < transport_pos < entertain_pos
