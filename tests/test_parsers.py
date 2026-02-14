"""Tests for expense_tracker.parsers — bank CSV parsing and registry."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from expense_tracker.parsers import PARSERS, get_parser
from expense_tracker.parsers.capital_one import parse as capital_one_parse
from expense_tracker.parsers.chase import parse as chase_parse
from expense_tracker.parsers.elevations import parse as elevations_parse

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------


class TestParserRegistry:
    """Tests for the PARSERS dict and get_parser() function."""

    def test_registry_contains_all_parsers(self):
        """All three bank parsers are registered."""
        assert "chase" in PARSERS
        assert "capital_one" in PARSERS
        assert "elevations" in PARSERS

    def test_get_parser_returns_callable(self):
        """get_parser returns the correct parse function."""
        assert get_parser("chase") is chase_parse
        assert get_parser("capital_one") is capital_one_parse
        assert get_parser("elevations") is elevations_parse

    def test_get_parser_unknown_raises_key_error(self):
        """get_parser raises KeyError for unknown parser names."""
        with pytest.raises(KeyError):
            get_parser("nonexistent_bank")


# ---------------------------------------------------------------------------
# Chase parser
# ---------------------------------------------------------------------------


class TestChaseParser:
    """Tests for the Chase credit card CSV parser."""

    def test_happy_path(self):
        """Valid Chase CSV produces correct Transaction objects."""
        result = chase_parse(FIXTURES / "chase_valid.csv", "chase", "Chase CC")

        assert result.errors == []
        assert result.warnings == []
        assert len(result.transactions) == 4

        # First row: expense
        txn0 = result.transactions[0]
        assert txn0.date == date(2026, 1, 15)
        assert txn0.merchant == "CHIPOTLE MEXICAN GRIL"
        assert txn0.description == "CHIPOTLE MEXICAN GRIL"
        assert txn0.amount == Decimal("-12.50")
        assert txn0.institution == "chase"
        assert txn0.account == "Chase CC"
        assert txn0.is_return is False
        assert txn0.category == "Uncategorized"
        assert txn0.source_file == str(FIXTURES / "chase_valid.csv")

        # Second row: grocery expense
        txn1 = result.transactions[1]
        assert txn1.date == date(2026, 1, 16)
        assert txn1.merchant == "KING SOOPERS #123"
        assert txn1.amount == Decimal("-87.32")
        assert txn1.is_return is False

        # Third row: refund (positive amount)
        txn2 = result.transactions[2]
        assert txn2.date == date(2026, 1, 18)
        assert txn2.merchant == "AMAZON.COM REFUND"
        assert txn2.amount == Decimal("25.99")
        assert txn2.is_return is True

        # Fourth row
        txn3 = result.transactions[3]
        assert txn3.date == date(2026, 1, 20)
        assert txn3.amount == Decimal("-5.75")

    def test_transaction_ids_are_deterministic(self):
        """Parsing the same file twice produces the same transaction IDs."""
        result1 = chase_parse(FIXTURES / "chase_valid.csv", "chase", "Chase CC")
        result2 = chase_parse(FIXTURES / "chase_valid.csv", "chase", "Chase CC")

        ids1 = [t.transaction_id for t in result1.transactions]
        ids2 = [t.transaction_id for t in result2.transactions]
        assert ids1 == ids2

    def test_transaction_ids_are_12_hex_chars(self):
        """All generated IDs are 12 lowercase hex characters."""
        result = chase_parse(FIXTURES / "chase_valid.csv", "chase", "Chase CC")
        for txn in result.transactions:
            assert len(txn.transaction_id) == 12
            assert all(c in "0123456789abcdef" for c in txn.transaction_id)

    def test_malformed_rows_skipped_with_warnings(self):
        """Malformed rows are skipped and produce warnings, valid rows kept."""
        result = chase_parse(FIXTURES / "chase_malformed.csv", "chase", "Chase CC")

        # 21 data rows, 1 malformed (missing date)
        assert len(result.transactions) == 20
        assert len(result.warnings) == 1
        assert "missing date" in result.warnings[0]
        assert result.errors == []

    def test_wrong_format_returns_error(self):
        """A CSV with wrong columns returns an error and no transactions."""
        result = chase_parse(FIXTURES / "chase_wrong_format.csv", "chase", "Chase CC")

        assert result.transactions == []
        assert len(result.errors) == 1
        assert "missing expected columns" in result.errors[0]

    def test_empty_file_returns_no_transactions(self):
        """A CSV with headers but no data rows returns empty result."""
        result = chase_parse(FIXTURES / "chase_empty.csv", "chase", "Chase CC")

        assert result.transactions == []
        assert result.warnings == []
        assert result.errors == []

    def test_file_not_found_returns_error(self):
        """A nonexistent file path returns an error."""
        result = chase_parse(FIXTURES / "nonexistent.csv", "chase", "Chase CC")

        assert result.transactions == []
        assert len(result.errors) == 1
        assert "file not found" in result.errors[0]

    def test_too_many_malformed_rows_fails_file(self):
        """If >10% of rows are malformed, the entire file is failed."""
        result = chase_parse(FIXTURES / "chase_all_malformed.csv", "chase", "Chase CC")

        # All 4 rows are malformed (100% > 10%)
        assert result.transactions == []
        assert len(result.errors) == 1
        assert "too many malformed rows" in result.errors[0]
        # Warnings are still recorded for individual rows
        assert len(result.warnings) == 4

    def test_all_rows_returned_no_date_filtering(self):
        """Parser returns all rows regardless of date -- no month filtering."""
        result = chase_parse(FIXTURES / "chase_valid.csv", "chase", "Chase CC")

        dates = {txn.date for txn in result.transactions}
        assert len(dates) == 4  # All 4 distinct dates present


# ---------------------------------------------------------------------------
# Capital One parser
# ---------------------------------------------------------------------------


class TestCapitalOneParser:
    """Tests for the Capital One credit card CSV parser."""

    def test_happy_path(self):
        """Valid Capital One CSV produces correct Transaction objects."""
        result = capital_one_parse(FIXTURES / "capital_one_valid.csv", "capital_one", "Cap One CC")

        assert result.errors == []
        assert result.warnings == []
        assert len(result.transactions) == 4

        # First row: debit (expense) -> negative amount
        txn0 = result.transactions[0]
        assert txn0.date == date(2026, 1, 15)
        assert txn0.merchant == "WHOLE FOODS MARKET"
        assert txn0.amount == Decimal("-45.67")
        assert txn0.institution == "capital_one"
        assert txn0.account == "Cap One CC"
        assert txn0.is_return is False

        # Second row: another debit
        txn1 = result.transactions[1]
        assert txn1.date == date(2026, 1, 17)
        assert txn1.merchant == "UBER EATS"
        assert txn1.amount == Decimal("-22.10")
        assert txn1.is_return is False

        # Third row: credit (refund) -> positive amount
        txn2 = result.transactions[2]
        assert txn2.date == date(2026, 1, 19)
        assert txn2.merchant == "REFUND - ONLINE ORDER"
        assert txn2.amount == Decimal("15.00")
        assert txn2.is_return is True

        # Fourth row
        txn3 = result.transactions[3]
        assert txn3.date == date(2026, 1, 21)
        assert txn3.merchant == "NETFLIX.COM"
        assert txn3.amount == Decimal("-17.99")

    def test_transaction_ids_are_deterministic(self):
        """Parsing the same file twice produces the same transaction IDs."""
        result1 = capital_one_parse(FIXTURES / "capital_one_valid.csv", "capital_one", "Cap One CC")
        result2 = capital_one_parse(FIXTURES / "capital_one_valid.csv", "capital_one", "Cap One CC")

        ids1 = [t.transaction_id for t in result1.transactions]
        ids2 = [t.transaction_id for t in result2.transactions]
        assert ids1 == ids2

    def test_transaction_ids_are_12_hex_chars(self):
        """All generated IDs are 12 lowercase hex characters."""
        result = capital_one_parse(FIXTURES / "capital_one_valid.csv", "capital_one", "Cap One CC")
        for txn in result.transactions:
            assert len(txn.transaction_id) == 12
            assert all(c in "0123456789abcdef" for c in txn.transaction_id)

    def test_malformed_rows_skipped_with_warnings(self):
        """Malformed rows produce warnings; valid rows are kept."""
        result = capital_one_parse(
            FIXTURES / "capital_one_malformed.csv", "capital_one", "Cap One CC"
        )

        # 20 rows total, 1 malformed (missing date)
        assert len(result.transactions) == 19
        assert len(result.warnings) == 1
        assert "missing date" in result.warnings[0]
        assert result.errors == []

    def test_wrong_format_returns_error(self):
        """A CSV with wrong columns returns an error and no transactions."""
        result = capital_one_parse(
            FIXTURES / "capital_one_wrong_format.csv", "capital_one", "Cap One CC"
        )

        assert result.transactions == []
        assert len(result.errors) == 1
        assert "missing expected columns" in result.errors[0]

    def test_empty_file_returns_no_transactions(self):
        """A CSV with headers but no data rows returns empty result."""
        result = capital_one_parse(FIXTURES / "capital_one_empty.csv", "capital_one", "Cap One CC")

        assert result.transactions == []
        assert result.warnings == []
        assert result.errors == []

    def test_file_not_found_returns_error(self):
        """A nonexistent file path returns an error."""
        result = capital_one_parse(FIXTURES / "nonexistent.csv", "capital_one", "Cap One CC")

        assert result.transactions == []
        assert len(result.errors) == 1
        assert "file not found" in result.errors[0]

    def test_debit_credit_sign_convention(self):
        """Debits produce negative amounts, credits produce positive amounts."""
        result = capital_one_parse(FIXTURES / "capital_one_valid.csv", "capital_one", "Cap One CC")

        # Debit rows: negative
        debit_txns = [t for t in result.transactions if not t.is_return]
        for t in debit_txns:
            assert t.amount < 0, f"Debit {t.merchant} should have negative amount"

        # Credit rows: positive
        credit_txns = [t for t in result.transactions if t.is_return]
        for t in credit_txns:
            assert t.amount > 0, f"Credit {t.merchant} should have positive amount"


# ---------------------------------------------------------------------------
# Elevations parser
# ---------------------------------------------------------------------------


class TestElevationsParser:
    """Tests for the Elevations Credit Union CSV parser."""

    def test_happy_path(self):
        """Valid Elevations CSV produces correct Transaction objects."""
        result = elevations_parse(
            FIXTURES / "elevations_valid.csv", "elevations", "Elevations Checking"
        )

        assert result.errors == []
        assert result.warnings == []
        assert len(result.transactions) == 4

        # First row: debit (expense)
        txn0 = result.transactions[0]
        assert txn0.date == date(2026, 1, 10)
        assert txn0.merchant == "XCEL ENERGY PAYMENT"
        assert txn0.description == "XCEL ENERGY PAYMENT"
        assert txn0.amount == Decimal("-150.00")
        assert txn0.institution == "elevations"
        assert txn0.account == "Elevations Checking"
        assert txn0.is_return is False

        # Second row: large debit
        txn1 = result.transactions[1]
        assert txn1.date == date(2026, 1, 12)
        assert txn1.merchant == "MORTGAGE PAYMENT"
        assert txn1.amount == Decimal("-2500.00")
        assert txn1.is_return is False

        # Third row: credit (positive amount)
        txn2 = result.transactions[2]
        assert txn2.date == date(2026, 1, 14)
        assert txn2.merchant == "VENMO CASHBACK"
        assert txn2.amount == Decimal("50.00")
        assert txn2.is_return is True

        # Fourth row
        txn3 = result.transactions[3]
        assert txn3.date == date(2026, 1, 16)
        assert txn3.merchant == "KING SOOPERS #456"
        assert txn3.amount == Decimal("-42.50")

    def test_transaction_ids_are_deterministic(self):
        """Parsing the same file twice produces the same transaction IDs."""
        result1 = elevations_parse(
            FIXTURES / "elevations_valid.csv", "elevations", "Elevations Checking"
        )
        result2 = elevations_parse(
            FIXTURES / "elevations_valid.csv", "elevations", "Elevations Checking"
        )

        ids1 = [t.transaction_id for t in result1.transactions]
        ids2 = [t.transaction_id for t in result2.transactions]
        assert ids1 == ids2

    def test_transaction_ids_are_12_hex_chars(self):
        """All generated IDs are 12 lowercase hex characters."""
        result = elevations_parse(
            FIXTURES / "elevations_valid.csv", "elevations", "Elevations Checking"
        )
        for txn in result.transactions:
            assert len(txn.transaction_id) == 12
            assert all(c in "0123456789abcdef" for c in txn.transaction_id)

    def test_malformed_rows_skipped_with_warnings(self):
        """Malformed rows produce warnings; valid rows are kept."""
        result = elevations_parse(
            FIXTURES / "elevations_malformed.csv", "elevations", "Elevations Checking"
        )

        # 20 rows total, 1 malformed (missing date)
        assert len(result.transactions) == 19
        assert len(result.warnings) == 1
        assert "missing date" in result.warnings[0]
        assert result.errors == []

    def test_wrong_format_returns_error(self):
        """A CSV with wrong columns returns an error and no transactions."""
        result = elevations_parse(
            FIXTURES / "elevations_wrong_format.csv", "elevations", "Elevations Checking"
        )

        assert result.transactions == []
        assert len(result.errors) == 1
        assert "missing expected columns" in result.errors[0]

    def test_empty_file_returns_no_transactions(self):
        """A CSV with headers but no data rows returns empty result."""
        result = elevations_parse(
            FIXTURES / "elevations_empty.csv", "elevations", "Elevations Checking"
        )

        assert result.transactions == []
        assert result.warnings == []
        assert result.errors == []

    def test_file_not_found_returns_error(self):
        """A nonexistent file path returns an error."""
        result = elevations_parse(FIXTURES / "nonexistent.csv", "elevations", "Elevations Checking")

        assert result.transactions == []
        assert len(result.errors) == 1
        assert "file not found" in result.errors[0]

    def test_sign_convention_preserved(self):
        """Negative amounts stay negative, positive amounts stay positive."""
        result = elevations_parse(
            FIXTURES / "elevations_valid.csv", "elevations", "Elevations Checking"
        )

        negative_txns = [t for t in result.transactions if t.amount < 0]
        positive_txns = [t for t in result.transactions if t.amount > 0]

        assert len(negative_txns) == 3  # XCEL, MORTGAGE, KING SOOPERS
        assert len(positive_txns) == 1  # VENMO CASHBACK


# ---------------------------------------------------------------------------
# Cross-parser tests
# ---------------------------------------------------------------------------


class TestCrossParsers:
    """Tests verifying consistent behavior across all parsers."""

    @pytest.mark.parametrize(
        "parser_func,fixture,institution,account",
        [
            (chase_parse, "chase_valid.csv", "chase", "Chase CC"),
            (capital_one_parse, "capital_one_valid.csv", "capital_one", "Cap One CC"),
            (elevations_parse, "elevations_valid.csv", "elevations", "Elevations Checking"),
        ],
    )
    def test_all_parsers_return_stage_result(self, parser_func, fixture, institution, account):
        """Every parser returns a StageResult with the expected attributes."""
        from expense_tracker.models import StageResult

        result = parser_func(FIXTURES / fixture, institution, account)

        assert isinstance(result, StageResult)
        assert isinstance(result.transactions, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.errors, list)

    @pytest.mark.parametrize(
        "parser_func,fixture,institution,account",
        [
            (chase_parse, "chase_valid.csv", "chase", "Chase CC"),
            (capital_one_parse, "capital_one_valid.csv", "capital_one", "Cap One CC"),
            (elevations_parse, "elevations_valid.csv", "elevations", "Elevations Checking"),
        ],
    )
    def test_all_parsers_set_source_file(self, parser_func, fixture, institution, account):
        """Every parser sets source_file on each transaction."""
        result = parser_func(FIXTURES / fixture, institution, account)

        for txn in result.transactions:
            assert txn.source_file != ""
            assert fixture.replace(".csv", "") in txn.source_file

    @pytest.mark.parametrize(
        "parser_func,fixture,institution,account",
        [
            (chase_parse, "chase_valid.csv", "chase", "Chase CC"),
            (capital_one_parse, "capital_one_valid.csv", "capital_one", "Cap One CC"),
            (elevations_parse, "elevations_valid.csv", "elevations", "Elevations Checking"),
        ],
    )
    def test_all_parsers_produce_unique_ids_within_file(
        self, parser_func, fixture, institution, account
    ):
        """Each transaction in a file gets a unique ID."""
        result = parser_func(FIXTURES / fixture, institution, account)

        ids = [t.transaction_id for t in result.transactions]
        assert len(ids) == len(set(ids)), "Duplicate transaction IDs found"

    @pytest.mark.parametrize(
        "parser_func,institution,account",
        [
            (chase_parse, "chase", "Chase CC"),
            (capital_one_parse, "capital_one", "Cap One CC"),
            (elevations_parse, "elevations", "Elevations Checking"),
        ],
    )
    def test_all_parsers_handle_missing_file(self, parser_func, institution, account):
        """All parsers gracefully handle a file that does not exist."""
        result = parser_func(Path("/tmp/does_not_exist_12345.csv"), institution, account)

        assert result.transactions == []
        assert len(result.errors) == 1
