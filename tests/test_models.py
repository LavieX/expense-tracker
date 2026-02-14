"""Tests for expense_tracker.models — dataclass construction and ID generation."""

from datetime import date
from decimal import Decimal

from expense_tracker.models import (
    AccountConfig,
    AppConfig,
    MerchantRule,
    StageResult,
    Transaction,
    generate_transaction_id,
)

# ---------------------------------------------------------------------------
# generate_transaction_id
# ---------------------------------------------------------------------------


class TestGenerateTransactionId:
    """Tests for deterministic transaction ID generation."""

    def test_basic_determinism(self):
        """Same inputs always produce the same ID."""
        kwargs = dict(
            institution="chase",
            txn_date=date(2026, 1, 15),
            merchant="CHIPOTLE MEXICAN GRIL",
            amount=Decimal("12.50"),
            row_ordinal=0,
        )
        id1 = generate_transaction_id(**kwargs)
        id2 = generate_transaction_id(**kwargs)
        assert id1 == id2

    def test_id_is_12_hex_chars(self):
        """ID should be exactly 12 lowercase hex characters."""
        tid = generate_transaction_id(
            institution="chase",
            txn_date=date(2026, 1, 15),
            merchant="STARBUCKS",
            amount=Decimal("5.75"),
            row_ordinal=0,
        )
        assert len(tid) == 12
        assert all(c in "0123456789abcdef" for c in tid)

    def test_different_institutions_produce_different_ids(self):
        """Changing the institution changes the ID."""
        common = dict(
            txn_date=date(2026, 1, 15),
            merchant="TARGET",
            amount=Decimal("42.00"),
            row_ordinal=0,
        )
        id_chase = generate_transaction_id(institution="chase", **common)
        id_cap1 = generate_transaction_id(institution="capital_one", **common)
        assert id_chase != id_cap1

    def test_different_dates_produce_different_ids(self):
        """Changing the date changes the ID."""
        common = dict(
            institution="chase",
            merchant="TARGET",
            amount=Decimal("42.00"),
            row_ordinal=0,
        )
        id_jan = generate_transaction_id(txn_date=date(2026, 1, 15), **common)
        id_feb = generate_transaction_id(txn_date=date(2026, 2, 15), **common)
        assert id_jan != id_feb

    def test_different_amounts_produce_different_ids(self):
        """Changing the amount changes the ID."""
        common = dict(
            institution="chase",
            txn_date=date(2026, 1, 15),
            merchant="TARGET",
            row_ordinal=0,
        )
        id_a = generate_transaction_id(amount=Decimal("42.00"), **common)
        id_b = generate_transaction_id(amount=Decimal("42.01"), **common)
        assert id_a != id_b

    def test_same_merchant_twice_same_day_different_ordinals(self):
        """Two purchases at the same merchant on the same day are
        distinguished by their row ordinal."""
        common = dict(
            institution="chase",
            txn_date=date(2026, 1, 15),
            merchant="STARBUCKS",
            amount=Decimal("5.75"),
        )
        id_row0 = generate_transaction_id(row_ordinal=0, **common)
        id_row1 = generate_transaction_id(row_ordinal=1, **common)
        assert id_row0 != id_row1

    def test_same_merchant_twice_same_day_same_ordinal(self):
        """Identical inputs (including ordinal) produce the same ID.
        This is the expected collision case -- same row re-processed
        yields the same ID for deduplication."""
        common = dict(
            institution="chase",
            txn_date=date(2026, 1, 15),
            merchant="STARBUCKS",
            amount=Decimal("5.75"),
            row_ordinal=3,
        )
        assert generate_transaction_id(**common) == generate_transaction_id(**common)

    def test_merchant_case_insensitivity(self):
        """Merchant name is uppercased before hashing, so case does not
        affect the ID."""
        common = dict(
            institution="chase",
            txn_date=date(2026, 1, 15),
            amount=Decimal("10.00"),
            row_ordinal=0,
        )
        id_lower = generate_transaction_id(merchant="starbucks", **common)
        id_upper = generate_transaction_id(merchant="STARBUCKS", **common)
        id_mixed = generate_transaction_id(merchant="Starbucks", **common)
        assert id_lower == id_upper == id_mixed

    def test_merchant_whitespace_stripped(self):
        """Leading/trailing whitespace on merchant is stripped before hashing."""
        common = dict(
            institution="chase",
            txn_date=date(2026, 1, 15),
            amount=Decimal("10.00"),
            row_ordinal=0,
        )
        id_clean = generate_transaction_id(merchant="STARBUCKS", **common)
        id_padded = generate_transaction_id(merchant="  STARBUCKS  ", **common)
        assert id_clean == id_padded

    def test_negative_amount(self):
        """Negative amounts (expenses) produce valid IDs and differ from
        positive amounts."""
        common = dict(
            institution="elevations",
            txn_date=date(2026, 3, 1),
            merchant="TRANSFER",
            row_ordinal=0,
        )
        id_neg = generate_transaction_id(amount=Decimal("-100.00"), **common)
        id_pos = generate_transaction_id(amount=Decimal("100.00"), **common)
        assert len(id_neg) == 12
        assert id_neg != id_pos


# ---------------------------------------------------------------------------
# Transaction dataclass
# ---------------------------------------------------------------------------


class TestTransaction:
    """Tests for Transaction dataclass construction."""

    def test_minimal_construction(self):
        """Transaction can be constructed with required fields only,
        using defaults for optional fields."""
        txn = Transaction(
            transaction_id="abcdef012345",
            date=date(2026, 1, 15),
            merchant="CHIPOTLE MEXICAN GRIL",
            description="CHIPOTLE MEXICAN GRIL  BOULDER CO",
            amount=Decimal("-12.50"),
            institution="chase",
            account="Chase Credit Card",
        )
        assert txn.transaction_id == "abcdef012345"
        assert txn.date == date(2026, 1, 15)
        assert txn.merchant == "CHIPOTLE MEXICAN GRIL"
        assert txn.amount == Decimal("-12.50")
        assert txn.category == "Uncategorized"
        assert txn.subcategory == ""
        assert txn.is_transfer is False
        assert txn.is_return is False
        assert txn.split_from == ""
        assert txn.source_file == ""

    def test_full_construction(self):
        """Transaction can be constructed with all fields specified."""
        txn = Transaction(
            transaction_id="abcdef012345",
            date=date(2026, 1, 15),
            merchant="KING SOOPERS",
            description="KING SOOPERS #123 BOULDER CO",
            amount=Decimal("-87.32"),
            institution="capital_one",
            account="Capital One Credit Card",
            category="Food & Dining",
            subcategory="Groceries",
            is_transfer=False,
            is_return=False,
            split_from="",
            source_file="input/capital-one/Activity2026.csv",
        )
        assert txn.category == "Food & Dining"
        assert txn.subcategory == "Groceries"
        assert txn.source_file == "input/capital-one/Activity2026.csv"

    def test_return_transaction(self):
        """A refund/return has a positive amount and is_return=True."""
        txn = Transaction(
            transaction_id="return123456",
            date=date(2026, 2, 1),
            merchant="AMAZON RETURNS",
            description="AMAZON.COM REFUND",
            amount=Decimal("25.99"),
            institution="chase",
            account="Chase Credit Card",
            is_return=True,
        )
        assert txn.amount > 0
        assert txn.is_return is True

    def test_split_transaction(self):
        """A split transaction references its parent via split_from."""
        txn = Transaction(
            transaction_id="abcdef012345-1",
            date=date(2026, 1, 20),
            merchant="TARGET ITEM 1",
            description="Split from TARGET purchase",
            amount=Decimal("-15.00"),
            institution="chase",
            account="Chase Credit Card",
            split_from="abcdef012345",
        )
        assert txn.split_from == "abcdef012345"


# ---------------------------------------------------------------------------
# StageResult dataclass
# ---------------------------------------------------------------------------


class TestStageResult:
    """Tests for StageResult dataclass construction."""

    def test_empty_result(self):
        """Default StageResult has empty lists."""
        result = StageResult()
        assert result.transactions == []
        assert result.warnings == []
        assert result.errors == []

    def test_result_with_data(self):
        """StageResult holds transactions, warnings, and errors."""
        txn = Transaction(
            transaction_id="aaa111222333",
            date=date(2026, 1, 1),
            merchant="TEST",
            description="Test transaction",
            amount=Decimal("-1.00"),
            institution="chase",
            account="test",
        )
        result = StageResult(
            transactions=[txn],
            warnings=["skipped 1 malformed row"],
            errors=["file not found: missing.csv"],
        )
        assert len(result.transactions) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 1

    def test_mutable_default_isolation(self):
        """Each StageResult instance gets its own list objects (no shared
        mutable defaults)."""
        r1 = StageResult()
        r2 = StageResult()
        r1.warnings.append("warning from r1")
        assert r2.warnings == []


# ---------------------------------------------------------------------------
# MerchantRule dataclass
# ---------------------------------------------------------------------------


class TestMerchantRule:
    """Tests for MerchantRule dataclass construction."""

    def test_user_rule(self):
        """User rule with explicit source."""
        rule = MerchantRule(
            pattern="KING SOOPERS",
            category="Food & Dining",
            subcategory="Groceries",
            source="user",
        )
        assert rule.pattern == "KING SOOPERS"
        assert rule.category == "Food & Dining"
        assert rule.subcategory == "Groceries"
        assert rule.source == "user"

    def test_learned_rule(self):
        """Learned rule created by the learn command."""
        rule = MerchantRule(
            pattern="CHIPOTLE",
            category="Food & Dining",
            subcategory="Fast Food",
            source="learned",
        )
        assert rule.source == "learned"

    def test_defaults(self):
        """Subcategory defaults to empty string, source defaults to 'user'."""
        rule = MerchantRule(pattern="NETFLIX", category="Entertainment")
        assert rule.subcategory == ""
        assert rule.source == "user"


# ---------------------------------------------------------------------------
# AccountConfig dataclass
# ---------------------------------------------------------------------------


class TestAccountConfig:
    """Tests for AccountConfig dataclass construction."""

    def test_construction(self):
        """AccountConfig stores all account configuration fields."""
        account = AccountConfig(
            name="Chase Credit Card",
            institution="chase",
            parser="chase",
            account_type="credit_card",
            input_dir="input/chase",
        )
        assert account.name == "Chase Credit Card"
        assert account.institution == "chase"
        assert account.parser == "chase"
        assert account.account_type == "credit_card"
        assert account.input_dir == "input/chase"

    def test_checking_account(self):
        """AccountConfig works for checking accounts too."""
        account = AccountConfig(
            name="Elevations Credit Union",
            institution="elevations",
            parser="elevations",
            account_type="checking",
            input_dir="input/elevations",
        )
        assert account.account_type == "checking"


# ---------------------------------------------------------------------------
# AppConfig dataclass
# ---------------------------------------------------------------------------


class TestAppConfig:
    """Tests for AppConfig dataclass construction."""

    def test_defaults(self):
        """AppConfig has sensible defaults for all fields."""
        config = AppConfig()
        assert config.accounts == []
        assert config.output_dir == "output"
        assert config.enrichment_cache_dir == "enrichment-cache"
        assert config.transfer_keywords == ["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"]
        assert config.transfer_date_window == 5
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.llm_api_key_env == "ANTHROPIC_API_KEY"

    def test_full_construction(self):
        """AppConfig can be constructed with all fields specified."""
        accounts = [
            AccountConfig(
                name="Chase Credit Card",
                institution="chase",
                parser="chase",
                account_type="credit_card",
                input_dir="input/chase",
            ),
            AccountConfig(
                name="Elevations Credit Union",
                institution="elevations",
                parser="elevations",
                account_type="checking",
                input_dir="input/elevations",
            ),
        ]
        config = AppConfig(
            accounts=accounts,
            output_dir="custom-output",
            enrichment_cache_dir="custom-cache",
            transfer_keywords=["PAYMENT", "TRANSFER"],
            transfer_date_window=3,
            llm_provider="none",
            llm_model="",
            llm_api_key_env="",
        )
        assert len(config.accounts) == 2
        assert config.output_dir == "custom-output"
        assert config.transfer_date_window == 3
        assert config.llm_provider == "none"

    def test_mutable_default_isolation(self):
        """Each AppConfig instance gets its own list objects."""
        c1 = AppConfig()
        c2 = AppConfig()
        c1.accounts.append(
            AccountConfig(
                name="Test",
                institution="test",
                parser="test",
                account_type="checking",
                input_dir="input/test",
            )
        )
        assert c2.accounts == []
        c1.transfer_keywords.append("WIRE")
        assert "WIRE" not in c2.transfer_keywords
