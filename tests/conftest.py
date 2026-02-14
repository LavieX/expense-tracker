"""Shared pytest fixtures for Expense Tracker tests.

Provides reusable fixtures for:
- tmp_project_dir: A temporary directory with full project structure (input dirs,
  config files, output dirs) for integration-style testing.
- sample_transactions: A list of realistic Transaction objects spanning multiple
  months, institutions, and edge cases (refunds, transfers, uncategorized).
- sample_rules: A list of MerchantRule objects covering user and learned rules.
- Convenience fixtures for fixture file paths.
"""

from __future__ import annotations

import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from expense_tracker.models import (
    MerchantRule,
    Transaction,
    generate_transaction_id,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_CONFIG_DIR = FIXTURES_DIR / "config"


# ---------------------------------------------------------------------------
# Fixture file path helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the tests/fixtures/ directory."""
    return FIXTURES_DIR


@pytest.fixture
def chase_sample_csv() -> Path:
    """Path to the Chase sample CSV fixture file."""
    return FIXTURES_DIR / "chase_sample.csv"


@pytest.fixture
def capital_one_sample_csv() -> Path:
    """Path to the Capital One sample CSV fixture file."""
    return FIXTURES_DIR / "capital_one_sample.csv"


@pytest.fixture
def elevations_sample_csv() -> Path:
    """Path to the Elevations sample CSV fixture file."""
    return FIXTURES_DIR / "elevations_sample.csv"


# ---------------------------------------------------------------------------
# tmp_project_dir -- temp directory with full project structure
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with the full Expense Tracker project structure.

    The directory contains:
    - config.toml, categories.toml, rules.toml (copied from test fixtures)
    - input/chase/ with chase_sample.csv
    - input/capital-one/ with capital_one_sample.csv
    - input/elevations/ with elevations_sample.csv
    - output/ (empty)
    - enrichment-cache/ (empty)

    Returns the Path to the temporary project root. The directory is
    automatically cleaned up after the test completes.
    """
    project = tmp_path / "expense-project"
    project.mkdir()

    # Create config files in project root
    for config_file in ("config.toml", "categories.toml", "rules.toml"):
        shutil.copy2(FIXTURES_CONFIG_DIR / config_file, project / config_file)

    # Create input directories with sample CSVs
    input_chase = project / "input" / "chase"
    input_chase.mkdir(parents=True)
    shutil.copy2(FIXTURES_DIR / "chase_sample.csv", input_chase / "Activity2026.csv")

    input_cap1 = project / "input" / "capital-one"
    input_cap1.mkdir(parents=True)
    shutil.copy2(
        FIXTURES_DIR / "capital_one_sample.csv",
        input_cap1 / "Activity2026.csv",
    )

    input_elev = project / "input" / "elevations"
    input_elev.mkdir(parents=True)
    shutil.copy2(
        FIXTURES_DIR / "elevations_sample.csv",
        input_elev / "Activity2026.csv",
    )

    # Create output and enrichment-cache directories
    (project / "output").mkdir()
    (project / "enrichment-cache").mkdir()

    return project


# ---------------------------------------------------------------------------
# sample_transactions -- realistic Transaction objects for pipeline testing
# ---------------------------------------------------------------------------


def _make_txn(
    institution: str,
    account: str,
    txn_date: date,
    merchant: str,
    description: str,
    amount: Decimal,
    row_ordinal: int,
    category: str = "Uncategorized",
    subcategory: str = "",
    is_transfer: bool = False,
    is_return: bool = False,
    source_file: str = "",
) -> Transaction:
    """Helper to build a Transaction with a deterministic ID."""
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
    )


@pytest.fixture
def sample_transactions() -> list[Transaction]:
    """A list of 15 realistic Transaction objects for pipeline testing.

    Includes:
    - Transactions from all three institutions (chase, capital_one, elevations)
    - Dates spanning January and February 2026
    - 1 refund (positive amount, is_return=True)
    - 1 transfer-like payment (is_transfer=True on two sides)
    - Mix of categorized and uncategorized transactions
    - Variety of merchants and amounts
    """
    return [
        # -- Chase credit card transactions --
        _make_txn(
            institution="chase",
            account="Chase Credit Card",
            txn_date=date(2026, 1, 15),
            merchant="WHOLEFDS LMT #10554",
            description="WHOLEFDS LMT #10554",
            amount=Decimal("-171.48"),
            row_ordinal=0,
            category="Food & Dining",
            subcategory="Groceries",
            source_file="input/chase/Activity2026.csv",
        ),
        _make_txn(
            institution="chase",
            account="Chase Credit Card",
            txn_date=date(2026, 1, 18),
            merchant="CHIPOTLE MEXICAN GRIL",
            description="CHIPOTLE MEXICAN GRIL  BOULDER CO",
            amount=Decimal("-12.50"),
            row_ordinal=1,
            category="Food & Dining",
            subcategory="Fast Food",
            source_file="input/chase/Activity2026.csv",
        ),
        _make_txn(
            institution="chase",
            account="Chase Credit Card",
            txn_date=date(2026, 1, 20),
            merchant="SHELL OIL 574726183",
            description="SHELL OIL 574726183",
            amount=Decimal("-48.75"),
            row_ordinal=2,
            category="Transportation",
            subcategory="Gas/Fuel",
            source_file="input/chase/Activity2026.csv",
        ),
        _make_txn(
            institution="chase",
            account="Chase Credit Card",
            txn_date=date(2026, 1, 25),
            merchant="TARGET 00022186",
            description="TARGET        00022186",
            amount=Decimal("-127.98"),
            row_ordinal=3,
            source_file="input/chase/Activity2026.csv",
        ),
        _make_txn(
            institution="chase",
            account="Chase Credit Card",
            txn_date=date(2026, 2, 8),
            merchant="AMAZON.COM REFUND",
            description="AMAZON.COM REFUND",
            amount=Decimal("25.99"),
            row_ordinal=4,
            is_return=True,
            category="Shopping",
            source_file="input/chase/Activity2026.csv",
        ),
        # -- Capital One credit card transactions --
        _make_txn(
            institution="capital_one",
            account="Capital One Credit Card",
            txn_date=date(2026, 1, 10),
            merchant="FIRST WATCH 0307 PAT",
            description="FIRST WATCH 0307 PAT",
            amount=Decimal("-41.78"),
            row_ordinal=0,
            category="Food & Dining",
            subcategory="Restaurant",
            source_file="input/capital-one/Activity2026.csv",
        ),
        _make_txn(
            institution="capital_one",
            account="Capital One Credit Card",
            txn_date=date(2026, 1, 12),
            merchant="REI.COM 800-426-4840",
            description="REI.COM  800-426-4840",
            amount=Decimal("-160.65"),
            row_ordinal=1,
            source_file="input/capital-one/Activity2026.csv",
        ),
        _make_txn(
            institution="capital_one",
            account="Capital One Credit Card",
            txn_date=date(2026, 2, 2),
            merchant="PETCO 1234",
            description="PETCO 1234",
            amount=Decimal("-45.67"),
            row_ordinal=2,
            category="Pets",
            subcategory="Supplies",
            source_file="input/capital-one/Activity2026.csv",
        ),
        _make_txn(
            institution="capital_one",
            account="Capital One Credit Card",
            txn_date=date(2026, 2, 12),
            merchant="CAPITAL ONE AUTOPAY PYMT",
            description="CAPITAL ONE AUTOPAY PYMT",
            amount=Decimal("474.90"),
            row_ordinal=3,
            is_transfer=True,
            is_return=False,
            source_file="input/capital-one/Activity2026.csv",
        ),
        _make_txn(
            institution="capital_one",
            account="Capital One Credit Card",
            txn_date=date(2026, 2, 5),
            merchant="CVS/PHARMACY #08432",
            description="CVS/PHARMACY #08432",
            amount=Decimal("-18.50"),
            row_ordinal=4,
            category="Healthcare",
            subcategory="Pharmacy",
            source_file="input/capital-one/Activity2026.csv",
        ),
        # -- Elevations checking account transactions --
        _make_txn(
            institution="elevations",
            account="Elevations Credit Union",
            txn_date=date(2026, 1, 15),
            merchant="PRIMROSE SCHOOL",
            description="PRIMROSE SCHOOL TYPE: 3037741919  ID: 1470259040 CO: PRIMROSE SCHOOL",
            amount=Decimal("-445.00"),
            row_ordinal=0,
            category="Kids",
            subcategory="Preschool",
            source_file="input/elevations/Activity2026.csv",
        ),
        _make_txn(
            institution="elevations",
            account="Elevations Credit Union",
            txn_date=date(2026, 1, 18),
            merchant="LPC Nextlight",
            description="LPC Nextlight TYPE: LPC BB  ID: 4846000608 CO: LPC Nextlight",
            amount=Decimal("-49.95"),
            row_ordinal=1,
            category="Utilities",
            subcategory="Electric/Water/Internet",
            source_file="input/elevations/Activity2026.csv",
        ),
        _make_txn(
            institution="elevations",
            account="Elevations Credit Union",
            txn_date=date(2026, 2, 5),
            merchant="CHASE CREDIT CRD",
            description="CHASE CREDIT CRD TYPE: AUTOPAY  ID: 4760039224 CO: CHASE CREDIT CRD",
            amount=Decimal("-2150.00"),
            row_ordinal=2,
            is_transfer=True,
            source_file="input/elevations/Activity2026.csv",
        ),
        _make_txn(
            institution="elevations",
            account="Elevations Credit Union",
            txn_date=date(2026, 2, 8),
            merchant="CAPITAL ONE",
            description="CAPITAL ONE TYPE: CRCARDPMT  ID: 9541719318 CO: CAPITAL ONE  NAME: LAVIE TOBEY",
            amount=Decimal("-474.90"),
            row_ordinal=3,
            is_transfer=True,
            source_file="input/elevations/Activity2026.csv",
        ),
        _make_txn(
            institution="elevations",
            account="Elevations Credit Union",
            txn_date=date(2026, 2, 1),
            merchant="VMWARE INC",
            description="VMWARE INC TYPE: PAYROLL  ID: 9111111103 CO: VMWARE INC",
            amount=Decimal("2254.14"),
            row_ordinal=4,
            source_file="input/elevations/Activity2026.csv",
        ),
    ]


# ---------------------------------------------------------------------------
# sample_rules -- MerchantRule objects for categorizer testing
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rules() -> list[MerchantRule]:
    """A list of MerchantRule objects for testing the categorization engine.

    Includes:
    - 9 user rules (hand-authored, never overwritten)
    - 6 learned rules (system-managed via the learn command)
    - Rules covering groceries, dining, gas, entertainment, kids, health,
      pets, pharmacy, and shopping
    - Both rules with subcategories and rules without

    Rules are ordered with user rules first, then learned rules, matching
    the ordering that config.load_rules() produces.
    """
    return [
        # User rules
        MerchantRule(
            pattern="KING SOOPERS",
            category="Food & Dining",
            subcategory="Groceries",
            source="user",
        ),
        MerchantRule(
            pattern="CHIPOTLE",
            category="Food & Dining",
            subcategory="Fast Food",
            source="user",
        ),
        MerchantRule(
            pattern="WHOLEFDS",
            category="Food & Dining",
            subcategory="Groceries",
            source="user",
        ),
        MerchantRule(
            pattern="SPROUTS FARMERS",
            category="Food & Dining",
            subcategory="Groceries",
            source="user",
        ),
        MerchantRule(
            pattern="STARBUCKS",
            category="Food & Dining",
            subcategory="Coffee",
            source="user",
        ),
        MerchantRule(
            pattern="NETFLIX",
            category="Entertainment",
            subcategory="Subscriptions",
            source="user",
        ),
        MerchantRule(
            pattern="SHELL OIL",
            category="Transportation",
            subcategory="Gas/Fuel",
            source="user",
        ),
        MerchantRule(
            pattern="PRIMROSE SCHOOL",
            category="Kids",
            subcategory="Preschool",
            source="user",
        ),
        MerchantRule(
            pattern="PLANET FITNESS",
            category="Health & Fitness",
            subcategory="Gym/Classes",
            source="user",
        ),
        # Learned rules
        MerchantRule(
            pattern="TARGET",
            category="Shopping",
            subcategory="",
            source="learned",
        ),
        MerchantRule(
            pattern="APPLE.COM/BILL",
            category="Entertainment",
            subcategory="Subscriptions",
            source="learned",
        ),
        MerchantRule(
            pattern="SPOTIFY",
            category="Entertainment",
            subcategory="Subscriptions",
            source="learned",
        ),
        MerchantRule(
            pattern="CVS/PHARMACY",
            category="Healthcare",
            subcategory="Pharmacy",
            source="learned",
        ),
        MerchantRule(
            pattern="PETCO",
            category="Pets",
            subcategory="Supplies",
            source="learned",
        ),
        MerchantRule(
            pattern="CHEWY.COM",
            category="Pets",
            subcategory="Food",
            source="learned",
        ),
    ]
