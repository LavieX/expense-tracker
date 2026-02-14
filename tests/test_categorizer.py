"""Tests for the categorization engine.

Covers:
- match_rules: overlapping patterns, tie-breaking, case insensitivity,
  user-before-learned ordering, no-match cases.
- categorize: rule matching integration, LLM fallback with a mock adapter,
  already-categorized transactions, LLM failure handling, no-LLM mode.
- learn: new rules from corrections, updating existing learned rules,
  user-rule conflict skipping, unchanged transactions.
"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from expense_tracker.categorizer import (
    LLMAdapter,
    categorize,
    learn,
    match_rules,
)
from expense_tracker.models import (
    LearnResult,
    MerchantRule,
    StageResult,
    Transaction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_txn(
    merchant: str,
    *,
    description: str = "",
    amount: Decimal = Decimal("-10.00"),
    txn_date: date = date(2026, 1, 15),
    category: str = "Uncategorized",
    subcategory: str = "",
    transaction_id: str = "",
) -> Transaction:
    """Build a minimal Transaction for categorizer tests."""
    if not description:
        description = merchant
    if not transaction_id:
        transaction_id = f"test_{merchant[:8].lower().replace(' ', '_')}"
    return Transaction(
        transaction_id=transaction_id,
        date=txn_date,
        merchant=merchant,
        description=description,
        amount=amount,
        institution="chase",
        account="Chase Credit Card",
        category=category,
        subcategory=subcategory,
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a list of dicts as a CSV file with standard output columns."""
    fieldnames = [
        "transaction_id",
        "date",
        "merchant",
        "description",
        "amount",
        "institution",
        "account",
        "category",
        "subcategory",
        "is_return",
        "split_from",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MockLLMAdapter:
    """A mock LLM adapter for testing the categorizer's LLM fallback.

    Accepts a dict mapping merchant names (uppercased) to (category,
    subcategory) tuples.  ``categorize_batch`` returns suggestions for
    merchants found in the mapping.
    """

    def __init__(
        self,
        suggestions: dict[str, tuple[str, str]] | None = None,
        *,
        should_fail: bool = False,
    ):
        self.suggestions = suggestions or {}
        self.should_fail = should_fail
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        self.calls.append((transactions, categories))
        if self.should_fail:
            raise ConnectionError("LLM service unavailable")
        result = []
        for txn in transactions:
            merchant = txn["merchant"]
            key = merchant.upper()
            if key in self.suggestions:
                cat, subcat = self.suggestions[key]
                result.append(
                    {"merchant": merchant, "category": cat, "subcategory": subcat}
                )
        return result


SAMPLE_CATEGORIES = [
    {"name": "Food & Dining", "subcategories": ["Groceries", "Fast Food", "Coffee"]},
    {"name": "Transportation", "subcategories": ["Gas/Fuel"]},
    {"name": "Shopping", "subcategories": ["Clothing", "Electronics"]},
    {"name": "Entertainment", "subcategories": ["Subscriptions"]},
    {"name": "Healthcare", "subcategories": ["Pharmacy"]},
]


# ===================================================================
# match_rules tests
# ===================================================================


class TestMatchRules:
    """Tests for the match_rules function."""

    def test_exact_substring_match(self):
        """A rule pattern that is a substring of the merchant should match."""
        rules = [
            MerchantRule(pattern="CHIPOTLE", category="Food & Dining", subcategory="Fast Food"),
        ]
        result = match_rules("CHIPOTLE MEXICAN GRIL", rules)
        assert result is not None
        assert result.pattern == "CHIPOTLE"
        assert result.category == "Food & Dining"

    def test_case_insensitive_matching(self):
        """Matching should be case-insensitive on both merchant and pattern."""
        rules = [
            MerchantRule(pattern="chipotle", category="Food & Dining", subcategory="Fast Food"),
        ]
        result = match_rules("CHIPOTLE MEXICAN GRIL", rules)
        assert result is not None
        assert result.category == "Food & Dining"

    def test_case_insensitive_merchant(self):
        """A lowercase merchant should still match an uppercase pattern."""
        rules = [
            MerchantRule(pattern="STARBUCKS", category="Food & Dining", subcategory="Coffee"),
        ]
        result = match_rules("starbucks store #1234", rules)
        assert result is not None
        assert result.category == "Food & Dining"

    def test_no_match_returns_none(self):
        """When no rule matches, None should be returned."""
        rules = [
            MerchantRule(pattern="CHIPOTLE", category="Food & Dining", subcategory="Fast Food"),
        ]
        result = match_rules("SUBWAY RESTAURANT", rules)
        assert result is None

    def test_empty_rules_returns_none(self):
        """An empty rule list should return None."""
        result = match_rules("CHIPOTLE MEXICAN GRIL", [])
        assert result is None

    def test_longest_pattern_wins(self):
        """When multiple rules match, the longest pattern wins."""
        rules = [
            MerchantRule(
                pattern="TARGET",
                category="Shopping",
                subcategory="",
                source="learned",
            ),
            MerchantRule(
                pattern="TARGET 00022186",
                category="Shopping",
                subcategory="Home Goods",
                source="learned",
            ),
        ]
        result = match_rules("TARGET 00022186", rules)
        assert result is not None
        assert result.pattern == "TARGET 00022186"
        assert result.subcategory == "Home Goods"

    def test_longest_pattern_wins_regardless_of_order(self):
        """Longest match wins even if a shorter rule appears first."""
        rules = [
            MerchantRule(
                pattern="SHELL",
                category="Shopping",
                subcategory="",
                source="user",
            ),
            MerchantRule(
                pattern="SHELL OIL",
                category="Transportation",
                subcategory="Gas/Fuel",
                source="learned",
            ),
        ]
        result = match_rules("SHELL OIL 574726183", rules)
        assert result is not None
        assert result.pattern == "SHELL OIL"
        assert result.category == "Transportation"

    def test_tie_broken_by_list_order_user_before_learned(self):
        """When two rules have equal-length patterns, the first in the list wins.

        Since user rules come before learned rules in the list, user rules
        win ties.
        """
        rules = [
            MerchantRule(
                pattern="NETFLIX",
                category="Entertainment",
                subcategory="Subscriptions",
                source="user",
            ),
            MerchantRule(
                pattern="NETFLIX",
                category="Shopping",
                subcategory="",
                source="learned",
            ),
        ]
        result = match_rules("NETFLIX.COM/BILL", rules)
        assert result is not None
        assert result.source == "user"
        assert result.category == "Entertainment"

    def test_tie_broken_by_insertion_order_within_same_source(self):
        """When two same-source rules have equal-length patterns, first wins."""
        rules = [
            MerchantRule(
                pattern="CVS",
                category="Healthcare",
                subcategory="Pharmacy",
                source="learned",
            ),
            MerchantRule(
                pattern="CVS",
                category="Shopping",
                subcategory="",
                source="learned",
            ),
        ]
        result = match_rules("CVS STORE #1234", rules)
        assert result is not None
        assert result.category == "Healthcare"

    def test_multiple_overlapping_patterns(self):
        """Three overlapping rules: the longest match wins."""
        rules = [
            MerchantRule(pattern="KING", category="Misc", subcategory="", source="user"),
            MerchantRule(
                pattern="KING SOOPERS",
                category="Food & Dining",
                subcategory="Groceries",
                source="user",
            ),
            MerchantRule(
                pattern="KING SOOPERS FUEL",
                category="Transportation",
                subcategory="Gas/Fuel",
                source="user",
            ),
        ]
        # Should match "KING SOOPERS FUEL" (longest)
        result = match_rules("KING SOOPERS FUEL #0047", rules)
        assert result is not None
        assert result.pattern == "KING SOOPERS FUEL"
        assert result.category == "Transportation"

        # Without "FUEL" in merchant, should match "KING SOOPERS"
        result2 = match_rules("KING SOOPERS #0099", rules)
        assert result2 is not None
        assert result2.pattern == "KING SOOPERS"
        assert result2.category == "Food & Dining"


# ===================================================================
# categorize tests
# ===================================================================


class TestCategorize:
    """Tests for the categorize function."""

    def test_rule_matching_applies_categories(self):
        """Transactions matching rules should get categorized."""
        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
            MerchantRule(
                pattern="SHELL OIL",
                category="Transportation",
                subcategory="Gas/Fuel",
                source="user",
            ),
        ]
        txns = [
            _make_txn("CHIPOTLE MEXICAN GRIL"),
            _make_txn("SHELL OIL 574726183"),
        ]

        result = categorize(txns, rules, SAMPLE_CATEGORIES)

        assert isinstance(result, StageResult)
        assert result.transactions[0].category == "Food & Dining"
        assert result.transactions[0].subcategory == "Fast Food"
        assert result.transactions[1].category == "Transportation"
        assert result.transactions[1].subcategory == "Gas/Fuel"

    def test_already_categorized_transactions_skipped(self):
        """Transactions with a category other than Uncategorized are not re-categorized."""
        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
        ]
        txn = _make_txn("CHIPOTLE MEXICAN GRIL", category="Shopping", subcategory="Clothing")
        result = categorize([txn], rules, SAMPLE_CATEGORIES)

        assert result.transactions[0].category == "Shopping"
        assert result.transactions[0].subcategory == "Clothing"

    def test_llm_fallback_for_uncategorized(self):
        """Uncategorized transactions (no rule match) should be sent to the LLM."""
        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
        ]
        mock_llm = MockLLMAdapter(
            suggestions={
                "NEW MERCHANT XYZ": ("Shopping", "Electronics"),
            }
        )
        txns = [
            _make_txn("CHIPOTLE MEXICAN GRIL"),
            _make_txn("NEW MERCHANT XYZ"),
        ]

        result = categorize(txns, rules, SAMPLE_CATEGORIES, llm_adapter=mock_llm)

        # Chipotle matched by rule
        assert result.transactions[0].category == "Food & Dining"
        # New merchant categorized by LLM
        assert result.transactions[1].category == "Shopping"
        assert result.transactions[1].subcategory == "Electronics"
        # LLM was called once
        assert len(mock_llm.calls) == 1
        # Only the uncategorized transaction was sent to LLM
        assert len(mock_llm.calls[0][0]) == 1
        assert mock_llm.calls[0][0][0]["merchant"] == "NEW MERCHANT XYZ"

    def test_llm_receives_categories(self):
        """The LLM adapter should receive the category taxonomy."""
        rules = []
        mock_llm = MockLLMAdapter()
        txns = [_make_txn("SOME STORE")]

        categorize(txns, rules, SAMPLE_CATEGORIES, llm_adapter=mock_llm)

        assert len(mock_llm.calls) == 1
        assert mock_llm.calls[0][1] == SAMPLE_CATEGORIES

    def test_llm_not_called_when_all_categorized(self):
        """If all transactions match rules, the LLM should not be called."""
        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
        ]
        mock_llm = MockLLMAdapter()
        txns = [_make_txn("CHIPOTLE MEXICAN GRIL")]

        result = categorize(txns, rules, SAMPLE_CATEGORIES, llm_adapter=mock_llm)

        assert result.transactions[0].category == "Food & Dining"
        assert len(mock_llm.calls) == 0

    def test_llm_failure_leaves_uncategorized(self):
        """If the LLM raises an exception, transactions stay Uncategorized."""
        rules = []
        mock_llm = MockLLMAdapter(should_fail=True)
        txns = [_make_txn("UNKNOWN MERCHANT")]

        result = categorize(txns, rules, SAMPLE_CATEGORIES, llm_adapter=mock_llm)

        assert result.transactions[0].category == "Uncategorized"
        assert any("LLM categorization failed" in w for w in result.warnings)

    def test_no_llm_adapter_warning(self):
        """When llm_adapter is None and transactions are uncategorized, warn."""
        rules = []
        txns = [_make_txn("UNKNOWN MERCHANT")]

        result = categorize(txns, rules, SAMPLE_CATEGORIES, llm_adapter=None)

        assert result.transactions[0].category == "Uncategorized"
        assert any("LLM unavailable" in w for w in result.warnings)

    def test_llm_partial_response(self):
        """If the LLM only categorizes some transactions, others stay Uncategorized."""
        rules = []
        mock_llm = MockLLMAdapter(
            suggestions={
                "MERCHANT A": ("Shopping", ""),
                # MERCHANT B not in suggestions
            }
        )
        txns = [
            _make_txn("MERCHANT A"),
            _make_txn("MERCHANT B"),
        ]

        result = categorize(txns, rules, SAMPLE_CATEGORIES, llm_adapter=mock_llm)

        assert result.transactions[0].category == "Shopping"
        assert result.transactions[1].category == "Uncategorized"
        assert any("could not be parsed" in w for w in result.warnings)

    def test_empty_transaction_list(self):
        """An empty transaction list should return an empty StageResult."""
        result = categorize([], [], SAMPLE_CATEGORIES)
        assert result.transactions == []
        assert result.warnings == []
        assert result.errors == []

    def test_categorize_with_sample_fixtures(self, sample_transactions, sample_rules):
        """Integration test using the shared fixtures from conftest."""
        # Only categorize January transactions that are uncategorized
        uncategorized_txns = [
            _make_txn(
                "TARGET 00022186",
                transaction_id="test_target",
            ),
            _make_txn(
                "VMWARE INC",
                transaction_id="test_vmware",
            ),
        ]

        mock_llm = MockLLMAdapter(
            suggestions={
                "VMWARE INC": ("Business", ""),
            }
        )

        result = categorize(
            uncategorized_txns,
            sample_rules,
            SAMPLE_CATEGORIES,
            llm_adapter=mock_llm,
        )

        # TARGET should match the learned rule
        assert result.transactions[0].category == "Shopping"
        # VMWARE should be categorized by LLM
        assert result.transactions[1].category == "Business"

    def test_llm_batch_contains_correct_fields(self):
        """The batch sent to the LLM should have merchant, description, amount, date."""
        rules = []
        mock_llm = MockLLMAdapter()
        txn = _make_txn(
            "TEST MERCHANT",
            description="TEST MERCHANT DESCRIPTION",
            amount=Decimal("-42.50"),
            txn_date=date(2026, 1, 20),
        )

        categorize([txn], rules, SAMPLE_CATEGORIES, llm_adapter=mock_llm)

        assert len(mock_llm.calls) == 1
        batch = mock_llm.calls[0][0]
        assert len(batch) == 1
        assert batch[0]["merchant"] == "TEST MERCHANT"
        assert batch[0]["description"] == "TEST MERCHANT DESCRIPTION"
        assert batch[0]["amount"] == "-42.50"
        assert batch[0]["date"] == "2026-01-20"


# ===================================================================
# learn tests
# ===================================================================


class TestLearn:
    """Tests for the learn function."""

    def test_new_rule_added(self, tmp_path):
        """A correction for a new merchant should add a new learned rule."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        rows = [
            {
                "transaction_id": "abc123",
                "date": "2026-01-15",
                "merchant": "NEW STORE",
                "description": "NEW STORE #1234",
                "amount": "-25.00",
                "institution": "chase",
                "account": "Chase Credit Card",
                "category": "Uncategorized",
                "subcategory": "",
                "is_return": "False",
                "split_from": "",
            }
        ]
        corrected_rows = [
            {
                **rows[0],
                "category": "Shopping",
                "subcategory": "Electronics",
            }
        ]

        _write_csv(original, rows)
        _write_csv(corrected, corrected_rows)

        rules: list[MerchantRule] = []
        result = learn(original, corrected, rules)

        assert isinstance(result, LearnResult)
        assert result.added == 1
        assert result.updated == 0
        assert result.skipped == 0
        assert len(result.rules) == 1
        assert result.rules[0].pattern == "NEW STORE"
        assert result.rules[0].category == "Shopping"
        assert result.rules[0].subcategory == "Electronics"
        assert result.rules[0].source == "learned"

    def test_existing_learned_rule_updated(self, tmp_path):
        """A correction for a merchant with an existing learned rule should update it."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        rows = [
            {
                "transaction_id": "abc123",
                "date": "2026-01-15",
                "merchant": "TARGET",
                "description": "TARGET #1234",
                "amount": "-55.00",
                "institution": "chase",
                "account": "Chase Credit Card",
                "category": "Shopping",
                "subcategory": "",
                "is_return": "False",
                "split_from": "",
            }
        ]
        corrected_rows = [
            {
                **rows[0],
                "category": "Shopping",
                "subcategory": "Home Goods",
            }
        ]

        _write_csv(original, rows)
        _write_csv(corrected, corrected_rows)

        rules = [
            MerchantRule(
                pattern="TARGET",
                category="Shopping",
                subcategory="",
                source="learned",
            ),
        ]
        result = learn(original, corrected, rules)

        assert result.added == 0
        assert result.updated == 1
        assert result.skipped == 0
        # The existing rule should be updated in place
        assert result.rules[0].category == "Shopping"
        assert result.rules[0].subcategory == "Home Goods"

    def test_user_rule_conflict_skipped(self, tmp_path):
        """A correction for a merchant covered by a user rule should be skipped."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        rows = [
            {
                "transaction_id": "abc123",
                "date": "2026-01-15",
                "merchant": "CHIPOTLE MEXICAN GRIL",
                "description": "CHIPOTLE MEXICAN GRIL BOULDER CO",
                "amount": "-12.50",
                "institution": "chase",
                "account": "Chase Credit Card",
                "category": "Food & Dining",
                "subcategory": "Fast Food",
                "is_return": "False",
                "split_from": "",
            }
        ]
        corrected_rows = [
            {
                **rows[0],
                "category": "Food & Dining",
                "subcategory": "Restaurant",
            }
        ]

        _write_csv(original, rows)
        _write_csv(corrected, corrected_rows)

        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
        ]
        result = learn(original, corrected, rules)

        assert result.added == 0
        assert result.updated == 0
        assert result.skipped == 1
        # The user rule should not be modified
        assert result.rules[0].category == "Food & Dining"
        assert result.rules[0].subcategory == "Fast Food"

    def test_unchanged_transactions_ignored(self, tmp_path):
        """Transactions with no category change should not create rules."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        rows = [
            {
                "transaction_id": "abc123",
                "date": "2026-01-15",
                "merchant": "CHIPOTLE",
                "description": "CHIPOTLE MEXICAN GRIL",
                "amount": "-12.50",
                "institution": "chase",
                "account": "Chase Credit Card",
                "category": "Food & Dining",
                "subcategory": "Fast Food",
                "is_return": "False",
                "split_from": "",
            }
        ]

        _write_csv(original, rows)
        _write_csv(corrected, rows)  # Same data

        rules: list[MerchantRule] = []
        result = learn(original, corrected, rules)

        assert result.added == 0
        assert result.updated == 0
        assert result.skipped == 0
        assert len(result.rules) == 0

    def test_multiple_corrections(self, tmp_path):
        """Multiple corrections in one learn call should all be processed."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        base = {
            "date": "2026-01-15",
            "description": "",
            "institution": "chase",
            "account": "Chase Credit Card",
            "is_return": "False",
            "split_from": "",
        }

        original_rows = [
            {
                **base,
                "transaction_id": "tx1",
                "merchant": "STORE A",
                "amount": "-10.00",
                "category": "Uncategorized",
                "subcategory": "",
            },
            {
                **base,
                "transaction_id": "tx2",
                "merchant": "STORE B",
                "amount": "-20.00",
                "category": "Uncategorized",
                "subcategory": "",
            },
            {
                **base,
                "transaction_id": "tx3",
                "merchant": "CHIPOTLE GRILL",
                "amount": "-15.00",
                "category": "Food & Dining",
                "subcategory": "Fast Food",
            },
        ]

        corrected_rows = [
            {
                **original_rows[0],
                "category": "Shopping",
                "subcategory": "Electronics",
            },
            {
                **original_rows[1],
                "category": "Entertainment",
                "subcategory": "Subscriptions",
            },
            {
                **original_rows[2],
                "category": "Food & Dining",
                "subcategory": "Restaurant",
            },
        ]

        _write_csv(original, original_rows)
        _write_csv(corrected, corrected_rows)

        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
        ]
        result = learn(original, corrected, rules)

        assert result.added == 2  # STORE A and STORE B
        assert result.updated == 0
        assert result.skipped == 1  # CHIPOTLE GRILL (covered by user rule)

    def test_transaction_only_in_corrected_ignored(self, tmp_path):
        """Transactions present only in the corrected file should be skipped."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        original_rows = [
            {
                "transaction_id": "tx1",
                "date": "2026-01-15",
                "merchant": "STORE A",
                "description": "",
                "amount": "-10.00",
                "institution": "chase",
                "account": "Chase Credit Card",
                "category": "Uncategorized",
                "subcategory": "",
                "is_return": "False",
                "split_from": "",
            }
        ]
        corrected_rows = [
            {
                **original_rows[0],
                "category": "Shopping",
                "subcategory": "",
            },
            {
                "transaction_id": "tx_new",
                "date": "2026-01-16",
                "merchant": "STORE B",
                "description": "",
                "amount": "-20.00",
                "institution": "chase",
                "account": "Chase Credit Card",
                "category": "Entertainment",
                "subcategory": "",
                "is_return": "False",
                "split_from": "",
            },
        ]

        _write_csv(original, original_rows)
        _write_csv(corrected, corrected_rows)

        rules: list[MerchantRule] = []
        result = learn(original, corrected, rules)

        assert result.added == 1  # Only STORE A
        assert result.rules[0].pattern == "STORE A"

    def test_subcategory_only_change(self, tmp_path):
        """A subcategory-only change should still create/update a rule."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        row = {
            "transaction_id": "tx1",
            "date": "2026-01-15",
            "merchant": "REI.COM",
            "description": "REI.COM 800-426-4840",
            "amount": "-160.65",
            "institution": "capital_one",
            "account": "Capital One Credit Card",
            "category": "Shopping",
            "subcategory": "",
            "is_return": "False",
            "split_from": "",
        }
        corrected_row = {
            **row,
            "subcategory": "Equipment & Maintenance",
        }

        _write_csv(original, [row])
        _write_csv(corrected, [corrected_row])

        rules: list[MerchantRule] = []
        result = learn(original, corrected, rules)

        assert result.added == 1
        assert result.rules[0].pattern == "REI.COM"
        assert result.rules[0].category == "Shopping"
        assert result.rules[0].subcategory == "Equipment & Maintenance"

    def test_user_rule_substring_match_prevents_learned_rule(self, tmp_path):
        """If a user rule is a substring match for the merchant, skip the correction.

        Even if the user rule pattern is shorter than the merchant name,
        the merchant is still "covered" by the user rule.
        """
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        row = {
            "transaction_id": "tx1",
            "date": "2026-01-15",
            "merchant": "STARBUCKS STORE #1234",
            "description": "STARBUCKS STORE #1234",
            "amount": "-5.75",
            "institution": "chase",
            "account": "Chase Credit Card",
            "category": "Food & Dining",
            "subcategory": "Coffee",
            "is_return": "False",
            "split_from": "",
        }
        corrected_row = {
            **row,
            "category": "Food & Dining",
            "subcategory": "Fast Food",
        }

        _write_csv(original, [row])
        _write_csv(corrected, [corrected_row])

        rules = [
            MerchantRule(
                pattern="STARBUCKS",
                category="Food & Dining",
                subcategory="Coffee",
                source="user",
            ),
        ]
        result = learn(original, corrected, rules)

        assert result.skipped == 1
        assert result.added == 0
        assert result.updated == 0

    def test_learn_preserves_existing_rules(self, tmp_path):
        """The learn function should preserve all existing rules in the output."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        row = {
            "transaction_id": "tx1",
            "date": "2026-01-15",
            "merchant": "NEW PLACE",
            "description": "NEW PLACE",
            "amount": "-30.00",
            "institution": "chase",
            "account": "Chase Credit Card",
            "category": "Uncategorized",
            "subcategory": "",
            "is_return": "False",
            "split_from": "",
        }
        corrected_row = {
            **row,
            "category": "Entertainment",
            "subcategory": "Tickets/Events",
        }

        _write_csv(original, [row])
        _write_csv(corrected, [corrected_row])

        existing_rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
            MerchantRule(
                pattern="TARGET",
                category="Shopping",
                subcategory="",
                source="learned",
            ),
        ]
        result = learn(original, corrected, existing_rules)

        assert result.added == 1
        # All original rules plus the new one
        assert len(result.rules) == 3
        assert result.rules[0].pattern == "CHIPOTLE"
        assert result.rules[1].pattern == "TARGET"
        assert result.rules[2].pattern == "NEW PLACE"

    def test_category_change_creates_rule(self, tmp_path):
        """A category change (not just subcategory) should also create a rule."""
        original = tmp_path / "original.csv"
        corrected = tmp_path / "corrected.csv"

        row = {
            "transaction_id": "tx1",
            "date": "2026-01-15",
            "merchant": "COSTCO WHOLESALE",
            "description": "COSTCO WHOLESALE #1234",
            "amount": "-250.00",
            "institution": "chase",
            "account": "Chase Credit Card",
            "category": "Shopping",
            "subcategory": "",
            "is_return": "False",
            "split_from": "",
        }
        corrected_row = {
            **row,
            "category": "Food & Dining",
            "subcategory": "Groceries",
        }

        _write_csv(original, [row])
        _write_csv(corrected, [corrected_row])

        rules: list[MerchantRule] = []
        result = learn(original, corrected, rules)

        assert result.added == 1
        assert result.rules[0].category == "Food & Dining"
        assert result.rules[0].subcategory == "Groceries"
