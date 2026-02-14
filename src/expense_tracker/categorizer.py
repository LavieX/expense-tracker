"""Categorization engine: rule matching, LLM fallback, and learn workflow.

This module implements the two-tier categorization system described in
Section 8 of the architecture doc:

1. **Tier 1 -- Rule matching:** Case-insensitive substring matching against
   merchant names. Longest pattern wins; ties broken by list order (user
   rules before learned rules, then insertion order within each group).

2. **Tier 2 -- LLM fallback:** Uncategorized transactions are batched and
   sent to an LLM adapter for suggestions. The adapter must conform to the
   ``LLMAdapter`` protocol defined here.

The ``learn`` function compares an original pipeline output CSV with a
user-corrected version and extracts new/updated learned rules, never
overwriting user rules.

Depends on ``models.py`` only (zero other internal imports at the module
level).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

from expense_tracker.models import (
    LearnResult,
    MerchantRule,
    StageResult,
    Transaction,
)


# ---------------------------------------------------------------------------
# LLM Adapter Protocol
# ---------------------------------------------------------------------------


class LLMAdapter(Protocol):
    """Protocol that LLM adapters must implement.

    The categorizer accepts any object conforming to this protocol.  The
    concrete ``AnthropicAdapter`` in ``llm.py`` is the primary
    implementation; tests use a mock.
    """

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        """Send a batch of transactions to the LLM for categorization.

        Args:
            transactions: List of dicts with keys ``merchant``,
                ``description``, ``amount``, ``date``.
            categories: The category taxonomy as a list of
                ``{"name": str, "subcategories": list[str]}`` dicts.

        Returns:
            A list of dicts, each with keys ``merchant``, ``category``,
            and ``subcategory``.  May return fewer items than the input
            if some could not be categorized.  Returns an empty list if
            the LLM is unavailable or the response cannot be parsed.
        """
        ...


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------


def match_rules(merchant: str, rules: list[MerchantRule]) -> MerchantRule | None:
    """Find the best matching rule for a merchant string.

    Strategy: case-insensitive substring match, longest pattern wins.
    Rules are expected to be pre-sorted with user rules first, then
    learned rules (as produced by ``config.load_rules()``).  Among all
    matching rules, the one with the longest pattern wins.  Ties are
    broken by list order: the first match at the longest length wins,
    which naturally favors user rules over learned rules.

    Args:
        merchant: The merchant name to match against.
        rules: Sorted list of ``MerchantRule`` objects (user first,
            then learned).

    Returns:
        The best-matching ``MerchantRule``, or ``None`` if no rule
        matches.
    """
    merchant_upper = merchant.upper()
    best: MerchantRule | None = None
    for rule in rules:
        if rule.pattern.upper() in merchant_upper:
            if best is None or len(rule.pattern) > len(best.pattern):
                best = rule
    return best


# ---------------------------------------------------------------------------
# Categorize stage
# ---------------------------------------------------------------------------


def categorize(
    transactions: list[Transaction],
    rules: list[MerchantRule],
    categories: list[dict],
    llm_adapter: LLMAdapter | None = None,
) -> StageResult:
    """Categorize transactions using rule matching and optional LLM fallback.

    This is the Stage 5 pipeline function.  It processes every transaction
    in two passes:

    1. **Rule matching** -- For each uncategorized transaction, attempt to
       find a matching rule via ``match_rules()``.  If found, apply the
       rule's category and subcategory.
    2. **LLM fallback** -- Gather all still-uncategorized transactions
       into a single batch and send them to the LLM adapter (if provided
       and not ``None``).  Apply any suggestions returned.
    3. Any transactions that remain uncategorized keep
       ``category="Uncategorized"``.

    Args:
        transactions: List of transactions to categorize (may already have
            some categorized from earlier stages).
        rules: Sorted list of merchant rules (user first, then learned).
        categories: Category taxonomy for LLM context.
        llm_adapter: Optional LLM adapter implementing the
            ``LLMAdapter`` protocol.  Pass ``None`` to skip LLM
            categorization.

    Returns:
        A ``StageResult`` containing all transactions (with categories
        applied where possible) and any warnings.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Pass 1: Rule matching
    uncategorized: list[Transaction] = []
    for txn in transactions:
        if txn.category != "Uncategorized":
            # Already categorized (e.g. from enrichment stage).
            continue
        rule = match_rules(txn.merchant, rules)
        if rule is not None:
            txn.category = rule.category
            txn.subcategory = rule.subcategory
        else:
            uncategorized.append(txn)

    # Pass 2: LLM fallback for remaining uncategorized
    if uncategorized and llm_adapter is not None:
        batch = [
            {
                "merchant": txn.merchant,
                "description": txn.description,
                "amount": str(txn.amount),
                "date": txn.date.isoformat(),
            }
            for txn in uncategorized
        ]

        try:
            suggestions = llm_adapter.categorize_batch(batch, categories)
        except Exception as exc:
            warnings.append(f"LLM categorization failed: {exc}")
            suggestions = []

        if suggestions:
            # Build a lookup from merchant name to suggestion.
            suggestion_map: dict[str, dict] = {}
            for s in suggestions:
                merchant_key = s.get("merchant", "").upper()
                if merchant_key:
                    suggestion_map[merchant_key] = s

            applied = 0
            for txn in uncategorized:
                suggestion = suggestion_map.get(txn.merchant.upper())
                if suggestion:
                    cat = suggestion.get("category", "")
                    subcat = suggestion.get("subcategory", "")
                    if cat:
                        txn.category = cat
                        txn.subcategory = subcat or ""
                        applied += 1

            unapplied = len(uncategorized) - applied
            if unapplied > 0:
                warnings.append(
                    f"LLM: {unapplied} transaction(s) could not be parsed from response"
                )
    elif uncategorized and llm_adapter is None:
        warnings.append(
            f"LLM unavailable: {len(uncategorized)} transaction(s) left uncategorized"
        )

    return StageResult(transactions=transactions, warnings=warnings, errors=errors)


# ---------------------------------------------------------------------------
# Learn workflow
# ---------------------------------------------------------------------------


def learn(
    original_path: Path,
    corrected_path: Path,
    rules: list[MerchantRule],
) -> LearnResult:
    """Compare original and corrected CSVs, extract new/updated rules.

    Reads both CSV files, indexes transactions by ``transaction_id``, and
    for every transaction where the category or subcategory differs between
    the two files:

    - If a **user rule** already matches the merchant, skip (never
      overwrite user rules).
    - If a **learned rule** already exists for the exact merchant pattern,
      update it with the corrected category.
    - Otherwise, add a new learned rule.

    The ``merchant`` field value from the corrected CSV is used verbatim
    as the rule pattern -- no additional normalization is applied.  Parsers
    are responsible for producing stable, matchable merchant strings.

    Args:
        original_path: Path to the pipeline's original output CSV.
        corrected_path: Path to the user-corrected CSV.
        rules: The current rule list (user + learned), as returned by
            ``config.load_rules()``.

    Returns:
        A ``LearnResult`` with counts of added, updated, and skipped
        rules, plus the complete updated rule list.
    """
    original_txns = _read_csv_indexed(original_path)
    corrected_txns = _read_csv_indexed(corrected_path)

    added = 0
    updated = 0
    skipped = 0

    # Build mutable copies of rule lookups for efficient access.
    # User rules: keyed by pattern.upper() for matching checks.
    user_patterns: set[str] = set()
    for r in rules:
        if r.source == "user":
            user_patterns.add(r.pattern.upper())

    # Learned rules: keyed by pattern (exact, case-preserved) for
    # update-in-place.  We work on the actual list elements.
    learned_by_pattern: dict[str, MerchantRule] = {}
    for r in rules:
        if r.source == "learned":
            learned_by_pattern[r.pattern] = r

    for txn_id, corrected in corrected_txns.items():
        original = original_txns.get(txn_id)
        if original is None:
            # Transaction only in corrected file -- skip.
            continue

        orig_cat = original.get("category", "")
        orig_sub = original.get("subcategory", "")
        corr_cat = corrected.get("category", "")
        corr_sub = corrected.get("subcategory", "")

        if orig_cat == corr_cat and orig_sub == corr_sub:
            # No change.
            continue

        merchant = corrected.get("merchant", "")
        if not merchant:
            continue

        # Check if a user rule covers this merchant.
        if _has_user_rule_match(merchant, rules):
            skipped += 1
            continue

        # Check if a learned rule exists for this exact merchant pattern.
        if merchant in learned_by_pattern:
            existing = learned_by_pattern[merchant]
            existing.category = corr_cat
            existing.subcategory = corr_sub
            updated += 1
        else:
            new_rule = MerchantRule(
                pattern=merchant,
                category=corr_cat,
                subcategory=corr_sub,
                source="learned",
            )
            rules.append(new_rule)
            learned_by_pattern[merchant] = new_rule
            added += 1

    return LearnResult(
        added=added,
        updated=updated,
        skipped=skipped,
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_user_rule_match(merchant: str, rules: list[MerchantRule]) -> bool:
    """Check if any user rule matches (substring, case-insensitive) the merchant."""
    merchant_upper = merchant.upper()
    for rule in rules:
        if rule.source == "user" and rule.pattern.upper() in merchant_upper:
            return True
    return False


def _read_csv_indexed(path: Path) -> dict[str, dict[str, str]]:
    """Read a CSV file and return rows indexed by transaction_id.

    Args:
        path: Path to the CSV file.

    Returns:
        A dict mapping ``transaction_id`` to a dict of column values.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the CSV does not contain a ``transaction_id`` column.
    """
    result: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txn_id = row["transaction_id"]
            result[txn_id] = dict(row)
    return result
