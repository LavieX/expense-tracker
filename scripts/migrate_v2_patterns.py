#!/usr/bin/env python3
"""Migrate v2 merchant patterns from ai_context.json to v3 rules.toml.

Reads the ``merchant_patterns`` section from v2's ``ai_context.json`` and
converts each entry into a learned rule in v3's ``rules.toml`` format.

Usage::

    python scripts/migrate_v2_patterns.py \
        --source ../expenses-v2/ai_context.json \
        --target rules.toml

Patterns whose category is not present in the v3 category taxonomy
(loaded from ``categories.toml`` next to the target ``rules.toml``) are
skipped and reported.  The ``[user_rules]`` section of the target file
is preserved verbatim; only ``[learned_rules]`` is rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running as a standalone script by adding the project src to sys.path.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from expense_tracker.config import load_categories, load_rules, save_learned_rules
from expense_tracker.models import MerchantRule


def _load_v2_patterns(source: Path) -> dict[str, dict]:
    """Load and return the ``merchant_patterns`` dict from *source*.

    Each value is a dict with at least ``category`` and ``subcategory``
    keys (and optionally ``notes``).

    Raises:
        FileNotFoundError: If *source* does not exist.
        KeyError: If the JSON has no ``merchant_patterns`` key.
    """
    with open(source, encoding="utf-8") as f:
        data = json.load(f)

    if "merchant_patterns" not in data:
        raise KeyError(
            f"No 'merchant_patterns' section found in {source}"
        )

    return data["merchant_patterns"]


def _build_valid_categories(root: Path) -> dict[str, list[str]]:
    """Return a mapping of valid category names to their subcategory lists.

    Loads ``categories.toml`` from *root* (the directory containing the
    target ``rules.toml``).
    """
    cats = load_categories(root)
    return {c["name"]: c["subcategories"] for c in cats}


def migrate(
    source: Path,
    target: Path,
    *,
    verbose: bool = False,
) -> tuple[int, int, int, list[str]]:
    """Migrate v2 merchant patterns into the target rules.toml.

    Args:
        source: Path to v2's ``ai_context.json``.
        target: Path to v3's ``rules.toml``.
        verbose: If True, print details for each pattern processed.

    Returns:
        A tuple of (total_found, migrated, skipped, skip_reasons).
    """
    v2_patterns = _load_v2_patterns(source)
    total_found = len(v2_patterns)

    # Determine the project root that contains the target rules.toml.
    target_root = target.parent

    # Build set of valid v3 categories.
    valid_categories = _build_valid_categories(target_root)

    # Load existing rules so we preserve user rules and any existing learned
    # rules that are NOT being replaced by v2 data.
    existing_rules = load_rules(target_root)
    user_rules = [r for r in existing_rules if r.source == "user"]
    existing_learned = {r.pattern: r for r in existing_rules if r.source == "learned"}

    # Build the merged learned-rules dict.  Start with existing learned rules
    # so we preserve any that are not overwritten by v2 data.
    merged_learned: dict[str, MerchantRule] = dict(existing_learned)

    migrated = 0
    skipped = 0
    skip_reasons: list[str] = []

    for pattern, entry in v2_patterns.items():
        category = entry.get("category", "").strip()
        subcategory = entry.get("subcategory", "").strip()

        # Validate category against v3 taxonomy.
        if not category or category not in valid_categories:
            skipped += 1
            reason = f"  SKIP  {pattern!r}: category {category!r} not in v3 taxonomy"
            skip_reasons.append(reason)
            if verbose:
                print(reason)
            continue

        # Validate subcategory if provided.
        if subcategory and subcategory not in valid_categories[category]:
            skipped += 1
            reason = (
                f"  SKIP  {pattern!r}: subcategory {subcategory!r} "
                f"not valid under {category!r}"
            )
            skip_reasons.append(reason)
            if verbose:
                print(reason)
            continue

        rule = MerchantRule(
            pattern=pattern,
            category=category,
            subcategory=subcategory,
            source="learned",
        )

        merged_learned[pattern] = rule
        migrated += 1
        if verbose:
            value = f"{category}:{subcategory}" if subcategory else category
            print(f"  OK    {pattern!r} -> {value!r}")

    # Build the complete learned-rules list (preserving insertion order of
    # the merged dict).
    all_learned = list(merged_learned.values())

    # Combine user + learned for save_learned_rules (it filters by source).
    all_rules = user_rules + all_learned
    save_learned_rules(target_root, all_rules)

    return total_found, migrated, skipped, skip_reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate v2 merchant patterns to v3 rules.toml format.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to v2's ai_context.json.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to v3's rules.toml.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for each pattern processed.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()

    if not source.exists():
        print(f"Error: source file not found: {source}", file=sys.stderr)
        sys.exit(1)
    if not target.exists():
        print(f"Error: target file not found: {target}", file=sys.stderr)
        sys.exit(1)

    print(f"Source: {source}")
    print(f"Target: {target}")
    print()

    total, migrated, skipped, skip_reasons = migrate(
        source, target, verbose=args.verbose
    )

    print()
    print("== Migration Summary ==")
    print(f"  Total patterns found:  {total}")
    print(f"  Patterns migrated:     {migrated}")
    print(f"  Patterns skipped:      {skipped}")

    if skip_reasons and not args.verbose:
        print()
        print("Skipped patterns:")
        for reason in skip_reasons:
            print(reason)

    # Verify the output is loadable.
    try:
        rules = load_rules(target.parent)
        learned_count = sum(1 for r in rules if r.source == "learned")
        print()
        print(f"Verification: rules.toml loads successfully ({learned_count} learned rules)")
    except Exception as exc:
        print(f"\nVerification FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
