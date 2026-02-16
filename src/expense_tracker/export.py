"""CSV export writer and processing summary printer.

This module implements Stage 6 of the pipeline:

- :func:`export` filters out transfers, sorts by date/institution/amount,
  and writes the fixed column schema to a monthly CSV file.
- :func:`print_summary` prints a human-readable processing summary to stdout,
  including source counts, categorization breakdown, top uncategorized
  merchants, and spending by category.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from expense_tracker.models import PipelineResult, Transaction

# Fixed output column order (Section 4 of the architecture doc).
# ``is_transfer`` and ``source_file`` are intentionally excluded.
CSV_COLUMNS = [
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
]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export(
    transactions: list[Transaction],
    output_dir: str | Path,
    month: str,
) -> Path:
    """Write a monthly CSV of non-transfer transactions.

    1. Filters out transactions where ``is_transfer`` is True.
    2. Sorts by date, then institution, then amount.
    3. Writes CSV with the fixed column schema to
       ``output_dir/YYYY-MM.csv``.  Overwrites if the file already exists.

    Args:
        transactions: Fully-processed transaction list from the pipeline.
        output_dir: Directory to write the CSV file into.
        month: Target month as ``"YYYY-MM"`` string, used as the filename.

    Returns:
        The :class:`~pathlib.Path` to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{month}.csv"

    # Filter out transfers
    exportable = [txn for txn in transactions if not txn.is_transfer]

    # Sort: date ascending, then institution ascending, then amount ascending
    exportable.sort(key=lambda t: (t.date, t.institution, t.amount))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for txn in exportable:
            writer.writerow(
                {
                    "transaction_id": txn.transaction_id,
                    "date": txn.date.isoformat(),
                    "month": txn.date.strftime("%Y-%m"),
                    "merchant": txn.merchant,
                    "description": txn.description,
                    "amount": str(txn.amount),
                    "institution": txn.institution,
                    "account": txn.account,
                    "category": txn.category,
                    "subcategory": txn.subcategory,
                    "is_return": str(txn.is_return),
                    "is_recurring": str(txn.is_recurring),
                    "split_from": txn.split_from,
                }
            )

    return output_path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(pipeline_result: PipelineResult, month: str) -> None:
    """Print a human-readable processing summary to stdout.

    The summary includes:

    - Source counts per institution.
    - Total transactions and transfers excluded.
    - Enrichment statistics (how many were split from a parent).
    - Categorization breakdown: rule match, LLM, uncategorized.
    - Top uncategorized merchants with transaction count and total amount.
    - Spending by category (sorted by absolute amount descending).
    - Warnings and errors, if any.

    Args:
        pipeline_result: The :class:`~expense_tracker.models.PipelineResult`
            from a completed pipeline run.
        month: Target month as ``"YYYY-MM"`` string, used in the header.
    """
    txns = pipeline_result.transactions

    # --- Partition: transfers vs. non-transfers ---
    transfers = [t for t in txns if t.is_transfer]
    non_transfers = [t for t in txns if not t.is_transfer]
    transfer_count = len(transfers)
    active_count = len(non_transfers)

    # --- Source counts ---
    institution_counts: Counter[str] = Counter()
    for t in txns:
        institution_counts[t.institution] += 1

    # --- Enrichment ---
    enriched_count = sum(1 for t in non_transfers if t.split_from)

    # --- Categorization breakdown ---
    # We cannot distinguish rule-match from LLM in the Transaction model
    # itself (the pipeline does not tag the method).  However, the
    # architecture summary shows them separately.  By convention:
    #   - "Uncategorized" -> uncategorized
    #   - All others are "categorized" (rule or LLM).  We approximate:
    #     any non-transfer non-Uncategorized transaction is counted.
    uncategorized = [t for t in non_transfers if t.category == "Uncategorized"]
    categorized = [t for t in non_transfers if t.category != "Uncategorized"]
    categorized_count = len(categorized)
    uncategorized_count = len(uncategorized)

    cat_pct = categorized_count / active_count * 100 if active_count > 0 else 0.0

    # --- Top uncategorized merchants ---
    uncat_merchant_amount: defaultdict[str, Decimal] = defaultdict(Decimal)
    uncat_merchant_count: Counter[str] = Counter()
    for t in uncategorized:
        uncat_merchant_count[t.merchant] += 1
        uncat_merchant_amount[t.merchant] += t.amount

    top_uncategorized = sorted(
        uncat_merchant_count.keys(),
        key=lambda m: (-uncat_merchant_count[m], uncat_merchant_amount[m]),
    )[:10]

    # --- Spending by category (only expenses, not returns) ---
    category_totals: defaultdict[str, Decimal] = defaultdict(Decimal)
    for t in non_transfers:
        if t.amount < 0:
            category_totals[t.category] += t.amount

    sorted_categories = sorted(
        category_totals.items(),
        key=lambda pair: pair[1],  # most negative first
    )

    # --- Print ---
    print()
    print(f"== Processing Summary: {month} ==")

    # Sources
    if institution_counts:
        source_parts = []
        for inst in sorted(institution_counts.keys()):
            count = institution_counts[inst]
            label = inst.replace("_", " ").title()
            source_parts.append(f"{label} ({count} txns)")
        print(f"Sources:  {', '.join(source_parts)}")
    else:
        print("Sources:  (none)")

    # Totals
    print(f"Total:    {len(txns)} transactions ({transfer_count} transfers excluded)")

    # Enrichment
    if enriched_count > 0:
        print(f"Enriched: {enriched_count} split line items")
    else:
        print("Enriched: 0 (no enrichment data)")

    # Categorization
    print(f"Categorized: {categorized_count} / {active_count} ({cat_pct:.1f}%)")
    print(f"  - Categorized: {categorized_count}")
    print(f"  - Uncategorized: {uncategorized_count}")

    # Top uncategorized
    if top_uncategorized:
        print()
        print("Top uncategorized merchants:")
        for i, merchant in enumerate(top_uncategorized, start=1):
            count = uncat_merchant_count[merchant]
            total = abs(uncat_merchant_amount[merchant])
            print(f"  {i:>2}. {merchant:<30} ({count} txns, ${total:,.2f})")

    # Spending by category
    if sorted_categories:
        print()
        print("Spending by category:")
        for cat, total in sorted_categories:
            print(f"  {cat + ':':<25} ${abs(total):,.2f}")

    # Warnings
    if pipeline_result.warnings:
        print()
        print(f"Warnings: {len(pipeline_result.warnings)}")
        for w in pipeline_result.warnings:
            print(f"  - {w}")

    # Errors
    if pipeline_result.errors:
        print()
        print(f"Errors: {len(pipeline_result.errors)}")
        for e in pipeline_result.errors:
            print(f"  - {e}")

    print()
