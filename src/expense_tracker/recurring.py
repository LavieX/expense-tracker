"""Recurring transaction detection.

This module scans historical transaction data to identify merchants that appear
regularly with similar amounts, likely indicating subscriptions or recurring
bills.

Detection algorithm:
- A merchant is considered recurring if it appears in 3+ distinct months
- Amounts must be within 20% variance to be considered "similar"
- Results are used to auto-flag transactions as recurring when no explicit
  rule exists
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from expense_tracker.models import Transaction


def detect_recurring(transactions: list[Transaction], output_dir: Path) -> list[str]:
    """Detect merchants that appear to be recurring charges.

    Scans existing output CSVs in the output directory to build a historical
    view of all transactions, then identifies merchants that:
    1. Appear in 3 or more distinct months
    2. Have similar amounts (within 20% variance)

    Args:
        transactions: Current batch of transactions (not used for detection,
            but included for potential future enhancements).
        output_dir: Directory containing historical monthly CSV files.

    Returns:
        A list of merchant patterns that appear to be recurring.
    """
    # Collect all historical transactions from output CSVs
    historical = _load_historical_transactions(output_dir)

    # Group by merchant (case-insensitive)
    merchant_data: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
    for txn in historical:
        merchant_upper = txn["merchant"].upper()
        month = txn["date"][:7]  # Extract YYYY-MM
        amount = abs(Decimal(txn["amount"]))
        merchant_data[merchant_upper].append((month, amount))

    # Analyze each merchant for recurring pattern
    recurring_merchants: list[str] = []

    for merchant, occurrences in merchant_data.items():
        if len(occurrences) < 3:
            continue

        # Group by month (multiple transactions in same month count as one)
        months_seen: dict[str, list[Decimal]] = defaultdict(list)
        for month, amount in occurrences:
            months_seen[month].append(amount)

        if len(months_seen) < 3:
            continue

        # Check if amounts are similar (within 20% variance)
        # Use the median amount from each month for comparison
        monthly_amounts = [
            sum(amounts) / len(amounts) for amounts in months_seen.values()
        ]

        if _amounts_are_similar(monthly_amounts):
            recurring_merchants.append(merchant)

    return recurring_merchants


def _load_historical_transactions(output_dir: Path) -> list[dict[str, str]]:
    """Load all transactions from CSV files in the output directory.

    Args:
        output_dir: Directory containing monthly CSV files (YYYY-MM.csv).

    Returns:
        List of transaction dicts with keys matching CSV columns.
    """
    historical: list[dict[str, str]] = []

    if not output_dir.is_dir():
        return historical

    # Find all CSV files matching YYYY-MM.csv pattern
    for csv_path in sorted(output_dir.glob("*.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    historical.append(dict(row))
        except (OSError, csv.Error):
            # Skip files we can't read
            continue

    return historical


def _amounts_are_similar(amounts: list[Decimal], variance_threshold: float = 0.20) -> bool:
    """Check if a list of amounts are similar (within variance threshold).

    Args:
        amounts: List of transaction amounts to compare.
        variance_threshold: Maximum allowed variance as a fraction (0.20 = 20%).

    Returns:
        True if all amounts are within variance_threshold of the median.
    """
    if not amounts:
        return False

    if len(amounts) == 1:
        return True

    # Calculate median
    sorted_amounts = sorted(amounts)
    n = len(sorted_amounts)
    if n % 2 == 0:
        median = (sorted_amounts[n // 2 - 1] + sorted_amounts[n // 2]) / 2
    else:
        median = sorted_amounts[n // 2]

    if median == 0:
        return False

    # Check if all amounts are within threshold
    for amount in amounts:
        variance = abs(amount - median) / median
        if variance > Decimal(str(variance_threshold)):
            return False

    return True
