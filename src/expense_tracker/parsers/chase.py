"""Chase credit card CSV parser.

Chase CSV format:
    Transaction Date, Post Date, Description, Category, Type, Amount, Memo

Sign convention:
    Negative amounts are charges (expenses).
    Positive amounts are refunds/credits.

This parser uses Transaction Date (not Post Date) per the architecture spec.
"""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from expense_tracker.models import StageResult, Transaction, generate_transaction_id

EXPECTED_COLUMNS = {"Transaction Date", "Post Date", "Description", "Category", "Type", "Amount"}


def parse(file_path: Path, institution: str, account: str) -> StageResult:
    """Parse a Chase credit card CSV file into normalized Transactions.

    Args:
        file_path: Path to the CSV file.
        institution: Institution key, e.g. "chase".
        account: Account identifier from config.

    Returns:
        A StageResult containing parsed transactions, warnings for skipped
        rows, and errors if the file cannot be parsed at all.
    """
    transactions: list[Transaction] = []
    warnings: list[str] = []
    errors: list[str] = []
    source = str(file_path)

    try:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate expected columns exist
            if reader.fieldnames is None:
                errors.append(f"{source}: empty file or no header row")
                return StageResult(transactions=[], warnings=warnings, errors=errors)

            actual_columns = set(reader.fieldnames)
            missing = EXPECTED_COLUMNS - actual_columns
            if missing:
                errors.append(f"{source}: missing expected columns: {', '.join(sorted(missing))}")
                return StageResult(transactions=[], warnings=warnings, errors=errors)

            rows = list(reader)

    except FileNotFoundError:
        errors.append(f"{source}: file not found")
        return StageResult(transactions=[], warnings=warnings, errors=errors)
    except OSError as exc:
        errors.append(f"{source}: {exc}")
        return StageResult(transactions=[], warnings=warnings, errors=errors)

    if not rows:
        return StageResult(transactions=[], warnings=warnings, errors=errors)

    total_rows = len(rows)
    malformed_count = 0

    for row_ordinal, row in enumerate(rows):
        # Parse date
        date_str = (row.get("Transaction Date") or "").strip()
        if not date_str:
            malformed_count += 1
            warnings.append(f"{source}: skipped malformed row {row_ordinal} (missing date)")
            continue

        try:
            txn_date = datetime.strptime(date_str, "%m/%d/%Y").date()
        except ValueError:
            malformed_count += 1
            warnings.append(
                f"{source}: skipped malformed row {row_ordinal} (invalid date: {date_str!r})"
            )
            continue

        # Parse amount
        amount_str = (row.get("Amount") or "").strip()
        if not amount_str:
            malformed_count += 1
            warnings.append(f"{source}: skipped malformed row {row_ordinal} (missing amount)")
            continue

        try:
            amount = Decimal(amount_str)
        except InvalidOperation:
            malformed_count += 1
            warnings.append(
                f"{source}: skipped malformed row {row_ordinal} (invalid amount: {amount_str!r})"
            )
            continue

        # Extract merchant/description
        description = (row.get("Description") or "").strip()
        if not description:
            malformed_count += 1
            warnings.append(f"{source}: skipped malformed row {row_ordinal} (missing description)")
            continue

        merchant = description
        is_return = amount > 0

        txn_id = generate_transaction_id(
            institution=institution,
            txn_date=txn_date,
            merchant=merchant,
            amount=amount,
            row_ordinal=row_ordinal,
        )

        transactions.append(
            Transaction(
                transaction_id=txn_id,
                date=txn_date,
                merchant=merchant,
                description=description,
                amount=amount,
                institution=institution,
                account=account,
                is_return=is_return,
                source_file=source,
            )
        )

    # Fail entire file if >10% malformed
    if total_rows > 0 and malformed_count / total_rows > 0.10:
        errors.append(
            f"{source}: too many malformed rows ({malformed_count}/{total_rows}), "
            f"skipping entire file"
        )
        return StageResult(transactions=[], warnings=warnings, errors=errors)

    return StageResult(transactions=transactions, warnings=warnings, errors=errors)
