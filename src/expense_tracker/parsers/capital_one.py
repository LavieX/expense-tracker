"""Capital One credit card CSV parser.

Capital One CSV format:
    Transaction Date, Posted Date, Card No., Description, Category, Debit, Credit

Sign convention:
    Debit column contains charge amounts (expenses) -- these become negative.
    Credit column contains payment/refund amounts -- these become positive.
    Exactly one of Debit or Credit is populated per row.

This parser uses Transaction Date per the architecture spec.
"""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from expense_tracker.models import StageResult, Transaction, generate_transaction_id

EXPECTED_COLUMNS = {"Transaction Date", "Posted Date", "Card No.", "Description", "Debit", "Credit"}


def parse(file_path: Path, institution: str, account: str) -> StageResult:
    """Parse a Capital One credit card CSV file into normalized Transactions.

    Args:
        file_path: Path to the CSV file.
        institution: Institution key, e.g. "capital_one".
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
            txn_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            malformed_count += 1
            warnings.append(
                f"{source}: skipped malformed row {row_ordinal} (invalid date: {date_str!r})"
            )
            continue

        # Parse amount from Debit/Credit columns
        debit_str = (row.get("Debit") or "").strip()
        credit_str = (row.get("Credit") or "").strip()

        if not debit_str and not credit_str:
            malformed_count += 1
            warnings.append(
                f"{source}: skipped malformed row {row_ordinal} (no debit or credit amount)"
            )
            continue

        try:
            if debit_str:
                # Debit = charge/expense -> negative amount
                amount = -Decimal(debit_str)
                is_return = False
            else:
                # Credit = payment/refund -> positive amount
                amount = Decimal(credit_str)
                is_return = True
        except InvalidOperation:
            malformed_count += 1
            raw = debit_str or credit_str
            warnings.append(
                f"{source}: skipped malformed row {row_ordinal} (invalid amount: {raw!r})"
            )
            continue

        # Extract merchant/description
        description = (row.get("Description") or "").strip()
        if not description:
            malformed_count += 1
            warnings.append(f"{source}: skipped malformed row {row_ordinal} (missing description)")
            continue

        merchant = description

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
