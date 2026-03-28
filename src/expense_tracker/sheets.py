"""Google Sheets integration for pushing processed transaction data.

Supports month-level upsert: when pushing a specific month, only that
month's rows are replaced in the sheet.  Other months are untouched.

When pushing all months (--all), the sheet is cleared and rewritten.
"""

from __future__ import annotations

import logging
from pathlib import Path

from expense_tracker.models import SheetsConfig, Transaction

logger = logging.getLogger(__name__)

# Column order matches the CSV export.
COLUMNS = [
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
    "source",
]


def _txn_to_row(txn: Transaction) -> list:
    """Convert a Transaction to a list of cell values."""
    return [
        txn.transaction_id,
        txn.date.isoformat(),
        txn.date.strftime("%Y-%m"),
        txn.merchant,
        txn.description,
        float(txn.amount),
        txn.institution,
        txn.account,
        txn.category,
        txn.subcategory,
        "TRUE" if txn.is_return else "FALSE",
        "TRUE" if txn.is_recurring else "FALSE",
        txn.split_from,
        txn.source,
    ]


def _get_worksheet(client, config: SheetsConfig):
    """Open (or create) the target worksheet."""
    import gspread

    spreadsheet = client.open_by_key(config.spreadsheet_id)
    try:
        return spreadsheet.worksheet(config.worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(
            title=config.worksheet_name,
            rows=5000,
            cols=len(COLUMNS),
        )


def _authenticate(config: SheetsConfig, root: Path):
    """Authenticate and return a gspread client."""
    import gspread
    from google.oauth2.service_account import Credentials

    creds_path = Path(config.credentials_file).expanduser()
    if not creds_path.is_absolute():
        creds_path = root / creds_path

    if not creds_path.exists():
        raise FileNotFoundError(
            f"Google service account credentials not found: {creds_path}"
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
    return gspread.authorize(credentials)


def push_to_sheets(
    transactions: list[Transaction],
    config: SheetsConfig,
    root: Path,
    month: str | None = None,
) -> int:
    """Push transactions to Google Sheets with month-level upsert.

    When *month* is provided (e.g. ``"2026-02"``), only rows matching
    that month are deleted and replaced.  Other months in the sheet are
    untouched.

    When *month* is ``None``, the entire sheet is cleared and rewritten
    with all provided transactions.

    Args:
        transactions: Transaction objects to push.
        config: Sheets config (credentials, spreadsheet ID, worksheet).
        root: Project root for resolving relative credential paths.
        month: If set, only replace this month's data (upsert mode).

    Returns:
        Number of data rows written.
    """
    try:
        import gspread
    except ImportError as exc:
        raise ImportError(
            "Google Sheets integration requires gspread and google-auth. "
            "Install them with: pip install 'expense-tracker[sheets]'"
        ) from exc

    client = _authenticate(config, root)
    worksheet = _get_worksheet(client, config)

    if month is not None:
        return _upsert_month(worksheet, transactions, month)
    else:
        return _replace_all(worksheet, transactions)


def _replace_all(worksheet, transactions: list[Transaction]) -> int:
    """Clear the sheet and write all transactions."""
    rows = [COLUMNS]  # header
    for txn in transactions:
        rows.append(_txn_to_row(txn))

    worksheet.clear()
    worksheet.update(rows, value_input_option="USER_ENTERED")
    logger.info("Replaced all data: %d rows", len(transactions))
    return len(transactions)


def _upsert_month(
    worksheet, transactions: list[Transaction], month: str,
) -> int:
    """Replace only the specified month's rows, preserving everything else.

    1. Read existing data from the sheet.
    2. Remove rows where the ``month`` column matches.
    3. Append the new transactions for that month.
    4. Sort by date and write back.
    """
    # Read current sheet data
    existing = worksheet.get_all_values()

    if not existing:
        # Empty sheet — write header + data
        rows = [COLUMNS]
        for txn in transactions:
            rows.append(_txn_to_row(txn))
        worksheet.update(rows, value_input_option="USER_ENTERED")
        logger.info("Sheet was empty, wrote %d rows for %s", len(transactions), month)
        return len(transactions)

    header = existing[0]
    data_rows = existing[1:]

    # Find the month column index
    try:
        month_col = header.index("month")
    except ValueError:
        # No month column — fall back to full replace
        logger.warning("No 'month' column in sheet header. Doing full replace.")
        return _replace_all(worksheet, transactions)

    # Filter out rows for the target month
    kept_rows = [row for row in data_rows if row[month_col] != month]
    removed_count = len(data_rows) - len(kept_rows)

    # Build new rows for the target month
    new_rows = [_txn_to_row(txn) for txn in transactions]

    # Combine: kept rows + new rows
    all_data = kept_rows + new_rows

    # Sort by date (column index 1 = "date")
    date_col = header.index("date") if "date" in header else 1
    all_data.sort(key=lambda r: r[date_col] if len(r) > date_col else "")

    # Write back: header + sorted data
    output = [header] + all_data
    worksheet.clear()
    worksheet.update(output, value_input_option="USER_ENTERED")

    logger.info(
        "Upserted %s: removed %d old rows, wrote %d new rows (%d total)",
        month, removed_count, len(new_rows), len(all_data),
    )
    return len(new_rows)
