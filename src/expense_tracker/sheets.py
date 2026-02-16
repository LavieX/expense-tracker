"""Google Sheets integration for pushing processed transaction data.

This module provides functionality to push processed monthly transactions to
a Google Sheets spreadsheet using the gspread library and service account
authentication.
"""

from __future__ import annotations

from pathlib import Path

from expense_tracker.models import SheetsConfig, Transaction


def push_to_sheets(
    transactions: list[Transaction],
    config: SheetsConfig,
    root: Path,
) -> int:
    """Push transactions to Google Sheets.

    This function:
    1. Authenticates via service account JSON
    2. Opens spreadsheet by ID
    3. Finds or creates worksheet by name
    4. Clears existing data
    5. Writes header row + all transaction rows in one batch API call
    6. Returns count of rows written

    The caller is responsible for gathering transactions across all desired
    months.  This function writes whatever it receives in a single batch.

    Args:
        transactions: List of Transaction objects to push to the sheet.
            May span multiple months.
        config: SheetsConfig with credentials, spreadsheet ID, and worksheet name.
        root: Project root directory (used to resolve relative credentials path).

    Returns:
        The number of data rows written (excludes header row).

    Raises:
        ImportError: If gspread or google-auth are not installed.
        FileNotFoundError: If the credentials file does not exist.
        Exception: For various Google Sheets API errors (auth, network, etc.).
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:
        raise ImportError(
            "Google Sheets integration requires gspread and google-auth. "
            "Install them with: pip install 'expense-tracker[sheets]'"
        ) from exc

    # Resolve credentials file path (may be relative to project root or use ~)
    creds_path = Path(config.credentials_file).expanduser()
    if not creds_path.is_absolute():
        creds_path = root / creds_path

    if not creds_path.exists():
        raise FileNotFoundError(
            f"Google service account credentials not found: {creds_path}"
        )

    # Define the required scopes for Google Sheets API
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # Authenticate using service account
    credentials = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
    client = gspread.authorize(credentials)

    # Open the spreadsheet by ID
    spreadsheet = client.open_by_key(config.spreadsheet_id)

    # Find or create the worksheet
    try:
        worksheet = spreadsheet.worksheet(config.worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=config.worksheet_name,
            rows=1000,
            cols=20,
        )

    # Define column order to match CSV export.
    # The instructions say another agent will update export.py to add is_recurring
    # after is_return. We'll include it here based on the plan.
    columns = [
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

    # Build rows: header + data rows
    rows = [columns]  # Header row

    for txn in transactions:
        rows.append([
            txn.transaction_id,
            txn.date.isoformat(),  # Format date as YYYY-MM-DD
            txn.date.strftime("%Y-%m"),  # Month as YYYY-MM
            txn.merchant,
            txn.description,
            float(txn.amount),  # Convert Decimal to float for Sheets
            txn.institution,
            txn.account,
            txn.category,
            txn.subcategory,
            "TRUE" if txn.is_return else "FALSE",  # Boolean as TRUE/FALSE string
            "TRUE" if txn.is_recurring else "FALSE",
            txn.split_from,
            txn.source,
        ])

    # Clear existing content and write all data in one batch update
    worksheet.clear()
    worksheet.update(rows, value_input_option="USER_ENTERED")

    # Return count of data rows (excluding header)
    return len(transactions)
