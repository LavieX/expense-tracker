"""Venmo transaction enrichment provider.

Scrapes Venmo web transaction history to extract the memo/note for each
transaction, then matches to bank transactions (Elevations checking) by
date and amount.

Venmo transactions appear in the bank statement as generic
"VENMO TYPE: PAYMENT..." entries.  This enrichment replaces the generic
description with the actual Venmo memo (e.g. "Babysitter", "Hockey",
"Dinner split") so the transaction can be properly categorized.

Two Venmo accounts are supported:
- Colleen's: colleentenedios@gmail.com
- Lavie's: lavie.tobey@gmail.com

Uses persistent browser profiles so "Remember Me" / device trust persists.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

VENMO_LOGIN_URL = "https://venmo.com/account/sign-in"
VENMO_STATEMENTS_URL = "https://account.venmo.com/statements"

# KeePass entry titles for the two accounts
VENMO_ACCOUNTS = [
    {"label": "colleen", "entry_title": "Venmo", "url": "https://venmo.com"},
    {"label": "lavie", "entry_title": "Venmo (Lavie)", "url": "https://id.venmo.com"},
]

MFA_TIMEOUT = 180  # seconds


@dataclass
class VenmoTransaction:
    """A single Venmo transaction with memo detail."""
    transaction_id: str
    date: date
    amount: Decimal
    note: str  # The memo/description from Venmo
    payer: str
    payee: str
    transaction_type: str  # "Payment", "Charge"
    account_label: str


def _is_logged_in(page) -> bool:
    """Check if we're on an authenticated Venmo page."""
    url = page.url.lower()
    return any(kw in url for kw in [
        "account.venmo.com", "venmo.com/account",
        "venmo.com/home", "venmo.com/statements",
    ]) and "sign-in" not in url


def _is_login_page(page) -> bool:
    url = page.url.lower()
    return "sign-in" in url or "login" in url or "auth" in url


def scrape_venmo_transactions(
    month: str,
    auth_dir: Path | None = None,
    accounts: list[dict] | None = None,
) -> list[VenmoTransaction]:
    """Scrape Venmo transaction history for all configured accounts.

    Args:
        month: Target month as "YYYY-MM".
        auth_dir: Base auth directory. Defaults to ".auth".
        accounts: List of account configs. Defaults to VENMO_ACCOUNTS.

    Returns:
        List of VenmoTransaction objects for the target month.
    """
    from playwright.sync_api import sync_playwright

    if auth_dir is None:
        auth_dir = Path(".auth")
    if accounts is None:
        accounts = VENMO_ACCOUNTS

    year, mon = month.split("-")
    year_int, mon_int = int(year), int(mon)

    all_transactions: list[VenmoTransaction] = []

    for acct in accounts:
        label = acct["label"]
        logger.info("Scraping Venmo transactions (%s)...", label)

        try:
            from expense_tracker.download.base import get_credentials
            username, password = get_credentials(
                entry_title=acct["entry_title"],
                url=acct.get("url", ""),
            )
        except Exception as exc:
            logger.error("Could not get credentials for Venmo (%s): %s", label, exc)
            continue

        profile = auth_dir / f"venmo-{label}" / "browser-profile"
        profile.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                headless=False,
                viewport={"width": 1280, "height": 900},
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            page = context.pages[0] if context.pages else context.new_page()

            try:
                txns = _scrape_account(
                    page, username, password, label, year_int, mon_int
                )
                all_transactions.extend(txns)
                logger.info(
                    "Venmo (%s): found %d transactions for %s", label, len(txns), month
                )
            except Exception as exc:
                logger.error("Venmo scraping failed (%s): %s", label, exc)
            finally:
                context.close()

    return all_transactions


def _scrape_account(
    page, username: str, password: str, label: str,
    year: int, month: int,
) -> list[VenmoTransaction]:
    """Scrape one Venmo account's transactions for the given month."""

    # Navigate to statements page
    page.goto(VENMO_STATEMENTS_URL, wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    # Check if we need to log in
    if _is_login_page(page):
        logger.info("Venmo (%s) login required. Filling credentials...", label)

        # Fill login form
        try:
            email_input = page.query_selector(
                'input[name="phoneEmailUsername"], '
                'input[type="email"], '
                'input[name="email"]'
            )
            if email_input:
                email_input.fill(username)

            pass_input = page.query_selector(
                'input[name="password"], input[type="password"]'
            )
            if pass_input:
                pass_input.fill(password)

            time.sleep(1)

            signin_btn = page.query_selector(
                'button:has-text("Sign In"), '
                'button[type="submit"]'
            )
            if signin_btn:
                signin_btn.click()

            print(f"  Venmo ({label}): Creds filled. Complete MFA if prompted...")
        except Exception as exc:
            logger.warning("Could not fill Venmo login: %s. Please log in manually.", exc)

        # Wait for login to complete
        start_time = time.time()
        while time.time() - start_time < MFA_TIMEOUT:
            if _is_logged_in(page):
                logger.info("Venmo (%s) login successful!", label)
                break
            time.sleep(3)
        else:
            raise RuntimeError(f"Venmo ({label}) login timed out.")

        # Navigate to statements after login
        page.goto(VENMO_STATEMENTS_URL, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(3)

    # Try to download CSV via the statements page
    transactions = _download_statement_csv(page, label, year, month)

    if not transactions:
        # Fallback: scrape from the transaction feed
        transactions = _scrape_transaction_feed(page, label, year, month)

    return transactions


def _download_statement_csv(
    page, label: str, year: int, month: int,
) -> list[VenmoTransaction]:
    """Try to download the CSV statement for the given month."""
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)

    # Look for statement download options on the page
    # Venmo statements page has date selectors and download buttons
    page_text = page.evaluate("document.body.innerText.substring(0, 3000)")

    # Try to find and click CSV download
    # Look for the month's statement or a date range selector
    csv_link = page.query_selector(
        'a:has-text("CSV"), '
        'button:has-text("CSV"), '
        'a:has-text("Download"), '
        'button:has-text("Download")'
    )

    if csv_link:
        # Try to get the CSV content
        try:
            with page.expect_download(timeout=15000) as dl_info:
                csv_link.click()
            dl = dl_info.value
            content = dl.path().read_text(encoding="utf-8")
            return _parse_venmo_csv(content, label, month_start, month_end)
        except Exception:
            pass

    logger.info("Venmo (%s): No direct CSV download found, trying feed scrape.", label)
    return []


def _scrape_transaction_feed(
    page, label: str, year: int, month: int,
) -> list[VenmoTransaction]:
    """Scrape transactions from the Venmo transaction feed/history."""
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)

    # Navigate to transaction history
    page.goto("https://account.venmo.com/activity", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    transactions: list[VenmoTransaction] = []

    # Scroll and collect transactions
    # Venmo's feed shows transactions with date, amount, note, and participants
    for scroll_attempt in range(20):
        entries = page.evaluate(r'''() => {
            const results = [];
            // Look for transaction items in the feed
            const items = document.querySelectorAll(
                '[data-testid*="transaction"], ' +
                '[class*="transaction"], ' +
                '[class*="story"], ' +
                'div[role="listitem"]'
            );
            for (const item of items) {
                const text = item.innerText.trim();
                results.push(text.substring(0, 300));
            }
            return results;
        }''')

        if entries:
            for entry_text in entries:
                txn = _parse_feed_entry(entry_text, label, month_start, month_end)
                if txn and txn.transaction_id not in {t.transaction_id for t in transactions}:
                    transactions.append(txn)

        # Check if we've scrolled past our target month
        # (Venmo shows newest first, so once we see dates before our month, stop)
        if entries:
            last_text = entries[-1].lower()
            # Simple heuristic - if we see dates from before our month, stop
            for prev_month in range(month - 1, 0, -1):
                month_name = date(year, prev_month, 1).strftime("%B").lower()
                if month_name in last_text:
                    logger.debug("Reached %s in feed, stopping scroll.", month_name)
                    return transactions

        # Scroll down for more
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)

    return transactions


def _parse_feed_entry(
    text: str, label: str, month_start: date, month_end: date
) -> VenmoTransaction | None:
    """Parse a single transaction entry from the Venmo feed text."""
    # Venmo feed entries typically look like:
    # "You paid Person Name\n$XX.XX\nMemo note\nJan 15"
    # or "Person Name paid You\n$XX.XX\nMemo note\nJan 15"

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return None

    # Try to extract amount
    amount = None
    for line in lines:
        m = re.search(r"\$[\d,]+\.?\d*", line)
        if m:
            amount = Decimal(m.group().replace("$", "").replace(",", ""))
            break

    if amount is None:
        return None

    # Try to extract date
    txn_date = None
    for line in lines:
        for fmt in ["%b %d", "%b %d, %Y", "%B %d", "%B %d, %Y"]:
            try:
                parsed = datetime.strptime(line.strip(), fmt)
                if parsed.year == 1900:  # No year in format
                    parsed = parsed.replace(year=month_start.year)
                txn_date = parsed.date()
                break
            except ValueError:
                continue
        if txn_date:
            break

    if txn_date is None or txn_date < month_start or txn_date > month_end:
        return None

    # Extract note/memo (usually the line that isn't a date, amount, or payer info)
    note = ""
    payer = ""
    payee = ""
    txn_type = "Payment"
    for line in lines:
        lower = line.lower()
        if "you paid" in lower:
            txn_type = "Payment"
            payee = line.split("paid")[-1].strip() if "paid" in lower else ""
            payer = "You"
        elif "paid you" in lower:
            txn_type = "Charge"
            payer = line.split("paid")[0].strip()
            payee = "You"
        elif "charged" in lower:
            txn_type = "Charge"
        elif not re.search(r"\$[\d,]+\.?\d*", line) and not re.match(r"^[A-Z][a-z]{2}\s+\d", line):
            if line and len(line) > 1 and line not in (payer, payee):
                note = line

    return VenmoTransaction(
        transaction_id=f"venmo-{label}-{txn_date.isoformat()}-{amount}",
        date=txn_date,
        amount=amount,
        note=note,
        payer=payer,
        payee=payee,
        transaction_type=txn_type,
        account_label=label,
    )


def _parse_venmo_csv(
    content: str, label: str, month_start: date, month_end: date,
) -> list[VenmoTransaction]:
    """Parse a Venmo CSV export into VenmoTransaction objects."""
    transactions: list[VenmoTransaction] = []

    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        # Venmo CSV fields vary, but common ones:
        # ID, Datetime, Type, Status, Note, From, To, Amount (total/fee/etc)
        try:
            # Try multiple date field names
            date_str = row.get("Datetime") or row.get("Date") or row.get("datetime") or ""
            if not date_str:
                continue

            # Parse date (Venmo uses ISO format or MM/DD/YYYY)
            txn_date = None
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y-%m-%d"]:
                try:
                    txn_date = datetime.strptime(date_str.split("T")[0] if "T" in date_str else date_str, fmt.split("T")[0]).date()
                    break
                except ValueError:
                    continue

            if txn_date is None or txn_date < month_start or txn_date > month_end:
                continue

            # Amount
            amt_str = (
                row.get("Amount (total)") or row.get("Amount") or
                row.get("amount") or "0"
            )
            amt_str = amt_str.replace("$", "").replace(",", "").replace("+", "").replace(" ", "")
            if not amt_str or amt_str == "0":
                continue
            amount = Decimal(amt_str)

            note = row.get("Note") or row.get("note") or row.get("Description") or ""
            payer = row.get("From") or row.get("from") or ""
            payee = row.get("To") or row.get("to") or ""
            txn_type = row.get("Type") or row.get("type") or "Payment"
            txn_id = row.get("ID") or row.get("id") or f"{txn_date}-{amount}"

            transactions.append(VenmoTransaction(
                transaction_id=f"venmo-{label}-{txn_id}",
                date=txn_date,
                amount=abs(amount),
                note=note.strip(),
                payer=payer.strip(),
                payee=payee.strip(),
                transaction_type=txn_type.strip(),
                account_label=label,
            ))
        except Exception as exc:
            logger.debug("Skipped Venmo CSV row: %s", exc)
            continue

    return transactions


def match_venmo_to_bank(
    venmo_txns: list[VenmoTransaction],
    bank_txns: list[dict],
    date_window: int = 3,
) -> list[dict]:
    """Match Venmo transactions to bank transactions by date and amount.

    Bank transactions from Elevations appear as:
        "VENMO TYPE: PAYMENT  ID: ... CO: VENMO NAME: ..."

    Returns list of matches: [{bank_txn_id, venmo_txn, ...}]
    """
    matches = []
    matched_bank_ids = set()
    matched_venmo_ids = set()

    # Filter bank txns to just Venmo entries
    venmo_bank = [
        t for t in bank_txns
        if "venmo" in (t.get("merchant", "") + t.get("description", "")).lower()
    ]

    for vtxn in venmo_txns:
        best_match = None
        best_date_diff = date_window + 1

        for btxn in venmo_bank:
            if btxn.get("transaction_id") in matched_bank_ids:
                continue

            # Compare amounts (bank has negative sign for debits)
            bank_amt = abs(Decimal(str(btxn.get("amount", 0))))
            if bank_amt != vtxn.amount:
                continue

            # Compare dates (within window)
            try:
                bank_date = date.fromisoformat(btxn["date"])
            except (KeyError, ValueError):
                continue

            diff = abs((bank_date - vtxn.date).days)
            if diff <= date_window and diff < best_date_diff:
                best_match = btxn
                best_date_diff = diff

        if best_match:
            matches.append({
                "bank_transaction_id": best_match["transaction_id"],
                "venmo_note": vtxn.note,
                "venmo_payer": vtxn.payer,
                "venmo_payee": vtxn.payee,
                "venmo_type": vtxn.transaction_type,
                "venmo_account": vtxn.account_label,
            })
            matched_bank_ids.add(best_match["transaction_id"])
            matched_venmo_ids.add(vtxn.transaction_id)

    return matches


def enrich_venmo(
    month: str,
    transactions: list[dict],
    cache_dir: Path,
    auth_dir: Path | None = None,
) -> dict:
    """Scrape Venmo and write enrichment cache files.

    Returns summary dict with counts.
    """
    venmo_txns = scrape_venmo_transactions(month, auth_dir=auth_dir)
    matches = match_venmo_to_bank(venmo_txns, transactions)

    cache_dir.mkdir(parents=True, exist_ok=True)

    for match in matches:
        cache_file = cache_dir / f"{match['bank_transaction_id']}.json"
        cache_data = {
            "transaction_id": match["bank_transaction_id"],
            "source": "venmo",
            "venmo_note": match["venmo_note"],
            "venmo_payer": match["venmo_payer"],
            "venmo_payee": match["venmo_payee"],
            "venmo_type": match["venmo_type"],
            "venmo_account": match["venmo_account"],
            "matched_at": datetime.now().isoformat(),
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Wrote Venmo enrichment: %s", cache_file)

    return {
        "venmo_transactions": len(venmo_txns),
        "matched": len(matches),
        "cache_written": len(matches),
    }
