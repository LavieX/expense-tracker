"""Capital One credit card CSV download automation.

Uses a persistent browser profile for session persistence.  Downloads
via the Capital One API endpoint::

    /web-api/protected/17463/credit-cards/accounts/{ACCOUNT_ID}/
    transactions/download?fromTransactionDate=...&toTransactionDate=...
    &documentFormatType=application/csv

The API is called via ``fetch()`` from the authenticated page context.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

from .base import DEFAULT_AUTH_DIR, download_dir, get_credentials, launch_persistent

logger = logging.getLogger(__name__)

BANK = "capital-one"
KEEPASS_ENTRY = "Capital One REI Mastercard"
LOGIN_URL = "https://verified.capitalone.com/auth/"
DASHBOARD_URL = "https://myaccounts.capitalone.com/accountSummary"

# Known account ID (URL-encoded).  Extracted from the user's curl.
DEFAULT_ACCOUNT_ID = "9iDGj77W%252FysHNCWw8cgl5ZIKWwmkKpPFTQxyK49gGDo%253D"

DASHBOARD_KW = ["account summary", "current balance", "available credit", "payment due", "recent transactions"]
MFA_TIMEOUT = 180


def _is_dashboard(text: str) -> bool:
    return any(kw in text.lower() for kw in DASHBOARD_KW)


async def download_capital_one(
    month: str,
    root: Path = Path.cwd(),
    auth_dir: Path = DEFAULT_AUTH_DIR,
    keepass_file: str | None = None,
    keepass_password: str | None = None,
    headless: bool = False,
) -> Path | None:
    username, password = get_credentials(
        keepass_path=keepass_file,
        keepass_password=keepass_password,
        entry_title=KEEPASS_ENTRY,
    )

    year, mon = month.split("-")
    start = date(int(year), int(mon), 1)
    end = date(int(year), int(mon) + 1, 1) - timedelta(days=1) if int(mon) < 12 else date(int(year), 12, 31)

    async with async_playwright() as pw:
        context = await launch_persistent(pw, BANK, auth_dir, headless=headless)
        page = context.pages[0] if context.pages else await context.new_page()

        # Check if already logged in
        print("  Loading Capital One...")
        await page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(5)

        text = ""
        for i in range(10):
            await asyncio.sleep(2)
            try:
                text = await page.evaluate("document.body.innerText.substring(0, 500)")
                if _is_dashboard(text):
                    print("  Already logged in!")
                    break
            except:
                pass

        if not _is_dashboard(text):
            if headless:
                print("  Capital One requires interactive login. Re-run without --headless.")
                await context.close()
                return None

            # Navigate to login
            await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)

            user_input = await page.query_selector(
                '#username, input[name="username"], input[autocomplete="username"]'
            )
            pass_input = await page.query_selector(
                '#password, input[name="password"], input[type="password"]'
            )

            if user_input and pass_input:
                await user_input.fill(username)
                await pass_input.fill(password)
                signin = await page.query_selector(
                    'button:has-text("Sign In"), button[type="submit"]'
                )
                if signin:
                    await signin.click()
                print("  Creds filled. Complete any MFA...")
            else:
                print("  Enter credentials manually...")

            for i in range(MFA_TIMEOUT // 3):
                await asyncio.sleep(3)
                try:
                    text = await page.evaluate("document.body.innerText.substring(0, 500)")
                    if _is_dashboard(text):
                        print("  Dashboard loaded!")
                        break
                except:
                    pass
            else:
                print("  Login timed out.")
                await context.close()
                return None

        # Extract account ID from page (or use default)
        account_id = await page.evaluate(r'''() => {
            const links = Array.from(document.querySelectorAll('a[href*="/Card/"]'));
            for (const a of links) {
                const m = a.href.match(/\/Card\/([^/]+)/);
                if (m) return m[1];
            }
            return null;
        }''')
        account_id = account_id or DEFAULT_ACCOUNT_ID

        # Download via API
        url = (
            f"https://myaccounts.capitalone.com/web-api/protected/17463/"
            f"credit-cards/accounts/{account_id}/transactions/download"
            f"?fromTransactionDate={start.isoformat()}"
            f"&toTransactionDate={end.isoformat()}"
            f"&documentFormatType=application/csv"
            f"&acceptLanguage=en-US"
        )

        print("  Downloading via API...")
        result = await page.evaluate('''async (url) => {
            try {
                const resp = await fetch(url, {
                    credentials: 'include',
                    headers: {
                        'accept': 'application/json;v=1',
                        'x-user-action': 'ease.downloadTransactions',
                    },
                });
                if (!resp.ok) return {error: resp.status, text: (await resp.text()).substring(0, 500)};
                return {ok: true, text: await resp.text()};
            } catch(e) { return {fetchError: e.message}; }
        }''', url)

        dest = download_dir(BANK, root)

        if isinstance(result, dict) and result.get("ok") and len(result.get("text", "")) > 20:
            out = dest / f"capital-one_{month}.csv"
            out.write_text(result["text"], encoding="utf-8")
            lines = result["text"].strip().count("\n")
            print(f"  ✓ {out} ({lines} transactions)")
            await context.close()
            return out
        else:
            logger.error("API download failed: %s", result)
            print(f"  ✗ API failed: {result}")
            await context.close()
            return None
