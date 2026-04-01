"""Elevations Credit Union checking account CSV download automation.

Uses a persistent browser profile.  Elevations has Cloudflare Turnstile
on the login page — first run requires the user to click "Verify you are
human".  With a persistent profile, subsequent runs may skip this.

Downloads via the Elevations export API::

    POST https://secure.elevationscu.com/MyAccountsV2/Export

Multipart form with format=54 (CSV), date range, and account UUID.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

from .base import DEFAULT_AUTH_DIR, download_dir, get_credentials, launch_persistent

logger = logging.getLogger(__name__)

BANK = "elevations"
KEEPASS_ENTRY = "Elevations CU (Joint)"
LOGIN_URL = "https://www.elevationscu.com/login"
DASHBOARD_URL = "https://secure.elevationscu.com/Dashboard"
EXPORT_URL = "https://secure.elevationscu.com/MyAccountsV2/Export"
ACCOUNTS_URL = "https://secure.elevationscu.com/MyAccountsV2"

DEFAULT_ACCOUNT_ID = "6c3e3322-e340-4048-ae01-e3c7662bef6c"
FORMAT_CSV = "54"

DASHBOARD_KW = ["checking", "savings", "account balance", "available balance", "transaction"]
MFA_TIMEOUT = 180


def _is_dashboard(text: str) -> bool:
    return any(kw in text.lower() for kw in DASHBOARD_KW)


async def download_elevations(
    month: str,
    root: Path = Path.cwd(),
    auth_dir: Path = DEFAULT_AUTH_DIR,
    keepass_file: str | None = None,
    keepass_password: str | None = None,
    headless: bool = False,
    account_id: str = DEFAULT_ACCOUNT_ID,
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
        print("  Loading Elevations...")
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
                print("  Elevations requires interactive login (Turnstile). Re-run without --headless.")
                await context.close()
                return None

            await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)

            # Fill username and password
            try:
                await page.focus("#username")
                await page.keyboard.type(username, delay=50)
                await page.focus("#password")
                await page.keyboard.type(password, delay=50)
                print("  Creds filled.")
            except Exception as exc:
                print(f"  Could not fill creds: {exc}")

            # Wait for Turnstile to pass, then click Log In
            await asyncio.sleep(2)
            for _attempt in range(15):
                try:
                    disabled = await page.evaluate(
                        'document.querySelector("#btn_submitCredentials")'
                        '?.getAttribute("aria-disabled")'
                    )
                    if disabled != "true":
                        await page.click("#btn_submitCredentials", force=True)
                        print("  Clicked Log In.")
                        break
                except:
                    pass
                await asyncio.sleep(2)
            else:
                print("  Log In button stayed disabled. Click it manually.")

            for i in range(MFA_TIMEOUT // 3):
                await asyncio.sleep(3)
                try:
                    text = await page.evaluate("document.body.innerText.substring(0, 1000)")
                    if _is_dashboard(text):
                        print("  Dashboard loaded!")
                        break
                except:
                    pass
            else:
                print("  Login timed out.")
                await context.close()
                return None

        # Let the session fully establish before navigating
        await asyncio.sleep(5)

        # Navigate to dashboard first (always works after login)
        print("  Fetching CSRF token...")
        await page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)

        # Then to accounts page for CSRF
        await page.goto(ACCOUNTS_URL, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)

        csrf = await page.evaluate('''() => {
            const el = document.querySelector('input[name="__RequestVerificationToken"]');
            return el ? el.value : null;
        }''')

        if not csrf:
            print("  ✗ Could not get CSRF token — session may have expired.")
            await context.close()
            return None

        # Download via API
        print("  Downloading via API...")
        csv_result = await page.evaluate('''async (params) => {
            const formData = new FormData();
            formData.append('__RequestVerificationToken', params.csrf);
            formData.append('Parameters.TransactionCategoryId', '');
            formData.append('Parameters.Debit', '');
            formData.append('Parameters.Description', '');
            formData.append('Parameters.MaximumAmount', '-1');
            formData.append('Parameters.MinimumAmount', '');
            formData.append('Parameters.TransactionTypeId', '');
            formData.append('Parameters.StartCheckNumber', '');
            formData.append('Parameters.EndCheckNumber', '');
            formData.append('format', params.format);
            formData.append('Parameters.StartDate', params.startDate);
            formData.append('Parameters.EndDate', params.endDate);
            formData.append('AccountIdentifiers', params.accountId);

            const resp = await fetch(params.url, {
                method: 'POST',
                body: formData,
                credentials: 'include',
            });
            if (!resp.ok) return {error: resp.status, text: (await resp.text()).substring(0, 500)};
            return {ok: true, text: await resp.text()};
        }''', {
            "url": EXPORT_URL,
            "csrf": csrf,
            "format": FORMAT_CSV,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "accountId": account_id,
        })

        dest = download_dir(BANK, root)

        if isinstance(csv_result, dict) and csv_result.get("ok") and len(csv_result.get("text", "")) > 20:
            out = dest / f"elevations_{month}.csv"
            out.write_text(csv_result["text"], encoding="utf-8")
            lines = csv_result["text"].strip().count("\n")
            print(f"  ✓ {out} ({lines} transactions)")
            await context.close()
            return out
        else:
            logger.error("API download failed: %s", csv_result)
            print(f"  ✗ API failed: {csv_result}")
            await context.close()
            return None
