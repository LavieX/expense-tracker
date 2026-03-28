"""Chase credit card CSV download automation.

Uses a persistent browser profile so "Remember Me" persists across runs.
On first run, user completes MFA manually.  Subsequent runs skip login.

Download is done via the Chase download UI at::

    #/dashboard/accountDetails/downloadAccountTransactions/
    index;params=CARD,BAC,{ACCOUNT_ID}

Chase uses deeply nested shadow-DOM ``MDS-*`` web components.  Standard
selectors can't reach them, so we recursively walk shadow roots to find
elements by text content and click them by coordinate.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

from .base import DEFAULT_AUTH_DIR, download_dir, get_credentials, launch_persistent

logger = logging.getLogger(__name__)

BANK = "chase"
KEEPASS_ENTRY = "Chase Bank"
LOGIN_URL = "https://secure.chase.com"
ACCOUNT_ID = "747565458"
DOWNLOAD_URL = (
    "https://secure.chase.com/web/auth/dashboard"
    "#/dashboard/accountDetails/downloadAccountTransactions"
    f"/index;params=CARD,BAC,{ACCOUNT_ID}"
)

DASHBOARD_KW = ["current balance", "payment due", "available credit", "last statement"]
MFA_TIMEOUT = 180


def _is_dashboard(text: str) -> bool:
    return any(kw in text.lower() for kw in DASHBOARD_KW)


async def _ensure_logged_in(page, username: str, password: str, headless: bool) -> bool:
    """Navigate to Chase and log in if needed.  Returns True if on dashboard."""
    await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)

    for i in range(10):
        await asyncio.sleep(2)
        try:
            text = await page.evaluate("document.body.innerText.substring(0, 500)")
            if _is_dashboard(text):
                print("  Already logged in (session remembered)!")
                return True
        except:
            pass

    if headless:
        print("  Chase requires interactive login. Re-run without --headless.")
        return False

    iframe_el = await page.query_selector("#logonbox")
    if iframe_el:
        iframe = await iframe_el.content_frame()
        await iframe.fill("#userId-input-field-input", username)
        await iframe.fill("#password-input-field-input", password)
        try:
            await iframe.check("#rememberMe")
        except:
            pass
        print("  Creds filled. Complete MFA and sign in...")

    for i in range(MFA_TIMEOUT // 3):
        await asyncio.sleep(3)
        try:
            text = await page.evaluate("document.body.innerText.substring(0, 500)")
            if _is_dashboard(text):
                print("  Dashboard loaded!")
                return True
        except:
            pass

    return False


async def _find_shadow_element(page, text_content: str) -> dict | None:
    """Recursively walk all shadow roots to find a visible element by text.

    Returns ``{x, y}`` center coordinates for clicking, or *None*.
    """
    results = await page.evaluate(
        """(target) => {
            function walk(root, depth) {
                const hits = [];
                for (const el of root.querySelectorAll('*')) {
                    const t = (el.textContent || '').trim();
                    if (t === target) {
                        const r = el.getBoundingClientRect();
                        if (r.height > 1 && r.width > 1) {
                            hits.push({x: r.x + r.width/2, y: r.y + r.height/2, depth});
                        }
                    }
                    if (el.shadowRoot) hits.push(...walk(el.shadowRoot, depth + 1));
                }
                return hits;
            }
            return walk(document, 0);
        }""",
        text_content,
    )
    # Return the deepest visible hit (most specific element).
    if not results:
        return None
    results.sort(key=lambda r: r["depth"], reverse=True)
    return results[0]


async def download_chase(
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
    end = (
        date(int(year), int(mon) + 1, 1) - timedelta(days=1)
        if int(mon) < 12
        else date(int(year), 12, 31)
    )

    async with async_playwright() as pw:
        context = await launch_persistent(pw, BANK, auth_dir, headless=headless)
        page = context.pages[0] if context.pages else await context.new_page()

        if not await _ensure_logged_in(page, username, password, headless):
            await context.close()
            return None

        # Navigate to download page.
        print("  Opening download page...")
        await page.goto(DOWNLOAD_URL, wait_until="domcontentloaded", timeout=30_000)
        for _ in range(15):
            await asyncio.sleep(2)
            count = await page.evaluate(
                'document.querySelectorAll("mds-select").length'
            )
            if count >= 3:
                break
        await asyncio.sleep(2)

        # Open the Activity dropdown.
        await page.evaluate(
            """() => {
                const s = document.querySelector('#downloadActivityOptionId');
                if (s && s.shadowRoot) {
                    const btn = s.shadowRoot.querySelector('button');
                    if (btn) btn.click();
                }
            }"""
        )
        await asyncio.sleep(1)

        # Click "Choose a date range" via coordinate (shadow DOM workaround).
        target = await _find_shadow_element(page, "Choose a date range")
        if target:
            await page.mouse.click(target["x"], target["y"])
            await asyncio.sleep(3)
        else:
            print("  Could not find 'Choose a date range' option.")
            print("  Please select it manually, fill dates, and download.")
            try:
                dl = await page.wait_for_event("download", timeout=180_000)
                out = download_dir(BANK, root) / f"chase_{month}.csv"
                await dl.save_as(str(out))
                print(f"  ✓ {out}")
                await context.close()
                return out
            except:
                await context.close()
                return None

        # Fill date inputs (they appear after selecting date range).
        for mds_input in await page.locator(
            "mds-text-input, mds-date-input"
        ).all():
            label = (await mds_input.get_attribute("label") or "").lower()
            try:
                inner = mds_input.locator("input").first
                if "from" in label or "start" in label:
                    await inner.fill(start.strftime("%m/%d/%Y"))
                    await inner.dispatch_event("change")
                elif "to" in label or "end" in label:
                    await inner.fill(end.strftime("%m/%d/%Y"))
                    await inner.dispatch_event("change")
            except Exception as exc:
                logger.debug("Could not fill date input %s: %s", label, exc)
        await asyncio.sleep(1)

        # Ensure CSV format.
        await page.evaluate(
            """() => {
                const s = document.querySelector('#downloadFileTypeOption');
                if (s) {
                    s.value = 'CSV';
                    s.dispatchEvent(new Event('change', {bubbles: true}));
                }
            }"""
        )

        # Click Download.
        dest = download_dir(BANK, root)
        try:
            async with page.expect_download(timeout=30_000) as dl_info:
                await page.locator("mds-button#download").locator("button").click()
            dl = await dl_info.value
            out = dest / f"chase_{month}.csv"
            await dl.save_as(str(out))
            content = out.read_text()
            lines = content.strip().count("\n")
            print(f"  ✓ {out} ({lines} transactions)")
            await context.close()
            return out
        except Exception as exc:
            logger.error("Auto-download failed: %s", exc)
            print("  Auto-download failed. Complete manually.")
            try:
                dl = await page.wait_for_event("download", timeout=120_000)
                out = dest / f"chase_{month}.csv"
                await dl.save_as(str(out))
                print(f"  ✓ {out}")
                await context.close()
                return out
            except:
                await context.close()
                return None
