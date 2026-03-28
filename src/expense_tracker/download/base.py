"""Shared browser helpers for bank download and enrichment automation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default auth state directory (relative to project root).
DEFAULT_AUTH_DIR = Path(".auth")


def get_credentials(
    entry_title: str = "",
    url: str = "",
    keepass_path: str | Path | None = None,
    keepass_password: str | None = None,
) -> tuple[str, str]:
    """Read *username* and *password* for a service.

    Tries KPX first (preferred — uses the running credential server).
    Falls back to direct pykeepass if KPX is not available.
    """
    # Try KPX first
    try:
        from kpx.client import KPXClient

        kpx = KPXClient()
        if kpx.is_available():
            # Search by title or URL
            if url:
                creds = kpx.get_credentials(url)
                if creds and creds.get("username"):
                    logger.info("Credentials from KPX (url: %s)", url)
                    return creds["username"], creds["password"]
            if entry_title:
                results = kpx.search(entry_title)
                if results:
                    # Get the first match with full details
                    entry = kpx.get_entry(
                        uuid=results[0]["uuid"],
                        db_path=results[0]["db_path"],
                    )
                    if entry and entry.get("username"):
                        logger.info("Credentials from KPX (title: %s)", entry_title)
                        return entry["username"], entry["password"]
    except (ImportError, Exception) as exc:
        logger.debug("KPX not available: %s", exc)

    # Fallback to direct pykeepass
    try:
        from pykeepass import PyKeePass
    except ImportError:
        raise ImportError(
            "Neither kpx nor pykeepass is available for credential lookup. "
            "Start kpx server or install pykeepass."
        )

    kp_path = keepass_path or os.getenv("KEEPASS_FILE")
    kp_pass = keepass_password or os.getenv("KEEPASS_PASSWORD")

    if not kp_path:
        raise ValueError(
            "No KeePass file specified. Set KEEPASS_FILE or pass --keepass-file."
        )
    if not kp_pass:
        raise ValueError(
            "No KeePass password specified. Set KEEPASS_PASSWORD or pass --keepass-password."
        )

    kp = PyKeePass(str(kp_path), password=kp_pass)
    entry = kp.find_entries(title=entry_title, first=True)
    if entry is None:
        raise ValueError(f"KeePass entry '{entry_title}' not found.")

    return entry.username, entry.password


def profile_dir(bank: str, auth_dir: Path = DEFAULT_AUTH_DIR) -> Path:
    """Return (and create) the persistent browser profile directory for *bank*."""
    d = auth_dir / bank / "browser-profile"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_dir(bank: str, root: Path = Path.cwd()) -> Path:
    """Return (and create) the input directory for *bank*."""
    d = root / "input" / bank
    d.mkdir(parents=True, exist_ok=True)
    return d


async def launch_persistent(
    playwright, bank: str, auth_dir: Path = DEFAULT_AUTH_DIR, headless: bool = False
):
    """Launch a Chromium browser with a persistent profile.

    The profile directory persists cookies, localStorage, and device
    fingerprints across sessions — so "Remember Me" actually works.

    Returns a ``BrowserContext`` (which also acts as the browser handle).
    """
    pdir = profile_dir(bank, auth_dir)

    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(pdir),
        headless=headless,
        viewport={"width": 1280, "height": 900},
        accept_downloads=True,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )
    return context


def launch_persistent_sync(
    playwright, bank: str, auth_dir: Path = DEFAULT_AUTH_DIR, headless: bool = False
):
    """Sync version of launch_persistent for enrichment scrapers."""
    pdir = profile_dir(bank, auth_dir)

    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(pdir),
        headless=headless,
        viewport={"width": 1280, "height": 900},
        accept_downloads=True,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )
    return context
