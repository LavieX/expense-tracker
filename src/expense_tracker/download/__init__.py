"""Bank transaction CSV download automation.

Uses Playwright browser automation to log into bank websites and download
transaction CSVs for a given date range.

Supported banks:
- Chase (credit card)
- Capital One (credit card)
- Elevations Credit Union (checking)

Session persistence
-------------------
All three banks employ bot detection (Cloudflare Turnstile, device
fingerprinting, MFA challenges) that prevent fully headless automation on
first login.  The workflow mirrors the Target enrichment scraper:

1. **First run** — a visible browser opens.  The user handles CAPTCHA / MFA
   manually, then the session is saved to ``.auth/<bank>/state.json``.
2. **Subsequent runs** — the saved session is reused.  If the session has
   expired the script falls back to interactive login.

Credentials are read from a KeePass ``.kdbx`` file whose path and master
password are provided via environment variables or CLI flags.
"""
