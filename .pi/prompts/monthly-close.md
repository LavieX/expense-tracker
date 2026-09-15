---
description: Run the full monthly expense workflow for a month, with human gates
argument-hint: "<YYYY-MM>"
---

Run the monthly expense close for month $1. Follow the working agreements in
AGENTS.md (never read/modify real data files yourself outside the CLI, run
lint+tests if you change code). Use the project venv binaries (`.venv/bin/`).

Execute these steps in order, stopping at each GATE for user confirmation:

1. **Download bank CSVs** (Playwright, hits real bank accounts — confirm with
   the user first). May require interactive MFA:
   `.venv/bin/expense download --month $1`
2. **Enrich** (also browser automation against real retailer accounts — confirm
   first). Skip any source the user doesn't want:
   `.venv/bin/expense enrich --month $1 --source amazon`
   `.venv/bin/expense enrich --month $1 --source target`
   `.venv/bin/expense enrich --month $1 --source venmo`
3. **Process**:
   `.venv/bin/expense process --month $1 --verbose`
4. **GATE — review.** Show the user the processing summary: categorization
   rate, top uncategorized merchants, warnings, and spending by category.
   Ask whether they want to correct anything before learning.
5. **Learn** (only after the user produces a corrected CSV):
   `.venv/bin/expense learn --original output/$1.csv --corrected output/$1-corrected.csv --verbose`
   If corrections were learned, offer to re-run step 3 so the output reflects
   the new rules.
6. **GATE — push.** Confirm before writing to the real Google Sheet, then
   upsert only this month (never `--all` unless explicitly requested):
   `.venv/bin/expense push --month $1 --verbose`
7. Report a final recap: transactions processed, categorized %, rules learned,
   rows pushed.

If any step fails, stop, show the error, and ask how to proceed — do not
retry browser-automation steps in a loop (risk of account lockouts).
