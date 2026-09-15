# AGENTS.md — Expense Tracker

Context for AI coding agents (pi, Claude Code, etc.) working in this repository.

## What This Is

Expense Tracker **v3**: a Python CLI (`expense`) that turns raw bank CSV exports
(Chase, Capital One, Elevations CU) into categorized monthly expense reports.
Pipeline: parse → filter month → exclude → dedupe → detect transfers → enrich
(item-level splits from Amazon/Target/Venmo scrapes) → categorize (rules +
Claude LLM) → export CSV → optionally push to Google Sheets.

- Python 3.12, src layout (`src/expense_tracker/`), Click CLI, ~9.3k LOC
- Full design doc: `docs/architecture.md` (read it before non-trivial changes)
- Issue tracking: **Beads** (`bd`) — see "Issue Tracking" below

## Commands

Use the project venv (`.venv/`) — either activate it or call binaries directly:

```bash
.venv/bin/pytest                     # run tests (or: make test)
.venv/bin/ruff check src/ tests/     # lint (or: make lint)
.venv/bin/ruff format src/ tests/    # format (or: make format)
.venv/bin/ruff format --check src/ tests/   # format check
.venv/bin/pip install -e ".[dev]"    # reinstall after dependency changes
```

Run a single test file while iterating: `.venv/bin/pytest tests/test_parsers.py -x`

CI (`.github/workflows/ci.yml`) runs `ruff check src/ tests/` and `pytest` on
Python 3.12. Note: `pyproject.toml` specifies `ruff>=0.8` unpinned, so CI
installs the latest ruff, which currently reports pre-existing violations
(see the note below).

**Before considering any task done: run `pytest` (must pass — currently 453
tests) and `ruff check` on the files you touched.** Note: the repo carries
pre-existing ruff debt (~110 violations under ruff 0.15.x, mostly in
`download/`, `enrichment/`, and E501 long lines) — don't add *new*
violations, and don't mass-fix unrelated lint as part of a feature change.

## Repo Layout

```
src/expense_tracker/
    cli.py            # Click commands: process, learn, enrich, push, download, init
    pipeline.py       # Stage orchestration (pure functions over Transaction lists)
    models.py         # Dataclasses. ZERO internal imports — everything depends on it
    config.py         # TOML load/save, project init
    categorizer.py    # Rule matching (merchant + description), learn workflow
    llm.py            # ClaudeCodeAdapter (primary), AnthropicAdapter, NullAdapter
    recurring.py      # Auto-detect recurring merchants from historical CSVs
    export.py         # Monthly CSV writer + stdout summary
    sheets.py         # Google Sheets push (month upsert or full rewrite)
    parsers/          # chase.py, capital_one.py, elevations.py + registry
    enrichment/       # amazon.py, target.py, venmo.py, cache.py + provider registry
    download/         # Playwright bank CSV downloaders + base.py (KeePass creds)
tests/                # pytest; fixtures in tests/fixtures/, helpers in conftest.py
docs/architecture.md  # The design document
.pi/settings.json     # Project pi settings (requires project trust to load)
.pi/prompts/          # Prompt templates: /monthly-close, /add-parser
rules.toml            # Merchant→category knowledge base (the crown jewel)
categories.toml       # 18-category taxonomy
config.toml.example   # Template; real config.toml is gitignored
```

Scoped agent docs: read `src/AGENTS.md` before modifying application code and
`tests/AGENTS.md` before writing tests.

## Hard Rules

1. **NEVER read, modify, print, or commit real data files.** These are real
   financial data and credentials, and are gitignored for a reason:
   - `config.toml` (real account config)
   - `input/`, `output/` (real bank CSVs and reports)
   - `enrichment-cache/` (real order history)
   - `.auth/` (browser sessions, Google service account, KeePass paths)
   - Any `*.csv` outside `tests/fixtures/`

   Use `config.toml.example`, `tests/fixtures/`, and `conftest.py`'s
   `tmp_project_dir` fixture instead.

2. **Never hand-edit the `[learned_rules]` section of `rules.toml`.** It is
   managed by `expense learn`. User rules go in `[user_rules]`.

3. **Conventional commits:** `feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
   `chore:` — matching existing git history.

4. **Run tests before declaring done; keep lint clean on files you touch**
   (see Commands — note the pre-existing lint debt above).

## Cautions

- `expense download` and `expense enrich` drive **Playwright against real
  bank/retailer accounts** using KeePass credentials and saved sessions in
  `.auth/`. Only run them when the user explicitly asks; they may trigger MFA
  prompts or bot detection.
- `expense push` writes to a real Google Sheet. `--all` clears and rewrites
  the entire sheet — prefer `--month YYYY-MM` upserts, and confirm with the
  user before pushing.
- Default LLM categorization shells out to the `claude` CLI subprocess
  (user's Max subscription, ~120s timeout per 80-txn batch). In tests, always
  inject a mock/`NullAdapter` — never invoke a real LLM.

## Issue Tracking (Beads)

This repo uses [Beads](https://github.com/steveyegge/beads) (`bd`, prefix
`exp-`). Issues live in the repo's Dolt database.

```bash
bd list                  # all issues
bd ready                 # unblocked work
bd show exp-<id>         # details
bd create "Title"        # new issue
bd update exp-<id> --claim          # claim before starting work
bd update exp-<id> --status done    # close when finished
```

When asked to pick up work, check `bd ready` first. Claim issues you start,
close them when done, and file a `bd create` for notable bugs/TODOs you
discover but don't fix.

## Conventions

- Line length 100; ruff rules `E,F,I,UP,B,SIM`; `from __future__ import annotations`
- Docstrings on all public functions (Google style); type hints everywhere
- Decimal for money, `date` for dates — never float/str in the domain model
- Parser modules expose `parse(file_path, institution, account) -> StageResult`
  and register in `parsers/__init__.py`'s `PARSERS` dict
- Deterministic transaction IDs via `models.generate_transaction_id` — parsers
  own merchant normalization; learned rules depend on stable merchant strings
- `models.py` must keep **zero internal imports**. Pipeline stages are pure
  functions returning `StageResult` — no exceptions across stage boundaries
  (partial failure over total failure)
