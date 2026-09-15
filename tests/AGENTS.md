# AGENTS.md — tests

Test conventions for the expense tracker suite. Read the root `AGENTS.md` first.

## Running

```bash
.venv/bin/pytest                     # full suite
.venv/bin/pytest tests/test_parsers.py -x        # one file
.venv/bin/pytest -k "chase"          # by name
```

Config lives in `pyproject.toml`: `testpaths = ["tests"]`, `pythonpath = ["src"]`
(no install needed to run tests, but `pip install -e ".[dev]"` is the norm).

## Layout & Fixtures

- `tests/fixtures/` — sample bank CSVs per institution: `*_valid.csv`,
  `*_malformed.csv`, `*_empty.csv`, `*_wrong_format.csv`, `*_sample.csv`.
  **These are the only CSVs tests may use.** Add new parser edge cases as new
  fixture files here (fixtures are whitelisted in `.gitignore`).
- `conftest.py` provides:
  - `tmp_project_dir` — full temp project tree (input dirs, config files,
    output dirs) for integration-style tests
  - `sample_transactions` — realistic `Transaction` list spanning months,
    institutions, refunds, transfers
  - `sample_rules` — user + learned `MerchantRule` objects
  - Path helpers for fixture files

## Rules

1. **No real data, ever.** No reading `config.toml`, `input/`, `output/`,
   `enrichment-cache/`, or `.auth/` from the repo root. Everything runs
   against fixtures or `tmp_path`/`tmp_project_dir`.
2. **No real LLM calls.** Inject a mock adapter or `NullAdapter` into
   `categorizer.categorize`. Do not shell out to `claude` or hit the
   Anthropic API.
3. **No network, no Playwright.** Browser automation (`download/`,
   `enrichment/` scrapers) is tested via unit tests of pure helpers
   (parsing, matching, cache I/O) — e.g. `test_enrichment_amazon.py` and
   `test_enrichment_target.py` exercise `match_orders_to_transactions`,
   price/date parsers, and cache writers with synthetic data.
4. **Money is `Decimal` constructed from strings** (`Decimal("-12.34")`),
   dates are `datetime.date`. Comparing floats will bite you.
5. New behavior needs a test in the matching `test_<module>.py`; bug fixes
   get a regression test first when practical.
6. Use `tmp_path` for filesystem writes; never write into the repo tree.
