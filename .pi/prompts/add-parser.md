---
description: Add a new bank CSV parser to the pipeline
argument-hint: "<bank-name>"
---

Add a parser for a new bank: $1. Follow `docs/architecture.md` Section 5 and
the conventions in `src/AGENTS.md`. Checklist:

1. Ask the user for a sample CSV export from $1 (or the column layout). Do
   NOT read real files from `input/` — have the user paste the header row and
   1-2 anonymized rows, or create a synthetic sample.
2. Create `src/expense_tracker/parsers/$1.py` exposing
   `parse(file_path: Path, institution: str, account: str) -> StageResult`,
   modeled on the existing parsers (chase.py / capital_one.py / elevations.py):
   - Validate expected columns; fail the file with an error if missing.
   - Skip malformed rows with warnings; fail the file if >10% are malformed.
   - Normalize amounts to the sign convention: negative = expense,
     positive = refund/credit. Use `Decimal` parsed from strings.
   - Normalize merchant names (strip bank prefixes, reference numbers) —
     learned rules match these strings verbatim.
   - Generate IDs via `models.generate_transaction_id` with the row ordinal.
   - Return ALL rows; never filter by month.
3. Register the parser in `src/expense_tracker/parsers/__init__.py` (`PARSERS`
   dict + import).
4. Add a `[[accounts]]` entry to `config.toml.example` (NOT the real
   `config.toml`).
5. Add test fixtures to `tests/fixtures/`: `$1_valid.csv`, `$1_malformed.csv`,
   `$1_wrong_format.csv`, `$1_empty.csv` (synthetic data only).
6. Add tests in `tests/test_parsers.py` covering: valid parse, sign
   convention, malformed rows skipped, wrong format fails, empty file,
   deterministic IDs.
7. Update `docs/architecture.md` (module structure + parser list).
8. Verify: `.venv/bin/ruff check src/ tests/`,
   `.venv/bin/ruff format --check src/ tests/`, `.venv/bin/pytest`.
