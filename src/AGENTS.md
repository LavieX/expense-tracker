# AGENTS.md — src/expense_tracker

Application code conventions. Read the root `AGENTS.md` first; full design
rationale lives in `docs/architecture.md`.

## Module Map

| Module | Responsibility | Key exports |
|--------|----------------|-------------|
| `models.py` | Dataclasses + ID hashing. **Zero internal imports — keep it that way.** | `Transaction`, `StageResult`, `MerchantRule`, `AppConfig`, `generate_transaction_id` |
| `pipeline.py` | Stage orchestration, pure functions | `run(month, config, categories, rules, root, exclude_patterns)` |
| `parsers/` | Per-bank CSV → `Transaction` | `parse(file_path, institution, account) -> StageResult`, `PARSERS` registry |
| `categorizer.py` | Rule matching + learn workflow | `match_rules`, `categorize`, `learn` |
| `llm.py` | LLM adapters | `ClaudeCodeAdapter` (primary), `AnthropicAdapter`, `NullAdapter` |
| `config.py` | TOML load/save, `expense init` scaffolding | `load_config`, `load_rules`, `save_learned_rules` |
| `recurring.py` | Recurring-merchant auto-detection from historical CSVs | `detect_recurring` |
| `export.py` | Monthly CSV + stdout summary | `export`, `print_summary` |
| `sheets.py` | Google Sheets push (month upsert) | `push_to_sheets` |
| `enrichment/` | Retailer scrapers → cache JSON | `AmazonEnrichmentProvider`, `enrich_target`, `enrich_venmo` |
| `download/` | Playwright bank CSV downloaders | `download_chase`, `download_capital_one`, `download_elevations` |

## Dependency Rules

- `models.py` imports nothing from the package. Everything may import it.
- `config.py`, parsers, `categorizer.py` depend only on `models.py` (+ stdlib/vendored libs).
- The dependency graph must stay acyclic and shallow. `pipeline.py` orchestrates;
  `cli.py` is the only place that wires LLM adapters, export, and sheets together.
- Pipeline stages are pure functions `(list[Transaction], ...) -> StageResult`.
  Errors are collected in `StageResult.errors`/`warnings`, never raised across
  stage boundaries. Partial failure over total failure.

## Adding Things

- **New bank parser:** create `parsers/new_bank.py` with `parse()`, register in
  `parsers/__init__.py`'s `PARSERS` dict, add account entry to config. Nothing
  else changes. Parser owns merchant normalization (stable strings matter —
  learned rules match against them verbatim).
- **New LLM adapter:** implement `categorize_batch(transactions, categories)
  -> list[dict]` (see `LLMAdapter` protocol in `llm.py`), wire selection in
  `cli.py process`.
- **New enrichment source:** module in `enrichment/` writing
  `enrichment-cache/{transaction_id}.json` with an `items` list
  (`merchant`, `description`, `amount`). The pipeline's enrich stage consumes
  the cache — producers never touch the pipeline.

## Gotchas

- **Sign convention:** negative = expense, positive = refund/credit, normalized
  by each parser. Chase/Capital One CSVs already use this; Elevations checking
  needs care.
- **Transaction IDs are deterministic** (`institution|date|merchant|amount|row_ordinal`
  → SHA-256[:12]). Changing any input component breaks dedup against historical
  output — treat the hash recipe as a compatibility boundary.
- **Categorization is AI-primary.** The pipeline applies rules first (stage 5);
  `cli.py` then sends everything still uncategorized to the LLM adapter. Rules
  only carry the full load under `--no-llm`. `match_rules` checks the
  *description* when the merchant match is generic (no subcategory) or absent —
  preserve this, enriched Amazon/Target splits depend on it.
- **`enrichment/target.py` is ~2.6k lines** of selector-engineering against a
  React SPA. Selector lists are ordered most-current-first, legacy kept as
  fallbacks — extend, don't replace.
- **`download/` and `enrichment/` hit real accounts.** Keep Playwright code
  defensive (bot detection, session expiry) and never log credentials.
  `download/base.py` reads credentials from KPX/KeePass only.
- Amounts are `Decimal`; construct from `str`, never `float`.
