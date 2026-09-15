# Expense Tracker v3 — Architecture

*Last updated: 2026-07 — rewritten to match the implemented system. The
original MVP design (and its review changelog) is preserved in git history;
ADRs below note where the implementation superseded the original decisions.*

## 1. System Overview

Expense Tracker v3 is a Python CLI that transforms raw bank CSV exports into a
categorized monthly expense report, pushes results to Google Sheets, and
learns from user corrections. Around the core pipeline sit three automation
layers: bank CSV download (Playwright), retailer/Venmo enrichment (Playwright
scrapers writing a local cache), and Google Sheets sync.

The codebase has grown past the original ~1,500-line MVP target to roughly
9,300 lines — the bulk of the growth is browser automation
(`enrichment/target.py` alone is ~2,600 lines of selector engineering) and the
download/enrichment/sheets integrations. The core pipeline modules remain
small and follow the original design principles.

### High-Level Data Flow

```
Bank websites ──(expense download: Playwright)──> input/<account>/*.csv
                                                        |
                                                        v
                                            [ 1. parse ]  per-bank parser -> Transaction
                                                        |
                                            [ 1b. filter to target month ]
                                                        |
                                            [ 1c. exclude ]  salary/income patterns
                                                        |
                                            [ 2. deduplicate ]  deterministic IDs
                                                        |
                                            [ 3. detect_transfers ]  checking debit <-> CC credit
                                                        |
Amazon/Target/Venmo ──(expense enrich)──> enrichment-cache/{txn_id}.json
                                                        |
                                            [ 4. enrich ]  cache lookup, split line items
                                                        |
                                            [ 4b. tag sources ]  Amazon / Target
                                                        |
                                            [ 5. categorize (rules) ]  longest substring match
                                                        |
                                            [ 6. detect recurring ]  history scan
                                                        |
                                            [ 7. categorize (LLM) ]  Claude, batch (CLI layer)
                                                        |
                                            [ 8. export ]  output/YYYY-MM.csv + summary
                                                        |
                                                        v
                                        (expense push) Google Sheets (month upsert)
                                                        |
                              user corrects in Sheets/CSV -> (expense learn) -> rules.toml
```

### Design Principles

1. **Pipeline, not framework.** Each stage is a pure function over a list of
   transactions. No inversion of control, no event bus. Data flows in one
   direction.
2. **Partial failure over total failure.** Every stage processes what it can
   and reports what it could not. The pipeline never halts on a single bad row
   or an unavailable LLM.
3. **Knowledge base is the product.** The TOML rule files are the most
   valuable artifact. They are human-readable, version-controlled, and survive
   refactors.
4. **Flat files are the system of record.** Monthly CSVs are self-contained
   and human-readable. No database required to view your own data.
5. **Real data never enters version control.** `config.toml`, `input/`,
   `output/`, `enrichment-cache/`, and `.auth/` are gitignored. Tests run
   against synthetic fixtures only.

---

## 2. Module Structure

All application code lives under `src/expense_tracker/`.

```
src/expense_tracker/                 (~9,300 lines total)
    __init__.py            # Package root, version
    cli.py                 # Click commands: process, learn, enrich, push, download, init (~850)
    pipeline.py            # Stage orchestration (~610)
    models.py              # Dataclasses + transaction ID hashing (~285)
    config.py              # TOML load/save, init scaffolding (~460)
    categorizer.py         # Rule matching, AI categorize driver, learn workflow (~420)
    llm.py                 # ClaudeCodeAdapter / AnthropicAdapter / NullAdapter (~320)
    recurring.py           # Recurring-merchant auto-detection (~140)
    export.py              # Monthly CSV writer + stdout summary (~240)
    sheets.py              # Google Sheets push with month upsert (~200)
    parsers/
        __init__.py        # Parser registry (PARSERS dict, get_parser)
        chase.py           # Chase credit card CSV parser
        capital_one.py     # Capital One credit card CSV parser
        elevations.py      # Elevations Credit Union checking parser
    enrichment/
        __init__.py        # EnrichmentProvider protocol, registry, result types
        cache.py           # Enrichment cache read/write (JSON)
        amazon.py          # Amazon order history scraper, multi-account (~980)
        target.py          # Target.com order history scraper (~2,650)
        venmo.py           # Venmo statement/feed scraper (~540)
    download/
        __init__.py        # Package docs: session persistence strategy
        base.py            # Shared helpers: KeePass/KPX credential lookup, auth dirs
        chase.py           # Chase CSV downloader (Playwright)
        capital_one.py     # Capital One CSV downloader (Playwright)
        elevations.py      # Elevations CSV downloader (Playwright)
```

### Dependency Rules

- **`models.py` has zero internal imports.** Everything depends on it; it
  depends on nothing.
- `config.py`, parsers, and `categorizer.py` depend only on `models.py`.
- `pipeline.py` imports parsers, models, and recurring. It deliberately does
  **not** import `llm.py` — the LLM tier is driven by the CLI layer (see
  Section 8).
- `cli.py` is the composition root: it loads config, selects the LLM adapter,
  runs the pipeline, invokes LLM categorization, exports, and prints the
  summary.
- The dependency graph is acyclic and shallow by construction.

---

## 3. Pipeline Stages

Each stage receives a list of `Transaction` objects and returns a
`StageResult` containing the (possibly modified) list plus warnings and
errors. `pipeline.run()` composes the stages and accumulates everything into
a `PipelineResult`.

### Stage 1: Parse

Discovers CSV files per account (glob `*.csv`, case-insensitive,
non-recursive, skipping names starting with `.`, `~`, `_`), dispatches to the
registered parser, concatenates results. Parsers validate expected columns
(fail the file if missing), skip malformed rows with warnings, and fail the
whole file if >10% of rows are malformed. Parsers return **all rows** — they
do not filter by month.

### Stage 1b: Filter to Target Month

The pipeline (not parsers) owns the date boundary: keeps transactions whose
`date` falls within the `--month` argument.

### Stage 1c: Exclude

Removes transactions whose merchant matches any `[exclude].patterns` entry in
`rules.toml` (case-insensitive substring). Used for salary, income, and
internal noise that should never reach reports. Exclusions are reported as a
warning count.

### Stage 2: Deduplicate

By `transaction_id`; first occurrence wins; count reported.

### Stage 3: Detect Transfers

Checking-account debits matching `transfer_detection.keywords` (merchant or
description, case-insensitive) are paired with credit-card credits of the
same absolute amount within `date_window_days` (default 5). Both sides get
`is_transfer=True`. Transfers stay in the list for auditability and are
filtered at export.

### Stage 4: Enrich

Pure cache lookup — **this stage never fetches**. For each transaction it
looks for `enrichment-cache/{transaction_id}.json` (written out-of-band by
`expense enrich`). If found, the transaction is replaced by split line items:

- Split ID: `{parent_id}-{n}` (1-indexed), `split_from` = parent ID
- Merchant: the retailer tag (e.g. `Amazon`, `Target`) — the product name
  goes in `description` so rules can match product-specific text
- Validation: split amounts must sum to the original within $0.01, else the
  original is kept with a warning

### Stage 4b: Tag Sources

Transactions not tagged by enrichment get `source = "Amazon"` / `"Target"`
from merchant-name pattern matching (`AMAZON`/`AMZN`/`AMZ`, `TARGET`).
`source` flows into the output CSV and the LLM prompt.

### Stage 5: Categorize (Rules)

Applies `categorizer.match_rules` to every uncategorized transaction (see
Section 8). Rule matches also set `is_recurring` when the matched rule
declares it.

### Stage 6: Detect Recurring

`recurring.detect_recurring` scans historical `output/*.csv` files for
merchants appearing in 3+ distinct months with amounts within 20% variance
(median-based). Matching transactions are auto-flagged `is_recurring=True`
unless a rule already set recurring status — **explicit rules always win**.
Failure of this stage degrades to a warning, never an error.

### Stage 7: Categorize (LLM) — CLI Layer

Not part of `pipeline.run()`. After the pipeline returns, `cli.py` calls
`categorizer.categorize()` with the selected LLM adapter, which sends **all**
still-uncategorized transactions to the LLM in batches (Section 8).

### Stage 8: Export

Filters out transfers, sorts by (date, institution, amount), writes
`output/YYYY-MM.csv` (overwrites), prints the summary: per-institution
counts, transfer count, enrichment stats, categorization rate, top
uncategorized merchants, spending by category, and accumulated
warnings/errors.

---

## 4. Data Model

Defined in `models.py`. All structures are `@dataclass` classes.

### Transaction

```python
@dataclass
class Transaction:
    transaction_id: str          # Deterministic hash, 12 hex chars
    date: date                   # Transaction date (not post date)
    merchant: str                # Normalized merchant/payee name
    description: str             # Original description from bank CSV
    amount: Decimal              # Negative = expense, positive = refund
    institution: str             # "chase", "capital_one", "elevations"
    account: str                 # Account display name from config
    category: str = "Uncategorized"
    subcategory: str = ""
    is_transfer: bool = False
    is_return: bool = False      # True if amount > 0
    is_recurring: bool = False   # Subscription/bill flag (rule or auto-detect)
    split_from: str = ""         # Parent transaction_id for split line items
    source: str = ""             # Retailer tag: "Amazon", "Target", or ""
    source_file: str = ""        # Debug only; never exported
```

### Transaction ID Generation

```python
raw = f"{institution}|{date.isoformat()}|{merchant.strip().upper()}|{amount}|{row_ordinal}"
id = hashlib.sha256(raw.encode()).hexdigest()[:12]
```

Deterministic; stable across re-runs and overlapping downloads. Row ordinal
(0-based within the source CSV) disambiguates identical same-day purchases.
**Compatibility boundary:** changing any component invalidates IDs in
historical output and enrichment caches.

### Output CSV Schema

Fixed column order (`export.CSV_COLUMNS`):

```
transaction_id, date, month, merchant, description, amount, institution,
account, category, subcategory, is_return, is_recurring, split_from, source
```

`month` (YYYY-MM) denormalizes the date for Sheets pivoting and drives the
Sheets month-upsert. `is_transfer` and `source_file` are intentionally
excluded. The Google Sheets push uses the same column order.

### Other Models

```python
@dataclass
class StageResult:                       # Universal stage return type
    transactions: list[Transaction]
    warnings: list[str]
    errors: list[str]

@dataclass
class MerchantRule:
    pattern: str                         # Case-insensitive substring
    category: str
    subcategory: str = ""
    recurring: bool = False              # Marks merchant as recurring charge
    source: str = "user"                 # "user" | "learned"

@dataclass
class LearnResult:
    added: int; updated: int; skipped: int
    rules: list[MerchantRule]

@dataclass
class AccountConfig:
    name: str; institution: str; parser: str
    account_type: str                    # "credit_card" | "checking"
    input_dir: str

@dataclass
class AmazonAccountConfig:
    label: str = "default"               # Per-account browser session label

@dataclass
class SheetsConfig:
    credentials_file: str                # Service account JSON path
    spreadsheet_id: str
    worksheet_name: str = "Raw Data"

@dataclass
class AppConfig:
    accounts: list[AccountConfig]
    output_dir: str = "output"
    enrichment_cache_dir: str = "enrichment-cache"
    transfer_keywords: list[str]         # ["PAYMENT", "AUTOPAY", ...]
    transfer_date_window: int = 5
    llm_provider: str = "anthropic"      # "anthropic" | "none" | other -> Claude Code
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"
    amazon_accounts: list[AmazonAccountConfig]
    sheets: SheetsConfig | None

@dataclass
class PipelineResult:
    transactions: list[Transaction]
    warnings: list[str]
    errors: list[str]
```

---

## 5. Plugin Architecture

### Parsers

Each parser module exposes:

```python
def parse(file_path: Path, institution: str, account: str) -> StageResult: ...
```

`parsers/__init__.py` maps names to functions in the `PARSERS` dict; account
config references the name. **Adding a bank:** create the module, register in
`PARSERS`, add a `[[accounts]]` entry — no other code changes. Parsers own
merchant normalization (stripping bank prefixes, trailing reference numbers):
learned rules match merchant strings verbatim, so stable normalized strings
are a correctness requirement.

Sign convention is normalized by parsers: **negative = expense, positive =
refund/credit**.

### LLM Adapters

Protocol (`llm.py`):

```python
class LLMAdapter(Protocol):
    def categorize_batch(
        self,
        transactions: list[dict],   # {id, merchant, description, amount, date, source}
        categories: list[dict],      # {name, subcategories}
    ) -> list[dict]:                 # [{id, category, subcategory}] (merchant fallback ok)
        ...
```

Three implementations:

| Adapter | Mechanism | Use |
|---------|-----------|-----|
| `ClaudeCodeAdapter` | Shells out to `claude --print --model sonnet --max-turns 3` | **Primary/default** — uses the user's Claude Max subscription, no API key |
| `AnthropicAdapter` | Direct `httpx` POST to the Messages API | Fallback for headless/CI; requires `ANTHROPIC_API_KEY` credits |
| `NullAdapter` | Returns `[]` | `--no-llm` mode |

All adapters share `_build_prompt` (household context + taxonomy + batch) and
`_parse_response` (extracts and validates the JSON array). Batches of 80
transactions; 120s subprocess timeout. Any failure returns an empty list —
the categorizer treats this as "LLM unavailable" and leaves transactions
uncategorized with a warning.

### Enrichment Providers

Protocol (`enrichment/__init__.py`):

```python
class EnrichmentSource(Protocol):
    def fetch(self, transactions: list[Transaction]) -> dict[str, list[dict]]:
        """Returns {transaction_id: [{item_name, amount, category_hint}, ...]}"""
```

Providers (Amazon, Target, Venmo) scrape order history via Playwright, match
orders to bank transactions (date proximity + amount matching, with tolerance
for e.g. Target RedCard 5% discounts), and write cache files. The pipeline's
enrich stage is a pure consumer of the cache.

### Cache Format

`enrichment-cache/{transaction_id}.json`:

```json
{
  "transaction_id": "abc123...",
  "source": "amazon",
  "order_id": "...",
  "matched_at": "...",
  "retailer": "Amazon",
  "items": [
    {"merchant": "...", "description": "...", "amount": "-30.00"}
  ]
}
```

The pipeline reads only `items` (+ `retailer` for source tagging); extra
metadata keys are ignored, keeping the format backward-compatible.

### Bank Downloaders

`download/` modules automate bank CSV export via Playwright with **persistent
browser sessions** in `.auth/<bank>/state.json`:

1. First run: visible browser, user completes CAPTCHA/MFA manually, session
   saved.
2. Subsequent runs: saved session reused (headless possible); expiry falls
   back to interactive login.

Credentials come from KeePass — `download/base.py` tries a running **KPX**
credential server first, then direct `pykeepass` (path + master password via
`--keepass-file`/`KEEPASS_FILE`, `--keepass-password`/`KEEPASS_PASSWORD`).
Credentials are never logged or stored by the app.

---

## 6. CLI Design

Framework: **Click**. Entry point `expense` (`pyproject.toml [project.scripts]`).

```
expense process  --month YYYY-MM [--no-llm] [--verbose] [--debug]
expense learn    --original PATH --corrected PATH [--verbose]
expense enrich   --month YYYY-MM --source amazon|target|venmo [--headless] [--verbose] [--debug]
expense push     [--month YYYY-MM | --all] [--verbose]
expense download --month YYYY-MM [--source chase|capital-one|elevations|all]
                 [--auth BANK] [--headless] [--keepass-file PATH] [--keepass-password PW]
expense init     [--dir PATH]
```

### `expense process`

The primary command:

1. Load `config.toml`, `categories.toml`, `rules.toml` (+ exclude patterns).
2. Select LLM adapter: `--no-llm`/`provider="none"` → NullAdapter;
   `provider="anthropic"` → AnthropicAdapter; **anything else →
   ClaudeCodeAdapter (default)**.
3. `pipeline.run(...)` → stages 1–6.
4. `categorizer.categorize(...)` with the LLM adapter → stage 7.
5. `export(...)` + `print_summary(...)`.

Month validation is strict (`YYYY-MM`, 01–12). Errors print to stderr with
exit code 1; LLM failures degrade to warnings.

### `expense learn`

Compares the pipeline's original output CSV with a user-corrected copy,
indexed by `transaction_id`. For every changed category/subcategory:

- User rule already covers the merchant → **skip** (never overwrite user rules)
- Learned rule exists for the exact merchant pattern → update in place
- Otherwise → append new learned rule

Writes only the `[learned_rules]` section of `rules.toml`; everything above
it (comments, `[exclude]`, `[user_rules]`) is preserved verbatim. Prints
added/updated/skipped counts.

### `expense enrich`

Runs the full pipeline parse path to build the transaction list, then invokes
the selected provider:

- **amazon**: multi-account (`[[enrichment.amazon]]` labels, separate browser
  sessions per account); per-account stats in the summary
- **target**: Playwright scrape of target.com order history; matches orders
  to transactions by date proximity (±3 days) and amount (with RedCard
  tolerance); skips gift-card-only orders
- **venmo**: scrapes Venmo statements/feed, matches to bank transactions,
  writes cache files

### `expense push`

Pushes processed data to Google Sheets (service account auth):

- `--month YYYY-MM`: **upsert** — reads the sheet, drops rows whose `month`
  column matches, appends fresh rows, re-sorts by date, rewrites. Other
  months untouched.
- `--all` (or no flag): clears the sheet and rewrites every month CSV found
  in `output/` (files matching `YYYY-MM.csv` only).

### `expense download`

Per-bank Playwright downloaders writing into each account's `input_dir`.
`--auth BANK` runs an interactive login-only pass to (re)save the session
without downloading.

### `expense init`

Idempotent scaffolding: creates `input/{chase,capital-one,elevations}/`,
`output/`, `enrichment-cache/`, and default `config.toml`, `categories.toml`,
`rules.toml`. Never overwrites existing files.

---

## 7. Configuration

Three TOML files in the project root (`config.toml` is gitignored — see
`config.toml.example` for a shareable template).

### config.toml

```toml
[general]
output_dir = "output"
enrichment_cache_dir = "enrichment-cache"

[transfer_detection]
keywords = ["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"]
date_window_days = 5

[llm]
provider = "anthropic"          # "anthropic" (API) | "none" | anything else -> Claude Code subprocess
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[[accounts]]
name = "Chase Credit Card"
institution = "chase"
parser = "chase"
account_type = "credit_card"    # or "checking"
input_dir = "input/chase"
# ... one per account

# Optional: multiple Amazon accounts, each with its own browser session
# [[enrichment.amazon]]
# label = "primary"

# Optional: Google Sheets push
# [sheets]
# credentials_file = ".auth/google-service-account.json"
# spreadsheet_id = "..."
# worksheet_name = "Raw Data"
```

### categories.toml

18 top-level categories, each with a `subcategories` list (possibly empty):
Housing, Utilities, Food & Dining, Transportation, Kids, Health & Fitness,
Healthcare, Entertainment, Shopping, Home & Garden, Personal Care, Pets,
Gifts & Charity, Travel, Education, Insurance, Business, Miscellaneous.

### rules.toml

```toml
[exclude]
patterns = ["PAYROLL", ...]      # removed in pipeline stage 1c

[user_rules]                     # hand-authored; the system never modifies these
"KING SOOPERS" = "Food & Dining:Groceries"
"NETFLIX" = { category = "Entertainment", subcategory = "Subscriptions", recurring = true }

[learned_rules]                  # written only by `expense learn`; do not hand-edit
```

Rule values are either `"Category"` / `"Category:Subcategory"` strings or an
inline dict with `category`, `subcategory`, `recurring` keys.

`config.save_learned_rules()` rewrites only the `[learned_rules]` section,
preserving everything before it byte-for-byte.

---

## 8. Categorization Engine

### Flow (as implemented)

```
pipeline.run():  stage 5 applies rules to all uncategorized transactions
       |
cli.py:          categorizer.categorize() with selected LLM adapter
       |
       +-- LLM adapter present: send ALL remaining uncategorized
       |   transactions to the LLM (batches of 80)
       |   -> apply suggestions by transaction ID (merchant name fallback)
       |
       +-- No adapter (--no-llm): rules are the only engine;
           unmatched stay "Uncategorized" with a warning
```

This is **AI-primary** categorization: rules act as a fast first pass and a
constraint-free fallback, but the LLM is expected to categorize the bulk of
long-tail merchants. LLM suggestions are applied to the output but are NOT
persisted to `rules.toml` — `expense learn` is the only path to new rules.

### Rule Matching

```python
match_rules(merchant, rules, description="")
```

1. Case-insensitive substring match against merchant; **longest pattern
   wins**; ties break by list order (user rules before learned).
2. If the best merchant match is *generic* (no subcategory — e.g. bare
   "Shopping" for Amazon) and a description exists, try matching the
   **description**; a specific description match (with subcategory) beats the
   generic merchant match.
3. If no merchant match at all, description matching is the final fallback.

Steps 2–3 exist for enriched splits: after enrichment the merchant is the
retailer ("Amazon") and the product name lives in the description, so
product-specific rules ("DOG FOOD" → Pets:Food) must be reachable.

### LLM Prompt

Single prompt per batch containing: household context (location, family,
pets, known local merchants — see `llm.HOUSEHOLD_CONTEXT`), categorization
rules (refunds match original category, sales-tax lines match their split,
etc.), the full taxonomy, and the transaction lines
`ID | Merchant | Description | Amount | Date [| source]`. Response: strict
JSON array of `{id, category, subcategory}`.

### Learn Workflow

`categorizer.learn(original_path, corrected_path, rules) -> LearnResult`.
Rule patterns are the transaction's `merchant` value **verbatim** — parsers
own normalization, so cross-bank variants of the same merchant produce
separate rules (longest-match-wins tolerates this).

---

## 9. Recurring Detection

`recurring.detect_recurring(transactions, output_dir)`:

1. Loads all historical `output/*.csv` files.
2. Groups by uppercased merchant; needs **3+ distinct months**.
3. Compares per-month median amounts; all within **20% variance** → recurring.
4. Pipeline stage 6 flags matching transactions `is_recurring=True`, but
   explicit rule-based recurring flags (either direction) take precedence.

The flag flows to the CSV (`is_recurring` column) and Sheets, enabling
subscription/bill rollups in pivot tables.

---

## 10. Error Handling

**Partial failure over total failure.** Errors never propagate as exceptions
across stage boundaries; every stage returns `StageResult` and the pipeline
accumulates warnings/errors for the final summary.

| Stage | Error case | Behavior |
|-------|-----------|----------|
| Parse | File unreadable / wrong columns | Skip file, add error |
| Parse | Malformed row | Skip row, add warning |
| Parse | >10% malformed rows | Fail the file (likely format change) |
| Exclude | (none) | Reports exclusion count as warning |
| Dedup | (no failure modes) | Reports count |
| Transfers | No pair found | Not an error |
| Enrich | No cache file | Pass through unchanged |
| Enrich | Splits don't sum to original (±$0.01) | Keep original, warn |
| Categorize | No rule match | Falls through to LLM tier |
| Categorize | LLM unavailable / unparseable | Leave uncategorized, warn |
| Recurring | Any exception | Warn, continue unflagged |
| Export | Output dir not writable | Fatal (cannot produce output) |
| Push | Missing credentials / API failure | Fatal for that command only |

Browser automation (download/enrich) additionally treats bot detection,
session expiry, and selector drift as routine: interactive fallback for auth,
debug HTML dumps for failed scrapes.

---

## 11. Key ADRs

### ADR-1: Flat Files Over SQLite

Monthly CSVs and TOML config, no database. Primary consumption is Google
Sheets with pivot tables; CSVs import directly and stay human-inspectable.
SQLite could later be added as a read-only index — not needed so far.

### ADR-2: Deterministic Transaction IDs via Hash

SHA-256 of `institution|date|merchant_upper|amount|row_ordinal`, truncated to
12 hex chars. Banks provide no unique IDs; the deterministic hash enables
dedup across re-runs and overlapping downloads. Proven across v2 (two years,
three banks).

### ADR-3: Substring Matching, Not Regex

Case-insensitive substring with longest-match-wins. Handles 95%+ of merchant
patterns with zero syntax overhead. Regex could be added later as an opt-in
per-rule field.

### ADR-4: Enrichment as Separate Pre-Processing

`expense enrich` scrapes and writes the cache; the pipeline only reads the
cache. Scrapers are slow and flaky; the fast path (`expense process` without
enrichment) must stay fast and reliable.

### ADR-5: LLM Suggestions Auto-Applied, Persisted Only After Learn

LLM output lands in the CSV immediately; `rules.toml` only grows via `learn`
(user-confirmed corrections). Prevents bad suggestions from polluting the
knowledge base.

### ADR-6: Click for CLI

Explicit decorators, mature ecosystem. (Typer wraps Click anyway.)

### ADR-7: Raw HTTP / Subprocess for LLM, No Framework

Direct Anthropic Messages API via `httpx`, or a `claude` CLI subprocess. No
LangChain/LiteLLM — a single prompt-response pattern doesn't justify hundreds
of transitive dependencies.

### ADR-8: Three TOML Config Files

`config.toml` (settings), `categories.toml` (taxonomy, rarely changes),
`rules.toml` (grows constantly via `learn`). Separation minimizes merge
conflicts and keeps `learn`'s blast radius to one section of one file.

### ADR-9: AI-Primary Categorization via Claude Code Subprocess

*Supersedes the MVP's "rules first, LLM as fallback" ordering.* The default
categorizer is `ClaudeCodeAdapter`, which invokes the `claude` CLI
(`--print --model sonnet --max-turns 3`) as a subprocess. **Rationale:** the
user's Claude Max subscription makes per-month categorization effectively
free versus metered API credits, and LLM accuracy on long-tail merchants
exceeds a hand-maintained rule base. Rules still run first (cheap, exact,
needed for `--no-llm`), and the Anthropic API adapter remains for headless/CI
use. Household context in the prompt encodes local knowledge (merchant
aliases, family specifics) that would otherwise require hundreds of rules.

### ADR-10: Playwright + Persistent Sessions for Bank/Retailer Automation

Banks and Target employ bot detection (Cloudflare Turnstile, device
fingerprinting, MFA). Fully headless first-login is not feasible. Decision:
interactive first run with session state persisted under `.auth/<bank>/`,
then reuse (headless-capable) with interactive fallback on expiry. Selector
strategy for React SPAs: comma-ordered selector lists, most-current-first,
legacy kept as fallbacks; debug HTML dumps on failure. Credentials come from
KeePass (KPX server preferred, `pykeepass` fallback) — never from config
files or env vars in plaintext.

### ADR-11: Google Sheets as the Analysis Layer

The CSV is the system of record; Google Sheets is the consumption/analysis
layer (pivot tables, charts). `expense push` supports month-level upsert so
re-processing a month never disturbs other months' data. A `month` column was
added to the export schema to make upserts and pivots trivial.

### ADR-12: Description-Based Rule Matching for Enriched Splits

Enriched splits normalize `merchant` to the retailer and put the product in
`description`. Categorization therefore matches rules against descriptions
when the merchant match is generic or absent. **Rationale:** "Amazon" alone
is unclassifiable; the product text is the signal. This keeps one generic
retailer rule from masking product-specific rules.

---

## Appendix: Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `click` | CLI framework |
| `httpx` | Anthropic API calls |
| `tomli-w` | Writing learned rules (`tomllib` is read-only) |
| `gspread`, `google-auth` | Google Sheets push |
| `playwright` | Bank download + retailer enrichment automation |
| `pykeepass` / `kpx` (optional) | Credential lookup for browser automation |

`tomllib`, `csv`, `hashlib`, `json`, `subprocess` from stdlib.

### Dev

| Package | Purpose |
|---------|---------|
| `pytest`, `pytest-cov` | Testing |
| `ruff` | Lint + format (line-length 100; rules E, F, I, UP, B, SIM) |

### CI

GitHub Actions (`.github/workflows/ci.yml`): Python 3.12, `pip install -e
".[dev]"`, `ruff check src/ tests/`, `pytest`.

---

## Changelog

### 2026-07 — Full rewrite to match implemented system

The original MVP architecture (authored 2026-02) is superseded by this
document. Major deltas:

- **AI-primary categorization** via Claude Code subprocess (ADR-9); rules run
  first in the pipeline, LLM tier moved to the CLI layer
- **New modules:** `recurring.py` (auto recurring detection), `sheets.py`
  (Google Sheets push with month upsert), `enrichment/` (Amazon multi-account,
  Target, Venmo providers + cache), `download/` (Playwright bank downloaders,
  KeePass credentials)
- **Pipeline stages added:** exclude (1c), source tagging (4b), recurring (6)
- **Model additions:** `is_recurring`, `source` on Transaction; `recurring`
  on MerchantRule; `AmazonAccountConfig`, `SheetsConfig`; `month`,
  `is_recurring`, `source` columns in the export schema
- **New CLI commands:** `push`, `download`; `enrich` implemented with three
  providers
- **rules.toml gained `[exclude]`** and the inline-dict rule format with
  `recurring`

### 2026-02-14 — Address architecture review findings (M1–M5)

(Historical, from the MVP doc.) Clarified enrichment trigger semantics (cache
lookup only), merchant pattern extraction in learn (verbatim from `merchant`
field), month filtering ownership (pipeline, not parsers), `StageResult` as
the universal stage return type, and CSV file discovery rules.
