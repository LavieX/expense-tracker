# Expense Tracker v3 -- Architecture

## 1. System Overview

Expense Tracker v3 is a Python CLI tool that transforms raw bank CSV exports into a categorized monthly expense report. It replaces v2's 8-10 command workflow with a single-command pipeline, maintains a learnable knowledge base of merchant-to-category mappings, and keeps the codebase under ~1,500 lines.

### High-Level Data Flow

```
Bank CSVs (Chase, Capital One, Elevations)
    |
    v
[ parse ] -- per-bank parser normalizes to Transaction
    |
    v
[ deduplicate ] -- deterministic transaction IDs, drop duplicates
    |
    v
[ detect_transfers ] -- match checking debits to CC credits, flag pairs
    |
    v
[ enrich ] -- look up cached enrichment data, split if available
    |
    v
[ categorize ] -- tier 1: rule matching, tier 2: LLM fallback
    |
    v
[ export ] -- write monthly CSV, print summary
```

### Design Principles

1. **Pipeline, not framework.** Each stage is a pure function over a list of transactions. No inversion of control, no event bus. Data flows in one direction.
2. **Partial failure over total failure.** Every stage processes what it can and reports what it could not. The pipeline never halts on a single bad row or an unavailable LLM.
3. **Knowledge base is the product.** The TOML rule files are the most valuable artifact. They are human-readable, version-controlled, and survive refactors.
4. **Flat files are the system of record.** Monthly CSVs are self-contained and human-readable. No database required to view your own data.

---

## 2. Module Structure

All application code lives under `src/expense_tracker/`. Target: ~1,500 lines total, ~300 lines per module.

```
src/expense_tracker/
    __init__.py          # Package root, version
    cli.py               # Click command definitions (~200 lines)
    pipeline.py          # Orchestrates the processing pipeline (~150 lines)
    models.py            # Data classes: Transaction, Rule, Config (~200 lines)
    parsers/
        __init__.py      # Parser registry and base protocol (~50 lines)
        chase.py         # Chase credit card CSV parser (~100 lines)
        capital_one.py   # Capital One credit card CSV parser (~100 lines)
        elevations.py    # Elevations Credit Union CSV parser (~100 lines)
    categorizer.py       # Rule matching + LLM fallback (~250 lines)
    llm.py               # LLM adapter interface + Anthropic impl (~150 lines)
    config.py            # TOML loading/writing, path resolution (~150 lines)
    export.py            # CSV output writer + summary printer (~100 lines)
```

Estimated total: ~1,550 lines (within tolerance of the ~1,500 target).

### Dependency Graph

```
cli.py
  |
  v
pipeline.py
  |---> parsers/ (parse stage)
  |---> models.py (Transaction, used everywhere)
  |---> categorizer.py (categorize stage)
  |       |---> llm.py (tier 2 fallback)
  |       |---> config.py (loads rules.toml)
  |---> export.py (output stage)
  |---> config.py (loads config.toml, categories.toml)
```

Key rule: `models.py` has zero internal imports. Everything depends on it; it depends on nothing. `config.py` depends only on `models.py`. Parsers depend only on `models.py`. This keeps the dependency graph acyclic and shallow.

---

## 3. Data Flow

Each pipeline stage receives a list of `Transaction` objects and returns a `StageResult` (see Section 9) containing the (possibly modified) list plus any warnings and errors from that stage. Stages are composed in `pipeline.py`, which accumulates warnings and errors across all stages for the final summary.

### Stage 1: Parse

**Input:** File paths to bank CSV files, grouped by account config.
**Output:** `list[Transaction]` with all fields populated except `category` and `subcategory`.

Each parser implements the `Parser` protocol (see Section 5). The parser registry in `parsers/__init__.py` maps parser names from config to parser functions. Each parser:

1. Opens the CSV with `csv.DictReader`.
2. Validates that expected columns exist (fail the file if not).
3. Iterates rows, skipping malformed ones (with warnings). If >10% of rows are malformed, fails the entire file.
4. For each valid row: extracts date, merchant, amount (applying sign conventions), and description. Generates the deterministic transaction ID.
5. Returns a list of `Transaction` objects for **all rows** in the file (no date filtering).

Transactions from all accounts are concatenated into a single list. The pipeline then filters this combined list to only transactions whose `date` falls within the target month. Parsers do not receive or apply month filtering -- they return everything, and the pipeline owns the date boundary logic. This keeps parsers simple and stateless.

### Stage 2: Deduplicate

**Input:** `list[Transaction]` (possibly containing duplicates from overlapping CSV downloads or re-runs).
**Output:** `list[Transaction]` with duplicates removed.

Deduplication is by `transaction_id`. When duplicates are found, keep the first occurrence. Log the count of duplicates removed.

### Stage 3: Detect Transfers

**Input:** `list[Transaction]`.
**Output:** `list[Transaction]` with `is_transfer=True` on matched pairs.

Algorithm:
1. Collect all checking account debits that match transfer keywords (`PAYMENT`, `AUTOPAY`, `ONLINE PAYMENT`, etc., configurable in `config.toml`).
2. For each, search for a matching credit card credit within the configurable date window (default: 5 days) with the same absolute amount.
3. Mark both sides of the match as `is_transfer=True`.

Transfers remain in the transaction list (for auditability) but are excluded from the output CSV by the export stage.

### Stage 4: Enrich

**Input:** `list[Transaction]`.
**Output:** `list[Transaction]`, possibly expanded (one transaction may become multiple split line items).

1. For each transaction, look up its `transaction_id` in the enrichment cache directory (`enrichment-cache/`).
2. If cached enrichment data exists (a JSON file keyed by transaction ID), replace the transaction with split line items. Each split gets:
   - ID: `{parent_id}-{n}` (1-indexed numeric suffix)
   - `split_from`: the parent transaction ID
   - Item-level merchant/description from enrichment data
   - Amount from enrichment item prices
3. Validate: split amounts must sum to the original transaction amount (within $0.01 tolerance for rounding). If validation fails, keep the original unsplit transaction and warn.
4. If no enrichment data exists, pass the transaction through unchanged.

This stage is a consumer of enrichment data, not a producer. Production of enrichment data is handled by the separate `expense enrich` command (Phase 2). Note: when the requirements describe transactions as "eligible for retailer enrichment," this means the transaction has cached enrichment data on disk -- it does not mean the pipeline triggers a fetch. The `enrich` stage is a pure cache lookup; all data acquisition happens out-of-band via `expense enrich`.

### Stage 5: Categorize

**Input:** `list[Transaction]` (uncategorized or partially categorized).
**Output:** `list[Transaction]` with `category` and `subcategory` populated where possible.

Two-tier system (see Section 8 for full detail):
1. **Tier 1 -- Rule matching:** Apply merchant-to-category rules from `rules.toml`. Substring match, case-insensitive, longest match wins. User rules checked before learned rules.
2. **Tier 2 -- LLM fallback:** Batch all still-uncategorized transactions into a single LLM call. Apply suggestions to transactions. Do NOT write suggestions to rules.toml (that happens via `learn`).
3. Transactions that remain uncategorized (no rule match, LLM unavailable or returned nothing) get `category="Uncategorized"`.

### Stage 6: Export

**Input:** `list[Transaction]` (fully processed).
**Output:** Monthly CSV file written to `output/YYYY-MM.csv`. Summary printed to stdout.

1. Filter out transactions where `is_transfer=True`.
2. Sort by date, then by institution, then by amount.
3. Write CSV with the fixed column schema (see Section 4). The output is designed to be consumed in Google Sheets via import, where the user builds pivot tables for analysis. The column schema, sort order, and well-structured tabular format are optimized for this workflow.
4. Print processing summary: total transactions, total spending, categorization rate, top uncategorized merchants, spending by top-level category.

---

## 4. Data Model

Defined in `models.py`. All data structures are `@dataclass` classes.

### Transaction

The core data object that flows through every pipeline stage.

```python
@dataclass
class Transaction:
    transaction_id: str          # Deterministic hash, 12-16 hex chars
    date: date                   # Transaction date (not post date)
    merchant: str                # Normalized merchant/payee name
    description: str             # Original description from bank CSV
    amount: Decimal              # Negative = expense, positive = refund
    institution: str             # e.g., "chase", "capital_one", "elevations"
    account: str                 # Account identifier from config
    category: str                # Top-level category or "Uncategorized"
    subcategory: str             # Subcategory or empty string
    is_transfer: bool            # True if detected as a transfer
    is_return: bool              # True if amount > 0
    split_from: str              # Parent transaction_id, or empty string
    source_file: str             # Path to the source CSV (for debugging)
```

### Transaction ID Generation

```python
import hashlib

def generate_transaction_id(
    institution: str,
    txn_date: date,
    merchant: str,
    amount: Decimal,
    row_ordinal: int,
) -> str:
    """Deterministic transaction ID from the components that define uniqueness."""
    raw = f"{institution}|{txn_date.isoformat()}|{merchant.strip().upper()}|{amount}|{row_ordinal}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
```

Components: institution name (lowercase), ISO date, uppercased/stripped merchant string, amount as string, 0-based row ordinal within the source CSV file. SHA-256, truncated to 12 hex characters.

### Output CSV Column Schema

Fixed column order, written by `export.py`:

```
transaction_id, date, merchant, description, amount, institution, account,
category, subcategory, is_return, split_from
```

`is_transfer` is not in the output (transfers are filtered out). `source_file` is not in the output (internal debugging field).

### MerchantRule

```python
@dataclass
class MerchantRule:
    pattern: str                 # Substring to match (case-insensitive)
    category: str                # Target category
    subcategory: str             # Target subcategory, or empty string
    source: str                  # "user" or "learned"
```

### AccountConfig

```python
@dataclass
class AccountConfig:
    name: str                    # Display name, e.g., "Chase Credit Card"
    institution: str             # Internal key, e.g., "chase"
    parser: str                  # Parser name, e.g., "chase"
    account_type: str            # "credit_card" or "checking"
    input_dir: str               # Relative path, e.g., "input/chase"
```

### AppConfig

```python
@dataclass
class AppConfig:
    accounts: list[AccountConfig]
    output_dir: str              # Default: "output"
    enrichment_cache_dir: str    # Default: "enrichment-cache"
    transfer_keywords: list[str] # Default: ["PAYMENT", "AUTOPAY", ...]
    transfer_date_window: int    # Default: 5 (days)
    llm_provider: str            # "anthropic" or "none"
    llm_model: str               # e.g., "claude-sonnet-4-20250514"
    llm_api_key_env: str         # Env var name, e.g., "ANTHROPIC_API_KEY"
```

---

## 5. Plugin Architecture

### Parser Protocol

Each bank parser is a module in `src/expense_tracker/parsers/` that exposes a single function:

```python
from expense_tracker.models import Transaction, StageResult
from decimal import Decimal
from datetime import date
from pathlib import Path

def parse(file_path: Path, institution: str, account: str) -> StageResult:
    """Parse a bank CSV file and return normalized Transaction objects.

    Returns a StageResult. Skipped rows are reported as warnings.
    If the file cannot be parsed at all (wrong format, >10% bad rows),
    returns an empty transaction list with the error in StageResult.errors.
    """
    ...
```

### Parser Registry

`parsers/__init__.py` maintains a simple dict mapping parser names to parse functions:

```python
from expense_tracker.parsers import chase, capital_one, elevations

PARSERS: dict[str, Callable] = {
    "chase": chase.parse,
    "capital_one": capital_one.parse,
    "elevations": elevations.parse,
}

def get_parser(name: str) -> Callable:
    """Look up a parser by name. Raises KeyError if not found."""
    return PARSERS[name]
```

### Adding a New Parser

1. Create `src/expense_tracker/parsers/new_bank.py` implementing the `parse()` function.
2. Import it in `parsers/__init__.py` and add it to the `PARSERS` dict.
3. Add an account entry in `config.toml` referencing the parser name.

No changes to `pipeline.py`, `categorizer.py`, or any other module. The `PARSERS` dict is the registration mechanism -- simple, explicit, no magic.

### LLM Adapter Protocol

Defined in `llm.py`:

```python
from typing import Protocol

class LLMAdapter(Protocol):
    def categorize_batch(
        self,
        transactions: list[dict],    # [{merchant, description, amount, date}, ...]
        categories: list[dict],       # [{name, subcategories: [...]}, ...]
    ) -> list[dict]:                  # [{merchant, category, subcategory}, ...]
        """Send a batch of transactions to the LLM for categorization."""
        ...
```

The Anthropic implementation (`AnthropicAdapter`) in the same file:
- Reads the API key from the environment variable specified in config.
- Constructs a single prompt containing all uncategorized transactions and the category taxonomy.
- Sends one HTTP POST to the Anthropic Messages API via `httpx`.
- Parses the structured response (JSON in the LLM output).
- Returns a list of category suggestions.

If the API call fails (network error, auth error, rate limit), it returns an empty list. The categorizer treats this as "LLM unavailable" and leaves those transactions uncategorized.

### Enrichment Source Interface (Phase 2)

Not built in MVP, but the interface is defined for forward compatibility:

```python
class EnrichmentSource(Protocol):
    def fetch(self, transactions: list[Transaction]) -> dict[str, list[dict]]:
        """Fetch enrichment data for matching transactions.

        Returns: {transaction_id: [{item_name, amount, category_hint}, ...]}
        """
        ...
```

Each enrichment source is a separate module. Enrichment data is written to `enrichment-cache/{transaction_id}.json` by the `expense enrich` command and read by the pipeline's enrich stage.

---

## 6. CLI Design

CLI framework: **Click**. Click is chosen over Typer for its mature ecosystem, explicit decorator-based command definitions, and broader compatibility. The CLI entry point is `expense` (defined in `pyproject.toml` as `project.scripts`).

### Commands

```
expense process --month YYYY-MM [--verbose] [--debug]
expense learn --original PATH --corrected PATH [--verbose]
expense enrich --month YYYY-MM --source NAME [--verbose]
expense init [--dir PATH]
```

#### `expense process`

The primary command. Runs the full pipeline for a given month.

```
expense process --month 2026-01
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--month` | `YYYY-MM` | Required | Target month to process |
| `--no-llm` | flag | False | Skip LLM categorization (cache-only) |
| `--verbose` | flag | False | Detailed progress output |
| `--debug` | flag | False | Developer-level diagnostics (full transaction records, matching details, LLM prompts/responses) |

Behavior:
1. Load config from `config.toml`, `categories.toml`, `rules.toml`.
2. For each configured account, discover CSV files in the account's input directory using these rules:
   - Glob `*.csv` (case-insensitive) in the account's `input_dir`. Non-recursive (subdirectories are ignored).
   - Exclude hidden files (names starting with `.`) and temp files (names starting with `~` or `_`).
   - All matching files are passed to the parser. Since parsers return all rows and the pipeline filters by target month, there is no need for filename-based date filtering.
3. Run the pipeline: parse -> filter to target month -> deduplicate -> detect transfers -> enrich -> categorize -> export.
4. Write `output/YYYY-MM.csv`. Print summary to stdout.

If an output file already exists for the target month, overwrite it.

#### `expense learn`

The learning loop. Compares original output to user-corrected version.

```
expense learn --original output/2026-01.csv --corrected output/2026-01-corrected.csv
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--original` | path | Required | Path to the pipeline's output CSV |
| `--corrected` | path | Required | Path to the user-corrected CSV |
| `--verbose` | flag | False | Show details of each learned rule |

Behavior:
1. Read both CSVs. Index by `transaction_id`.
2. For each transaction where `category` or `subcategory` differs:
   - Extract the merchant pattern and the corrected category.
   - If a user rule already exists for this pattern, skip (do not overwrite user rules).
   - If a learned rule exists, update it with the correction.
   - If no rule exists, add a new learned rule.
3. Write updated rules to the `[learned_rules]` section of `rules.toml`.
4. Print a summary: N new rules added, N rules updated, N conflicts skipped.

#### `expense enrich` (Phase 2 -- not in MVP)

Run enrichment for a given month and source.

```
expense enrich --month 2026-01 --source target
```

Fetches item-level data for matching transactions and writes to `enrichment-cache/`. The pipeline's enrich stage reads from this cache.

#### `expense init`

Initialize a new data directory with the standard structure.

```
expense init
expense init --dir ~/expenses
```

Creates:
- `config.toml` with default settings and example account entries
- `categories.toml` with the default 18-category taxonomy
- `rules.toml` with empty `[user_rules]` and `[learned_rules]` sections
- `input/` directory
- `output/` directory
- `enrichment-cache/` directory

If files/directories already exist, skip them (do not overwrite).

### CLI-to-Pipeline Mapping

```
expense process  -->  pipeline.run(month, config)
expense learn    -->  categorizer.learn(original_path, corrected_path, rules)
expense init     -->  config.initialize(target_dir)
expense enrich   -->  (Phase 2: enrichment_source.fetch + cache write)
```

The CLI layer (`cli.py`) handles argument parsing, config loading, and error display. It delegates all business logic to `pipeline.py`, `categorizer.py`, and `config.py`.

---

## 7. Configuration

Three TOML files, all in the project root directory.

### config.toml

```toml
# Expense Tracker v3 configuration

[general]
output_dir = "output"
enrichment_cache_dir = "enrichment-cache"

[transfer_detection]
keywords = ["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"]
date_window_days = 5

[llm]
provider = "anthropic"          # "anthropic" or "none"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"  # Name of env var containing the API key

# Account definitions
[[accounts]]
name = "Chase Credit Card"
institution = "chase"
parser = "chase"
account_type = "credit_card"
input_dir = "input/chase"

[[accounts]]
name = "Capital One Credit Card"
institution = "capital_one"
parser = "capital_one"
account_type = "credit_card"
input_dir = "input/capital-one"

[[accounts]]
name = "Elevations Credit Union"
institution = "elevations"
parser = "elevations"
account_type = "checking"
input_dir = "input/elevations"
```

### categories.toml

```toml
# Category taxonomy -- the valid categories and subcategories

[Housing]
subcategories = ["Mortgage"]

[Utilities]
subcategories = ["Electric/Water/Internet", "Natural Gas", "Mobile Phone", "Television"]

[Food & Dining]
subcategories = ["Groceries", "Restaurant", "Fast Food", "Coffee", "Alcohol", "Delivery"]

[Transportation]
subcategories = ["Gas/Fuel", "Parking/Tolls", "Public Transit", "Rideshare", "Service & Maintenance", "Registration/DMV"]

[Kids]
subcategories = ["Clothing", "Supplies", "Activities", "Toys", "School", "Preschool", "Camps"]

[Health & Fitness]
subcategories = ["Gym/Classes", "Skiing", "Biking", "Hockey", "Race/Event Fees", "Equipment & Maintenance"]

[Healthcare]
subcategories = ["Doctor", "Dental", "Vision", "Pharmacy", "Therapy"]

[Entertainment]
subcategories = ["Tickets/Events", "Games", "Movies", "Subscriptions"]

[Shopping]
subcategories = ["Clothing", "Electronics", "Home Goods", "Books", "Jewelry"]

[Home & Garden]
subcategories = ["Maintenance & Repairs", "Furniture & Decor", "Appliances", "Garden & Lawn", "Tools & Hardware", "Home Services"]

[Personal Care]
subcategories = ["Haircut/Barber", "Beauty/Spa", "Cosmetics"]

[Pets]
subcategories = ["Food", "Vet", "Daycare/Boarding", "Grooming", "Supplies"]

[Gifts & Charity]
subcategories = ["Gifts", "Donations"]

[Travel]
subcategories = ["Flight", "Hotel/Lodging", "Rental Car/Transport", "Vacation/Activities", "Baggage/Fees"]

[Education]
subcategories = ["Tuition", "Books & Supplies", "Courses/Training"]

[Insurance]
subcategories = []

[Business]
subcategories = []

[Miscellaneous]
subcategories = []
```

### rules.toml

```toml
# Merchant-to-category mapping rules
# User rules always take precedence over learned rules.
# Matching: case-insensitive substring, longest match wins.

[user_rules]
# Manually authored rules. The system never modifies this section.
# Format: pattern = "Category" or pattern = "Category:Subcategory"

# Examples:
# "KING SOOPERS" = "Food & Dining:Groceries"
# "CHIPOTLE" = "Food & Dining:Fast Food"

[learned_rules]
# System-managed rules from the learn command. Do not hand-edit.
# Same format as user_rules.
```

### Config Loading (`config.py`)

`config.py` provides these functions:

```python
def load_config(root: Path) -> AppConfig:
    """Load config.toml and return an AppConfig."""

def load_categories(root: Path) -> list[dict]:
    """Load categories.toml and return the taxonomy as a list of
    {name: str, subcategories: list[str]}."""

def load_rules(root: Path) -> list[MerchantRule]:
    """Load rules.toml and return a sorted list of MerchantRule objects.
    User rules come first, then learned rules. Within each group,
    rules are in file order (insertion order)."""

def save_learned_rules(root: Path, rules: list[MerchantRule]) -> None:
    """Write updated learned rules to the [learned_rules] section of rules.toml.
    Preserves the [user_rules] section exactly as-is."""

def initialize(target_dir: Path) -> None:
    """Create the standard directory structure and default config files."""
```

Writing TOML requires `tomli-w` (since `tomllib` is read-only). This is the only TOML-writing dependency.

---

## 8. Categorization Engine

The categorizer is the most important module. It implements the two-tier system and the learn workflow.

### Categorization Flow

```
For each transaction:
    |
    [1] Match against rules (user rules first, then learned)
    |-- Match found? --> Apply category, done.
    |
    [2] No match: add to LLM batch
    |
After all transactions checked:
    |
    [3] Send LLM batch (if any and if LLM enabled)
    |-- LLM returned suggestions? --> Apply to transactions
    |-- LLM unavailable/failed? --> Leave as "Uncategorized"
    |
    [4] Return all transactions
```

### Rule Matching Algorithm

```python
def match_rules(merchant: str, rules: list[MerchantRule]) -> MerchantRule | None:
    """Find the best matching rule for a merchant string.

    Strategy: case-insensitive substring, longest match wins.
    User rules and learned rules are pre-sorted by source (user first)
    and original file order. Among all matching rules, the one with
    the longest pattern wins. Ties broken by list order (user > learned,
    then insertion order within group).
    """
    merchant_upper = merchant.upper()
    best: MerchantRule | None = None
    for rule in rules:
        if rule.pattern.upper() in merchant_upper:
            if best is None or len(rule.pattern) > len(best.pattern):
                best = rule
    return best
```

User rules beat learned rules only when patterns have the same length, because they appear first in the list and ties go to insertion order. In practice, the longest-match-wins rule handles most cases correctly regardless of source.

### LLM Batch Categorization

When uncategorized transactions remain after rule matching:

1. Build a prompt containing:
   - The category taxonomy (from `categories.toml`)
   - A list of uncategorized transactions (merchant, description, amount, date)
   - Instructions to return a JSON array of `{merchant, category, subcategory}` objects
2. Send a single API call to the configured LLM provider.
3. Parse the JSON response.
4. Apply suggestions to the matching transactions in the pipeline.

The LLM suggestions are applied to the output CSV so the user sees a fully categorized report. They are NOT written to `rules.toml`. The learn command is the only path to persisting new rules.

### LLM Prompt Structure

```
You are categorizing household expenses. For each transaction below,
assign the most appropriate category and subcategory from the provided taxonomy.

## Category Taxonomy
{formatted taxonomy from categories.toml}

## Transactions to Categorize
{formatted list: merchant | description | amount | date}

## Response Format
Return a JSON array. Each element:
{"merchant": "...", "category": "...", "subcategory": "..."}

Use only categories and subcategories from the taxonomy above.
If no subcategory applies, use an empty string for subcategory.
```

### Learn Workflow

```python
def learn(original_path: Path, corrected_path: Path, rules: list[MerchantRule]) -> LearnResult:
    """Compare original and corrected CSVs, extract new rules.

    Returns a LearnResult with counts of added, updated, and skipped rules.
    """
```

Algorithm:
1. Parse both CSVs, index by `transaction_id`.
2. For each transaction present in both files where category or subcategory changed:
   a. Extract the rule pattern from the transaction's `merchant` field exactly as stored (already normalized by the parser). No additional normalization is applied -- the pattern is the `merchant` value verbatim. This means parsers are responsible for producing stable, matchable merchant strings (stripping bank-specific prefixes, trailing reference numbers, etc.), because the quality of learned patterns depends directly on parser normalization. If the same merchant appears with different names across banks (e.g., "CHIPOTLE MEXICAN GRIL" vs. "CHIPOTLE ONLINE"), they will produce separate rules; the longest-match-wins algorithm will still match correctly as long as patterns share a common substring with the real merchant names.
   b. Check if a user rule already covers this merchant. If yes, skip (never overwrite user rules).
   c. Check if a learned rule already exists for this exact pattern. If yes, update its category.
   d. Otherwise, add a new learned rule.
3. Write the updated learned rules via `config.save_learned_rules()`.

---

## 9. Error Handling

### Strategy

Partial failure is always preferred over total failure. Each pipeline stage handles errors independently and reports them.

### Error Propagation

Errors do not propagate as exceptions across pipeline stages. Instead, every stage function returns a `StageResult`. The pipeline collects warnings and errors from each stage's result and includes them in the final summary. This is the universal return type for all pipeline stage functions, including parsers.

```python
@dataclass
class StageResult:
    transactions: list[Transaction]
    warnings: list[str]
    errors: list[str]
```

Defined in `models.py` alongside `Transaction`.

### Per-Stage Error Behavior

| Stage | Error Case | Behavior |
|-------|-----------|----------|
| **Parse** | File not found / unreadable | Skip file, add error. Process other files. |
| **Parse** | Wrong CSV format (missing expected columns) | Skip file, add error. |
| **Parse** | Malformed row | Skip row, add warning. |
| **Parse** | >10% malformed rows in one file | Skip entire file, add error (likely format change). |
| **Deduplicate** | (No failure modes) | -- |
| **Detect Transfers** | No matching transfer pair | Not an error; transaction is not a transfer. |
| **Enrich** | Missing enrichment cache | Not an error; skip enrichment for that transaction. |
| **Enrich** | Split amounts do not sum to original | Keep original unsplit, add warning. |
| **Categorize** | No rule match | Fall through to LLM tier. |
| **Categorize** | LLM unavailable | Leave uncategorized, add warning. |
| **Categorize** | LLM returns unparseable response | Leave uncategorized, add warning. |
| **Export** | Output directory not writable | Fatal error (cannot produce output). |

### Summary Output

After processing, the summary includes:

```
== Processing Summary: 2026-01 ==
Sources:  Chase (148 txns), Capital One (89 txns), Elevations (42 txns)
Total:    279 transactions (6 transfers excluded)
Enriched: 0 of 12 eligible (no enrichment cache)
Categorized: 251 / 273 (91.9%)
  - Rule match:  238
  - LLM:          13
  - Uncategorized: 22

Top uncategorized merchants:
  1. NEW MERCHANT XYZ         (3 txns, $142.50)
  2. UNKNOWN STORE            (2 txns, $67.30)
  ...

Spending by category:
  Food & Dining:     $1,247.32
  Transportation:      $438.10
  ...

Warnings: 2
  - chase/Activity2026.csv: skipped 1 malformed row (row 47)
  - LLM: 3 transactions could not be parsed from response
```

---

## 10. Key ADRs

### ADR-1: Flat Files Over SQLite

**Decision:** Monthly CSVs and TOML config files. No database.

**Rationale:** The primary consumption path is Google Sheets with pivot tables. Monthly CSV files import directly into Sheets and are self-contained, portable, and version-controllable. The user also inspects and edits data in text editors and spreadsheets for the learn workflow. SQLite would add a dependency and make data opaque for marginal query benefits that are not needed until Phase 3. If cross-month queries become necessary, SQLite can be added as a read-only index over existing CSVs.

### ADR-2: Deterministic Transaction IDs via Hash

**Decision:** SHA-256 hash of `institution|date|merchant_upper|amount|row_ordinal`, truncated to 12 hex characters.

**Rationale:** Banks do not provide unique transaction IDs in CSV exports. A deterministic hash ensures the same input always produces the same ID, enabling deduplication across re-runs and overlapping downloads. Row ordinal is included to distinguish identical transactions on the same day (e.g., two charges at the same coffee shop). This approach has worked reliably in v2 across two years and three banks.

### ADR-3: Substring Matching, Not Regex

**Decision:** Case-insensitive substring matching with longest-match-wins for merchant rules.

**Rationale:** Substring matching handles 95%+ of merchant patterns. It is easy to understand, easy to debug, and requires no special syntax knowledge. Regex adds complexity and footguns for marginal benefit. If a case arises that truly requires regex, it can be added as an opt-in `type: regex` field on individual rules.

### ADR-4: Enrichment as Separate Pre-Processing

**Decision:** Enrichment (Amazon/Target item-level data) is a separate `enrich` command that writes to a local cache. The main pipeline reads from the cache.

**Rationale:** Enrichment sources (browser automation, external APIs) are inherently slow and unreliable. Making them a blocking pipeline stage would make the common case (no enrichment) slower and less reliable. Separating them keeps the fast path fast: `expense process` completes in seconds. Users who want enrichment run `expense enrich` first, then `expense process`.

### ADR-5: LLM Suggestions Auto-Applied, Cached Only After Learn

**Decision:** LLM categorization suggestions are written to the output CSV immediately but are not persisted to `rules.toml` until the user runs `learn`.

**Rationale:** This gives the user a fully categorized report on first run (no "Uncategorized" for LLM-handled transactions) while ensuring the knowledge base only grows from confirmed categorizations. The review-and-learn cycle IS the confirmation step. This prevents bad LLM suggestions from polluting the rules file.

### ADR-6: Click for CLI Framework

**Decision:** Click over Typer.

**Rationale:** Both are viable. Click is chosen for its explicit decorator syntax, mature ecosystem, and broad compatibility. The CLI is simple enough that either framework works. The choice avoids Typer's dependency on Click (it wraps Click internally) and its reliance on type annotation parsing, which can be surprising in edge cases.

### ADR-7: Raw HTTP for LLM, No Framework

**Decision:** Direct HTTP calls to the Anthropic Messages API via `httpx`. No LangChain, no LiteLLM.

**Rationale:** The LLM interaction is a single prompt-and-response pattern. A framework adds hundreds of transitive dependencies for functionality we do not use. A thin adapter (~50 lines for the Anthropic implementation) is sufficient, testable, and transparent. The `LLMAdapter` protocol enables adding providers later without a framework.

### ADR-8: Three TOML Config Files

**Decision:** Separate `config.toml`, `categories.toml`, and `rules.toml`.

**Rationale:** Separation of concerns. `rules.toml` will grow to 500+ entries and changes frequently (via `learn`). `categories.toml` rarely changes. `config.toml` is application settings. Keeping them separate means the `learn` command only touches `rules.toml`, reduces merge conflicts in version control, and keeps each file focused and readable.

---

## Appendix: Dependencies

### MVP Direct Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `click` | CLI framework | Command definitions, argument parsing |
| `httpx` | HTTP client | LLM API calls |
| `tomli-w` | TOML writing | Writing learned rules to rules.toml (`tomllib` is read-only) |

Three direct runtime dependencies. `tomllib` (stdlib) handles TOML reading. `csv` (stdlib) handles CSV I/O. `hashlib` (stdlib) handles transaction ID generation.

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `pytest` | Testing |
| `ruff` | Linting and formatting |

### Phase 2 Additions (not in MVP)

| Package | Purpose |
|---------|---------|
| `gspread` | Google Sheets API |
| `google-auth` | Google authentication |
| `playwright` | Browser automation for Amazon enrichment |

---

## Changelog

### 2026-02-14 -- Address architecture review findings (M1-M5)

Targeted updates to resolve five moderate concerns identified by the architecture reviewer:

- **M1 (Enrichment trigger ambiguity):** Added clarifying language in Stage 4 (Enrich) that "eligible for enrichment" means "has cached enrichment data on disk," not "triggers a fetch." The enrich stage is a pure cache lookup.
- **M2 (Merchant pattern extraction underspecified):** Expanded the learn algorithm (Section 8, step 2a) to specify that patterns are extracted from the `merchant` field verbatim as stored in the Transaction. Parsers own normalization; no additional normalization is applied at learn time. Documented behavior for cross-bank merchant name variations.
- **M3 (Month filtering has no clear home):** Resolved the contradiction between the Parser Protocol (no month parameter) and the CLI text ("parsers handle date filtering"). Parsers now explicitly return all rows; the pipeline owns month filtering. Updated Stage 1 description, Parser Protocol signature, and CLI behavior text for consistency.
- **M4 (`StageResult` not integrated into stage signatures):** Updated Section 3 preamble, Section 5 Parser Protocol, and Section 9 to establish `StageResult` as the universal return type for all pipeline stage functions. Moved `StageResult` definition to `models.py`.
- **M5 (CSV file discovery unspecified):** Added file discovery rules to the `expense process` behavior: glob `*.csv` (case-insensitive), non-recursive, excluding hidden and temp files.

Additional context incorporated from Product Owner: primary data consumption is via Google Sheets with pivot tables. Updated Stage 6 (Export) and ADR-1 to reflect this.
