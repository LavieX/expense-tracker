# Expense Tracker

A Python CLI tool for multi-account household expense tracking and categorization. This is **v3** — a fresh rewrite from the ground up.

## What It Does

- **Multi-bank CSV parsing** — Imports and normalizes transaction exports from Chase, Capital One, and Elevations Credit Union
- **Deduplication** — Detects and removes duplicate transactions across accounts and import runs
- **Two-tier categorization** — Cached merchant-to-category mappings for speed, with LLM fallback for unknown merchants
- **Correction-based learning** — Manual corrections feed back into the merchant cache so the same merchant is categorized correctly next time
- **Monthly reporting** — Exports categorized transactions as CSV for downstream use

## Setup

```bash
# Clone the repo
git clone git@github.com:LavieX/expense-tracker.git
cd expense-tracker

# Create a virtual environment and install in dev mode
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
expense --help
```

## Development

```bash
# Run tests
pytest

# Run linter
ruff check src/ tests/
```

## License

MIT
