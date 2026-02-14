"""Core data models for Expense Tracker.

This module defines all dataclasses and utility functions used throughout the
pipeline. It has zero internal imports -- everything depends on it, but it
depends on nothing within the package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


def generate_transaction_id(
    institution: str,
    txn_date: date,
    merchant: str,
    amount: Decimal,
    row_ordinal: int,
) -> str:
    """Generate a deterministic transaction ID from uniqueness components.

    The ID is a 12-character hex string derived from a SHA-256 hash of the
    pipe-delimited concatenation of: institution (as-is), ISO date, uppercased
    and stripped merchant string, amount as string, and 0-based row ordinal.

    This ensures that:
    - The same CSV row always produces the same ID (deterministic).
    - Two identical purchases on the same day at the same merchant are
      distinguished by their row ordinal in the source file.
    - IDs are stable across re-runs and overlapping CSV downloads.

    Args:
        institution: Institution key, e.g. "chase", "capital_one".
        txn_date: Transaction date.
        merchant: Merchant/payee name (will be stripped and uppercased).
        amount: Transaction amount as Decimal.
        row_ordinal: 0-based row index within the source CSV file.

    Returns:
        A 12-character lowercase hex string.
    """
    raw = f"{institution}|{txn_date.isoformat()}|{merchant.strip().upper()}|{amount}|{row_ordinal}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class Transaction:
    """A single financial transaction flowing through the pipeline.

    This is the core data object that every pipeline stage receives and
    returns. Fields are populated progressively: parsers fill in the base
    fields, then later stages add category, transfer, and split information.

    Attributes:
        transaction_id: Deterministic 12-char hex hash identifying this
            transaction uniquely.
        date: Transaction date (not post date).
        merchant: Normalized merchant/payee name.
        description: Original description from the bank CSV.
        amount: Signed decimal amount. Negative means expense, positive
            means refund or credit.
        institution: Internal institution key, e.g. "chase".
        account: Account identifier from config.
        category: Top-level category, or "Uncategorized" if not yet
            categorized.
        subcategory: Subcategory within the top-level category, or empty
            string if none applies.
        is_transfer: True if this transaction was detected as an internal
            transfer between accounts.
        is_return: True if the amount is positive (refund/credit).
        split_from: Parent transaction_id if this is a split line item,
            or empty string if it is not a split.
        source_file: Path to the source CSV file (for debugging; not
            included in output).
    """

    transaction_id: str
    date: date
    merchant: str
    description: str
    amount: Decimal
    institution: str
    account: str
    category: str = "Uncategorized"
    subcategory: str = ""
    is_transfer: bool = False
    is_return: bool = False
    split_from: str = ""
    source_file: str = ""


@dataclass
class StageResult:
    """Return type for every pipeline stage function.

    Each stage processes what it can and reports what it could not. The
    pipeline accumulates warnings and errors across all stages for the
    final summary.

    Attributes:
        transactions: The list of transactions after this stage's
            processing (possibly modified, filtered, or expanded).
        warnings: Non-fatal issues encountered during processing, such
            as skipped rows or unparseable LLM responses.
        errors: Fatal issues for individual items or files, such as
            unreadable files or format mismatches. The stage still
            returns whatever it could process successfully.
    """

    transactions: list[Transaction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class MerchantRule:
    """A merchant-to-category mapping rule.

    Rules are matched via case-insensitive substring matching against the
    transaction's merchant field. When multiple rules match, the one with
    the longest pattern wins. User rules take precedence over learned
    rules when pattern lengths are equal.

    Attributes:
        pattern: Substring to match against merchant names
            (case-insensitive).
        category: Target top-level category.
        subcategory: Target subcategory, or empty string if none.
        source: Origin of this rule: "user" (hand-authored, never
            overwritten) or "learned" (system-managed via the learn
            command).
    """

    pattern: str
    category: str
    subcategory: str = ""
    source: str = "user"


@dataclass
class AccountConfig:
    """Configuration for a single bank account.

    Defines how to locate and parse CSV files for one account.

    Attributes:
        name: Human-readable display name, e.g. "Chase Credit Card".
        institution: Internal key used in transaction IDs, e.g. "chase".
        parser: Parser name that maps to a parser function in the
            registry, e.g. "chase".
        account_type: Either "credit_card" or "checking".
        input_dir: Relative path to the directory containing CSV files
            for this account, e.g. "input/chase".
    """

    name: str
    institution: str
    parser: str
    account_type: str
    input_dir: str


@dataclass
class AppConfig:
    """Top-level application configuration loaded from config.toml.

    Attributes:
        accounts: List of configured bank accounts.
        output_dir: Directory for output CSV files. Default: "output".
        enrichment_cache_dir: Directory for enrichment cache JSON files.
            Default: "enrichment-cache".
        transfer_keywords: Keywords that indicate a transaction is an
            internal transfer (e.g. bill payments). Matched
            case-insensitively against merchant/description.
        transfer_date_window: Maximum number of days between a checking
            debit and a credit card credit to consider them a transfer
            pair. Default: 5.
        llm_provider: LLM provider name. "anthropic" or "none".
        llm_model: Model identifier, e.g. "claude-sonnet-4-20250514".
        llm_api_key_env: Name of the environment variable containing
            the API key.
    """

    accounts: list[AccountConfig] = field(default_factory=list)
    output_dir: str = "output"
    enrichment_cache_dir: str = "enrichment-cache"
    transfer_keywords: list[str] = field(
        default_factory=lambda: ["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"]
    )
    transfer_date_window: int = 5
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"
