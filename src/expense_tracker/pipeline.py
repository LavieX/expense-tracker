"""Pipeline orchestration for Expense Tracker.

Composes the six processing stages: parse, filter, deduplicate, detect
transfers, enrich, and categorize.  Each stage receives a list of
:class:`~expense_tracker.models.Transaction` objects and returns a
:class:`~expense_tracker.models.StageResult`.  The pipeline accumulates
warnings and errors from every stage into a final
:class:`~expense_tracker.models.PipelineResult`.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from expense_tracker.models import (
    AppConfig,
    MerchantRule,
    PipelineResult,
    StageResult,
    Transaction,
)
from expense_tracker.parsers import get_parser
from expense_tracker.recurring import detect_recurring

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    month: str,
    config: AppConfig,
    categories: list[dict],
    rules: list[MerchantRule],
    root: Path,
    exclude_patterns: list[str] | None = None,
) -> PipelineResult:
    """Run the full processing pipeline for *month*.

    Stages executed in order:

    1. **Parse** -- discover CSVs per account, call parsers, concatenate.
    2. **Filter** -- keep only transactions within the target month.
    3. **Exclude** -- filter out transactions matching exclude patterns.
    4. **Deduplicate** -- remove duplicate ``transaction_id`` values.
    5. **Detect transfers** -- match checking debits to CC credits.
    6. **Enrich** -- look up enrichment-cache, split if found.
    7. **Categorize** -- apply rules (tier 1); LLM fallback is not
       invoked by this module (a callable may be injected later).

    Args:
        month: Target month as ``"YYYY-MM"`` string.
        config: Application configuration.
        categories: Category taxonomy (list of dicts with ``name`` and
            ``subcategories`` keys).
        rules: Merchant-to-category mapping rules, ordered user-first.
        root: Project root directory (paths in *config* are relative to
            this).
        exclude_patterns: Optional list of patterns for transactions to
            exclude entirely (e.g., salary, internal transfers). If not
            provided, defaults to empty list.

    Returns:
        A :class:`PipelineResult` with the final transaction list and all
        accumulated warnings and errors.
    """
    all_warnings: list[str] = []
    all_errors: list[str] = []

    if exclude_patterns is None:
        exclude_patterns = []

    # -- Stage 1: Parse -------------------------------------------------------
    parse_result = _parse_stage(config, root)
    all_warnings.extend(parse_result.warnings)
    all_errors.extend(parse_result.errors)
    transactions = parse_result.transactions

    # -- Stage 1b: Filter to target month -------------------------------------
    filter_result = _filter_month(transactions, month)
    all_warnings.extend(filter_result.warnings)
    all_errors.extend(filter_result.errors)
    transactions = filter_result.transactions

    # -- Stage 1c: Exclude transactions ---------------------------------------
    exclude_result = _exclude_transactions(transactions, exclude_patterns)
    all_warnings.extend(exclude_result.warnings)
    all_errors.extend(exclude_result.errors)
    transactions = exclude_result.transactions

    # -- Stage 2: Deduplicate -------------------------------------------------
    dedup_result = _deduplicate(transactions)
    all_warnings.extend(dedup_result.warnings)
    all_errors.extend(dedup_result.errors)
    transactions = dedup_result.transactions

    # -- Stage 3: Detect transfers --------------------------------------------
    transfer_result = _detect_transfers(transactions, config)
    all_warnings.extend(transfer_result.warnings)
    all_errors.extend(transfer_result.errors)
    transactions = transfer_result.transactions

    # -- Stage 4: Enrich ------------------------------------------------------
    enrich_result = _enrich(transactions, root, config)
    all_warnings.extend(enrich_result.warnings)
    all_errors.extend(enrich_result.errors)
    transactions = enrich_result.transactions

    # -- Stage 4b: Tag sources ------------------------------------------------
    source_result = _tag_sources(transactions)
    transactions = source_result.transactions

    # -- Stage 5: Categorize --------------------------------------------------
    cat_result = _categorize(transactions, rules)
    all_warnings.extend(cat_result.warnings)
    all_errors.extend(cat_result.errors)
    transactions = cat_result.transactions

    # -- Stage 6: Detect recurring --------------------------------------------
    recurring_result = _detect_recurring_stage(transactions, root, config, rules)
    all_warnings.extend(recurring_result.warnings)
    all_errors.extend(recurring_result.errors)
    transactions = recurring_result.transactions

    return PipelineResult(
        transactions=transactions,
        warnings=all_warnings,
        errors=all_errors,
    )


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _discover_csv_files(input_dir: Path) -> list[Path]:
    """Discover CSV files in *input_dir*, excluding hidden and temp files.

    Matches ``*.csv`` case-insensitively, non-recursive.  Excludes files
    whose name starts with ``.``, ``~``, or ``_``.

    Returns:
        Sorted list of matching file paths.
    """
    if not input_dir.is_dir():
        return []

    csv_files: list[Path] = []
    for entry in input_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        # Exclude hidden/temp files
        if name.startswith(".") or name.startswith("~") or name.startswith("_"):
            continue
        # Case-insensitive .csv check
        if name.lower().endswith(".csv"):
            csv_files.append(entry)

    csv_files.sort()
    return csv_files


def _parse_stage(config: AppConfig, root: Path) -> StageResult:
    """Stage 1: discover CSVs per account, call parsers, concatenate."""
    all_transactions: list[Transaction] = []
    warnings: list[str] = []
    errors: list[str] = []

    for acct in config.accounts:
        input_dir = root / acct.input_dir

        try:
            parser_fn = get_parser(acct.parser)
        except KeyError:
            errors.append(f"Unknown parser {acct.parser!r} for account {acct.name!r}")
            continue

        csv_files = _discover_csv_files(input_dir)
        if not csv_files:
            warnings.append(f"No CSV files found in {input_dir}")
            continue

        for csv_path in csv_files:
            result = parser_fn(csv_path, acct.institution, acct.name)
            all_transactions.extend(result.transactions)
            warnings.extend(result.warnings)
            errors.extend(result.errors)

    return StageResult(
        transactions=all_transactions,
        warnings=warnings,
        errors=errors,
    )


def _filter_month(transactions: list[Transaction], month: str) -> StageResult:
    """Filter transactions to only those in the target month.

    Args:
        transactions: All parsed transactions (may span many months).
        month: Target month as ``"YYYY-MM"`` string.

    Returns:
        StageResult with only transactions whose date falls in *month*.
    """
    year, mon = month.split("-")
    year_int = int(year)
    mon_int = int(mon)

    first_day = date(year_int, mon_int, 1)
    # Compute last day: go to next month, subtract one day.
    if mon_int == 12:
        last_day = date(year_int + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year_int, mon_int + 1, 1) - timedelta(days=1)

    filtered = [
        txn for txn in transactions if first_day <= txn.date <= last_day
    ]

    return StageResult(transactions=filtered)


def _exclude_transactions(
    transactions: list[Transaction],
    exclude_patterns: list[str],
) -> StageResult:
    """Filter out transactions matching exclude patterns.

    Exclude patterns are used to filter out transactions like salary,
    income, or internal transfers that should not be processed.

    Args:
        transactions: All transactions to filter.
        exclude_patterns: List of patterns to match against merchant field.
            Matching is case-insensitive substring match.

    Returns:
        StageResult with excluded transactions removed and a warning
        reporting the count of excluded transactions.
    """
    if not exclude_patterns:
        return StageResult(transactions=transactions)

    # Convert patterns to uppercase for case-insensitive matching
    patterns_upper = [p.upper() for p in exclude_patterns]

    # Filter out transactions whose merchant matches any exclude pattern
    excluded_count = 0
    filtered: list[Transaction] = []

    for txn in transactions:
        merchant_upper = txn.merchant.upper()
        is_excluded = any(pattern in merchant_upper for pattern in patterns_upper)

        if is_excluded:
            excluded_count += 1
        else:
            filtered.append(txn)

    warnings: list[str] = []
    if excluded_count > 0:
        warnings.append(f"Excluded {excluded_count} transaction(s) matching exclude patterns")

    return StageResult(transactions=filtered, warnings=warnings)


def _deduplicate(transactions: list[Transaction]) -> StageResult:
    """Stage 2: remove duplicates by ``transaction_id``, keep first occurrence."""
    seen: dict[str, Transaction] = {}
    unique: list[Transaction] = []
    dup_count = 0

    for txn in transactions:
        if txn.transaction_id not in seen:
            seen[txn.transaction_id] = txn
            unique.append(txn)
        else:
            dup_count += 1

    warnings: list[str] = []
    if dup_count > 0:
        warnings.append(f"Removed {dup_count} duplicate transaction(s)")

    return StageResult(transactions=unique, warnings=warnings)


def _detect_transfers(
    transactions: list[Transaction],
    config: AppConfig,
) -> StageResult:
    """Stage 3: match checking debits to credit card credits.

    Algorithm:
    1. Collect all checking-account debits whose merchant/description matches
       any of the configured transfer keywords (case-insensitive).
    2. For each, search for a credit card credit within the date window with
       the same absolute amount.
    3. Mark both sides ``is_transfer=True``.
    """
    keywords_upper = [kw.upper() for kw in config.transfer_keywords]
    window = config.transfer_date_window

    # Build a lookup of account types by institution
    acct_types: dict[str, str] = {}
    for acct in config.accounts:
        acct_types[acct.institution] = acct.account_type

    # Find checking debits matching transfer keywords
    checking_debits: list[Transaction] = []
    for txn in transactions:
        if acct_types.get(txn.institution) != "checking":
            continue
        if txn.amount >= 0:
            continue  # Not a debit
        # Check if merchant or description matches any keyword
        merchant_upper = txn.merchant.upper()
        desc_upper = txn.description.upper()
        for kw in keywords_upper:
            if kw in merchant_upper or kw in desc_upper:
                checking_debits.append(txn)
                break

    # Find credit card credits (positive amounts on credit_card accounts)
    cc_credits: list[Transaction] = []
    for txn in transactions:
        if acct_types.get(txn.institution) != "credit_card":
            continue
        if txn.amount <= 0:
            continue  # Not a credit
        cc_credits.append(txn)

    # Match pairs: checking debit to CC credit by amount and date window
    matched_cc_ids: set[str] = set()
    for debit in checking_debits:
        debit_abs = abs(debit.amount)
        for credit in cc_credits:
            if credit.transaction_id in matched_cc_ids:
                continue
            if abs(credit.amount) != debit_abs:
                continue
            day_diff = abs((credit.date - debit.date).days)
            if day_diff > window:
                continue
            # Match found -- mark both
            debit.is_transfer = True
            credit.is_transfer = True
            matched_cc_ids.add(credit.transaction_id)
            break  # Move to next checking debit

    return StageResult(transactions=transactions)




def _detect_retailer_source(retailer_hint: str, merchant: str) -> str:
    """Determine the retailer source tag from enrichment data or merchant name.

    Checks the enrichment cache ``retailer`` field first, then falls back to
    pattern-matching the original merchant name.

    Returns:
        A clean retailer tag like ``"Amazon"`` or ``"Target"``, or empty string
        if the retailer cannot be determined.
    """
    hint_upper = retailer_hint.upper()
    if "AMAZON" in hint_upper or "AMZN" in hint_upper or "AMZ" in hint_upper:
        return "Amazon"
    if "TARGET" in hint_upper:
        return "Target"

    merchant_upper = merchant.upper()
    if "AMAZON" in merchant_upper or "AMZN" in merchant_upper or "AMZ" in merchant_upper:
        return "Amazon"
    if "TARGET" in merchant_upper:
        return "Target"

    return ""


def _tag_sources(transactions: list[Transaction]) -> StageResult:
    """Tag transactions with a retailer ``source`` based on merchant name.

    For transactions that were not enriched (and thus have no ``source`` set),
    pattern-match the merchant/description to detect known retailers:

    - Any merchant containing "AMAZON", "AMZN", or "AMZ" -> source = "Amazon"
    - Any merchant containing "TARGET" -> source = "Target"

    Transactions that already have a ``source`` (set by enrichment) are left
    unchanged.
    """
    for txn in transactions:
        if txn.source:
            continue  # Already tagged by enrichment
        merchant_upper = txn.merchant.upper()
        if "AMAZON" in merchant_upper or "AMZN" in merchant_upper or "AMZ" in merchant_upper:
            txn.source = "Amazon"
        elif "TARGET" in merchant_upper:
            txn.source = "Target"

    return StageResult(transactions=transactions)


def _enrich(
    transactions: list[Transaction],
    root: Path,
    config: AppConfig,
) -> StageResult:
    """Stage 4: look up enrichment cache, split transactions if data exists.

    For each transaction, check ``enrichment-cache/{transaction_id}.json``.
    If found, replace the transaction with split line items.  Validate that
    split amounts sum to the original (within $0.01 tolerance).
    """
    cache_dir = root / config.enrichment_cache_dir
    warnings: list[str] = []
    result: list[Transaction] = []

    for txn in transactions:
        cache_file = cache_dir / f"{txn.transaction_id}.json"
        if not cache_file.is_file():
            result.append(txn)
            continue

        # Load enrichment data
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            warnings.append(
                f"Could not read enrichment cache for {txn.transaction_id}: {exc}"
            )
            result.append(txn)
            continue

        # Expect data to be a dict with an "items" list
        items = data.get("items", [])
        if not items:
            result.append(txn)
            continue

        # Determine the retailer source from the enrichment data or
        # the parent transaction's merchant name.
        enrichment_source = _detect_retailer_source(
            data.get("retailer", ""), txn.merchant
        )

        # Build split transactions
        splits: list[Transaction] = []
        split_total = Decimal("0")
        for i, item in enumerate(items, start=1):
            item_amount = Decimal(str(item.get("amount", "0")))
            split_total += item_amount
            # Use the retailer name (e.g. "Amazon", "Target") as the
            # merchant for enriched splits.  The product-specific name
            # lives in the description field so categorization rules can
            # match against it.
            if enrichment_source:
                enriched_merchant = enrichment_source
            else:
                enriched_merchant = item.get("merchant", txn.merchant)
            split_txn = Transaction(
                transaction_id=f"{txn.transaction_id}-{i}",
                date=txn.date,
                merchant=enriched_merchant,
                description=item.get("description", txn.description),
                amount=item_amount,
                institution=txn.institution,
                account=txn.account,
                category="Uncategorized",
                subcategory="",
                is_transfer=txn.is_transfer,
                is_return=item_amount > 0,
                is_recurring=txn.is_recurring,
                split_from=txn.transaction_id,
                source=enrichment_source,
                source_file=txn.source_file,
            )
            splits.append(split_txn)

        # Validate sum: must match original within $0.01
        if abs(split_total - txn.amount) > Decimal("0.01"):
            warnings.append(
                f"Enrichment split amounts for {txn.transaction_id} sum to "
                f"{split_total}, expected {txn.amount}; keeping original"
            )
            result.append(txn)
        else:
            result.extend(splits)

    return StageResult(transactions=result, warnings=warnings)


def _categorize(
    transactions: list[Transaction],
    rules: list[MerchantRule],
) -> StageResult:
    """Stage 5: apply rule-based categorization (tier 1).

    For each uncategorized transaction, find the best matching rule using
    case-insensitive substring matching with longest-match-wins.  If no rule
    matches, the transaction retains ``category="Uncategorized"``.

    When the best merchant match is a generic category (no subcategory)
    and the transaction has a description, the description is checked
    for a more specific match first.  This allows product-specific rules
    to override generic retailer rules for enriched transactions from
    multi-product retailers like Amazon and Target.

    LLM fallback (tier 2) is not implemented in this module.  The
    ``categorizer`` module will handle that when available.
    """
    from expense_tracker.categorizer import match_rules as cat_match_rules

    for txn in transactions:
        if txn.category != "Uncategorized":
            continue
        match = cat_match_rules(txn.merchant, rules, description=txn.description)
        if match is not None:
            txn.category = match.category
            txn.subcategory = match.subcategory
            txn.is_recurring = match.recurring

    return StageResult(transactions=transactions)


def _detect_recurring_stage(
    transactions: list[Transaction],
    root: Path,
    config: AppConfig,
    rules: list[MerchantRule],
) -> StageResult:
    """Stage 6: auto-detect recurring merchants from historical data.

    Scans historical output CSVs to find merchants that appear regularly
    (3+ months) with similar amounts (within 20% variance). If a transaction's
    merchant matches a detected recurring pattern AND no rule explicitly sets
    recurring status, mark it as recurring.

    Rule-based recurring flags always take precedence over auto-detection.
    """
    output_dir = root / config.output_dir

    # Detect recurring merchants from historical data
    try:
        recurring_merchants = detect_recurring(transactions, output_dir)
    except Exception as exc:
        # Don't fail the pipeline if recurring detection fails
        return StageResult(
            transactions=transactions,
            warnings=[f"Recurring detection failed: {exc}"],
        )

    if not recurring_merchants:
        return StageResult(transactions=transactions)

    # Build a set of merchants that have explicit recurring rules
    # (so we don't override them)
    explicit_recurring_merchants: set[str] = set()
    for rule in rules:
        if rule.recurring:
            explicit_recurring_merchants.add(rule.pattern.upper())

    # Apply auto-detection to transactions
    auto_flagged = 0
    for txn in transactions:
        # Skip if already flagged as recurring by a rule
        if txn.is_recurring:
            continue

        # Skip if merchant has an explicit recurring rule
        merchant_upper = txn.merchant.upper()
        has_explicit_rule = any(
            pattern in merchant_upper for pattern in explicit_recurring_merchants
        )
        if has_explicit_rule:
            continue

        # Check if merchant matches any auto-detected recurring pattern
        if merchant_upper in recurring_merchants:
            txn.is_recurring = True
            auto_flagged += 1

    warnings = []
    if auto_flagged > 0:
        warnings.append(f"Auto-flagged {auto_flagged} recurring transaction(s)")

    return StageResult(transactions=transactions, warnings=warnings)


def _match_rules(merchant: str, rules: list[MerchantRule]) -> MerchantRule | None:
    """Find the best matching rule for *merchant*.

    Strategy: case-insensitive substring, longest pattern wins.  Ties are
    broken by list order (user rules come first in *rules*).
    """
    merchant_upper = merchant.upper()
    best: MerchantRule | None = None
    for rule in rules:
        if rule.pattern.upper() in merchant_upper:
            if best is None or len(rule.pattern) > len(best.pattern):
                best = rule
    return best
