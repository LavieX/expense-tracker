"""Click CLI entry point for the expense command.

Handles argument parsing, config loading, and error display. All business
logic is delegated to ``pipeline``, ``categorizer``, ``config``, and
``export`` modules.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import click

from expense_tracker import __version__


def _validate_month(month: str) -> str:
    """Validate that *month* matches ``YYYY-MM`` and represents a real month.

    Returns the validated month string, or raises ``click.BadParameter``.
    """
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise click.BadParameter(
            f"Invalid month format: {month!r}. Expected YYYY-MM (e.g. 2026-01)."
        )
    _, mon = month.split("-")
    mon_int = int(mon)
    if mon_int < 1 or mon_int > 12:
        raise click.BadParameter(
            f"Invalid month: {month!r}. Month must be between 01 and 12."
        )
    return month


def _configure_logging(verbose: bool, debug: bool) -> None:
    """Set up logging based on verbosity flags."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        force=True,
    )


@click.group()
@click.version_option(version=__version__, prog_name="expense-tracker")
def cli() -> None:
    """CLI tool for multi-account household expense tracking and categorization."""


@cli.command()
@click.option("--month", required=True, help="Target month in YYYY-MM format.")
@click.option("--no-llm", is_flag=True, default=False, help="Skip LLM categorization.")
@click.option("--verbose", is_flag=True, default=False, help="Detailed progress output.")
@click.option("--debug", is_flag=True, default=False, help="Developer-level diagnostics.")
def process(month: str, no_llm: bool, verbose: bool, debug: bool) -> None:
    """Run the full processing pipeline for a given month."""
    _configure_logging(verbose, debug)

    try:
        month = _validate_month(month)
    except click.BadParameter as exc:
        click.echo(f"Error: {exc.format_message()}", err=True)
        sys.exit(1)

    root = Path.cwd()

    # Load configuration
    try:
        from expense_tracker.config import (
            load_categories,
            load_config,
            load_exclude_patterns,
            load_rules,
        )

        config = load_config(root)
        categories = load_categories(root)
        rules = load_rules(root)
        exclude_patterns = load_exclude_patterns(root)
    except FileNotFoundError as exc:
        click.echo(
            f"Error: {exc}. Run 'expense init' to create the project structure.",
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error loading configuration: {exc}", err=True)
        sys.exit(1)

    # Select LLM adapter
    from expense_tracker.categorizer import categorize
    from expense_tracker.llm import AnthropicAdapter, NullAdapter

    if no_llm or config.llm_provider == "none":
        llm_adapter = NullAdapter()
        if verbose:
            click.echo("LLM categorization disabled.")
    else:
        llm_adapter = AnthropicAdapter(
            model=config.llm_model,
            api_key_env=config.llm_api_key_env,
        )
        if verbose:
            click.echo(f"Using LLM: {config.llm_provider} ({config.llm_model})")

    # Run the pipeline (stages 1-5: parse, filter, exclude, dedup, transfers, enrich, rule-categorize)
    from expense_tracker.pipeline import run

    if verbose:
        click.echo(f"Processing month: {month}")

    try:
        pipeline_result = run(
            month=month,
            config=config,
            categories=categories,
            rules=rules,
            root=root,
            exclude_patterns=exclude_patterns,
        )
    except Exception as exc:
        click.echo(f"Error running pipeline: {exc}", err=True)
        sys.exit(1)

    # Run LLM categorization on remaining uncategorized transactions
    try:
        cat_result = categorize(
            transactions=pipeline_result.transactions,
            rules=rules,
            categories=categories,
            llm_adapter=llm_adapter,
        )
        pipeline_result.transactions = cat_result.transactions
        pipeline_result.warnings.extend(cat_result.warnings)
        pipeline_result.errors.extend(cat_result.errors)
    except Exception as exc:
        click.echo(f"Warning: LLM categorization failed: {exc}", err=True)

    # Export results
    from expense_tracker.export import export, print_summary

    try:
        output_path = export(
            transactions=pipeline_result.transactions,
            output_dir=root / config.output_dir,
            month=month,
        )
        if verbose:
            click.echo(f"Wrote output to {output_path}")
    except Exception as exc:
        click.echo(f"Error writing output: {exc}", err=True)
        sys.exit(1)

    # Print summary
    print_summary(pipeline_result, month)


@cli.command()
@click.option(
    "--original", required=True, type=click.Path(exists=True), help="Original output CSV."
)
@click.option(
    "--corrected", required=True, type=click.Path(exists=True), help="User-corrected CSV."
)
@click.option("--verbose", is_flag=True, default=False, help="Show details of each learned rule.")
def learn(original: str, corrected: str, verbose: bool) -> None:
    """Compare original and corrected CSVs to learn new categorization rules."""
    _configure_logging(verbose, debug=False)
    root = Path.cwd()

    # Load rules
    try:
        from expense_tracker.config import load_rules, save_learned_rules

        rules = load_rules(root)
    except FileNotFoundError as exc:
        click.echo(
            f"Error: {exc}. Run 'expense init' to create the project structure.",
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error loading rules: {exc}", err=True)
        sys.exit(1)

    # Run the learn workflow
    from expense_tracker.categorizer import learn as categorizer_learn

    try:
        result = categorizer_learn(
            original_path=Path(original),
            corrected_path=Path(corrected),
            rules=rules,
        )
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except KeyError as exc:
        click.echo(
            f"Error: CSV file is missing required column: {exc}",
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error during learning: {exc}", err=True)
        sys.exit(1)

    # Save the updated learned rules
    try:
        save_learned_rules(root, result.rules)
    except Exception as exc:
        click.echo(f"Error saving learned rules: {exc}", err=True)
        sys.exit(1)

    # Print summary
    click.echo()
    click.echo("== Learn Summary ==")
    click.echo(f"  New rules added:    {result.added}")
    click.echo(f"  Rules updated:      {result.updated}")
    click.echo(f"  Conflicts skipped:  {result.skipped}")

    if verbose and (result.added > 0 or result.updated > 0):
        click.echo()
        learned = [r for r in result.rules if r.source == "learned"]
        click.echo("Learned rules:")
        for rule in learned:
            value = f"{rule.category}:{rule.subcategory}" if rule.subcategory else rule.category
            click.echo(f'  "{rule.pattern}" -> {value}')

    click.echo()


@cli.command()
@click.option("--month", required=True, help="Target month in YYYY-MM format.")
@click.option(
    "--source",
    required=True,
    type=click.Choice(["amazon", "target"], case_sensitive=False),
    help="Enrichment source to scrape (amazon or target).",
)
@click.option("--headless", is_flag=True, default=False, help="Run browser in headless mode.")
@click.option("--verbose", is_flag=True, default=False, help="Detailed progress output.")
@click.option("--debug", is_flag=True, default=False, help="Developer-level diagnostics.")
def enrich(month: str, source: str, headless: bool, verbose: bool, debug: bool) -> None:
    """Scrape retailer order history and match to bank transactions."""
    _configure_logging(verbose, debug)

    try:
        month = _validate_month(month)
    except click.BadParameter as exc:
        click.echo(f"Error: {exc.format_message()}", err=True)
        sys.exit(1)

    root = Path.cwd()

    # Load configuration
    try:
        from expense_tracker.config import load_config

        config = load_config(root)
    except FileNotFoundError as exc:
        click.echo(
            f"Error: {exc}. Run 'expense init' to create the project structure.",
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error loading configuration: {exc}", err=True)
        sys.exit(1)

    cache_dir = root / config.enrichment_cache_dir

    # Build transaction list from pipeline parse stage for matching
    from expense_tracker.pipeline import run as pipeline_run

    try:
        from expense_tracker.config import (
            load_categories,
            load_exclude_patterns,
            load_rules,
        )

        categories = load_categories(root)
        rules = load_rules(root)
        exclude_patterns = load_exclude_patterns(root)
        pipeline_result = pipeline_run(
            month=month,
            config=config,
            categories=categories,
            rules=rules,
            root=root,
            exclude_patterns=exclude_patterns,
        )
        # Convert Transaction objects to dicts for the enrichment provider
        txn_dicts = [
            {
                "transaction_id": t.transaction_id,
                "date": t.date.isoformat(),
                "amount": str(t.amount),
                "merchant": t.merchant,
            }
            for t in pipeline_result.transactions
            if not t.is_transfer
        ]
    except Exception as exc:
        click.echo(f"Error loading transactions: {exc}", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"Found {len(txn_dicts)} non-transfer transactions for {month}")

    source_lower = source.lower()

    if source_lower == "target":
        from expense_tracker.enrichment.target import enrich_target

        try:
            result = enrich_target(
                month=month,
                transactions=txn_dicts,
                cache_dir=cache_dir,
                headless=headless,
            )
        except ImportError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        except Exception as exc:
            click.echo(f"Error during Target enrichment: {exc}", err=True)
            sys.exit(1)

    elif source_lower == "amazon":
        from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider
        from expense_tracker.models import AmazonAccountConfig

        provider = AmazonEnrichmentProvider()

        # Convert ISO date strings back to date objects and amounts to Decimal
        # for the matching algorithm.
        from datetime import date as date_cls
        from decimal import Decimal as Dec

        normalized_txns = []
        for txn in txn_dicts:
            d = txn["date"]
            if isinstance(d, str):
                d = date_cls.fromisoformat(d)
            a = txn["amount"]
            if isinstance(a, str):
                a = Dec(a)
            normalized_txns.append({
                **txn,
                "date": d,
                "amount": a,
            })

        # Determine Amazon accounts from config.
        # If no [[enrichment.amazon]] sections exist, fall back to a single
        # default account for backward compatibility.
        amazon_accounts = config.amazon_accounts
        if not amazon_accounts:
            amazon_accounts = [AmazonAccountConfig(label="default")]

        if verbose:
            labels = [a.label for a in amazon_accounts]
            click.echo(f"Amazon accounts to scrape: {', '.join(labels)}")

        try:
            enrich_result = provider.enrich_multi_account(
                month=month,
                root=root,
                amazon_accounts=amazon_accounts,
                transactions=normalized_txns,
            )
        except ImportError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        except Exception as exc:
            click.echo(f"Error during Amazon enrichment: {exc}", err=True)
            sys.exit(1)

        # Print summary
        click.echo()
        click.echo("== Amazon Enrichment Summary ==")

        # Show per-account breakdown if there are multiple accounts.
        if enrich_result.account_stats and len(enrich_result.account_stats) > 1:
            for stat in enrich_result.account_stats:
                click.echo(
                    f'  Account "{stat.label}": '
                    f"{stat.orders_found} orders found, "
                    f"{stat.orders_matched} matched"
                )
            click.echo(
                f"  Total: {enrich_result.orders_found} orders, "
                f"{enrich_result.orders_matched} matched, "
                f"{enrich_result.orders_unmatched} unmatched"
            )
        else:
            click.echo(f"  Orders found:         {enrich_result.orders_found}")
            click.echo(f"  Orders matched:       {enrich_result.orders_matched}")
            click.echo(f"  Orders unmatched:     {enrich_result.orders_unmatched}")

        click.echo(f"  Cache files written:  {enrich_result.cache_files_written}")

        if enrich_result.unmatched_details:
            click.echo()
            click.echo("Unmatched orders (review manually):")
            for detail in enrich_result.unmatched_details:
                click.echo(f"  - {detail}")

        if enrich_result.errors:
            click.echo()
            for err in enrich_result.errors:
                click.echo(f"Error: {err}", err=True)

        click.echo()
        return

    else:
        click.echo(f"Error: Unknown source: {source!r}", err=True)
        sys.exit(1)

    # Print summary (for Target and other dict-returning providers)
    click.echo()
    click.echo(f"== {source.title()} Enrichment Summary ==")
    click.echo(f"  Orders scraped:       {result['orders_scraped']}")
    click.echo(f"  Orders matched:       {result['orders_matched']}")
    click.echo(f"  Cache files written:  {result['cache_files_written']}")
    if result.get("skipped_gift_card", 0) > 0:
        click.echo(f"  Gift card (skipped):  {result['skipped_gift_card']}")
    click.echo()


def _read_csv_transactions(csv_path: Path) -> list:
    """Read a CSV file and return a list of Transaction objects.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of Transaction objects parsed from the CSV.

    Raises:
        KeyError: If a required column is missing.
        Exception: For other CSV parsing errors.
    """
    import csv as csv_mod
    from datetime import date as date_cls
    from decimal import Decimal

    from expense_tracker.models import Transaction

    transactions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            txn = Transaction(
                transaction_id=row["transaction_id"],
                date=date_cls.fromisoformat(row["date"]),
                merchant=row["merchant"],
                description=row["description"],
                amount=Decimal(row["amount"]),
                institution=row["institution"],
                account=row["account"],
                category=row["category"],
                subcategory=row["subcategory"],
                is_return=row["is_return"].lower() == "true",
                is_recurring=row.get("is_recurring", "false").lower() == "true",
                split_from=row["split_from"],
            )
            transactions.append(txn)
    return transactions


def _find_month_csvs(output_dir: Path) -> list[Path]:
    """Find all month CSV files in the output directory.

    Matches files named ``YYYY-MM.csv`` (e.g. ``2025-07.csv``) and excludes
    files with suffixes like ``-corrected`` (e.g. ``2025-07-corrected.csv``).

    Args:
        output_dir: Directory to search for CSV files.

    Returns:
        Sorted list of Path objects for matching CSV files.
    """
    month_csv_pattern = re.compile(r"^\d{4}-\d{2}\.csv$")
    csvs = [
        p for p in output_dir.iterdir()
        if p.is_file() and month_csv_pattern.match(p.name)
    ]
    return sorted(csvs)


@cli.command()
@click.option("--month", default=None, help="Target month in YYYY-MM format.")
@click.option("--all", "push_all", is_flag=True, default=False, help="Push all months.")
@click.option("--verbose", is_flag=True, default=False, help="Detailed progress output.")
def push(month: str | None, push_all: bool, verbose: bool) -> None:
    """Push processed transaction data to Google Sheets.

    By default (no flags), pushes ALL month CSVs found in the output directory.
    With --month, pushes the specified month combined with all other existing
    months so the sheet always has the full picture.
    """
    _configure_logging(verbose, debug=False)

    if month is not None:
        try:
            month = _validate_month(month)
        except click.BadParameter as exc:
            click.echo(f"Error: {exc.format_message()}", err=True)
            sys.exit(1)

    if month is not None and push_all:
        click.echo("Error: --month and --all are mutually exclusive.", err=True)
        sys.exit(1)

    root = Path.cwd()

    # Load configuration
    try:
        from expense_tracker.config import load_config

        config = load_config(root)
    except FileNotFoundError as exc:
        click.echo(
            f"Error: {exc}. Run 'expense init' to create the project structure.",
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error loading configuration: {exc}", err=True)
        sys.exit(1)

    # Check if sheets config exists
    if config.sheets is None:
        click.echo(
            "Error: Google Sheets is not configured. Add a [sheets] section to config.toml.",
            err=True,
        )
        sys.exit(1)

    if not config.sheets.spreadsheet_id:
        click.echo(
            "Error: spreadsheet_id is not set in the [sheets] section of config.toml.",
            err=True,
        )
        sys.exit(1)

    output_dir = root / config.output_dir

    # Determine which CSV files to read
    if month is not None:
        # --month provided: ensure the requested month exists, then also
        # gather all other month CSVs so nothing is lost when the sheet
        # is cleared and rewritten.
        target_csv = output_dir / f"{month}.csv"
        if not target_csv.exists():
            click.echo(
                f"Error: Output file not found: {target_csv}\n"
                f"Run 'expense process --month {month}' first.",
                err=True,
            )
            sys.exit(1)

        csv_files = _find_month_csvs(output_dir)
        # Ensure the target is included (it should be, but be explicit)
        if target_csv not in csv_files:
            csv_files.append(target_csv)
            csv_files.sort()
    else:
        # --all or default: read everything
        csv_files = _find_month_csvs(output_dir)

    if not csv_files:
        click.echo(
            f"Error: No month CSV files found in {output_dir}.\n"
            "Run 'expense process --month YYYY-MM' first.",
            err=True,
        )
        sys.exit(1)

    if verbose:
        click.echo(f"Found {len(csv_files)} month file(s):")
        for f in csv_files:
            click.echo(f"  {f.name}")

    # Read and combine transactions from all CSV files
    transactions = []
    for csv_path in csv_files:
        try:
            file_txns = _read_csv_transactions(csv_path)
            if verbose:
                click.echo(f"  {csv_path.name}: {len(file_txns)} transactions")
            transactions.extend(file_txns)
        except KeyError as exc:
            click.echo(
                f"Error: CSV file {csv_path.name} is missing required column: {exc}",
                err=True,
            )
            sys.exit(1)
        except Exception as exc:
            click.echo(f"Error reading {csv_path.name}: {exc}", err=True)
            sys.exit(1)

    # Sort all transactions by date
    transactions.sort(key=lambda t: (t.date, t.institution, t.amount))

    if verbose:
        click.echo(f"Total: {len(transactions)} transactions across {len(csv_files)} month(s)")

    # Push to Google Sheets
    try:
        from expense_tracker.sheets import push_to_sheets

        rows_written = push_to_sheets(
            transactions=transactions,
            config=config.sheets,
            root=root,
        )
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error pushing to Google Sheets: {exc}", err=True)
        sys.exit(1)

    # Print summary
    sheet_url = f"https://docs.google.com/spreadsheets/d/{config.sheets.spreadsheet_id}"
    click.echo()
    click.echo(f"Successfully pushed {rows_written} transactions to Google Sheets")
    click.echo(f"  Months: {', '.join(f.stem for f in csv_files)}")
    click.echo(f"  Worksheet: {config.sheets.worksheet_name}")
    click.echo(f"  URL: {sheet_url}")
    click.echo()


@cli.command()
@click.option(
    "--dir", "target_dir", default=".", type=click.Path(), help="Directory to initialize."
)
def init(target_dir: str) -> None:
    """Initialize a new data directory with the standard structure."""
    from expense_tracker.config import initialize

    target = Path(target_dir).resolve()

    try:
        initialize(target)
    except Exception as exc:
        click.echo(f"Error initializing project: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Initialized expense tracker project in {target}")
