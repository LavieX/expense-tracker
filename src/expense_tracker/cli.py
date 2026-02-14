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
        from expense_tracker.config import load_categories, load_config, load_rules

        config = load_config(root)
        categories = load_categories(root)
        rules = load_rules(root)
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

    # Run the pipeline (stages 1-5: parse, filter, dedup, transfers, enrich, rule-categorize)
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
