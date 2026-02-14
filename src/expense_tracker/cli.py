"""Click CLI entry point for the expense command."""

import click

from expense_tracker import __version__


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
    raise SystemExit("Not yet implemented.")


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
    raise SystemExit("Not yet implemented.")


@cli.command()
@click.option(
    "--dir", "target_dir", default=".", type=click.Path(), help="Directory to initialize."
)
def init(target_dir: str) -> None:
    """Initialize a new data directory with the standard structure."""
    raise SystemExit("Not yet implemented.")
