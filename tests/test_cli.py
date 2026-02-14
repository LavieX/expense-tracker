"""Tests for the Click CLI layer.

Uses Click's CliRunner to invoke commands without spawning subprocesses.
Mocks the business logic modules to isolate CLI behavior (argument parsing,
error handling, output formatting) from pipeline internals.
"""

from __future__ import annotations

import csv
import shutil
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from expense_tracker.cli import cli
from expense_tracker.models import (
    AccountConfig,
    AppConfig,
    LearnResult,
    MerchantRule,
    PipelineResult,
    StageResult,
    Transaction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_CONFIG_DIR = Path(__file__).parent / "fixtures" / "config"


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli_project_dir(tmp_path: Path) -> Path:
    """Temporary directory with valid config files for CLI tests."""
    project = tmp_path / "project"
    project.mkdir()

    for config_file in ("config.toml", "categories.toml", "rules.toml"):
        shutil.copy2(FIXTURES_CONFIG_DIR / config_file, project / config_file)

    (project / "input" / "chase").mkdir(parents=True)
    (project / "input" / "capital-one").mkdir(parents=True)
    (project / "input" / "elevations").mkdir(parents=True)
    (project / "output").mkdir()
    (project / "enrichment-cache").mkdir()

    return project


def _make_app_config() -> AppConfig:
    """Build a minimal AppConfig for testing."""
    return AppConfig(
        accounts=[
            AccountConfig(
                name="Chase Credit Card",
                institution="chase",
                parser="chase",
                account_type="credit_card",
                input_dir="input/chase",
            ),
        ],
        output_dir="output",
        enrichment_cache_dir="enrichment-cache",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        llm_api_key_env="ANTHROPIC_API_KEY",
    )


def _make_pipeline_result(n_transactions: int = 3) -> PipelineResult:
    """Build a minimal PipelineResult with n transactions."""
    txns = []
    for i in range(n_transactions):
        txns.append(
            Transaction(
                transaction_id=f"abc{i:09d}",
                date=date(2026, 1, 15 + i),
                merchant=f"MERCHANT_{i}",
                description=f"Description {i}",
                amount=Decimal(f"-{10 + i}.00"),
                institution="chase",
                account="Chase Credit Card",
                category="Food & Dining" if i % 2 == 0 else "Uncategorized",
                subcategory="Groceries" if i % 2 == 0 else "",
            )
        )
    return PipelineResult(transactions=txns)


def _make_categories() -> list[dict]:
    """Build a minimal category taxonomy."""
    return [
        {"name": "Food & Dining", "subcategories": ["Groceries", "Restaurant"]},
        {"name": "Shopping", "subcategories": ["Electronics"]},
    ]


def _make_rules() -> list[MerchantRule]:
    """Build a minimal rules list."""
    return [
        MerchantRule(
            pattern="CHIPOTLE",
            category="Food & Dining",
            subcategory="Fast Food",
            source="user",
        ),
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as a CSV file."""
    if not rows:
        path.write_text("transaction_id,date,merchant,category,subcategory\n")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# expense --help
# ===========================================================================


class TestCLIHelp:
    """Verify top-level and subcommand help output."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "expense tracking" in output_lower or "categorization" in output_lower
        assert "process" in result.output
        assert "learn" in result.output
        assert "init" in result.output

    def test_process_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--month" in result.output
        assert "--no-llm" in result.output
        assert "--verbose" in result.output
        assert "--debug" in result.output

    def test_learn_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["learn", "--help"])
        assert result.exit_code == 0
        assert "--original" in result.output
        assert "--corrected" in result.output
        assert "--verbose" in result.output

    def test_init_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "--dir" in result.output

    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "expense-tracker" in result.output


# ===========================================================================
# expense process
# ===========================================================================


class TestProcessCommand:
    """Tests for the `expense process` command."""

    def test_month_validation_invalid_format(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["process", "--month", "2026-1"])
        assert result.exit_code != 0
        assert "YYYY-MM" in result.output or "Invalid" in result.output

    def test_month_validation_invalid_month_number(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["process", "--month", "2026-13"])
        assert result.exit_code != 0
        assert "Invalid" in result.output

    def test_month_validation_zero_month(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["process", "--month", "2026-00"])
        assert result.exit_code != 0
        assert "Invalid" in result.output

    def test_month_validation_not_numeric(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["process", "--month", "abcd-ef"])
        assert result.exit_code != 0

    @patch("expense_tracker.config.load_config")
    def test_missing_config_file(
        self,
        mock_load_config: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Process should fail gracefully when config.toml is missing."""
        mock_load_config.side_effect = FileNotFoundError(
            "[Errno 2] No such file or directory: 'config.toml'"
        )
        result = runner.invoke(
            cli, ["process", "--month", "2026-01"],
            catch_exceptions=False,
        )
        # Should mention running 'expense init' or show a file-not-found error
        assert result.exit_code != 0
        # The error message should be user-facing, not a traceback
        assert "Error" in result.output
        assert "expense init" in result.output

    @patch("expense_tracker.export.print_summary")
    @patch("expense_tracker.export.export")
    @patch("expense_tracker.categorizer.categorize")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_success(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_categorize: MagicMock,
        mock_export: MagicMock,
        mock_print_summary: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Successful process invocation calls pipeline, categorize, export, summary."""
        mock_load_config.return_value = _make_app_config()
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()

        pipeline_result = _make_pipeline_result()
        mock_pipeline_run.return_value = pipeline_result
        mock_categorize.return_value = StageResult(
            transactions=pipeline_result.transactions
        )
        mock_export.return_value = Path("output/2026-01.csv")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01"],
            catch_exceptions=False,
        )

        # CLI should succeed
        assert result.exit_code == 0, result.output

        # All components should have been called
        mock_load_config.assert_called_once()
        mock_load_categories.assert_called_once()
        mock_load_rules.assert_called_once()
        mock_pipeline_run.assert_called_once()
        mock_categorize.assert_called_once()
        mock_export.assert_called_once()
        mock_print_summary.assert_called_once()

        # Verify pipeline.run was called with the right month
        call_kwargs = mock_pipeline_run.call_args
        assert call_kwargs.kwargs.get("month") == "2026-01"

    @patch("expense_tracker.export.print_summary")
    @patch("expense_tracker.export.export")
    @patch("expense_tracker.categorizer.categorize")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_no_llm_flag(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_categorize: MagicMock,
        mock_export: MagicMock,
        mock_print_summary: MagicMock,
        runner: CliRunner,
    ) -> None:
        """--no-llm flag should use NullAdapter."""
        mock_load_config.return_value = _make_app_config()
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()

        pipeline_result = _make_pipeline_result()
        mock_pipeline_run.return_value = pipeline_result
        mock_categorize.return_value = StageResult(
            transactions=pipeline_result.transactions
        )
        mock_export.return_value = Path("output/2026-01.csv")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        # categorize should have been called with a NullAdapter
        mock_categorize.assert_called_once()
        call_kwargs = mock_categorize.call_args
        from expense_tracker.llm import NullAdapter
        adapter = call_kwargs.kwargs.get("llm_adapter")
        assert isinstance(adapter, NullAdapter)

    @patch("expense_tracker.export.print_summary")
    @patch("expense_tracker.export.export")
    @patch("expense_tracker.categorizer.categorize")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_llm_provider_none_in_config(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_categorize: MagicMock,
        mock_export: MagicMock,
        mock_print_summary: MagicMock,
        runner: CliRunner,
    ) -> None:
        """llm_provider='none' in config should use NullAdapter even without --no-llm."""
        config = _make_app_config()
        config.llm_provider = "none"
        mock_load_config.return_value = config
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()

        pipeline_result = _make_pipeline_result()
        mock_pipeline_run.return_value = pipeline_result
        mock_categorize.return_value = StageResult(
            transactions=pipeline_result.transactions
        )
        mock_export.return_value = Path("output/2026-01.csv")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        from expense_tracker.llm import NullAdapter
        call_kwargs = mock_categorize.call_args
        adapter = call_kwargs.kwargs.get("llm_adapter")
        assert isinstance(adapter, NullAdapter)

    @patch("expense_tracker.export.print_summary")
    @patch("expense_tracker.export.export")
    @patch("expense_tracker.categorizer.categorize")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_anthropic_adapter_when_llm_enabled(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_categorize: MagicMock,
        mock_export: MagicMock,
        mock_print_summary: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Default config (anthropic provider) should use AnthropicAdapter."""
        mock_load_config.return_value = _make_app_config()
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()

        pipeline_result = _make_pipeline_result()
        mock_pipeline_run.return_value = pipeline_result
        mock_categorize.return_value = StageResult(
            transactions=pipeline_result.transactions
        )
        mock_export.return_value = Path("output/2026-01.csv")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        from expense_tracker.llm import AnthropicAdapter
        call_kwargs = mock_categorize.call_args
        adapter = call_kwargs.kwargs.get("llm_adapter")
        assert isinstance(adapter, AnthropicAdapter)

    @patch("expense_tracker.export.print_summary")
    @patch("expense_tracker.export.export")
    @patch("expense_tracker.categorizer.categorize")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_verbose_output(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_categorize: MagicMock,
        mock_export: MagicMock,
        mock_print_summary: MagicMock,
        runner: CliRunner,
    ) -> None:
        """--verbose flag should produce progress messages."""
        mock_load_config.return_value = _make_app_config()
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()

        pipeline_result = _make_pipeline_result()
        mock_pipeline_run.return_value = pipeline_result
        mock_categorize.return_value = StageResult(
            transactions=pipeline_result.transactions
        )
        mock_export.return_value = Path("output/2026-01.csv")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm", "--verbose"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        # Should see progress information
        assert "Processing month" in result.output or "LLM" in result.output

    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_pipeline_error(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Pipeline exceptions should be caught and displayed as user-facing errors."""
        mock_load_config.return_value = _make_app_config()
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()
        mock_pipeline_run.side_effect = RuntimeError("disk full")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01"],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "disk full" in result.output

    @patch("expense_tracker.export.print_summary")
    @patch("expense_tracker.export.export")
    @patch("expense_tracker.categorizer.categorize")
    @patch("expense_tracker.pipeline.run")
    @patch("expense_tracker.config.load_rules")
    @patch("expense_tracker.config.load_categories")
    @patch("expense_tracker.config.load_config")
    def test_process_export_error(
        self,
        mock_load_config: MagicMock,
        mock_load_categories: MagicMock,
        mock_load_rules: MagicMock,
        mock_pipeline_run: MagicMock,
        mock_categorize: MagicMock,
        mock_export: MagicMock,
        mock_print_summary: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Export failure should be caught and shown as a user-facing error."""
        mock_load_config.return_value = _make_app_config()
        mock_load_categories.return_value = _make_categories()
        mock_load_rules.return_value = _make_rules()

        pipeline_result = _make_pipeline_result()
        mock_pipeline_run.return_value = pipeline_result
        mock_categorize.return_value = StageResult(
            transactions=pipeline_result.transactions
        )
        mock_export.side_effect = PermissionError("output dir not writable")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01"],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "output" in result.output.lower() or "writable" in result.output.lower()

    def test_process_missing_month_option(self, runner: CliRunner) -> None:
        """Process without --month should fail with Click's missing option error."""
        result = runner.invoke(cli, ["process"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--month" in result.output


# ===========================================================================
# expense learn
# ===========================================================================


class TestLearnCommand:
    """Tests for the `expense learn` command."""

    def test_learn_missing_original(self, runner: CliRunner) -> None:
        """Should fail when --original file does not exist."""
        result = runner.invoke(
            cli,
            ["learn", "--original", "/nonexistent.csv", "--corrected", "/also-nonexistent.csv"],
        )
        assert result.exit_code != 0

    @patch("expense_tracker.config.save_learned_rules")
    @patch("expense_tracker.categorizer.learn")
    @patch("expense_tracker.config.load_rules")
    def test_learn_success(
        self,
        mock_load_rules: MagicMock,
        mock_learn: MagicMock,
        mock_save: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Successful learn invocation calls learn, saves rules, prints summary."""
        # Create original and corrected CSV files
        original_path = tmp_path / "original.csv"
        corrected_path = tmp_path / "corrected.csv"

        rows = [
            {
                "transaction_id": "abc000000000",
                "date": "2026-01-15",
                "merchant": "NEW STORE",
                "category": "Uncategorized",
                "subcategory": "",
            },
        ]
        _write_csv(original_path, rows)

        corrected_rows = [
            {
                "transaction_id": "abc000000000",
                "date": "2026-01-15",
                "merchant": "NEW STORE",
                "category": "Shopping",
                "subcategory": "Electronics",
            },
        ]
        _write_csv(corrected_path, corrected_rows)

        rules = [
            MerchantRule(
                pattern="CHIPOTLE",
                category="Food & Dining",
                subcategory="Fast Food",
                source="user",
            ),
        ]
        mock_load_rules.return_value = rules

        learn_result = LearnResult(added=1, updated=0, skipped=0, rules=rules)
        mock_learn.return_value = learn_result

        result = runner.invoke(
            cli,
            ["learn", "--original", str(original_path), "--corrected", str(corrected_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Learn Summary" in result.output
        assert "New rules added" in result.output
        assert "1" in result.output

        mock_learn.assert_called_once()
        mock_save.assert_called_once()

    @patch("expense_tracker.config.save_learned_rules")
    @patch("expense_tracker.categorizer.learn")
    @patch("expense_tracker.config.load_rules")
    def test_learn_verbose(
        self,
        mock_load_rules: MagicMock,
        mock_learn: MagicMock,
        mock_save: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--verbose should show learned rule details."""
        original_path = tmp_path / "original.csv"
        corrected_path = tmp_path / "corrected.csv"
        _write_csv(original_path, [{"transaction_id": "x", "category": "A", "subcategory": ""}])
        _write_csv(corrected_path, [{"transaction_id": "x", "category": "B", "subcategory": ""}])

        learned_rule = MerchantRule(
            pattern="SOME STORE", category="Shopping", subcategory="Electronics", source="learned"
        )
        rules = [learned_rule]
        mock_load_rules.return_value = []
        mock_learn.return_value = LearnResult(added=1, updated=0, skipped=0, rules=rules)

        result = runner.invoke(
            cli,
            [
                "learn",
                "--original", str(original_path),
                "--corrected", str(corrected_path),
                "--verbose",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Learned rules" in result.output
        assert "SOME STORE" in result.output
        assert "Shopping:Electronics" in result.output

    @patch("expense_tracker.config.save_learned_rules")
    @patch("expense_tracker.categorizer.learn")
    @patch("expense_tracker.config.load_rules")
    def test_learn_no_changes(
        self,
        mock_load_rules: MagicMock,
        mock_learn: MagicMock,
        mock_save: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """When there are no differences, summary shows zeros."""
        original_path = tmp_path / "original.csv"
        corrected_path = tmp_path / "corrected.csv"
        _write_csv(original_path, [{"transaction_id": "x", "category": "A", "subcategory": ""}])
        _write_csv(corrected_path, [{"transaction_id": "x", "category": "A", "subcategory": ""}])

        mock_load_rules.return_value = []
        mock_learn.return_value = LearnResult(added=0, updated=0, skipped=0, rules=[])

        result = runner.invoke(
            cli,
            ["learn", "--original", str(original_path), "--corrected", str(corrected_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Learn Summary" in result.output

    @patch("expense_tracker.config.load_rules")
    def test_learn_missing_rules_toml(
        self,
        mock_load_rules: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Learn should fail gracefully when rules.toml is missing."""
        mock_load_rules.side_effect = FileNotFoundError("rules.toml not found")

        original_path = tmp_path / "original.csv"
        corrected_path = tmp_path / "corrected.csv"
        _write_csv(original_path, [])
        _write_csv(corrected_path, [])

        result = runner.invoke(
            cli,
            ["learn", "--original", str(original_path), "--corrected", str(corrected_path)],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("expense_tracker.categorizer.learn")
    @patch("expense_tracker.config.load_rules")
    def test_learn_csv_missing_column(
        self,
        mock_load_rules: MagicMock,
        mock_learn: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Learn should show a clear error when CSV is missing transaction_id column."""
        original_path = tmp_path / "original.csv"
        corrected_path = tmp_path / "corrected.csv"
        _write_csv(original_path, [{"transaction_id": "x", "category": "A", "subcategory": ""}])
        _write_csv(corrected_path, [{"transaction_id": "x", "category": "A", "subcategory": ""}])

        mock_load_rules.return_value = []
        mock_learn.side_effect = KeyError("transaction_id")

        result = runner.invoke(
            cli,
            ["learn", "--original", str(original_path), "--corrected", str(corrected_path)],
            catch_exceptions=False,
        )

        assert result.exit_code != 0
        assert "Error" in result.output
        assert "transaction_id" in result.output

    def test_learn_missing_required_options(self, runner: CliRunner) -> None:
        """Learn without required options should fail."""
        result = runner.invoke(cli, ["learn"])
        assert result.exit_code != 0


# ===========================================================================
# expense init
# ===========================================================================


class TestInitCommand:
    """Tests for the `expense init` command."""

    def test_init_creates_structure(self, runner: CliRunner, tmp_path: Path) -> None:
        """Init should create the standard project structure."""
        target = tmp_path / "new-project"

        result = runner.invoke(
            cli,
            ["init", "--dir", str(target)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Initialized" in result.output

        # Verify files and directories were created
        assert (target / "config.toml").is_file()
        assert (target / "categories.toml").is_file()
        assert (target / "rules.toml").is_file()
        assert (target / "input" / "chase").is_dir()
        assert (target / "input" / "capital-one").is_dir()
        assert (target / "input" / "elevations").is_dir()
        assert (target / "output").is_dir()
        assert (target / "enrichment-cache").is_dir()

    def test_init_idempotent(self, runner: CliRunner, tmp_path: Path) -> None:
        """Running init twice should not fail or overwrite existing files."""
        target = tmp_path / "idempotent-project"

        # First run
        result1 = runner.invoke(cli, ["init", "--dir", str(target)], catch_exceptions=False)
        assert result1.exit_code == 0

        # Write custom content to a config file
        config_path = target / "config.toml"
        original_content = config_path.read_text()
        custom_content = original_content + "\n# custom comment\n"
        config_path.write_text(custom_content)

        # Second run
        result2 = runner.invoke(cli, ["init", "--dir", str(target)], catch_exceptions=False)
        assert result2.exit_code == 0

        # Config file should not have been overwritten
        assert config_path.read_text() == custom_content

    def test_init_default_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        """Init without --dir should use current directory."""
        result = runner.invoke(
            cli,
            ["init"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Initialized" in result.output

    def test_init_output_includes_path(self, runner: CliRunner, tmp_path: Path) -> None:
        """Init should print the resolved path of the initialized directory."""
        target = tmp_path / "show-path"
        result = runner.invoke(
            cli,
            ["init", "--dir", str(target)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert str(target) in result.output


# ===========================================================================
# Month validation unit tests
# ===========================================================================


class TestMonthValidation:
    """Unit tests for _validate_month helper."""

    def test_valid_months(self) -> None:
        from expense_tracker.cli import _validate_month

        assert _validate_month("2026-01") == "2026-01"
        assert _validate_month("2026-12") == "2026-12"
        assert _validate_month("2025-06") == "2025-06"

    def test_invalid_format(self) -> None:
        from click import BadParameter

        from expense_tracker.cli import _validate_month

        with pytest.raises(BadParameter):
            _validate_month("2026-1")
        with pytest.raises(BadParameter):
            _validate_month("26-01")
        with pytest.raises(BadParameter):
            _validate_month("2026/01")
        with pytest.raises(BadParameter):
            _validate_month("january")

    def test_invalid_month_number(self) -> None:
        from click import BadParameter

        from expense_tracker.cli import _validate_month

        with pytest.raises(BadParameter):
            _validate_month("2026-00")
        with pytest.raises(BadParameter):
            _validate_month("2026-13")
