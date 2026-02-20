"""End-to-end integration tests for the full Expense Tracker pipeline.

Exercises the complete workflow via the CLI (CliRunner):
  init -> place fixture CSVs -> process -> verify output -> learn -> process again

Test scenarios:
(a) Happy path with all three banks, transfers detected, categories applied
(b) LLM fallback with mocked HTTP (uncategorized transactions get LLM suggestions)
(c) Partial failure (one bank CSV is malformed, others process successfully)
(d) Enrichment cache present (create a mock enrichment JSON, verify split transactions)
(e) Re-run idempotency (process twice, same output)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from expense_tracker.cli import cli
from expense_tracker.export import CSV_COLUMNS

# ---------------------------------------------------------------------------
# Shared CSV data -- clean data that all three parsers accept
# ---------------------------------------------------------------------------

# Chase credit card: 7 rows spanning Jan and Feb 2026.
# Includes a grocery store, fast food, gas, retail, subscription, a refund,
# and a credit card autopay credit (transfer).
_CHASE_CSV = """\
Transaction Date,Post Date,Description,Category,Type,Amount,Memo
01/15/2026,01/16/2026,WHOLEFDS LMT #10554,Groceries,Sale,-171.48,
01/18/2026,01/19/2026,CHIPOTLE MEXICAN GRIL,Food & Drink,Sale,-12.50,
01/20/2026,01/21/2026,SHELL OIL 574726183,Gas,Sale,-48.75,
01/25/2026,01/26/2026,TARGET        00022186,Groceries,Sale,-127.98,
02/05/2026,02/06/2026,APPLE.COM/BILL,Shopping,Sale,-42.85,
02/08/2026,02/09/2026,AMAZON.COM REFUND,Shopping,Return,25.99,
02/10/2026,02/11/2026,CHASE CREDIT CRD AUTOPAY,Payment,Payment,2150.00,
"""

# Capital One credit card: 6 rows spanning Jan and Feb 2026.
# Includes dining, retail, pharmacy, streaming, and an autopay credit (transfer).
_CAPITAL_ONE_CSV = """\
Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit
2026-01-10,2026-01-11,4218,FIRST WATCH 0307 PAT,Dining,41.78,
2026-01-12,2026-01-13,4218,REI.COM  800-426-4840,Merchandise,160.65,
2026-02-02,2026-02-03,4218,PETCO 1234,Merchandise,45.67,
2026-02-05,2026-02-06,4218,CVS/PHARMACY #08432,Health,18.50,
2026-02-08,2026-02-09,4218,SPOTIFY USA,Entertainment,10.99,
2026-02-12,2026-02-12,4218,CAPITAL ONE AUTOPAY PYMT,Payment/Credit,,474.90
"""

# Elevations checking: 5 rows spanning Jan and Feb 2026.
# Includes a preschool payment, utility, payroll credit, and two CC
# payment debits (transfers matching Chase and Capital One).
_ELEVATIONS_CSV = """\
"Transaction ID","Posting Date","Effective Date","Transaction Type","Amount","Check Number","Reference Number","Description","Transaction Category","Type","Balance","Memo","Extended Description"
"row0","1/15/2026","1/15/2026","Debit","-445.00000","","1001","PRIMROSE SCHOOL TYPE: 3037741919  ID: 1470259040 CO: PRIMROSE SCHOOL","Child","ACH","10000.00","",""
"row1","1/18/2026","1/18/2026","Debit","-49.95000","","1002","LPC Nextlight TYPE: LPC BB  ID: 4846000608 CO: LPC Nextlight","Utilities","ACH","10000.00","",""
"row2","2/1/2026","2/1/2026","Credit","2254.14000","","1003","VMWARE INC TYPE: PAYROLL  ID: 9111111103 CO: VMWARE INC","Paychecks","ACH","12000.00","",""
"row3","2/5/2026","2/5/2026","Debit","-2150.00000","","1004","CHASE CREDIT CRD TYPE: AUTOPAY  ID: 4760039224 CO: CHASE CREDIT CRD","CC Payment","ACH","10000.00","",""
"row4","2/8/2026","2/8/2026","Debit","-474.90000","","1005","CAPITAL ONE TYPE: CRCARDPMT  ID: 9541719318 CO: CAPITAL ONE  NAME: LAVIE TOBEY","CC Payment","ACH","9500.00","",""
"""

# Malformed Capital One CSV: wrong column headers entirely.
_CAPITAL_ONE_MALFORMED_CSV = """\
Wrong,Column,Headers,Here
some,garbage,data,row
more,garbage,data,row
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_project(root: Path) -> None:
    """Populate a project directory with clean CSV data and test config."""
    fixtures_config = Path(__file__).parent / "fixtures" / "config"

    # Copy config files
    for config_file in ("config.toml", "categories.toml", "rules.toml"):
        shutil.copy2(fixtures_config / config_file, root / config_file)

    # Create input directories with clean CSVs
    chase_dir = root / "input" / "chase"
    chase_dir.mkdir(parents=True)
    (chase_dir / "Activity2026.csv").write_text(_CHASE_CSV, encoding="utf-8")

    cap1_dir = root / "input" / "capital-one"
    cap1_dir.mkdir(parents=True)
    (cap1_dir / "Activity2026.csv").write_text(_CAPITAL_ONE_CSV, encoding="utf-8")

    elev_dir = root / "input" / "elevations"
    elev_dir.mkdir(parents=True)
    (elev_dir / "Activity2026.csv").write_text(_ELEVATIONS_CSV, encoding="utf-8")

    # Create output and enrichment-cache directories
    (root / "output").mkdir()
    (root / "enrichment-cache").mkdir()


def _read_output_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV and return (fieldnames, list-of-row-dicts)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _make_anthropic_response(suggestions: list[dict]) -> dict:
    """Build a mock Anthropic Messages API response body."""
    return {
        "id": "msg_test_integration",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": json.dumps(suggestions),
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 100},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def integration_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fully-populated temporary project directory for integration tests.

    Uses clean CSV data (no malformed rows) so all three banks parse
    successfully. The config has llm.provider = "none" (from test fixtures).

    Importantly, this fixture also changes the working directory to the
    project root, because the CLI uses ``Path.cwd()`` to locate config
    files and input directories.
    """
    project = tmp_path / "integration-project"
    project.mkdir()
    _build_project(project)
    monkeypatch.chdir(project)
    return project


# ===========================================================================
# (a) Happy path with all three banks
# ===========================================================================


class TestHappyPath:
    """End-to-end: init -> place CSVs -> process -> verify output -> learn -> process again."""

    def test_init_creates_project_structure(self, runner: CliRunner, tmp_path: Path) -> None:
        """expense init creates all config files and directories."""
        target = tmp_path / "new-project"

        result = runner.invoke(cli, ["init", "--dir", str(target)])

        assert result.exit_code == 0, result.output
        assert "Initialized" in result.output
        assert (target / "config.toml").is_file()
        assert (target / "categories.toml").is_file()
        assert (target / "rules.toml").is_file()
        assert (target / "input" / "chase").is_dir()
        assert (target / "input" / "capital-one").is_dir()
        assert (target / "input" / "elevations").is_dir()
        assert (target / "output").is_dir()
        assert (target / "enrichment-cache").is_dir()

    def test_full_pipeline_january(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Process January 2026: all three banks parsed, categories applied,
        transfers detected, output CSV written with correct schema."""
        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Processing Summary: 2026-01" in result.output

        # Verify output file exists
        output_path = integration_project / "output" / "2026-01.csv"
        assert output_path.is_file()

        # Verify CSV schema
        fieldnames, rows = _read_output_csv(output_path)
        assert fieldnames == CSV_COLUMNS

        # January transactions from the clean CSVs:
        #   Chase: WHOLEFDS (1/15), CHIPOTLE (1/18), SHELL OIL (1/20), TARGET (1/25) = 4
        #   Capital One: FIRST WATCH (1/10), REI.COM (1/12) = 2
        #   Elevations: PRIMROSE SCHOOL (1/15), LPC Nextlight (1/18) = 2
        # Total: 8 transactions in January, none are transfers
        assert len(rows) == 8, f"Expected 8 rows, got {len(rows)}"

        # Verify all dates are in January 2026
        for row in rows:
            assert row["date"].startswith("2026-01"), f"Unexpected date: {row['date']}"

        # Verify institutions present
        institutions = {row["institution"] for row in rows}
        assert "chase" in institutions
        assert "capital_one" in institutions
        assert "elevations" in institutions

        # Verify categorization from rules (rules.toml has user and learned rules).
        # Parsers set merchant = description, so merchants are the full
        # Description field from the CSV.
        cats_by_merchant: dict[str, str] = {}
        for row in rows:
            cats_by_merchant[row["merchant"]] = row["category"]

        # User rules: WHOLEFDS -> Food & Dining, CHIPOTLE -> Food & Dining,
        #             SHELL OIL -> Transportation
        assert cats_by_merchant.get("WHOLEFDS LMT #10554") == "Food & Dining"
        assert cats_by_merchant.get("CHIPOTLE MEXICAN GRIL") == "Food & Dining"
        assert cats_by_merchant.get("SHELL OIL 574726183") == "Transportation"

        # Learned rule: TARGET -> Shopping
        assert cats_by_merchant.get("TARGET        00022186") == "Shopping"

        # User rule: PRIMROSE SCHOOL -> Kids (Elevations merchant is the full Description)
        primrose_merchants = [m for m in cats_by_merchant if "PRIMROSE SCHOOL" in m]
        assert len(primrose_merchants) == 1
        assert cats_by_merchant[primrose_merchants[0]] == "Kids"

        # Verify summary output mentions source institutions
        assert "Chase" in result.output
        assert "Capital One" in result.output
        assert "Elevations" in result.output

    def test_full_pipeline_february_with_transfers(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Process February 2026: transfers detected and excluded from output."""
        result = runner.invoke(
            cli,
            ["process", "--month", "2026-02", "--no-llm"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        output_path = integration_project / "output" / "2026-02.csv"
        assert output_path.is_file()

        fieldnames, rows = _read_output_csv(output_path)
        assert fieldnames == CSV_COLUMNS

        # February transactions (10 total):
        #   Chase: APPLE.COM/BILL (2/5), AMAZON.COM REFUND (2/8), CHASE AUTOPAY (2/10)
        #   Capital One: PETCO (2/2), CVS/PHARMACY (2/5), SPOTIFY (2/8), CAP ONE AUTOPAY (2/12)
        #   Elevations: VMWARE PAYROLL (2/1), CHASE CRD AUTOPAY (2/5), CAPITAL ONE CRCARDPMT (2/8)
        #
        # Transfer detection matches checking debits (keyword match) to CC credits:
        #   - Elevations "CHASE CREDIT CRD TYPE: AUTOPAY..." (-2150.00) contains "AUTOPAY"
        #     -> matched to Chase "CHASE CREDIT CRD AUTOPAY" (+2150.00) = 1 pair = 2 transfers
        #   - Elevations "CAPITAL ONE TYPE: CRCARDPMT..." (-474.90) does NOT contain
        #     any transfer keyword (PAYMENT, AUTOPAY, ONLINE PAYMENT, PAYOFF)
        #     -> NOT detected as a transfer
        #
        # So 2 transactions are excluded as transfers (the Chase pair).
        # Remaining non-transfer transactions: 8

        # Verify that the Chase transfer pair is excluded from output
        output_merchants = [row["merchant"] for row in rows]
        assert "CHASE CREDIT CRD AUTOPAY" not in output_merchants

        # Verify the Elevations CHASE CREDIT CRD debit (other side of transfer) is also excluded
        assert not any("CHASE CREDIT CRD" in m and "AUTOPAY" in m for m in output_merchants)

        # Verify transfers are mentioned in the summary
        assert "transfers excluded" in result.output

        # Verify non-transfer February transactions are present
        merchant_set = set(output_merchants)
        assert "APPLE.COM/BILL" in merchant_set
        assert "PETCO 1234" in merchant_set
        assert "CVS/PHARMACY #08432" in merchant_set
        assert "SPOTIFY USA" in merchant_set
        # Capital One autopay is NOT detected as transfer (Elevations side lacks keyword)
        assert "CAPITAL ONE AUTOPAY PYMT" in merchant_set

        # Verify a refund is flagged correctly
        refund_rows = [r for r in rows if "REFUND" in r["merchant"]]
        for refund in refund_rows:
            assert refund["is_return"] == "True"
            assert Decimal(refund["amount"]) > 0

        # Should have 8 non-transfer transactions (10 total - 2 transfers)
        assert len(rows) == 8, f"Expected 8 non-transfer rows, got {len(rows)}"

    def test_learn_workflow_and_reprocess(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Full learn loop: process -> modify output -> learn -> process again.

        Verifies that:
        1. Initial process produces uncategorized transactions
        2. User corrections via learn create new rules in rules.toml
        3. Re-processing applies the learned rules
        """
        # Step 1: Process January
        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        original_path = integration_project / "output" / "2026-01.csv"
        assert original_path.is_file()

        _, original_rows = _read_output_csv(original_path)

        # Find transactions that are Uncategorized (these would be merchants
        # not covered by the fixture rules).
        # Merchants NOT covered by rules: FIRST WATCH 0307 PAT, REI.COM  800-426-4840,
        # and LPC Nextlight TYPE:... (no rule for LPC Nextlight)
        uncategorized = [r for r in original_rows if r["category"] == "Uncategorized"]
        assert len(uncategorized) > 0, (
            "Expected some uncategorized transactions but all were categorized. "
            f"Categories: {[(r['merchant'], r['category']) for r in original_rows]}"
        )

        # Step 2: Create a corrected version with user fixes
        corrected_path = integration_project / "output" / "2026-01-corrected.csv"
        corrected_rows = []
        corrections_applied: dict[str, tuple[str, str]] = {}

        for row in original_rows:
            new_row = dict(row)
            merchant = row["merchant"]
            if row["category"] == "Uncategorized":
                # Apply corrections based on merchant keywords.
                # Note: PRIMROSE SCHOOL is covered by a user rule so it won't
                # be Uncategorized, but LPC Nextlight has no rule.
                if "FIRST WATCH" in merchant:
                    new_row["category"] = "Food & Dining"
                    new_row["subcategory"] = "Restaurant"
                    corrections_applied[merchant] = ("Food & Dining", "Restaurant")
                elif "REI.COM" in merchant:
                    new_row["category"] = "Shopping"
                    new_row["subcategory"] = "Clothing"
                    corrections_applied[merchant] = ("Shopping", "Clothing")
                elif "LPC" in merchant or "Nextlight" in merchant:
                    new_row["category"] = "Utilities"
                    new_row["subcategory"] = "Electric/Water/Internet"
                    corrections_applied[merchant] = (
                        "Utilities",
                        "Electric/Water/Internet",
                    )
            corrected_rows.append(new_row)

        assert len(corrections_applied) > 0, "No corrections were applied"

        # Write corrected CSV
        with open(corrected_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(corrected_rows)

        # Step 3: Run learn
        learn_result = runner.invoke(
            cli,
            [
                "learn",
                "--original",
                str(original_path),
                "--corrected",
                str(corrected_path),
            ],
            catch_exceptions=False,
        )

        assert learn_result.exit_code == 0, learn_result.output
        assert "Learn Summary" in learn_result.output

        # Verify rules were added
        rules_text = (integration_project / "rules.toml").read_text(encoding="utf-8")
        assert "[learned_rules]" in rules_text
        for merchant in corrections_applied:
            # The merchant pattern should appear in the learned rules section
            assert merchant in rules_text, (
                f"Expected merchant '{merchant}' in rules.toml"
            )

        # Step 4: Re-process January -- learned rules should now apply
        reprocess_result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        assert reprocess_result.exit_code == 0, reprocess_result.output

        _, reprocessed_rows = _read_output_csv(original_path)

        # Verify the previously-uncategorized transactions are now categorized
        for row in reprocessed_rows:
            merchant = row["merchant"]
            if merchant in corrections_applied:
                expected_cat, expected_sub = corrections_applied[merchant]
                assert row["category"] == expected_cat, (
                    f"After learn, {merchant} should be '{expected_cat}' "
                    f"but got '{row['category']}'"
                )
                assert row["subcategory"] == expected_sub, (
                    f"After learn, {merchant} subcategory should be '{expected_sub}' "
                    f"but got '{row['subcategory']}'"
                )

    def test_process_january_transaction_counts(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Verify exact transaction counts from three banks in January."""
        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        _, rows = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )

        # Count by institution
        by_inst: dict[str, int] = {}
        for row in rows:
            inst = row["institution"]
            by_inst[inst] = by_inst.get(inst, 0) + 1

        # Chase January: WHOLEFDS, CHIPOTLE, SHELL OIL, TARGET = 4
        assert by_inst.get("chase", 0) == 4
        # Capital One January: FIRST WATCH, REI.COM = 2
        assert by_inst.get("capital_one", 0) == 2
        # Elevations January: PRIMROSE SCHOOL, LPC Nextlight = 2
        assert by_inst.get("elevations", 0) == 2
        # Total
        assert len(rows) == 8

    def test_output_csv_sort_order(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Output CSV is sorted by date, then institution, then amount."""
        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )

        _, rows = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )

        # Verify sort: date ascending, then institution ascending, then amount ascending
        prev_key = None
        for row in rows:
            key = (row["date"], row["institution"], Decimal(row["amount"]))
            if prev_key is not None:
                assert key >= prev_key, (
                    f"Sort order violated: {prev_key} should come before {key}"
                )
            prev_key = key


# ===========================================================================
# (b) LLM fallback with mocked HTTP
# ===========================================================================


class TestLLMFallback:
    """Uncategorized transactions trigger LLM suggestions when LLM is enabled."""

    def test_llm_categorizes_uncategorized_transactions(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With LLM enabled and mocked HTTP, uncategorized transactions get
        LLM-suggested categories applied to the output."""
        project = tmp_path / "llm-project"
        project.mkdir()
        _build_project(project)
        monkeypatch.chdir(project)

        # Modify config to enable LLM
        config_text = (project / "config.toml").read_text(encoding="utf-8")
        config_text = config_text.replace(
            'provider = "none"', 'provider = "anthropic"'
        )
        config_text = config_text.replace(
            'model = ""', 'model = "claude-sonnet-4-20250514"'
        )
        config_text = config_text.replace(
            'api_key_env = ""', 'api_key_env = "TEST_ANTHROPIC_KEY"'
        )
        (project / "config.toml").write_text(config_text, encoding="utf-8")

        # Build mock LLM response for uncategorized merchants.
        # From Jan data, merchants not in rules: FIRST WATCH 0307 PAT,
        # REI.COM  800-426-4840, and LPC Nextlight TYPE:...
        llm_suggestions = [
            {
                "merchant": "FIRST WATCH 0307 PAT",
                "category": "Food & Dining",
                "subcategory": "Restaurant",
            },
            {
                "merchant": "REI.COM  800-426-4840",
                "category": "Shopping",
                "subcategory": "Clothing",
            },
        ]

        mock_response = httpx.Response(
            status_code=200,
            json=_make_anthropic_response(llm_suggestions),
            request=httpx.Request(
                "POST", "https://api.anthropic.com/v1/messages"
            ),
        )

        with (
            patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch(
                "expense_tracker.llm.httpx.post", return_value=mock_response
            ) as mock_post,
        ):
            result = runner.invoke(
                cli,
                ["process", "--month", "2026-01"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output

        # Verify the LLM was actually called
        assert mock_post.called

        # Read output and check that LLM suggestions were applied
        _, rows = _read_output_csv(project / "output" / "2026-01.csv")

        cats_by_merchant = {row["merchant"]: row["category"] for row in rows}

        # Rule-matched merchants should still be correct
        assert cats_by_merchant.get("WHOLEFDS LMT #10554") == "Food & Dining"
        assert cats_by_merchant.get("CHIPOTLE MEXICAN GRIL") == "Food & Dining"

        # LLM-suggested merchants
        assert cats_by_merchant.get("FIRST WATCH 0307 PAT") == "Food & Dining"
        assert cats_by_merchant.get("REI.COM  800-426-4840") == "Shopping"

        # Verify LLM suggestions are NOT written to rules.toml (only learn does that)
        rules_text = (project / "rules.toml").read_text(encoding="utf-8")
        assert "FIRST WATCH" not in rules_text
        assert "REI.COM" not in rules_text

    def test_llm_failure_leaves_uncategorized(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the LLM API fails, transactions remain Uncategorized
        and the pipeline still completes successfully."""
        project = tmp_path / "llm-fail-project"
        project.mkdir()
        _build_project(project)
        monkeypatch.chdir(project)

        # Enable LLM in config
        config_text = (project / "config.toml").read_text(encoding="utf-8")
        config_text = config_text.replace(
            'provider = "none"', 'provider = "anthropic"'
        )
        config_text = config_text.replace(
            'model = ""', 'model = "claude-sonnet-4-20250514"'
        )
        config_text = config_text.replace(
            'api_key_env = ""', 'api_key_env = "TEST_ANTHROPIC_KEY"'
        )
        (project / "config.toml").write_text(config_text, encoding="utf-8")

        # Mock LLM to return a 500 error
        mock_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request(
                "POST", "https://api.anthropic.com/v1/messages"
            ),
        )

        with (
            patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch(
                "expense_tracker.llm.httpx.post", return_value=mock_response
            ),
        ):
            result = runner.invoke(
                cli,
                ["process", "--month", "2026-01"],
                catch_exceptions=False,
            )

        # Pipeline should still succeed (partial failure model)
        assert result.exit_code == 0, result.output

        # Verify output was still written
        output_path = project / "output" / "2026-01.csv"
        assert output_path.is_file()

        _, rows = _read_output_csv(output_path)
        assert len(rows) > 0

        # Rule-matched transactions should still be categorized
        rule_matched = [
            r
            for r in rows
            if r["merchant"] in ("WHOLEFDS LMT #10554", "CHIPOTLE MEXICAN GRIL")
        ]
        for r in rule_matched:
            assert r["category"] != "Uncategorized"

        # Some transactions should remain Uncategorized (LLM failed)
        uncategorized = [r for r in rows if r["category"] == "Uncategorized"]
        assert len(uncategorized) > 0


# ===========================================================================
# (c) Partial failure (one bank CSV is malformed)
# ===========================================================================


class TestPartialFailure:
    """One bank's CSV is malformed; other banks process successfully."""

    def test_malformed_capital_one_others_succeed(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When Capital One CSV has wrong format, Chase and Elevations
        still process, and the summary shows errors."""
        project = tmp_path / "partial-fail-project"
        project.mkdir()
        _build_project(project)
        monkeypatch.chdir(project)

        # Replace Capital One CSV with a malformed one
        cap1_csv = project / "input" / "capital-one" / "Activity2026.csv"
        cap1_csv.write_text(_CAPITAL_ONE_MALFORMED_CSV, encoding="utf-8")

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "Processing Summary: 2026-01" in result.output

        # Verify output was written
        output_path = project / "output" / "2026-01.csv"
        assert output_path.is_file()

        _, rows = _read_output_csv(output_path)

        # Should have transactions from Chase and Elevations, but not Capital One
        institutions = {row["institution"] for row in rows}
        assert "chase" in institutions
        assert "elevations" in institutions
        assert "capital_one" not in institutions

        # Verify the error is mentioned in output (errors are printed in summary)
        assert "Error" in result.output or "error" in result.output.lower()

        # Verify Chase January transactions are present
        chase_rows = [r for r in rows if r["institution"] == "chase"]
        assert len(chase_rows) == 4  # 4 Chase January transactions

        # Verify Elevations January transactions are present
        elev_rows = [r for r in rows if r["institution"] == "elevations"]
        assert len(elev_rows) == 2  # 2 Elevations January transactions

    def test_malformed_csv_with_some_good_rows(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CSV with a few malformed rows (under 10%) still processes
        the good rows and reports warnings."""
        project = tmp_path / "few-bad-rows-project"
        project.mkdir()
        _build_project(project)
        monkeypatch.chdir(project)

        # Add a Chase CSV with 1 bad row out of 20 (5% < 10% threshold)
        chase_csv_with_bad_row = (
            "Transaction Date,Post Date,Description,Category,Type,Amount,Memo\n"
            "01/15/2026,01/16/2026,STORE A,Groceries,Sale,-10.00,\n"
            "01/16/2026,01/17/2026,STORE B,Groceries,Sale,-20.00,\n"
            "01/17/2026,01/18/2026,STORE C,Groceries,Sale,-30.00,\n"
            "01/18/2026,01/19/2026,STORE D,Groceries,Sale,-40.00,\n"
            "01/19/2026,01/20/2026,STORE E,Groceries,Sale,-50.00,\n"
            "01/20/2026,01/21/2026,STORE F,Groceries,Sale,-60.00,\n"
            "01/21/2026,01/22/2026,STORE G,Groceries,Sale,-70.00,\n"
            "01/22/2026,01/23/2026,STORE H,Groceries,Sale,-80.00,\n"
            "01/23/2026,01/24/2026,STORE I,Groceries,Sale,-90.00,\n"
            "01/24/2026,01/25/2026,STORE J,Groceries,Sale,-100.00,\n"
            "01/25/2026,01/26/2026,STORE K,Groceries,Sale,-110.00,\n"
            "01/26/2026,01/27/2026,STORE L,Groceries,Sale,-120.00,\n"
            "01/27/2026,01/28/2026,STORE M,Groceries,Sale,-130.00,\n"
            "01/28/2026,01/29/2026,STORE N,Groceries,Sale,-140.00,\n"
            "01/29/2026,01/30/2026,STORE O,Groceries,Sale,-150.00,\n"
            "01/30/2026,01/31/2026,STORE P,Groceries,Sale,-160.00,\n"
            "01/31/2026,02/01/2026,STORE Q,Groceries,Sale,-170.00,\n"
            "this is a bad row\n"
            "02/01/2026,02/02/2026,STORE R,Groceries,Sale,-180.00,\n"
            "02/02/2026,02/03/2026,STORE S,Groceries,Sale,-190.00,\n"
        )
        chase_dir = project / "input" / "chase"
        (chase_dir / "Activity2026.csv").write_text(
            chase_csv_with_bad_row, encoding="utf-8"
        )

        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        _, rows = _read_output_csv(project / "output" / "2026-01.csv")

        # Chase should have 17 January rows (rows A through Q, minus the bad row)
        chase_rows = [r for r in rows if r["institution"] == "chase"]
        assert len(chase_rows) == 17

        # Warnings about the malformed row should appear
        assert "malformed" in result.output.lower() or "Warnings" in result.output


# ===========================================================================
# (d) Enrichment cache present
# ===========================================================================


class TestEnrichmentCache:
    """Enrichment cache JSON triggers split transactions in output."""

    def test_enrichment_splits_transaction(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """A transaction with enrichment cache data is split into
        line items in the output CSV."""
        # First, process to get real transaction IDs
        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )

        _, rows_before = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )

        # Find the TARGET transaction (we know it is in January)
        target_row = None
        for row in rows_before:
            if "TARGET" in row["merchant"]:
                target_row = row
                break
        assert target_row is not None, "TARGET transaction not found in output"

        txn_id = target_row["transaction_id"]
        original_amount = Decimal(target_row["amount"])

        # Create enrichment cache JSON for this transaction
        # Split the TARGET purchase into two items that sum to the original
        half = original_amount / 2
        other_half = original_amount - half
        enrichment_data = {
            "items": [
                {
                    "merchant": "TARGET - Diapers",
                    "description": "Pampers Size 4 Pack",
                    "amount": str(half),
                },
                {
                    "merchant": "TARGET - Snacks",
                    "description": "Goldfish Crackers",
                    "amount": str(other_half),
                },
            ]
        }
        cache_dir = integration_project / "enrichment-cache"
        (cache_dir / f"{txn_id}.json").write_text(
            json.dumps(enrichment_data), encoding="utf-8"
        )

        # Re-process with enrichment data
        result = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        _, rows_after = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )

        # Should have 1 more row (original replaced by 2 splits)
        assert len(rows_after) == len(rows_before) + 1

        # Verify split transactions exist
        splits = [r for r in rows_after if r["split_from"] == txn_id]
        assert len(splits) == 2

        # Verify split IDs
        split_ids = {r["transaction_id"] for r in splits}
        assert f"{txn_id}-1" in split_ids
        assert f"{txn_id}-2" in split_ids

        # Verify split merchants -- normalized to retailer name "Target".
        # Product names are in the description field.
        split_merchants = {r["merchant"] for r in splits}
        assert "Target" in split_merchants
        split_descriptions = {r["description"] for r in splits}
        assert "Pampers Size 4 Pack" in split_descriptions
        assert "Goldfish Crackers" in split_descriptions

        # Verify split amounts sum to original
        split_total = sum(Decimal(r["amount"]) for r in splits)
        assert abs(split_total - original_amount) <= Decimal("0.01")

        # Verify the original TARGET transaction is no longer in the output
        original_in_output = [
            r for r in rows_after if r["transaction_id"] == txn_id
        ]
        assert len(original_in_output) == 0

        # Verify enrichment is mentioned in the summary
        assert "split line items" in result.output or "Enriched" in result.output


# ===========================================================================
# (e) Re-run idempotency
# ===========================================================================


class TestIdempotency:
    """Processing the same month twice produces identical output."""

    def test_process_twice_same_output(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Running process for the same month twice produces bit-for-bit
        identical CSV output."""
        # First run
        result1 = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        assert result1.exit_code == 0, result1.output

        output_path = integration_project / "output" / "2026-01.csv"
        content1 = output_path.read_text(encoding="utf-8")

        # Second run
        result2 = runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        assert result2.exit_code == 0, result2.output

        content2 = output_path.read_text(encoding="utf-8")

        # Output should be identical
        assert content1 == content2, "Output CSV differs between two runs"

    def test_process_twice_same_row_count(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Running process twice produces the same number of rows."""
        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        _, rows1 = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )

        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        _, rows2 = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )

        assert len(rows1) == len(rows2)

    def test_process_twice_same_transaction_ids(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Deterministic transaction IDs are stable across runs."""
        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        _, rows1 = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )
        ids1 = [r["transaction_id"] for r in rows1]

        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        _, rows2 = _read_output_csv(
            integration_project / "output" / "2026-01.csv"
        )
        ids2 = [r["transaction_id"] for r in rows2]

        assert ids1 == ids2

    def test_process_different_months_independent(
        self, runner: CliRunner, integration_project: Path
    ) -> None:
        """Processing two different months produces separate output files."""
        runner.invoke(
            cli,
            ["process", "--month", "2026-01", "--no-llm"],
            catch_exceptions=False,
        )
        runner.invoke(
            cli,
            ["process", "--month", "2026-02", "--no-llm"],
            catch_exceptions=False,
        )

        jan_path = integration_project / "output" / "2026-01.csv"
        feb_path = integration_project / "output" / "2026-02.csv"

        assert jan_path.is_file()
        assert feb_path.is_file()

        _, jan_rows = _read_output_csv(jan_path)
        _, feb_rows = _read_output_csv(feb_path)

        # No overlap in dates
        jan_dates = {r["date"] for r in jan_rows}
        feb_dates = {r["date"] for r in feb_rows}
        assert jan_dates.isdisjoint(feb_dates)

        # No overlap in transaction IDs
        jan_ids = {r["transaction_id"] for r in jan_rows}
        feb_ids = {r["transaction_id"] for r in feb_rows}
        assert jan_ids.isdisjoint(feb_ids)
