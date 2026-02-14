"""Tests for expense_tracker.config — loading, saving, and initialization."""

from pathlib import Path

from expense_tracker.config import (
    initialize,
    load_categories,
    load_config,
    load_rules,
    save_learned_rules,
)
from expense_tracker.models import AccountConfig, AppConfig, MerchantRule


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for loading config.toml into an AppConfig."""

    def test_loads_default_config(self, tmp_path: Path):
        """Default config.toml produced by initialize() is loadable."""
        initialize(tmp_path)
        config = load_config(tmp_path)

        assert isinstance(config, AppConfig)
        assert len(config.accounts) == 3
        assert config.output_dir == "output"
        assert config.enrichment_cache_dir == "enrichment-cache"

    def test_account_fields(self, tmp_path: Path):
        """Each account entry maps to a correct AccountConfig."""
        initialize(tmp_path)
        config = load_config(tmp_path)

        chase = config.accounts[0]
        assert isinstance(chase, AccountConfig)
        assert chase.name == "Chase Credit Card"
        assert chase.institution == "chase"
        assert chase.parser == "chase"
        assert chase.account_type == "credit_card"
        assert chase.input_dir == "input/chase"

        cap_one = config.accounts[1]
        assert cap_one.name == "Capital One Credit Card"
        assert cap_one.institution == "capital_one"
        assert cap_one.input_dir == "input/capital-one"

        elevations = config.accounts[2]
        assert elevations.name == "Elevations Credit Union"
        assert elevations.institution == "elevations"
        assert elevations.account_type == "checking"

    def test_transfer_detection(self, tmp_path: Path):
        """Transfer detection settings are loaded correctly."""
        initialize(tmp_path)
        config = load_config(tmp_path)

        assert config.transfer_keywords == ["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"]
        assert config.transfer_date_window == 5

    def test_llm_settings(self, tmp_path: Path):
        """LLM settings are loaded correctly."""
        initialize(tmp_path)
        config = load_config(tmp_path)

        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.llm_api_key_env == "ANTHROPIC_API_KEY"

    def test_custom_config(self, tmp_path: Path):
        """A hand-crafted config.toml loads with the correct values."""
        (tmp_path / "config.toml").write_text(
            """\
[general]
output_dir = "custom-output"
enrichment_cache_dir = "custom-cache"

[transfer_detection]
keywords = ["PAYMENT"]
date_window_days = 3

[llm]
provider = "none"
model = ""
api_key_env = ""

[[accounts]]
name = "Test Account"
institution = "test"
parser = "test"
account_type = "checking"
input_dir = "input/test"
""",
            encoding="utf-8",
        )
        config = load_config(tmp_path)

        assert config.output_dir == "custom-output"
        assert config.enrichment_cache_dir == "custom-cache"
        assert config.transfer_keywords == ["PAYMENT"]
        assert config.transfer_date_window == 3
        assert config.llm_provider == "none"
        assert len(config.accounts) == 1
        assert config.accounts[0].name == "Test Account"

    def test_missing_config_raises(self, tmp_path: Path):
        """FileNotFoundError is raised when config.toml does not exist."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path)


# ---------------------------------------------------------------------------
# load_categories
# ---------------------------------------------------------------------------


class TestLoadCategories:
    """Tests for loading categories.toml into a list of dicts."""

    def test_loads_default_categories(self, tmp_path: Path):
        """Default categories.toml produced by initialize() is loadable."""
        initialize(tmp_path)
        categories = load_categories(tmp_path)

        assert isinstance(categories, list)
        assert len(categories) == 18

    def test_category_structure(self, tmp_path: Path):
        """Each category dict has 'name' and 'subcategories' keys."""
        initialize(tmp_path)
        categories = load_categories(tmp_path)

        for cat in categories:
            assert "name" in cat
            assert "subcategories" in cat
            assert isinstance(cat["name"], str)
            assert isinstance(cat["subcategories"], list)

    def test_specific_categories(self, tmp_path: Path):
        """Spot-check a few specific categories and their subcategories."""
        initialize(tmp_path)
        categories = load_categories(tmp_path)

        by_name = {c["name"]: c for c in categories}

        assert "Food & Dining" in by_name
        food = by_name["Food & Dining"]
        assert "Groceries" in food["subcategories"]
        assert "Restaurant" in food["subcategories"]
        assert "Fast Food" in food["subcategories"]

        assert "Insurance" in by_name
        assert by_name["Insurance"]["subcategories"] == []

        assert "Miscellaneous" in by_name
        assert by_name["Miscellaneous"]["subcategories"] == []

    def test_custom_categories(self, tmp_path: Path):
        """A hand-crafted categories.toml loads correctly."""
        (tmp_path / "categories.toml").write_text(
            """\
[Food]
subcategories = ["Groceries", "Restaurant"]

[Transport]
subcategories = []
""",
            encoding="utf-8",
        )
        categories = load_categories(tmp_path)

        assert len(categories) == 2
        assert categories[0]["name"] == "Food"
        assert categories[0]["subcategories"] == ["Groceries", "Restaurant"]
        assert categories[1]["name"] == "Transport"
        assert categories[1]["subcategories"] == []

    def test_missing_categories_raises(self, tmp_path: Path):
        """FileNotFoundError is raised when categories.toml does not exist."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_categories(tmp_path)


# ---------------------------------------------------------------------------
# load_rules
# ---------------------------------------------------------------------------


class TestLoadRules:
    """Tests for loading rules.toml into a list of MerchantRule objects."""

    def test_loads_default_rules(self, tmp_path: Path):
        """Default rules.toml (empty sections) loads as an empty list."""
        initialize(tmp_path)
        rules = load_rules(tmp_path)

        assert isinstance(rules, list)
        assert len(rules) == 0

    def test_user_rules_loaded(self, tmp_path: Path):
        """User rules are loaded with source='user'."""
        (tmp_path / "rules.toml").write_text(
            """\
[user_rules]
"KING SOOPERS" = "Food & Dining:Groceries"
"CHIPOTLE" = "Food & Dining:Fast Food"

[learned_rules]
""",
            encoding="utf-8",
        )
        rules = load_rules(tmp_path)

        assert len(rules) == 2
        assert rules[0].pattern == "KING SOOPERS"
        assert rules[0].category == "Food & Dining"
        assert rules[0].subcategory == "Groceries"
        assert rules[0].source == "user"

        assert rules[1].pattern == "CHIPOTLE"
        assert rules[1].category == "Food & Dining"
        assert rules[1].subcategory == "Fast Food"
        assert rules[1].source == "user"

    def test_learned_rules_loaded(self, tmp_path: Path):
        """Learned rules are loaded with source='learned'."""
        (tmp_path / "rules.toml").write_text(
            """\
[user_rules]

[learned_rules]
"STARBUCKS" = "Food & Dining:Coffee"
""",
            encoding="utf-8",
        )
        rules = load_rules(tmp_path)

        assert len(rules) == 1
        assert rules[0].pattern == "STARBUCKS"
        assert rules[0].source == "learned"

    def test_user_rules_before_learned(self, tmp_path: Path):
        """User rules appear before learned rules in the returned list."""
        (tmp_path / "rules.toml").write_text(
            """\
[user_rules]
"USER MERCHANT" = "Shopping"

[learned_rules]
"LEARNED MERCHANT" = "Entertainment"
""",
            encoding="utf-8",
        )
        rules = load_rules(tmp_path)

        assert len(rules) == 2
        assert rules[0].source == "user"
        assert rules[0].pattern == "USER MERCHANT"
        assert rules[1].source == "learned"
        assert rules[1].pattern == "LEARNED MERCHANT"

    def test_category_without_subcategory(self, tmp_path: Path):
        """Rules with category only (no colon) have empty subcategory."""
        (tmp_path / "rules.toml").write_text(
            """\
[user_rules]
"NETFLIX" = "Entertainment"

[learned_rules]
""",
            encoding="utf-8",
        )
        rules = load_rules(tmp_path)

        assert len(rules) == 1
        assert rules[0].category == "Entertainment"
        assert rules[0].subcategory == ""

    def test_missing_rules_raises(self, tmp_path: Path):
        """FileNotFoundError is raised when rules.toml does not exist."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_rules(tmp_path)


# ---------------------------------------------------------------------------
# save_learned_rules
# ---------------------------------------------------------------------------


class TestSaveLearnedRules:
    """Tests for writing learned rules back to rules.toml."""

    def test_save_to_empty_learned_section(self, tmp_path: Path):
        """Saving rules to an empty [learned_rules] section works."""
        initialize(tmp_path)

        learned = [
            MerchantRule(
                pattern="STARBUCKS",
                category="Food & Dining",
                subcategory="Coffee",
                source="learned",
            ),
        ]
        save_learned_rules(tmp_path, learned)

        # Verify it round-trips.
        rules = load_rules(tmp_path)
        assert len(rules) == 1
        assert rules[0].pattern == "STARBUCKS"
        assert rules[0].category == "Food & Dining"
        assert rules[0].subcategory == "Coffee"
        assert rules[0].source == "learned"

    def test_preserves_user_rules(self, tmp_path: Path):
        """Saving learned rules does not alter user rules."""
        (tmp_path / "rules.toml").write_text(
            """\
[user_rules]
"KING SOOPERS" = "Food & Dining:Groceries"
"CHIPOTLE" = "Food & Dining:Fast Food"

[learned_rules]
""",
            encoding="utf-8",
        )

        learned = [
            MerchantRule(
                pattern="STARBUCKS",
                category="Food & Dining",
                subcategory="Coffee",
                source="learned",
            ),
        ]
        save_learned_rules(tmp_path, learned)

        rules = load_rules(tmp_path)
        user_rules = [r for r in rules if r.source == "user"]
        learned_rules = [r for r in rules if r.source == "learned"]

        assert len(user_rules) == 2
        assert user_rules[0].pattern == "KING SOOPERS"
        assert user_rules[1].pattern == "CHIPOTLE"
        assert len(learned_rules) == 1
        assert learned_rules[0].pattern == "STARBUCKS"

    def test_user_rules_section_verbatim(self, tmp_path: Path):
        """The [user_rules] section text is preserved exactly, including
        comments and blank lines."""
        user_section = """\
[user_rules]
# My custom rules
"KING SOOPERS" = "Food & Dining:Groceries"

# Fast food
"CHIPOTLE" = "Food & Dining:Fast Food"

"""
        (tmp_path / "rules.toml").write_text(
            user_section + "[learned_rules]\n",
            encoding="utf-8",
        )

        save_learned_rules(tmp_path, [])

        text = (tmp_path / "rules.toml").read_text(encoding="utf-8")
        assert text.startswith(user_section)

    def test_filters_non_learned_rules(self, tmp_path: Path):
        """Only rules with source='learned' are written; user rules in the
        input list are ignored."""
        initialize(tmp_path)

        rules = [
            MerchantRule(pattern="USER RULE", category="Shopping", source="user"),
            MerchantRule(pattern="LEARNED RULE", category="Entertainment", source="learned"),
        ]
        save_learned_rules(tmp_path, rules)

        loaded = load_rules(tmp_path)
        learned = [r for r in loaded if r.source == "learned"]
        assert len(learned) == 1
        assert learned[0].pattern == "LEARNED RULE"

    def test_overwrites_previous_learned_rules(self, tmp_path: Path):
        """Saving a new set of learned rules replaces the old set."""
        initialize(tmp_path)

        # First save.
        save_learned_rules(
            tmp_path,
            [MerchantRule(pattern="OLD", category="Shopping", source="learned")],
        )
        assert load_rules(tmp_path)[0].pattern == "OLD"

        # Second save with different rules.
        save_learned_rules(
            tmp_path,
            [MerchantRule(pattern="NEW", category="Entertainment", source="learned")],
        )
        rules = load_rules(tmp_path)
        assert len(rules) == 1
        assert rules[0].pattern == "NEW"

    def test_round_trip_multiple_rules(self, tmp_path: Path):
        """Multiple learned rules survive a save-then-load cycle."""
        initialize(tmp_path)

        learned = [
            MerchantRule(
                pattern="STARBUCKS", category="Food & Dining", subcategory="Coffee", source="learned"
            ),
            MerchantRule(
                pattern="NETFLIX", category="Entertainment", subcategory="Subscriptions", source="learned"
            ),
            MerchantRule(
                pattern="SHELL OIL", category="Transportation", subcategory="Gas/Fuel", source="learned"
            ),
        ]
        save_learned_rules(tmp_path, learned)
        loaded = load_rules(tmp_path)

        assert len(loaded) == 3
        for original, reloaded in zip(learned, loaded):
            assert reloaded.pattern == original.pattern
            assert reloaded.category == original.category
            assert reloaded.subcategory == original.subcategory
            assert reloaded.source == "learned"

    def test_round_trip_with_user_and_learned(self, tmp_path: Path):
        """Full round-trip: user rules + learned rules, save, reload."""
        (tmp_path / "rules.toml").write_text(
            """\
[user_rules]
"KING SOOPERS" = "Food & Dining:Groceries"

[learned_rules]
"OLD LEARNED" = "Shopping"
""",
            encoding="utf-8",
        )

        new_learned = [
            MerchantRule(
                pattern="STARBUCKS", category="Food & Dining", subcategory="Coffee", source="learned"
            ),
            MerchantRule(
                pattern="NETFLIX", category="Entertainment", subcategory="Subscriptions", source="learned"
            ),
        ]
        save_learned_rules(tmp_path, new_learned)

        rules = load_rules(tmp_path)
        user_rules = [r for r in rules if r.source == "user"]
        learned_rules = [r for r in rules if r.source == "learned"]

        assert len(user_rules) == 1
        assert user_rules[0].pattern == "KING SOOPERS"
        assert len(learned_rules) == 2
        assert learned_rules[0].pattern == "STARBUCKS"
        assert learned_rules[1].pattern == "NETFLIX"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    """Tests for initializing the project directory structure."""

    def test_creates_directories(self, tmp_path: Path):
        """All required directories are created."""
        initialize(tmp_path)

        assert (tmp_path / "input" / "chase").is_dir()
        assert (tmp_path / "input" / "capital-one").is_dir()
        assert (tmp_path / "input" / "elevations").is_dir()
        assert (tmp_path / "output").is_dir()
        assert (tmp_path / "enrichment-cache").is_dir()

    def test_creates_config_files(self, tmp_path: Path):
        """All three TOML config files are created."""
        initialize(tmp_path)

        assert (tmp_path / "config.toml").is_file()
        assert (tmp_path / "categories.toml").is_file()
        assert (tmp_path / "rules.toml").is_file()

    def test_config_files_are_valid_toml(self, tmp_path: Path):
        """All three created files are valid TOML that can be parsed."""
        initialize(tmp_path)

        # These will raise if TOML is invalid.
        load_config(tmp_path)
        load_categories(tmp_path)
        load_rules(tmp_path)

    def test_idempotent_does_not_overwrite(self, tmp_path: Path):
        """Running initialize twice does not overwrite existing files."""
        initialize(tmp_path)

        # Write custom content to config.toml.
        custom_content = "# Custom config\n"
        (tmp_path / "config.toml").write_text(custom_content, encoding="utf-8")

        # Run initialize again.
        initialize(tmp_path)

        # The custom content should be preserved.
        assert (tmp_path / "config.toml").read_text(encoding="utf-8") == custom_content

    def test_idempotent_preserves_all_files(self, tmp_path: Path):
        """Running initialize twice preserves all three config files."""
        initialize(tmp_path)

        # Read the original content.
        original_config = (tmp_path / "config.toml").read_text(encoding="utf-8")
        original_categories = (tmp_path / "categories.toml").read_text(encoding="utf-8")
        original_rules = (tmp_path / "rules.toml").read_text(encoding="utf-8")

        # Run initialize again.
        initialize(tmp_path)

        assert (tmp_path / "config.toml").read_text(encoding="utf-8") == original_config
        assert (tmp_path / "categories.toml").read_text(encoding="utf-8") == original_categories
        assert (tmp_path / "rules.toml").read_text(encoding="utf-8") == original_rules

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        """Initialize creates the target directory itself if it doesn't exist."""
        target = tmp_path / "nested" / "deep" / "expenses"
        initialize(target)

        assert target.is_dir()
        assert (target / "config.toml").is_file()
        assert (target / "input" / "chase").is_dir()

    def test_partial_existing_structure(self, tmp_path: Path):
        """Initialize fills in missing pieces without touching existing ones."""
        # Pre-create some directories but not all.
        (tmp_path / "input" / "chase").mkdir(parents=True)
        (tmp_path / "output").mkdir()

        # Pre-create one config file with custom content.
        (tmp_path / "config.toml").write_text("# Pre-existing\n", encoding="utf-8")

        initialize(tmp_path)

        # Missing directories should be created.
        assert (tmp_path / "input" / "capital-one").is_dir()
        assert (tmp_path / "input" / "elevations").is_dir()
        assert (tmp_path / "enrichment-cache").is_dir()

        # Pre-existing config should be untouched.
        assert (tmp_path / "config.toml").read_text(encoding="utf-8") == "# Pre-existing\n"

        # Missing config files should be created.
        assert (tmp_path / "categories.toml").is_file()
        assert (tmp_path / "rules.toml").is_file()
