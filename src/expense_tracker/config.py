"""Configuration loading, writing, and project initialization.

Reads TOML config files using stdlib ``tomllib`` and writes them using
``tomli_w``.  Depends only on ``models.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from expense_tracker.models import AccountConfig, AppConfig, MerchantRule

# ---------------------------------------------------------------------------
# Default file content -- matches architecture doc Section 7 exactly
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_TOML = """\
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
"""

_DEFAULT_CATEGORIES_TOML = """\
# Category taxonomy -- the valid categories and subcategories

[Housing]
subcategories = ["Mortgage"]

[Utilities]
subcategories = ["Electric/Water/Internet", "Natural Gas", "Mobile Phone", "Television"]

["Food & Dining"]
subcategories = ["Groceries", "Restaurant", "Fast Food", "Coffee", "Alcohol", "Delivery"]

[Transportation]
subcategories = ["Gas/Fuel", "Parking/Tolls", "Public Transit", "Rideshare", "Service & Maintenance", "Registration/DMV"]

[Kids]
subcategories = ["Clothing", "Supplies", "Activities", "Toys", "School", "Preschool", "Camps"]

["Health & Fitness"]
subcategories = ["Gym/Classes", "Skiing", "Biking", "Hockey", "Race/Event Fees", "Equipment & Maintenance"]

[Healthcare]
subcategories = ["Doctor", "Dental", "Vision", "Pharmacy", "Therapy"]

[Entertainment]
subcategories = ["Tickets/Events", "Games", "Movies", "Subscriptions"]

[Shopping]
subcategories = ["Clothing", "Electronics", "Home Goods", "Books", "Jewelry"]

["Home & Garden"]
subcategories = ["Maintenance & Repairs", "Furniture & Decor", "Appliances", "Garden & Lawn", "Tools & Hardware", "Home Services"]

["Personal Care"]
subcategories = ["Haircut/Barber", "Beauty/Spa", "Cosmetics"]

[Pets]
subcategories = ["Food", "Vet", "Daycare/Boarding", "Grooming", "Supplies"]

["Gifts & Charity"]
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
"""

_DEFAULT_RULES_TOML = """\
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
"""

# Directories that ``initialize`` creates.
_INIT_DIRS = [
    "input/chase",
    "input/capital-one",
    "input/elevations",
    "output",
    "enrichment-cache",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(root: Path) -> AppConfig:
    """Load ``config.toml`` from *root* and return an :class:`AppConfig`.

    Args:
        root: Project root directory containing ``config.toml``.

    Returns:
        A fully-populated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If ``config.toml`` does not exist.
    """
    data = _read_toml(root / "config.toml")

    general = data.get("general", {})
    transfer = data.get("transfer_detection", {})
    llm = data.get("llm", {})

    accounts = [
        AccountConfig(
            name=a["name"],
            institution=a["institution"],
            parser=a["parser"],
            account_type=a["account_type"],
            input_dir=a["input_dir"],
        )
        for a in data.get("accounts", [])
    ]

    return AppConfig(
        accounts=accounts,
        output_dir=general.get("output_dir", "output"),
        enrichment_cache_dir=general.get("enrichment_cache_dir", "enrichment-cache"),
        transfer_keywords=transfer.get(
            "keywords", ["PAYMENT", "AUTOPAY", "ONLINE PAYMENT", "PAYOFF"]
        ),
        transfer_date_window=transfer.get("date_window_days", 5),
        llm_provider=llm.get("provider", "anthropic"),
        llm_model=llm.get("model", "claude-sonnet-4-20250514"),
        llm_api_key_env=llm.get("api_key_env", "ANTHROPIC_API_KEY"),
    )


def load_categories(root: Path) -> list[dict]:
    """Load ``categories.toml`` and return the taxonomy.

    Args:
        root: Project root directory containing ``categories.toml``.

    Returns:
        A list of ``{"name": str, "subcategories": list[str]}`` dicts,
        one per top-level category, preserving file order.

    Raises:
        FileNotFoundError: If ``categories.toml`` does not exist.
    """
    data = _read_toml(root / "categories.toml")
    return [
        {"name": name, "subcategories": list(section.get("subcategories", []))}
        for name, section in data.items()
        if isinstance(section, dict)
    ]


def load_rules(root: Path) -> list[MerchantRule]:
    """Load ``rules.toml`` and return a sorted list of rules.

    User rules come first, then learned rules.  Within each group rules
    are in file order (insertion order preserved by ``tomllib``).

    Args:
        root: Project root directory containing ``rules.toml``.

    Returns:
        A list of :class:`MerchantRule` objects.

    Raises:
        FileNotFoundError: If ``rules.toml`` does not exist.
    """
    data = _read_toml(root / "rules.toml")
    rules: list[MerchantRule] = []

    for pattern, value in data.get("user_rules", {}).items():
        cat, subcat = _parse_category_value(value)
        rules.append(MerchantRule(pattern=pattern, category=cat, subcategory=subcat, source="user"))

    for pattern, value in data.get("learned_rules", {}).items():
        cat, subcat = _parse_category_value(value)
        rules.append(
            MerchantRule(pattern=pattern, category=cat, subcategory=subcat, source="learned")
        )

    return rules


def save_learned_rules(root: Path, rules: list[MerchantRule]) -> None:
    """Write learned rules to the ``[learned_rules]`` section of ``rules.toml``.

    The ``[user_rules]`` section (and everything above it) is preserved
    verbatim.  Only the ``[learned_rules]`` section is rewritten.

    Args:
        root: Project root directory containing ``rules.toml``.
        rules: The complete list of learned rules to write.  Only rules
            with ``source="learned"`` are written; others are ignored.
    """
    rules_path = root / "rules.toml"
    original_text = rules_path.read_text(encoding="utf-8")

    # Find where [learned_rules] starts and preserve everything before it.
    marker = "[learned_rules]"
    idx = original_text.find(marker)
    if idx == -1:
        # No [learned_rules] section yet -- append one.
        prefix = original_text.rstrip() + "\n\n"
    else:
        prefix = original_text[:idx]

    # Build the new [learned_rules] section using tomli_w for correct quoting.
    learned = [r for r in rules if r.source == "learned"]
    learned_dict = {r.pattern: _format_category_value(r) for r in learned}

    section_header = "[learned_rules]\n"
    section_comment = (
        "# System-managed rules from the learn command. Do not hand-edit.\n"
        "# Same format as user_rules.\n"
    )

    if learned_dict:
        # Use tomli_w to serialize just the key-value pairs.  We serialize
        # the full table then strip its header so we can insert our own
        # header + comment.
        kv_text = tomli_w.dumps({"learned_rules": learned_dict})
        # Remove the "[learned_rules]\n" header that tomli_w generates.
        kv_lines = kv_text.split("\n", 1)[1] if "\n" in kv_text else ""
        new_section = section_header + section_comment + kv_lines
    else:
        new_section = section_header + section_comment

    rules_path.write_text(prefix + new_section, encoding="utf-8")


def initialize(target_dir: Path) -> None:
    """Create the standard directory structure and default config files.

    Idempotent: existing directories are left alone and existing files
    are **not** overwritten.

    Args:
        target_dir: The directory in which to create the project structure.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories.
    for d in _INIT_DIRS:
        (target_dir / d).mkdir(parents=True, exist_ok=True)

    # Write default config files (skip if they already exist).
    _write_if_missing(target_dir / "config.toml", _DEFAULT_CONFIG_TOML)
    _write_if_missing(target_dir / "categories.toml", _DEFAULT_CATEGORIES_TOML)
    _write_if_missing(target_dir / "rules.toml", _DEFAULT_RULES_TOML)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict:
    """Read and parse a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_category_value(value: str) -> tuple[str, str]:
    """Parse a ``"Category"`` or ``"Category:Subcategory"`` string."""
    if ":" in value:
        cat, subcat = value.split(":", 1)
        return cat.strip(), subcat.strip()
    return value.strip(), ""


def _format_category_value(rule: MerchantRule) -> str:
    """Format a rule's category/subcategory back to TOML value form."""
    if rule.subcategory:
        return f"{rule.category}:{rule.subcategory}"
    return rule.category


def _write_if_missing(path: Path, content: str) -> None:
    """Write *content* to *path* only if the file does not already exist."""
    if not path.exists():
        path.write_text(content, encoding="utf-8")
