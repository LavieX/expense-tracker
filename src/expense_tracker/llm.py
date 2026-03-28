"""LLM adapter for AI-driven transaction categorization.

The primary adapter (ClaudeCodeAdapter) invokes Claude Code as a
subprocess — using the user's existing Max subscription. No API key
needed. The AnthropicAdapter is kept as a fallback for headless/CI use.

Transactions are batched and sent with the full category taxonomy and
household context. Claude reads each merchant/description and assigns
the best category.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Protocol

logger = logging.getLogger(__name__)

# Household context helps Claude understand recurring merchants
HOUSEHOLD_CONTEXT = """
This is a household in Longmont, Colorado (Boulder County area).
- Family with young kids (preschool/elementary age)
- Pets: dog(s) and fish/aquarium
- Drives a Tesla (EV charging, not gas — Tesla Supercharger = EV Charging)
- Dad plays hockey, family skis
- Colleen (mom) runs a photography/design business (Home Craft Design)
- Common local merchants: King Soopers (groceries), Sprouts (groceries),
  Safeway (groceries), Moe's Broadway Bagel (restaurant), Spruce Airport (coffee shop, NOT an airport),
  Camp Bow Wow (dog daycare), Primrose School (preschool)
- Venmo payments to "Colleen Tobey" or between family are internal transfers
- BREEZE THRU is a drive-thru liquor store, not transportation
- Apple.com/Bill charges are subscriptions (iCloud, Apple Music, Apple One), not electronics
- Peacock = NBC streaming service subscription
- ALIGN PT = physical therapy
- IVY SESSION = therapy
- Steve and Kate's = kids camp program
- Winners Circle Longmont = kids arcade/entertainment
"""

# Max transactions per batch
BATCH_SIZE = 80


class LLMAdapter(Protocol):
    """Protocol for LLM-based transaction categorization."""

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        ...


def _build_prompt(transactions: list[dict], categories: list[dict]) -> str:
    """Build the categorization prompt."""
    taxonomy_lines: list[str] = []
    for cat in categories:
        name = cat["name"]
        subs = cat.get("subcategories", [])
        if subs:
            taxonomy_lines.append(f"- {name}: {', '.join(subs)}")
        else:
            taxonomy_lines.append(f"- {name}")
    taxonomy_text = "\n".join(taxonomy_lines)

    txn_lines: list[str] = []
    for txn in transactions:
        line = f"{txn['id']} | {txn['merchant']} | {txn['description']} | {txn['amount']} | {txn['date']}"
        if txn.get("source"):
            line += f" | source:{txn['source']}"
        txn_lines.append(line)
    txn_text = "\n".join(txn_lines)

    return (
        "You are categorizing household expenses. For EVERY transaction below,\n"
        "assign the single best category and subcategory from the taxonomy.\n"
        "\n"
        "## Household Context\n"
        f"{HOUSEHOLD_CONTEXT}\n"
        "\n"
        "## Rules\n"
        "- Categorize based on what was ACTUALLY purchased, not the retailer.\n"
        "  Amazon/Target/Walmart items: read the description (product name).\n"
        "- Food items (produce, meat, dairy, beverages, snacks) from ANY store = Food & Dining:Groceries\n"
        "- Children's medicine (Motrin, Tylenol, Pepto) = Healthcare:Pharmacy\n"
        "- Skin care products (Aquaphor, lotions) for people = Healthcare:Pharmacy or Personal Care\n"
        "- Makeup/cosmetics (Maybelline, e.l.f.) = Personal Care:Cosmetics\n"
        "- Kids' clothing/shoes (Cat & Jack, toddler items) = Kids:Clothing\n"
        "- Flowers from florists = Gifts & Charity:Flowers\n"
        "- Hotels/lodging = Travel:Hotel/Lodging\n"
        "- Sales tax line items: match the category of the other items in the same split\n"
        "- Refunds/returns (positive amounts): use the same category as the original charge\n"
        "- If a transaction is clearly an internal transfer (Venmo between spouses), use Miscellaneous:Transfers\n"
        "\n"
        "## Category Taxonomy\n"
        f"{taxonomy_text}\n"
        "\n"
        "## Transactions\n"
        f"ID | Merchant | Description | Amount | Date [| source]\n"
        f"{txn_text}\n"
        "\n"
        "## Response Format\n"
        "Return ONLY a JSON array with one object per transaction:\n"
        '[{"id": "...", "category": "...", "subcategory": "..."}]\n'
        "\n"
        "Use the transaction ID from the first column. Use ONLY categories and\n"
        "subcategories from the taxonomy. Empty string for subcategory if none fits.\n"
        "You MUST return exactly one entry per transaction — do not skip any.\n"
        "Return ONLY the JSON array, no other text."
    )


def _parse_response(text: str, expected_count: int = 0) -> list[dict]:
    """Extract the JSON array from the LLM response."""
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM response does not contain a JSON array")
        return []

    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON from LLM response: %s", exc)
        return []

    if not isinstance(result, list):
        logger.warning("LLM response JSON is not a list")
        return []

    validated: list[dict] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        if "id" in item and "category" in item:
            validated.append({
                "id": str(item["id"]),
                "category": str(item["category"]),
                "subcategory": str(item.get("subcategory", "")),
            })
        elif "merchant" in item and "category" in item:
            validated.append({
                "id": "",
                "merchant": str(item["merchant"]),
                "category": str(item["category"]),
                "subcategory": str(item.get("subcategory", "")),
            })

    if expected_count and len(validated) < expected_count:
        logger.warning(
            "LLM returned %d categorizations for %d transactions",
            len(validated), expected_count,
        )

    return validated


class ClaudeCodeAdapter:
    """Categorization via Claude Code subprocess (uses Max subscription).

    Invokes ``claude`` CLI with the categorization prompt. No API key
    needed — uses the same subscription as the interactive Claude Code
    session.
    """

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        if not transactions:
            return []

        all_results: list[dict] = []

        for i in range(0, len(transactions), BATCH_SIZE):
            batch = transactions[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(transactions) + BATCH_SIZE - 1) // BATCH_SIZE

            if total_batches > 1:
                logger.info(
                    "Categorizing batch %d/%d (%d transactions)",
                    batch_num, total_batches, len(batch),
                )

            results = self._invoke_claude(batch, categories)
            all_results.extend(results)

        return all_results

    def _invoke_claude(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        """Call claude CLI as a subprocess."""
        prompt = _build_prompt(transactions, categories)

        try:
            result = subprocess.run(
                [
                    "claude",
                    "--print",  # Output only, no interactive mode
                    "--model", self.model,
                    "--max-turns", "1",
                    "-p", prompt,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning(
                    "claude CLI returned exit code %d: %s",
                    result.returncode, result.stderr[:200],
                )
                return []

            response_text = result.stdout
            if not response_text.strip():
                logger.warning("claude CLI returned empty response")
                return []

            return _parse_response(response_text, len(transactions))

        except FileNotFoundError:
            logger.error(
                "claude CLI not found. Install Claude Code: "
                "https://docs.anthropic.com/en/docs/claude-code"
            )
            return []
        except subprocess.TimeoutExpired:
            logger.warning("claude CLI timed out after 120s")
            return []
        except Exception as exc:
            logger.warning("claude CLI invocation failed: %s", exc)
            return []


class AnthropicAdapter:
    """Direct Anthropic API adapter (requires API credits)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 8192,
        timeout: float = 90.0,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        if not transactions:
            return []

        import httpx

        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            logger.warning(
                "LLM API key not found in '%s'", self.api_key_env,
            )
            return []

        all_results: list[dict] = []
        for i in range(0, len(transactions), BATCH_SIZE):
            batch = transactions[i : i + BATCH_SIZE]
            results = self._call_api(batch, categories, api_key, httpx)
            all_results.extend(results)

        return all_results

    def _call_api(self, transactions, categories, api_key, httpx):
        prompt = _build_prompt(transactions, categories)
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            text = "\n".join(
                b["text"] for b in body.get("content", []) if b.get("type") == "text"
            )
            return _parse_response(text, len(transactions))
        except Exception as exc:
            logger.warning("Anthropic API call failed: %s", exc)
            return []


class NullAdapter:
    """No-op adapter for --no-llm mode."""

    def categorize_batch(self, transactions, categories):
        return []
