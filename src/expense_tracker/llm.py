"""LLM adapter interface and Anthropic implementation.

Defines the LLMAdapter protocol for categorizing transactions via an LLM,
plus two implementations:
- AnthropicAdapter: sends batches to the Anthropic Messages API via httpx.
- NullAdapter: no-op adapter that always returns an empty list (for --no-llm mode).

This module depends only on the standard library and httpx. It has no internal
imports from expense_tracker -- the categorizer passes plain dicts, not
Transaction objects, to keep the boundary clean.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


class LLMAdapter(Protocol):
    """Protocol for LLM-based transaction categorization.

    Implementations receive a batch of transaction dicts and a category
    taxonomy, and return a list of category suggestion dicts. On any
    failure, implementations must return an empty list rather than raising.
    """

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        """Send a batch of transactions to the LLM for categorization.

        Args:
            transactions: List of dicts, each with keys:
                merchant, description, amount, date.
            categories: List of dicts, each with keys:
                name (str), subcategories (list[str]).

        Returns:
            List of dicts, each with keys: merchant, category, subcategory.
            Empty list on any failure.
        """
        ...


def _build_prompt(transactions: list[dict], categories: list[dict]) -> str:
    """Construct the categorization prompt per the architecture doc Section 8.

    The prompt contains the category taxonomy, the list of uncategorized
    transactions, and instructions for the expected JSON response format.

    Args:
        transactions: Transaction dicts with merchant, description, amount, date.
        categories: Category taxonomy dicts with name and subcategories.

    Returns:
        The fully formatted prompt string.
    """
    # Format the taxonomy
    taxonomy_lines: list[str] = []
    for cat in categories:
        name = cat["name"]
        subs = cat.get("subcategories", [])
        if subs:
            sub_str = ", ".join(subs)
            taxonomy_lines.append(f"- {name}: {sub_str}")
        else:
            taxonomy_lines.append(f"- {name}")
    taxonomy_text = "\n".join(taxonomy_lines)

    # Format the transactions
    txn_lines: list[str] = []
    for txn in transactions:
        txn_lines.append(
            f"{txn['merchant']} | {txn['description']} | {txn['amount']} | {txn['date']}"
        )
    txn_text = "\n".join(txn_lines)

    return (
        "You are categorizing household expenses. For each transaction below,\n"
        "assign the most appropriate category and subcategory from the provided taxonomy.\n"
        "\n"
        "## Category Taxonomy\n"
        f"{taxonomy_text}\n"
        "\n"
        "## Transactions to Categorize\n"
        f"{txn_text}\n"
        "\n"
        "## Response Format\n"
        'Return a JSON array. Each element:\n'
        '{"merchant": "...", "category": "...", "subcategory": "..."}\n'
        "\n"
        "Use only categories and subcategories from the taxonomy above.\n"
        "If no subcategory applies, use an empty string for subcategory."
    )


def _parse_response(text: str) -> list[dict]:
    """Extract and parse the JSON array from the LLM response text.

    The LLM may wrap the JSON in markdown code fences or include
    explanatory text. This function finds the first '[' and last ']'
    to extract the array.

    Args:
        text: Raw text from the LLM response.

    Returns:
        Parsed list of dicts, or empty list if parsing fails.
    """
    # Find the JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM response does not contain a JSON array")
        return []

    json_str = text[start : end + 1]
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON from LLM response: %s", exc)
        return []

    if not isinstance(result, list):
        logger.warning("LLM response JSON is not a list")
        return []

    # Validate each element has the required keys
    validated: list[dict] = []
    for item in result:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item in LLM response: %s", item)
            continue
        if "merchant" not in item or "category" not in item:
            logger.warning("Skipping item missing required keys: %s", item)
            continue
        validated.append(
            {
                "merchant": str(item["merchant"]),
                "category": str(item["category"]),
                "subcategory": str(item.get("subcategory", "")),
            }
        )

    return validated


class AnthropicAdapter:
    """LLM adapter that calls the Anthropic Messages API via httpx.

    Reads the API key from the environment variable specified in config
    (``api_key_env``). Constructs a single prompt containing all
    uncategorized transactions and the category taxonomy, sends one
    HTTP POST, and parses the structured JSON response.

    On any failure (missing API key, network error, auth error, rate
    limit, unparseable response), returns an empty list. The categorizer
    treats this as "LLM unavailable" and leaves transactions uncategorized.

    Args:
        model: The Anthropic model identifier, e.g. "claude-sonnet-4-20250514".
        api_key_env: Name of the environment variable containing the API key.
        max_tokens: Maximum tokens in the LLM response. Default: 4096.
        timeout: HTTP request timeout in seconds. Default: 60.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str,
        max_tokens: int = 4096,
        timeout: float = 60.0,
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
        """Send a batch of transactions to Anthropic for categorization.

        Args:
            transactions: List of dicts with merchant, description, amount, date.
            categories: Category taxonomy as list of dicts with name and subcategories.

        Returns:
            List of suggestion dicts with merchant, category, subcategory.
            Empty list on any failure.
        """
        if not transactions:
            return []

        # Read the API key from the environment
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            logger.warning(
                "LLM API key not found in environment variable '%s'",
                self.api_key_env,
            )
            return []

        prompt = _build_prompt(transactions, categories)

        request_body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        try:
            response = httpx.post(
                ANTHROPIC_API_URL,
                json=request_body,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.warning("LLM request timed out")
            return []
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "LLM API returned HTTP %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return []
        except httpx.HTTPError as exc:
            logger.warning("LLM request failed: %s", exc)
            return []

        # Extract text from the Anthropic response format
        try:
            body = response.json()
            content_blocks = body.get("content", [])
            text_parts: list[str] = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
            response_text = "\n".join(text_parts)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to extract text from LLM response: %s", exc)
            return []

        if not response_text:
            logger.warning("LLM response contained no text content")
            return []

        return _parse_response(response_text)


class NullAdapter:
    """No-op LLM adapter for --no-llm mode.

    Always returns an empty list. Used when LLM categorization is
    disabled via the --no-llm flag or when llm_provider is set to "none"
    in config.
    """

    def categorize_batch(
        self,
        transactions: list[dict],
        categories: list[dict],
    ) -> list[dict]:
        """Return an empty list (no LLM categorization).

        Args:
            transactions: Ignored.
            categories: Ignored.

        Returns:
            An empty list, always.
        """
        return []
