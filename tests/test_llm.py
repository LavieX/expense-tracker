"""Tests for expense_tracker.llm -- LLM adapter, prompt construction, and response parsing.

All tests use mocked HTTP responses. No real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from expense_tracker.llm import (
    AnthropicAdapter,
    NullAdapter,
    _build_prompt,
    _parse_response,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_TRANSACTIONS = [
    {
        "merchant": "TARGET 00022186",
        "description": "TARGET        00022186",
        "amount": "-127.98",
        "date": "2026-01-25",
    },
    {
        "merchant": "REI.COM 800-426-4840",
        "description": "REI.COM  800-426-4840",
        "amount": "-160.65",
        "date": "2026-01-12",
    },
]

SAMPLE_CATEGORIES = [
    {"name": "Food & Dining", "subcategories": ["Groceries", "Restaurant", "Fast Food"]},
    {"name": "Shopping", "subcategories": ["Clothing", "Electronics", "Home Goods"]},
    {"name": "Health & Fitness", "subcategories": ["Gym/Classes", "Skiing"]},
    {"name": "Insurance", "subcategories": []},
]

SAMPLE_LLM_SUGGESTIONS = [
    {"merchant": "TARGET 00022186", "category": "Shopping", "subcategory": "Home Goods"},
    {"merchant": "REI.COM 800-426-4840", "category": "Shopping", "subcategory": "Clothing"},
]


def _make_anthropic_response(suggestions: list[dict]) -> dict:
    """Build a mock Anthropic Messages API response body."""
    return {
        "id": "msg_test123",
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
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


# ---------------------------------------------------------------------------
# _build_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_contains_taxonomy(self):
        """Prompt includes all category names and their subcategories."""
        prompt = _build_prompt(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)
        assert "## Category Taxonomy" in prompt
        assert "Food & Dining: Groceries, Restaurant, Fast Food" in prompt
        assert "Shopping: Clothing, Electronics, Home Goods" in prompt
        assert "Health & Fitness: Gym/Classes, Skiing" in prompt
        # Insurance has no subcategories -- listed without colon-separated subs
        assert "- Insurance" in prompt

    def test_contains_transactions(self):
        """Prompt includes all transaction details in pipe-delimited format."""
        prompt = _build_prompt(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)
        assert "## Transactions to Categorize" in prompt
        assert "TARGET 00022186 | TARGET        00022186 | -127.98 | 2026-01-25" in prompt
        assert "REI.COM 800-426-4840 | REI.COM  800-426-4840 | -160.65 | 2026-01-12" in prompt

    def test_contains_response_format_instructions(self):
        """Prompt includes JSON response format instructions."""
        prompt = _build_prompt(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)
        assert "## Response Format" in prompt
        assert "JSON array" in prompt
        assert '"merchant"' in prompt
        assert '"category"' in prompt
        assert '"subcategory"' in prompt

    def test_contains_system_instructions(self):
        """Prompt starts with the categorization instruction."""
        prompt = _build_prompt(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)
        assert prompt.startswith("You are categorizing household expenses.")

    def test_empty_transactions(self):
        """Prompt is still well-formed with zero transactions."""
        prompt = _build_prompt([], SAMPLE_CATEGORIES)
        assert "## Transactions to Categorize" in prompt
        assert "## Category Taxonomy" in prompt

    def test_empty_categories(self):
        """Prompt is still well-formed with zero categories."""
        prompt = _build_prompt(SAMPLE_TRANSACTIONS, [])
        assert "## Category Taxonomy" in prompt
        assert "## Transactions to Categorize" in prompt


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for LLM response parsing."""

    def test_clean_json_array(self):
        """Parses a clean JSON array."""
        text = json.dumps(SAMPLE_LLM_SUGGESTIONS)
        result = _parse_response(text)
        assert len(result) == 2
        assert result[0]["merchant"] == "TARGET 00022186"
        assert result[0]["category"] == "Shopping"
        assert result[0]["subcategory"] == "Home Goods"
        assert result[1]["merchant"] == "REI.COM 800-426-4840"

    def test_json_in_code_fence(self):
        """Parses JSON wrapped in markdown code fences."""
        text = (
            "Here are the categorizations:\n\n"
            "```json\n"
            + json.dumps(SAMPLE_LLM_SUGGESTIONS)
            + "\n```"
        )
        result = _parse_response(text)
        assert len(result) == 2

    def test_json_with_surrounding_text(self):
        """Parses JSON with explanatory text before and after."""
        text = (
            "I've analyzed the transactions. Here are my suggestions:\n\n"
            + json.dumps(SAMPLE_LLM_SUGGESTIONS)
            + "\n\nLet me know if you need any changes."
        )
        result = _parse_response(text)
        assert len(result) == 2

    def test_missing_subcategory_defaults_to_empty(self):
        """Items without subcategory get an empty string default."""
        text = json.dumps([{"merchant": "COSTCO", "category": "Shopping"}])
        result = _parse_response(text)
        assert len(result) == 1
        assert result[0]["subcategory"] == ""

    def test_no_json_array_returns_empty(self):
        """Returns empty list when no JSON array is found."""
        result = _parse_response("I don't know how to categorize these.")
        assert result == []

    def test_invalid_json_returns_empty(self):
        """Returns empty list for malformed JSON."""
        result = _parse_response('[{"merchant": "TEST", "category": incomplete')
        assert result == []

    def test_non_list_json_returns_empty(self):
        """Returns empty list when JSON is not an array."""
        result = _parse_response('{"merchant": "TEST", "category": "Shopping"}')
        assert result == []

    def test_skips_items_missing_required_keys(self):
        """Items missing merchant or category are skipped."""
        text = json.dumps(
            [
                {"merchant": "TARGET", "category": "Shopping", "subcategory": ""},
                {"merchant": "REI"},  # missing category
                {"category": "Pets"},  # missing merchant
                {"merchant": "CVS", "category": "Healthcare", "subcategory": "Pharmacy"},
            ]
        )
        result = _parse_response(text)
        assert len(result) == 2
        assert result[0]["merchant"] == "TARGET"
        assert result[1]["merchant"] == "CVS"

    def test_skips_non_dict_items(self):
        """Non-dict items in the array are skipped."""
        text = json.dumps(
            [
                {"merchant": "TARGET", "category": "Shopping", "subcategory": ""},
                "not a dict",
                42,
                None,
            ]
        )
        result = _parse_response(text)
        assert len(result) == 1

    def test_empty_array(self):
        """An empty JSON array returns an empty list."""
        result = _parse_response("[]")
        assert result == []

    def test_values_coerced_to_strings(self):
        """Non-string values for merchant/category/subcategory are coerced."""
        text = json.dumps([{"merchant": 123, "category": True, "subcategory": None}])
        result = _parse_response(text)
        assert len(result) == 1
        assert result[0]["merchant"] == "123"
        assert result[0]["category"] == "True"
        assert result[0]["subcategory"] == "None"


# ---------------------------------------------------------------------------
# NullAdapter tests
# ---------------------------------------------------------------------------


class TestNullAdapter:
    """Tests for the NullAdapter (no-op for --no-llm mode)."""

    def test_returns_empty_list(self):
        """NullAdapter always returns an empty list."""
        adapter = NullAdapter()
        result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)
        assert result == []

    def test_returns_empty_list_with_empty_inputs(self):
        """NullAdapter returns empty list even with empty inputs."""
        adapter = NullAdapter()
        assert adapter.categorize_batch([], []) == []

    def test_conforms_to_protocol(self):
        """NullAdapter has the categorize_batch method with correct signature."""
        adapter = NullAdapter()
        assert hasattr(adapter, "categorize_batch")
        assert callable(adapter.categorize_batch)


# ---------------------------------------------------------------------------
# AnthropicAdapter tests
# ---------------------------------------------------------------------------


class TestAnthropicAdapter:
    """Tests for the AnthropicAdapter with mocked HTTP responses."""

    def _make_adapter(self, api_key_env: str = "TEST_ANTHROPIC_KEY") -> AnthropicAdapter:
        """Create an adapter with test defaults."""
        return AnthropicAdapter(
            model="claude-sonnet-4-20250514",
            api_key_env=api_key_env,
        )

    # -- Successful categorization --

    def test_successful_categorization(self):
        """Adapter sends correct request and parses successful response."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            json=_make_anthropic_response(SAMPLE_LLM_SUGGESTIONS),
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response) as mock_post,
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert len(result) == 2
        assert result[0]["merchant"] == "TARGET 00022186"
        assert result[0]["category"] == "Shopping"

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["x-api-key"] == "sk-ant-test-key"
        assert call_kwargs.kwargs["headers"]["anthropic-version"] == "2023-06-01"
        assert call_kwargs.kwargs["headers"]["content-type"] == "application/json"

        request_body = call_kwargs.kwargs["json"]
        assert request_body["model"] == "claude-sonnet-4-20250514"
        assert request_body["max_tokens"] == 4096
        assert len(request_body["messages"]) == 1
        assert request_body["messages"][0]["role"] == "user"

    def test_prompt_content_in_request(self):
        """The request body contains the correctly constructed prompt."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            json=_make_anthropic_response(SAMPLE_LLM_SUGGESTIONS),
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response) as mock_post,
        ):
            adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "Category Taxonomy" in prompt
        assert "TARGET 00022186" in prompt
        assert "REI.COM 800-426-4840" in prompt
        assert "Food & Dining" in prompt

    # -- Empty inputs --

    def test_empty_transactions_returns_empty(self):
        """Adapter returns empty list without making an API call for empty input."""
        adapter = self._make_adapter()
        with patch("expense_tracker.llm.httpx.post") as mock_post:
            result = adapter.categorize_batch([], SAMPLE_CATEGORIES)

        assert result == []
        mock_post.assert_not_called()

    # -- Missing API key --

    def test_missing_api_key_returns_empty(self):
        """Adapter returns empty list when the API key env var is not set."""
        adapter = self._make_adapter(api_key_env="NONEXISTENT_KEY_VAR")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("expense_tracker.llm.httpx.post") as mock_post,
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []
        mock_post.assert_not_called()

    def test_empty_api_key_returns_empty(self):
        """Adapter returns empty list when the API key env var is empty string."""
        adapter = self._make_adapter()
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": ""}),
            patch("expense_tracker.llm.httpx.post") as mock_post,
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []
        mock_post.assert_not_called()

    # -- HTTP error handling --

    def test_auth_error_returns_empty(self):
        """Adapter returns empty list on 401 Unauthorized."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-bad-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_rate_limit_returns_empty(self):
        """Adapter returns empty list on 429 Too Many Requests."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_server_error_returns_empty(self):
        """Adapter returns empty list on 500 Internal Server Error."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_timeout_returns_empty(self):
        """Adapter returns empty list on request timeout."""
        adapter = self._make_adapter()
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch(
                "expense_tracker.llm.httpx.post",
                side_effect=httpx.TimeoutException("Connection timed out"),
            ),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_network_error_returns_empty(self):
        """Adapter returns empty list on network connection error."""
        adapter = self._make_adapter()
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch(
                "expense_tracker.llm.httpx.post",
                side_effect=httpx.ConnectError("Connection refused"),
            ),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    # -- Malformed response handling --

    def test_unparseable_response_body_returns_empty(self):
        """Adapter returns empty list when response body is not valid JSON."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            text="This is not JSON",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_response_with_no_content_blocks(self):
        """Adapter returns empty list when response has no content blocks."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            json={"id": "msg_test", "type": "message", "content": []},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_response_with_non_text_content(self):
        """Adapter returns empty list when content blocks have no text type."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "msg_test",
                "type": "message",
                "content": [{"type": "image", "source": {}}],
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    def test_response_text_without_json_returns_empty(self):
        """Adapter returns empty list when LLM text contains no JSON."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "msg_test",
                "type": "message",
                "content": [
                    {"type": "text", "text": "I cannot categorize these transactions."}
                ],
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response),
        ):
            result = adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert result == []

    # -- Custom configuration --

    def test_custom_max_tokens(self):
        """Adapter uses the configured max_tokens value."""
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-20250514",
            api_key_env="TEST_ANTHROPIC_KEY",
            max_tokens=2048,
        )
        mock_response = httpx.Response(
            status_code=200,
            json=_make_anthropic_response(SAMPLE_LLM_SUGGESTIONS),
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response) as mock_post,
        ):
            adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert mock_post.call_args.kwargs["json"]["max_tokens"] == 2048

    def test_custom_timeout(self):
        """Adapter uses the configured timeout value."""
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-20250514",
            api_key_env="TEST_ANTHROPIC_KEY",
            timeout=30.0,
        )
        mock_response = httpx.Response(
            status_code=200,
            json=_make_anthropic_response(SAMPLE_LLM_SUGGESTIONS),
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response) as mock_post,
        ):
            adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert mock_post.call_args.kwargs["timeout"] == 30.0

    def test_conforms_to_protocol(self):
        """AnthropicAdapter has the categorize_batch method with correct signature."""
        adapter = self._make_adapter()
        assert hasattr(adapter, "categorize_batch")
        assert callable(adapter.categorize_batch)

    def test_posts_to_correct_url(self):
        """Adapter sends the POST to the Anthropic Messages API URL."""
        adapter = self._make_adapter()
        mock_response = httpx.Response(
            status_code=200,
            json=_make_anthropic_response(SAMPLE_LLM_SUGGESTIONS),
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        with (
            patch.dict("os.environ", {"TEST_ANTHROPIC_KEY": "sk-ant-test-key"}),
            patch("expense_tracker.llm.httpx.post", return_value=mock_response) as mock_post,
        ):
            adapter.categorize_batch(SAMPLE_TRANSACTIONS, SAMPLE_CATEGORIES)

        assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"
