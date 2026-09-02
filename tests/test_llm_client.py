"""
Tests for the provider-agnostic LLM abstraction in ingestion/llm/.

Covers the GeminiClient's error categorization (connectivity vs API vs
generation vs model-unavailable), the retry/no-retry behavior for each, and
log_llm_error's per-category log output. No live API calls — the google-genai
client is stubbed.

Run:
    pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import logging

import pytest

from ingestion.llm import build_llm_client
from ingestion.llm.base import (
    LLMAPIError,
    LLMConnectivityError,
    LLMGenerationError,
    LLMModelUnavailableError,
    log_llm_error,
)
from ingestion.llm.gemini import GeminiClient


# ---------------------------------------------------------------------------
# Fakes for google.genai.Client — no network, deterministic responses.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    """Returns/raises the next item in a scripted sequence on each call."""

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self.call_count = 0

    def generate_content(self, **kwargs):
        self.call_count += 1
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _FakeResponse(item)


class _FakeGenaiClient:
    def __init__(self, items: list) -> None:
        self.models = _FakeModels(items)


def _client_with_scripted_calls(items: list, **kwargs) -> GeminiClient:
    """A GeminiClient whose underlying SDK client is pre-set to a fake that
    returns/raises `items` in order — bypasses _get_client()'s real
    genai.Client() construction and verify ping."""
    client = GeminiClient(model_name="gemini-x", project_id="p", location="l", **kwargs)
    client._client = _FakeGenaiClient(items)
    client._model_verified = True
    return client


# ---------------------------------------------------------------------------
# GeminiClient.generate_json
# ---------------------------------------------------------------------------


class TestGenerateJson:
    def test_returns_parsed_json_on_success(self) -> None:
        client = _client_with_scripted_calls(['{"a": 1}'])
        assert client.generate_json("prompt") == {"a": 1}

    def test_non_json_output_raises_generation_error_with_raw_text(self) -> None:
        client = _client_with_scripted_calls(["not valid json at all"])
        with pytest.raises(LLMGenerationError) as excinfo:
            client.generate_json("prompt")
        assert excinfo.value.raw_text == "not valid json at all"

    def test_non_json_output_is_not_retried(self) -> None:
        """A malformed-JSON response is deterministic at temperature 0 — retrying
        would just reproduce the same bad output, so only one call should happen
        even though max_retries > 1."""
        client = _client_with_scripted_calls(["still not json"], max_retries=3, retry_base_delay=0)
        with pytest.raises(LLMGenerationError):
            client.generate_json("prompt")
        assert client._client.models.call_count == 1

    def test_connectivity_error_is_retried_then_succeeds(self) -> None:
        client = _client_with_scripted_calls(
            [TimeoutError("slow"), '{"ok": true}'], max_retries=3, retry_base_delay=0
        )
        assert client.generate_json("prompt") == {"ok": True}
        assert client._client.models.call_count == 2

    def test_connectivity_error_exhausts_retries_and_raises(self) -> None:
        client = _client_with_scripted_calls(
            [TimeoutError("a"), TimeoutError("b"), TimeoutError("c")],
            max_retries=3, retry_base_delay=0,
        )
        with pytest.raises(LLMConnectivityError):
            client.generate_json("prompt")
        assert client._client.models.call_count == 3

    def test_api_error_is_not_retried(self) -> None:
        """A request-level failure (bad auth/quota/argument) won't be fixed by
        retrying identically, so only one call should happen."""
        client = _client_with_scripted_calls(
            [ValueError("PermissionDenied: no access")], max_retries=3, retry_base_delay=0
        )
        with pytest.raises(LLMAPIError):
            client.generate_json("prompt")
        assert client._client.models.call_count == 1


class TestVerifyModelAvailable:
    def test_raises_model_unavailable_when_ping_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _PingFailModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("404 NOT_FOUND")

        class _PingFailClient:
            def __init__(self, **kwargs) -> None:
                self.models = _PingFailModels()

        monkeypatch.setattr("ingestion.llm.gemini.genai.Client", _PingFailClient)
        client = GeminiClient(model_name="gemini-x", project_id="p", location="l")
        with pytest.raises(LLMModelUnavailableError):
            client.verify_model_available()

    def test_succeeds_and_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        class _OkModels:
            def generate_content(self, **kwargs):
                calls["n"] += 1
                return _FakeResponse("pong")

        class _OkClient:
            def __init__(self, **kwargs) -> None:
                self.models = _OkModels()

        monkeypatch.setattr("ingestion.llm.gemini.genai.Client", _OkClient)
        client = GeminiClient(model_name="gemini-x", project_id="p", location="l")
        client.verify_model_available()
        client.verify_model_available()  # second call should not re-ping
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# build_llm_client factory
# ---------------------------------------------------------------------------


class TestBuildLlmClient:
    def test_gemini_returns_gemini_client(self) -> None:
        client = build_llm_client("gemini", model_name="gemini-x", project_id="p", location="l")
        assert isinstance(client, GeminiClient)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            build_llm_client("not-a-real-provider")


# ---------------------------------------------------------------------------
# log_llm_error — segregated per-category reporting
# ---------------------------------------------------------------------------


class TestLogLlmError:
    def test_generation_error_logs_raw_text(self, caplog: pytest.LogCaptureFixture) -> None:
        exc = LLMGenerationError("bad output", raw_text="{totally broken")
        with caplog.at_level(logging.ERROR):
            log_llm_error(logging.getLogger("test"), exc, "tagging chunk x")
        assert "bad output" in caplog.text
        assert "{totally broken" in caplog.text

    def test_connectivity_error_labeled_distinctly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            log_llm_error(logging.getLogger("test"), LLMConnectivityError("timeout"))
        assert "connectivity" in caplog.text.lower()

    def test_api_error_labeled_distinctly(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            log_llm_error(logging.getLogger("test"), LLMAPIError("quota exceeded"))
        assert "api problem" in caplog.text.lower()

    def test_generation_error_without_raw_text_says_none_captured(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR):
            log_llm_error(logging.getLogger("test"), LLMGenerationError("empty response"))
        assert "no output text was captured" in caplog.text.lower()
