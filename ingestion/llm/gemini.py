"""
LLMClient implementation backed by Vertex AI Gemini via the google-genai SDK.

This is the only concrete provider today; it exists as a sibling module so that
openai/anthropic/mistralai/openrouter clients can be added later (each its own
file implementing LLMClient) without metadata_gen.py or any caller changing —
see ingestion/llm/__init__.py::build_llm_client.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from google import genai
from google.genai import types

from config.settings import settings

from .base import (
    _MAX_RAW_TEXT_CHARS,
    LLMAPIError,
    LLMClient,
    LLMConnectivityError,
    LLMError,
    LLMGenerationError,
    LLMModelUnavailableError,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled each attempt

# Substrings of google.api_core exception class names that indicate a
# transport/availability problem rather than a request/account problem —
# these are the ones worth retrying.
_CONNECTIVITY_MARKERS = (
    "DeadlineExceeded",
    "ServiceUnavailable",
    "ConnectionError",
    "Timeout",
    "TransportError",
    "RetryError",
)


def _log_raw_response(text: str) -> None:
    """DEBUG-log the exact text Gemini returned for one call, truncated.

    This is the only place the raw response text is visible on the SUCCESS
    path (log_llm_error in base.py only surfaces raw_text when a call fails).
    Silent by default (DEBUG); run any script with --verbose
    (scripts/_gcp_logging.py::setup_logging) to see it — useful for auditing
    exactly what the model produced even when generation "worked".
    """
    preview = text[:_MAX_RAW_TEXT_CHARS]
    note = (
        f" (showing first {_MAX_RAW_TEXT_CHARS} of {len(text)} chars)"
        if len(text) > _MAX_RAW_TEXT_CHARS
        else ""
    )
    logger.debug("Gemini raw response%s:\n%s", note, preview)


class GeminiClient(LLMClient):
    """LLMClient backed by Vertex AI Gemini (google-genai SDK).

    Auth: Vertex AI via Application Default Credentials — no API key (see
    CLAUDE.md). Uses the raw `gcp_location` compute region (unlike Discovery
    Engine, which needs the derived multi-region).
    """

    def __init__(
        self,
        model_name: str,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        max_retries: int = _MAX_RETRIES,
        retry_base_delay: float = _RETRY_BASE_DELAY,
    ) -> None:
        """
        Args:
            model_name: Gemini model identifier (e.g. from GEMINI_MODEL_METADATA).
            project_id: GCP project ID; defaults to settings.gcp_project_id.
            location: Vertex AI location; defaults to settings.gcp_location.
            max_retries: Attempts for transient (connectivity/API) failures.
            retry_base_delay: Seconds before the first retry; doubled each attempt.
        """
        self._model_name = model_name
        self._project_id = project_id or settings.gcp_project_id
        self._location = location or settings.gcp_location
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client: Optional[genai.Client] = None
        self._model_verified = False

    def verify_model_available(self) -> None:
        self._get_client()

    def generate_json(self, prompt: str, *, temperature: Optional[float] = None) -> dict:
        """Call Gemini with retry/backoff and return the parsed JSON response.

        Uses response_mime_type="application/json" to request structured
        output. Transport/API failures are retried with exponential backoff;
        a non-JSON response is NOT retried (a deterministic prompt at
        temperature 0 would just reproduce the same malformed output) and is
        raised immediately as LLMGenerationError with the raw text attached.
        """
        client = self._get_client()
        temp = 0.0 if temperature is None else temperature
        last_exc: LLMError = LLMConnectivityError("No attempts made")
        delay = self._retry_base_delay

        for attempt in range(self._max_retries):
            try:
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temp,
                        response_mime_type="application/json",
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — categorized immediately below
                last_exc = self._wrap_call_exception(exc)
                is_last_attempt = attempt >= self._max_retries - 1
                if isinstance(last_exc, LLMConnectivityError) and not is_last_attempt:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise last_exc from exc
            else:
                text = getattr(response, "text", None) or ""
                _log_raw_response(text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise LLMGenerationError(
                        f"Gemini returned non-JSON output: {exc}", raw_text=text,
                    ) from exc

        raise last_exc  # pragma: no cover — loop always returns or raises above

    def _wrap_call_exception(self, exc: Exception) -> LLMError:
        """Categorize a google-genai/api_core exception as connectivity vs API."""
        name = type(exc).__name__
        if any(marker in name for marker in _CONNECTIVITY_MARKERS):
            return LLMConnectivityError(f"{name}: {exc}")
        return LLMAPIError(f"{name}: {exc}")

    def _get_client(self) -> genai.Client:
        """Lazy-initialize the Vertex AI client and verify the model once."""
        if self._client is None:
            client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._location,
            )
            self._verify_model_available(client)
            self._client = client
        return self._client

    def _verify_model_available(self, client: genai.Client) -> None:
        """Confirm `self._model_name` is actually servable in `self._location`.

        `client.models.get()` / `.list()` are NOT sufficient: they hit a global
        model-garden catalog and report a model as present even when it isn't
        deployed to this location. A real `generate_content` call is the only
        signal that matches what generate_json() will actually do. Runs once
        per process (cached via `_model_verified`).
        """
        if self._model_verified:
            return
        try:
            client.models.generate_content(
                model=self._model_name,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=1, temperature=0.0),
            )
        except Exception as exc:
            raise LLMModelUnavailableError(
                f"Gemini model '{self._model_name}' is not usable in Vertex AI "
                f"location '{self._location}' (project '{self._project_id}'). It "
                f"may only be available in a different region (e.g. 'global') "
                f"than GCP_LOCATION. Run `PYTHONPATH=. python scripts/check_llm.py "
                f"--list` to see what's servable here, then update "
                f"GEMINI_MODEL_METADATA in .env.\nUnderlying error: {exc}"
            ) from exc
        self._model_verified = True
