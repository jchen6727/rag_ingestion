"""
Provider-agnostic interface for the "call an LLM, get structured JSON back"
step used by metadata_gen.py.

Every concrete provider (Gemini today; openai/anthropic/mistralai/openrouter as
future siblings — see ingestion/llm/__init__.py::build_llm_client) implements
LLMClient and is responsible for translating its own SDK's exceptions into one
of the four categories below. Callers (metadata_gen.py, inspect_chunks.py) then
handle and log every provider the same way, via log_llm_error(), instead of
special-casing each SDK's exception types.

The categories exist because they call for different operator responses:
  - LLMConnectivityError — transient, safe to retry, nothing to inspect.
  - LLMAPIError          — a credentials/quota/request problem; check config.
  - LLMGenerationError    — the call succeeded but the MODEL's output was bad
                            (non-JSON, schema-invalid). This is the one that
                            silently vanished before: the fallback path caught
                            it right alongside API errors and never surfaced
                            what the model actually produced. It carries
                            `raw_text` for exactly that reason.
  - LLMModelUnavailableError — fatal; every subsequent call would fail the same
                            way, so callers should let this propagate rather
                            than retry/fallback per-chunk.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

# Characters of raw model output to include in a single log line. Long enough
# to see a truncated/malformed JSON body, short enough not to flood logs when
# batch-processing hundreds of chunks.
_MAX_RAW_TEXT_CHARS = 4000


class LLMError(Exception):
    """Base class for every error raised by an LLMClient implementation."""


class LLMConnectivityError(LLMError):
    """Transport-level failure reaching the provider (timeout, DNS, connection
    reset, TLS) — no model output was produced. Safe to retry."""


class LLMAPIError(LLMError):
    """The provider's API rejected or failed the request (auth, quota, invalid
    argument, service error) — an account/request-level problem, not the
    model's output."""


class LLMGenerationError(LLMError):
    """The call succeeded at the transport/API level, but the model's output
    could not be used (non-JSON, empty, or schema-invalid after coercion).

    Carries `raw_text`: whatever text the model actually produced, so it can be
    logged and inspected instead of being discarded on the fallback path.
    """

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


class LLMModelUnavailableError(LLMError):
    """The configured model can't be served at all in this deployment/region.

    Fatal — never caught by a per-call fallback path, since every subsequent
    call would fail identically. Callers should let it propagate and halt.
    """


class LLMClient(ABC):
    """Provider-agnostic interface for extracting structured JSON from a chunk
    of text via a chat/completion model.

    A concrete subclass wraps one provider's SDK (see ingestion/llm/gemini.py
    for google-genai/Vertex AI) and translates that SDK's exceptions into the
    LLMError subclasses above.
    """

    @abstractmethod
    def verify_model_available(self) -> None:
        """Eagerly confirm the configured model is actually servable.

        Call once at startup so a misconfigured/unavailable model fails fast
        with one clear error instead of degrading every subsequent call to a
        fallback over a long batch run.

        Raises:
            LLMModelUnavailableError: If the model can't be reached.
        """

    @abstractmethod
    def generate_json(self, prompt: str, *, temperature: Optional[float] = None) -> dict:
        """Send `prompt` to the model and return its response parsed as JSON.

        Args:
            prompt: Full prompt text.
            temperature: Sampling temperature; None uses the client's default.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            LLMConnectivityError: Transport-level failure.
            LLMAPIError: The provider rejected or failed the request.
            LLMGenerationError: The model responded but its output isn't usable
                JSON (`raw_text` carries what it actually produced).
            LLMModelUnavailableError: The model isn't servable at all.
        """


def log_llm_error(logger: logging.Logger, exc: BaseException, context: str = "") -> None:
    """Log an LLMError (or any exception) with category-appropriate detail.

    This is the "segregated handling and reporting" point for the three
    per-call error categories, plus the fatal model-unavailable case:
      - LLMGenerationError logs the raw model output (truncated), which is the
        piece that previously vanished into a generic warning.
      - LLMConnectivityError / LLMAPIError log a short, category-labeled line
        so an operator can immediately tell "retry" from "check config" apart.
      - Anything else (including non-LLM exceptions passed in by callers that
        haven't been fully migrated) gets a generic fallback line so no
        information is silently dropped.

    Args:
        logger: Logger to emit on.
        exc: The caught exception.
        context: Optional short description of what was being attempted
            (e.g. "tagging chunk abc123_00007").
    """
    where = f" while {context}" if context else ""
    name = type(exc).__name__

    if isinstance(exc, LLMGenerationError):
        raw = (exc.raw_text or "").strip()
        if raw:
            preview = raw[:_MAX_RAW_TEXT_CHARS]
            note = (
                f" (showing first {_MAX_RAW_TEXT_CHARS} of {len(raw)} chars)"
                if len(raw) > _MAX_RAW_TEXT_CHARS
                else ""
            )
            logger.error(
                "LLM generation problem%s [%s]: %s\n"
                "  -> Raw model output%s:\n%s",
                where, name, exc, note, preview,
            )
        else:
            logger.error(
                "LLM generation problem%s [%s]: %s\n"
                "  -> No output text was captured.",
                where, name, exc,
            )
    elif isinstance(exc, LLMConnectivityError):
        logger.error(
            "LLM connectivity problem%s [%s]: %s\n"
            "  -> Transient/network failure. Safe to retry; check API endpoint reachability.",
            where, name, exc,
        )
    elif isinstance(exc, LLMAPIError):
        logger.error(
            "LLM API problem%s [%s]: %s\n"
            "  -> Check credentials, quota, and request parameters for the configured provider.",
            where, name, exc,
        )
    elif isinstance(exc, LLMModelUnavailableError):
        logger.error("LLM model unavailable%s [%s]: %s", where, name, exc)
    else:
        logger.error("Unrecognized error%s [%s]: %s", where, name, exc)

    logger.debug("Full traceback:", exc_info=exc)
