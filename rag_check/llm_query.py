"""
Thin, direct google-genai wrapper for rag_check's LLM calls, with wall-clock
latency measurement.

Deliberately does NOT go through ingestion/llm/ (LLMClient / build_llm_client).
That abstraction's one method, generate_json(), is scoped to the ingestion
pipeline's contract of "always force response_mime_type=json and return
ChunkMetadata-shaped output" (see ingestion/llm/base.py) — it has no seed,
thinking-budget, top_p/top_k, or max-output-tokens knobs, because metadata
generation never needed them. rag_check needs plain-text generation with
those request-level `genai` parameters exposed directly for a latency
comparison, so it talks to google.genai itself rather than stretch
LLMClient's narrower interface to fit a different use case.

Auth matches the rest of the repo: Vertex AI via Application Default
Credentials, routed through settings.gcp_project_id / settings.gcp_location
(the raw compute location — Gemini calls use this, NOT the Discovery Engine
multi-region; see CLAUDE.md "Region routing").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types

from config.settings import settings


@dataclass
class GenAIParams:
    """Request-level generation knobs, all exposed on the rag_check CLI.

    `thinking_budget`: token budget for Gemini's internal "thinking" (only
    meaningful on models that support it, e.g. the 2.5 family). None leaves
    it at the model default; 0 disables thinking where the model allows it.
    """

    model: str
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None


@dataclass
class LLMCallResult:
    text: str
    latency_seconds: float
    usage: dict = field(default_factory=dict)


def _build_config(params: GenAIParams) -> types.GenerateContentConfig:
    kwargs: dict = {"temperature": params.temperature}
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.top_k is not None:
        kwargs["top_k"] = params.top_k
    if params.seed is not None:
        kwargs["seed"] = params.seed
    if params.max_output_tokens is not None:
        kwargs["max_output_tokens"] = params.max_output_tokens
    if params.thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=params.thinking_budget
        )
    return types.GenerateContentConfig(**kwargs)


def run_query(prompt: str, params: GenAIParams) -> LLMCallResult:
    """Send `prompt` to Vertex AI Gemini and time the round trip.

    Args:
        prompt: Full prompt text (context + question already assembled by
            the caller — see compare_latency.py::build_prompt).
        params: Model/sampling/thinking configuration.

    Returns:
        LLMCallResult with the response text, wall-clock latency in seconds
        (time.perf_counter around the single generate_content call — no
        retry/backoff here, unlike ingestion/llm/gemini.py, since retries
        would corrupt the latency measurement this tool exists to produce),
        and whatever usage_metadata the SDK reports (token counts).

    Raises:
        Whatever google-genai/google-api-core raises on failure — not
        wrapped in LLMError here; this is a standalone measurement script,
        not part of the ingestion error-handling path.
    """
    client = genai.Client(
        vertexai=True,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )
    config = _build_config(params)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=params.model,
        contents=prompt,
        config=config,
    )
    elapsed = time.perf_counter() - start

    usage: dict = {}
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        usage = {
            "prompt_token_count": getattr(um, "prompt_token_count", None),
            "candidates_token_count": getattr(um, "candidates_token_count", None),
            "thoughts_token_count": getattr(um, "thoughts_token_count", None),
            "total_token_count": getattr(um, "total_token_count", None),
        }

    return LLMCallResult(
        text=getattr(response, "text", None) or "",
        latency_seconds=elapsed,
        usage=usage,
    )
