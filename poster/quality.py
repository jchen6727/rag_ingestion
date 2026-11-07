"""
JSON-output quality scoring for poster/plot_results.py.

Separate from rag_check/ (which only measures latency): these heuristics
exist purely to characterize the failure mode the poster figure needs to
surface — a response that came back *fast* for a reason other than
genuinely efficient retrieval+generation. Two distinct mechanisms count as
a "failure" here, and are kept as separate quality_flag values rather than
lumped together, because the live data showed they dominate different arms:

  - `declined_insufficient_context`: the model explicitly refused (per
    compare_latency.py's system preamble, which instructs exactly this when
    retrieved context doesn't cover the question) rather than attempting
    the requested structure at all. Observed live to dominate the
    `filtered` arm's failures on several mid-complexity tiers — a narrow
    `top_k` makes "nothing retrieved looks relevant enough" far more likely
    than it is for `full_corpus`, which has the entire corpus to search.
  - `malformed_json` / `missing_keys` / `short_lists`: the model DID
    attempt the structure but botched it (truncated, wrong shape, or fewer
    list items than asked for) — the "gave up mid-answer" mode originally
    anticipated for a large, unfiltered context.

See poster/prompts.py for each tier's expected schema and poster/README.md
for the full rationale, including which mechanism actually showed up in the
first live run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Strips a leading/trailing ```json ... ``` (or bare ```...```) fence — Gemini
# routinely wraps JSON output in one even when the prompt asks for raw JSON.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# compare_latency.py's system preamble instructs the model to explicitly
# decline rather than guess when retrieved context doesn't cover the
# question. That's a distinct failure mode from "attempted JSON and botched
# it" — conflating the two would hide which mechanism is actually driving a
# given arm's failure rate (observed live: the filtered arm, with a narrow
# top_k, declines far more often than full_corpus, which rarely lacks SOME
# matching chunk) — so it gets its own quality_flag.
_REFUSAL_RE = re.compile(
    r"\b(does not|do not|doesn't|don't)\s+contain\b.{0,40}\b(relevant|information|guidance)",
    re.IGNORECASE,
)


def _extract_json_candidate(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object substring from free-form model
    output: strip a markdown code fence, then try the whole remainder, then
    fall back to the first '{' .. last '}' span."""
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]
    return None


@dataclass
class QualityResult:
    expects_json: bool
    json_found: bool = False
    json_valid: bool = False
    missing_keys: list[str] = field(default_factory=list)
    short_list_fields: list[str] = field(default_factory=list)  # "KEY:got<expected"
    response_chars: int = 0
    # "n/a" (free-text tier) | "ok" | "empty_response" | "declined_insufficient_context" |
    # "malformed_json" | "missing_keys" | "short_lists"
    quality_flag: str = "n/a"


# Flags that count as a genuine structural failure for the poster's failure-rate
# panel. "n/a" (free-text tiers never asked for structure) and "ok" don't.
FAILURE_FLAGS = frozenset({
    "empty_response", "declined_insufficient_context",
    "malformed_json", "missing_keys", "short_lists",
})


def evaluate_quality(
    answer_text: str,
    expects_json: bool,
    schema: Optional[dict[str, dict]] = None,
) -> QualityResult:
    """Score one response against its tier's expectations.

    Args:
        answer_text: The model's raw response text.
        expects_json: Whether this tier's prompt asked for a JSON object.
        schema: {key: {"min_items": N}} for keys expected in the parsed JSON;
            `min_items` is only checked for list/dict-valued keys (omit it,
            or leave the dict empty, for a key with no length requirement).

    Returns:
        A QualityResult; `quality_flag` is the single value the poster's
        failure-rate panel aggregates on.
    """
    text = answer_text or ""
    result = QualityResult(expects_json=expects_json, response_chars=len(text))

    if not text.strip():
        result.quality_flag = "empty_response"
        return result

    if not expects_json:
        result.quality_flag = "n/a"
        return result

    if _REFUSAL_RE.search(text):
        result.quality_flag = "declined_insufficient_context"
        return result

    candidate = _extract_json_candidate(text)
    result.json_found = candidate is not None
    if candidate is None:
        result.quality_flag = "malformed_json"
        return result

    try:
        parsed: Any = json.loads(candidate)
    except json.JSONDecodeError:
        result.quality_flag = "malformed_json"
        return result

    result.json_valid = True
    if not isinstance(parsed, dict):
        result.quality_flag = "malformed_json"
        return result

    for key, spec in (schema or {}).items():
        if key not in parsed:
            result.missing_keys.append(key)
            continue
        min_items = spec.get("min_items")
        if min_items is None:
            continue
        value = parsed[key]
        length = len(value) if isinstance(value, (list, dict)) else None
        if length is None or length < min_items:
            result.short_list_fields.append(f"{key}:{length}<{min_items}")

    if result.missing_keys:
        result.quality_flag = "missing_keys"
    elif result.short_list_fields:
        result.quality_flag = "short_lists"
    else:
        result.quality_flag = "ok"
    return result
