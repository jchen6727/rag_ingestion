"""
JSON-output quality scoring for poster/plot_results.py.

Separate from rag_check/ (which only measures latency): these heuristics
exist purely to characterize the failure mode the poster figure needs to
surface — a response that came back *fast* for a reason other than
genuinely efficient retrieval+generation. Two distinct mechanisms count as
a hard failure here, and are kept as separate quality_flag values rather
than lumped together, because `poster/trials/trial_0.md` found they
dominate different arms:

  - `declined_insufficient_context`: the model explicitly refused (per
    compare_latency.py's system preamble, which instructs exactly this when
    retrieved context doesn't cover the question) rather than attempting
    the requested structure at all. In trial_0 this dominated the
    `filtered` arm's failures — a narrow `top_k` makes "nothing retrieved
    looks relevant enough" far more likely than it is for `full_corpus`,
    which has the entire corpus to search.
  - `malformed_json` / `missing_keys` / `exceeded_char_budget`: the model
    DID attempt the structure but botched it (truncated, wrong shape,
    missing a required key, or blew through an explicit character budget).

Trial_0 ALSO hard-failed on a third category, `short_lists` (a list field
with fewer items than a prompt-mandated minimum) — removed after trial_0.
Minimum item counts turned out to be confounded with the thing being
measured: mandating "at least 3 cautions" when only 1-2 are genuinely
relevant forces a choice between fabricating filler or refusing, so a
"failure" there was often really a side effect of an unrealistic minimum,
not evidence of the model's reliability. See poster/trials/trial_0.md.

In its place, `list_fields` responses are now scored for **cross-repeat
variability** (coefficient of variation of item count across the 5 repeats
of the same tier/arm — computed in plot_results.py, not here, since it's
an aggregate over rows) as a softer inaccuracy proxy: a field whose item
count swings wildly run to run, even when every individual response parses
fine, suggests the model is improvising structure rather than reporting
something stable it actually found.
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
# question — see module docstring.
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
    # {field_name: item_count}; None for a list field that was missing or
    # not actually a list. Consumed by plot_results.py to compute
    # cross-repeat variability — not judged pass/fail here.
    list_item_counts: dict[str, Optional[int]] = field(default_factory=dict)
    response_chars: int = 0
    json_chars: Optional[int] = None  # length of just the extracted JSON, for max_chars checks
    within_char_budget: Optional[bool] = None  # None if the tier has no max_chars
    # "n/a" (free-text tier) | "ok" | "empty_response" |
    # "declined_insufficient_context" | "malformed_json" | "missing_keys" |
    # "exceeded_char_budget"
    quality_flag: str = "n/a"


# Flags that count as a genuine hard failure for the poster's failure-rate
# panel. "n/a" (free-text tiers never asked for structure) and "ok" don't.
FAILURE_FLAGS = frozenset({
    "empty_response", "declined_insufficient_context",
    "malformed_json", "missing_keys", "exceeded_char_budget",
})


def evaluate_quality(
    answer_text: str,
    expects_json: bool,
    required_keys: Optional[list[str]] = None,
    list_fields: Optional[list[str]] = None,
    max_chars: Optional[int] = None,
) -> QualityResult:
    """Score one response against its tier's expectations.

    Args:
        answer_text: The model's raw response text.
        expects_json: Whether this tier's prompt asked for a JSON object.
        required_keys: Keys that must be present in the parsed JSON for the
            response to count as structurally complete (no count/length
            requirement — just presence).
        list_fields: Keys expected to hold a JSON array; their item counts
            are recorded (not judged) for plot_results.py's cross-repeat
            variability metric.
        max_chars: If set, the extracted JSON substring must be no longer
            than this many characters — flags `exceeded_char_budget`.

    Returns:
        A QualityResult; `quality_flag` is the single value the poster's
        failure-rate panel aggregates on, `list_item_counts` feeds the
        separate variability panel.
    """
    text = answer_text or ""
    result = QualityResult(expects_json=expects_json, response_chars=len(text))

    if not text.strip():
        result.quality_flag = "empty_response"
        return result

    if not expects_json:
        result.quality_flag = "n/a"
        return result

    candidate = _extract_json_candidate(text)
    result.json_found = candidate is not None
    if candidate is None:
        # Refusal check only applies here, to the case where NO JSON was
        # attempted at all — e.g. "The context does not contain a relevant
        # answer." with nothing else. Checking this against the raw text
        # BEFORE extraction (as an earlier version of this function did)
        # produces false positives: a response can legitimately produce
        # valid, well-formed JSON whose *content* says the guidance doesn't
        # address the scenario (e.g. {"ADDRESSED_BY_GUIDANCE": false, ...})
        # — that's the model correctly reporting a negative finding inside
        # the requested structure, not refusing to use it, and must score
        # as "ok" once required_keys/list_fields checks below pass.
        result.quality_flag = (
            "declined_insufficient_context" if _REFUSAL_RE.search(text) else "malformed_json"
        )
        return result

    result.json_chars = len(candidate)
    if max_chars is not None:
        result.within_char_budget = result.json_chars <= max_chars

    try:
        parsed: Any = json.loads(candidate)
    except json.JSONDecodeError:
        result.quality_flag = "malformed_json"
        return result

    result.json_valid = True
    if not isinstance(parsed, dict):
        result.quality_flag = "malformed_json"
        return result

    for key in (required_keys or []):
        if key not in parsed:
            result.missing_keys.append(key)

    for key in (list_fields or []):
        value = parsed.get(key)
        result.list_item_counts[key] = len(value) if isinstance(value, list) else None

    if result.missing_keys:
        result.quality_flag = "missing_keys"
    elif result.within_char_budget is False:
        result.quality_flag = "exceeded_char_budget"
    else:
        result.quality_flag = "ok"
    return result
