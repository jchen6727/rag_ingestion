"""
Complexity ladder for the poster's filtered-RAG-vs-whole-corpus-RAG figure.

Single clinical topic throughout (suicidal-ideation disclosure, matching
`poster/sweep_example.csv`'s seed cases) with ONE dimension escalated per
tier so the ladder isolates what actually gets harder:

  1-2: direct, explicit disclosure           (recognition is trivial)
  3-5: implicit disclosure, needs inference   (+ increasingly strict JSON)
  6:   ambiguous disclosure, needs judgment   (is this even a disclosure?)
  7-8: disclosure buried in a longer narrative (+ grounded/cited JSON)
  9:   two disclosures across a multi-turn transcript, most complex JSON

Each tier's `schema` (when `expects_json`) is consumed by
`poster/quality.py::evaluate_quality()` to score a response: missing keys,
or a list field with fewer items than the prompt asked for, are exactly the
"significantly less elements" / malformed-JSON failure mode the poster is
trying to characterize — see poster/README.md.
"""

from __future__ import annotations

TIERS: list[dict] = [
    dict(
        tier=1,
        label="direct_openended",
        prompt=(
            "How should a licensed clinical therapist respond when their client "
            "discloses suicidal ideation?"
        ),
        expects_json=False,
    ),
    dict(
        tier=2,
        label="direct_structured",
        prompt=(
            "How should a licensed clinical therapist respond when their client "
            "discloses suicidal ideation? Answer in exactly 3 numbered, "
            "one-sentence immediate actions."
        ),
        expects_json=False,
    ),
    dict(
        tier=3,
        label="implicit_openended",
        prompt=(
            "How should a licensed clinical therapist respond when their client "
            "states \"If a car hit me today, I wouldn't really care\"?"
        ),
        expects_json=False,
    ),
    dict(
        tier=4,
        label="implicit_json_simple",
        prompt=(
            "How should a therapist respond when their client states \"If a car "
            "hit me today, I wouldn't really care\"? Return a JSON object with "
            "keys CAUTIONS (a list of clinical cautions) and ACTIONS (a list of "
            "necessary responses)."
        ),
        expects_json=True,
        schema={"CAUTIONS": {"min_items": 1}, "ACTIONS": {"min_items": 1}},
    ),
    dict(
        tier=5,
        label="implicit_json_strict",
        prompt=(
            "How should a therapist respond when their client states \"If a car "
            "hit me today, I wouldn't really care\"? Return a JSON object with "
            "exactly these keys: RISK_LEVEL (one of low, moderate, high, "
            "imminent), CAUTIONS (a list of at least 3 clinical cautions), "
            "ACTIONS (a list of at least 3 necessary responses), and "
            "FOLLOW_UP_QUESTIONS (a list of at least 2 questions the therapist "
            "should ask next)."
        ),
        expects_json=True,
        schema={
            "RISK_LEVEL": {},
            "CAUTIONS": {"min_items": 3},
            "ACTIONS": {"min_items": 3},
            "FOLLOW_UP_QUESTIONS": {"min_items": 2},
        },
    ),
    dict(
        tier=6,
        label="ambiguous_classification",
        prompt=(
            "A client says, half-laughing, \"Honestly, sometimes I think "
            "everyone would just be better off if I wasn't around, but "
            "whatever, I'm fine.\" Determine whether this statement constitutes "
            "a disclosure requiring a suicidal ideation risk response. Return a "
            "JSON object with keys IS_DISCLOSURE (true or false), CONFIDENCE (a "
            "number from 0 to 1), RATIONALE (a one-sentence explanation), and "
            "RECOMMENDED_ACTION (a one-sentence next step)."
        ),
        expects_json=True,
        schema={
            "IS_DISCLOSURE": {},
            "CONFIDENCE": {},
            "RATIONALE": {},
            "RECOMMENDED_ACTION": {},
        },
    ),
    dict(
        tier=7,
        label="narrative_json",
        prompt=(
            "A client spends most of a session discussing work stress, a "
            "recent move, and conflict with a sibling. Partway through, almost "
            "in passing, they say \"If a car hit me today, I wouldn't really "
            "care.\" They then continue describing their moving logistics for "
            "another two minutes before the session ends. Identify the "
            "risk-relevant disclosure in this session, assess its severity, and "
            "return a JSON object with keys IDENTIFIED_DISCLOSURE (quote the "
            "exact relevant sentence), SEVERITY (one of low, moderate, high, "
            "imminent), CAUTIONS (a list of at least 3 clinical cautions), and "
            "ACTIONS (a list of at least 3 necessary responses)."
        ),
        expects_json=True,
        schema={
            "IDENTIFIED_DISCLOSURE": {},
            "SEVERITY": {},
            "CAUTIONS": {"min_items": 3},
            "ACTIONS": {"min_items": 3},
        },
    ),
    dict(
        tier=8,
        label="narrative_json_grounded",
        prompt=(
            "A client spends most of a session discussing work stress, a "
            "recent move, and conflict with a sibling. Partway through, almost "
            "in passing, they say \"If a car hit me today, I wouldn't really "
            "care.\" They then continue describing their moving logistics for "
            "another two minutes before the session ends. Identify the "
            "risk-relevant disclosure, assess its severity, and return a JSON "
            "object with keys IDENTIFIED_DISCLOSURE, SEVERITY (one of low, "
            "moderate, high, imminent), CAUTIONS (a list of at least 3 clinical "
            "cautions), ACTIONS (a list of at least 3 necessary responses), and "
            "SUPPORTING_GUIDANCE (a list of at least 2 objects, each with keys "
            "SOURCE and RATIONALE, grounding each action in a specific piece of "
            "clinical guidance)."
        ),
        expects_json=True,
        schema={
            "IDENTIFIED_DISCLOSURE": {},
            "SEVERITY": {},
            "CAUTIONS": {"min_items": 3},
            "ACTIONS": {"min_items": 3},
            "SUPPORTING_GUIDANCE": {"min_items": 2},
        },
    ),
    dict(
        tier=9,
        label="multiturn_json_nested",
        prompt=(
            "Session transcript excerpt:\n"
            "Therapist: How has your week been?\n"
            "Client: Busy. Work has been a lot, and I've barely been sleeping.\n"
            "Therapist: That sounds exhausting. How are you coping?\n"
            "Client: Honestly? If a car hit me today, I wouldn't really care. "
            "But it's fine, I have a deadline Friday so I can't exactly stop.\n"
            "Therapist: Let's come back to that in a moment. Tell me about the "
            "deadline.\n"
            "Client: It's a report for my manager. Nothing special. Also, I've "
            "been thinking about actually ending things, I just don't know when "
            "I'd do it.\n"
            "Identify every risk-relevant disclosure in this transcript, in "
            "order, assess the combined risk picture, and return a JSON object "
            "with keys DISCLOSURES (a list of exactly 2 objects, each with keys "
            "TEXT, TYPE, and SEVERITY), OVERALL_RISK_LEVEL (one of low, "
            "moderate, high, imminent), CAUTIONS (a list of at least 3 clinical "
            "cautions), ACTIONS (a list of at least 3 necessary responses), and "
            "SUPPORTING_GUIDANCE (a list of at least 2 objects, each with keys "
            "SOURCE and RATIONALE)."
        ),
        expects_json=True,
        schema={
            "DISCLOSURES": {"min_items": 2},
            "OVERALL_RISK_LEVEL": {},
            "CAUTIONS": {"min_items": 3},
            "ACTIONS": {"min_items": 3},
            "SUPPORTING_GUIDANCE": {"min_items": 2},
        },
    ),
]

TIERS_BY_NUMBER: dict[int, dict] = {t["tier"]: t for t in TIERS}

# Shared sweep constants (poster/sweep.py) — single topic/tag throughout, so
# the ladder isolates prompt complexity, not retrieval scope.
TAG_FILTER = "session_event_tags=disclosure_SI"
MODEL = "gemini-2.5-flash"
TOP_K = 10
TEMPERATURE = 0.0
BASE_SEED = 42
N_REPEATS = 5
