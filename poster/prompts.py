"""
Complexity ladder for the poster's filtered-RAG-vs-whole-corpus-RAG figure.

Single clinical topic throughout (suicidal-ideation disclosure, matching
`poster/sweep_example.csv`'s seed cases) with ONE dimension escalated per
tier so the ladder isolates what actually gets harder:

  1-2: direct, explicit disclosure           (recognition is trivial)
  3-5: implicit disclosure, needs inference   (+ JSON structure, + priority ordering)
  6:   ambiguous disclosure, needs judgment   (does the guidance even address this?)
  7-8: disclosure buried in a longer narrative (+ grounded/cited JSON)
  9:   two disclosures across a multi-turn transcript + a character budget

Redesigned after `poster/trials/trial_0.md` found two problems with the first
version of this ladder (see that doc for the full writeup):

  1. Several fields (RISK_LEVEL, IS_DISCLOSURE, CONFIDENCE, SEVERITY) asked
     the model to invent a clinical judgment the corpus — treatment manuals,
     not a risk-stratification instrument — doesn't actually contain. Every
     tier here now asks only for content plausibly *extractable* from
     retrieved guidance text: cautions/actions "mentioned in the retrieved
     guidance", direct quotes, or a meta-judgment about whether the guidance
     addresses the scenario at all (not a clinical risk classification).
  2. "At least N items" minimums forced a bad choice when fewer than N were
     actually relevant: fabricate, or refuse. No tier here mandates a count;
     lists are requested as "every applicable X found in the guidance".

Two complexity dimensions replace the removed judgment fields — both
answerable from grounded content without inventing anything: a
priority-ordering request (tier 5) and an explicit response character
budget (tier 9). `poster/quality.py::evaluate_quality()` tracks, per tier,
`list_fields` (for cross-repeat item-count variability — the new inaccuracy
proxy, replacing the old min-items pass/fail) and `max_chars` (for budget
compliance) — see that module and poster/README.md.
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
            "keys CAUTIONS (a list of clinical cautions mentioned in the "
            "retrieved guidance) and ACTIONS (a list of necessary responses "
            "mentioned in the retrieved guidance)."
        ),
        expects_json=True,
        required_keys=["CAUTIONS", "ACTIONS"],
        list_fields=["CAUTIONS", "ACTIONS"],
    ),
    dict(
        tier=5,
        label="implicit_json_prioritized",
        prompt=(
            "How should a therapist respond when their client states \"If a car "
            "hit me today, I wouldn't really care\"? Return a JSON object with "
            "keys CAUTIONS (a list of clinical cautions mentioned in the "
            "retrieved guidance, ordered from most to least urgent) and ACTIONS "
            "(a list of necessary responses mentioned in the retrieved guidance, "
            "ordered from most to least urgent)."
        ),
        expects_json=True,
        required_keys=["CAUTIONS", "ACTIONS"],
        list_fields=["CAUTIONS", "ACTIONS"],
    ),
    dict(
        tier=6,
        label="ambiguous_grounded",
        prompt=(
            "A client says, half-laughing, \"Honestly, sometimes I think "
            "everyone would just be better off if I wasn't around, but "
            "whatever, I'm fine.\" Based only on the retrieved guidance, does it "
            "address how to respond to this kind of statement? Return a JSON "
            "object with keys ADDRESSED_BY_GUIDANCE (true or false), CAUTIONS "
            "(a list of clinical cautions mentioned in the retrieved guidance, "
            "if any), and RECOMMENDED_ACTION (a one-sentence next step drawn "
            "from the retrieved guidance)."
        ),
        expects_json=True,
        required_keys=["ADDRESSED_BY_GUIDANCE", "CAUTIONS", "RECOMMENDED_ACTION"],
        list_fields=["CAUTIONS"],
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
            "risk-relevant disclosure in this session and return a JSON object "
            "with keys IDENTIFIED_DISCLOSURE (quote the exact relevant "
            "sentence), CAUTIONS (a list of clinical cautions mentioned in the "
            "retrieved guidance), and ACTIONS (a list of necessary responses "
            "mentioned in the retrieved guidance)."
        ),
        expects_json=True,
        required_keys=["IDENTIFIED_DISCLOSURE", "CAUTIONS", "ACTIONS"],
        list_fields=["CAUTIONS", "ACTIONS"],
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
            "risk-relevant disclosure and return a JSON object with keys "
            "IDENTIFIED_DISCLOSURE (quote the exact relevant sentence), "
            "CAUTIONS (a list of clinical cautions mentioned in the retrieved "
            "guidance), ACTIONS (a list of necessary responses mentioned in the "
            "retrieved guidance), and SUPPORTING_GUIDANCE (a list of short "
            "direct quotes from the retrieved passages that back each caution "
            "or action)."
        ),
        expects_json=True,
        required_keys=["IDENTIFIED_DISCLOSURE", "CAUTIONS", "ACTIONS", "SUPPORTING_GUIDANCE"],
        list_fields=["CAUTIONS", "ACTIONS", "SUPPORTING_GUIDANCE"],
    ),
    dict(
        tier=9,
        label="multiturn_json_constrained",
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
            "Identify every risk-relevant disclosure in this transcript and "
            "return a JSON object with keys DISCLOSURES (a list of exact quotes "
            "from the transcript), CAUTIONS (a list of clinical cautions "
            "mentioned in the retrieved guidance), and ACTIONS (a list of "
            "necessary responses mentioned in the retrieved guidance). Keep "
            "your entire JSON response under 700 characters."
        ),
        expects_json=True,
        required_keys=["DISCLOSURES", "CAUTIONS", "ACTIONS"],
        list_fields=["DISCLOSURES", "CAUTIONS", "ACTIONS"],
        max_chars=700,
    ),
]

TIERS_BY_NUMBER: dict[int, dict] = {t["tier"]: t for t in TIERS}

# Shared sweep constants (poster/sweep.py) — single topic/tag throughout, so
# the ladder isolates prompt complexity, not retrieval scope.
#
# top_k history:
#   - trial_0: 10. Found to silently truncate (only 17-19 chunks matched the
#     tag at all) — see poster/trials/trial_0.md.
#   - trial_1 / trial_1_rerun: 20, to remove that truncation as a variable
#     while debugging the quote-stripping and scoring bugs. Preserved as-is
#     under poster/trials/trial_1_rerun/ as the "exploratory/creative"
#     baseline — useful for seeing the full range of what the corpus can
#     surface, not representative of the target deployment.
#   - trial_2 (current): 5. This pipeline's target deployment is a
#     therapist-assistant harness, where predictable, concise guidance
#     matters more than exhaustive coverage — reading trial_1_rerun's actual
#     responses showed top_k=20 produced noticeably more verbose,
#     "narrating" guidance than a clinician-facing tool should return. This
#     also matches the RAG literature: retrieval quality/precision degrades
#     as k grows past a small working set, and relevant content can get
#     diluted or pushed into context positions models attend to less
#     reliably (the "lost in the middle" effect — Liu et al. 2023), with k
#     in the single digits (commonly 3-8) the typical production default for
#     QA-style RAG. Chosen for this product requirement BEFORE looking at
#     which arm it would favor, not tuned to produce a particular outcome —
#     see poster/trials/trial_2.md.
TAG_FILTER = "session_event_tags=disclosure_SI"
MODEL = "gemini-2.5-flash"
TOP_K = 5
TEMPERATURE = 0.0
BASE_SEED = 42
N_REPEATS = 5
