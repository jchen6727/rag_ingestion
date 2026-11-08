# poster/ — filtered-RAG vs. whole-corpus RAG, scientific-poster figure

**New here or picking this back up cold? Read [`poster/summary.md`](summary.md)
first** — one-page priorities, what's been found, and bootstrapping
instructions. This file is the pipeline/mechanics reference; `summary.md` is
the narrative entry point.

Built on top of `rag_check/` (the latency harness) to answer one question for
the poster: **as a prompt gets harder — more inference required, more
distractor text, stricter output structure — does metadata-filtered
retrieval hold up better than dumping the entire corpus into context, and at
what cost?**

**Read `poster/trials/trial_0.md` and `poster/trials/trial_1.md` before
touching this methodology — three real bugs were found and fixed across
these two trials, each one initially looked like a model-behavior finding
before turning out to be a measurement problem:**

1. **Trial_0**: schema fields the corpus couldn't support (`RISK_LEVEL`,
   `CONFIDENCE`, ...) and prompt-mandated minimum item counts that forced
   fabrication-or-refusal — fixed by redesigning the schema and raising
   `top_k`.
2. **Trial_1, scoring bug**: `quality.py`'s refusal detector matched
   refusal-like phrasing *inside otherwise-valid JSON* (a response honestly
   reporting an empty finding), mis-scoring it as a hard failure — fixed by
   only checking for refusal when no JSON was found at all.
3. **Trial_1, retrieval bug**: several prompts quote example client text in
   literal `"..."` marks; Discovery Engine parses that as an **exact-phrase
   search operator**, so the default search-query-equals-prompt behavior
   silently retrieved **zero chunks** on every tier that quoted text not
   verbatim present in the corpus — not a top_k or filter problem, a query-
   construction one. Fixed in `poster/sweep.py` (strips quote characters
   from the search query only, not the LLM-facing prompt) and documented
   more generally in `rag_check/README.md` for anyone else building a
   search query from natural-language prompt text.

**Trial_1_rerun** (all three bugs above fixed) also surfaced a retrieval
*confound*, not a bug with one correct fix: the two "narrative distractor"
tiers (T7/T8) use their full prompt — distractor text included — as the
search query, which measurably changed how many chunks got retrieved for
those two tiers specifically (7/18 vs. 17–18/18 everywhere else). See
trial_1.md's Addendum 2.

**Trial_2 — `top_k` reduced from 20 to 5**, chosen for the actual target
deployment (a therapist-assistant harness, where predictable/concise
guidance matters more than exhaustive coverage — reading trial_1_rerun's
responses showed `top_k=20` producing noticeably more verbose, "narrating"
output than that product needs) and consistent with the RAG literature
(single-digit `k` is the typical production default; see trial_2.md for the
full rationale). Chosen for the deployment requirement *before* looking at
which arm it would favor, not tuned to produce an outcome — `top_k=20`
remains preserved unmodified under `trials/trial_1_rerun/` as the
exploratory/creative baseline, not overwritten. This also happened to
resolve the T7/T8 confound as a side effect: at `top_k=5` every tier now
retrieves exactly 5 chunks.

**Current status (`poster/trials/trial_2/`, the current root-level data):**
the cleanest result of the three trials — `filtered` faster than
`full_corpus` on every single tier, 0% hard failures for `filtered`
everywhere, 0% for `full_corpus` on 8 of 9 tiers (T9's character-budget
tier is the one exception, carried over from trial_1). One metric artifact
to know about: T6's variability bar spikes to ~122% CV purely because its
list-field mean count is near zero (0-1 items) — a coefficient-of-variation
distortion, not a real consistency problem; see trial_2.md. No structural
hallucination-paradox signal (the original motivating hypothesis for this
whole poster) has appeared in any trial after the bugs were fixed; whether
`full_corpus` actually hallucinates more is still unverified, not refuted —
see "Known limitations" below. Each trial's raw data, figure, and code
snapshot are preserved under `poster/trials/trial_N/` precisely so history
doesn't get lost as the methodology moves on.

## Pipeline

```
poster/prompts.py   — the 9-tier complexity ladder (single topic, one escalated axis)
        │
        ▼
poster/sweep.py      — runs it through rag_check.param_sweep.run_sweep()
        │
        ▼
poster/sweep_results.csv
        │
        ▼
poster/plot_results.py — scores output quality (poster/quality.py) + renders the figure
        │
        ▼
poster/figure.png  (+ figure.html, interactive table-view twin)
```

```bash
PYTHONPATH=. python poster/sweep.py --dry-run                           # sanity-check the plan, no API calls
PYTHONPATH=. python poster/sweep.py --out poster/sweep_results.csv      # the real run (90 LLM queries, ~20-40 min)
PYTHONPATH=. python poster/plot_results.py --csv poster/sweep_results.csv --out poster/figure.png
```

## The complexity ladder (`poster/prompts.py`)

Single clinical topic throughout — suicidal-ideation disclosure, same tag
filter (`session_event_tags=disclosure_SI`) on every tier, matching the seed
cases in `poster/sweep_example.csv` — so the ladder isolates *prompt*
complexity, not retrieval scope. One dimension escalates per tier:

| Tier | Label | What's harder |
|---|---|---|
| 1 | `direct_openended` | baseline: explicit disclosure, free-text answer |
| 2 | `direct_structured` | + a minimal structural constraint (3 numbered actions) |
| 3 | `implicit_openended` | disclosure is implicit — requires inference to even recognize it |
| 4 | `implicit_json_simple` | + an open-ended JSON schema (CAUTIONS, ACTIONS — no minimum count) |
| 5 | `implicit_json_prioritized` | + the same fields, but ordered from most to least urgent |
| 6 | `ambiguous_grounded` | disclosure is ambiguous/deniable — asks whether the retrieved guidance even addresses it, not an invented risk classification |
| 7 | `narrative_json` | the disclosure is buried in a longer, mostly-unrelated narrative |
| 8 | `narrative_json_grounded` | + each caution/action must be backed by a direct quote from retrieved guidance |
| 9 | `multiturn_json_constrained` | multi-turn transcript, two disclosures to find, **and** the entire JSON response must fit under 700 characters |

### Why no RISK_LEVEL / IS_DISCLOSURE / CONFIDENCE / SEVERITY, and no "at least N items"

Trial_0's ladder asked for those fields and minimums; `poster/trials/trial_0.md`
documents in detail why both choices were wrong and were removed:

- Fields like `RISK_LEVEL` or a `CONFIDENCE` score ask the model to invent a
  clinical judgment that a corpus of treatment manuals and textbooks simply
  doesn't state — there's no passage to ground that number in. Every field
  in the current ladder asks only for content plausibly **extractable** from
  retrieved guidance text: cautions/actions "mentioned in the retrieved
  guidance", direct quotes, or a meta-judgment about whether the guidance
  addresses the scenario at all (not a clinical risk classification).
- "A list of at least 3 cautions" forces a bad choice when only 1-2 are
  actually relevant: fabricate a 3rd, or refuse outright. No tier here
  mandates a count — lists are requested as "every applicable X found in the
  guidance", open-ended.

## Quality scoring (`poster/quality.py`)

Deliberately separate from `rag_check/`, which only measures latency. Given
a response's raw text and its tier's definition (`required_keys`,
`list_fields`, optional `max_chars`):

1. Detect an explicit refusal (`compare_latency.py`'s system preamble
   instructs the model to decline rather than guess when context is
   insufficient) → `declined_insufficient_context`. This is its own category,
   not lumped in with malformed output — trial_0 found it dominates the
   `filtered` arm specifically, which is a different story than "the model
   tried and botched it."
2. Strip a markdown code fence if present, parse as JSON → `malformed_json`
   if that fails (covers truncation too — a cut-off response won't parse).
3. Check every `required_keys` entry is present → `missing_keys` if not.
4. If `max_chars` is set, check the extracted JSON length against it →
   `exceeded_char_budget` if over.
5. Otherwise `ok`. Free-text tiers (no JSON requested) are never scored as a
   failure unless the response is empty.

Separately, **every `list_fields` entry's item count is recorded** (not
judged) on every response that parsed. `plot_results.py` turns this into a
**cross-repeat variability** metric — see next section — instead of a
pass/fail against a minimum, which is what trial_0 got wrong.

## The figure (`poster/plot_results.py`)

Three panels, same x-axis (complexity tier 1-9), stacked — not a dual-axis
chart (three different scales; see the dataviz skill's anti-patterns on why
dual axes invent correlations that aren't in the data):

- **Top: mean total latency**, filtered vs. full_corpus, error bars = SEM
  across the 5 repeats.
- **Middle: hard failure rate** — `declined_insufficient_context` /
  `malformed_json` / `missing_keys` / `exceeded_char_budget` — same two
  series, same x-axis.
- **Bottom: structural variability** — mean coefficient of variation (CV%)
  of `list_fields` item counts across the 5 repeats, for responses that
  otherwise parsed fine. This is the inaccuracy proxy that replaced
  trial_0's strict minimum-item-count check: a field whose item count swings
  wildly run to run, even when every individual response is well-formed,
  suggests the model is improvising structure rather than reporting
  something stable it actually found in the retrieved guidance.

Read the top two together first: wherever one arm's latency bar dips
*below* the other's while its failure-rate bar spikes at the same tier,
that's the latency/reliability paradox this poster reports — it only "won"
on speed because it gave up on the output, not because retrieval+generation
was actually more efficient. The script auto-annotates the single clearest
instance of this pattern with a direct-label arrow, in whichever direction
the live data actually shows it (deliberately not hardcoded to either arm —
see `_find_paradox_annotation()`). The bottom panel is a second, independent
lens: an arm can have a *low* failure rate and *still* show high
variability, which is itself worth noting on the poster as "technically
valid, but inconsistent."

Colors are the dataviz skill's validated default palette, categorical slots
1-2 (blue = filtered, green = full_corpus), fixed order throughout, checked
against the skill's palette validator for this exact pair before shipping:

```bash
node <dataviz-skill>/scripts/validate_palette.js "#2a78d6,#008300" --mode light
# -> ALL CHECKS PASS
```

Output is `poster/figure.png` (2x-scale PNG for print) plus
`poster/figure.html` — an interactive twin that is also the chart's
"table view" equivalent (hover any bar for its exact value).

## Known limitations (open as of trial_2)

- **Grounding is not independently verified.** A structurally well-formed,
  `ok`-flagged response is not a verified-accurate one — nothing here checks
  whether the content is actually grounded in the retrieved passages versus
  the model's own prior knowledge. `full_corpus`'s near-0% hard-failure rate
  is a structural-validity claim only; whether it also hallucinates more
  than `filtered` (the original hypothesis this whole poster set out to
  test) needs a clinician or LLM-judge grounding pass, not just this
  pipeline. No structural signal either way has appeared in any trial after
  the bugs were fixed — treat it as a fully open question, not a settled
  finding in either direction.
- **Coefficient of variation is unreliable near a zero mean.** T6's
  variability bar in trial_2 spikes to ~122% purely because its list
  field's typical count is 0-1 items — a metric artifact, not a real
  consistency problem (see trial_2.md). Worth switching to an absolute-range
  measure for low-count list fields before the variability panel goes on
  the poster without this caveat attached.
- **T9's character budget (700 chars) is tight.** `full_corpus` fails it
  20% of the time at `top_k=5` (worse at `top_k=20` — see trial_1_rerun);
  `filtered` never has in any trial so far. Not yet stress-tested at other
  `top_k` values to see where that boundary actually sits.
- **Priority ordering (tier 5) is requested but not independently scored.**
  There's no cross-repeat rank-agreement metric — free-text list items
  aren't reliably matchable across repeats without a fuzzy-matching step.
  Tracked as future work.

## Files

| File | Role |
|---|---|
| `prompts.py` | the complexity ladder + sweep constants (single source of truth) |
| `quality.py` | per-response quality scoring, independent of `rag_check/` |
| `sweep.py` | builds the sweep DataFrame, runs it via `rag_check.param_sweep` |
| `plot_results.py` | scores quality + renders the three-panel figure |
| `sweep_example.csv` | the original 3 hand-written seed cases this ladder expands on |
| `sweep_results.csv` | generated by `sweep.py` (not checked in — regenerate) |
| `figure.png` / `figure.html` | generated by `plot_results.py` (not checked in — regenerate) |
| `trials/trial_0.md`, `trials/trial_1.md`, `trials/trial_2.md` | findings writeups, in order — **read these first** |
| `trials/trial_0/` | frozen snapshot of trial_0's code, raw data, and figure |
| `trials/trial_1/` | trial_1's **first pass** — scoring bug + zero-chunk retrieval bug both still present; kept for the audit trail, not for numbers |
| `trials/trial_1_rerun/` | trial_1's **corrected** run (`top_k=20`) — the exploratory/creative baseline, preserved unmodified |
| `trials/trial_2/` | **current** run (`top_k=5`, production-configured for the therapist-assistant deployment) — this is the current, citable data |
