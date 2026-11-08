# Summary — filtered-RAG vs. whole-corpus RAG (poster work)

One-page entry point for this body of work. Read this first; it links out to
everything else. Written for a human skimming for what matters, and for
another agent picking this up cold — see "Bootstrapping for another agent"
at the bottom.

## Priorities — what to do next, ranked by impact

1. **Verify grounding/hallucination, not just structure.** This is the
   single biggest gap. Every trial so far measures whether a response
   *parses* correctly (valid JSON, required keys present, under budget) —
   none of them check whether the content is actually *grounded* in
   retrieved guidance versus the model's own prior knowledge. The original
   motivating hypothesis for this whole poster — does `full_corpus` trade
   accuracy for speed — has never actually been tested. This needs a
   clinician read-through or an LLM-judge grounding pass (feed the
   response + the actual retrieved chunks to a judge, ask "is every claim
   traceable to a chunk") against `trial_2`'s data, before any claim about
   accuracy goes on the poster. Nothing currently measures this.
2. **Fix the CV-near-zero-mean metric artifact before the variability panel
   is poster-ready.** `poster/quality.py` / `poster/plot_results.py`'s
   coefficient-of-variation metric blows up arithmetically when a list
   field's mean count is near zero (trial_2's T6: counts `[0,0,1,1,0]` →
   122% CV, not a real consistency problem — see `trials/trial_2.md`).
   Switch to an absolute-range or count-of-distinct-values measure for
   low-count fields, or clip/flag CV when mean < some threshold.
3. **Decide what goes on the poster: `top_k=5` (production-configured) or
   `top_k=20` (exploratory), and say why, explicitly.** Both are real,
   honestly-obtained results (`trials/trial_2/` and `trials/trial_1_rerun/`
   respectively) measuring different things — `top_k=5` is what the target
   therapist-assistant deployment would actually run; `top_k=20` shows more
   of what the corpus can surface. Don't present one without naming which
   and why, given how much of this project's effort went into making sure
   every number is defensible.
4. **This ladder is single-topic (`session_event_tags=disclosure_SI`) and
   single-corpus (3 PDFs, ~900 chunks).** Every finding here is scoped to
   that one tag and that one corpus. Generalizing to other topics
   (`therapeutic_modality=DBT`, `directionality=contraindicated`, etc.)
   before claiming anything about filtered-vs-corpus RAG *in general* is an
   open, not-yet-attempted next step.
5. **Priority ordering (tier 5) is requested in the prompt but never
   independently scored** — no cross-repeat rank-agreement metric exists.
   Lower priority than 1–4, but a known gap in what the ladder claims to
   test.

## What this work is

Two layers, built in this order:

- **`rag_check/`** — a general-purpose latency-comparison harness: one LLM
  call, two retrieval strategies (`filtered`: Discovery Engine search
  restricted by a metadata tag filter; `full_corpus`: every chunk, no
  ranking), same prompt, same model params, timed separately. Also ships
  `param_sweep.py` for running many such comparisons from a CSV or
  DataFrame. Not part of `rag_ingestion`'s stated scope (ingestion +
  tagging only) — a deliberate, narrow exception, documented in the
  top-level `CLAUDE.md`.
- **`poster/`** — built on top of `rag_check/`, purpose-built for one
  scientific-poster figure: a 9-tier escalating-complexity prompt ladder on
  a single clinical topic (suicidal-ideation disclosure), run through
  `rag_check.param_sweep`, scored for structural JSON quality
  (`poster/quality.py`), and rendered as a three-panel chart (latency /
  hard-failure rate / structural variability).

## What was actually found — in order, because each one corrected the last

Four trials, each documented in its own file under `poster/trials/`,
**read in order** — later trials correct specific, named mistakes in
earlier ones rather than silently replacing them:

| Trial | What changed | What was found |
|---|---|---|
| [`trial_0.md`](trials/trial_0.md) | First live run, 9-tier ladder, `top_k=10` | Paradox ran *backwards* from the hypothesis: `filtered` (not `full_corpus`) was the fast-but-unreliable arm, refusing ~100% of the time on structured tiers. Root causes: schema fields (`RISK_LEVEL`, `CONFIDENCE`, ...) the corpus couldn't support, and prompt-mandated minimum item counts forcing fabrication-or-refusal. |
| [`trial_1.md`](trials/trial_1.md) | Redesigned schema (grounded fields only, no minimums), `top_k→20` | Two more bugs found **while investigating this trial's own data**: (a) a scoring bug — valid JSON honestly reporting an empty finding was mis-flagged as a hard failure; (b) a retrieval bug — literal quote marks in prompts were parsed by Discovery Engine as an exact-phrase search operator, silently returning **zero chunks** on 6 of 9 tiers. Both fixed; corrected rerun archived as `trial_1_rerun/`. A third issue (not a bug, a confound) was also found and documented, not fixed: two "narrative distractor" tiers retrieved fewer chunks than the rest because their off-topic filler content diluted the search query. |
| [`trial_2.md`](trials/trial_2.md) | `top_k→5`, matching the real target deployment (concise, predictable therapist-assistant output, not exhaustive/narrating corpus coverage) | Cleanest result yet: `filtered` faster than `full_corpus` on every tier, ~0% hard failures nearly everywhere, and the top_k reduction incidentally fixed trial_1's narrative-distractor confound (every tier now retrieves the same 5 chunks). One metric artifact caught and flagged, not hidden (see Priority 2 above). |

**No trial has found structural evidence either way for the original
hallucination-paradox hypothesis.** That's not a negative result — it's an
untested one (see Priority 1).

## Where the numbers actually are

- **Current/citable**: `poster/trials/trial_2/` (`top_k=5`, production-configured) —
  also mirrored at the repo root (`poster/sweep_results.csv`, `poster/figure.png`)
  as the latest run.
- **Exploratory baseline**: `poster/trials/trial_1_rerun/` (`top_k=20`, same
  schema/fixes as trial_2, higher-`k` corpus coverage).
- **Historical / buggy, kept for audit trail only**: `poster/trials/trial_0/`
  and `poster/trials/trial_1/` — do not cite numbers from these; they
  predate fixes documented in the corresponding `.md`.

## Bootstrapping for another agent

If you're picking this up with no prior context, in order:

1. Read this file fully (you're doing that now).
2. Read `poster/trials/trial_0.md` → `trial_1.md` → `trial_2.md`, in that
   order. Each one's "what was found" section explains a mistake the
   *previous* trial made — don't skip to trial_2 and assume its numbers
   need no caveats, read the chain.
3. Read `poster/README.md` for the mechanical pipeline (`prompts.py` →
   `sweep.py` → `sweep_results.csv` → `plot_results.py` → `figure.png`) and
   the column/schema reference.
4. Read `rag_check/README.md` for the layer underneath — what
   `compare_latency.py` and `param_sweep.py` actually do, and two more live
   -confirmed Discovery Engine gotchas that aren't poster-specific:
   - `SearchService.search()` doesn't return chunk text inline in this
     project even with `content_config=CONTENT_REQUIRED` — `search_filtered()`
     falls back to a per-chunk `GetDocument` call, which is why filtered
     retrieval costs `1 + top_k` round trips, not 1.
   - Discovery Engine's filter grammar needs `field: ANY("value")` for
     **every** text-indexed field (scalar or array) — `=` only works on the
     one true integer field (`year_published`); this was wrong in an
     earlier version of `rag_check/retrieval.py` and is now fixed and
     documented there.
5. **Before changing anything**, run `PYTHONPATH=. python poster/sweep.py
   --dry-run` to see the current plan without spending API quota, and skim
   `poster/prompts.py::TIERS` to see the actual prompt text per tier.
6. **Before rerunning live**, know the cost: 9 tiers × 5 repeats × 2 arms =
   90 LLM queries, ~15–40 minutes depending on `top_k` and model, against a
   real GCP project (`rag_check/README.md`'s Prerequisites). Confirm with
   the user before spending that budget — every live run in this history
   was confirmed first.
7. **If a result looks surprising, suspect the pipeline before the model.**
   Every "finding" in trial_0 and trial_1's first pass turned out to be a
   bug in this code, not a real result, until it was specifically checked
   against the raw `num_chunks` / `answer_text` columns in
   `sweep_results.csv`. The pattern that worked every time: when a result
   doesn't match expectation, pull the raw rows for a few individual
   responses and read them verbatim before writing up a conclusion.
8. **Never tune a parameter to produce a preferred outcome.** `top_k` went
   from 10→20→5 across this history, and every change was justified on
   independent grounds (truncation, debugging, then a real deployment
   requirement) *before* checking which arm it favored — see trial_2.md's
   explicit discussion of this. If asked to change a parameter "to get a
   better result for X," push back and ask for the actual underlying
   reason, the same way this conversation did.

## Files at a glance

```
poster/
  summary.md          <- you are here
  README.md            pipeline mechanics, column reference, current limitations
  prompts.py           the 9-tier ladder + sweep constants (single source of truth)
  quality.py           per-response structural scoring
  sweep.py              builds the sweep DataFrame, runs rag_check.param_sweep
  plot_results.py      scores + renders the 3-panel figure
  sweep_example.csv     original 3 hand-written seed cases this ladder expands on
  sweep_results.csv     latest run's raw data (currently = trial_2)
  figure.png/.html      latest run's figure (currently = trial_2)
  trials/
    trial_0.md, trial_1.md, trial_2.md   <- READ IN ORDER
    trial_0/, trial_1/, trial_1_rerun/, trial_2/   frozen code+data+figure per trial

rag_check/
  README.md             the latency-harness layer poster/ is built on
  compare_latency.py, retrieval.py, llm_query.py, param_sweep.py
  examples/
```
