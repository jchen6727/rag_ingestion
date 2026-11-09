# Research TODO — evaluating the efficacy of LLM/RAG system design

This file is forward-looking: what we've learned about *how to evaluate* a
filtered-RAG-vs-whole-corpus design (not just the current poster numbers),
and the concrete next investigations that would make those evaluations
trustworthy. For the current citable result and the narrative of how we got
here, see [`summary.md`](summary.md) and `trials/trial_0.md` →
`trial_1.md` → `trial_2.md` in order.

## Findings — what this work has actually established

These are the validated, defensible takeaways. Each is scoped to what was
actually tested (single topic, single corpus, `gemini-2.5-flash`) — see
"Generalization" below for what hasn't been.

1. **Filtered retrieval is faster than whole-corpus retrieval, consistently,
   once the pipeline itself is correct.** In `trial_2` (the clean run),
   `filtered` beat `full_corpus` on every one of 9 prompts, by roughly
   1.3x–2.7x. This held regardless of `top_k` (also true at `top_k=20` in
   `trial_1_rerun`). This is the one result that survived every bug fix and
   methodology revision — it's the most defensible claim this work can make.
2. **A faster-but-less-reliable arm is a real failure mode worth watching
   for, but it was never a genuine *model* behavior in this work — every
   instance traced to a measurement bug.** `trial_0` and `trial_1`'s first
   pass both showed `filtered` "failing" at high rates; both times the
   actual cause was broken retrieval (zero chunks returned, in trial_1's
   case) or an overly strict schema (trial_0), not the model behaving badly
   with real context. **Lesson for future evaluation design**: before
   concluding "arm X is less reliable," verify `num_chunks` / the actual
   retrieved content for the failing rows. A 100% failure rate with zero
   variance across repeats is a strong signal of a deterministic pipeline
   bug, not stochastic model behavior — model failures should scatter, not
   repeat identically 5/5 times.
3. **Schema design has to match what the corpus can actually support.**
   Asking for synthesized clinical judgments (`RISK_LEVEL`, `CONFIDENCE`
   scores) that no retrieved passage states caused a strictly-grounded
   model to correctly refuse rather than fabricate. That's good model
   behavior, but it made the evaluation measure "did we ask for something
   answerable" rather than "is the model reliable" — a confound between
   prompt design and model capability.
4. **Natural-language prompt text is not safe to reuse verbatim as a search
   query.** Discovery Engine (and likely other search backends using
   similar query syntax) treats literal quote characters as an exact-phrase
   operator. Any evaluation harness that derives retrieval queries from
   LLM-facing prompts needs an explicit, tested translation step — not an
   assumption that "the text the model sees" and "the text that should
   drive retrieval" are interchangeable.
5. **Distractor content in a prompt can silently change what gets
   retrieved, not just how the model reasons.** Two tiers designed to test
   "can the model find a disclosure buried in off-topic narrative" also,
   as a side effect, retrieved fewer relevant chunks than other tiers
   (their own distractor text diluted the search query's relevance
   ranking). An evaluation that varies prompt content for one reason (test
   reasoning under distraction) can accidentally vary retrieval too — these
   need to be deliberately decoupled or deliberately coupled, not left
   ambiguous.
6. **A consistency/variability metric needs an agreed ground truth before
   it means anything.** The coefficient-of-variation panel was removed
   after supervisor review surfaced interrater disagreement on what
   "correct" structure even looks like for several tiers. A statistic that
   quantifies disagreement with an assumed-correct answer is not useful
   until there's consensus on what correct means — and even then, CV
   specifically breaks down near a zero mean, which a small-count field
   (a list that's often empty or near-empty) will frequently be.
7. **Structural validity (JSON parses, keys present) is not the same claim
   as content accuracy.** Every metric built in this work — including the
   "quality_flag" scoring — checks shape, not truth. `full_corpus`'s
   consistently low hard-failure rate says nothing about whether its
   content is grounded in retrieved guidance or fabricated from the
   model's general training. This gap was never closed.

## Future directions for evaluating LLM/RAG design efficacy

Ranked roughly by how much they'd change what we can actually claim.

### 1. Grounding / hallucination verification (highest priority)

Nothing in this pipeline checks whether a response's content is traceable
to the retrieved chunks. This is the single biggest gap, and it's the
original question the whole poster set out to answer. Concrete next steps:

- **LLM-judge grounding pass**: for each response, feed the judge the
  response text *and* the actual retrieved chunks (not the whole corpus),
  and ask it to flag any claim not supported by the provided passages.
  Needs its own validation — judge agreement with a human rater on a
  sample, before trusting it at scale.
  Yields a `grounded` / `partially_grounded` / `ungrounded` label per claim
  or per response, to actually test the original hypothesis
  (`full_corpus` hallucinates more).
- **Clinician spot-check**: a small, structured rubric (is each caution
  clinically accurate, is each action appropriate, is anything dangerous or
  contraindicated recommended) over a sample of responses — expensive but
  the actual ground truth a judge model would be validated against.
- **Citation-forcing variant**: tiers that already ask for
  `SUPPORTING_GUIDANCE` quotes (T8) could be checked programmatically —
  does the quoted text actually appear in the retrieved chunk it's
  attributed to? This is cheap (no judge call needed) and catches outright
  fabricated citations, though it doesn't catch subtler ungrounded
  paraphrasing.

### 2. Retrieval parameter space — this work only tested one axis at a time

`top_k` went 10 → 20 → 5 across three trials, each a point estimate, never
a swept comparison. Worth a proper sweep (via `rag_check.param_sweep`,
already built for exactly this) across:

- `top_k` ∈ {3, 5, 8, 10, 15, 20} at fixed everything else, to find where
  latency/reliability/verbosity actually trade off, rather than picking a
  single value from literature + one qualitative read.
- Filter breadth: single-tag vs. multi-tag OR (e.g.
  `session_event_tags=disclosure_SI OR risk_dimension_tags=suicide_ideation_chronic`)
  — trial_0 found the single tag already covers 89% of plausibly-relevant
  chunks for this topic, but that number is topic-specific and unverified
  elsewhere.
- Chunk size / `chunk_config.yaml` parameters — never varied in this work;
  retrieval quality is sensitive to chunking strategy and this is a
  completely untested axis.

### 3. Query-construction robustness

The quote-stripping fix (trial_1) was reactive — found because a specific
query happened to collide with Discovery Engine's syntax. A more
systematic approach:

- Audit every prompt for other characters/patterns a search backend might
  treat specially (`AND`/`OR`/`NOT` as keywords, `:`, `-` as exclusion,
  etc.) rather than discovering them one at a time.
- Consider a dedicated `search_query` per tier (already supported by
  `rag_check`) chosen independently of prompt wording, vs. the current
  default of deriving it from the prompt — trial_1's Addendum 2 flagged
  this as an open design decision, never resolved.

### 4. Generalization beyond this one topic and corpus

Every finding here is scoped to `session_event_tags=disclosure_SI` against
a 3-PDF, ~900-chunk corpus. Before claiming anything about filtered-vs-
whole-corpus RAG *in general*:

- Repeat the ladder (or a reduced version) against other tags —
  `therapeutic_modality=DBT`, `directionality=contraindicated`,
  `risk_dimension_tags=*` — to see whether the latency advantage and
  reliability pattern hold, or whether `disclosure_SI` (17-19 matching
  chunks, a relatively narrow, well-defined topic) was a favorable case.
- Test against a larger/more diverse corpus — current findings may not
  hold at corpus sizes where `full_corpus`'s per-call cost (currently
  dominated by listing ~900 chunks) grows further, or where a topic has
  far more or far fewer matching chunks than disclosure_SI's ~18.

### 5. Statistical power and design

n=5 repeats per cell is enough to see clear effects (SEM bars in the
current figure are mostly tight) but not enough for the less-clean tiers
(trial_2's T6 variability spike came from a mean of 0.4 across 5 points).
Consider:

- More repeats for any tier/metric close to a decision boundary.
- Formal power analysis once a specific, pre-registered hypothesis exists
  (e.g. "filtered reduces mean latency by at least 30%") rather than
  exploratory n=5 sweeps.
- Varying `seed` was the only repeat-to-repeat variation in this work;
  consider also varying `temperature` to separately characterize
  sampling-driven vs. retrieval-driven variability.

### 6. Human/interrater evaluation protocol

The structural-variability panel was pulled specifically because of
interrater disagreement on "correct" structure. That disagreement is
itself worth investigating properly rather than just removing the metric:

- Define a rubric with concrete examples before asking raters to judge
  structure, rather than relying on each rater's implicit sense of
  "correct."
- Measure interrater agreement (e.g. Cohen's kappa) explicitly, and report
  it alongside any future structural-quality metric — if agreement is low,
  that's a finding in itself about how well-specified the schema is.

### 7. Production-readiness criteria for the therapist-assistant deployment

This whole evaluation is in service of a real deployment target. Worth
defining, explicitly, what "good enough" means for that target before more
trials:

- A latency SLO (e.g. p95 under N seconds) — currently only means are
  reported.
- A hard floor on `declined_insufficient_context` rate that's acceptable
  in production (some refusal is correct/safe behavior; too much makes the
  tool useless).
- Whether `top_k=5`'s remaining T9 character-budget failures (20% in
  trial_2) are acceptable, or whether that specific tier's schema needs
  simplifying for deployment rather than further parameter tuning.
