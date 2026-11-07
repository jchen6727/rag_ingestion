# poster/ — filtered-RAG vs. whole-corpus RAG, scientific-poster figure

Built on top of `rag_check/` (the latency harness) to answer one question for
the poster: **as a prompt gets harder — more inference required, more
distractor text, stricter output structure — does metadata-filtered
retrieval hold up better than dumping the entire corpus into context, and at
what cost?** Preliminary findings include a paradox worth showing explicitly:
**full-corpus retrieval sometimes comes back *faster*, but only because the
model gave up on the requested output structure** (malformed JSON, missing
keys, or list fields with far fewer items than asked for) — not because
retrieval + generation was actually more efficient. This package exists to
make that paradox visible in one figure, not just anecdotally noticed in a
few answer_text cells.

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
poster/plot_results.py — scores JSON quality (poster/quality.py) + renders the figure
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
| 4 | `implicit_json_simple` | + a 2-key JSON schema (CAUTIONS, ACTIONS) |
| 5 | `implicit_json_strict` | + a 4-key schema with minimum list lengths (≥3 items) |
| 6 | `ambiguous_classification` | disclosure is ambiguous/deniable — requires a judgment call, not just recognition, plus a confidence score |
| 7 | `narrative_json` | the disclosure is buried in a longer, mostly-unrelated narrative |
| 8 | `narrative_json_grounded` | + each action must cite a specific supporting-guidance source (nested JSON) |
| 9 | `multiturn_json_nested` | multi-turn transcript with **two** disclosures to find, combined risk assessment, most complex nested schema |

Each tier that requests JSON carries a `schema` dict (`poster/prompts.py`)
consumed by `poster/quality.py::evaluate_quality()`: required keys, and a
`min_items` for list-valued keys. A tier's output is scored `"ok"` only if
every key is present and every list meets its minimum — anything else
(`malformed_json`, `missing_keys`, `short_lists`, `empty_response`) counts
toward the figure's failure-rate panel.

## The sweep (`poster/sweep.py`)

9 tiers × 5 repeats (seed 42–46, everything else held constant) × 2 arms
(filtered / full_corpus, via `rag_check.param_sweep.run_sweep`) = **90 LLM
queries**, under the 100-query budget. Repeats vary only `seed`, so they
sample latency variance under otherwise-identical conditions rather than
testing prompt variations. Model is pinned to `gemini-2.5-flash` for every
tier (cost/time; override in `poster/prompts.py` if needed).

This hits the real DataStore and the real Gemini model — same prerequisites
as `rag_check/` (see `rag_check/README.md`). The `full_corpus` arm sends the
entire corpus (hundreds of chunks) as context on every one of its 45 calls,
so budget **20–40 minutes** for a full run. Always `--dry-run` first.

## Quality scoring (`poster/quality.py`)

Deliberately separate from `rag_check/`, which only measures latency. Given
a response's raw text and its tier's expected schema:

1. Strip a markdown code fence if present (Gemini routinely wraps JSON in
   ` ```json ... ``` ` even when asked for raw JSON).
2. Parse as JSON; `malformed_json` if that fails (covers truncation — a cut-
   off response won't parse).
3. Check every expected key is present; `missing_keys` if not.
4. Check every list-valued key meets its `min_items`; `short_lists` if not
   — this is the "significantly fewer elements" failure mode (e.g. a
   `CAUTIONS` list asked for ≥3 items that comes back with 1).
5. Otherwise `"ok"`. Free-text tiers (no JSON requested) are never scored as
   a failure unless the response is empty.

## The figure (`poster/plot_results.py`)

Two panels, same x-axis (complexity tier 1–9), stacked — not a dual-axis
chart (latency in seconds and failure % are different scales; see the
dataviz skill's anti-patterns on why dual axes mislead):

- **Top: mean total latency**, filtered vs. full_corpus, error bars = SEM
  across the 5 repeats.
- **Bottom: % of repeats that failed structurally** (`quality.py`'s
  `FAILURE_FLAGS`), same two series, same x-axis.

Read the two together: wherever full_corpus's latency bar dips *below*
filtered's while its failure-rate bar spikes at the same tier, that's the
paradox — full_corpus "won" on speed only because it gave up on the
requested structure. The script auto-annotates the single clearest instance
of this pattern with a direct-label arrow (deliberately sparing — not a
number on every bar, per the dataviz skill's labeling rule).

Colors are the dataviz skill's validated default palette, categorical slots
1–2 (blue = filtered, green = full_corpus), fixed order throughout, checked
against the skill's palette validator for this exact pair before shipping:

```bash
node <dataviz-skill>/scripts/validate_palette.js "#2a78d6,#008300" --mode light
# -> ALL CHECKS PASS
```

Output is `poster/figure.png` (2x-scale PNG for print) plus
`poster/figure.html` — an interactive twin that is also the chart's
"table view" equivalent (hover any bar for its exact value).

## Files

| File | Role |
|---|---|
| `prompts.py` | the complexity ladder + sweep constants (single source of truth) |
| `quality.py` | JSON-output quality scoring, independent of `rag_check/` |
| `sweep.py` | builds the sweep DataFrame, runs it via `rag_check.param_sweep` |
| `plot_results.py` | scores quality + renders the two-panel figure |
| `sweep_example.csv` | the original 3 hand-written seed cases this ladder expands on |
| `sweep_results.csv` | generated by `sweep.py` (not checked in — regenerate) |
| `figure.png` / `figure.html` | generated by `plot_results.py` (not checked in — regenerate) |
