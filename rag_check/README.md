# rag_check — RAG latency comparison

**Not part of `rag_ingestion`'s stated scope.** Per the repo's `CLAUDE.md`,
`rag_ingestion` is ingestion + metadata tagging + metadata visualization
only — "the query/retrieval/generation side ... is intentionally not part
of this package." `rag_check/` is a separate, narrow measurement harness
that *reads from* a Vertex AI Search DataStore already populated by
`scripts/batch_ingest.py`. It does not write to the DataStore, does not
import from `ingestion/`, and is not exercised by CI.

## What it measures

One LLM call latency, under two different retrieval strategies, against the
**same prompt** and the **same Gemini request parameters**:

| Arm | Retrieval | What it answers |
|---|---|---|
| `filtered` | Discovery Engine `Search` — ranked, optionally restricted to chunks matching one or more `--tag field=value` metadata filters | "How fast is a response when we only hand the model retrieved, tagged guidance?" |
| `full_corpus` | Discovery Engine `ListDocuments` — every chunk in the DataStore, unranked, unfiltered | "How fast is a response when we just dump the entire corpus into context instead?" |

Both arms assemble a prompt of the **identical shape** (same instructions,
same question — see `build_prompt()` in `compare_latency.py`) so the only
deliberate variable between runs is retrieval strategy / context size.
Retrieval time and generation time are timed separately and reported
alongside the total, so you can tell whether a latency difference comes
from the Discovery Engine call or from the Gemini call processing a bigger
prompt.

## Prerequisites

- A DataStore already provisioned (`scripts/setup_vertex_search.py`) and
  populated (`scripts/batch_ingest.py`) — same `.env` / `GCP_PROJECT_ID`,
  `VERTEX_SEARCH_DATASTORE_ID`, `VERTEX_SEARCH_ENGINE_ID` as the rest of the
  pipeline.
- Application Default Credentials set up (`gcloud auth application-default
  login`) — same auth as every other script here.
- No new dependencies: `google-cloud-discoveryengine` and `google-genai`
  are already in `requirements.txt`.

## Usage

All commands run from the repo root with `PYTHONPATH=.` (same convention as
every other script in this repo).

```bash
# Basic comparison: a tag-filtered arm vs. the whole corpus.
PYTHONPATH=. python rag_check/compare_latency.py \
    --prompt "What should I do about a client disclosing suicidal ideation mid-session?" \
    --tag session_event_tags=disclosure_SI \
    --tag directionality=contraindicated

# Pick the model and tune genai request params:
PYTHONPATH=. python rag_check/compare_latency.py \
    --prompt "..." \
    --tag therapeutic_modality=DBT \
    --model gemini-2.5-flash \
    --temperature 0.2 \
    --seed 42 \
    --thinking-budget 1024 \
    --max-output-tokens 2048

# No --tag at all: the "filtered" arm still ranks (unfiltered) across the
# whole corpus — distinct from full_corpus, which sends every chunk with no
# ranking at all. Useful as a 3rd data point (ranked-but-unfiltered vs.
# unranked-everything):
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --top-k 5

# Only run one arm (e.g. to sanity-check full_corpus alone, or to bound a
# very large corpus to a fixed number of chunks):
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --skip-filtered
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --skip-full-corpus
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --max-corpus-chunks 200

# Save the full run (including answer text, per-chunk IDs retrieved, and all
# params) as JSON for later diffing/plotting:
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --tag domain=dialectical_behavior_therapy --out results.json

# See exactly what's happening (Discovery Engine + Gemini request/response
# detail):
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --tag domain=cognitive_behavioral_therapy --verbose
```

### Inputs accepted

| Flag | Meaning |
|---|---|
| `--prompt` (required) | The question sent to the LLM. |
| `--search-query` | Search text for the `filtered` arm's ranking, if you want it to differ from `--prompt` (defaults to `--prompt`). |
| `--tag FIELD=VALUE` | Repeatable, AND-combined metadata filter for the `filtered` arm — e.g. `--tag session_event_tags=disclosure_SI`. **Omit it entirely to retrieve/rank across all documents** in that arm too (see table above for how that still differs from `full_corpus`). |
| `--top-k` | Max chunks the `filtered` arm retrieves (default 10). |
| `--max-corpus-chunks` | Cap on chunks the `full_corpus` arm retrieves (default: no cap — the literal entire corpus). |
| `--skip-filtered` / `--skip-full-corpus` | Run only one arm. |
| `--model` | Gemini model id (default: `GEMINI_MODEL_GENERATION` from `.env`, same setting `config/settings.py::gemini_model_generation` exposes). |
| `--temperature`, `--top-p`, `--sampling-top-k`, `--seed`, `--max-output-tokens`, `--thinking-budget` | Passed straight through to `google.genai.types.GenerateContentConfig`. `--sampling-top-k` is the *sampling* top-k, kept distinct from `--top-k` (the *retrieval* count) — don't confuse the two. |
| `--out PATH` | Write the full JSON result (answer text, chunk IDs, all params) to a file. |
| `--verbose` | DEBUG logging, including Discovery Engine/Gemini request details. |

### Which metadata fields can go in `--tag`

Must match a field registered as filterable in
`scripts/setup_vertex_search.py` (see `config/rta_v1.json`
`notes.vertex_ai_search`) — passing an unregistered/misspelled field name
raises a clear error from `rag_check/retrieval.py::build_filter_expression`
instead of silently building a filter that no-ops at the API layer:

- **Text fields** — scalar or array, same `field: ANY("value")` operator;
  for arrays a chunk matches if it *contains* that value (confirmed against
  a live DataStore: Discovery Engine rejects `=`/`<`/`>` on any
  text-indexed field, scalar included — e.g. `directionality = "x"` fails
  with "Unsupported field ... on comparison operators"):
  `therapeutic_modality`, `clinical_presentation`, `session_event_tags`,
  `applies_when`, `risk_dimension_tags`, `patient_population`,
  `clinical_measure_tags`, `technique_tags`, `clinical_caution` (arrays);
  `domain`, `doc_type`, `session_phase`, `directionality`, `source_file`,
  `title`, `chapter`, `subdomain` (scalars).
- **Integer field** (the one exception — uses `=`, unquoted): `year_published`.

Multiple `--tag` args are AND-combined. There is currently no CLI way to
express OR across fields or multiple values for the same array field in one
filter — if you need that, call `rag_check.retrieval.search_filtered()`
directly with your own filter string (Discovery Engine's filter grammar
supports `ANY("a", "b")` for an OR within one field).

## Output

Console summary (always printed):

```
==============================================================================
arm             chunks  prompt_chars  retrieval_s  generate_s   total_s
------------------------------------------------------------------------------
filtered             8          4213        0.312       1.845     2.157
full_corpus         412        187340       0.891       6.203     7.094
==============================================================================

'filtered' was faster by 4.937s total (filtered=2.157s, full_corpus=7.094s)
```

With `--out`, the same run is written as JSON with the full answer text,
every retrieved `chunk_id`, the resolved filter expression, and every
`genai` param used — enough to reproduce or diff two runs later.

## Parameter sweeps (`param_sweep.py`)

`rag_check/param_sweep.py` runs the same filtered-vs-full-corpus comparison
once per row of an input table, instead of once per CLI invocation. Both
entry points call the exact same `compare_latency.run_comparison()` the CLI
does, so a sweep row can't drift from what a one-off `compare_latency.py`
run would measure for the same inputs:

- `run_sweep_from_csv(csv_path, out_csv=...)` — CLI / file-based.
- `run_sweep(df, out_csv=...)` — the method wrapper for a DataFrame you
  built programmatically instead of a file.

A column is entirely optional: if it's absent from the table, or blank/NaN
for a given row, that argument falls back to the same default
`compare_latency.py`'s CLI uses. Only `prompt` is required.

| Column | Maps to | Default if missing/blank |
|---|---|---|
| `prompt` | `--prompt` | *(required, no default)* |
| `search_query` | `--search-query` | `prompt` |
| `tags` | `--tag` (repeated) | none — see below |
| `top_k` | `--top-k` | `10` |
| `max_corpus_chunks` | `--max-corpus-chunks` | no cap |
| `skip_filtered` | `--skip-filtered` | `False` |
| `skip_full_corpus` | `--skip-full-corpus` | `False` |
| `model` | `--model` | `settings.gemini_model_generation` |
| `temperature` | `--temperature` | `0.0` |
| `top_p` | `--top-p` | model default |
| `sampling_top_k` | `--sampling-top-k` | model default |
| `seed` | `--seed` | unseeded |
| `max_output_tokens` | `--max-output-tokens` | model default |
| `thinking_budget` | `--thinking-budget` | model default |

`tags` is a single semicolon-delimited cell (a CSV row can't repeat a
column the way `--tag` is repeatable on the CLI): e.g.
`"session_event_tags=disclosure_SI;directionality=contraindicated"` for an
AND-combined filter, or leave the cell blank for an unfiltered-but-ranked
search. Bool-ish columns accept `true`/`false`/`1`/`0`/`yes`/`no`.

Output is **long format**: one row per (input row × arm actually run for
that row) — so a row with neither skip flag set produces two output rows
(`filtered` and `full_corpus`), each with every input column echoed back
under an `input_` prefix plus `arm`, `num_chunks`, `prompt_chars`,
`retrieval_latency_seconds`, `generation_latency_seconds`,
`total_latency_seconds`, token-usage columns, `chunk_ids`, and
`answer_text`. A row whose run raises an exception (e.g. an unregistered
`--tag` field) gets one output row with `arm="error"` and the exception
message in `error`, instead of halting the whole sweep.

### Example 1 — input CSV

`rag_check/examples/sweep_example.csv`:

```csv
prompt,tags,top_k,model,temperature,seed
"What should I do about a client disclosing suicidal ideation mid-session?",session_event_tags=disclosure_SI,5,gemini-2.5-flash,0.0,42
"What should I do about a client disclosing suicidal ideation mid-session?",session_event_tags=disclosure_SI,10,gemini-2.5-flash,0.0,42
"How do I handle a therapeutic alliance rupture?",session_event_tags=rupture;directionality=indicated,5,gemini-2.5-flash,0.0,42
"What DBT skills help with emotional dysregulation?",therapeutic_modality=DBT,10,gemini-2.5-pro,0.0,42
"What DBT skills help with emotional dysregulation?",,10,gemini-2.5-pro,0.0,42
```

Note row 5's blank `tags` cell (falls back to an unfiltered ranked
search) and that `search_query`/`max_corpus_chunks`/`skip_*`/`top_p`/etc.
are omitted entirely as columns — every one resorts to its default.

```bash
PYTHONPATH=. python rag_check/param_sweep.py \
    --input-csv rag_check/examples/sweep_example.csv \
    --out sweep_results.csv
```

### Example 2 — generated DataFrame

`rag_check/examples/dataframe_sweep_example.py` builds a `top_k x
thinking_budget` grid with `itertools.product` — the kind of sweep a CSV is
awkward for, since you'd otherwise hand-write every combination as a row —
and passes the DataFrame straight to `run_sweep()`:

```python
import itertools
import pandas as pd
from rag_check.param_sweep import run_sweep

rows = [
    {
        "prompt": "What should I do about a client disclosing suicidal ideation mid-session?",
        "tags": "session_event_tags=disclosure_SI",
        "top_k": top_k,
        "thinking_budget": thinking_budget,
        "model": "gemini-2.5-flash",
        "seed": 42,
        "skip_full_corpus": True,  # isolate the filtered arm for this grid
    }
    for top_k, thinking_budget in itertools.product([3, 10], [0, 2048])
]
sweep_df = pd.DataFrame(rows)

results = run_sweep(sweep_df, out_csv="dataframe_sweep_results.csv")
```

Run it directly:

```bash
PYTHONPATH=. python rag_check/examples/dataframe_sweep_example.py
```

## Design notes for maintainers

- **Why not reuse `ingestion/llm/` (`LLMClient`)?** That abstraction
  (`ingestion/llm/base.py`) has exactly one method, `generate_json()`,
  scoped to the ingestion pipeline's contract: always force
  `response_mime_type="application/json"`, no `seed`/`thinking_budget`/
  `top_p`/`top_k`/`max_output_tokens` knobs, because metadata generation
  never needed them. `rag_check` needs plain-text generation with those
  parameters exposed directly for a latency comparison, so
  `rag_check/llm_query.py` talks to `google.genai` directly instead of
  stretching `LLMClient`'s narrower interface to fit a different job. Its
  single `generate_content` call also intentionally has **no retry/backoff**
  (unlike `ingestion/llm/gemini.py`) — a retry would corrupt the exact
  latency measurement this tool exists to produce.
- **Chunk text retrieval**: confirmed against a live DataStore,
  `SearchService.search()` does **not** return the `Document.content`
  field inline on its results in this project, even though the DataStore
  is provisioned with `content_config=CONTENT_REQUIRED`
  (`scripts/setup_vertex_search.py::create_datastore`) — only
  `ListDocuments`/`GetDocument` return it (the same field
  `scripts/review_datastore.py` reads). So `search_filtered()`'s
  per-chunk `DocumentService.get_document()` fallback is the **normal**
  path here, not a rare edge case: retrieval for the `filtered` arm costs
  `1 + top_k` Discovery Engine round trips, not 1 — real latency, baked
  into `retrieval_latency_seconds`, not a bug. (The first decode attempt on
  the bare search result stays silent about missing content for exactly
  this reason; a warning only fires if the `get_document` fallback also
  comes back empty, which would indicate a genuinely missing chunk.)
- **Region routing**: Discovery Engine calls (`search_filtered`,
  `list_all_chunks`) go through `settings.discovery_engine_location` /
  `discovery_engine_endpoint`; the Gemini call (`llm_query.run_query`) uses
  the raw `settings.gcp_location` — same split documented in the top-level
  `CLAUDE.md` under "Region routing," and the same mistake (crossing the
  two) is what broke the first ingest.
