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

- **Array fields** (use `--tag field=single_value`; each chunk matches if
  the array *contains* that value): `therapeutic_modality`,
  `clinical_presentation`, `session_event_tags`, `applies_when`,
  `risk_dimension_tags`, `patient_population`, `clinical_measure_tags`,
  `technique_tags`, `clinical_caution`.
- **Scalar fields**: `domain`, `doc_type`, `session_phase`,
  `directionality`, `source_file`, `title`, `chapter`, `year_published`,
  `subdomain`.

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
- **Chunk text retrieval**: `search_filtered()` reads the `Document` proto
  inline on each `SearchResult` first (one Discovery Engine call, not two)
  — the DataStore is provisioned with `content_config=CONTENT_REQUIRED`
  (`scripts/setup_vertex_search.py::create_datastore`), so `content.rawBytes`
  is expected to come back the same way `scripts/review_datastore.py`
  already reads it off `ListDocuments`. If a particular result comes back
  without inline content (API behavior here has changed across Discovery
  Engine API versions — don't assume it), it falls back to a per-chunk
  `DocumentService.get_document()` call rather than silently dropping that
  chunk from the context.
- **Region routing**: Discovery Engine calls (`search_filtered`,
  `list_all_chunks`) go through `settings.discovery_engine_location` /
  `discovery_engine_endpoint`; the Gemini call (`llm_query.run_query`) uses
  the raw `settings.gcp_location` — same split documented in the top-level
  `CLAUDE.md` under "Region routing," and the same mistake (crossing the
  two) is what broke the first ingest.
