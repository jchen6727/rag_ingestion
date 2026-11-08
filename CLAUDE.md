# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

`rag_ingestion` ingests clinical PDFs into a tagged, searchable corpus and provides
tools to review the tags. Scope is **ingestion + metadata tagging + metadata
visualization only** — the query/retrieval/generation side of the original
research project is intentionally **not** part of this package. The one
exception is `rag_check/`, a separate, narrow latency-measurement harness
that reads from an already-provisioned DataStore without extending this
package's retrieval/generation surface — see "rag_check/" below.

Pipeline: `corpus/*.pdf → extractor → chunker → metadata_gen (Gemini) → uploader
(GCS) → indexer (Vertex AI Search)`, orchestrated by `scripts/batch_ingest.py`.

## Commands

Python 3.12+. There is no `pyproject.toml`; all imports are absolute from the repo
root, so **prefix commands with `PYTHONPATH=.`**.

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
pip install plotly                                                    # for the chart scripts only

PYTHONPATH=. pytest tests/                       # test suite
PYTHONPATH=. python scripts/inspect_chunks.py corpus/x.pdf --no-metadata   # local chunk preview (no cloud)
PYTHONPATH=. python scripts/show_prompt.py                                # print the exact ingestion prompt (no cloud, no PDF needed)
PYTHONPATH=. python scripts/show_prompt.py --pdf corpus/x.pdf --chunk-index 0 --constants  # + a real chunk + resolved bucket/datastore/model/etc.
PYTHONPATH=. python scripts/check_llm.py         # verify the Gemini model is reachable
scripts/preflight_check.sh                       # verify GCP setup before provisioning
PYTHONPATH=. python scripts/setup_vertex_search.py   # create datastore + register schema (once)
PYTHONPATH=. python scripts/batch_ingest.py --file corpus/x.pdf --dry-run
PYTHONPATH=. python scripts/batch_ingest.py --file corpus/x.pdf
```

Config is read from `.env` (auto-loaded on import of `config.settings`; copy from
`.env.example`). Override the schema path with `METADATA_SCHEMA_PATH`, the strategy
with `INGEST_STRATEGY` / `INGEST_CONCURRENCY`.

## Architecture notes

- **Active schema: `config/rta_v1.json`** (23 fields, real-time-analysis schema).
  It uses the `directionality` + `applies_when` model for cautions/contraindications
  and a generated enum vocabulary. `config/metadata_schema.json` (37-field unified
  schema) is retained for reference but is **not** loaded by default.
- **`config/schema_loader.py::SchemaVocabulary`** derives the controlled vocabulary
  (enums, arrays, defaults, coercion) from the schema. `metadata_gen.py` and
  `setup_vertex_search.py` both consume it, so the code never hard-codes enums.
- **`models.py::ChunkMetadata`** (pydantic) mirrors the 23 schema fields and is
  serialized to Vertex AI Search `structData`. Keep it in sync with `rta_v1.json`
  (guarded by `tests/test_rta_schema.py`).
- **LLM calls go through `ingestion/llm/`, not `google-genai` directly.**
  `MetadataGenerator` holds `self._llm: LLMClient` (built via
  `build_llm_client("gemini", ...)`) and never imports a provider SDK itself;
  `ingestion/llm/gemini.py::GeminiClient` is the only implementation today and
  owns the actual `genai.Client(vertexai=True, project=…, location=GCP_LOCATION)`
  call — no API key (`GEMINI_API_KEY` is deprecated/unused). Add `openai` /
  `anthropic` / `mistralai` / `openrouter` by writing a new `LLMClient` subclass
  and registering it in `ingestion/llm/__init__.py::_PROVIDERS`; no caller
  changes needed. Errors are categorized as `LLMConnectivityError` /
  `LLMAPIError` / `LLMGenerationError` (malformed model output — carries
  `raw_text`) / `LLMModelUnavailableError` (fatal), and
  `ingestion/llm/base.py::log_llm_error()` logs each category differently —
  in particular, it's what surfaces the actual raw text a model produced when
  `metadata_gen._validate_and_coerce()` rejects it, instead of a generic
  Google-error hint with no indication of what the model said.
  `GeminiExtractionError` / `GeminiModelUnavailableError` are still importable
  from `ingestion.metadata_gen` as aliases for
  `LLMGenerationError`/`LLMModelUnavailableError`.
- **Region routing:** Discovery Engine calls go through
  `settings.discovery_engine_location` / `discovery_engine_endpoint` (derived
  `us`/`eu`/`global` + regional endpoint) — never the raw `gcp_location`. Vertex
  Gemini, by contrast, uses the raw `gcp_location`.
- **Prompt config — single source, no code-side fallback:** hand-authored
  tagging guidance lives in **exactly one file**,
  `config/ingestion_prompt.yaml` (`settings.ingestion_prompt_path`, override
  with `INGESTION_PROMPT_PATH`; see `INGESTION_FOR_CLINICIANS.md`). The enum
  vocabulary is generated from the schema, not stored here. `metadata_gen.py`
  has **no** built-in copy of this text to fall back to — a **missing** file
  is treated the same as an **invalid** one (bad YAML, not a mapping, or a
  missing/empty required key: `system_preamble`, `output_instruction`,
  `allowed_values_header`, `closing_instruction`, `guidance`): both raise
  `IngestionPromptConfigError` and halt `MetadataGenerator` construction
  before any chunk is tagged. `batch_ingest.py` and `inspect_chunks.py` both
  catch this at startup with an actionable message. (An earlier version of
  this code merged the yaml over a hardcoded Python dict when the file was
  missing — that meant a deleted/misconfigured file silently tagged the
  corpus against text nobody was looking at. Removed; ship a valid
  `config/ingestion_prompt.yaml`, or point `INGESTION_PROMPT_PATH` at your
  own complete copy — see the file's own header comment for the required
  keys.)
  - `guidance` (list of hand-edited bullets) is the clinician-facing part.
    `schema_notes_keys` (list of `config/rta_v1.json` `notes` keys, e.g.
    `domain_vs_modality`, `directionality_applies_when`) is a second,
    explicit list in the same yaml naming which schema-tied constraints
    (like the directionality/applies_when pairing/mutual-exclusivity rule)
    get appended to the guidance block automatically — this used to be a
    tuple hardcoded in `metadata_gen.py::_extraction_guidance()`; it is now
    config, not code, so what feeds the prompt is visible in one place.
  - **See the exact prompt** any config edit produces with
    `scripts/show_prompt.py` (no cloud calls) — see Commands above.
- **Processing strategy:** `ingestion/processing_strategy.py` is the extension
  point for how chunks are batched/contextualized. Default `ChapterContextStrategy`
  tags each chunk with its chapter context; chapters run concurrently, chunks
  within a chapter sequentially. Add a strategy by subclassing and registering in
  `_STRATEGIES`.
- **Checkpoint/resume:** `batch_ingest.py` writes each chunk's metadata to
  `ingestion_checkpoints/<doc_id>.jsonl` as produced; re-running resumes and reuses
  cached chunks (delete the file to force a full re-tag). Two granularity levels:
  - Per-file: `ingest_file()` checks `GCSUploader.is_already_uploaded(doc_id)`
    *before* extracting/chunking. If this doc_id's chunk JSONL is already in
    GCS (and neither `--force` nor `--dry-run`), extraction/chunking/metadata
    are skipped entirely and the Vertex AI Search import LRO is started
    directly against the existing object — the resume path for a crash
    between "chunks uploaded" and "import LRO confirmed".
  - Per-chunk: `_generate_all_metadata()` short-circuits straight to loading
    cached metadata (no thread pool, no Gemini calls) when the checkpoint
    already covers every chunk in the document.
- **Chunk identity:** `chunk_id = f"{doc_id}_{chunk_index:05d}"`, where `doc_id` is
  the SHA-256 of the PDF bytes. It's the Vertex AI Search document ID — keep it
  stable across re-ingestion.
- **Seeing exact LLM output:** every script here (`batch_ingest.py`,
  `inspect_chunks.py`, `check_llm.py`, `show_prompt.py`) takes `--verbose`,
  which sets logging to DEBUG (`scripts/_gcp_logging.py::setup_logging`). At
  DEBUG, `ingestion/llm/gemini.py::GeminiClient.generate_json()` logs the raw
  response text for **every** call, success or failure (truncated to 4000
  chars). On failure specifically, `log_llm_error()` (`ingestion/llm/base.py`)
  logs that same raw text at ERROR level even without `--verbose`, since a
  `LLMGenerationError` carries it as `.raw_text` — this is what surfaces a
  malformed/non-JSON Gemini response instead of a generic error with no
  indication of what the model actually said. For a durable record rather
  than a log line: successfully-tagged chunks' *coerced* output (not the raw
  pre-coercion text) is written to `ingestion_checkpoints/<doc_id>.jsonl`
  as-produced.
- **LLM output determinism:** temperature is 0.0 by default but this does
  **not** make output byte-reproducible — see `determinism.txt` at the repo
  root for the investigation (rules out this codebase's threading/chapter
  strategy with code-level evidence; the actual cause is server-side
  batching/floating-point non-determinism in the hosted Gemini API, outside
  this pipeline's control).

## Metadata review scripts (operate on a chunks `.jsonl`)

- `scripts/metadata_viz.py` — tag-frequency charts (plotly → HTML).
- `scripts/metadata_jaccard.py` — within-field tag co-occurrence (Jaccard heatmap).
- `scripts/view_chunks.py` — filter chunks by tag, print decoded **text**.
- `scripts/compare_jsonl.py` — diff two chunks `.jsonl` (e.g. two runs of the
  same PDF across commits, or two independent ingestions of the same PDF —
  see `determinism.txt`) by `chunk_id`, flagging any tag that differs for the
  same passage. Trailing positional args restrict both the diff and the
  chart to specific fields, e.g. `compare_jsonl.py a.jsonl b.jsonl
  applies_when session_event_tags clinical_presentation`; a requested field
  name not found in either file logs a warning (silently zero-diffing an
  unknown field would otherwise look identical to "no mismatches"). Writes
  `diff_<label1>_<label2>[_fields-...].log` (per-chunk diffs + a
  mismatch-count-by-field table) and the matching `.html` (plotly: per-tag
  frequency in file1 vs. file2, sorted by `|difference|`) next to `file1` —
  neither input `.jsonl` is modified. Provenance fields (`doc_id`,
  `source_file`, `page_start`, `page_end`, `chunk_index`) are always excluded
  since they're copied from the `Chunk`, not produced by the LLM.

`metadata_viz.py` and `metadata_jaccard.py` only need the flat tag dict, so
both auto-detect, per line, which of two shapes they were handed — no
conversion step needed:
  - the `structData`-format JSONL ingestion uploads to GCS (download with
    `gcloud storage cp gs://<bucket>/chunks/<doc_id>.jsonl`): `{"id": ...,
    "structData": {...tags...}}`;
  - the local, pre-upload JSONL you already have right after ingestion: either
    `ingestion_checkpoints/<doc_id>.jsonl` (batch_ingest.py's checkpoint) or
    `inspect_chunks.py`'s `<name>.<docid8>.chunks.jsonl` review file — both
    shaped as `{"chunk_id": ..., "metadata": {...tags...}}`.
  A line matching neither shape raises a clear error naming the keys it did
  find, rather than a bare `KeyError`.

`view_chunks.py` is the one exception: it also needs each chunk's decoded
**text**, which is only present in the uploaded format (`content.rawBytes`,
base64) — the local checkpoint/review formats don't carry chunk text at all
(by design, to stay small), so it still requires the GCS-uploaded JSONL.

## `rag_check/` (RAG latency measurement — separate from ingestion scope)

A narrow measurement harness that reads from a DataStore already provisioned
(`scripts/setup_vertex_search.py`) and populated (`scripts/batch_ingest.py`);
it does not write to the DataStore and does not import from `ingestion/`.
Full usage, column reference, and worked examples in `rag_check/README.md`.
`poster/` is built on top of this (a 9-tier prompt-complexity sweep for a
scientific poster, multiple trials, several real bugs found and fixed) —
see [`poster/summary.md`](poster/summary.md) for that narrative and current
priorities.

```bash
PYTHONPATH=. python rag_check/compare_latency.py --prompt "..." --tag session_event_tags=disclosure_SI
PYTHONPATH=. python rag_check/param_sweep.py --input-csv rag_check/examples/sweep_example.csv --out results.csv
```

- **`compare_latency.py`** times one LLM call under two retrieval
  strategies against the same prompt/model params: `filtered` (Discovery
  Engine `Search`, ranked, optional `--tag field=value` metadata filter —
  omit `--tag` entirely for "retrieve/rank across all documents") vs.
  `full_corpus` (`ListDocuments`, every chunk, unranked/unfiltered).
  Accepts the prompt, the retrieval tag, and the relevant `genai` request
  knobs (`--model`, `--temperature`, `--top-p`, `--sampling-top-k`,
  `--seed`, `--max-output-tokens`, `--thinking-budget`). `run_comparison()`
  is the one implementation both this CLI and `param_sweep.py` call.
- **`retrieval.py`** — `build_filter_expression()`: Discovery Engine
  rejects `=`/`<`/`>` on every text-indexed field, scalar included
  (confirmed live: `directionality = "x"` fails with "Unsupported field
  ... on comparison operators") — every text field (scalar or array) uses
  `field: ANY("value")`, and only the one integer field (`year_published`)
  uses `=`. Also confirmed live: `SearchService.search()` does not return
  `Document.content` inline in this project even with
  `content_config=CONTENT_REQUIRED` — only `ListDocuments`/`GetDocument` do
  — so `search_filtered()` always falls back to a per-chunk `GetDocument`
  call, making the filtered arm's retrieval cost `1 + top_k` round trips,
  not 1 (real latency, not a bug).
- **`llm_query.py`** calls `google.genai` directly, not `ingestion/llm/`'s
  `LLMClient` (that's JSON-only, no seed/thinking/top_p knobs), with no
  retry/backoff — a retry would corrupt the latency measurement.
- **`param_sweep.py`** runs `run_comparison()` once per row of a CSV or an
  in-memory DataFrame (`run_sweep_from_csv()` / `run_sweep()`); any
  argument column absent or blank falls back to `compare_latency.py`'s own
  default. Output is long-format (one row per input row × arm run), inputs
  echoed back under `input_*`. See `rag_check/examples/` for a CSV and an
  `itertools.product`-built DataFrame sweep.

## Gotchas

- **Front matter is not filtered.** `rta_v1.json`'s `doc_type` has no `front_matter`
  value, so title/TOC pages are chunked and indexed. Spot-check the first chunks of
  each document.
- **Pick a current Gemini model.** Google retires older models; set
  `GEMINI_MODEL_METADATA` in `.env` and verify with `scripts/check_llm.py`.
- **Shell scripts:** `preflight_check.sh` uses REST (curl) for its checks; only
  interactive auth and API-enable stay on `gcloud`. `verify_datastore.sh` has an
  editable hardcoded project ID — treat it as a scratch helper.
- **Keep `.env` private** (it points at a cloud project); never commit it.
- **Tagging isn't perfectly reproducible run-to-run**, even at temperature=0 —
  see `determinism.txt`. Don't re-run ingestion expecting a different tag for
  the same chunk; delete the specific line (or file) in
  `ingestion_checkpoints/` if you deliberately want a fresh Gemini call.
- CI (`.github/workflows/ci.yml`) runs the schema + core tests on PRs to `main` and
  manual dispatch. Some local tests need PDF fixtures in `tests/fixtures/`.
