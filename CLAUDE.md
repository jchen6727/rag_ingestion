# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

`rag_ingestion` ingests clinical PDFs into a tagged, searchable corpus and provides
tools to review the tags. Scope is **ingestion + metadata tagging + metadata
visualization only** — the query/retrieval/generation side of the original
research project is intentionally **not** part of this package.

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
- **Prompt config:** hand-authored tagging guidance lives in
  `config/ingestion_prompt.yaml` (see `INGESTION_FOR_CLINICIANS.md`); the enum
  vocabulary is generated from the schema. A **missing** file falls back to
  built-in defaults in `metadata_gen.py` (expected/normal). A file that
  **exists but is invalid** (bad YAML, not a mapping, or missing/empty a
  required framing key) raises `IngestionPromptConfigError` and halts
  `MetadataGenerator` construction — `batch_ingest.py` and `inspect_chunks.py`
  both catch this at startup with an actionable message before any chunk is
  processed, rather than silently tagging the corpus against unintended
  defaults.
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

## Metadata review scripts (operate on an exported chunks `.jsonl`)

These read the `structData`-format JSONL that ingestion uploads to GCS (download
with `gcloud storage cp gs://<bucket>/chunks/<doc_id>.jsonl`), not the local
`inspect_chunks` review file:

- `scripts/view_chunks.py` — filter chunks by tag, print decoded text.
- `scripts/metadata_viz.py` — tag-frequency charts (plotly → HTML).
- `scripts/metadata_jaccard.py` — within-field tag co-occurrence (Jaccard heatmap).

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
- CI (`.github/workflows/ci.yml`) runs the schema + core tests on PRs to `main` and
  manual dispatch. Some local tests need PDF fixtures in `tests/fixtures/`.
