# Changelog

All notable changes to this package.

> **Origin.** This package (`rag_ingestion`) was extracted as a clean, standalone
> tool from a larger internal research project (`rag_guidance`). Every change
> below was made **during work on `rag_guidance`** and is recorded here so the
> extracted package carries its own history. Dates are when the work was done in
> that project.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

### Added
- **Provider-agnostic LLM client abstraction** (2026-08-31): `ingestion/llm/`
  (`LLMClient` ABC + `GeminiClient`) now owns every direct `google-genai` call
  that used to live in `metadata_gen.py` (`_get_client`/`_verify_model_available`/
  the retry loop). `MetadataGenerator` talks only to `self._llm: LLMClient`, built
  via `build_llm_client("gemini", ...)`. Swapping in `openai` / `anthropic` /
  `mistralai` / `openrouter` means writing one new `LLMClient` subclass (see
  `ingestion/llm/gemini.py` for the shape) and registering it in
  `ingestion/llm/__init__.py::_PROVIDERS` — no caller changes.
  - Exceptions are categorized so callers can handle/report each kind
    differently: `LLMConnectivityError` (transient, safe to retry),
    `LLMAPIError` (auth/quota/request problem), `LLMGenerationError` (the call
    succeeded but the model's output wasn't usable JSON — carries `raw_text`),
    and `LLMModelUnavailableError` (fatal, mirrors the existing
    `GeminiModelUnavailableError`). `GeminiExtractionError` and
    `GeminiModelUnavailableError` remain importable from `ingestion.metadata_gen`
    as aliases for `LLMGenerationError`/`LLMModelUnavailableError`, so existing
    callers don't need to change.
- **Malformed-LLM-output logging** (2026-08-31): `ingestion/llm/base.py::log_llm_error()`
  logs the actual raw text a model produced when `_validate_and_coerce()` rejects
  it (non-JSON response, or schema-invalid after coercion) — previously this was
  either dropped entirely (`_call_gemini`'s `JSONDecodeError` path never captured
  `response.text`) or reduced to a generic Google-error hint with no indication
  of what the model actually said (`scripts/inspect_chunks.py` calling
  `log_api_error` on a plain pydantic `ValidationError`). Wired into both
  `MetadataGenerator.generate()` and `scripts/inspect_chunks.py::_generate_metadata()`.
- **`config/ingestion_prompt.yaml` validation** (2026-08-31): a MISSING file is
  still the documented, expected case (falls back to built-in defaults, INFO
  log). A file that EXISTS but fails to parse, isn't a YAML mapping, or is
  missing/mistypes a required framing key (`system_preamble`,
  `output_instruction`, `allowed_values_header`, `closing_instruction`,
  `guidance`) now raises `IngestionPromptConfigError` and halts
  `MetadataGenerator` construction, instead of silently tagging the whole
  corpus against defaults nobody asked for — same fail-fast philosophy as
  `GeminiModelUnavailableError`. `scripts/batch_ingest.py` and
  `scripts/inspect_chunks.py` both catch it at startup with an actionable
  message and exit before any chunk is processed.
- **Finer-grained ingestion checkpointing** (2026-08-31):
  - `scripts/batch_ingest.py::ingest_file()` now checks
    `GCSUploader.is_already_uploaded(doc_id)` *before* extracting/chunking a
    PDF. If this doc_id's chunk JSONL is already sitting in GCS (and neither
    `--force` nor `--dry-run` was passed), extraction, chunking, and metadata
    generation are skipped entirely and the Vertex AI Search import LRO is
    started directly against the existing object (`_import_existing_chunks()`).
    This is the resume path for a crash between "chunks uploaded" and "import
    LRO confirmed" — exactly the gap the LRO bug above could leave a run in,
    which previously meant re-extracting and re-chunking the whole PDF for
    nothing.
  - `scripts/batch_ingest.py::_generate_all_metadata()` now checks whether the
    per-chunk metadata checkpoint already covers every chunk
    (`_checkpoint_covers_all()`) before building the strategy's units/thread
    pool, and short-circuits straight to loading cached metadata if so —
    previously the checkpoint was only consulted chunk-by-chunk inside an
    already-spun-up thread pool.
  - `ingestion/uploader.py`: added `GCSUploader.gcs_pdf_uri()` /
    `gcs_chunks_uri()` (derive the URI without a network call) and
    `count_uploaded_chunks()` (line count of an already-uploaded chunks JSONL,
    for reporting on the short-circuit path).

### Fixed
- **`batch_ingest.py` failing LRO calls at the end of metadata generation.
  ** (2026-08-31)
  - `ingestion/indexer.py` `_transport.operations_client` object in 
    `discoveryengine.DocumentServiceClient`, method `.get_operation` incorrectly 
    handled (need to provide a string to `name`). Additionally, the op returned 
    by the method has metadata and response error values that must be retrieved
    separately.
- **`batch_ingest.py` silently degrading to fallback metadata when the
  configured Gemini model isn't servable in `GCP_LOCATION`** (2026-08-20):
  `gemini-3.6-flash` appeared in `client.models.list()` / `.get()` (both of
  which query a global model-garden catalog, not the client's configured
  region) and so looked available, but a real `generate_content` call against
  `us-central1` returned `404 NOT_FOUND` — as of this writing the model is only
  servable via the `global` Vertex AI location, not `us-central1` (or any other
  region-scoped endpoint tested). Because `metadata_gen.py::generate()` catches
  all exceptions per chunk and falls back to rule-based extraction, this
  previously failed *silently*: three retries of the same 404 per chunk,
  logged only as `WARNING`, then the entire corpus got tagged with degraded
  fallback metadata instead of the run stopping.
  - `ingestion/metadata_gen.py`: added `GeminiModelUnavailableError` and
    `MetadataGenerator._verify_model_available()`, which makes one real
    `generate_content` call (the only reliable availability signal) the first
    time the client is initialized, cached for the process lifetime. `generate()`
    now re-raises this error instead of swallowing it into the fallback path.
  - `scripts/batch_ingest.py`: calls `metadata_gen.verify_model_available()`
    once at startup, inside the existing client-init `try/except` that already
    exits with code 1 on failure — so a bad model now aborts before any file is
    processed, not partway through with degraded output.
  - `scripts/preflight_check.sh`: added a check (section 6) that makes the same
    real `generateContent` REST call for `GEMINI_MODEL_METADATA` against
    `GCP_LOCATION` before any provisioning, with a pointer to
    `check_llm.py --list` to find a model that IS servable in the region.
  - `scripts/check_llm.py --list` had the same catalog-vs-servable gap — it
    printed every model the catalog listed as "available", `gemini-3.6-flash`
    included. It now probes each candidate `gemini-*` model with a real
    `generate_content` ping (concurrently) and splits the output into
    "servable here" vs. "listed but not servable here", so it actually answers
    the question it claims to.
  - `scripts/_gcp_logging.py`: added `google_genai` to the noisy-logger list —
    it's a separate root logger from `google.*`, so its "AFC is enabled..."
    line was printing once per API call regardless of `--verbose`.
  - `GCP_LOCATION` is intentionally left as a hard compute-region setting (used
    by Discovery Engine/GCS routing too) — the fix does not attempt to route
    Gemini calls to a different location. To resolve the actual 404, set
    `GEMINI_MODEL_METADATA` in `.env` to a model already available in your
    `GCP_LOCATION` (verify with `scripts/check_llm.py --list`), or wait for the
    model to roll out to that region.

## [0.1.0] — Initial internal release (extracted 2026-07-30)

First standalone release for the internal clinical liaison team: PDF ingestion,
metadata tagging, and metadata review/visualization.

### Added
- **Metadata visualization & inspection tools** (2026-07-30): `scripts/metadata_viz.py`
  (tag-frequency charts), `scripts/metadata_jaccard.py` (within-field tag
  co-occurrence via Jaccard similarity), and `scripts/view_chunks.py` (filter and
  read chunk text by tag) — all operate on an exported chunks `.jsonl`.
- **Clinician-facing prompt guide** (`INGESTION_FOR_CLINICIANS.md`): how to tune
  the metadata-tagging instructions without coding.
- **Pluggable processing strategies** (2026-07-27): `ingestion/processing_strategy.py`
  with `IndependentStrategy` and the default `ChapterContextStrategy` (tags each
  chunk with its chapter context; chapters run concurrently, chunks within a
  chapter sequentially). Configurable via `INGEST_STRATEGY` / `INGEST_CONCURRENCY`
  or `--strategy` / `--concurrency`.
- **Crash-safe checkpointing & resume** (2026-07-26): `scripts/batch_ingest.py`
  writes each chunk's metadata to `ingestion_checkpoints/<doc_id>.jsonl` as it is
  produced; a re-run resumes and does not re-pay for already-tagged chunks.
- **Progress logging** at the default level (`…metadata X/N (Y%)`).
- **`.env` auto-loading**: importing `config.settings` loads `.env` — no manual
  `source .env` needed.
- **Model availability check** (`scripts/check_llm.py`): verifies the configured
  Gemini model is reachable on Vertex AI before a long run.
- **Externalized ingestion prompt**: hand-authored tagging guidance moved to
  `config/ingestion_prompt.yaml` (the tag vocabulary stays generated from the
  schema); a missing/invalid file falls back to built-in defaults.
- **Preflight coverage** (`scripts/preflight_check.sh`): full end-to-end IAM
  permission set and required-API list (incl. Gemini/Vertex).
- **Starter CI** (`.github/workflows/ci.yml`): runs the schema + core tests on
  pull requests to `main` and on manual dispatch.

### Changed
- **LLM backend migrated to `google-genai` + Vertex AI (ADC)** (2026-07-27):
  metadata generation now authenticates via Application Default Credentials — no
  API key. Replaces the deprecated `google-generativeai` package.
- **Active metadata schema is `config/rta_v1.json`** (the 23-field real-time
  schema with the `directionality` + `applies_when` model). The 37-field unified
  schema `config/metadata_schema.json` is retained for reference/future work.
- **Discovery Engine region routing** now derives the correct multi-region
  (`us`/`eu`/`global`) and regional endpoint from `GCP_LOCATION` via
  `config/settings.py`, fixing an import-time region mismatch.
- **Shell checks use REST**: `preflight_check.sh` performs its project / billing /
  enabled-API checks via the Cloud REST APIs (curl); only interactive auth and
  API-enable commands remain on `gcloud`.

### Fixed
- Vertex AI Search import failing with an `INVALID_ARGUMENT` "endpoint can only
  serve global region" error when `GCP_LOCATION` was a compute region such as
  `us-central1`.

### Notes / not yet done
- Pick a current Gemini model in `.env` (`GEMINI_MODEL_METADATA`) — older models
  are being retired; verify with `scripts/check_llm.py`.
- The query/retrieval side of the original project is **not** part of this
  package; this package is ingestion + metadata review only.
