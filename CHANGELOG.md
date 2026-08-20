# Changelog

All notable changes to this package.

> **Origin.** This package (`rag_ingestion`) was extracted as a clean, standalone
> tool from a larger internal research project (`rag_guidance`). Every change
> below was made **during work on `rag_guidance`** and is recorded here so the
> extracted package carries its own history. Dates are when the work was done in
> that project.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com).

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
