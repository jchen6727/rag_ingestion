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
- **`poster/` — 9-tier prompt-complexity sweep + figure for a scientific
  poster** (2026-11-07): built on `rag_check/param_sweep.py` to answer
  whether metadata-filtered RAG holds up better than whole-corpus RAG as
  prompt complexity escalates, on a single clinical topic
  (`session_event_tags=disclosure_SI`). **Start at
  `poster/summary.md`** — the narrative entry point, with ranked priorities
  and bootstrapping instructions for picking this up cold; `poster/README.md`
  is the mechanics/column reference.
  - Four trials, each correcting a specific, named mistake in the one
    before it rather than silently replacing it — all documented, with
    frozen code+data+figure snapshots, under `poster/trials/`:
    - **trial_0**: first live run surfaced a paradox *opposite* the one
      motivating the poster (`filtered`, not `full_corpus`, was the
      fast-but-unreliable arm) — root-caused to schema fields the corpus
      couldn't support (`RISK_LEVEL`, `CONFIDENCE`, ...) and prompt-mandated
      minimum item counts forcing fabrication-or-refusal.
    - **trial_1**: redesigned schema + raised `top_k`, which in turn exposed
      two more bugs found while investigating the new data: a quality-
      scoring false positive (valid JSON honestly reporting an empty finding
      was mis-flagged as a hard failure), and a retrieval bug — Discovery
      Engine parses literal `"..."` quote marks in a search query as an
      exact-phrase operator, so prompts quoting invented example text
      silently retrieved **zero chunks** on 6 of 9 tiers. Both fixed
      (`poster/quality.py`, `poster/sweep.py`); corrected rerun archived as
      `trial_1_rerun/`. A related confound (two "narrative distractor"
      tiers retrieving fewer chunks than the rest, from their own off-topic
      content diluting the search query) was found and documented, not
      fixed.
    - **trial_2**: `top_k` lowered 20→5 to match the actual target
      deployment (a therapist-assistant harness needing concise, predictable
      output, not exhaustive corpus narration) — chosen on that basis before
      checking which arm it would favor, with the `top_k=20` run preserved
      unmodified as an exploratory baseline. Cleanest result of the four
      trials, and the `top_k` reduction incidentally resolved trial_1's
      narrative-distractor confound as a side effect.
      - **Figure revised after supervisor review** (same trial_2 data, no
        rerun): `poster/plot_results.py` went from three stacked panels to
        one. X-axis changed from a labeled "complexity tier" to a plain
        prompt number (supervisors didn't want these 9 prompts framed as a
        validated difficulty ladder); every individual repeat is now
        plotted as a jittered point on top of its bar, not just the mean
        (SEM error bars kept); the structural-variability panel was removed
        (interrater disagreement on what counts as "correct" output
        structure, surfaced during review — `_aggregate_variability()`
        stays in the code, unused, for later); the hard-failure-rate panel
        was folded into the latency plot as a per-point flag (a reserved
        status-red "x" marker + one legend entry) instead of a separate
        chart. The underlying latency finding was unchanged and called
        defensible as-is.
  - `rag_check/retrieval.py` also gained two real fixes discovered while
    building this: Discovery Engine's filter grammar needs
    `field: ANY("value")` for **every** text-indexed field (scalar or
    array), not `=` for scalars as originally assumed (confirmed live:
    `directionality = "x"` errors; only the one integer field,
    `year_published`, uses `=`); and `SearchService.search()` doesn't return
    chunk content inline in this project even with
    `content_config=CONTENT_REQUIRED`, so `search_filtered()` now falls back
    to a per-chunk `GetDocument` call rather than silently dropping chunks.
  - **No structural evidence for or against the original hallucination-
    paradox hypothesis has appeared in any trial after the bugs were
    fixed** — flagged as the top open priority in `poster/summary.md`:
    nothing in this pipeline verifies response content is actually grounded
    in retrieved passages versus the model's prior knowledge, only that it
    parses correctly.
- **`rag_check/` — RAG latency comparison + parameter sweep harness**
  (2026-11-06): a separate, narrow measurement tool, explicitly **not** part
  of this package's ingestion/tagging/visualization scope (see "What this
  is") — it reads from a DataStore already provisioned/populated by
  `scripts/setup_vertex_search.py` / `scripts/batch_ingest.py` and does not
  write to it or import from `ingestion/`. Full usage in `rag_check/README.md`.
  - **`compare_latency.py`**: times one LLM call under two retrieval
    strategies against the same prompt and the same Gemini request
    parameters — `filtered` (Discovery Engine `Search`, ranked, optionally
    restricted by one or more `--tag field=value` metadata filters) vs.
    `full_corpus` (`ListDocuments`, every chunk, unranked and unfiltered).
    Retrieval time and generation time are measured separately. Accepts the
    prompt, the retrieval tag/filter (including an "all documents" mode —
    simply omit `--tag`), and the full set of relevant `genai` request
    knobs: `--model`, `--temperature`, `--top-p`, `--sampling-top-k`,
    `--seed`, `--max-output-tokens`, `--thinking-budget`. Its core,
    `run_comparison()`, is the single implementation both this CLI and
    `param_sweep.py` call, so a sweep row can't measure something subtly
    different from what the one-off CLI would for the same inputs.
  - **`retrieval.py`**: `build_filter_expression()` builds the Discovery
    Engine filter string for `--tag`, validating every field name against
    the schema's actually-registered filterable fields
    (`config/rta_v1.json` `notes.vertex_ai_search`) rather than letting an
    unregistered/misspelled field silently no-op the filter at the API
    layer. Confirmed against a live DataStore while building this: Discovery
    Engine rejects `=`/`<`/`>` comparison operators on **every** text-indexed
    field, scalar included — not just arrays (`directionality = "x"` fails
    with "Unsupported field ... on comparison operators"). Every text field
    (scalar or array) therefore uses `field: ANY("value")`; only the one
    true integer field, `year_published`, uses `=`. Also confirmed live:
    `SearchService.search()` does not return `Document.content` inline in
    this project even though the DataStore has
    `content_config=CONTENT_REQUIRED` — only `ListDocuments`/`GetDocument`
    do — so `search_filtered()` always falls back to a per-chunk
    `GetDocument` call for chunk text, meaning the `filtered` arm's
    retrieval cost is `1 + top_k` Discovery Engine round trips, not 1 (this
    is real, reported latency, not a bug). The decode helper only logs a
    warning if that fallback *also* comes back empty, to avoid a
    false-alarm warning on the now-expected missing-inline-content case.
  - **`llm_query.py`**: calls `google.genai` directly rather than going
    through `ingestion/llm/`'s `LLMClient` — that abstraction's one method,
    `generate_json()`, is scoped to always force JSON output for
    `ChunkMetadata` and has no `seed`/`thinking_budget`/`top_p`/`top_k`/
    `max_output_tokens` knobs. Its single `generate_content` call also has
    no retry/backoff (unlike `ingestion/llm/gemini.py`), since a retry would
    corrupt the latency measurement this tool exists to produce.
  - **`param_sweep.py`**: runs `compare_latency.run_comparison()` once per
    row of an input table — `run_sweep_from_csv(csv_path, out_csv=...)` for
    a CSV file, or `run_sweep(df, out_csv=...)` as the method wrapper for a
    DataFrame built programmatically (e.g. `itertools.product` over a
    parameter grid). Every argument is an optional column; a column absent
    from the table, or blank/NaN for a given row, falls back to
    `compare_latency.py`'s own default for that argument — only `prompt` is
    required. Output is long-format (one row per input row × arm actually
    run for that row), every input column echoed back under an `input_`
    prefix alongside the measured latency/token/answer outputs; a row whose
    run raises an exception gets `arm="error"` with the message instead of
    halting the whole sweep. `rag_check/examples/sweep_example.csv` and
    `rag_check/examples/dataframe_sweep_example.py` are worked examples of
    each entry point.

### Added
- **`scripts/compare_jsonl.py`** (2026-10-16): diffs two chunks `.jsonl` files
  by `chunk_id` and flags every tag that differs for the same passage (e.g.
  re-tagging the same PDF at temperature=0 across two runs — see
  `determinism.txt` — or checking whether a prompt/schema/model change
  between two commits shifted tags; `chunk_archive/NOTES.md` has this repo's
  `run_000N.jsonl` <-> source-PDF naming). Accepts the same dual input format
  as `metadata_viz.py`/`metadata_jaccard.py` (GCS-uploaded `structData` or the
  local checkpoint/review `metadata` shape — the two inputs may even differ
  from each other) and never modifies either input file. Writes
  `diff_<label1>_<label2>[_fields-...].log` (summary counts, chunk_ids present
  in only one file, a mismatch-count-by-field table, then every per-chunk
  field diff) and the matching `.html` (plotly grouped-bar chart of each
  tag's frequency in file1 vs. file2, sorted by `|difference|`, top N) next to
  `file1` by default (`--out-dir` to redirect, `--label1`/`--label2` to avoid
  a filename collision when both inputs share a stem). Any number of trailing
  positional field names (e.g. `applies_when session_event_tags
  clinical_presentation`) restricts both the diff and the chart to just those
  fields and is reflected in the output filename; a requested field not found
  in either file logs a warning rather than silently producing an
  all-clear-looking empty diff. Provenance fields (`doc_id`, `source_file`,
  `page_start`, `page_end`, `chunk_index`) are always excluded since they're
  copied from the `Chunk`, never produced by the LLM, so a difference there
  would only ever
  mean the two chunk_ids aren't the same passage.

### Fixed
- **`metadata_viz.py` / `metadata_jaccard.py` crashed (`KeyError: 'structData'`)
  on the local, pre-upload chunks `.jsonl`** (2026-10-02): both scripts
  unconditionally read `json.loads(line)["structData"]`, a key that only
  exists in the format `ingestion/uploader.py` writes to GCS. Running either
  script against `ingestion_checkpoints/<doc_id>.jsonl` (or
  `inspect_chunks.py`'s `*.chunks.jsonl` review file) — both shaped as
  `{"chunk_id": ..., "metadata": {...}}` — raised immediately, before a
  document was ever uploaded. Both scripts now auto-detect, per line, which
  of the two shapes they were handed (`structData` vs. `metadata`) with no
  need to pre-convert the file; an unrecognized shape raises a clear error
  naming the keys actually found instead of a bare `KeyError`. `view_chunks.py`
  is unchanged — it also needs each chunk's decoded text, which only exists
  in the GCS-uploaded format, so it still requires that format specifically.

### Added
- **`config/ingestion_prompt.yaml` is now the single, required source of the
  ingestion prompt** (2026-09-04): removed the hardcoded `_DEFAULT_PROMPT_CFG`
  fallback dict from `ingestion/metadata_gen.py`. Previously a **missing**
  file silently fell back to a Python-side copy of (nominally) the same text,
  which meant the prompt an operator edited in the yaml and the prompt
  actually used by a run without that file could silently diverge — two
  sources of truth for one prompt. Now a missing file is treated exactly like
  an invalid one: `IngestionPromptConfigError` and a halt before any chunk is
  tagged (same as before for invalid-but-present files). Ship a valid
  `config/ingestion_prompt.yaml` (already checked into the repo), or point
  `INGESTION_PROMPT_PATH` at your own complete copy.
  - `INGESTION_FOR_CLINICIANS.md` and `CLAUDE.md` updated to describe the
    halt-on-missing behavior; the old "quietly falls back" language was no
    longer accurate and has been removed everywhere it appeared.
- **`schema_notes_keys` — prompt-crafting rules made explicit in config, not
  code** (2026-09-04): which `config/rta_v1.json` `notes` entries get appended
  to the extraction guidance block (the `directionality`/`applies_when`
  mutual-exclusivity/pairing rule, and the `domain` vs. `therapeutic_modality`
  distinction) used to be a tuple hardcoded in
  `metadata_gen.py::_extraction_guidance()`. It is now a `schema_notes_keys`
  list in `config/ingestion_prompt.yaml` (optional, defaults to `[]`, validated
  as a list of strings) — an operator can see and change what schema-tied text
  augments the prompt without touching Python.
- **`scripts/show_prompt.py`** (2026-09-04): prints the exact prompt
  `metadata_gen.py` sends to the LLM — either for a placeholder chunk (no PDF,
  no cloud calls, useful for checking a prompt edit landed) or for a real
  chunk of a real PDF via `--pdf ... --chunk-index N` (includes whatever
  chapter context the active `INGEST_STRATEGY` would attach). `--constants`
  additionally prints every resolved runtime constant a real run would use:
  GCP project/location, GCS bucket, Vertex AI Search datastore/engine ID,
  Gemini model, ingest strategy/concurrency, and the prompt/schema/checkpoint
  paths — each read defensively so an incomplete `.env` prints `<not set>`
  per-value instead of crashing the whole command.
- **Raw LLM response now logged at DEBUG on success, not just failure**
  (2026-09-04): `ingestion/llm/gemini.py::GeminiClient.generate_json()` logs
  the exact (truncated) response text for every call. Previously the raw text
  was only ever surfaced via `LLMGenerationError.raw_text` on a *failed* call
  (see `log_llm_error()` below); a successful call's exact output was never
  written anywhere but the post-coercion `ChunkMetadata`. Run any script with
  `--verbose` to see it (see CLAUDE.md "Seeing exact LLM output").
- **`determinism.txt`** (2026-09-04): investigates reports of non-reproducible
  tagging at `temperature=0.0`. Rules out this codebase's own concurrency
  (`ThreadPoolExecutor` in `batch_ingest.py`) and the `ChapterContextStrategy`
  chapter-context composition as the cause, with code-level evidence that the
  exact prompt built for a given chunk is independent of thread scheduling and
  `INGEST_CONCURRENCY`. Identifies the actual cause as inherent to hosted
  Gemini API serving (batching/floating-point non-associativity, no `seed`
  pinned, unpinned model alias) — outside this pipeline's control — and notes
  the checkpoint files already exist for exactly this reason (a re-run reuses
  cached tags rather than re-rolling them).
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
