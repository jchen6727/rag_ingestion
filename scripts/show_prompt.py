"""
Print the exact prompt sent to the ingestion LLM for metadata tagging, and the
resolved runtime constants (GCS bucket, Vertex AI Search datastore/engine, the
Gemini model, GCP location, and the config file paths) that
scripts/batch_ingest.py uses for a real run.

No cloud calls are made. Building the prompt only needs config/rta_v1.json and
config/ingestion_prompt.yaml to load successfully — a live GCP connection or
credentials are never required, so this also works to sanity-check a prompt
edit before `gcloud auth application-default login`.

Usage:
    # Constants only (bucket, datastore, engine, model, ...):
    PYTHONPATH=. python scripts/show_prompt.py --constants

    # The exact prompt for a placeholder chunk (checks structure/wording only):
    PYTHONPATH=. python scripts/show_prompt.py

    # The exact prompt for a real chunk of a real PDF, including whatever
    # chapter context the active INGEST_STRATEGY would attach to it:
    PYTHONPATH=. python scripts/show_prompt.py --pdf corpus/x.pdf --chunk-index 3

    # Both prompt and constants:
    PYTHONPATH=. python scripts/show_prompt.py --pdf corpus/x.pdf --constants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._gcp_logging import setup_logging

from config.settings import settings
from ingestion.chunker import ChunkerConfig, ContextAwareChunker
from ingestion.extractor import PDFExtractor
from ingestion.metadata_gen import IngestionPromptConfigError, MetadataGenerator
from ingestion.processing_strategy import build_strategy
from models import Chunk

logger = None  # set in main() via setup_logging

_PLACEHOLDER_TEXT = (
    "<chunk text goes here — this is a placeholder shown so you can inspect the "
    "prompt's structure/wording without a PDF. Pass --pdf to see it filled in "
    "with a real passage.>"
)


def _print_constants() -> None:
    """Print the resolved runtime constants a real ingestion run would use.

    Each value is read defensively (not settings.validate_all()) so this still
    works to preview config with an incomplete .env — unset required values
    print as "<not set>" instead of raising.
    """
    def _get(label: str, fn) -> None:
        try:
            print(f"  {label}: {fn()}")
        except Exception as exc:  # noqa: BLE001 — EnvironmentError from settings._require
            print(f"  {label}: <not set> ({exc})")

    print("=" * 68)
    print("Resolved runtime constants")
    print("=" * 68)
    _get("GCP project", lambda: settings.gcp_project_id)
    _get("GCP location (compute, used for Gemini)", lambda: settings.gcp_location)
    _get("Discovery Engine location (derived, used for Vertex AI Search)",
         lambda: settings.discovery_engine_location)
    _get("GCS bucket", lambda: settings.gcs_bucket_name)
    _get("Vertex AI Search datastore ID", lambda: settings.vertex_search_datastore_id)
    _get("Vertex AI Search engine ID", lambda: settings.vertex_search_engine_id)
    _get("Gemini model (metadata tagging)", lambda: settings.gemini_model_metadata)
    _get("Ingest strategy", lambda: settings.ingest_strategy)
    _get("Ingest concurrency", lambda: settings.ingest_concurrency)
    _get("Ingestion prompt config path", lambda: settings.ingestion_prompt_path)
    _get("Metadata schema path", lambda: settings.metadata_schema_path)
    _get("Checkpoint dir", lambda: settings.checkpoint_dir)
    print("=" * 68)


def _build_placeholder_chunk() -> Chunk:
    return Chunk(
        chunk_id="placeholder_00000",
        doc_id="placeholder",
        text=_PLACEHOLDER_TEXT,
        page_start=1,
        page_end=1,
        chunk_index=0,
        parent_section="",
    )


def _load_real_chunk(pdf: Path, chunk_index: int) -> tuple[Chunk, str]:
    """Extract+chunk `pdf` the same way inspect_chunks.py does, and compute the
    context_window the active INGEST_STRATEGY would attach to chunk_index.

    Returns:
        (chunk, context_window)
    """
    extractor = PDFExtractor(use_document_ai=False)
    try:
        config = ChunkerConfig.from_yaml(settings.chunk_config_path)
    except Exception:
        config = ChunkerConfig()
    chunker = ContextAwareChunker(config)

    doc = extractor.extract(pdf)
    chunks = chunker.chunk(doc)
    if not chunks:
        raise ValueError(f"{pdf} produced no chunks.")
    if not (0 <= chunk_index < len(chunks)):
        raise ValueError(
            f"--chunk-index {chunk_index} out of range: {pdf} has {len(chunks)} "
            f"chunk(s) (0..{len(chunks) - 1})."
        )

    strategy = build_strategy(settings.ingest_strategy, settings.ingest_concurrency)
    target = chunks[chunk_index]
    for unit in strategy.units(chunks):
        if target in unit:
            idx_in_unit = unit.index(target)
            context = strategy.context_for(target, unit, idx_in_unit)
            return target, context
    return target, ""  # pragma: no cover — every chunk belongs to some unit


def main() -> None:
    global logger
    parser = argparse.ArgumentParser(
        description="Print the exact ingestion (metadata-tagging) prompt and/or "
                     "the resolved pipeline constants. No cloud calls.",
    )
    parser.add_argument("--pdf", type=Path, default=None,
                        help="Show the prompt for a real chunk of this PDF "
                             "(chunked the same way the real pipeline does). "
                             "Omit to see a placeholder-text prompt instead.")
    parser.add_argument("--chunk-index", type=int, default=0,
                        help="Which chunk of --pdf to use (default: 0).")
    parser.add_argument("--constants", action="store_true",
                        help="Also print resolved runtime constants (bucket, "
                             "datastore, engine, model, ...).")
    parser.add_argument("--constants-only", action="store_true",
                        help="Print only the constants; skip the prompt.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    if args.constants_only:
        _print_constants()
        return

    try:
        gen = MetadataGenerator(
            model_name=settings.gemini_model_metadata,
            schema_path=settings.metadata_schema_path,
        )
    except IngestionPromptConfigError as exc:
        parser.error(
            "config/ingestion_prompt.yaml is missing or invalid — it is the sole "
            "source of the ingestion prompt (no built-in fallback). Fix it (see "
            f"INGESTION_FOR_CLINICIANS.md) and re-run:\n{exc}"
        )

    if args.pdf is not None:
        if not args.pdf.exists():
            parser.error(f"File not found: {args.pdf}")
        chunk, context = _load_real_chunk(args.pdf, args.chunk_index)
        print(f"# Prompt for {args.pdf.name} chunk {args.chunk_index} "
              f"({chunk.chunk_id}), strategy '{settings.ingest_strategy}'\n")
    else:
        chunk, context = _build_placeholder_chunk(), ""
        print("# Prompt for a placeholder chunk (pass --pdf for a real one)\n")

    prompt = gen._build_extraction_prompt(chunk, context)
    print(prompt)
    print(f"\n# ({len(prompt)} characters)")

    if args.constants:
        print()
        _print_constants()


if __name__ == "__main__":
    main()
