"""
Batch ingestion script — process all PDFs in corpus/ in one run.

Use this to populate a fresh DataStore from an existing corpus directory,
or to re-ingest specific files after a config change.

The script:
  1. Scans corpus/ for PDF files
  2. Skips files already in the ingestion manifest (unless --force)
  3. Runs the full pipeline per file: extract → chunk → metadata → upload → index
  4. Waits for each Vertex AI Search import operation before moving to the next file
  5. Writes a summary report on completion

Usage:
    python scripts/batch_ingest.py [--dry-run] [--force] [--domain DOMAIN] [--file FILE]

Examples:
    # Ingest all new PDFs in corpus/
    python scripts/batch_ingest.py

    # Re-ingest a specific file, overwriting existing index entries
    python scripts/batch_ingest.py --file corpus/cardiology_textbook.pdf --force

    # Preview what would be ingested without running
    python scripts/batch_ingest.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Imported first so its warnings filter is registered before the google.generativeai
# import (below, via the ingestion package) prints its end-of-life FutureWarning.
from scripts._gcp_logging import log_api_error, setup_logging

from config.settings import settings
from ingestion.extractor import PDFExtractor
from ingestion.chunker import ContextAwareChunker
from ingestion.metadata_gen import MetadataGenerator
from ingestion.uploader import GCSUploader
from ingestion.indexer import VertexSearchIndexer
from ingestion.processing_strategy import ProcessingStrategy, build_strategy
from models import ChunkMetadata

logger = logging.getLogger(__name__)


def _generate_all_metadata(
    chunks: list,
    strategy: ProcessingStrategy,
    metadata_gen: MetadataGenerator,
    source_file: str,
    checkpoint_path: Path,
) -> tuple[int, int]:
    """Tag every chunk, honoring the processing strategy, checkpoint, and concurrency.

    Units run concurrently (up to ``strategy.max_concurrency``); chunks within a
    unit run in order so each can see its predecessors' context. Each chunk's
    metadata is written to the checkpoint as it is produced (crash-safe), and
    chunks already in the checkpoint are reused instead of re-tagged. Progress is
    logged at the default level.

    Returns:
        (n_failures, n_reused).
    """
    cache = _load_metadata_checkpoint(checkpoint_path)
    if cache:
        logger.info("  Resuming from checkpoint %s: %d chunk(s) already tagged.",
                    checkpoint_path, len(cache))
    units = strategy.units(chunks)
    n = len(chunks)
    step = max(1, n // 20)  # progress roughly every 5%
    lock = threading.Lock()
    state = {"done": 0, "reused": 0, "failures": 0}
    logger.info("  Tagging %d chunk(s) via '%s' strategy: %d unit(s), concurrency %d...",
                n, strategy.name, len(units), strategy.max_concurrency)

    def process_unit(unit: list) -> None:
        for idx, chunk in enumerate(unit):
            if chunk.chunk_id in cache:
                chunk.metadata = ChunkMetadata(**cache[chunk.chunk_id])
                with lock:
                    state["reused"] += 1
            else:
                context = strategy.context_for(chunk, unit, idx)
                try:
                    metadata = metadata_gen.generate(chunk, context)
                    metadata.source_file = source_file
                    chunk.metadata = metadata
                    with lock:
                        _append_metadata_checkpoint(checkpoint_path, chunk)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("  Metadata failed for %s: %s", chunk.chunk_id, exc)
                    with lock:
                        state["failures"] += 1
            with lock:
                state["done"] += 1
                done = state["done"]
            if done % step == 0 or done == n:
                logger.info("  ...metadata %d/%d (%d%%)%s", done, n, int(100 * done / n),
                            f"  [{state['reused']} reused]" if state["reused"] else "")

    with ThreadPoolExecutor(max_workers=max(1, strategy.max_concurrency)) as pool:
        futures = [pool.submit(process_unit, u) for u in units]
        for fut in as_completed(futures):
            fut.result()  # surface any unexpected worker exception

    return state["failures"], state["reused"]


def _load_metadata_checkpoint(path: Path) -> dict[str, dict]:
    """Return {chunk_id: metadata_dict} from a checkpoint JSONL, or {} if absent.

    Lets a re-run skip chunks already tagged in a previous (possibly crashed) run,
    so the expensive Gemini calls are not repeated. Delete the file to force a
    full re-tag (e.g. after fixing credentials that caused fallback tagging).
    """
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("chunk_id") and rec.get("metadata") is not None:
                cache[rec["chunk_id"]] = rec["metadata"]
    return cache


def _append_metadata_checkpoint(path: Path, chunk) -> None:
    """Append one chunk's metadata to the checkpoint JSONL (crash-safe, incremental)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "chunk_id": chunk.chunk_id,
        "metadata": chunk.metadata.model_dump() if chunk.metadata else None,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ingest_file(
    pdf_path: Path,
    extractor: PDFExtractor,
    chunker: ContextAwareChunker,
    metadata_gen: MetadataGenerator,
    uploader: GCSUploader,
    indexer: VertexSearchIndexer,
    strategy: ProcessingStrategy,
    dry_run: bool = False,
) -> dict:
    """Run the full ingestion pipeline for a single PDF.

    Steps in order:
      1. Compute doc_id (content hash)
      2. Extract text and structure from the PDF
      3. Chunk the extracted document
      4. Generate metadata for each chunk via Gemini
      5. Filter out skip_doc_types chunks (e.g. front_matter)
      6. Upload PDF and chunk JSONL to GCS
      7. Import chunks into Vertex AI Search
      8. Wait for import LRO completion

    Args:
        pdf_path: Absolute path to the PDF file.
        extractor: Initialized PDFExtractor.
        chunker: Initialized ContextAwareChunker.
        metadata_gen: Initialized MetadataGenerator.
        uploader: Initialized GCSUploader.
        indexer: Initialized VertexSearchIndexer.
        strategy: Chunk-processing strategy (units + context + concurrency).
        dry_run: If True, run through extract/chunk/metadata but skip GCS upload
                 and Vertex AI import.

    Returns:
        Summary dict with keys: file, doc_id, n_chunks, n_metadata_failures,
        gcs_pdf_uri, gcs_chunks_uri, import_success, import_failures, elapsed_s.
    """
    t0 = time.time()

    # 1. Compute doc_id
    doc_id = uploader.compute_doc_id(pdf_path)
    logger.info("  doc_id: %s...", doc_id[:16])

    # 2. Extract
    logger.info("  Extracting text from %s...", pdf_path.name)
    doc = extractor.extract(pdf_path)
    logger.info("  Extracted %d page(s)", len(doc.pages))

    # 3. Chunk
    logger.info("  Chunking...")
    chunks = chunker.chunk(doc)
    logger.info("  Produced %d chunk(s)", len(chunks))

    # 4. Generate metadata (strategy-driven, concurrent, checkpoint-aware/resumable)
    checkpoint_path = settings.checkpoint_dir / f"{doc_id}.jsonl"
    n_failures, _ = _generate_all_metadata(
        chunks, strategy, metadata_gen, doc.source_file, checkpoint_path
    )

    # 5. Filter skip_doc_types
    skip_types = set(chunker._config.skip_doc_types)
    if skip_types:
        before = len(chunks)
        chunks = [
            c for c in chunks
            if c.metadata is None or c.metadata.doc_type not in skip_types
        ]
        skipped = before - len(chunks)
        if skipped:
            logger.info("  Filtered %d chunk(s) with skip_doc_types", skipped)

    gcs_pdf_uri = ""
    gcs_chunks_uri = ""
    import_success = 0
    import_failures = 0

    if not dry_run and chunks:
        # 6. Upload to GCS
        logger.info("  Uploading PDF and %d chunk(s) to GCS...", len(chunks))
        gcs_pdf_uri = uploader.upload_pdf(pdf_path, doc_id)
        gcs_chunks_uri = uploader.upload_chunks(chunks, doc_id)

        # 7. Import into Vertex AI Search
        logger.info("  Starting Vertex AI Search import...")
        op_name = indexer.import_chunks(gcs_chunks_uri, doc_id)

        # 8. Wait for LRO completion
        logger.info("  Waiting for import LRO...")
        result = indexer.wait_for_import(op_name)
        import_success = result.success_count
        import_failures = result.failure_count
        if result.errors:
            for err in result.errors[:5]:
                logger.warning("  Import error sample: %s", err)

    elif dry_run:
        logger.info("  [DRY RUN] Skipping upload and index steps.")

    elapsed = time.time() - t0
    return {
        "file": pdf_path.name,
        "doc_id": doc_id,
        "n_chunks": len(chunks),
        "n_metadata_failures": n_failures,
        "gcs_pdf_uri": gcs_pdf_uri,
        "gcs_chunks_uri": gcs_chunks_uri,
        "import_success": import_success,
        "import_failures": import_failures,
        "elapsed_s": round(elapsed, 1),
    }


def scan_corpus(corpus_dir: Path, manifest_path: Path, force: bool = False) -> list[Path]:
    """Return PDF paths in corpus_dir that have not yet been ingested.

    Args:
        corpus_dir: Directory to scan for PDFs.
        manifest_path: Path to the ingestion manifest JSON.
        force: If True, return all PDFs regardless of manifest state.

    Returns:
        List of PDF paths to ingest, sorted alphabetically.
    """
    all_pdfs = sorted(corpus_dir.glob("*.pdf"))

    if force or not manifest_path.exists():
        return all_pdfs

    with open(manifest_path) as f:
        manifest: dict[str, str] = json.load(f)

    return [p for p in all_pdfs if p.name not in manifest]


def _update_manifest(manifest_path: Path, file_name: str, doc_id: str) -> None:
    """Append a successfully ingested file to the manifest."""
    import os
    import tempfile

    manifest: dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest[file_name] = doc_id

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(manifest_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).rename(manifest_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def print_summary(results: list[dict]) -> None:
    """Print a formatted ingestion summary table to stdout.

    Args:
        results: List of result dicts from ingest_file().
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Ingestion Summary  ({len(results)} file(s))")
    print(sep)
    total_chunks = 0
    total_failures = 0
    for r in results:
        status = "WARN" if r.get("import_failures", 0) > 0 else "OK  "
        print(f"[{status}] {r['file']}")
        print(f"       doc_id  : {r['doc_id'][:24]}...")
        print(f"       chunks  : {r['n_chunks']}  (metadata failures: {r['n_metadata_failures']})")
        if r["gcs_pdf_uri"]:
            print(f"       pdf     : {r['gcs_pdf_uri']}")
        if r["gcs_chunks_uri"]:
            print(f"       chunks  : {r['gcs_chunks_uri']}")
        print(f"       import  : {r['import_success']} ok / {r['import_failures']} failed")
        print(f"       elapsed : {r['elapsed_s']}s")
        total_chunks += r["n_chunks"]
        total_failures += r["import_failures"]
    print(sep)
    print(f"Total: {total_chunks} chunks across {len(results)} file(s)")
    if total_failures:
        print(f"WARNING: {total_failures} import failure(s) — check logs above")
    print(sep)


def main() -> None:
    """Parse arguments and run batch ingestion."""
    parser = argparse.ArgumentParser(description="Batch ingest PDFs into the RAG corpus.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk without uploading or indexing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files already in the manifest.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Ingest a single specific PDF file instead of scanning corpus/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging, including Google/gRPC internals — use to debug API failures.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Chunk-processing strategy: 'chapter' (default) or 'independent'. "
             "Overrides INGEST_STRATEGY.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max chapters/chunks tagged concurrently. Overrides INGEST_CONCURRENCY.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    strategy = build_strategy(
        args.strategy or settings.ingest_strategy,
        args.concurrency if args.concurrency is not None else settings.ingest_concurrency,
    )
    logger.info("Processing strategy: %s (concurrency %d)", strategy.name, strategy.max_concurrency)

    # Fail fast with an actionable message on missing config or bad credentials
    # rather than a raw traceback deep in the first API call.
    try:
        settings.validate_all()
    except Exception as exc:  # missing/empty env vars
        logger.error("Configuration problem — check your .env:\n%s", exc)
        sys.exit(2)

    logger.info("Active metadata schema: %s", settings.metadata_schema_path)

    try:
        extractor = PDFExtractor(
            use_document_ai=settings.use_document_ai,
            document_ai_processor_id=settings.document_ai_processor_id or "",
            project_id=settings.gcp_project_id,
        )
        chunker = ContextAwareChunker()
        metadata_gen = MetadataGenerator(
            model_name=settings.gemini_model_metadata,
            schema_path=settings.metadata_schema_path,
        )
        uploader = GCSUploader(
            bucket_name=settings.gcs_bucket_name,
            project_id=settings.gcp_project_id,
        )
        indexer = VertexSearchIndexer(
            project_id=settings.gcp_project_id,
            location=settings.discovery_engine_location,      # 'global'/'us'/'eu', not the raw compute region
            datastore_id=settings.vertex_search_datastore_id,
            api_endpoint=settings.discovery_engine_endpoint,  # regional endpoint (fixes the import crash)
        )
    except Exception as exc:  # noqa: BLE001 — clientinit/credential errors
        log_api_error(logger, exc, "initializing the pipeline clients")
        sys.exit(1)

    if args.file:
        pdf_paths = [args.file]
    else:
        pdf_paths = scan_corpus(settings.corpus_dir, settings.manifest_path, args.force)

    logger.info("Found %d PDF(s) to ingest.", len(pdf_paths))
    if args.dry_run:
        logger.info("DRY RUN: upload and index steps will be skipped.")

    results = []
    for pdf_path in pdf_paths:
        logger.info("Processing: %s", pdf_path.name)
        try:
            result = ingest_file(
                pdf_path, extractor, chunker, metadata_gen, uploader, indexer,
                strategy, dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001 — one bad file/API call shouldn't kill the batch
            log_api_error(logger, exc, f"ingesting {pdf_path.name}")
            continue
        results.append(result)

        # Update manifest after each successful (non-dry-run) ingestion
        if not args.dry_run and result["import_failures"] == 0:
            _update_manifest(
                settings.manifest_path, result["file"], result["doc_id"]
            )

    print_summary(results)


if __name__ == "__main__":
    main()
