"""
DataStore purge utility — wipe all documents and optionally re-ingest.

USE WITH CAUTION. This script deletes all indexed documents. Use it when:
  - The metadata schema has changed and all chunks must be re-imported
  - Chunking parameters changed and the whole corpus must be re-chunked
  - Test data needs to be cleared from a development DataStore

The script requires explicit --confirm to prevent accidental data loss.
It also supports a --doc-id flag to delete a single document's chunks
without wiping the entire DataStore.

Usage:
    # Wipe entire DataStore (DESTRUCTIVE)
    python scripts/purge_datastore.py --confirm

    # Delete a single document's chunks
    python scripts/purge_datastore.py --doc-id <sha256_doc_id> --confirm

    # Wipe and re-ingest immediately
    python scripts/purge_datastore.py --confirm --reingest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from ingestion.indexer import VertexSearchIndexer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def purge_all(indexer: VertexSearchIndexer, dry_run: bool = False) -> int:
    """Delete all documents from the DataStore.

    Lists all document IDs via indexer.list_documents() and calls
    indexer.delete_document() for each. Logs progress every 100 deletions.

    Args:
        indexer: Initialized VertexSearchIndexer.
        dry_run: If True, log what would be deleted without deleting.

    Returns:
        Number of documents deleted (or that would be deleted in dry-run).
    """
    logger.info("Listing all documents in DataStore...")
    doc_ids = indexer.list_documents()
    n = len(doc_ids)
    logger.info("Found %d document(s).", n)

    if dry_run:
        logger.info("[DRY RUN] Would delete %d document(s).", n)
        return n

    for i, doc_id in enumerate(doc_ids):
        indexer.delete_document(doc_id)
        if (i + 1) % 100 == 0:
            logger.info("Deleted %d/%d document(s)...", i + 1, n)

    logger.info("Deleted all %d document(s).", n)
    return n


def purge_document(
    doc_id: str, indexer: VertexSearchIndexer, dry_run: bool = False
) -> int:
    """Delete all chunks belonging to a single source document.

    Chunk IDs follow the pattern "{doc_id}_{index:05d}". This function
    lists all DataStore documents with IDs matching the prefix and deletes them.

    Args:
        doc_id: The source document SHA-256 ID.
        indexer: Initialized VertexSearchIndexer.
        dry_run: If True, log without deleting.

    Returns:
        Number of chunks deleted.
    """
    prefix = f"{doc_id}_"
    all_ids = indexer.list_documents()
    matching = [d for d in all_ids if d.startswith(prefix)]
    n = len(matching)

    if dry_run:
        logger.info("[DRY RUN] Would delete %d chunk(s) for doc %s.", n, doc_id[:16])
        return n

    for chunk_id in matching:
        indexer.delete_document(chunk_id)

    logger.info("Deleted %d chunk(s) for doc %s.", n, doc_id[:16])
    return n


def reset_manifest(manifest_path: Path, doc_id: str | None = None) -> None:
    """Remove entries from the ingestion manifest after a purge.

    If doc_id is None, clears the entire manifest.
    If doc_id is provided, removes only the entry for that document.

    Args:
        manifest_path: Path to the ingestion manifest JSON.
        doc_id: Optional doc ID to remove; clears all if None.
    """
    if not manifest_path.exists():
        return

    if doc_id is None:
        manifest_path.write_text("{}\n")
        logger.info("Manifest cleared: %s", manifest_path)
        return

    with open(manifest_path) as f:
        manifest: dict[str, str] = json.load(f)

    # Remove entries where value matches the doc_id
    before = len(manifest)
    manifest = {k: v for k, v in manifest.items() if v != doc_id}
    removed = before - len(manifest)

    fd, tmp = tempfile.mkstemp(dir=str(manifest_path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        Path(tmp).rename(manifest_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

    logger.info("Removed %d manifest entry/entries for doc %s.", removed, doc_id[:16])


def main() -> None:
    """Parse arguments and run purge."""
    parser = argparse.ArgumentParser(
        description="Purge documents from the Vertex AI Search DataStore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="Required flag to confirm destructive operation.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Delete only chunks for this specific document SHA-256 ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be deleted without performing deletions.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Run batch_ingest.py after purge completes (full corpus re-ingest).",
    )
    args = parser.parse_args()

    settings.validate_all()

    indexer = VertexSearchIndexer(
        project_id=settings.gcp_project_id,
        location=settings.discovery_engine_location,
        datastore_id=settings.vertex_search_datastore_id,
        api_endpoint=settings.discovery_engine_endpoint,
    )

    if args.doc_id:
        logger.info("Purging document: %s", args.doc_id)
        n = purge_document(args.doc_id, indexer, dry_run=args.dry_run)
        logger.info("Deleted %d chunk(s) for doc %s.", n, args.doc_id)
        if not args.dry_run:
            reset_manifest(settings.manifest_path, doc_id=args.doc_id)
    else:
        logger.warning(
            "Purging ALL documents from DataStore: %s",
            settings.vertex_search_datastore_id,
        )
        n = purge_all(indexer, dry_run=args.dry_run)
        logger.info("Deleted %d documents total.", n)
        if not args.dry_run:
            reset_manifest(settings.manifest_path)

    if args.reingest and not args.dry_run:
        logger.info("Launching batch re-ingest...")
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/batch_ingest.py", "--force"],
            check=True,
        )


if __name__ == "__main__":
    main()
