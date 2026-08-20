"""
Corpus scanning for the corpus/ directory.

Primary usage — one-time scan (no daemon):
    scanner = CorpusScanner(corpus_dir, manifest_path, pipeline_fn)
    scanner.scan()   # single pass over corpus/; returns when done

During deployment: ``scripts/batch_ingest.py`` uses this path. Intended to run
as a batch job, CI step, or cron-scheduled container.

Tracks processed files in a JSON manifest (basename → doc_id) so re-runs
process only new files. The manifest is written atomically on every update.

Production note: for deployments requiring sub-minute ingestion latency,
replace the scheduled scan with a GCS Eventarc trigger (Cloud Functions).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class CorpusScanner:
    """
    Orchestrates corpus scanning and coordinates per-document ingestion.

    Primary lifecycle (one-time scan):
        scanner = CorpusScanner(corpus_dir, manifest_path, pipeline_fn)
        scanner.scan()   # processes all unprocessed PDFs, then returns
    """

    def __init__(
        self,
        corpus_dir: Path,
        manifest_path: Path,
        pipeline_callback: Callable[[Path], None],
    ) -> None:
        """
        Args:
            corpus_dir: Directory to scan. Must exist before calling scan().
            manifest_path: Path to the JSON manifest (created if absent).
            pipeline_callback: Called with the PDF path to run the full ingestion
                               pipeline (extract → chunk → metadata → upload → index).
        """
        self._corpus_dir = corpus_dir
        self._manifest_path = manifest_path
        self._pipeline_callback = pipeline_callback
        self._manifest: dict[str, str] = {}  # basename -> doc_id

    def scan(self) -> None:
        """Process any PDFs in corpus_dir not yet recorded in the manifest.

        Primary ingestion entry point. Scans corpus_dir once, runs the pipeline
        for each unprocessed file in alphabetical order, and returns. Manifest is
        updated after each successful file so partial runs are resumable.
        """
        self._load_manifest()
        pdf_paths = sorted(self._corpus_dir.glob("*.pdf"))
        logger.info("Scanning %s: found %d PDF(s)", self._corpus_dir, len(pdf_paths))

        for path in pdf_paths:
            if not self.is_processed(path):
                self.process_pdf(path)
            else:
                logger.debug("Skipping already-processed: %s", path.name)

    def process_pdf(self, path: Path) -> None:
        """Guard-check then invoke the ingestion pipeline for one PDF.

        Skips the file if already in the manifest. Marks it processed after the
        pipeline_callback returns without raising.

        Args:
            path: Absolute path to the PDF.
        """
        if self.is_processed(path):
            logger.debug("Skipping already-processed: %s", path.name)
            return

        logger.info("Processing: %s", path.name)
        self._pipeline_callback(path)

        doc_id = self._compute_file_hash(path)
        self.mark_processed(path, doc_id)

    def is_processed(self, path: Path) -> bool:
        """Return True if this file's basename exists in the in-memory manifest.

        Args:
            path: Path to the PDF file.

        Returns:
            True if already ingested.
        """
        return path.name in self._manifest

    def mark_processed(self, path: Path, doc_id: str) -> None:
        """Record a file in the manifest and persist to disk.

        Args:
            path: Path to the PDF file.
            doc_id: Content-hash ID assigned by GCSUploader.compute_doc_id().
        """
        self._manifest[path.name] = doc_id
        self._save_manifest()
        logger.debug("Manifest updated: %s -> %s", path.name, doc_id[:16])

    def _load_manifest(self) -> None:
        """Read manifest JSON into self._manifest; create empty if missing."""
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                self._manifest = json.load(f)
            logger.debug("Loaded manifest with %d entries", len(self._manifest))
        else:
            self._manifest = {}

    def _save_manifest(self) -> None:
        """Write self._manifest to disk atomically (temp file + rename)."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._manifest_path.parent), suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._manifest, f, indent=2)
            Path(tmp_path).rename(self._manifest_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _compute_file_hash(self, path: Path) -> str:
        """Return the SHA-256 hex digest of the file at path."""
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
