"""
Tests for the checkpoint-granularity short-circuit in scripts/batch_ingest.py:
_checkpoint_covers_all() and ingest_file()'s already-uploaded-to-GCS fast path,
which skips extraction/chunking/metadata generation entirely and starts the
Vertex AI Search import LRO directly against the existing chunks JSONL.

No cloud calls — uploader/indexer/extractor/chunker/metadata_gen are all
stubbed. extract()/chunk()/generate() being called at all is treated as a test
failure on the shortcut path, since the whole point is that they're skipped.

Run:
    pytest tests/test_batch_ingest.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.processing_strategy import IndependentStrategy
from models import Chunk, ImportResult


def _chunk(idx: int, doc_id: str = "docid123") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_{idx:05d}",
        doc_id=doc_id,
        text="body",
        page_start=idx + 1,
        page_end=idx + 1,
        chunk_index=idx,
        parent_section="",
    )


# ---------------------------------------------------------------------------
# _checkpoint_covers_all
# ---------------------------------------------------------------------------


class TestCheckpointCoversAll:
    def test_true_when_every_chunk_cached(self) -> None:
        from scripts.batch_ingest import _checkpoint_covers_all

        chunks = [_chunk(i) for i in range(3)]
        cache = {c.chunk_id: {} for c in chunks}
        assert _checkpoint_covers_all(chunks, cache) is True

    def test_false_when_partially_cached(self) -> None:
        from scripts.batch_ingest import _checkpoint_covers_all

        chunks = [_chunk(i) for i in range(3)]
        cache = {chunks[0].chunk_id: {}}
        assert _checkpoint_covers_all(chunks, cache) is False

    def test_false_when_no_chunks(self) -> None:
        from scripts.batch_ingest import _checkpoint_covers_all

        assert _checkpoint_covers_all([], {}) is False


# ---------------------------------------------------------------------------
# ingest_file() — GCS-already-uploaded short-circuit
# ---------------------------------------------------------------------------


class _RefusingExtractor:
    def extract(self, path):
        raise AssertionError("extract() must not run when chunks are already uploaded")


class _RefusingChunker:
    class _Cfg:
        skip_doc_types: list = []

    _config = _Cfg()

    def chunk(self, doc):
        raise AssertionError("chunk() must not run when chunks are already uploaded")


class _RefusingMetadataGen:
    def generate(self, chunk, context=""):
        raise AssertionError("generate() must not run when chunks are already uploaded")

    def verify_model_available(self) -> None:
        pass


class _FakeUploader:
    def __init__(self, already_uploaded: bool, n_chunks: int = 5) -> None:
        self.already_uploaded = already_uploaded
        self.n_chunks = n_chunks
        self.upload_pdf_called = False
        self.upload_chunks_called = False

    def compute_doc_id(self, path: Path) -> str:
        return "docid123"

    def is_already_uploaded(self, doc_id: str) -> bool:
        return self.already_uploaded

    def gcs_pdf_uri(self, doc_id: str) -> str:
        return f"gs://bucket/pdfs/{doc_id}.pdf"

    def gcs_chunks_uri(self, doc_id: str) -> str:
        return f"gs://bucket/chunks/{doc_id}.jsonl"

    def count_uploaded_chunks(self, doc_id: str) -> int:
        return self.n_chunks

    def upload_pdf(self, local_path: Path, doc_id: str) -> str:
        self.upload_pdf_called = True
        return self.gcs_pdf_uri(doc_id)

    def upload_chunks(self, chunks, doc_id: str) -> str:
        self.upload_chunks_called = True
        return self.gcs_chunks_uri(doc_id)


class _FakeIndexer:
    def __init__(self) -> None:
        self.import_calls: list[tuple[str, str]] = []

    def import_chunks(self, gcs_jsonl_uri: str, doc_id: str) -> str:
        self.import_calls.append((gcs_jsonl_uri, doc_id))
        return "op-name"

    def wait_for_import(self, operation_name: str) -> ImportResult:
        return ImportResult(
            operation_name=operation_name, success_count=5, failure_count=0, completed=True,
        )


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")
    return path


class TestIngestFileAlreadyUploadedShortcut:
    def test_skips_pipeline_and_imports_directly(self, fake_pdf: Path) -> None:
        from scripts.batch_ingest import ingest_file

        uploader = _FakeUploader(already_uploaded=True, n_chunks=7)
        indexer = _FakeIndexer()

        result = ingest_file(
            fake_pdf, _RefusingExtractor(), _RefusingChunker(), _RefusingMetadataGen(),
            uploader, indexer, IndependentStrategy(1), dry_run=False, force=False,
        )

        assert result["resumed_from_upload"] is True
        assert result["n_chunks"] == 7
        assert result["n_metadata_failures"] == 0
        assert result["import_success"] == 5
        assert result["import_failures"] == 0
        assert indexer.import_calls == [("gs://bucket/chunks/docid123.jsonl", "docid123")]
        assert not uploader.upload_pdf_called
        assert not uploader.upload_chunks_called

    def test_force_bypasses_shortcut(self, fake_pdf: Path) -> None:
        from scripts.batch_ingest import ingest_file

        uploader = _FakeUploader(already_uploaded=True)
        indexer = _FakeIndexer()

        with pytest.raises(AssertionError, match="extract"):
            ingest_file(
                fake_pdf, _RefusingExtractor(), _RefusingChunker(), _RefusingMetadataGen(),
                uploader, indexer, IndependentStrategy(1), dry_run=False, force=True,
            )

    def test_dry_run_bypasses_shortcut(self, fake_pdf: Path) -> None:
        from scripts.batch_ingest import ingest_file

        uploader = _FakeUploader(already_uploaded=True)
        indexer = _FakeIndexer()

        with pytest.raises(AssertionError, match="extract"):
            ingest_file(
                fake_pdf, _RefusingExtractor(), _RefusingChunker(), _RefusingMetadataGen(),
                uploader, indexer, IndependentStrategy(1), dry_run=True, force=False,
            )

    def test_not_yet_uploaded_runs_full_pipeline(self, fake_pdf: Path) -> None:
        """Sanity check: when nothing is uploaded yet, the shortcut must NOT
        trigger — extract() runs (and, via the refusing fake, raises), proving
        is_already_uploaded()==False takes the normal path."""
        from scripts.batch_ingest import ingest_file

        uploader = _FakeUploader(already_uploaded=False)
        indexer = _FakeIndexer()

        with pytest.raises(AssertionError, match="extract"):
            ingest_file(
                fake_pdf, _RefusingExtractor(), _RefusingChunker(), _RefusingMetadataGen(),
                uploader, indexer, IndependentStrategy(1), dry_run=False, force=False,
            )
