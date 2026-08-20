"""
Tests for ingestion/processing_strategy.py and the concurrent, checkpoint-aware
tagging helper in scripts/batch_ingest.py. No cloud/Gemini calls — the metadata
generator is stubbed.

Run:
    pytest tests/test_processing_strategy.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.processing_strategy import (
    ChapterContextStrategy,
    IndependentStrategy,
    build_strategy,
)
from models import Chunk, ChunkMetadata


def _chunk(idx: int, section: str, text: str = "body text") -> Chunk:
    return Chunk(
        chunk_id=f"doc_{idx:05d}",
        doc_id="doc",
        text=text,
        page_start=idx + 1,
        page_end=idx + 1,
        chunk_index=idx,
        parent_section=section,
    )


class TestIndependentStrategy:
    def test_one_unit_per_chunk(self) -> None:
        chunks = [_chunk(i, "A") for i in range(3)]
        units = IndependentStrategy().units(chunks)
        assert [len(u) for u in units] == [1, 1, 1]

    def test_no_context(self) -> None:
        chunks = [_chunk(0, "A")]
        s = IndependentStrategy()
        assert s.context_for(chunks[0], chunks, 0) == ""


class TestChapterContextStrategy:
    def test_groups_consecutive_sections(self) -> None:
        chunks = [_chunk(0, "Ch1"), _chunk(1, "Ch1"), _chunk(2, "Ch2"), _chunk(3, "Ch1")]
        units = ChapterContextStrategy().units(chunks)
        # Ch1(2) | Ch2(1) | Ch1(1) — consecutive runs, not global grouping
        assert [len(u) for u in units] == [2, 1, 1]

    def test_context_has_heading_and_preceding_text(self) -> None:
        chunks = [_chunk(0, "Chapter 4", "first para"), _chunk(1, "Chapter 4", "second para")]
        s = ChapterContextStrategy()
        ctx0 = s.context_for(chunks[0], chunks, 0)
        ctx1 = s.context_for(chunks[1], chunks, 1)
        assert "Chapter 4" in ctx0 and "first para" not in ctx0   # nothing precedes chunk 0
        assert "Chapter 4" in ctx1 and "first para" in ctx1        # chunk 0 precedes chunk 1

    def test_context_budget_truncates_to_most_recent(self) -> None:
        big = "x" * 5000
        chunks = [_chunk(0, "C", big), _chunk(1, "C", "y")]
        s = ChapterContextStrategy(context_char_budget=100)
        ctx1 = s.context_for(chunks[1], chunks, 1)
        # heading line + at most budget chars of preceding text
        assert len(ctx1) <= 100 + len("Chapter/section: C") + 2


class TestBuildStrategy:
    def test_known(self) -> None:
        assert build_strategy("chapter", 4).name == "chapter"
        assert build_strategy("independent", 8).name == "independent"

    def test_concurrency_applied(self) -> None:
        assert build_strategy("chapter", 3).max_concurrency == 3

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            build_strategy("bogus", 1)


class _StubGen:
    """Records generate() calls and returns deterministic metadata."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, chunk: Chunk, context: str = "") -> ChunkMetadata:
        self.calls.append((chunk.chunk_id, context))
        return ChunkMetadata(doc_id="a" * 64, keywords=[chunk.chunk_id])


class TestGenerateAllMetadata:
    def test_tags_all_and_writes_checkpoint(self, tmp_path: Path) -> None:
        import scripts.batch_ingest as bi

        chunks = [_chunk(i, "Ch1") for i in range(4)]
        gen = _StubGen()
        ckpt = tmp_path / "doc.jsonl"
        failures, reused = bi._generate_all_metadata(
            chunks, ChapterContextStrategy(max_concurrency=1), gen, "f.pdf", ckpt
        )
        assert failures == 0 and reused == 0
        assert all(c.metadata is not None for c in chunks)
        assert len(gen.calls) == 4
        # checkpoint has one line per chunk
        assert ckpt.exists() and len(ckpt.read_text().strip().splitlines()) == 4

    def test_resume_skips_cached(self, tmp_path: Path) -> None:
        import scripts.batch_ingest as bi

        chunks = [_chunk(i, "Ch1") for i in range(4)]
        gen1 = _StubGen()
        ckpt = tmp_path / "doc.jsonl"
        bi._generate_all_metadata(chunks, IndependentStrategy(1), gen1, "f.pdf", ckpt)

        # Second run over the same chunks: everything should be reused, no new calls.
        fresh = [_chunk(i, "Ch1") for i in range(4)]
        gen2 = _StubGen()
        failures, reused = bi._generate_all_metadata(
            fresh, IndependentStrategy(1), gen2, "f.pdf", ckpt
        )
        assert reused == 4 and failures == 0
        assert gen2.calls == []  # nothing re-tagged
