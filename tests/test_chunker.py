"""
Tests for ingestion/chunker.py.

Validates structural splitting, semantic boundary detection, token budget
enforcement, chunk ID generation, and overlap behavior.

Run:
    pytest tests/test_chunker.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.chunker import ChunkerConfig, ContextAwareChunker
from models import Chunk, ExtractedDocument, Page


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_chunker() -> ContextAwareChunker:
    return ContextAwareChunker(ChunkerConfig())


@pytest.fixture
def small_max_chunker() -> ContextAwareChunker:
    """Chunker with a very small token budget for easy overflow testing."""
    return ContextAwareChunker(ChunkerConfig(max_tokens=50, min_tokens=10, overlap_tokens=5))


def make_extracted_doc(pages_text: list[str], doc_id: str = "abc123") -> ExtractedDocument:
    """Build an ExtractedDocument from a list of page text strings."""
    pages = [Page(page_num=i + 1, text=t) for i, t in enumerate(pages_text)]
    return ExtractedDocument(doc_id=doc_id, source_file="test.pdf", source_path="/tmp/test.pdf", pages=pages)


def _constant_embeddings(sentences: list[str]) -> np.ndarray:
    """Return identical embeddings for all sentences (cosine sim = 1.0 → no semantic splits)."""
    return np.ones((len(sentences), 4), dtype=float)


# ---------------------------------------------------------------------------
# ContextAwareChunker.chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_returns_list_of_chunks(self, default_chunker: ContextAwareChunker) -> None:
        """chunk() should return a non-empty list of Chunk objects."""
        doc = make_extracted_doc(["This is the first sentence. Here is the second sentence."])
        with patch.object(default_chunker, "_embed_sentences", side_effect=_constant_embeddings):
            chunks = default_chunker.chunk(doc)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_document_returns_empty_list(self, default_chunker: ContextAwareChunker) -> None:
        """chunk() should return [] for a document with no text on any page."""
        doc = make_extracted_doc(["", "   ", "\n"])
        with patch.object(default_chunker, "_embed_sentences", side_effect=_constant_embeddings):
            chunks = default_chunker.chunk(doc)
        assert chunks == []

    def test_chunk_ids_are_unique(self, default_chunker: ContextAwareChunker) -> None:
        """All chunk IDs in the output should be unique strings."""
        text = " ".join([f"Sentence number {i}." for i in range(30)])
        doc = make_extracted_doc([text])
        with patch.object(default_chunker, "_embed_sentences", side_effect=_constant_embeddings):
            chunks = default_chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_are_sequential(self, default_chunker: ContextAwareChunker) -> None:
        """chunk_index values should be 0, 1, 2, ... with no gaps."""
        text = " ".join([f"Sentence {i}." for i in range(20)])
        doc = make_extracted_doc([text])
        with patch.object(default_chunker, "_embed_sentences", side_effect=_constant_embeddings):
            chunks = default_chunker.chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_no_chunk_exceeds_max_tokens(self, small_max_chunker: ContextAwareChunker) -> None:
        """No returned chunk should have a token count above ChunkerConfig.max_tokens."""
        # Build text with many short sentences so the chunker must split
        text = " ".join([f"Word{i} Word{i}." for i in range(200)])
        doc = make_extracted_doc([text])
        with patch.object(small_max_chunker, "_embed_sentences", side_effect=_constant_embeddings):
            chunks = small_max_chunker.chunk(doc)
        for chunk in chunks:
            assert small_max_chunker._token_count(chunk.text) <= small_max_chunker._config.max_tokens


# ---------------------------------------------------------------------------
# ContextAwareChunker._structural_split
# ---------------------------------------------------------------------------


class TestStructuralSplit:
    def test_splits_on_chapter_header(self, default_chunker: ContextAwareChunker) -> None:
        """_structural_split should create a new section at 'Chapter N' headers."""
        doc = make_extracted_doc([
            "Some introductory text.",
            "Chapter 1\nFirst chapter content here.",
            "Chapter 2\nSecond chapter content here.",
        ])
        sections = default_chunker._structural_split(doc)
        section_names = [name for name, _ in sections]
        assert any("Chapter 1" in name for name in section_names)
        assert any("Chapter 2" in name for name in section_names)

    def test_preamble_section_for_pre_header_content(
        self, default_chunker: ContextAwareChunker
    ) -> None:
        """Pages before the first detected header should be grouped as 'preamble'."""
        doc = make_extracted_doc([
            "Title page content.",
            "Chapter 1\nFirst chapter.",
        ])
        sections = default_chunker._structural_split(doc)
        first_name, first_pages = sections[0]
        assert first_name == "preamble"

    def test_no_headers_returns_single_section(
        self, default_chunker: ContextAwareChunker
    ) -> None:
        """Documents with no detectable headers should produce one section."""
        doc = make_extracted_doc([
            "This is normal body text with no headers.",
            "Another page with more body text.",
        ])
        sections = default_chunker._structural_split(doc)
        assert len(sections) == 1
        assert sections[0][0] == "preamble"


# ---------------------------------------------------------------------------
# ContextAwareChunker._detect_section_headers
# ---------------------------------------------------------------------------


class TestDetectSectionHeaders:
    def test_detects_chapter_header(self, default_chunker: ContextAwareChunker) -> None:
        headers = default_chunker._detect_section_headers("Chapter 3\nSome body text.")
        assert "Chapter 3" in headers

    def test_detects_numbered_section(self, default_chunker: ContextAwareChunker) -> None:
        headers = default_chunker._detect_section_headers("3.1 Introduction\nBody text follows.")
        assert len(headers) > 0
        assert "3.1 Introduction" in headers

    def test_detects_all_caps_header(self, default_chunker: ContextAwareChunker) -> None:
        headers = default_chunker._detect_section_headers("METHODS\nWe recruited patients.")
        assert "METHODS" in headers

    def test_ignores_normal_prose(self, default_chunker: ContextAwareChunker) -> None:
        """Normal body text sentences should not be detected as headers."""
        text = (
            "This is a normal sentence that should not be a header. "
            "The patient was treated with medication. No headers here."
        )
        headers = default_chunker._detect_section_headers(text)
        assert headers == []


# ---------------------------------------------------------------------------
# ContextAwareChunker._make_chunk_id
# ---------------------------------------------------------------------------


class TestMakeChunkId:
    def test_format_is_correct(self, default_chunker: ContextAwareChunker) -> None:
        """chunk_id should be '{doc_id}_{index:05d}'."""
        result = default_chunker._make_chunk_id("abc123", 7)
        assert result == "abc123_00007"

    def test_zero_padded_to_five_digits(self, default_chunker: ContextAwareChunker) -> None:
        """Single-digit indices should be zero-padded to 5 digits."""
        result = default_chunker._make_chunk_id("deadbeef", 0)
        assert result == "deadbeef_00000"

        result = default_chunker._make_chunk_id("deadbeef", 99999)
        assert result == "deadbeef_99999"


# ---------------------------------------------------------------------------
# ContextAwareChunker._token_count
# ---------------------------------------------------------------------------


class TestTokenCount:
    def test_empty_string_is_zero(self, default_chunker: ContextAwareChunker) -> None:
        assert default_chunker._token_count("") == 0

    def test_single_word_is_one(self, default_chunker: ContextAwareChunker) -> None:
        assert default_chunker._token_count("hello") == 1

    def test_whitespace_only_is_zero(self, default_chunker: ContextAwareChunker) -> None:
        assert default_chunker._token_count("   \t\n   ") == 0


# ---------------------------------------------------------------------------
# ChunkerConfig.from_yaml
# ---------------------------------------------------------------------------


class TestChunkerConfigFromYaml:
    def test_loads_max_tokens_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "chunk_config.yaml"
        yaml_file.write_text("max_tokens: 256\nmin_tokens: 32\n")
        config = ChunkerConfig.from_yaml(yaml_file)
        assert config.max_tokens == 256
        assert config.min_tokens == 32

    def test_falls_back_to_defaults_for_missing_keys(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "chunk_config.yaml"
        yaml_file.write_text("max_tokens: 300\n")
        config = ChunkerConfig.from_yaml(yaml_file)
        assert config.max_tokens == 300
        assert config.overlap_tokens == ChunkerConfig().overlap_tokens

    def test_loads_header_patterns(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "chunk_config.yaml"
        yaml_file.write_text('header_patterns:\n  - "^Chapter"\n')
        config = ChunkerConfig.from_yaml(yaml_file)
        assert config.header_patterns == ["^Chapter"]
