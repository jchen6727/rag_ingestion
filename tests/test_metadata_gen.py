"""
Tests for ingestion/metadata_gen.py.

Validates the Gemini call path, JSON response validation, type coercion,
and fallback behavior. All Gemini API calls are mocked — no live API keys
are required to run these tests.

Run:
    pytest tests/test_metadata_gen.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.metadata_gen import GeminiExtractionError, MetadataGenerator
from models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_REAL_SCHEMA_PATH = (
    Path(__file__).parent.parent / "config" / "metadata_schema.json"
)


@pytest.fixture
def generator() -> MetadataGenerator:
    """MetadataGenerator pointed at the authoritative psychotherapy schema.

    Coercion is schema-driven (config/schema_loader.py), so the tests exercise
    the real controlled vocabulary rather than a placeholder schema.
    """
    return MetadataGenerator(
        model_name="gemini-1.5-flash", schema_path=_REAL_SCHEMA_PATH
    )


def make_chunk(text: str = "Sample clinical text.", doc_id: str = "abc123") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_00000",
        doc_id=doc_id,
        text=text,
        page_start=1,
        page_end=1,
        chunk_index=0,
        parent_section="Introduction",
    )


def _valid_gemini_response(chunk: Chunk) -> dict:
    """Return a minimal valid Gemini response dict for the given chunk."""
    return {
        "doc_id": chunk.doc_id,
        "source_file": "test.pdf",
        "domain": "cognitive_behavioral",
        "doc_type": "treatment_manual",
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "title": "Test Title",
        "keywords": ["keyword1"],
    }


# ---------------------------------------------------------------------------
# MetadataGenerator.generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_chunk_metadata(self, generator: MetadataGenerator) -> None:
        """generate() should return a ChunkMetadata instance."""
        chunk = make_chunk()
        with patch.object(generator, "_call_gemini", return_value=_valid_gemini_response(chunk)):
            result = generator.generate(chunk)
        assert isinstance(result, ChunkMetadata)

    def test_populates_doc_id_from_chunk(self, generator: MetadataGenerator) -> None:
        """Returned metadata.doc_id should match chunk.doc_id."""
        chunk = make_chunk(doc_id="deadbeef" * 8)
        raw = _valid_gemini_response(chunk)
        raw["doc_id"] = "wrong_id"  # should be overridden by validate_and_coerce
        with patch.object(generator, "_call_gemini", return_value=raw):
            result = generator.generate(chunk)
        assert result.doc_id == chunk.doc_id

    def test_falls_back_when_gemini_raises(self, generator: MetadataGenerator) -> None:
        """generate() should return fallback metadata rather than propagating Gemini errors."""
        chunk = make_chunk()
        with patch.object(generator, "_call_gemini", side_effect=GeminiExtractionError("API error")):
            result = generator.generate(chunk)
        assert isinstance(result, ChunkMetadata)
        assert result.doc_id == chunk.doc_id

    def test_falls_back_on_invalid_json(self, generator: MetadataGenerator) -> None:
        """generate() should use _fallback_extraction if Gemini returns malformed JSON."""
        chunk = make_chunk()
        with patch.object(
            generator, "_call_gemini", side_effect=GeminiExtractionError("Non-JSON response")
        ):
            result = generator.generate(chunk)
        assert isinstance(result, ChunkMetadata)


# ---------------------------------------------------------------------------
# MetadataGenerator._validate_and_coerce
# ---------------------------------------------------------------------------


class TestValidateAndCoerce:
    def test_coerces_string_year_to_int(self, generator: MetadataGenerator) -> None:
        """'2019' as a string should be coerced to int 2019 in year_published."""
        chunk = make_chunk()
        raw = _valid_gemini_response(chunk)
        raw["year_published"] = "2019"
        result = generator._validate_and_coerce(raw, chunk)
        assert result.year_published == 2019
        assert isinstance(result.year_published, int)

    def test_coerces_single_keyword_string_to_list(
        self, generator: MetadataGenerator
    ) -> None:
        """A bare string for 'keywords' should be wrapped in a list."""
        chunk = make_chunk()
        raw = _valid_gemini_response(chunk)
        raw["keywords"] = "single_keyword"
        result = generator._validate_and_coerce(raw, chunk)
        assert result.keywords == ["single_keyword"]

    def test_unknown_doc_type_is_set_to_empty(self, generator: MetadataGenerator) -> None:
        """An unrecognized doc_type value should be coerced to '' to avoid filter errors."""
        chunk = make_chunk()
        raw = _valid_gemini_response(chunk)
        raw["doc_type"] = "unknown_type_xyz"
        result = generator._validate_and_coerce(raw, chunk)
        assert result.doc_type == ""

    def test_page_provenance_overrides_gemini_values(
        self, generator: MetadataGenerator
    ) -> None:
        """page_start, page_end, chunk_index, and doc_id should always come from the Chunk,
        not from the Gemini response."""
        chunk = make_chunk()
        chunk.page_start = 5
        chunk.page_end = 7
        chunk.chunk_index = 3
        raw = _valid_gemini_response(chunk)
        raw["page_start"] = 99
        raw["page_end"] = 99
        raw["chunk_index"] = 99
        raw["doc_id"] = "wrong_doc_id"
        result = generator._validate_and_coerce(raw, chunk)
        assert result.page_start == 5
        assert result.page_end == 7
        assert result.chunk_index == 3
        assert result.doc_id == chunk.doc_id

    def test_invalid_domain_falls_back_to_other(self, generator: MetadataGenerator) -> None:
        """An unrecognized domain value should be coerced to 'other'."""
        chunk = make_chunk()
        raw = _valid_gemini_response(chunk)
        raw["domain"] = "not_a_real_domain"
        result = generator._validate_and_coerce(raw, chunk)
        assert result.domain == "other"

    def test_invalid_year_string_becomes_none(self, generator: MetadataGenerator) -> None:
        """Non-numeric year_published strings should become None."""
        chunk = make_chunk()
        raw = _valid_gemini_response(chunk)
        raw["year_published"] = "not-a-year"
        result = generator._validate_and_coerce(raw, chunk)
        assert result.year_published is None


# ---------------------------------------------------------------------------
# MetadataGenerator._fallback_extraction
# ---------------------------------------------------------------------------


class TestFallbackExtraction:
    def test_returns_metadata_with_correct_doc_id(
        self, generator: MetadataGenerator
    ) -> None:
        """Fallback metadata should have the chunk's doc_id set correctly."""
        chunk = make_chunk(doc_id="deadbeef" * 8)
        result = generator._fallback_extraction(chunk)
        assert result.doc_id == chunk.doc_id

    def test_keywords_list_is_not_none(self, generator: MetadataGenerator) -> None:
        """Fallback metadata.keywords should be a list (possibly empty), never None."""
        chunk = make_chunk()
        result = generator._fallback_extraction(chunk)
        assert result.keywords is not None
        assert isinstance(result.keywords, list)

    def test_does_not_raise_on_empty_chunk_text(
        self, generator: MetadataGenerator
    ) -> None:
        """_fallback_extraction should not raise for a chunk with empty text."""
        chunk = make_chunk(text="")
        result = generator._fallback_extraction(chunk)
        assert isinstance(result, ChunkMetadata)
        assert result.title == ""

    def test_page_fields_match_chunk(self, generator: MetadataGenerator) -> None:
        """Fallback metadata page_start/page_end should come from the chunk."""
        chunk = make_chunk()
        chunk.page_start = 4
        chunk.page_end = 6
        result = generator._fallback_extraction(chunk)
        assert result.page_start == 4
        assert result.page_end == 6


# ---------------------------------------------------------------------------
# MetadataGenerator.generate_batch
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def test_returns_same_count_as_input(self, generator: MetadataGenerator) -> None:
        """generate_batch should return exactly len(chunks) metadata objects."""
        chunks = [make_chunk(text=f"Chunk {i}.", doc_id="abc123") for i in range(5)]
        for i, c in enumerate(chunks):
            c.chunk_index = i
            c.chunk_id = f"abc123_{i:05d}"
        with patch.object(
            generator, "_call_gemini", side_effect=lambda p: {
                "source_file": "test.pdf", "domain": "other", "doc_type": "textbook",
            }
        ):
            results = generator.generate_batch(chunks, delay_between_calls=0)
        assert len(results) == len(chunks)

    def test_order_is_preserved(self, generator: MetadataGenerator) -> None:
        """Output metadata[i] should correspond to input chunks[i]."""
        chunk_a = make_chunk(text="Chunk A text.", doc_id="aaa")
        chunk_b = make_chunk(text="Chunk B text.", doc_id="bbb")

        def mock_gemini(prompt: str) -> dict:
            if "aaa" in prompt or "Chunk A" in prompt:
                return {"source_file": "a.pdf", "domain": "other", "doc_type": "textbook",
                        "doc_id": "aaa", "page_start": 1, "page_end": 1, "chunk_index": 0}
            return {"source_file": "b.pdf", "domain": "other", "doc_type": "textbook",
                    "doc_id": "bbb", "page_start": 1, "page_end": 1, "chunk_index": 0}

        with patch.object(generator, "_call_gemini", side_effect=mock_gemini):
            results = generator.generate_batch([chunk_a, chunk_b], delay_between_calls=0)

        # Both entries should be ChunkMetadata instances in order
        assert len(results) == 2
        assert results[0].doc_id == chunk_a.doc_id
        assert results[1].doc_id == chunk_b.doc_id
