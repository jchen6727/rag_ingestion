"""
Tests for ingestion/extractor.py.

Covers the PDFExtractor's text-layer detection, pdfplumber extraction path,
and Document AI fallback path. Uses sample PDFs in tests/fixtures/.

Run:
    pytest tests/test_extractor.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.extractor import PDFExtractor
from models import ExtractedDocument, Page


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def extractor() -> PDFExtractor:
    """PDFExtractor with Document AI disabled (default test configuration)."""
    return PDFExtractor(use_document_ai=False)


@pytest.fixture
def extractor_with_docai() -> PDFExtractor:
    """PDFExtractor configured to use Document AI."""
    return PDFExtractor(
        use_document_ai=True,
        document_ai_processor_id="projects/test/locations/us/processors/test-processor",
        project_id="test-project",
    )


def make_doc(pages_text: list[str], doc_id: str = "abc123") -> ExtractedDocument:
    """Build an ExtractedDocument from a list of page text strings."""
    pages = [Page(page_num=i + 1, text=t) for i, t in enumerate(pages_text)]
    return ExtractedDocument(
        doc_id=doc_id,
        source_file="test.pdf",
        source_path="/tmp/test.pdf",
        pages=pages,
    )


def make_mock_plumber_page(
    text: str = "", page_number: int = 1, images=None, tables=None
) -> MagicMock:
    """Return a mock pdfplumber page object."""
    page = MagicMock()
    page.page_number = page_number
    page.extract_text.return_value = text
    page.find_tables.return_value = tables if tables is not None else []
    page.images = images if images is not None else []
    return page


def plumber_ctx(pages: list) -> MagicMock:
    """Return a mock context manager wrapping a pdfplumber PDF with the given pages."""
    mock_pdf = MagicMock()
    mock_pdf.pages = pages
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_pdf)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# PDFExtractor.extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_raises_on_missing_file(self, extractor: PDFExtractor, tmp_path: Path) -> None:
        """extract() should raise FileNotFoundError for a non-existent path."""
        missing = tmp_path / "ghost.pdf"
        with pytest.raises(FileNotFoundError):
            extractor.extract(missing)

    def test_returns_extracted_document(self, extractor: PDFExtractor, tmp_path: Path) -> None:
        """extract() should return an ExtractedDocument for a valid PDF."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")

        plumber_page = make_mock_plumber_page(text="x" * 200)

        with patch("pdfplumber.open", return_value=plumber_ctx([plumber_page])):
            result = extractor.extract(pdf)

        assert isinstance(result, ExtractedDocument)
        assert result.source_file == "test.pdf"
        assert len(result.pages) == 1

    def test_falls_back_to_document_ai_when_no_text_layer(self, tmp_path: Path) -> None:
        """extract() should trigger Document AI when pdfplumber yields no text."""
        pdf = tmp_path / "scanned.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")

        # Needs a processor_id so the fallback path (not the warn-and-return path) runs.
        extractor = PDFExtractor(
            use_document_ai=False,
            document_ai_processor_id="projects/p/locations/us/processors/abc",
            project_id="test-project",
        )

        plumber_page = make_mock_plumber_page(text="")  # no text → below threshold

        docai_result = ExtractedDocument(
            doc_id="abc",
            source_file="scanned.pdf",
            source_path=str(pdf),
            pages=[Page(page_num=1, text="OCR text via Document AI")],
            extraction_method="document_ai",
        )

        with patch("pdfplumber.open", return_value=plumber_ctx([plumber_page])):
            with patch.object(
                extractor, "_extract_with_document_ai", return_value=docai_result
            ) as mock_docai:
                result = extractor.extract(pdf)

        mock_docai.assert_called_once()
        assert result.extraction_method == "document_ai"

    def test_sets_correct_extraction_method(self, extractor: PDFExtractor, tmp_path: Path) -> None:
        """ExtractedDocument.extraction_method should reflect which path was used."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")

        plumber_page = make_mock_plumber_page(text="x" * 200)

        with patch("pdfplumber.open", return_value=plumber_ctx([plumber_page])):
            result = extractor.extract(pdf)

        assert result.extraction_method == "pdfplumber"


# ---------------------------------------------------------------------------
# PDFExtractor._has_text_layer
# ---------------------------------------------------------------------------


class TestHasTextLayer:
    def test_returns_true_for_text_heavy_doc(self, extractor: PDFExtractor) -> None:
        """_has_text_layer returns True when average chars/page exceeds threshold."""
        doc = make_doc(["x" * 200])
        assert extractor._has_text_layer(doc) is True

    def test_returns_false_for_empty_pages(self, extractor: PDFExtractor) -> None:
        """_has_text_layer returns False when pages have no text."""
        doc = make_doc(["", ""])
        assert extractor._has_text_layer(doc) is False

    def test_uses_min_chars_threshold(self) -> None:
        """_has_text_layer respects the min_text_chars_per_page configuration."""
        doc = make_doc(["x" * 40, "x" * 40])  # avg = 40 chars/page

        strict = PDFExtractor(min_text_chars_per_page=50)
        assert strict._has_text_layer(doc) is False

        lenient = PDFExtractor(min_text_chars_per_page=30)
        assert lenient._has_text_layer(doc) is True

    def test_returns_false_for_no_pages(self, extractor: PDFExtractor) -> None:
        """_has_text_layer returns False for a document with zero pages."""
        doc = make_doc([])
        assert extractor._has_text_layer(doc) is False


# ---------------------------------------------------------------------------
# PDFExtractor._parse_pdfplumber_page
# ---------------------------------------------------------------------------


class TestParsePdfplumberPage:
    def test_preserves_page_number(self, extractor: PDFExtractor) -> None:
        """Returned Page.page_num should match the input page_num argument."""
        mock_page = make_mock_plumber_page(text="some text", page_number=7)
        result = extractor._parse_pdfplumber_page(mock_page, 7)
        assert result.page_num == 7

    def test_detects_figures_from_image_objects(self, extractor: PDFExtractor) -> None:
        """Page.has_figures should be True when the pdfplumber page has images."""
        mock_page = make_mock_plumber_page(images=[{"name": "Im0", "width": 100}])
        result = extractor._parse_pdfplumber_page(mock_page, 1)
        assert result.has_figures is True

    def test_no_figures_when_images_empty(self, extractor: PDFExtractor) -> None:
        """Page.has_figures should be False when the page has no image objects."""
        mock_page = make_mock_plumber_page(images=[])
        result = extractor._parse_pdfplumber_page(mock_page, 1)
        assert result.has_figures is False

    def test_extracts_tables(self, extractor: PDFExtractor) -> None:
        """_parse_pdfplumber_page should populate Page.tables for pages with tables."""
        mock_table = MagicMock()
        mock_table.extract.return_value = [["A", "B"], ["C", "D"]]
        mock_table.bbox = (0.0, 0.0, 100.0, 50.0)

        mock_page = make_mock_plumber_page(tables=[mock_table])
        result = extractor._parse_pdfplumber_page(mock_page, 1)

        assert len(result.tables) == 1
        assert result.tables[0]["rows"] == [["A", "B"], ["C", "D"]]

    def test_returns_page_instance(self, extractor: PDFExtractor) -> None:
        """_parse_pdfplumber_page should return a Page dataclass instance."""
        mock_page = make_mock_plumber_page(text="hello")
        result = extractor._parse_pdfplumber_page(mock_page, 1)
        assert isinstance(result, Page)


# ---------------------------------------------------------------------------
# PDFExtractor._extract_tables
# ---------------------------------------------------------------------------


class TestExtractTables:
    def test_returns_empty_list_for_no_tables(self, extractor: PDFExtractor) -> None:
        """_extract_tables should return [] when the page has no tables."""
        mock_page = MagicMock()
        mock_page.find_tables.return_value = []
        assert extractor._extract_tables(mock_page) == []

    def test_table_has_rows_key(self, extractor: PDFExtractor) -> None:
        """Each table dict should contain a 'rows' key with a list of row arrays."""
        mock_table = MagicMock()
        mock_table.extract.return_value = [["A", "B"]]
        mock_table.bbox = (0.0, 0.0, 100.0, 50.0)

        mock_page = MagicMock()
        mock_page.find_tables.return_value = [mock_table]

        result = extractor._extract_tables(mock_page)

        assert len(result) == 1
        assert "rows" in result[0]
        assert result[0]["rows"] == [["A", "B"]]

    def test_table_has_bbox_key(self, extractor: PDFExtractor) -> None:
        """Each table dict should contain a 'bbox' key."""
        mock_table = MagicMock()
        mock_table.extract.return_value = [["X"]]
        mock_table.bbox = (10.0, 20.0, 110.0, 70.0)

        mock_page = MagicMock()
        mock_page.find_tables.return_value = [mock_table]

        result = extractor._extract_tables(mock_page)

        assert "bbox" in result[0]
        assert result[0]["bbox"] == (10.0, 20.0, 110.0, 70.0)

    def test_skips_malformed_table(self, extractor: PDFExtractor) -> None:
        """_extract_tables should skip tables whose extract() raises an exception."""
        bad_table = MagicMock()
        bad_table.extract.side_effect = RuntimeError("corrupt table")

        mock_page = MagicMock()
        mock_page.find_tables.return_value = [bad_table]

        result = extractor._extract_tables(mock_page)
        assert result == []

    def test_multiple_tables_returned(self, extractor: PDFExtractor) -> None:
        """_extract_tables should return one dict per table on the page."""
        def make_table(rows):
            t = MagicMock()
            t.extract.return_value = rows
            t.bbox = (0.0, 0.0, 1.0, 1.0)
            return t

        mock_page = MagicMock()
        mock_page.find_tables.return_value = [make_table([["A"]]), make_table([["B"], ["C"]])]

        result = extractor._extract_tables(mock_page)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# PDFExtractor._compute_doc_id
# ---------------------------------------------------------------------------


class TestComputeDocId:
    def test_returns_64_char_hex_string(self, extractor: PDFExtractor, tmp_path: Path) -> None:
        """_compute_doc_id should return a 64-character SHA-256 hex digest."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf content")
        doc_id = extractor._compute_doc_id(pdf)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 64

    def test_stable_across_calls(self, extractor: PDFExtractor, tmp_path: Path) -> None:
        """_compute_doc_id should return the same value for identical file content."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"stable content")
        assert extractor._compute_doc_id(pdf) == extractor._compute_doc_id(pdf)

    def test_differs_for_different_content(self, extractor: PDFExtractor, tmp_path: Path) -> None:
        """Different file content must produce different doc IDs."""
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert extractor._compute_doc_id(a) != extractor._compute_doc_id(b)
