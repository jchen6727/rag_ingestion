"""
PDF text and structure extraction.

Primary path: pdfplumber (layout-aware, handles multi-column and tables).
Fallback path: Google Document AI (OCR for scanned PDFs with no text layer).

The public interface is PDFExtractor.extract(path) which returns an
ExtractedDocument regardless of which path was used.

See caveats.md §3 for known limitations of PDF extraction.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from models import ExtractedDocument, Page

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when both pdfplumber and Document AI fail to extract a PDF."""


class PDFExtractor:
    """
    Extracts text and structure from a PDF file.

    Automatically falls back to Document AI OCR when pdfplumber yields
    fewer than `min_text_chars_per_page` characters on average (indicating
    a scanned PDF without a text layer).
    """

    def __init__(
        self,
        use_document_ai: bool = False,
        document_ai_processor_id: str = "",
        project_id: str = "",
        location: str = "us",
        min_text_chars_per_page: int = 100,
    ) -> None:
        """
        Args:
            use_document_ai: If True, always use Document AI instead of pdfplumber.
                             If False, Document AI is used only as a fallback when
                             text-layer detection fails.
            document_ai_processor_id: Full resource name of the Document AI processor,
                                       e.g. projects/{id}/locations/{loc}/processors/{id}.
            project_id: GCP project ID (required if Document AI is enabled).
            location: GCP region for Document AI (typically "us" or "eu").
            min_text_chars_per_page: Average characters per page below which the PDF
                                     is classified as scanned and falls back to OCR.
        """
        self._use_document_ai = use_document_ai
        self._processor_id = document_ai_processor_id
        self._project_id = project_id
        self._location = location
        self._min_chars = min_text_chars_per_page

    def extract(self, path: Path) -> ExtractedDocument:
        """Extract text and structure from a PDF, choosing the appropriate method.

        If use_document_ai is False, attempts pdfplumber first. If the result
        fails the text-layer quality check, re-extracts with Document AI.

        Args:
            path: Absolute path to the PDF file. Must exist and be readable.

        Returns:
            ExtractedDocument with all pages populated.

        Raises:
            FileNotFoundError: If path does not exist.
            ExtractionError: If both pdfplumber and Document AI fail.
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_id = self._compute_doc_id(path)

        if self._use_document_ai:
            return self._extract_with_document_ai(path, doc_id)

        try:
            doc = self._extract_with_pdfplumber(path, doc_id)
        except Exception as exc:
            logger.warning("pdfplumber failed for %s: %s — trying Document AI", path.name, exc)
            if not self._processor_id:
                raise ExtractionError(
                    f"pdfplumber failed and Document AI is not configured: {exc}"
                ) from exc
            return self._extract_with_document_ai(path, doc_id)

        if not self._has_text_layer(doc):
            logger.info(
                "%s has no usable text layer (avg chars/page below %d), falling back to Document AI",
                path.name,
                self._min_chars,
            )
            if not self._processor_id:
                logger.warning(
                    "Document AI not configured; returning low-quality pdfplumber result for %s",
                    path.name,
                )
                return doc
            return self._extract_with_document_ai(path, doc_id)

        return doc

    def _has_text_layer(self, doc: ExtractedDocument) -> bool:
        """Heuristic check: return True if the document has a usable text layer.

        Computes the average characters per page across all extracted pages.
        Falls back to Document AI if below self._min_chars.

        Args:
            doc: Partially extracted document from pdfplumber.

        Returns:
            True if the document has sufficient text for indexing.
        """
        if not doc.pages:
            return False
        avg = sum(len(p.text) for p in doc.pages) / len(doc.pages)
        return avg >= self._min_chars

    def _extract_with_pdfplumber(self, path: Path, doc_id: str) -> ExtractedDocument:
        """Extract text and tables using pdfplumber.

        Preserves page boundaries and attempts to extract table cell content
        as structured rows. Multi-column layouts may produce interleaved text —
        see caveats.md §3.

        Args:
            path: Path to the PDF.

        Returns:
            ExtractedDocument with extraction_method = "pdfplumber".
        """
        import pdfplumber

        pages: list[Page] = []
        with pdfplumber.open(path) as pdf:
            for plumber_page in pdf.pages:
                pages.append(self._parse_pdfplumber_page(plumber_page, plumber_page.page_number))

        return ExtractedDocument(
            doc_id=doc_id,
            source_file=path.name,
            source_path=str(path),
            pages=pages,
            extraction_method="pdfplumber",
        )

    def _extract_with_document_ai(self, path: Path, doc_id: str) -> ExtractedDocument:
        """Extract text using Google Document AI OCR.

        Reads the PDF bytes, submits to the configured processor, and parses
        the response into Page objects. Significantly slower and more expensive
        than pdfplumber (~$1.50/1000 pages) — gate this behind _has_text_layer.

        Args:
            path: Path to the PDF.

        Returns:
            ExtractedDocument with extraction_method = "document_ai".

        Raises:
            ValueError: If document_ai_processor_id is not configured.
        """
        if not self._processor_id:
            raise ValueError(
                "document_ai_processor_id must be set to use Document AI extraction"
            )

        from google.cloud import documentai

        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{self._location}-documentai.googleapis.com"}
        )

        raw_bytes = path.read_bytes()
        request = documentai.ProcessRequest(
            name=self._processor_id,
            raw_document=documentai.RawDocument(
                content=raw_bytes,
                mime_type="application/pdf",
            ),
        )

        result = client.process_document(request=request)
        document = result.document

        # Document AI returns one document with pages; map them to our Page model.
        # Text is stored flat in document.text; page tokens carry layout offsets.
        pages: list[Page] = []
        full_text = document.text

        for dai_page in document.pages:
            page_num = dai_page.page_number  # 1-indexed
            # Collect text segments that belong to this page via their layout offsets.
            segments: list[str] = []
            for block in dai_page.blocks:
                for segment in block.layout.text_anchor.text_segments:
                    start = int(segment.start_index) if segment.start_index else 0
                    end = int(segment.end_index)
                    segments.append(full_text[start:end])
            page_text = "".join(segments)
            pages.append(Page(page_num=page_num, text=page_text, tables=[], has_figures=False))

        return ExtractedDocument(
            doc_id=doc_id,
            source_file=path.name,
            source_path=str(path),
            pages=pages,
            extraction_method="document_ai",
        )

    def _parse_pdfplumber_page(self, page: object, page_num: int) -> Page:
        """Convert a pdfplumber Page object into the internal Page model.

        Extracts plain text via page.extract_text() and tables via
        page.extract_tables(). Detects figure presence by checking for
        image objects on the page.

        Args:
            page: A pdfplumber page object.
            page_num: 1-indexed page number.

        Returns:
            Populated Page dataclass.
        """
        text: str = page.extract_text() or ""
        tables: list[dict] = self._extract_tables(page)
        has_figures: bool = bool(getattr(page, "images", None))

        return Page(
            page_num=page_num,
            text=text,
            tables=tables,
            has_figures=has_figures,
        )

    def _extract_tables(self, page: object) -> list[dict]:
        """Extract tables from a pdfplumber page as a list of row arrays.

        Each returned dict has:
            {"rows": [["cell00", "cell01"], ["cell10", "cell11"]], "bbox": tuple}

        Merged cells are represented as None in the row array. Complex table
        structures (spanning headers, nested tables) are not reliably supported.

        Args:
            page: A pdfplumber page object.

        Returns:
            List of table dicts, possibly empty.
        """
        result: list[dict] = []
        try:
            tables = page.find_tables()
        except Exception as exc:
            logger.debug("Table extraction failed on page %s: %s", getattr(page, "page_number", "?"), exc)
            return result

        for table in tables:
            try:
                rows = table.extract()
                bbox = tuple(table.bbox) if hasattr(table, "bbox") else ()
                result.append({"rows": rows, "bbox": bbox})
            except Exception as exc:
                logger.debug("Skipping malformed table: %s", exc)

        return result

    def _compute_doc_id(self, path: Path) -> str:
        """Return the SHA-256 hex digest of the PDF bytes (stable doc identifier)."""
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
