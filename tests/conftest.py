"""
Shared test fixtures for the RAG guidance pipeline.

Import fixtures from here instead of duplicating them across test files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models import Chunk, ExtractedDocument, Page


def make_chunk(
    text: str = "Sample clinical text.",
    doc_id: str = "abc123",
    chunk_index: int = 0,
    page_start: int = 1,
    page_end: int = 1,
    parent_section: str = "Introduction",
) -> Chunk:
    """Build a minimal Chunk for use in tests."""
    return Chunk(
        chunk_id=f"{doc_id}_{chunk_index:05d}",
        doc_id=doc_id,
        text=text,
        page_start=page_start,
        page_end=page_end,
        chunk_index=chunk_index,
        parent_section=parent_section,
    )


def make_extracted_doc(
    pages_text: list[str],
    doc_id: str = "abc123",
    source_file: str = "test.pdf",
) -> ExtractedDocument:
    """Build an ExtractedDocument from a list of page text strings."""
    pages = [Page(page_num=i + 1, text=t) for i, t in enumerate(pages_text)]
    return ExtractedDocument(
        doc_id=doc_id,
        source_file=source_file,
        source_path=f"/tmp/{source_file}",
        pages=pages,
    )
