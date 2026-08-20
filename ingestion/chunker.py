"""
Context-aware chunking of extracted PDF documents.

Two-pass strategy (see structure.md §3 and issues.md P1):
  1. Structural pass  — split on detected section headers and page group boundaries.
  2. Semantic pass    — sub-split sections that exceed the token budget using
                        sentence-embedding cosine similarity (split where similarity drops).

Each Chunk carries `parent_section`, `page_start`, and `page_end` provenance fields
so citations can be resolved to source pages.

Configuration is loaded from config/chunk_config.yaml via ChunkerConfig.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models import Chunk, ExtractedDocument, Page

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """Chunking parameters loaded from config/chunk_config.yaml."""

    max_tokens: int = 512
    """Hard token ceiling for a single chunk. Chunks never exceed this."""

    min_tokens: int = 64
    """Minimum chunk size; smaller fragments are merged with their neighbor."""

    overlap_tokens: int = 64
    """Token overlap between consecutive chunks to preserve context across boundaries."""

    semantic_similarity_threshold: float = 0.75
    """Cosine similarity below which a sentence boundary becomes a chunk boundary.
    Lower values = more splits. Tune per-domain — see issues.md P1."""

    embedding_model: str = "all-MiniLM-L6-v2"
    """sentence-transformers model for semantic boundary detection."""

    embedding_batch_size: int = 64
    """Sentences per forward pass through the embedding model."""

    max_section_pages: int = 50
    """Sections exceeding this page count are force-split at page boundaries
    before semantic splitting to avoid computing embeddings for huge sections."""

    header_patterns: list[str] = field(default_factory=lambda: [
        r"^(Chapter|Section|CHAPTER|SECTION)\s+\d+",
        r"^\d+\.\d*\s+[A-Z]",   # "3.1 Introduction" style
        r"^[A-Z][A-Z\s]{4,40}$",  # ALL CAPS headers (bounded to avoid false positives)
        r"^(Abstract|Introduction|Methods|Results|Discussion|Conclusion|Background|References)\s*$",
    ])
    """Regex patterns used to detect section headers in page text."""

    skip_doc_types: list[str] = field(default_factory=lambda: [
        "front_matter",
        "index",
        "bibliography",
    ])
    """Chunk metadata doc_types to exclude from the final output."""

    front_matter_indicators: list[str] = field(default_factory=lambda: [
        "Table of Contents",
        "Index",
        "Bibliography",
        "Acknowledgements",
        "Acknowledgments",
        "List of Figures",
        "List of Tables",
        "Preface",
        "Foreword",
    ])
    """Lines whose presence strongly suggests front-matter content."""

    @classmethod
    def from_yaml(cls, path: Path) -> "ChunkerConfig":
        """Load chunking parameters from config/chunk_config.yaml.

        Falls back to Python defaults for any key not present in the file.

        Args:
            path: Path to chunk_config.yaml.

        Returns:
            ChunkerConfig populated from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        defaults = cls()
        return cls(
            max_tokens=data.get("max_tokens", defaults.max_tokens),
            min_tokens=data.get("min_tokens", defaults.min_tokens),
            overlap_tokens=data.get("overlap_tokens", defaults.overlap_tokens),
            semantic_similarity_threshold=data.get(
                "semantic_similarity_threshold", defaults.semantic_similarity_threshold
            ),
            embedding_model=data.get("embedding_model", defaults.embedding_model),
            embedding_batch_size=data.get("embedding_batch_size", defaults.embedding_batch_size),
            max_section_pages=data.get("max_section_pages", defaults.max_section_pages),
            header_patterns=data.get("header_patterns", defaults.header_patterns),
            skip_doc_types=data.get("skip_doc_types", defaults.skip_doc_types),
            front_matter_indicators=data.get(
                "front_matter_indicators", defaults.front_matter_indicators
            ),
        )


class ContextAwareChunker:
    """
    Converts an ExtractedDocument into a list of semantically coherent Chunks.

    Usage:
        config = ChunkerConfig()
        chunker = ContextAwareChunker(config)
        chunks = chunker.chunk(extracted_doc)
    """

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        """
        Args:
            config: Chunking parameters. Defaults to ChunkerConfig() if None.
        """
        self._config = config or ChunkerConfig()
        self._embedding_model = None  # Lazy-loaded on first use

    def chunk(self, doc: ExtractedDocument) -> list[Chunk]:
        """Convert an ExtractedDocument into a list of Chunks.

        Runs the structural pass followed by the semantic pass on each section.
        Chunks are assigned sequential `chunk_index` values across the document.

        Args:
            doc: Fully extracted document from PDFExtractor.

        Returns:
            Ordered list of Chunks. Empty if the document has no extractable text.
        """
        sections = self._structural_split(doc)
        all_chunks: list[Chunk] = []
        chunk_index = 0

        for section_name, pages in sections:
            section_chunks = self._semantic_split(
                section_name, pages, doc.doc_id, chunk_index
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return all_chunks

    def _structural_split(
        self, doc: ExtractedDocument
    ) -> list[tuple[str, list[Page]]]:
        """Group pages into sections based on detected headers.

        Scans each page for lines matching ChunkerConfig.header_patterns.
        When a new header is found, the current section is closed and a new one
        begins. Pages with no detectable header belong to the most recent section.

        Args:
            doc: Extracted document.

        Returns:
            List of (section_name, pages) tuples in document order.
            `section_name` is the header text or "preamble" for pre-header pages.
        """
        sections: list[tuple[str, list[Page]]] = []
        current_header = "preamble"
        current_pages: list[Page] = []

        for page in doc.pages:
            headers = self._detect_section_headers(page.text)
            if headers:
                if current_pages:
                    sections.append((current_header, current_pages))
                current_header = headers[0]
                current_pages = [page]
            else:
                current_pages.append(page)

            # Force-split oversized sections at page boundaries
            if len(current_pages) >= self._config.max_section_pages:
                sections.append((current_header, current_pages))
                current_pages = []

        if current_pages:
            sections.append((current_header, current_pages))

        return sections if sections else [("preamble", doc.pages)]

    def _semantic_split(
        self,
        section_name: str,
        pages: list[Page],
        doc_id: str,
        start_chunk_idx: int,
    ) -> list[Chunk]:
        """Split a section's text into token-bounded, semantically coherent chunks.

        Steps:
          1. Concatenate page text; track page-number offsets per sentence.
          2. Embed all sentences using the configured sentence-transformers model.
          3. Find boundaries where cosine similarity between adjacent sentences
             drops below semantic_similarity_threshold.
          4. Merge spans into chunks respecting max_tokens and min_tokens.
          5. Add overlap tokens from the previous chunk's tail.

        Args:
            section_name: The section header text (used as parent_section).
            pages: Pages in this section.
            doc_id: Parent document ID.
            start_chunk_idx: The chunk_index to assign to the first chunk of this section.

        Returns:
            List of Chunks for this section.
        """
        # Step 1: collect sentences with page provenance
        all_sentences: list[str] = []
        sentence_pages: list[int] = []

        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            sents = re.split(r"(?<=[.!?])\s+", text)
            for sent in sents:
                sent = sent.strip()
                if sent:
                    all_sentences.append(sent)
                    sentence_pages.append(page.page_num)

        if not all_sentences:
            return []

        # Step 2: find semantic split points
        if len(all_sentences) > 1:
            embeddings = self._embed_sentences(all_sentences)
            semantic_boundaries = set(self._find_semantic_boundaries(embeddings))
        else:
            semantic_boundaries = {0}

        # Step 3: greedily build chunks, splitting at semantic boundaries and token budget
        chunks: list[Chunk] = []
        chunk_index = start_chunk_idx
        prev_tail_words: list[str] = []

        current_sents: list[str] = []
        current_pages_list: list[int] = []

        def emit() -> None:
            nonlocal chunk_index
            if not current_sents:
                return
            body = " ".join(current_sents)
            text_with_overlap = (
                (" ".join(prev_tail_words) + " " + body).strip()
                if prev_tail_words
                else body
            )
            page_start = min(current_pages_list)
            page_end = max(current_pages_list)
            chunks.append(
                Chunk(
                    chunk_id=self._make_chunk_id(doc_id, chunk_index),
                    doc_id=doc_id,
                    text=text_with_overlap,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=chunk_index,
                    parent_section=section_name,
                )
            )
            # Update overlap tail from the body (not including previous overlap)
            body_words = body.split()
            prev_tail_words.clear()
            if self._config.overlap_tokens > 0:
                prev_tail_words.extend(body_words[-self._config.overlap_tokens :])
            chunk_index += 1
            current_sents.clear()
            current_pages_list.clear()

        for i, (sent, page_num) in enumerate(zip(all_sentences, sentence_pages)):
            is_semantic_boundary = i in semantic_boundaries and i > 0
            projected_tokens = self._token_count(" ".join(current_sents + [sent]))

            if current_sents and projected_tokens > self._config.max_tokens:
                # Hard token limit reached — emit before adding this sentence
                emit()
            elif is_semantic_boundary and self._token_count(" ".join(current_sents)) >= self._config.min_tokens:
                # Semantic boundary with enough content — emit
                emit()

            current_sents.append(sent)
            current_pages_list.append(page_num)

        emit()
        return chunks

    def _detect_section_headers(self, text: str) -> list[str]:
        """Return all lines in text that match any configured header pattern.

        Args:
            text: Raw page text.

        Returns:
            List of matched header strings, in order of appearance.
        """
        matched: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for pattern in self._config.header_patterns:
                if re.match(pattern, line):
                    matched.append(line)
                    break
        return matched

    def _embed_sentences(self, sentences: list[str]) -> "np.ndarray":
        """Compute sentence embeddings using the configured model.

        Lazy-loads the sentence-transformers model on first call.

        Args:
            sentences: List of sentence strings.

        Returns:
            2D numpy array of shape (len(sentences), embedding_dim).
        """
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model.encode(
            sentences,
            batch_size=self._config.embedding_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _find_semantic_boundaries(
        self, embeddings: "np.ndarray"
    ) -> list[int]:
        """Return sentence indices where a chunk boundary should be placed.

        A boundary is placed at index i when the cosine similarity between
        embeddings[i-1] and embeddings[i] drops below
        ChunkerConfig.semantic_similarity_threshold.

        Args:
            embeddings: (N, D) embedding array from _embed_sentences.

        Returns:
            Sorted list of sentence indices that start new chunks.
            Index 0 is always included.
        """
        import numpy as np

        boundaries = [0]
        for i in range(1, len(embeddings)):
            a = embeddings[i - 1]
            b = embeddings[i]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0.0 or norm_b == 0.0:
                similarity = 0.0
            else:
                similarity = float(np.dot(a, b) / (norm_a * norm_b))
            if similarity < self._config.semantic_similarity_threshold:
                boundaries.append(i)
        return boundaries

    def _token_count(self, text: str) -> int:
        """Approximate token count using whitespace splitting (fast, not exact).

        For a precise count, swap this for a tiktoken or sentencepiece tokenizer
        matching the target model's vocabulary.

        Args:
            text: Input text string.

        Returns:
            Approximate token count.
        """
        return len(text.split())

    def _make_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Build the canonical chunk ID string.

        Format: "{doc_id}_{chunk_index:05d}"

        Args:
            doc_id: Parent document SHA-256 ID.
            chunk_index: Zero-based sequential chunk index.

        Returns:
            Unique chunk identifier string.
        """
        return f"{doc_id}_{chunk_index:05d}"

    def _load_embedding_model(self) -> None:
        """Lazy-initialize the sentence-transformers model.

        Called on first use of _embed_sentences to avoid loading the model
        during import (slow) or when Document AI OCR falls back (unnecessary).
        """
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", self._config.embedding_model)
        self._embedding_model = SentenceTransformer(self._config.embedding_model)
