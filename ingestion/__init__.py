"""
Ingestion package — corpus scanning, PDF extraction, chunking, metadata
generation, GCS upload, and Vertex AI Search indexing.

Primary entry point: CorpusScanner.scan() (via scripts/batch_ingest.py)

Typical call order for a single document:
    extractor  -> ExtractedDocument
    chunker    -> list[Chunk]
    metadata_gen (per chunk) -> ChunkMetadata attached to each Chunk
    uploader   -> GCS URIs (PDF + chunk JSONL)
    indexer    -> Vertex AI Search import operation
"""

from ingestion.scanner import CorpusScanner
from ingestion.extractor import PDFExtractor
from ingestion.chunker import ContextAwareChunker
from ingestion.metadata_gen import MetadataGenerator
from ingestion.uploader import GCSUploader
from ingestion.indexer import VertexSearchIndexer

__all__ = [
    "CorpusScanner",
    "PDFExtractor",
    "ContextAwareChunker",
    "MetadataGenerator",
    "GCSUploader",
    "VertexSearchIndexer",
]
