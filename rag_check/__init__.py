"""
rag_check — latency comparison harness: filtered-RAG vs. whole-corpus RAG.

Not part of the ingestion pipeline's stated scope (see CLAUDE.md: "the
query/retrieval/generation side ... is intentionally not part of this
package"). This is a separate, narrow measurement tool that reads from a
DataStore already populated by scripts/batch_ingest.py; it does not write to
it and does not import from ingestion/.

See rag_check/README.md for usage.
"""
