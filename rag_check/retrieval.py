"""
Retrieval helpers for rag_check — pulls chunks out of the already-provisioned
Vertex AI Search DataStore (populated by scripts/batch_ingest.py) for the two
latency-comparison arms in compare_latency.py:

  - `search_filtered()` — Discovery Engine SearchService.search() with an
    optional metadata filter, returning the top-K ranked chunks. This is the
    "retrieved tagged guidance" arm.
  - `list_all_chunks()` — Discovery Engine DocumentService.list_documents()
    with no filter and no ranking: every chunk in the DataStore. This is the
    "entire corpus" arm.

Both return `RetrievedChunk` with the full chunk text (decoded from
`content.rawBytes`, the same field scripts/review_datastore.py reads) so the
caller can build an LLM context without a second round trip per chunk.

Region routing follows CLAUDE.md: Discovery Engine calls go through
`settings.discovery_engine_location` / `discovery_engine_endpoint`, never the
raw `gcp_location`.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

from config.settings import settings

logger = logging.getLogger(__name__)

# Array-typed metadata fields registered as filterable in
# scripts/setup_vertex_search.py (see config/rta_v1.json notes.vertex_ai_search).
# Discovery Engine's filter grammar needs `field: ANY("value")` for these and
# `field = "value"` (or a bare number) for scalars — see build_filter_expression().
ARRAY_FIELDS = frozenset({
    "therapeutic_modality", "clinical_presentation", "session_event_tags",
    "applies_when", "risk_dimension_tags", "patient_population",
    "clinical_measure_tags", "technique_tags", "clinical_caution",
})

# Scalar filterable fields, for validation/error messages only.
SCALAR_FIELDS = frozenset({
    "domain", "doc_type", "session_phase", "directionality",
    "source_file", "title", "chapter", "year_published", "subdomain",
})

_INTEGER_FIELDS = frozenset({"year_published", "page_start", "page_end", "chunk_index"})

_RPC_TIMEOUT = 300


@dataclass
class RetrievedChunk:
    """One chunk pulled back from the DataStore, with its full text."""

    chunk_id: str
    text: str
    struct_data: dict = field(default_factory=dict)


class FilterFieldError(ValueError):
    """Raised when a --tag field is neither a known scalar nor array field.

    Not fatal by design choice elsewhere in this repo but raised here (rather
    than silently building a no-op filter) because an unregistered/misspelled
    filter field fails SILENTLY at the Discovery Engine API layer per
    config/rta_v1.json notes.vertex_ai_search — exactly the failure mode this
    module exists to avoid reproducing.
    """


def build_filter_expression(tag_filters: list[tuple[str, str]]) -> str:
    """Build a Discovery Engine filter string from (field, value) pairs.

    AND-combines all pairs. Array-typed fields (ARRAY_FIELDS) use
    `field: ANY("value")`; scalar string fields use `field = "value"`; the one
    scalar integer field in practice (year_published) is emitted unquoted.

    Args:
        tag_filters: List of (field, value) pairs, e.g.
            [("session_event_tags", "disclosure_SI"), ("directionality", "contraindicated")].

    Returns:
        A filter expression string, or "" if `tag_filters` is empty (no filter).

    Raises:
        FilterFieldError: If a field name is not a known filterable field —
            see ARRAY_FIELDS / SCALAR_FIELDS above.
    """
    clauses = []
    for raw_field, raw_value in tag_filters:
        field_name = raw_field.strip()
        value = raw_value.strip()
        if field_name in ARRAY_FIELDS:
            clauses.append(f'{field_name}: ANY("{value}")')
        elif field_name in SCALAR_FIELDS:
            if field_name in _INTEGER_FIELDS:
                clauses.append(f"{field_name} = {int(value)}")
            else:
                clauses.append(f'{field_name} = "{value}"')
        else:
            known = sorted(ARRAY_FIELDS | SCALAR_FIELDS)
            raise FilterFieldError(
                f"Unknown filter field '{field_name}'. Known filterable fields: {known}. "
                f"(An unregistered field would otherwise silently no-op the filter — "
                f"see config/rta_v1.json notes.vertex_ai_search.)"
            )
    return " AND ".join(clauses)


def _client_options() -> ClientOptions | None:
    endpoint = settings.discovery_engine_endpoint
    return ClientOptions(api_endpoint=endpoint) if endpoint else None


def _datastore_branch_parent() -> str:
    """Full resource name of the default branch documents live under."""
    return (
        f"projects/{settings.gcp_project_id}/locations/{settings.discovery_engine_location}"
        f"/collections/default_collection/dataStores/{settings.vertex_search_datastore_id}"
        f"/branches/default_branch"
    )


def _serving_config() -> str:
    """Full resource name of the engine's default search serving config."""
    return (
        f"projects/{settings.gcp_project_id}/locations/{settings.discovery_engine_location}"
        f"/collections/default_collection/engines/{settings.vertex_search_engine_id}"
        f"/servingConfigs/default_search"
    )


def _decode_text(doc_dict: dict) -> str:
    """Decode a Document's base64 `content.rawBytes` to plain text.

    Returns an empty string (logged at WARNING) if the document has no inline
    content — e.g. a CONTENT_REQUIRED-less DataStore, or a document imported
    before content storage was enabled.
    """
    content = doc_dict.get("content") or {}
    raw = content.get("rawBytes") or content.get("raw_bytes")
    if not raw:
        logger.warning("Document %s has no inline content; skipping.", doc_dict.get("id"))
        return ""
    try:
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        logger.warning("Document %s content could not be decoded.", doc_dict.get("id"))
        return ""


def _doc_to_chunk(doc_dict: dict) -> RetrievedChunk:
    struct = doc_dict.get("structData") or doc_dict.get("struct_data") or {}
    return RetrievedChunk(
        chunk_id=str(doc_dict.get("id", "")),
        text=_decode_text(doc_dict),
        struct_data=struct,
    )


def search_filtered(
    query: str,
    filter_expr: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Run a ranked Discovery Engine search, optionally metadata-filtered.

    This is the "retrieved tagged guidance" arm: semantic/keyword ranking
    against `query`, restricted to documents matching `filter_expr` (empty
    string = no filter, i.e. rank across the whole corpus).

    Args:
        query: Search query text (defaults to the LLM prompt in the CLI).
        filter_expr: Discovery Engine filter string from
            build_filter_expression(), or "" for no filter.
        top_k: Max number of ranked results to return.

    Returns:
        Up to `top_k` RetrievedChunk, each with full decoded text.
    """
    client = discoveryengine.SearchServiceClient(client_options=_client_options())
    request = discoveryengine.SearchRequest(
        serving_config=_serving_config(),
        query=query,
        filter=filter_expr or None,
        page_size=top_k,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            # We fetch full text ourselves (below), not snippets — this just
            # tells Discovery Engine not to spend time generating them.
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=False
            ),
        ),
    )
    logger.info(
        "Search: query=%r filter=%r top_k=%d", query, filter_expr or "(none)", top_k
    )
    response = client.search(request=request, timeout=_RPC_TIMEOUT)

    # Search results carry the Document proto inline (the DataStore was
    # provisioned with content_config=CONTENT_REQUIRED — see
    # scripts/setup_vertex_search.py::create_datastore), so this is usually
    # one Discovery Engine call, not two. If a given result comes back
    # without inline content anyway (API behavior here isn't guaranteed
    # across Discovery Engine versions), fall back to a per-document
    # DocumentService.get_document() call rather than silently dropping the
    # chunk from the context.
    doc_client: discoveryengine.DocumentServiceClient | None = None
    chunks: list[RetrievedChunk] = []
    for result in response.results:
        doc_dict = discoveryengine.Document.to_dict(result.document)
        chunk = _doc_to_chunk(doc_dict)
        if not chunk.text and chunk.chunk_id:
            if doc_client is None:
                doc_client = discoveryengine.DocumentServiceClient(
                    client_options=_client_options()
                )
            name = f"{_datastore_branch_parent()}/documents/{chunk.chunk_id}"
            try:
                full_doc = doc_client.get_document(name=name, timeout=_RPC_TIMEOUT)
                chunk = _doc_to_chunk(discoveryengine.Document.to_dict(full_doc))
            except Exception:
                logger.warning(
                    "get_document fallback failed for %s; dropping from context.",
                    chunk.chunk_id,
                )
        if chunk.text:
            chunks.append(chunk)
    return chunks


def list_all_chunks(max_chunks: int | None = None) -> list[RetrievedChunk]:
    """List every chunk in the DataStore, unranked and unfiltered.

    This is the "entire corpus" arm. Paginates via ListDocuments (same call
    scripts/review_datastore.py uses to audit what was indexed).

    Args:
        max_chunks: Optional cap (useful if the corpus is large and you want
            a bounded latency test rather than the literal whole thing).
            None means no cap.

    Returns:
        All (or up to `max_chunks`) RetrievedChunk in the DataStore.
    """
    client = discoveryengine.DocumentServiceClient(client_options=_client_options())
    parent = _datastore_branch_parent()
    request = discoveryengine.ListDocumentsRequest(parent=parent, page_size=100)
    logger.info("Listing all documents in: %s", parent)

    chunks: list[RetrievedChunk] = []
    for doc in client.list_documents(request=request, timeout=_RPC_TIMEOUT):
        doc_dict = discoveryengine.Document.to_dict(doc)
        chunk = _doc_to_chunk(doc_dict)
        if chunk.text:
            chunks.append(chunk)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks
