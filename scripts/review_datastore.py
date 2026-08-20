"""
Review what was actually INDEXED — fetch chunks (documents) back out of the
Vertex AI Search DataStore and print their tags and text.

Use this AFTER a real ingestion (scripts/batch_ingest.py) to confirm the chunks
and their metadata tags landed correctly in the cloud. It is the CLI counterpart
to browsing the DataStore in the Google Cloud console (cloud.google.com →
AI Applications / Agent Builder → Data Stores → your store → Documents).

Usage:
    # Show the first 20 indexed chunks with their tags:
    PYTHONPATH=. python scripts/review_datastore.py

    # Only chunks from one PDF (pass the doc_id, or its first characters):
    PYTHONPATH=. python scripts/review_datastore.py --doc-id ef0c4d1d

    # Show more, and save a Markdown report you can open:
    PYTHONPATH=. python scripts/review_datastore.py --limit 100 --out indexed_review.md

    # Just count how many chunks are in the store:
    PYTHONPATH=. python scripts/review_datastore.py --count-only

Reads GCP settings from your .env (GCP_PROJECT_ID, GCP_LOCATION,
VERTEX_SEARCH_DATASTORE_ID). If the store does not exist yet, run
scripts/setup_vertex_search.py first.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from scripts._gcp_logging import describe_google_error, log_api_error, setup_logging

logger = None  # set in main()

# Tags shown per chunk, in review order. Safety flags last so they stand out.
_SHOWN_TAGS = [
    "domain", "doc_type", "therapeutic_modality", "clinical_presentation",
    "session_event_tags", "session_phase", "technique_tags",
    "patient_population", "clinical_measure_tags",
    "directionality", "applies_when", "clinical_caution", "risk_dimension_tags",
]


def _datastore_parent() -> str:
    """Build the branch resource name that documents live under.

    Uses ``discovery_engine_location`` ('global'/'us'/'eu'), not the raw compute
    region, so it matches how the DataStore was created.
    """
    return (
        f"projects/{settings.gcp_project_id}/locations/{settings.discovery_engine_location}"
        f"/collections/default_collection/dataStores/{settings.vertex_search_datastore_id}"
        f"/branches/default_branch"
    )


def _doc_to_dict(doc) -> dict:
    """Convert a discoveryengine Document (proto-plus) to a plain dict."""
    from google.cloud import discoveryengine_v1 as discoveryengine
    return discoveryengine.Document.to_dict(doc)


def _content_preview(doc_dict: dict, n: int) -> str:
    """Best-effort plain-text preview from the document's content.rawBytes."""
    content = doc_dict.get("content") or {}
    raw = content.get("rawBytes") or content.get("raw_bytes")
    if not raw:
        return "(text stored in GCS; not returned inline)"
    try:
        text = base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        return "(could not decode inline content)"
    return " ".join(text.split())[:n]


def _render(doc_dict: dict, preview_chars: int) -> list[str]:
    """Render one indexed document as Markdown lines."""
    struct = doc_dict.get("structData") or doc_dict.get("struct_data") or {}
    out = [f"### {doc_dict.get('id', '(no id)')}"]
    page_s, page_e = struct.get("page_start"), struct.get("page_end")
    if page_s is not None:
        page = f"p.{page_s}" if page_s == page_e else f"pp.{page_s}–{page_e}"
        out.append(f"*{page} · {struct.get('source_file', '')}*\n")
    if not struct:
        out.append("- ⚠️ no structData (metadata) on this document — was the schema "
                   "registered before import? (setup_vertex_search.py)\n")
    for tag in _SHOWN_TAGS:
        if tag in struct:
            out.append(f"- {tag}: `{struct.get(tag)}`")
    out.append(f"\n> {_content_preview(doc_dict, preview_chars)}\n")
    return out


def main() -> None:
    global logger
    parser = argparse.ArgumentParser(
        description="Review chunks/tags already indexed in the Vertex AI Search DataStore.",
    )
    parser.add_argument("--doc-id", default=None,
                        help="Only show chunks whose ID starts with this (a doc_id or its prefix).")
    parser.add_argument("--limit", type=int, default=20, help="Max chunks to display (default 20).")
    parser.add_argument("--count-only", action="store_true", help="Only print the total document count.")
    parser.add_argument("--out", type=Path, default=None, help="Also write a Markdown report to this file.")
    parser.add_argument("--preview-chars", type=int, default=300, help="Chars of text shown per chunk.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging incl. Google/gRPC internals.")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        settings.validate_all()
    except Exception as exc:  # missing env vars
        logger.error("Configuration problem:\n%s", exc)
        sys.exit(2)

    try:
        from google.api_core.client_options import ClientOptions
        from google.cloud import discoveryengine_v1 as discoveryengine
        endpoint = settings.discovery_engine_endpoint
        client = discoveryengine.DocumentServiceClient(
            client_options=ClientOptions(api_endpoint=endpoint) if endpoint else None
        )
        parent = _datastore_parent()
        logger.info("Listing documents in: %s", parent)
        request = discoveryengine.ListDocumentsRequest(parent=parent, page_size=100)
        pager = client.list_documents(request=request)

        shown = 0
        total = 0
        report: list[str] = [f"# Indexed chunks — {settings.vertex_search_datastore_id}\n"]
        for doc in pager:
            total += 1
            if args.count_only:
                continue
            if args.doc_id and not str(doc.id).startswith(args.doc_id):
                continue
            if shown < args.limit:
                doc_dict = _doc_to_dict(doc)
                report.extend(_render(doc_dict, args.preview_chars))
                shown += 1
    except Exception as exc:  # noqa: BLE001 — surface any Google API error clearly
        log_api_error(logger, exc, "listing documents in the DataStore")
        sys.exit(1)

    print("\n" + "=" * 68)
    print(f"DataStore: {settings.vertex_search_datastore_id}  (location: {settings.gcp_location})")
    print(f"Total indexed chunks: {total}")
    if args.doc_id:
        print(f"Filtered to doc_id prefix '{args.doc_id}': showing {shown}")
    if total == 0:
        print("No documents found. Either nothing has been ingested yet, or GCP_LOCATION /"
              "\nVERTEX_SEARCH_DATASTORE_ID point at a different store. Run batch_ingest.py first.")
    print("=" * 68)

    if not args.count_only and shown:
        # Print the human-readable cards to stdout too (skip the H1).
        print("\n".join(report[1:]))

    if args.out and not args.count_only:
        args.out.write_text("\n".join(report), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
