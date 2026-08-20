"""
Inspect how one or more PDFs will be chunked and tagged — WITHOUT touching the
cloud. This is the "review tags and chunks" tool: run it before (or instead of)
a real ingestion to see exactly what the pipeline produces.

What it does, per PDF:
  1. computes the doc_id (SHA-256 of the file — identical to what the real
     pipeline uses, so chunk IDs match),
  2. extracts the text (pdfplumber),
  3. splits it into chunks (the same ContextAwareChunker the pipeline uses),
  4. (optional) generates the metadata TAGS for each chunk with Gemini,
  5. writes two review files per PDF into the output directory:
        <name>.<docid8>.chunks.jsonl   — one machine-readable line per chunk
        <name>.<docid8>.review.md      — a human-readable report you can open

Nothing is uploaded and nothing is indexed. With --no-metadata it does not call
any Google API at all (no credentials or cost) — useful to sanity-check chunking.

Usage:
    # One PDF, full tagging (needs Gemini creds — see test_ingestion.md):
    PYTHONPATH=. python scripts/inspect_chunks.py corpus/APA_Boswell_Constantino_Deliberate_Practice_CBT.pdf

    # Chunking only, no API calls (fast, free, offline):
    PYTHONPATH=. python scripts/inspect_chunks.py corpus/somefile.pdf --no-metadata

    # Several PDFs, tag only the first 20 chunks of each (cheap test):
    PYTHONPATH=. python scripts/inspect_chunks.py corpus/a.pdf corpus/b.pdf --limit 20

    # Everything in corpus/:
    PYTHONPATH=. python scripts/inspect_chunks.py --all

    # See exactly what Gemini/Google is doing:
    PYTHONPATH=. python scripts/inspect_chunks.py corpus/a.pdf --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Imported first so its warnings filter is registered before the google.generativeai
# import (via the ingestion package) prints its end-of-life FutureWarning.
from scripts._gcp_logging import log_api_error, setup_logging

from config.settings import settings
from ingestion.chunker import ChunkerConfig, ContextAwareChunker
from ingestion.extractor import PDFExtractor
from models import Chunk, ChunkMetadata

logger = None  # set in main() via setup_logging

# Metadata fields worth summarizing / highlighting in the review report.
_TAG_FIELDS = [
    "domain", "doc_type", "therapeutic_modality", "clinical_presentation",
    "session_event_tags", "session_phase", "directionality", "applies_when",
    "risk_dimension_tags", "clinical_caution", "technique_tags",
    "patient_population", "clinical_measure_tags", "keywords",
]
# The safety-critical fields, called out separately in each chunk card.
_SAFETY_FIELDS = ["directionality", "applies_when", "clinical_caution", "risk_dimension_tags"]


def compute_doc_id(path: Path) -> str:
    """SHA-256 hex digest of the file bytes (matches uploader/extractor)."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def _build_chunker() -> ContextAwareChunker:
    """Chunker configured from config/chunk_config.yaml (falls back to defaults)."""
    cfg_path = settings.chunk_config_path
    try:
        config = ChunkerConfig.from_yaml(cfg_path)
        logger.info("Loaded chunk config from %s", cfg_path)
    except Exception as exc:  # missing/invalid yaml — defaults are fine for a preview
        logger.warning("Could not load %s (%s); using ChunkerConfig defaults.", cfg_path, exc)
        config = ChunkerConfig()
    return ContextAwareChunker(config)


def _generate_metadata(gen, chunk: Chunk, source_file: str) -> tuple[ChunkMetadata, str]:
    """Generate metadata for one chunk, surfacing (not swallowing) API failures.

    Mirrors MetadataGenerator.generate() but reports whether Gemini succeeded or
    the rule-based fallback was used, and logs the underlying Google error with an
    actionable hint so failures are debuggable.

    Returns:
        (metadata, status) where status is "ok" or "fallback".
    """
    try:
        prompt = gen._build_extraction_prompt(chunk, "")
        raw = gen._call_gemini(prompt)
        md = gen._validate_and_coerce(raw, chunk)
        status = "ok"
    except Exception as exc:  # noqa: BLE001 — we want every failure class here
        log_api_error(logger, exc, f"tagging chunk {chunk.chunk_id}")
        md = gen._fallback_extraction(chunk)
        status = "fallback"
    md.source_file = source_file
    return md, status


def _tag_counts(chunks: list[Chunk]) -> dict[str, Counter]:
    """Frequency of each tag value across chunks (for the summary tables)."""
    counts: dict[str, Counter] = {f: Counter() for f in _TAG_FIELDS}
    for c in chunks:
        if c.metadata is None:
            continue
        for f in _TAG_FIELDS:
            val = getattr(c.metadata, f, None)
            if isinstance(val, list):
                counts[f].update(val or ["(empty)"])
            elif val in (None, ""):
                counts[f].update(["(empty)"])
            else:
                counts[f].update([str(val)])
    return counts


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    """One readable JSON object per chunk (text kept as plain text, not base64)."""
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "parent_section": c.parent_section,
                "text": c.text,
                "metadata": c.metadata.model_dump() if c.metadata else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_review_md(
    path: Path, pdf: Path, doc_id: str, n_pages: int, chunks: list[Chunk],
    n_fallback: int, metadata_on: bool, preview_chars: int,
) -> None:
    """Human-readable Markdown report of chunks + their tags."""
    lines: list[str] = []
    lines.append(f"# Ingestion review — {pdf.name}\n")
    lines.append(f"- **doc_id:** `{doc_id}`")
    lines.append(f"- **pages extracted:** {n_pages}")
    lines.append(f"- **chunks produced:** {len(chunks)}")
    if metadata_on:
        ok = len(chunks) - n_fallback
        lines.append(f"- **tagging:** Gemini — {ok} ok / {n_fallback} used fallback")
        if n_fallback:
            lines.append("  - ⚠️ Fallback chunks have minimal tags. A high count usually means a "
                         "Gemini/credentials/quota problem — see the log output above.")
    else:
        lines.append("- **tagging:** skipped (--no-metadata). Tags below are empty by design.")
    lines.append("")

    if metadata_on:
        lines.append("## Tag frequencies (across all chunks)\n")
        lines.append("How often each tag value appears. Good for spotting systematic mislabeling.\n")
        counts = _tag_counts(chunks)
        for field in _TAG_FIELDS:
            pairs = counts[field].most_common()
            rendered = ", ".join(f"`{v}`×{n}" for v, n in pairs) or "—"
            lines.append(f"- **{field}:** {rendered}")
        lines.append("")

    lines.append("## Chunks\n")
    lines.append("Each card is one chunk. Check: is the split sensible, are the therapy/"
                 "presentation tags right, and was any stated caution captured?\n")
    for c in chunks:
        md = c.metadata
        lines.append(f"### {c.chunk_id}")
        page = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}–{c.page_end}"
        section = c.parent_section or "(no section header)"
        lines.append(f"*{page} · section: {section}*\n")
        if md:
            for f in ["domain", "doc_type", "therapeutic_modality", "clinical_presentation",
                      "session_event_tags", "session_phase", "technique_tags",
                      "patient_population", "clinical_measure_tags", "keywords"]:
                v = getattr(md, f, None)
                lines.append(f"- {f}: `{v}`")
            lines.append("- **safety flags:**")
            for f in _SAFETY_FIELDS:
                v = getattr(md, f, None)
                lines.append(f"  - {f}: `{v}`")
        preview = " ".join(c.text.split())[:preview_chars]
        ellipsis = "…" if len(c.text) > preview_chars else ""
        lines.append(f"\n> {preview}{ellipsis}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def inspect_file(pdf: Path, extractor, chunker, gen, outdir: Path,
                 metadata_on: bool, limit: int, preview_chars: int) -> dict:
    """Extract, chunk, optionally tag one PDF; write review files; return a summary."""
    logger.info("=== %s ===", pdf.name)
    doc_id = compute_doc_id(pdf)
    logger.info("doc_id: %s", doc_id)

    logger.info("Extracting text...")
    doc = extractor.extract(pdf)
    logger.info("Extracted %d page(s).", len(doc.pages))

    logger.info("Chunking...")
    chunks = chunker.chunk(doc)
    logger.info("Produced %d chunk(s).", len(chunks))

    n_fallback = 0
    if metadata_on:
        targets = chunks if limit <= 0 else chunks[:limit]
        logger.info("Tagging %d chunk(s) with Gemini%s...",
                    len(targets), "" if limit <= 0 else f" (--limit {limit})")
        for i, c in enumerate(targets, 1):
            md, status = _generate_metadata(gen, c, pdf.name)
            c.metadata = md
            if status == "fallback":
                n_fallback += 1
            if i % 10 == 0:
                logger.info("  ...%d/%d tagged", i, len(targets))
        chunks = targets  # only report the chunks we actually tagged

    stem = f"{pdf.stem}.{doc_id[:8]}"
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{stem}.chunks.jsonl"
    md_path = outdir / f"{stem}.review.md"
    _write_jsonl(jsonl_path, chunks)
    _write_review_md(md_path, pdf, doc_id, len(doc.pages), chunks,
                     n_fallback, metadata_on, preview_chars)
    logger.info("Wrote %s", jsonl_path)
    logger.info("Wrote %s", md_path)

    return {
        "file": pdf.name, "doc_id": doc_id, "pages": len(doc.pages),
        "chunks": len(chunks), "fallback": n_fallback,
        "review_md": str(md_path), "chunks_jsonl": str(jsonl_path),
    }


def main() -> None:
    global logger
    parser = argparse.ArgumentParser(
        description="Preview chunks and tags for PDFs locally (no upload/index).",
    )
    parser.add_argument("files", nargs="*", type=Path, help="PDF path(s) to inspect.")
    parser.add_argument("--all", action="store_true",
                        help="Inspect every *.pdf in the corpus dir (CORPUS_DIR / 'corpus').")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Skip Gemini tagging (no Google API calls, no cost).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Tag at most N chunks per PDF (0 = all). Cheap way to test.")
    parser.add_argument("--out", type=Path, default=Path("ingestion_review"),
                        help="Output directory for review files (default: ingestion_review/).")
    parser.add_argument("--preview-chars", type=int, default=320,
                        help="Characters of chunk text shown per card in the .md report.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging incl. Google/gRPC internals.")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    if args.all:
        pdfs = sorted(settings.corpus_dir.glob("*.pdf"))
    else:
        pdfs = list(args.files)
    if not pdfs:
        parser.error("No PDFs given. Pass one or more paths, or use --all.")

    missing = [p for p in pdfs if not p.exists()]
    if missing:
        parser.error("File(s) not found: " + ", ".join(str(m) for m in missing))

    extractor = PDFExtractor(use_document_ai=False)
    chunker = _build_chunker()

    gen = None
    if not args.no_metadata:
        # Imported lazily so --no-metadata never even loads the Gemini client.
        from ingestion.metadata_gen import MetadataGenerator
        gen = MetadataGenerator(
            model_name=settings.gemini_model_metadata,
            schema_path=settings.metadata_schema_path,  # active RTA schema (rta_v1.json)
        )
        logger.info("Metadata schema: %s", settings.metadata_schema_path)

    results = []
    for pdf in pdfs:
        try:
            results.append(inspect_file(
                pdf, extractor, chunker, gen, args.out,
                metadata_on=not args.no_metadata, limit=args.limit,
                preview_chars=args.preview_chars,
            ))
        except Exception as exc:  # noqa: BLE001
            log_api_error(logger, exc, f"inspecting {pdf.name}")

    # Console summary
    print("\n" + "=" * 68)
    print(f"Inspected {len(results)} PDF(s). Review files in: {args.out}/")
    print("=" * 68)
    for r in results:
        tag = f"{r['fallback']} fallback" if not args.no_metadata else "no tags"
        print(f"  {r['file']}")
        print(f"    doc_id : {r['doc_id'][:16]}...")
        print(f"    pages  : {r['pages']}   chunks: {r['chunks']}   ({tag})")
        print(f"    open   : {r['review_md']}")
    print("=" * 68)
    if not args.no_metadata and any(r["fallback"] for r in results):
        print("NOTE: some chunks used the fallback tagger. Scroll up for the Google API "
              "error(s) explaining why, or re-run with --verbose.")


if __name__ == "__main__":
    main()
