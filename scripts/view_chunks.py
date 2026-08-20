#!/usr/bin/env python
"""View decoded chunk text filtered by metadata tags.

Filters are AND-combined across --filter args. A filter matches a chunk if:
  - the field is scalar and equals the given value, or
  - the field is a list and contains the given value.

Usage:
    python view_chunks.py boswell_chunks.jsonl --filter therapeutic_modality=CBT
    python view_chunks.py boswell_chunks.jsonl --filter domain=psychotherapy_general --filter session_phase=any
    python view_chunks.py boswell_chunks.jsonl --filter keywords=deliberate practice --keyword-contains "series editors"
    python view_chunks.py boswell_chunks.jsonl --filter doc_type=other --limit 3 --full
    python view_chunks.py boswell_chunks.jsonl --filter therapeutic_modality=CBT --output matches.txt
"""

import argparse
import base64
import json
from pathlib import Path

TRUNCATE_CHARS = 600


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def decode_content(record: dict) -> str:
    raw = record.get("content", {}).get("rawBytes", "")
    if not raw:
        return ""
    return base64.b64decode(raw).decode("utf-8", errors="replace")


def parse_filters(filter_args: list[str]) -> list[tuple[str, str]]:
    filters = []
    for f in filter_args:
        if "=" not in f:
            raise SystemExit(f"Invalid --filter '{f}', expected field=value")
        field, value = f.split("=", 1)
        filters.append((field.strip(), value.strip()))
    return filters


def matches(struct_data: dict, filters: list[tuple[str, str]], keyword_contains: str | None) -> bool:
    for field, value in filters:
        actual = struct_data.get(field)
        if actual is None:
            return False
        if isinstance(actual, list):
            if value not in actual:
                return False
        else:
            if str(actual) != value:
                return False

    if keyword_contains:
        keywords = struct_data.get("keywords", []) or []
        needle = keyword_contains.lower()
        if not any(needle in kw.lower() for kw in keywords):
            return False

    return True


def format_match(record: dict, full: bool) -> str:
    sd = record["structData"]
    text = decode_content(record)
    if not full and len(text) > TRUNCATE_CHARS:
        text = text[:TRUNCATE_CHARS] + "... [truncated, use --full]"

    lines = [
        "=" * 80,
        f"chunk_id: {record.get('id')}",
        f"source_file: {sd.get('source_file')}  chapter: {sd.get('chapter')}",
        f"pages: {sd.get('page_start')}-{sd.get('page_end')}  chunk_index: {sd.get('chunk_index')}",
        f"domain: {sd.get('domain')}  doc_type: {sd.get('doc_type')}  "
        f"modality: {sd.get('therapeutic_modality')}",
        f"session_event_tags: {sd.get('session_event_tags')}  "
        f"directionality: {sd.get('directionality')}",
        f"keywords: {sd.get('keywords')}",
        "-" * 80,
        text,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("jsonl_path", type=Path, help="Path to the chunks .jsonl file")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Metadata filter, repeatable. AND-combined. Scalar fields match by "
        "equality, list fields match by membership.",
    )
    parser.add_argument(
        "--keyword-contains",
        default=None,
        help="Case-insensitive substring match against the keywords list.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max chunks to show. Default: 10.")
    parser.add_argument("--full", action="store_true", help="Show full chunk text, not truncated.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write results to this file instead of stdout."
    )
    args = parser.parse_args()

    filters = parse_filters(args.filter)
    if not filters and not args.keyword_contains:
        raise SystemExit("Provide at least one --filter or --keyword-contains.")

    records = load_records(args.jsonl_path)
    if not records:
        raise SystemExit(f"No records found in {args.jsonl_path}")

    hits = [r for r in records if matches(r["structData"], filters, args.keyword_contains)]

    out_lines = [f"Matched {len(hits)} chunk(s) (showing up to {args.limit}):\n"]
    for record in hits[: args.limit]:
        out_lines.append(format_match(record, args.full))

    output_text = "\n".join(out_lines)

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(hits)} match(es) to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
