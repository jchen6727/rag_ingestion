#!/usr/bin/env python
"""Compare two ingestion-output .jsonl files chunk-by-chunk and flag any chunk
whose metadata tags differ between the two runs — e.g. the same passage
tagged with a different keyword list, modality, or directionality in one run
vs. the other. Typical uses: checking whether a prompt/schema/model change
(different commits — see chunk_archive/NOTES.md's run_000N <-> source-PDF
naming) shifted tags, or whether two independent ingestions of the same PDF
produced different tags at temperature=0 (see determinism.txt).

Accepts the same two input shapes as metadata_viz.py / metadata_jaccard.py,
auto-detected per line (no need to pre-convert either file):
  - the GCS-uploaded format: {"id": ..., "structData": {...tags...}}
  - the local checkpoint/review format: {"chunk_id": ..., "metadata": {...tags...}}
    (ingestion_checkpoints/<doc_id>.jsonl, or inspect_chunks.py's
    <name>.<docid8>.chunks.jsonl review file)
The two input files may even be in different formats from each other.

Chunks are matched by chunk_id, so both files should come from the same
chunking of the same source PDF(s) — chunk_id encodes doc_id (a hash of the
PDF bytes) + chunk_index, so a chunk_id that exists in only one file usually
means the two files cover different PDFs, or the chunker itself changed
between the two runs. Those are reported in the summary but are not (and
cannot be) diffed field-by-field.

Provenance fields (doc_id, source_file, page_start, page_end, chunk_index) are
always excluded from the diff — they're copied verbatim from the Chunk, never
produced by the LLM (see ingestion/metadata_gen.py _PROVENANCE_FIELDS), so a
"mismatch" there would only ever indicate the two chunk_ids don't actually
refer to the same passage, not a tagging difference.

Output, written next to the first input file unless --out-dir is given:
  diff_<label1>_<label2>.log    human-readable per-chunk diff + summary tables
  diff_<label1>_<label2>.html   plotly chart: per-tag frequency in file1 vs.
                                 file2, sorted by |difference|, top N

Pass any number of field names after the two files to restrict BOTH the
per-chunk diff and the frequency chart to just those fields — e.g. to look
only at the safety-relevant fields, or only at one field you suspect drifted:

    PYTHONPATH=. python scripts/compare_jsonl.py a.jsonl b.jsonl applies_when session_event_tags clinical_presentation

Omit field names to compare every non-provenance field present in either file.

Usage:
    PYTHONPATH=. python scripts/compare_jsonl.py run_0000.jsonl run_0000_rerun.jsonl
    PYTHONPATH=. python scripts/compare_jsonl.py chunk_archive/v0/run_0002.jsonl chunk_archive/v1/run_0002.jsonl
    PYTHONPATH=. python scripts/compare_jsonl.py a.jsonl b.jsonl domain directionality keywords
    PYTHONPATH=. python scripts/compare_jsonl.py a.jsonl b.jsonl --top-n 15 --out-dir diffs/
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import json

import plotly.graph_objects as go

# Copied from the chunk, never produced by the LLM — comparing these would
# only ever flag "these two chunk_ids aren't the same passage", not a
# tagging difference. Kept in sync conceptually with
# ingestion/metadata_gen.py::_PROVENANCE_FIELDS (+ source_file, which is
# filename-derived rather than LLM-produced too).
PROVENANCE_FIELDS = {"doc_id", "source_file", "page_start", "page_end", "chunk_index"}

_MISSING = object()  # sentinel: field absent from a record entirely


# ---------------------------------------------------------------------------
# Loading (same dual-format auto-detection as metadata_viz.py / metadata_jaccard.py)
# ---------------------------------------------------------------------------


def _extract(obj: dict, path: Path, line_no: int) -> tuple[str, dict] | None:
    """Return (chunk_id, tags_dict) for one parsed .jsonl line, or None if
    the line is a recognized shape but carries no tags yet (e.g. a
    checkpoint entry written before Gemini extraction completed)."""
    if "structData" in obj:
        chunk_id = obj.get("id")
        tags = obj["structData"]
    elif "metadata" in obj:
        chunk_id = obj.get("chunk_id")
        tags = obj["metadata"]
    else:
        raise ValueError(
            f"{path} line {line_no}: recognized neither the GCS-upload format "
            f"(expected an 'structData' key) nor the local checkpoint/review "
            f"format (expected a 'metadata' key). Got keys: {sorted(obj.keys())}"
        )
    if tags is None or chunk_id is None:
        return None
    return chunk_id, tags


def load_records(path: Path) -> dict[str, dict]:
    """Return {chunk_id: tags_dict}, last line wins if a chunk_id repeats."""
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = _extract(json.loads(line), path, line_no)
            if parsed is not None:
                chunk_id, tags = parsed
                records[chunk_id] = tags
    return records


# ---------------------------------------------------------------------------
# Field-level comparison
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> Any:
    """Order-insensitive comparison key for a field value: a list is a
    passage's set of tags for that field, so ['CBT', 'DBT'] and ['DBT', 'CBT']
    are the same tagging, not a mismatch."""
    if isinstance(value, list):
        try:
            return tuple(sorted(value))
        except TypeError:
            return tuple(sorted(value, key=str))
    return value


def _as_values(value: Any) -> list[str]:
    """Flatten a field value into the list of tag-value strings it
    contributes to frequency counting. Scalars count as a single value;
    empty/placeholder scalars contribute nothing."""
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "", "none", "not_specified")]
    if value in (None, "", "none", "not_specified"):
        return []
    return [str(value)]


def comparison_fields(rec1: dict, rec2: dict, only: set[str] | None) -> list[str]:
    fields = (set(rec1) | set(rec2)) - PROVENANCE_FIELDS
    if only:
        fields &= only
    return sorted(fields)


def diff_chunk(rec1: dict, rec2: dict, fields: list[str]) -> list[tuple[str, Any, Any]]:
    """Return [(field, value1, value2), ...] for every field that differs."""
    diffs = []
    for field in fields:
        v1 = rec1.get(field, _MISSING)
        v2 = rec2.get(field, _MISSING)
        if _normalize(v1) != _normalize(v2):
            diffs.append((field, v1, v2))
    return diffs


# ---------------------------------------------------------------------------
# Report + chart
# ---------------------------------------------------------------------------


def unknown_requested_fields(
    recs1: dict[str, dict], recs2: dict[str, dict], only_fields: set[str] | None
) -> set[str]:
    """Field names explicitly requested via CLI that don't appear in ANY
    record of either file — almost always a typo. Flagging this matters
    because an unknown field silently produces zero diffs and an empty
    chart, which otherwise looks identical to "no mismatches found"."""
    if not only_fields:
        return set()
    known: set[str] = set()
    for rec in recs1.values():
        known |= set(rec)
    for rec in recs2.values():
        known |= set(rec)
    return only_fields - known


def build_report(
    recs1: dict[str, dict],
    recs2: dict[str, dict],
    label1: str,
    label2: str,
    only_fields: set[str] | None,
    unknown_fields: set[str] | None = None,
) -> tuple[str, Counter, Counter]:
    """Returns (log_text, freq1, freq2) where freq1/freq2 are Counters keyed
    by "field:value" across ALL common chunks (not just mismatching ones) —
    the raw material for the frequency-delta chart."""
    common = sorted(set(recs1) & set(recs2))
    only_in_1 = sorted(set(recs1) - set(recs2))
    only_in_2 = sorted(set(recs2) - set(recs1))

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Chunk metadata diff: {label1}  vs.  {label2}")
    lines.append("=" * 72)
    lines.append(
        f"Fields compared: {', '.join(sorted(only_fields))}"
        if only_fields
        else "Fields compared: ALL (every field present in either file, minus "
             "provenance fields)"
    )
    if unknown_fields:
        lines.append(
            f"WARNING: requested field(s) not found in either file (typo?): "
            f"{', '.join(sorted(unknown_fields))}"
        )
    lines.append(f"{label1}: {len(recs1)} chunk(s)")
    lines.append(f"{label2}: {len(recs2)} chunk(s)")
    lines.append(f"Common chunk_id(s): {len(common)}")
    lines.append(f"Only in {label1}: {len(only_in_1)}")
    lines.append(f"Only in {label2}: {len(only_in_2)}")
    if only_in_1:
        lines.append(f"  e.g. {only_in_1[:5]}{' ...' if len(only_in_1) > 5 else ''}")
    if only_in_2:
        lines.append(f"  e.g. {only_in_2[:5]}{' ...' if len(only_in_2) > 5 else ''}")
    lines.append("")

    if not common:
        lines.append("No common chunk_id(s) — nothing to diff. Are these two "
                      "files from the same source PDF(s)?")
        return "\n".join(lines), Counter(), Counter()

    field_mismatch_counts: Counter = Counter()
    per_chunk_diffs: dict[str, list[tuple[str, Any, Any]]] = {}
    freq1: Counter = Counter()
    freq2: Counter = Counter()

    for chunk_id in common:
        rec1, rec2 = recs1[chunk_id], recs2[chunk_id]
        fields = comparison_fields(rec1, rec2, only_fields)
        for field in fields:
            for v in _as_values(rec1.get(field)):
                freq1[f"{field}:{v}"] += 1
            for v in _as_values(rec2.get(field)):
                freq2[f"{field}:{v}"] += 1
        diffs = diff_chunk(rec1, rec2, fields)
        if diffs:
            per_chunk_diffs[chunk_id] = diffs
            for field, _v1, _v2 in diffs:
                field_mismatch_counts[field] += 1

    n_mismatched = len(per_chunk_diffs)
    lines.append(
        f"Chunks with at least one differing tag: {n_mismatched}/{len(common)} "
        f"({100 * n_mismatched / len(common):.1f}%)"
    )
    lines.append("")

    if field_mismatch_counts:
        lines.append("-" * 72)
        lines.append("Mismatch count by field (out of chunks compared on that field)")
        lines.append("-" * 72)
        width = max(len(f) for f in field_mismatch_counts)
        for field, count in field_mismatch_counts.most_common():
            lines.append(f"  {field:<{width}} {count:>6}/{len(common)}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("Per-chunk diffs")
    lines.append("-" * 72)
    if not per_chunk_diffs:
        lines.append("(none — every compared field matched on every common chunk)")
    for chunk_id in sorted(per_chunk_diffs):
        lines.append(f"\n{chunk_id}")
        for field, v1, v2 in per_chunk_diffs[chunk_id]:
            v1_s = "<missing>" if v1 is _MISSING else repr(v1)
            v2_s = "<missing>" if v2 is _MISSING else repr(v2)
            lines.append(f"  {field}:")
            lines.append(f"    {label1}: {v1_s}")
            lines.append(f"    {label2}: {v2_s}")

    return "\n".join(lines), freq1, freq2


def build_figure(freq1: Counter, freq2: Counter, label1: str, label2: str, top_n: int) -> go.Figure:
    all_tags = set(freq1) | set(freq2)
    deltas = {tag: freq1.get(tag, 0) - freq2.get(tag, 0) for tag in all_tags}
    ranked = sorted(all_tags, key=lambda t: abs(deltas[t]), reverse=True)[:top_n]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ranked, y=[freq1.get(t, 0) for t in ranked], name=label1,
    ))
    fig.add_trace(go.Bar(
        x=ranked, y=[freq2.get(t, 0) for t in ranked], name=label2,
    ))
    fig.update_layout(
        barmode="group",
        title_text=f"Tag frequency, {label1} vs. {label2} — top {len(ranked)} by |difference|",
        xaxis_title="field:value",
        yaxis_title="chunk count",
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=45)
    return fig


_MAX_FIELDS_SUFFIX_CHARS = 120


def _fields_filename_suffix(only_fields: set[str] | None) -> str:
    """"_fields-a-b-c" for the output filename, kept readable: field names are
    listed in full as long as that stays a reasonable filename length; past
    that, list as many as fit and note how many were cut off, rather than
    collapsing the whole list to an opaque count."""
    if not only_fields:
        return ""
    names = sorted(only_fields)
    full = "_fields-" + "-".join(names)
    if len(full) <= _MAX_FIELDS_SUFFIX_CHARS:
        return full
    kept: list[str] = []
    for name in names:
        candidate = "_fields-" + "-".join([*kept, name])
        if len(candidate) > _MAX_FIELDS_SUFFIX_CHARS:
            break
        kept.append(name)
    omitted = len(names) - len(kept)
    return "_fields-" + "-".join(kept) + (f"-+{omitted}more" if omitted else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file1", type=Path, help="First .jsonl file.")
    parser.add_argument("file2", type=Path, help="Second .jsonl file.")
    parser.add_argument(
        "fields", nargs="*", default=None,
        help="Optional field name(s) to restrict the diff and frequency chart "
             "to (e.g. applies_when session_event_tags clinical_presentation). "
             "Default: every field present in either file, minus provenance "
             "fields.",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Max field:value tags to show in the frequency-delta chart. Default: 20.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory to write diff_*.log/.html into. Default: file1's directory.",
    )
    parser.add_argument(
        "--label1", default=None, help="Override the name used for file1 in output "
        "(default: file1's stem). Useful if both files share a stem.",
    )
    parser.add_argument(
        "--label2", default=None, help="Override the name used for file2 in output.",
    )
    args = parser.parse_args()

    if not args.file1.exists():
        parser.error(f"File not found: {args.file1}")
    if not args.file2.exists():
        parser.error(f"File not found: {args.file2}")

    label1 = args.label1 or args.file1.stem
    label2 = args.label2 or args.file2.stem
    only_fields = set(args.fields) if args.fields else None

    recs1 = load_records(args.file1)
    recs2 = load_records(args.file2)
    if not recs1:
        raise SystemExit(f"No records found in {args.file1}")
    if not recs2:
        raise SystemExit(f"No records found in {args.file2}")

    unknown_fields = unknown_requested_fields(recs1, recs2, only_fields)
    if unknown_fields:
        print(
            f"WARNING: field(s) not found in either file (typo?): "
            f"{', '.join(sorted(unknown_fields))}"
        )

    log_text, freq1, freq2 = build_report(
        recs1, recs2, label1, label2, only_fields, unknown_fields
    )

    out_dir = args.out_dir or args.file1.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _fields_filename_suffix(only_fields)
    log_path = out_dir / f"diff_{label1}_{label2}{suffix}.log"
    html_path = out_dir / f"diff_{label1}_{label2}{suffix}.html"

    log_path.write_text(log_text, encoding="utf-8")
    print(f"Wrote {log_path}")

    if freq1 or freq2:
        fig = build_figure(freq1, freq2, label1, label2, args.top_n)
        fig.write_html(str(html_path))
        print(f"Wrote {html_path}")
    else:
        print("No common chunks with comparable tags — skipped the frequency chart.")


if __name__ == "__main__":
    main()
