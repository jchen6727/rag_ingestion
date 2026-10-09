#!/usr/bin/env python
"""Visualize within-field tag co-occurrence via pairwise Jaccard similarity.

For a given metadata field (e.g. therapeutic_modality), this computes, for
every pair of distinct tag values in that field, the Jaccard similarity
between the sets of chunks each value appears on:

    J(a, b) = |chunks with a AND b| / |chunks with a OR b|

Scalar (non-list) fields will always show 0 off-diagonal, since a chunk can
only hold one value — this correctly reflects mutual exclusivity.

Accepts two input formats, auto-detected per line (no need to pre-convert):
  - the GCS-uploaded chunks .jsonl ({"id": ..., "structData": {...tags...}}),
    produced by ingestion/uploader.py once a doc has actually been uploaded.
  - the local, pre-upload .jsonl written by scripts/batch_ingest.py's
    checkpoint (ingestion_checkpoints/<doc_id>.jsonl, or a renamed copy of
    one) or by scripts/inspect_chunks.py's review .jsonl — both shaped as
    {"chunk_id": ..., "metadata": {...tags...}}. This is what you have
    immediately after ingestion/dry-run, before (or without ever) uploading.

Usage:
    conda run -n rag python metadata_jaccard.py boswell_chunks.jsonl
    conda run -n rag python metadata_jaccard.py boswell_chunks.jsonl --fields domain doc_type
    conda run -n rag python metadata_jaccard.py boswell_chunks.jsonl --top-n 15 --output out.html
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_FIELDS = [
    "therapeutic_modality",
    "clinical_presentation",
    "session_event_tags",
    "directionality",
]

PLACEHOLDER_VALUES = {"", "none", "not_specified"}


def _extract_tags(obj: dict, path: Path, line_no: int) -> dict | None:
    """Pull the flat tag dict out of one parsed .jsonl line, regardless of
    which of the two shapes described in the module docstring it is.

    Returns None for a line that is a recognized shape but carries no tags
    (e.g. a checkpoint entry written before Gemini extraction completed) —
    callers should skip those rather than treat them as malformed.
    """
    if "structData" in obj:
        return obj["structData"]
    if "metadata" in obj:
        return obj["metadata"]  # may be None; caller skips
    raise ValueError(
        f"{path} line {line_no}: recognized neither the GCS-upload format "
        f"(expected an 'structData' key) nor the local checkpoint/review "
        f"format (expected a 'metadata' key). Got keys: {sorted(obj.keys())}"
    )


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tags = _extract_tags(json.loads(line), path, line_no)
            if tags is not None:
                records.append(tags)
    return records


def value_sets_per_chunk(records: list[dict], field: str) -> list[set]:
    sets = []
    for rec in records:
        value = rec.get(field)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        values = {v for v in values if v not in PLACEHOLDER_VALUES}
        if values:
            sets.append(values)
    return sets


def jaccard_matrix(sets: list[set], top_n: int) -> tuple[list[str], list[list[float]]]:
    freq = Counter()
    for s in sets:
        freq.update(s)
    labels = [v for v, _ in freq.most_common(top_n)]

    chunks_containing = {v: {i for i, s in enumerate(sets) if v in s} for v in labels}

    n = len(labels)
    matrix = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(labels):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            b = labels[j]
            a_set, b_set = chunks_containing[a], chunks_containing[b]
            union = a_set | b_set
            score = len(a_set & b_set) / len(union) if union else 0.0
            matrix[i][j] = matrix[j][i] = score
    return labels, matrix


def build_figure(records: list[dict], fields: list[str], top_n: int) -> go.Figure:
    per_field = {}
    for field in fields:
        sets = value_sets_per_chunk(records, field)
        if not sets:
            continue
        labels, matrix = jaccard_matrix(sets, top_n)
        if len(labels) < 2:
            continue
        per_field[field] = (labels, matrix)

    if not per_field:
        raise ValueError("No field had >=2 distinct values to compare.")

    n = len(per_field)
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=[f"{field} — pairwise Jaccard" for field in per_field],
        vertical_spacing=min(0.15, 1 / (n * 2)),
    )

    for i, (field, (labels, matrix)) in enumerate(per_field.items(), start=1):
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                showscale=(i == 1),
            ),
            row=i,
            col=1,
        )
        fig.update_xaxes(tickangle=45, row=i, col=1)

    fig.update_layout(
        height=max(450, 450 * n),
        title_text="Metadata Tag Jaccard Similarity",
        template="plotly_white",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path, help="Path to the chunks .jsonl file")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help=(
            "Metadata fields to compute co-occurrence Jaccard for. "
            f"Defaults to: {' '.join(DEFAULT_FIELDS)}"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Max distinct tag values per field to include (most frequent first). Default: 20.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write figure to this HTML file instead of opening a browser window.",
    )
    args = parser.parse_args()

    fields = args.fields if args.fields else DEFAULT_FIELDS

    records = load_records(args.jsonl_path)
    if not records:
        raise SystemExit(f"No records found in {args.jsonl_path}")

    fig = build_figure(records, fields, args.top_n)

    if args.output:
        fig.write_html(str(args.output))
        print(f"Wrote {args.output}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
