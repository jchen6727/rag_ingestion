#!/usr/bin/env python
"""Visualize metadata tag frequency from a chunked-corpus .jsonl file.

Accepts two input formats, auto-detected per line (no need to pre-convert):
  - the GCS-uploaded chunks .jsonl ({"id": ..., "structData": {...tags...}}),
    produced by ingestion/uploader.py once a doc has actually been uploaded.
  - the local, pre-upload .jsonl written by scripts/batch_ingest.py's
    checkpoint (ingestion_checkpoints/<doc_id>.jsonl, or a renamed copy of
    one) or by scripts/inspect_chunks.py's review .jsonl — both shaped as
    {"chunk_id": ..., "metadata": {...tags...}}. This is what you have
    immediately after ingestion/dry-run, before (or without ever) uploading.

Usage:
    conda run -n rag python metadata_viz.py boswell_chunks.jsonl
    conda run -n rag python metadata_viz.py boswell_chunks.jsonl --fields domain doc_type
    conda run -n rag python metadata_viz.py boswell_chunks.jsonl --output out.html
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


def count_field(records: list[dict], field: str) -> Counter:
    counter = Counter()
    for rec in records:
        value = rec.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            counter.update(v for v in value if v not in ("", "none", "not_specified"))
        else:
            counter[value] += 1
    return counter


def build_figure(records: list[dict], fields: list[str]) -> go.Figure:
    counts = {field: count_field(records, field) for field in fields}
    counts = {field: c for field, c in counts.items() if c}

    if not counts:
        raise ValueError("No non-empty values found for the requested fields.")

    n = len(counts)
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=[f"{field} (n={sum(counts[field].values())})" for field in counts],
        vertical_spacing=min(0.15, 1 / (n * 2)),
    )

    for i, (field, counter) in enumerate(counts.items(), start=1):
        items = counter.most_common()
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        fig.add_trace(
            go.Bar(x=labels, y=values, name=field, showlegend=False),
            row=i,
            col=1,
        )
        fig.update_xaxes(tickangle=45, row=i, col=1)

    fig.update_layout(
        height=max(400, 350 * n),
        title_text="Metadata Tag Frequency",
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
            "Metadata fields to aggregate and plot. "
            f"Defaults to: {' '.join(DEFAULT_FIELDS)}"
        ),
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

    fig = build_figure(records, fields)

    if args.output:
        fig.write_html(str(args.output))
        print(f"Wrote {args.output}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
