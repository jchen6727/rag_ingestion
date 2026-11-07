#!/usr/bin/env python
"""
Parameter sweep over rag_check/compare_latency.py's filtered-vs-full-corpus
latency comparison.

Runs `compare_latency.run_comparison()` once per row of an input table (CSV
file or an in-memory pandas DataFrame), where each row supplies some subset
of the comparison's arguments as labeled columns. Any argument whose column
is absent from the table, or blank/NaN for a given row, falls back to
compare_latency.py's own default for that argument — a sweep CSV only needs
to specify the columns it's actually varying.

Two entry points:
  - `run_sweep_from_csv(csv_path, out_csv=...)` — CLI / file-based sweeps.
  - `run_sweep(df, out_csv=...)` — the method wrapper for a sweep built
    programmatically (e.g. `itertools.product` over a few params) as a
    pandas DataFrame instead of a file. `run_sweep_from_csv` is a two-line
    wrapper around this: `pd.read_csv(csv_path)` then `run_sweep(df, ...)`.

Output is a long-format DataFrame/CSV: one row per (input row x arm actually
run for that row), with every input column echoed back under an `input_`
prefix plus the measured outputs (num_chunks, retrieval/generation/total
latency, token usage, answer text). See rag_check/README.md for the full
column reference and two worked examples (CSV and DataFrame).

Usage:
    PYTHONPATH=. python rag_check/param_sweep.py \\
        --input-csv rag_check/examples/sweep_example.csv \\
        --out sweep_results.csv

    PYTHONPATH=. python rag_check/param_sweep.py \\
        --input-csv rag_check/examples/sweep_example.csv \\
        --out sweep_results.csv --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from rag_check.compare_latency import run_comparison  # noqa: E402
from rag_check.llm_query import GenAIParams  # noqa: E402
from rag_check.retrieval import FilterFieldError, build_filter_expression  # noqa: E402
from scripts._gcp_logging import log_api_error, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Recognized input columns. Every one is optional except "prompt"; a missing
# column or a blank/NaN cell falls back to the same default
# compare_latency.py's CLI uses for that argument.
_BOOL_TRUE = frozenset({"1", "true", "yes", "y", "t"})


def _cell(row: "pd.Series", col: str, default: Any) -> Any:
    """Read `row[col]`, falling back to `default` if the column is absent,
    NaN, or (for strings) blank."""
    if col not in row.index:
        return default
    val = row[col]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, str) and val.strip() == "":
        return default
    return val


def _as_bool(val: Any) -> bool:
    """Coerce a cell value already passed through _cell() to bool.

    `val` is either the Python bool default _cell() returned (column absent
    or blank) or an actual cell value (str/bool/int) — either way this
    normalizes it, so the caller never needs to special-case "was it the
    default".
    """
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in _BOOL_TRUE


def _as_optional_int(val: Any) -> Optional[int]:
    return None if val is None else int(val)


def _as_optional_float(val: Any) -> Optional[float]:
    return None if val is None else float(val)


def parse_tags_cell(val: Any) -> list[tuple[str, str]]:
    """Parse a "field=value;field=value" cell into (field, value) pairs.

    Mirrors compare_latency.py's repeated `--tag field=value`, but as a
    single semicolon-delimited cell since a CSV row can't repeat a column.
    """
    if val is None:
        return []
    text = str(val).strip()
    if not text:
        return []
    pairs = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid tags entry '{part}', expected field=value")
        field_name, _, value = part.partition("=")
        pairs.append((field_name.strip(), value.strip()))
    return pairs


def _row_to_run_kwargs(row: "pd.Series") -> dict:
    """Translate one input row into run_comparison() kwargs, applying
    defaults for every column that is absent or blank for this row."""
    prompt = _cell(row, "prompt", None)
    if prompt is None:
        raise ValueError("row is missing a 'prompt' value (required, no default)")
    prompt = str(prompt)

    tag_pairs = parse_tags_cell(_cell(row, "tags", None))
    # Validated eagerly (not just inside run_comparison) so a bad field name
    # in one row fails with a clear per-row message instead of a traceback
    # from deep inside retrieval.search_filtered.
    build_filter_expression(tag_pairs)

    model = _cell(row, "model", None)
    model = str(model) if model is not None else settings.gemini_model_generation

    genai_params = GenAIParams(
        model=model,
        temperature=float(_cell(row, "temperature", 0.0)),
        top_p=_as_optional_float(_cell(row, "top_p", None)),
        top_k=_as_optional_int(_cell(row, "sampling_top_k", None)),
        seed=_as_optional_int(_cell(row, "seed", None)),
        max_output_tokens=_as_optional_int(_cell(row, "max_output_tokens", None)),
        thinking_budget=_as_optional_int(_cell(row, "thinking_budget", None)),
    )

    return dict(
        prompt=prompt,
        search_query=_cell(row, "search_query", None),
        tag_pairs=tag_pairs,
        top_k=int(_cell(row, "top_k", 10)),
        max_corpus_chunks=_as_optional_int(_cell(row, "max_corpus_chunks", None)),
        skip_filtered=_as_bool(_cell(row, "skip_filtered", False)),
        skip_full_corpus=_as_bool(_cell(row, "skip_full_corpus", False)),
        genai_params=genai_params,
    )


def _flatten(row_id: Any, row: "pd.Series", record: dict) -> dict:
    """Combine one input row + one arm's result record into one output row."""
    out = {f"input_{col}": row[col] for col in row.index}
    out["row_id"] = row_id
    usage = record.get("usage") or {}
    out.update({
        "arm": record.get("arm"),
        "num_chunks": record.get("num_chunks"),
        "prompt_chars": record.get("prompt_chars"),
        "retrieval_latency_seconds": record.get("retrieval_latency_seconds"),
        "generation_latency_seconds": record.get("generation_latency_seconds"),
        "total_latency_seconds": record.get("total_latency_seconds"),
        "prompt_token_count": usage.get("prompt_token_count"),
        "candidates_token_count": usage.get("candidates_token_count"),
        "thoughts_token_count": usage.get("thoughts_token_count"),
        "total_token_count": usage.get("total_token_count"),
        "chunk_ids": record.get("chunk_ids"),
        "answer_text": record.get("answer_text"),
        "error": record.get("error"),
    })
    return out


def run_sweep(
    df: "pd.DataFrame",
    out_csv: Optional[Path] = None,
    verbose: bool = False,
) -> "pd.DataFrame":
    """Run the filtered-vs-full-corpus latency comparison once per row of `df`.

    This is the method wrapper for programmatic sweeps: build any DataFrame
    (e.g. `itertools.product` over a few params — see
    rag_check/README.md for a worked example) and pass it directly, instead
    of going through a CSV file.

    Recognized columns — every one optional except `prompt`; a missing
    column, or a blank/NaN cell within a present column, falls back to
    compare_latency.py's own default for that argument:

        prompt (required, str)        — the question sent to the LLM.
        search_query (str)            — filtered-arm search text; defaults to `prompt`.
        tags (str)                    — "field=value;field=value" (semicolons;
                                         compare_latency.py's CLI --tag is repeatable
                                         instead, since a CSV row can't repeat a column).
        top_k (int)                   — filtered-arm retrieval count. Default 10.
        max_corpus_chunks (int)       — cap on the full_corpus arm. Default: no cap.
        skip_filtered (bool-ish)      — "true"/"1"/"yes" etc. Default False.
        skip_full_corpus (bool-ish)   — same. Not both True in the same row.
        model (str)                   — default settings.gemini_model_generation.
        temperature (float)           — default 0.0.
        top_p (float)                 — sampling top_p. Default: model default.
        sampling_top_k (int)          — sampling top_k (distinct from retrieval `top_k`).
        seed (int)                    — default: model default (unseeded).
        max_output_tokens (int)       — default: model default.
        thinking_budget (int)         — default: model default. 0 disables thinking
                                         on models that support it.

    Args:
        df: One row per comparison run. Must have a `prompt` column.
        out_csv: If given, write the result DataFrame here as CSV.
        verbose: DEBUG logging, including Discovery Engine/Gemini request detail.

    Returns:
        A long-format DataFrame: one row per (input row x arm actually run
        for that row). Every input column is echoed back under an `input_`
        prefix; a row whose run raised an exception gets a single output
        row with `arm="error"` and the exception message in `error` rather
        than halting the whole sweep.
    """
    global logger
    logger = setup_logging(verbose)
    settings.validate_all()

    if "prompt" not in df.columns:
        raise ValueError("Input table must have a 'prompt' column.")

    output_rows: list[dict] = []
    for row_id, row in df.iterrows():
        try:
            kwargs = _row_to_run_kwargs(row)
        except (ValueError, FilterFieldError) as exc:
            logger.error("Row %s: invalid input, skipping: %s", row_id, exc)
            output_rows.append(_flatten(row_id, row, {"arm": "error", "error": str(exc)}))
            continue

        logger.info(
            "Row %s: prompt=%r tags=%s model=%s",
            row_id, kwargs["prompt"][:60], kwargs["tag_pairs"], kwargs["genai_params"].model,
        )
        try:
            records = run_comparison(**kwargs)
        except Exception as exc:  # noqa: BLE001 — keep sweeping past a bad row
            log_api_error(logger, exc, f"running sweep row {row_id}")
            output_rows.append(_flatten(row_id, row, {"arm": "error", "error": str(exc)}))
            continue

        for record in records:
            output_rows.append(_flatten(row_id, row, record))

    result_df = pd.DataFrame(output_rows).drop(columns=["chunk_ids"]) # remove chunk_ids
    if out_csv is not None:
        result_df.to_csv(out_csv, index=False)
        logger.info("Wrote %s (%d rows)", out_csv, len(result_df))
    return result_df


def run_sweep_from_csv(
    csv_path: Path,
    out_csv: Optional[Path] = None,
    verbose: bool = False,
) -> "pd.DataFrame":
    """Load `csv_path` and run_sweep() it. See run_sweep() for column reference."""
    df = pd.read_csv(csv_path)
    return run_sweep(df, out_csv=out_csv, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rag_check's filtered-vs-full-corpus latency comparison "
                    "once per row of an input CSV, collecting inputs+outputs into a CSV.",
    )
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="CSV with one comparison run per row — see run_sweep()'s "
                             "docstring or rag_check/README.md for the column reference.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Where to write the result CSV.")
    parser.add_argument("--verbose", action="store_true", help="Verbose/DEBUG logging.")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")
    # removing chunk_ids, as full corpus search returns too many tags
    result_df = run_sweep_from_csv(args.input_csv, out_csv=args.out, verbose=args.verbose)

    print(f"\nRan {len(result_df)} result row(s) from {args.input_csv} -> {args.out}")
    if not result_df.empty:
        print(result_df[["arm", "num_chunks", "retrieval_latency_seconds",
                          "generation_latency_seconds", "total_latency_seconds"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
