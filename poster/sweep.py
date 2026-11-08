#!/usr/bin/env python
"""
Parameter sweep for the poster's filtered-RAG-vs-whole-corpus-RAG figure.

Expands the 3 seed cases in `poster/sweep_example.csv` (direct disclosure ->
implicit disclosure -> implicit disclosure + JSON structure) into a 9-tier
complexity ladder on the SAME clinical topic (`poster/prompts.py`), escalating
one dimension at a time: how directly the text discloses risk, how much
distractor narrative surrounds it, and how strict the requested JSON
structure is. Each tier runs `rag_check.param_sweep.run_sweep()`'s filtered
vs. full_corpus comparison, repeated 5x per tier (varying `seed` only, to
sample latency variance under otherwise-identical conditions):

    9 tiers x 5 repeats = 45 rows x 2 arms (filtered + full_corpus) = 90 LLM
    queries total — under the 100-query budget.

This calls the real Vertex AI Search DataStore and Gemini model for every
query (same auth/config as the rest of rag_ingestion — see rag_check/README.md
Prerequisites). With gemini-2.5-flash, expect on the order of 20-40 minutes
end to end, dominated by the full_corpus arm sending the entire corpus
(hundreds of chunks) as context on every one of its 45 calls. Use --dry-run
first to sanity-check the planned sweep without spending any quota.

Usage:
    PYTHONPATH=. python poster/sweep.py --dry-run
    PYTHONPATH=. python poster/sweep.py --out poster/sweep_results.csv
    PYTHONPATH=. python poster/sweep.py --out poster/sweep_results.csv --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from poster.prompts import (  # noqa: E402
    BASE_SEED,
    MODEL,
    N_REPEATS,
    TAG_FILTER,
    TEMPERATURE,
    TIERS,
    TOP_K,
)
from rag_check.param_sweep import run_sweep  # noqa: E402

N_ARMS = 2  # filtered + full_corpus, run_comparison()'s default

# Discovery Engine's SearchService parses literal straight/curly double quotes
# in the query text as an exact-phrase operator (confirmed live — see
# poster/trials/trial_1.md "Investigation: zero chunks retrieved on the
# hard-failing tiers"). Several tiers' prompts quote an invented client
# statement for readability in the LLM-facing prompt (e.g. `states "If a car
# hit me today, I wouldn't really care"?`) — that invented quote doesn't
# appear verbatim anywhere in the corpus, so using the raw prompt as the
# Discovery Engine *search query* silently retrieved zero chunks regardless
# of how well the tag filter matched. The fix is query-side only: strip quote
# characters from the text used for ranking, while leaving `prompt` (what the
# LLM actually sees) untouched.
_QUOTE_CHARS = str.maketrans("", "", '"“”')


def _search_query_for(prompt: str) -> str:
    """Derive the Discovery Engine search query from an LLM prompt, stripped
    of quote characters that Discovery Engine treats as an exact-phrase
    operator (see module-level comment above)."""
    return prompt.translate(_QUOTE_CHARS)


def build_sweep_df() -> pd.DataFrame:
    """One row per (tier x repeat); `rag_check.param_sweep` runs both arms
    per row, so len(df) * N_ARMS is the total LLM-query count."""
    rows = []
    for tier in TIERS:
        for repeat in range(N_REPEATS):
            rows.append({
                "prompt": tier["prompt"],
                "search_query": _search_query_for(tier["prompt"]),
                "tags": TAG_FILTER,
                "top_k": TOP_K,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "seed": BASE_SEED + repeat,
                # Pass-through columns: not read by run_comparison(), just
                # echoed back as input_* so plot_results.py can group by them.
                "complexity_tier": tier["tier"],
                "complexity_label": tier["label"],
                "repeat": repeat,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the poster's 9-tier complexity sweep (filtered vs. "
                    "full_corpus RAG latency + output quality).",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "sweep_results.csv",
                        help="Where to write the result CSV (default: poster/sweep_results.csv).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned sweep and exit without calling any API.")
    parser.add_argument("--verbose", action="store_true", help="Verbose/DEBUG logging.")
    args = parser.parse_args()

    df = build_sweep_df()
    n_queries = len(df) * N_ARMS
    print(
        f"Planned sweep: {len(TIERS)} complexity tiers x {N_REPEATS} repeats "
        f"= {len(df)} rows x {N_ARMS} arms = {n_queries} total LLM queries."
    )
    if args.dry_run:
        print(df[["complexity_tier", "complexity_label", "repeat", "seed"]].to_string(index=False))
        return

    results = run_sweep(df, out_csv=args.out, verbose=args.verbose)
    n_errors = (results["arm"] == "error").sum() if "arm" in results.columns else 0
    print(f"\nWrote {len(results)} result rows to {args.out} ({n_errors} row-level error(s)).")


if __name__ == "__main__":
    main()
