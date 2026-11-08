#!/usr/bin/env python
"""
Worked example: build a sweep DataFrame programmatically (instead of a CSV)
and run it through rag_check.param_sweep.run_sweep() — the "method wrapper"
for in-memory DataFrames.

This sweeps top_k x thinking_budget for a fixed prompt/tag filter, which is
the kind of grid a CSV is awkward for (you'd otherwise hand-write every
combination as a row) but `itertools.product` + `pandas.DataFrame` makes
trivial.

Usage:
    PYTHONPATH=. python rag_check/examples/dataframe_sweep_example.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd  # noqa: E402

from rag_check.param_sweep import run_sweep  # noqa: E402

PROMPT = "What should I do about a client disclosing suicidal ideation mid-session?"
TAGS = "session_event_tags=disclosure_SI"

TOP_K_VALUES = [3, 10]
THINKING_BUDGET_VALUES = [0, 2048]  # 0 = thinking disabled, 2048 = budgeted thinking

rows = [
    {
        "prompt": PROMPT,
        "tags": TAGS,
        "top_k": top_k,
        "thinking_budget": thinking_budget,
        "model": "gemini-2.5-flash",
        "seed": 42,
        # Isolate the filtered arm only — this grid is about retrieval/thinking
        # cost, not the filtered-vs-full-corpus comparison itself.
        "skip_full_corpus": True,
    }
    for top_k, thinking_budget in itertools.product(TOP_K_VALUES, THINKING_BUDGET_VALUES)
]
sweep_df = pd.DataFrame(rows)

if __name__ == "__main__":
    results = run_sweep(sweep_df, out_csv=Path("dataframe_sweep_results.csv"))
    print(results[[
        "input_top_k", "input_thinking_budget",
        "retrieval_latency_seconds", "generation_latency_seconds", "total_latency_seconds",
        "total_token_count",
    ]].to_string(index=False))
