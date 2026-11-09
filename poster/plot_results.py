#!/usr/bin/env python
"""
Render the poster figure from poster/sweep.py's result CSV.

Single-panel design (revised after supervisor review of the earlier
3-panel version — see poster/trials/trial_2.md's figure-revision addendum):

  - X-axis is plain prompt number ("Prompt 1" .. "Prompt 9"), not a
    "complexity tier" — supervisors specifically asked not to frame these
    9 prompts as more-or-less complex; they're 9 distinct prompts on the
    same topic, numbered, nothing more. (The underlying code still orders
    them 1-9 and `poster/prompts.py` still documents what each one asks —
    that's an implementation detail for maintainers, not something this
    chart asserts to the viewer.)
  - Each arm's mean total latency is still a bar (± SEM error bar), but
    the 5 individual repeat values are now also plotted as discrete points
    jittered around each bar — supervisors wanted to see the actual runs,
    not just the aggregate.
  - A run that hard-failed (poster/quality.py: declined to answer,
    malformed JSON, a missing required key, or an explicit character
    budget blown) is drawn in a reserved status color (critical red) with
    an "x" marker instead of its arm's color/circle — not a separate
    failure-rate panel anymore, just a per-point flag on the same latency
    plot, with one extra legend entry.
  - The structural-variability panel (coefficient-of-variation of list
    item counts) is REMOVED for now, not just hidden — supervisors noted
    interrater disagreement on what counts as "correct" structure, so a
    metric that scores structural consistency against an assumed-correct
    shape isn't defensible yet. `_aggregate_variability()` is kept in the
    code (unused) in case that metric is revisited later with an agreed
    ground truth; it is deliberately not wired into build_figure().

Colors follow the dataviz skill's validated default palette: categorical
slots 1-2 for the two arms (blue=filtered, green=full_corpus), validated
for this pair:
    node <dataviz-skill>/scripts/validate_palette.js "#2a78d6,#008300" --mode light
and the skill's fixed status palette for the failed-run marker (critical,
reserved — never reused for a plain series, always paired with a legend
label here, per the skill's status-color rule).

Usage:
    PYTHONPATH=. python poster/plot_results.py
    PYTHONPATH=. python poster/plot_results.py --csv poster/sweep_results.csv --out poster/figure.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from poster.prompts import TIERS_BY_NUMBER  # noqa: E402
from poster.quality import FAILURE_FLAGS, evaluate_quality  # noqa: E402

# Validated categorical slots 1-2 (see module docstring) — fixed order, never cycled.
_COLOR_FILTERED = "#2a78d6"
_COLOR_FULL_CORPUS = "#008300"
_ARM_COLORS = {"filtered": _COLOR_FILTERED, "full_corpus": _COLOR_FULL_CORPUS}
_ARM_ORDER = ["filtered", "full_corpus"]
_ARM_LABELS = {"filtered": "Filtered (tagged guidance)", "full_corpus": "Whole corpus"}
_ARM_X_OFFSET = {"filtered": -0.18, "full_corpus": 0.18}

# Fixed status palette (dataviz skill) — reserved, never themed, always
# paired with a legend label. Used only for the failed-run marker.
_COLOR_CRITICAL = "#d03b3b"

_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_TEXT_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_SURFACE = "#fcfcfb"

# Small deterministic within-group spread so 5 repeats at the same bar
# don't sit exactly on top of each other. Centered on 0, well inside the
# 0.36-wide bar.
_JITTER_BY_REPEAT = {0: -0.09, 1: -0.045, 2: 0.0, 3: 0.045, 4: 0.09}


def _score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row quality_flag + list_item_counts by looking up each
    row's tier definition (poster/prompts.py)."""
    flags, item_counts = [], []
    for _, row in df.iterrows():
        tier_num = row.get("input_complexity_tier")
        tier = TIERS_BY_NUMBER.get(int(tier_num)) if pd.notna(tier_num) else None
        if tier is None:
            flags.append("n/a")
            item_counts.append({})
            continue
        result = evaluate_quality(
            row.get("answer_text") or "",
            expects_json=tier["expects_json"],
            required_keys=tier.get("required_keys"),
            list_fields=tier.get("list_fields"),
            max_chars=tier.get("max_chars"),
        )
        flags.append(result.quality_flag)
        item_counts.append(result.list_item_counts)
    df = df.copy()
    df["quality_flag"] = flags
    df["list_item_counts"] = item_counts
    df["is_failure"] = df["quality_flag"].isin(FAILURE_FLAGS)
    return df


def _aggregate_latency(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (tier, arm): mean/SEM latency, for the bars."""
    df = df[df["arm"].isin(_ARM_ORDER)].copy()
    grouped = df.groupby(["input_complexity_tier", "arm"])
    agg = grouped["total_latency_seconds"].agg(["mean", "sem", "count"]).reset_index()
    agg["sem"] = agg["sem"].fillna(0.0)
    return agg.sort_values(["input_complexity_tier", "arm"])


def _aggregate_variability(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (tier, arm): mean coefficient of variation (CV%) of
    list-field item counts across the 5 repeats. NOT currently used by
    build_figure() — see module docstring ("structural-variability panel
    removed for now"). Kept here, unused, in case this metric is revisited
    once there's an agreed ground truth for "correct" structure."""
    records = []
    for (tier, arm), group in df[df["arm"].isin(_ARM_ORDER)].groupby(["input_complexity_tier", "arm"]):
        tier_def = TIERS_BY_NUMBER.get(int(tier), {})
        list_fields = tier_def.get("list_fields") or []
        field_cvs = []
        for field_name in list_fields:
            counts = [
                counts_dict.get(field_name)
                for counts_dict in group["list_item_counts"]
                if isinstance(counts_dict, dict) and counts_dict.get(field_name) is not None
            ]
            if len(counts) < 2:
                continue
            mean_count = np.mean(counts)
            if mean_count > 0:
                field_cvs.append(float(np.std(counts) / mean_count * 100.0))
        records.append({
            "input_complexity_tier": tier,
            "arm": arm,
            "mean_cv_pct": float(np.mean(field_cvs)) if field_cvs else np.nan,
        })
    return pd.DataFrame(records).sort_values(["input_complexity_tier", "arm"])


def build_figure(df: pd.DataFrame, agg: pd.DataFrame) -> go.Figure:
    tier_order = sorted(agg["input_complexity_tier"].unique())
    x_pos_by_tier = {t: i for i, t in enumerate(tier_order)}
    tick_labels = [f"Prompt {t}" for t in tier_order]

    fig = go.Figure()

    for arm in _ARM_ORDER:
        sub = agg[agg["arm"] == arm].set_index("input_complexity_tier").reindex(tier_order)
        bar_x = [x_pos_by_tier[t] + _ARM_X_OFFSET[arm] for t in tier_order]
        fig.add_trace(go.Bar(
            x=bar_x, y=sub["mean"],
            error_y=dict(type="data", array=sub["sem"], color=_TEXT_SECONDARY, thickness=1.5, width=3),
            name=_ARM_LABELS[arm], marker=dict(color=_ARM_COLORS[arm], opacity=0.55),
            width=0.34, legendgroup=arm, showlegend=True,
        ))

    # Discrete per-repeat points, jittered around each bar. Split by
    # pass/fail so a failed run renders in the reserved status color with
    # its own legend entry, instead of a separate failure-rate panel.
    first_failed_point_plotted = False
    for arm in _ARM_ORDER:
        arm_rows = df[df["arm"] == arm]
        for failed in (False, True):
            subset = arm_rows[arm_rows["is_failure"] == failed]
            if subset.empty:
                continue
            xs, ys = [], []
            for _, row in subset.iterrows():
                tier = int(row["input_complexity_tier"])
                repeat = int(row.get("input_repeat", 0))
                jitter = _JITTER_BY_REPEAT.get(repeat, 0.0)
                xs.append(x_pos_by_tier[tier] + _ARM_X_OFFSET[arm] + jitter)
                ys.append(row["total_latency_seconds"])
            if failed:
                show_legend = not first_failed_point_plotted
                first_failed_point_plotted = True
                marker = dict(color=_COLOR_CRITICAL, size=10, symbol="x", line=dict(width=1, color=_SURFACE))
                name = "Failed run (hard failure)"
            else:
                show_legend = False
                marker = dict(
                    color=_ARM_COLORS[arm], size=7, symbol="circle",
                    line=dict(width=1, color=_SURFACE),
                )
                name = _ARM_LABELS[arm]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", marker=marker,
                name=name, legendgroup="failed" if failed else arm,
                showlegend=show_legend,
            ))

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=_TEXT_PRIMARY, size=15),
        title=dict(
            text="Mean response latency per prompt (± SEM; individual runs shown, n=5 each)",
            font=dict(size=17, color=_TEXT_PRIMARY),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=14)),
        margin=dict(l=70, r=30, t=100, b=70),
        width=1400,
        height=750,
    )
    fig.update_yaxes(
        title_text="Seconds",
        gridcolor=_GRIDLINE, zerolinecolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED),
        title_font=dict(color=_TEXT_SECONDARY), rangemode="tozero",
    )
    fig.update_xaxes(
        title_text="Prompt #",
        tickvals=list(range(len(tier_order))), ticktext=tick_labels,
        gridcolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED), title_font=dict(color=_TEXT_SECONDARY),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the filtered-vs-full-corpus latency poster figure."
    )
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "sweep_results.csv",
                        help="Result CSV from poster/sweep.py (default: poster/sweep_results.csv).")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figure.png",
                        help="Output image path (default: poster/figure.png). "
                             "Also writes a .html twin next to it (interactive table-view equivalent).")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Result CSV not found: {args.csv} (run poster/sweep.py first).")

    df = pd.read_csv(args.csv)
    df = _score_rows(df)
    agg = _aggregate_latency(df)

    fig = build_figure(df, agg)
    fig.write_image(str(args.out), scale=2)
    html_out = args.out.with_suffix(".html")
    fig.write_html(str(html_out), include_plotlyjs="cdn")

    print(f"Wrote {args.out}")
    print(f"Wrote {html_out} (interactive / table-view twin)")

    fail_rate = df[df["arm"].isin(_ARM_ORDER)].groupby(
        ["input_complexity_tier", "arm"]
    )["is_failure"].mean().reset_index(name="failure_rate")
    summary = agg.merge(fail_rate, on=["input_complexity_tier", "arm"])
    print("\nPer-prompt summary:")
    print(
        summary[["input_complexity_tier", "arm", "mean", "sem", "failure_rate", "count"]]
        .rename(columns={"mean": "mean_latency_s", "sem": "sem_latency_s", "input_complexity_tier": "prompt"})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
