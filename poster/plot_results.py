#!/usr/bin/env python
"""
Render the poster figure from poster/sweep.py's result CSV.

Three stacked small-multiple panels sharing one categorical x-axis
(complexity tier 1-9, ordered) — deliberately NOT a dual-axis chart (the
three measures are different scales; see dataviz anti-patterns on why two
y-scales on one plot invent correlations that aren't in the data):

  Panel A (top):     mean total latency per tier, filtered vs. full_corpus,
                      error bars = SEM across the 5 repeats per tier.
  Panel B (middle):   % of repeats with a hard structural failure
                      (poster/quality.py: declined to answer, malformed
                      JSON, a missing required key, or an explicit
                      character budget blown) — same two series, same
                      x-axis, aligned under panel A.
  Panel C (bottom):   structural variability across the 5 repeats — mean
                      coefficient of variation (CV%) of list-field item
                      counts for responses that otherwise parsed fine. High
                      CV, even with zero hard failures, means the model is
                      improvising how much structure to produce rather than
                      reporting something stable it actually found — the
                      softer inaccuracy proxy poster/trials/trial_0.md
                      adopted after trial_0's strict minimum-item-count
                      check turned out to be confounded with the thing it
                      was trying to measure (see that doc, and
                      poster/quality.py's docstring).

Panels A+B together are the headline: wherever one arm's bar in panel A
dips *below* the other's (it looks faster) while panel B shows that SAME
arm's failure rate spiking at the same tier, that's the paradox this poster
reports — it only "wins" on latency because it gave up on the output (or
declined to answer at all), not because retrieval+generation was actually
more efficient. `_find_paradox_annotation()` checks both directions and
labels whichever arm the live data actually shows this for — don't assume
it's full_corpus: trial_0 found the filtered arm was the one trading
reliability for speed (a narrow top_k + schema fields the corpus couldn't
support made it decline, per the "don't guess" system preamble, far more
often than full_corpus, which has the whole corpus to search for something
relevant — see poster/trials/trial_0.md for the full trial_0 writeup).

Colors follow the dataviz skill's validated default palette, categorical
slots 1-2 (blue=filtered, green=full_corpus) — validated for this pair:
    node <dataviz-skill>/scripts/validate_palette.js "#2a78d6,#008300" --mode light

Usage:
    PYTHONPATH=. python poster/plot_results.py
    PYTHONPATH=. python poster/plot_results.py --csv poster/sweep_results.csv --out poster/figure.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from poster.prompts import TIERS_BY_NUMBER  # noqa: E402
from poster.quality import FAILURE_FLAGS, evaluate_quality  # noqa: E402

# Validated categorical slots 1-2 (see module docstring) — fixed order, never cycled.
_COLOR_FILTERED = "#2a78d6"
_COLOR_FULL_CORPUS = "#008300"
_ARM_COLORS = {"filtered": _COLOR_FILTERED, "full_corpus": _COLOR_FULL_CORPUS}
_ARM_ORDER = ["filtered", "full_corpus"]
_ARM_LABELS = {"filtered": "Filtered (tagged guidance)", "full_corpus": "Whole corpus"}

_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_TEXT_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_SURFACE = "#fcfcfb"


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
    return df


def _aggregate_latency_and_failures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (tier, arm): mean/SEM latency + hard-failure rate."""
    df = df[df["arm"].isin(_ARM_ORDER)].copy()
    df["is_failure"] = df["quality_flag"].isin(FAILURE_FLAGS)

    grouped = df.groupby(["input_complexity_tier", "input_complexity_label", "arm"])
    agg = grouped["total_latency_seconds"].agg(["mean", "sem", "count"]).reset_index()
    fail_rate = grouped["is_failure"].mean().reset_index(name="failure_rate")
    agg = agg.merge(fail_rate, on=["input_complexity_tier", "input_complexity_label", "arm"])
    agg = agg.sort_values(["input_complexity_tier", "arm"])
    agg["sem"] = agg["sem"].fillna(0.0)
    return agg


def _aggregate_variability(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (tier, arm): mean coefficient of variation (CV%) of
    list-field item counts across the 5 repeats, averaged over every
    list_field that tier defines. NaN where fewer than 2 non-null
    observations exist for every field (can't compute a variance) — the
    figure just leaves that bar empty rather than fabricating a 0."""
    records = []
    for (tier, label, arm), group in df[df["arm"].isin(_ARM_ORDER)].groupby(
        ["input_complexity_tier", "input_complexity_label", "arm"]
    ):
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
            "input_complexity_label": label,
            "arm": arm,
            "mean_cv_pct": float(np.mean(field_cvs)) if field_cvs else np.nan,
        })
    return pd.DataFrame(records).sort_values(["input_complexity_tier", "arm"])


def _find_paradox_annotation(agg: pd.DataFrame, tier_order: list, x_labels: list) -> Optional[dict]:
    """Pick the single (tier, arm) pair that most clearly shows the paradox —
    one arm *faster* at that tier while *also* failing structurally more
    often than the other — and return one arrow annotation pointing at its
    latency bar, or None if no tier in this run shows the pattern in either
    direction. Checked both ways deliberately: which arm ends up faster-but-
    less-reliable is an empirical question, not an assumption (trial_0 found
    it was `filtered`, the opposite of the originally anticipated direction
    — see poster/trials/trial_0.md)."""
    pivot = agg.pivot(index="input_complexity_tier", columns="arm", values=["mean", "failure_rate"])
    best = None  # (score, tier, faster_arm, faster_mean, faster_fail)
    for tier in tier_order:
        try:
            means = {arm: pivot.loc[tier, ("mean", arm)] for arm in _ARM_ORDER}
            fails = {arm: pivot.loc[tier, ("failure_rate", arm)] for arm in _ARM_ORDER}
        except KeyError:
            continue
        for faster, slower in (("filtered", "full_corpus"), ("full_corpus", "filtered")):
            latency_gap = means[slower] - means[faster]  # faster arm being quicker -> positive
            failure_gap = fails[faster] - fails[slower]  # faster arm failing more -> positive
            if latency_gap > 0 and failure_gap > 0:
                score = latency_gap * failure_gap
                if best is None or score > best[0]:
                    best = (score, tier, faster, means[faster], fails[faster])

    if best is None:
        return None
    _, tier, faster_arm, faster_mean, faster_fail = best
    idx = tier_order.index(tier)
    return dict(
        text=f"{_ARM_LABELS[faster_arm]}: faster, but {faster_fail * 100:.0f}% failed",
        x=x_labels[idx], y=faster_mean,
        xref="x", yref="y",
        ax=40, ay=-50, showarrow=True, arrowcolor=_TEXT_SECONDARY, arrowwidth=1.5, arrowhead=2,
        font=dict(size=12, color=_TEXT_PRIMARY),
        bgcolor=_SURFACE, bordercolor=_TEXT_SECONDARY, borderwidth=1, borderpad=4,
    )


def build_figure(agg: pd.DataFrame, variability: pd.DataFrame) -> go.Figure:
    tier_order = sorted(agg["input_complexity_tier"].unique())
    x_labels = [f"T{t}" for t in tier_order]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.44, 0.26, 0.30],
        subplot_titles=(
            "Mean response latency by prompt complexity (± SEM, n=5 repeats)",
            "Hard failure rate (declined to answer / malformed JSON / missing key / over character budget)",
            "Structural variability across repeats (mean CV% of list item counts — lower is more consistent)",
        ),
    )

    for arm in _ARM_ORDER:
        sub = agg[agg["arm"] == arm].set_index("input_complexity_tier").reindex(tier_order)
        var_sub = variability[variability["arm"] == arm].set_index("input_complexity_tier").reindex(tier_order)
        fig.add_trace(
            go.Bar(
                x=x_labels, y=sub["mean"],
                error_y=dict(type="data", array=sub["sem"], color=_TEXT_SECONDARY, thickness=1.5, width=3),
                name=_ARM_LABELS[arm], marker=dict(color=_ARM_COLORS[arm]),
                width=0.36, legendgroup=arm, showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x_labels, y=sub["failure_rate"] * 100.0,
                name=_ARM_LABELS[arm], marker=dict(color=_ARM_COLORS[arm]),
                width=0.36, legendgroup=arm, showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x_labels, y=var_sub["mean_cv_pct"],
                name=_ARM_LABELS[arm], marker=dict(color=_ARM_COLORS[arm]),
                width=0.36, legendgroup=arm, showlegend=False,
            ),
            row=3, col=1,
        )

    tier_captions = [
        f"T{t}: {TIERS_BY_NUMBER[t]['label']}" for t in tier_order if t in TIERS_BY_NUMBER
    ]
    mid = (len(tier_captions) + 1) // 2
    tier_caption_text = (
        "  ·  ".join(tier_captions[:mid]) + "<br>" + "  ·  ".join(tier_captions[mid:])
    )

    # One sparing direct label on the single tier that best tells the paradox
    # story — not a number on every bar (see dataviz skill: label selectively).
    # Disabled (uncomment to re-enable): _find_paradox_annotation(agg, tier_order, x_labels)
    paradox_annotation = None

    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=_TEXT_PRIMARY, size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=14)),
        margin=dict(l=70, r=30, t=90, b=190),
        width=1400,
        height=1260,
        annotations=list(fig.layout.annotations) + [
            dict(
                text=tier_caption_text,
                xref="paper", yref="paper", x=0, y=-0.13,
                showarrow=False, align="left",
                font=dict(size=12, color=_TEXT_SECONDARY),
            ),
        ] + ([paradox_annotation] if paradox_annotation else []),
    )
    fig.update_yaxes(
        title_text="Seconds", row=1, col=1,
        gridcolor=_GRIDLINE, zerolinecolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED),
        title_font=dict(color=_TEXT_SECONDARY),
    )
    fig.update_yaxes(
        title_text="% of repeats", row=2, col=1,
        gridcolor=_GRIDLINE, zerolinecolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED),
        title_font=dict(color=_TEXT_SECONDARY), range=[0, 100],
    )
    fig.update_yaxes(
        title_text="CV %", row=3, col=1,
        gridcolor=_GRIDLINE, zerolinecolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED),
        title_font=dict(color=_TEXT_SECONDARY), rangemode="tozero",
    )
    fig.update_xaxes(gridcolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED), row=1, col=1)
    fig.update_xaxes(gridcolor=_GRIDLINE, tickfont=dict(color=_TEXT_MUTED), row=2, col=1)
    fig.update_xaxes(
        title_text="Prompt complexity tier (see key below)", gridcolor=_GRIDLINE,
        tickfont=dict(color=_TEXT_MUTED), title_font=dict(color=_TEXT_SECONDARY), row=3, col=1,
    )
    for annotation in fig.layout.annotations[:3]:
        annotation.font = dict(size=15, color=_TEXT_PRIMARY)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the filtered-vs-full-corpus latency/quality poster figure."
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
    agg = _aggregate_latency_and_failures(df)
    variability = _aggregate_variability(df)

    fig = build_figure(agg, variability)
    fig.write_image(str(args.out), scale=2)
    html_out = args.out.with_suffix(".html")
    fig.write_html(str(html_out), include_plotlyjs="cdn")

    print(f"Wrote {args.out}")
    print(f"Wrote {html_out} (interactive / table-view twin)")

    summary = agg.merge(
        variability[["input_complexity_tier", "arm", "mean_cv_pct"]],
        on=["input_complexity_tier", "arm"], how="left",
    )
    print("\nPer-tier summary:")
    print(
        summary[["input_complexity_label", "arm", "mean", "sem", "failure_rate", "mean_cv_pct", "count"]]
        .rename(columns={"mean": "mean_latency_s", "sem": "sem_latency_s"})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
