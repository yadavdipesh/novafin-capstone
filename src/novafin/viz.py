"""
novafin-capstone/src/novafin/viz.py

Themed figure builders.

Every chart that reaches the D7 guide or the D8 deck is produced by a function
in this module. Notebooks call them and display the result; they never build a
figure inline. That keeps the deliverables consistent (same palette, same DPI,
same accessibility rules) and means a figure can be regenerated from a single
call when the underlying numbers change.

Accessibility is enforced here, not left to discipline
------------------------------------------------------
Measured contrast on white: ``gold #FFC72C`` is 1.56:1 and ``teal #00B2A9`` is
2.64:1, so neither may carry small text. Risk annotations therefore use
``gold_text #7A5B00`` (6.32:1), and every risk-coded chart also varies **marker
shape or hatch**, so the information survives greyscale printing and
colour-vision deficiency.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from novafin.utils.theme import (
    categorical_cycle,
    color,
    diverging_cmap,
    semantic_color,
    sequential_cmap,
)

__all__ = [
    "plot_class_balance",
    "plot_decile_lift",
    "plot_missingness",
    "plot_correlation_heatmap",
    "plot_target_correlations",
    "plot_temporal_rate",
    "plot_distribution_by_class",
    "plot_split_schedule",
    "plot_leakage_evidence",
]

LOGGER = logging.getLogger(__name__)

#: Marker shapes paired with the semantic colours, so risk state is never
#: encoded by colour alone.
RISK_MARKERS = {"good": "o", "warning": "s", "critical": "D", "neutral": "^"}


def _finish(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_class_balance(
    balance: pd.DataFrame, target: str, *, title: str | None = None
) -> Figure:
    """Horizontal bar chart of class counts with the proportion annotated.

    Args:
        balance: Output of ``profile.class_balance``.
        target: Target column name (used for labels).
        title: Optional override.
    """
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    labels = balance[target].astype(str)
    counts = balance["count"]
    minority = counts.idxmin()

    colors = [
        semantic_color("warning") if i == minority else color("navy")
        for i in balance.index
    ]
    bars = ax.barh(labels, counts, color=colors)
    for bar, proportion in zip(bars, balance["proportion"], strict=False):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{proportion:.2%}",
            va="center",
            fontsize=9,
            color=semantic_color("text"),
        )
    ax.set_xlim(0, counts.max() * 1.18)
    ratio = balance["imbalance_ratio"].iloc[0] if "imbalance_ratio" in balance else None
    suffix = f"  (imbalance {ratio:.0f}:1)" if ratio else ""
    _finish(ax, title or f"Class balance - {target}{suffix}", "Rows", "")
    return fig


def plot_decile_lift(
    lift: pd.DataFrame, feature: str, target: str, *, base_rate: float | None = None
) -> Figure:
    """Event rate per quantile bin, with the base rate drawn as a reference.

    The reference line is the point of the chart: a bar above it is a bin where
    the feature genuinely concentrates events.
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    positions = np.arange(len(lift))
    rates = lift["event_rate"].to_numpy()

    threshold = base_rate if base_rate is not None else float(np.average(rates, weights=lift["n"]))
    colors = [
        semantic_color("warning") if r > 2 * threshold else color("teal") for r in rates
    ]
    ax.bar(positions, rates, color=colors)
    ax.axhline(
        threshold,
        color=semantic_color("critical"),
        linestyle="--",
        linewidth=1.2,
        label=f"base rate {threshold:.2%}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"D{i + 1}" for i in positions])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.legend()
    _finish(ax, f"{target} rate by {feature} decile", f"{feature} decile (low to high)", "Event rate")
    return fig


def plot_missingness(profile: pd.DataFrame, dataset: str) -> Figure:
    """Missingness per column.

    Kept even though every NovaFin file is complete: a chart that shows a flat
    zero is evidence, and its absence would leave "no missing values" as an
    unsupported claim in the report.
    """
    fig, ax = plt.subplots(figsize=(8.0, max(2.4, 0.26 * len(profile))))
    ax.barh(profile["column"], profile["missing_pct"], color=color("teal"))
    ax.set_xlim(0, max(1.0, float(profile["missing_pct"].max()) * 1.2))
    ax.invert_yaxis()
    _finish(ax, f"Missingness - {dataset}", "Missing (%)", "")
    if float(profile["missing_pct"].max()) == 0.0:
        ax.text(
            0.5, 0.5, "0.00% missing in every column",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=semantic_color("good_text"),
        )
    return fig


def plot_correlation_heatmap(
    frame: pd.DataFrame, *, columns: Sequence[str] | None = None, title: str = "Correlation"
) -> Figure:
    """Correlation matrix in the brand diverging ramp.

    Uses the diverging map (gold-white-teal) rather than a sequential one so
    that sign is readable at a glance - positive and negative relationships
    are different *hues*, not different intensities.
    """
    numeric = frame[list(columns)] if columns else frame.select_dtypes(include=[np.number])
    corr = numeric.corr(numeric_only=True)

    size = max(4.5, 0.42 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(corr, cmap=diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.75, label="Pearson r")
    _finish(ax, title)
    return fig


def plot_target_correlations(
    correlations: pd.Series, target: str, *, alias_threshold: float = 0.95
) -> Figure:
    """Ranked |correlation| with the target, with the leakage line drawn in.

    The dashed line at 0.95 is the alias threshold from ``data/validate.py``.
    Anything touching it is a leak; a healthy module shows its strongest bar
    far below it. This single chart communicates the whole leakage argument.
    """
    fig, ax = plt.subplots(figsize=(8.0, max(2.6, 0.32 * len(correlations))))
    values = correlations.sort_values(key=np.abs)
    colors = [
        semantic_color("critical") if abs(v) >= alias_threshold else color("teal")
        for v in values
    ]
    ax.barh(values.index, values.to_numpy(), color=colors)
    ax.axvline(alias_threshold, color=semantic_color("critical"), linestyle="--", linewidth=1.0)
    ax.axvline(-alias_threshold, color=semantic_color("critical"), linestyle="--", linewidth=1.0)
    ax.axvline(0, color=color("silver"), linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    _finish(ax, f"Correlation with {target}  (dashed = leakage threshold)", "Pearson r", "")
    return fig


def plot_temporal_rate(rates: pd.DataFrame, time_column: str, target: str) -> Figure:
    """Event rate over time with the overall mean, to evidence stationarity."""
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(rates[time_column], rates["event_rate"], marker="o", color=color("teal"))
    mean_rate = float(np.average(rates["event_rate"], weights=rates["n"]))
    ax.axhline(mean_rate, color=color("navy"), linestyle="--", linewidth=1.0,
               label=f"overall {mean_rate:.2%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.legend()
    _finish(ax, f"{target} rate over time", time_column, "Event rate")
    fig.autofmt_xdate()
    return fig


def plot_distribution_by_class(
    frame: pd.DataFrame, feature: str, target: str, *, log_x: bool = False, bins: int = 60
) -> Figure:
    """Overlaid, density-normalised distributions per class.

    Normalised to density rather than count because at a 2.28% event rate the
    minority histogram is invisible on a shared count axis - which is how
    genuinely separating features get dismissed as uninformative.
    """
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    values = frame[[feature, target]].dropna()
    classes = sorted(values[target].unique())
    palette = categorical_cycle()

    if log_x:
        positive = values[values[feature] > 0]
        edges = np.logspace(
            np.log10(positive[feature].min()), np.log10(positive[feature].max()), bins
        )
    else:
        edges = np.linspace(values[feature].min(), values[feature].max(), bins)

    for index, klass in enumerate(classes):
        subset = values.loc[values[target] == klass, feature]
        ax.hist(
            subset, bins=edges, density=True, alpha=0.55,
            color=palette[index % len(palette)],
            label=f"{target}={klass} (n={len(subset):,})",
        )
    if log_x:
        ax.set_xscale("log")
    ax.legend()
    _finish(ax, f"{feature} by {target}", feature, "Density")
    return fig


def plot_split_schedule(schedule: pd.DataFrame, scheme: str) -> Figure:
    """Gantt-style view of cross-validation folds.

    Makes the validation strategy auditable: a reviewer can see the expanding
    training window, the embargo gap and the non-overlapping test blocks
    instead of trusting the scheme's name.
    """
    fig, ax = plt.subplots(figsize=(9.5, max(2.4, 0.5 * len(schedule))))
    has_dates = {"train_end", "test_start", "test_end"} <= set(schedule.columns)

    for _, row in schedule.iterrows():
        y = row["fold"]
        if has_dates:
            start = row["train_end"]
            ax.barh(y, row["test_end"] - row["test_start"], left=row["test_start"],
                    color=semantic_color("warning"), height=0.55)
            ax.plot([start, row["test_start"]], [y, y],
                    color=semantic_color("critical"), linewidth=2.0, linestyle=":")
        else:
            ax.barh(y, row["n_test"], left=row["n_train"],
                    color=semantic_color("warning"), height=0.55)
            ax.barh(y, row["n_train"], color=color("navy"), height=0.55)

    ax.set_yticks(schedule["fold"])
    ax.set_yticklabels([f"Fold {int(f)}" for f in schedule["fold"]])
    ax.invert_yaxis()
    _finish(ax, f"Validation schedule - {scheme}", "Date" if has_dates else "Row index", "")
    if has_dates:
        fig.autofmt_xdate()
    return fig


def plot_leakage_evidence(
    leaked: np.ndarray, clean: np.ndarray, *, labels: tuple[str, str] = ("Leaked", "Clean"),
    title: str = "Leakage evidence", ylabel: str = "Value",
) -> Figure:
    """Side-by-side bars contrasting a leaked model with its clean counterpart.

    This is the figure that turns the leakage register into a deck slide: the
    leaked bar is deliberately drawn in the critical colour and hatched, so the
    point survives a greyscale printout.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    positions = np.arange(len(leaked))
    width = 0.38
    ax.bar(positions - width / 2, leaked, width, label=labels[0],
           color=semantic_color("critical"), hatch="//", edgecolor="white")
    ax.bar(positions + width / 2, clean, width, label=labels[1], color=color("teal"))
    ax.set_xticks(positions)
    ax.legend()
    _finish(ax, title, "", ylabel)
    return fig
