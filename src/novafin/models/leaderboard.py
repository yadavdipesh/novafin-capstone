"""
novafin-capstone/src/novafin/models/leaderboard.py

D5 - the auto-generated leaderboard.

Run as ``make leaderboard`` or ``python -m novafin.models.leaderboard``.

Why this exists as a module rather than a notebook cell
--------------------------------------------------------
The D5 requirement is "a leaderboard table auto-generated from runs". Generated
means exactly that: the table in the D7 guide and on the D8 results slide is
produced by this script from the run store, not typed by hand. A number that
was typed can drift from the run that produced it; a number that was generated
cannot.

It reads the run store directly rather than requiring ``mlflow ui``, which
cannot be reached from a Colab VM without a tunnel account.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from novafin.config import Config, load_config
from novafin.paths import resolve
from novafin.tracking import load_runs
from novafin.utils.io import atomic_write_text
from novafin.utils.logging_utils import setup_logging

__all__ = ["build_leaderboard", "format_leaderboard", "main"]

LOGGER = logging.getLogger(__name__)

#: The metric each module is ranked by, and whether higher is better.
#: These follow the Phase-0 audit: PR-AUC where the base rate is low, KS for
#: credit, rank IC for the equity panel, macro-F1 for the 3-class order book.
PRIMARY_METRIC: dict[str, tuple[str, bool]] = {
    "initiatives": ("pr_auc", True),
    "loans": ("ks", True),
    "transactions": ("pr_auc", True),
    "customers": ("pr_auc", True),
    "customers_clv": ("rmse", False),
    "market": ("mean_ic", True),
    "liquidity": ("mae", False),
    "options": ("rmse", False),
    "hft": ("macro_f1", True),
}


def _metric_columns(runs: pd.DataFrame) -> dict[str, str]:
    """Map bare metric names to their column names in the run frame.

    MLflow prefixes logged metrics with ``metrics.``; the JSON-lines fallback
    uses ``metric.``. Normalising here means the rest of the module - and the
    guide - never has to care which backend produced the run.
    """
    mapping: dict[str, str] = {}
    for column in runs.columns:
        for prefix in ("metrics.", "metric."):
            if column.startswith(prefix):
                mapping[column[len(prefix):]] = column
    return mapping


def build_leaderboard(cfg: Config | None = None, *, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble one ranked row per module/model from the run store.

    Returns:
        A frame with ``module, model, rank, primary_metric, value`` plus the
        common metrics. Empty when no runs exist yet, which is the correct
        state before Phase 4 has been executed.
    """
    cfg = cfg or load_config()
    frame = runs if runs is not None else load_runs(cfg)
    if frame is None or frame.empty:
        LOGGER.warning("No runs found in %s - run 03_baseline_models first.", cfg.paths.mlruns)
        return pd.DataFrame()

    metrics = _metric_columns(frame)

    module_column = next(
        (c for c in ("tags.module", "tag.module", "module") if c in frame.columns), None
    )
    model_column = next(
        (c for c in ("tags.model", "tag.model", "model") if c in frame.columns), None
    )
    if module_column is None or model_column is None:
        LOGGER.error("Run frame is missing module/model tags; cannot rank.")
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, run in frame.iterrows():
        module = str(run[module_column])
        primary, higher_is_better = PRIMARY_METRIC.get(module, ("roc_auc", True))
        row: dict[str, object] = {
            "module": module,
            "model": str(run[model_column]),
            "primary_metric": primary,
            "value": run.get(metrics.get(primary, ""), float("nan")),
            "higher_is_better": higher_is_better,
        }
        for name in (
            "roc_auc", "pr_auc", "brier", "ks", "macro_f1", "balanced_accuracy",
            "rmse", "mae", "r2", "mean_ic", "ic_ir", "optimal_threshold",
            "expected_cost", "cv_roc_auc_std", "cv_pr_auc_std", "cv_n_folds",
        ):
            if name in metrics:
                row[name] = run.get(metrics[name], float("nan"))
        rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        return board

    board["value"] = pd.to_numeric(board["value"], errors="coerce")
    board = board.dropna(subset=["value"])

    ranked: list[pd.DataFrame] = []
    for module, chunk in board.groupby("module", observed=True):
        ascending = not bool(chunk["higher_is_better"].iloc[0])
        chunk = chunk.sort_values("value", ascending=ascending).reset_index(drop=True)
        chunk.insert(2, "rank", range(1, len(chunk) + 1))
        ranked.append(chunk)

    out = pd.concat(ranked, ignore_index=True)
    return out.drop(columns=["higher_is_better"]).sort_values(["module", "rank"]).reset_index(drop=True)


def _to_markdown(frame: pd.DataFrame) -> str:
    """Render a frame as a GitHub markdown table.

    Hand-rolled because ``DataFrame.to_markdown`` requires ``tabulate``, which
    is not in requirements.txt. Adding a package so a report can print a table
    is a poor trade; this is eight lines and has no dependency.
    """
    header = list(frame.columns)
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_leaderboard(board: pd.DataFrame) -> str:
    """Render the leaderboard as markdown for the README and the D7 guide."""
    if board.empty:
        return "_No runs recorded yet - execute `notebooks/03_baseline_models.ipynb`._"

    lines: list[str] = []
    for module, chunk in board.groupby("module", observed=True):
        primary = chunk["primary_metric"].iloc[0]
        lines.append(f"\n### {module}  (ranked by `{primary}`)\n")
        # The primary metric is shown once, as the renamed `value` column -
        # listing it again in the extras produced a duplicated header.
        columns = ["rank", "model", "value"] + [
            c for c in ("roc_auc", "pr_auc", "brier", "ks", "macro_f1", "rmse", "mae", "mean_ic")
            if c != primary and c in chunk.columns and chunk[c].notna().any()
        ]
        table = chunk[columns].copy().rename(columns={"value": primary})
        lines.append(_to_markdown(table))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point - ``python -m novafin.models.leaderboard``."""
    parser = argparse.ArgumentParser(description="Generate the NovaFin model leaderboard.")
    parser.add_argument("--output", type=Path, default=None, help="CSV path (default from config)")
    parser.add_argument("--markdown", type=Path, default=None, help="Also write a markdown table")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    setup_logging("WARNING" if args.quiet else "INFO")
    cfg = load_config()

    board = build_leaderboard(cfg)
    if board.empty:
        print("No runs recorded yet. Execute notebooks/03_baseline_models.ipynb first.")
        return 1

    # cfg.mlflow.leaderboard_path is repo-relative ("reports/tables/..."), so
    # it must be resolved against the repo root, not against paths.reports -
    # joining it to paths.reports produced "reports/reports/tables/...".
    if args.output is not None:
        output = Path(args.output)
    else:
        output = resolve(cfg.mlflow.leaderboard_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, board.to_csv(index=False))
    print(f"Leaderboard written to {output}  ({len(board)} rows)")

    if args.markdown:
        atomic_write_text(args.markdown, format_leaderboard(board))
        print(f"Markdown written to {args.markdown}")
    else:
        print(format_leaderboard(board))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
