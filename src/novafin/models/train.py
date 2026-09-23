"""
novafin-capstone/src/novafin/models/train.py

D4 Level 1 - cross-validated baselines.

One loop, every module
----------------------
:func:`cross_validate_model` is the only place a model is fitted in this
project. It takes the splitter chosen in Phase 2, the pipeline built in Phase 3
and the metrics from :mod:`novafin.evaluate`, and produces fold-level results,
out-of-fold predictions and a versioned :class:`~novafin.utils.io.ModelBundle`.

The three properties that make it trustworthy
----------------------------------------------
1. **Everything is fitted inside the fold.** The pipeline carries the imputer,
   the scaler and the one-hot category list, and ``pipeline.fit(X_train)`` is
   called per fold. No transformation ever sees the validation rows.
2. **Out-of-fold predictions are assembled, not averaged.** Each row is
   predicted exactly once, by a model that did not train on it. That single
   OOF vector is what the threshold optimisation, the calibration curve and
   the permutation test consume - averaging fold metrics would hide the
   fold-to-fold variance that Phase 2 showed is large on two modules.
3. **Fold variance is reported, never hidden.** Every metric is returned as
   mean and standard deviation across folds. With a per-fold AUC standard
   error of 0.096 on the initiatives module, a bare mean would be misleading.

Note on the final artefact: the bundle is refitted on **all** training data
after cross-validation, which is standard - CV estimates performance, the
deployed model should use every row available. The metrics recorded in the
bundle are the cross-validated ones, never the in-sample ones.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from novafin.config import Config, load_config
from novafin.evaluate import (
    EvaluationResult,
    calibration_table,
    classification_metrics,
    gains_table,
    multiclass_metrics,
    optimal_threshold,
    rank_ic,
    regression_metrics,
)
from novafin.models.build import ModelSpec, build_pipeline
from novafin.utils.io import ModelBundle

__all__ = [
    "FoldResult",
    "CVResult",
    "cross_validate_model",
    "train_module",
    "make_bundle",
]

LOGGER = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Everything one fold produced."""

    fold: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    fit_seconds: float
    test_index: np.ndarray
    predictions: np.ndarray


@dataclass
class CVResult:
    """Aggregated cross-validation output for one model on one module."""

    module: str
    model_name: str
    task: str
    folds: list[FoldResult] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    oof_mask: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def fold_frame(self) -> pd.DataFrame:
        """One row per fold - the table that shows the variance."""
        return pd.DataFrame(
            [{"fold": f.fold, "n_train": f.n_train, "n_test": f.n_test,
              "fit_seconds": round(f.fit_seconds, 2), **f.metrics} for f in self.folds]
        )

    def summary(self) -> dict[str, float]:
        """Mean and standard deviation of every metric across folds.

        Both are reported for every metric. A mean alone invites the reader to
        treat 0.71 +/- 0.09 and 0.71 +/- 0.01 as the same result.
        """
        frame = self.fold_frame()
        numeric = frame.drop(columns=["fold", "n_train", "n_test"], errors="ignore")
        out: dict[str, float] = {}
        for column in numeric.columns:
            values = numeric[column].astype("float64")
            out[column] = float(values.mean())
            out[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out["n_folds"] = float(len(self.folds))
        out["total_fit_seconds"] = float(sum(f.fit_seconds for f in self.folds))
        return out


# =============================================================================
# Prediction helpers
# =============================================================================
def _predict(pipeline: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    """Get the prediction appropriate to the task.

    Classification returns a probability wherever the estimator exposes one:
    calibration, thresholding and expected cost all need probabilities, and a
    hard 0/1 label discards the information they depend on.
    """
    if task == "binary_classification":
        if hasattr(pipeline, "predict_proba"):
            return np.asarray(pipeline.predict_proba(X))[:, 1]
        if hasattr(pipeline, "decision_function"):
            scores = np.asarray(pipeline.decision_function(X), dtype="float64")
            return 1.0 / (1.0 + np.exp(-scores))   # monotone squash for ranking
        return np.asarray(pipeline.predict(X), dtype="float64")

    if task == "multiclass_classification":
        return np.asarray(pipeline.predict(X))

    return np.asarray(pipeline.predict(X), dtype="float64")


def _score(
    y_true: np.ndarray, y_pred: np.ndarray, task: str, *, groups: np.ndarray | None = None
) -> dict[str, float]:
    """Dispatch to the metric bundle appropriate to the task."""
    if task == "binary_classification":
        return classification_metrics(y_true, y_pred)
    if task == "multiclass_classification":
        return multiclass_metrics(y_true, y_pred)
    metrics = regression_metrics(y_true, y_pred)
    if groups is not None:
        metrics.update(rank_ic(y_true, y_pred, groups))
    return metrics


# =============================================================================
# The loop
# =============================================================================
def cross_validate_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Any,
    *,
    task: str,
    module: str,
    cfg: Config | None = None,
    split_kwargs: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    ic_groups: Sequence[Any] | None = None,
    max_folds: int | None = None,
    pipeline_factory: Callable[..., Any] | None = None,
    fold_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> CVResult:
    """Cross-validate one model, fitting the whole pipeline inside each fold.

    Args:
        spec: Model declaration from ``configs/models.yaml``.
        X: Feature matrix (already through ``make_feature_frame``).
        y: Target.
        splitter: Any object with a scikit-learn-style ``split``.
        task: ``binary_classification`` | ``multiclass_classification`` |
            ``regression`` | ``panel_regression`` | ``time_series_forecast``.
        module: Dataset key, for logging.
        cfg: Project config.
        split_kwargs: Extra arguments for ``splitter.split`` (dates, groups).
        defaults: Global model defaults.
        ic_groups: Rebalance-date labels, enabling rank IC for the equity panel.
        max_folds: Stop after this many folds. Used for smoke runs on the
            120k-row order book, where a full sweep is minutes rather than
            seconds - and flagged in the result notes so a truncated run can
            never be mistaken for a complete one.
        pipeline_factory: Override for :func:`~novafin.models.build.build_pipeline`.
            Two reasons it is exposed: the loop becomes testable without
            scikit-learn installed, and a D4 Level-4 campaign can substitute a
            stacked or calibrated pipeline without touching this function.
        fold_callback: Called as ``callback(fold_index, fold_metrics)`` after
            each fold. This is what makes D4 Level-2 PRUNING possible: the
            Optuna objective reports each fold's score as it lands and raises
            ``TrialPruned`` from inside the callback, so a hopeless trial dies
            after 3 fits instead of 50. The exception propagates deliberately -
            catching it here would silently disable pruning.

    Returns:
        A :class:`CVResult`.
    """
    factory = pipeline_factory or build_pipeline
    cfg = cfg or load_config()
    split_kwargs = dict(split_kwargs or {})

    y_values = np.asarray(y)
    oof = np.full(len(X), np.nan, dtype="float64" if task != "multiclass_classification" else object)
    mask = np.zeros(len(X), dtype=bool)

    folds: list[FoldResult] = []
    for index, (train_idx, test_idx) in enumerate(splitter.split(X, y, **split_kwargs), start=1):
        if max_folds is not None and index > max_folds:
            break

        started = time.perf_counter()
        pipeline = factory(spec, X, cfg=cfg, defaults=defaults)
        pipeline.fit(X.iloc[train_idx], y_values[train_idx])
        predictions = _predict(pipeline, X.iloc[test_idx], task)
        elapsed = time.perf_counter() - started

        groups = np.asarray(ic_groups)[test_idx] if ic_groups is not None else None
        metrics = _score(y_values[test_idx], predictions, task, groups=groups)

        oof[test_idx] = predictions
        mask[test_idx] = True
        folds.append(
            FoldResult(
                fold=index, n_train=len(train_idx), n_test=len(test_idx),
                metrics=metrics, fit_seconds=elapsed,
                test_index=test_idx, predictions=predictions,
            )
        )
        LOGGER.info(
            "%s/%s fold %d: n_train=%d n_test=%d (%.1fs)",
            module, spec.name, index, len(train_idx), len(test_idx), elapsed,
        )

        # Pruning hook. Any exception raised here (notably optuna.TrialPruned)
        # is allowed to propagate - that is the mechanism, not a bug.
        if fold_callback is not None:
            fold_callback(index, metrics)

    notes: list[str] = []
    if max_folds is not None:
        notes.append(f"TRUNCATED RUN: stopped after {max_folds} fold(s)")
    if not folds:
        notes.append("no folds produced - check the splitter configuration")

    return CVResult(
        module=module, model_name=spec.name, task=task, folds=folds,
        oof_predictions=oof, oof_mask=mask, params=dict(spec.params), notes=notes,
    )


def evaluate_cv(
    result: CVResult,
    y: pd.Series,
    *,
    cfg: Config | None = None,
    cost_false_negative: float | None = None,
    cost_false_positive: float | None = None,
) -> EvaluationResult:
    """Turn a :class:`CVResult` into the reportable evaluation.

    Metrics are computed on the **out-of-fold vector** - every row predicted
    once by a model that did not see it - rather than by averaging fold
    metrics. The two differ, and the OOF version is the one a threshold or a
    calibration curve can legitimately be fitted to.
    """
    cfg = cfg or load_config()
    y_values = np.asarray(y)
    mask = result.oof_mask if result.oof_mask is not None else np.ones(len(y_values), bool)
    truth = y_values[mask]
    predictions = np.asarray(result.oof_predictions)[mask]

    evaluation = EvaluationResult(
        module=result.module, model_name=result.model_name, task=result.task,
        fold_metrics=result.fold_frame(), notes=list(result.notes),
    )

    if result.task == "binary_classification":
        predictions = predictions.astype("float64")
        evaluation.metrics = classification_metrics(truth, predictions)
        evaluation.calibration = calibration_table(truth, predictions)
        evaluation.gains = gains_table(truth, predictions)
        if cost_false_negative is not None and cost_false_positive is not None:
            evaluation.cost = optimal_threshold(
                truth, predictions,
                cost_false_negative=cost_false_negative,
                cost_false_positive=cost_false_positive,
            )
            evaluation.metrics["optimal_threshold"] = evaluation.cost["threshold"]
            evaluation.metrics["expected_cost"] = evaluation.cost["total_cost"]
    elif result.task == "multiclass_classification":
        evaluation.metrics = multiclass_metrics(truth, predictions)
    else:
        evaluation.metrics = regression_metrics(truth, predictions.astype("float64"))

    # Fold variance travels with the headline number, always.
    for key, value in result.summary().items():
        if key.endswith("_std") or key in {"n_folds", "total_fit_seconds"}:
            evaluation.metrics[f"cv_{key}"] = value

    return evaluation


def make_bundle(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    evaluation: EvaluationResult,
    *,
    cfg: Config | None = None,
    defaults: dict[str, Any] | None = None,
    data_hashes: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    pipeline_factory: Callable[..., Any] | None = None,
) -> ModelBundle:
    """Refit on all data and package model + preprocessor + feature list.

    Refitting on the full training set after cross-validation is standard: CV
    is there to *estimate* generalisation, and the deployed artefact should use
    every row available. The metrics stored in the bundle are the
    cross-validated ones - never in-sample numbers, which would be a lie by
    omission.
    """
    cfg = cfg or load_config()
    factory = pipeline_factory or build_pipeline
    pipeline = factory(spec, X, cfg=cfg, defaults=defaults)
    pipeline.fit(X, np.asarray(y))

    return ModelBundle(
        model=pipeline.named_steps["model"],
        preprocessor=pipeline.named_steps["preprocess"],
        feature_names=list(X.columns),
        target_name=str(y.name),
        dataset_key=evaluation.module,
        metrics={k: v for k, v in evaluation.metrics.items() if isinstance(v, (int, float))},
        params=dict(spec.params),
        config_fingerprint=cfg.fingerprint(),
        data_hashes=dict(data_hashes or {}),
        metadata={
            "model_name": spec.name,
            "class_path": spec.class_path,
            "task": evaluation.task,
            "scaled": spec.scale,
            "cv_folds": len(evaluation.fold_metrics) if evaluation.fold_metrics is not None else 0,
            "refit_on": "all training rows after CV",
            **(metadata or {}),
        },
    )


def train_module(
    module: str,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Any,
    specs: Sequence[ModelSpec],
    *,
    task: str,
    cfg: Config | None = None,
    split_kwargs: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    ic_groups: Sequence[Any] | None = None,
    tracker: Any | None = None,
    cost_false_negative: float | None = None,
    cost_false_positive: float | None = None,
    max_folds: int | None = None,
    pipeline_factory: Callable[..., Any] | None = None,
    on_error: str = "skip",
) -> list[EvaluationResult]:
    """Cross-validate every declared model for one module.

    A failure in one model (a missing optional library, an estimator that does
    not support a task) is logged and skipped by default rather than aborting
    the sweep - otherwise one absent package costs you the other four results.

    Args:
        on_error: ``skip`` or ``raise``.

    Returns:
        One :class:`EvaluationResult` per model that trained successfully.
    """
    cfg = cfg or load_config()
    results: list[EvaluationResult] = []

    for spec in specs:
        if spec.unsupervised:
            LOGGER.info("Skipping unsupervised model '%s' in the supervised loop.", spec.name)
            continue
        try:
            cv = cross_validate_model(
                spec, X, y, splitter, task=task, module=module, cfg=cfg,
                split_kwargs=split_kwargs, defaults=defaults,
                ic_groups=ic_groups, max_folds=max_folds,
                pipeline_factory=pipeline_factory,
            )
            if not cv.folds:
                LOGGER.warning("%s/%s produced no folds; skipping.", module, spec.name)
                continue

            evaluation = evaluate_cv(
                cv, y, cfg=cfg,
                cost_false_negative=cost_false_negative,
                cost_false_positive=cost_false_positive,
            )
            results.append(evaluation)

            if tracker is not None:
                with tracker.run(module, spec.name) as handle:
                    handle.params.update(spec.describe())
                    handle.metrics.update(
                        {k: v for k, v in evaluation.metrics.items() if isinstance(v, (int, float))}
                    )
                    tracker.log_table(handle, evaluation.fold_metrics, "folds.csv")
                    if evaluation.calibration is not None:
                        tracker.log_table(handle, evaluation.calibration, "calibration.csv")
                    if evaluation.gains is not None:
                        tracker.log_table(handle, evaluation.gains, "gains.csv")

        except Exception as exc:
            if on_error == "raise":
                raise
            LOGGER.error("%s/%s failed: %s", module, spec.name, exc)

    return results
