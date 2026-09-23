"""
novafin-capstone/src/novafin/features/selection.py

Feature selection - three filters, applied in increasing order of cost.

Why select at all when gradient boosting "handles" irrelevant features
----------------------------------------------------------------------
It mostly does, but three things still argue for selection in this project:

1. **Explainability is graded.** A SHAP plot over 60 features is unreadable;
   over 20 it is a slide. The brief says model governance and explainability
   matter and that complexity is not rewarded.
2. **Noise features cost real accuracy on small data.** The initiatives module
   has 180 rows; every spurious column is another chance for a tree to split on
   coincidence.
3. **A demonstrable drop is evidence.** ``Sector_Risk`` correlates 0.029 with
   default over 3,978 distinct values - it is noise by construction. A pipeline
   that *identifies and removes* it is a much stronger exhibit than one that
   merely tolerates it.

The pipeline
------------
=========================  ====================  =================================
Stage                      Cost                  Removes
=========================  ====================  =================================
1. variance filter         O(n)                  constants and near-constants
2. correlation filter      O(k^2)                one of each redundant pair
3. null-importance filter  fits ~30 models       features no better than noise
=========================  ====================  =================================

Null importance is the only stage that is not a heuristic. It fits the model on
the real target, then repeatedly on a **shuffled** target, and keeps a feature
only when its real importance exceeds the 95th percentile of its own
null distribution. A feature that looks useful purely because it has many split
points - which is exactly how high-cardinality noise like ``Sector_Risk``
fools impurity-based importance - scores just as highly against a shuffled
target and is therefore rejected.

Reference: Altmann A, Tolosi L, Sander O, Lengauer T (2010), "Permutation
importance: a corrected feature importance measure", *Bioinformatics* 26(10).
https://doi.org/10.1093/bioinformatics/btq134
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SelectionReport",
    "variance_filter",
    "correlation_filter",
    "null_importance_filter",
    "select_features",
]

LOGGER = logging.getLogger(__name__)


@dataclass
class SelectionReport:
    """What was kept, what was dropped, and why."""

    kept: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    scores: pd.DataFrame | None = None

    def to_frame(self) -> pd.DataFrame:
        """Tabular view for the notebook and the D7 guide appendix."""
        rows = [{"feature": f, "status": "kept", "reason": ""} for f in self.kept]
        rows += [
            {"feature": f, "status": "dropped", "reason": r}
            for f, r in self.dropped.items()
        ]
        return pd.DataFrame(rows).sort_values(["status", "feature"]).reset_index(drop=True)

    def summary(self) -> str:
        return f"{len(self.kept)} kept, {len(self.dropped)} dropped"


# =============================================================================
# Stage 1 - variance
# =============================================================================
def variance_filter(
    X: pd.DataFrame, *, threshold: float = 1e-8, dominance: float = 0.995
) -> tuple[list[str], dict[str, str]]:
    """Drop constant and near-constant columns.

    Two criteria, because they catch different failures: zero variance catches
    a numeric constant, and value dominance catches a flag that is 99.5% one
    value - technically variable, practically useless, and capable of producing
    a fold in which it is genuinely constant.

    Returns:
        ``(kept, {dropped: reason})``.
    """
    kept: list[str] = []
    dropped: dict[str, str] = {}

    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            variance = float(np.nanvar(series.astype("float64")))
            if variance <= threshold:
                dropped[column] = f"variance {variance:.2e} <= {threshold:.0e}"
                continue
        counts = series.value_counts(normalize=True, dropna=False)
        if len(counts) and float(counts.iloc[0]) >= dominance:
            dropped[column] = (
                f"{counts.index[0]!r} accounts for {counts.iloc[0]:.1%} of rows"
            )
            continue
        kept.append(column)

    return kept, dropped


# =============================================================================
# Stage 2 - redundancy
# =============================================================================
def correlation_filter(
    X: pd.DataFrame,
    *,
    threshold: float = 0.95,
    priority: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Drop one column from every pair correlated above ``threshold``.

    Which one to drop is a real decision, not a coin toss. The rule here keeps
    the column that is *more interpretable*: anything in ``priority`` first,
    then the one with higher mean absolute correlation to everything else is
    dropped (it carries less unique information).

    In this project the redundancy is mostly by construction - ``total_debt``
    against ``Loan_Amount``, ``ltv`` against ``collateral_coverage``, the three
    revenue-derived initiative features - so removing it costs nothing and
    makes every coefficient and SHAP value easier to defend.

    Returns:
        ``(kept, {dropped: reason})``.
    """
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return list(X.columns), {}

    # `.to_numpy()` can return a read-only view in pandas 3, so build an
    # explicit writable copy before zeroing the diagonal.
    matrix = numeric.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(matrix, 0.0)
    corr = pd.DataFrame(matrix, index=numeric.columns, columns=numeric.columns)
    mean_corr = corr.mean()
    preferred = set(priority)

    dropped: dict[str, str] = {}
    keepers: set[str] = set()

    for a, b in ((a, b) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]):
        if a in dropped or b in dropped:
            continue
        value = float(corr.loc[a, b])
        if value < threshold:
            continue

        # A column already chosen as the survivor of an earlier pair is
        # protected. Without this, a redundancy CHAIN (x ~ y, y ~ z) can drop
        # both x and y and leave only z - silently discarding the most
        # interpretable member of the group. Observed on the equity panel,
        # where momentum was eliminated in favour of a derived ratio.
        if a in keepers and b in keepers:
            continue
        if a in keepers:
            loser, keeper = b, a
        elif b in keepers:
            loser, keeper = a, b
        elif a in preferred and b not in preferred:
            loser, keeper = b, a
        elif b in preferred and a not in preferred:
            loser, keeper = a, b
        else:
            loser, keeper = (a, b) if mean_corr[a] > mean_corr[b] else (b, a)

        dropped[loser] = f"|r| = {value:.3f} with '{keeper}'"
        keepers.add(keeper)

    kept = [c for c in X.columns if c not in dropped]
    return kept, dropped


# =============================================================================
# Stage 3 - null importance
# =============================================================================
def null_importance_filter(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_runs: int = 25,
    percentile: float = 95.0,
    task: str = "classification",
    random_state: int = 42,
    model: Any | None = None,
) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    """Keep only features that beat their own shuffled-target distribution.

    Procedure:

    1. fit the model once on the true target and record importances;
    2. refit ``n_runs`` times on a shuffled target, recording importances each
       time - this is the null distribution *for that feature*, which is the
       key point: a high-cardinality column gets a high null too;
    3. keep a feature when its real importance exceeds the ``percentile`` of
       its null distribution.

    Because each feature is compared against its own null, the test is immune
    to the cardinality bias that makes impurity importance untrustworthy.

    Args:
        X: Numeric feature frame (encode categoricals first).
        y: Target.
        n_runs: Number of shuffled refits. 25 gives a usable 95th percentile
            and costs ~25 cheap model fits.
        percentile: Null percentile a feature must clear.
        task: ``classification`` or ``regression``.
        random_state: Seed.
        model: Override estimator. Defaults to a small LightGBM.

    Returns:
        ``(kept, {dropped: reason}, scores_frame)``.

    Raises:
        ImportError: If neither LightGBM nor scikit-learn is available.
    """
    rng = np.random.default_rng(random_state)
    numeric = X.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    if model is None:
        model = _default_selector_model(task, random_state)

    def importances(target: np.ndarray) -> np.ndarray:
        estimator = _clone(model)
        estimator.fit(numeric, target)
        raw = getattr(estimator, "feature_importances_", None)
        if raw is None:  # pragma: no cover - linear fallback
            raw = np.abs(np.ravel(getattr(estimator, "coef_", np.zeros(numeric.shape[1]))))
        return np.asarray(raw, dtype="float64")

    y_values = np.asarray(y)
    actual = importances(y_values)

    null = np.zeros((n_runs, numeric.shape[1]), dtype="float64")
    for run in range(n_runs):
        shuffled = y_values.copy()
        rng.shuffle(shuffled)
        null[run] = importances(shuffled)

    cutoff = np.percentile(null, percentile, axis=0)
    scores = pd.DataFrame(
        {
            "feature": numeric.columns,
            "actual_importance": actual,
            "null_mean": null.mean(axis=0),
            f"null_p{percentile:.0f}": cutoff,
            "gain_over_null": actual - cutoff,
        }
    ).sort_values("gain_over_null", ascending=False).reset_index(drop=True)

    kept: list[str] = []
    dropped: dict[str, str] = {}
    for _, row in scores.iterrows():
        if row["gain_over_null"] > 0:
            kept.append(row["feature"])
        else:
            dropped[row["feature"]] = (
                f"importance {row['actual_importance']:.1f} <= null "
                f"p{percentile:.0f} of {row[f'null_p{percentile:.0f}']:.1f} "
                "- indistinguishable from noise"
            )

    # Non-numeric columns bypass this stage; they are handled by the encoders.
    kept += [c for c in X.columns if c not in numeric.columns]

    LOGGER.info(
        "Null-importance filter over %d runs: kept %d, dropped %d",
        n_runs, len(kept), len(dropped),
    )
    return kept, dropped, scores


def _default_selector_model(task: str, random_state: int) -> Any:
    """A small, fast model for selection - not the final estimator."""
    try:
        import lightgbm as lgb

        params = dict(
            n_estimators=120, learning_rate=0.1, num_leaves=15,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, verbose=-1, n_jobs=-1,
        )
        return lgb.LGBMClassifier(**params) if task == "classification" else lgb.LGBMRegressor(**params)
    except ImportError:
        pass
    try:
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )

        params = dict(n_estimators=150, max_depth=8, random_state=random_state, n_jobs=-1)
        return (
            RandomForestClassifier(**params) if task == "classification"
            else RandomForestRegressor(**params)
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "null_importance_filter needs LightGBM or scikit-learn; install "
            "requirements.txt."
        ) from exc


def _clone(estimator: Any) -> Any:
    """Clone an estimator, falling back to the class + params."""
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:  # pragma: no cover
        return estimator.__class__(**estimator.get_params())


# =============================================================================
# Pipeline
# =============================================================================
def select_features(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    task: str = "classification",
    correlation_threshold: float = 0.95,
    run_null_importance: bool = True,
    n_runs: int = 25,
    priority: Sequence[str] = (),
    random_state: int = 42,
    model: Any | None = None,
) -> SelectionReport:
    """Run the three filters in order and report the outcome.

    Stages are ordered cheapest-first so the expensive one operates on the
    fewest columns.

    **Fit this on training data only.** Like any learned transformation, a
    selected feature list derived from the full dataset leaks test information
    into the training representation - see ``features/encoders.py``.

    Args:
        X: Feature frame.
        y: Target. Required when ``run_null_importance`` is True.
        task: ``classification`` or ``regression``.
        correlation_threshold: Redundancy cut.
        run_null_importance: Run stage 3.
        n_runs: Shuffled refits for stage 3.
        priority: Features to prefer keeping in the correlation stage.
        random_state: Seed.
        model: Optional estimator for the null-importance stage. Exposed so a
            D4 Level-4 tuning campaign can swap the selector from YAML, and so
            the stage is testable without LightGBM installed. Defaults to a
            small LightGBM (or RandomForest) classifier/regressor.

    Returns:
        A :class:`SelectionReport`.
    """
    dropped: dict[str, str] = {}

    kept, stage_dropped = variance_filter(X)
    dropped.update({k: f"variance: {v}" for k, v in stage_dropped.items()})
    LOGGER.info("Variance filter: %d -> %d columns", X.shape[1], len(kept))

    kept, stage_dropped = correlation_filter(
        X[kept], threshold=correlation_threshold, priority=priority
    )
    dropped.update({k: f"correlation: {v}" for k, v in stage_dropped.items()})
    LOGGER.info("Correlation filter -> %d columns", len(kept))

    scores: pd.DataFrame | None = None
    if run_null_importance:
        if y is None:
            raise ValueError("null importance needs a target; pass y or disable it.")
        kept, stage_dropped, scores = null_importance_filter(
            X[kept], y, n_runs=n_runs, task=task,
            random_state=random_state, model=model,
        )
        dropped.update({k: f"null importance: {v}" for k, v in stage_dropped.items()})
        LOGGER.info("Null-importance filter -> %d columns", len(kept))

    return SelectionReport(kept=kept, dropped=dropped, scores=scores)
