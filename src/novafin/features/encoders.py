"""
novafin-capstone/src/novafin/features/encoders.py

Leak-safe encoding and preprocessing.

The rule this module exists to enforce
--------------------------------------
**Every statistic used to transform a feature must be estimated on training
rows only.** A mean for imputation, a standard deviation for scaling, a
category list for one-hot encoding, a category-level target mean - all of them
are *fitted parameters*, and fitting them on the full dataset lets test rows
influence the training representation. The resulting leak is small per feature
and utterly invisible in the metrics, which is exactly what makes it dangerous.

Concretely: ``StandardScaler().fit_transform(X)`` before splitting is wrong;
``Pipeline([...]).fit(X_train)`` inside each fold is right. Everything here is
built to make the right version the easy one.

Target encoding gets its own class
----------------------------------
:class:`OutOfFoldTargetEncoder` exists because naive target encoding is the
single most effective way to destroy a tabular project. Replacing a category
with the mean target of its rows lets each row see its **own** label. On a
high-cardinality column the model can then memorise the training set perfectly
and generalise not at all - and the cross-validation score looks superb,
because the encoding was fitted before the split.

The fix is an internal, stratified K-fold: each training row is encoded using
the *other* folds' rows only, so no row ever contributes to its own encoding.
Reference: Micci-Barreca D (2001), "A preprocessing scheme for high-cardinality
categorical attributes in classification and prediction problems",
*ACM SIGKDD Explorations* 3(1). https://doi.org/10.1145/507533.507538

No sklearn import at module level
---------------------------------
The two custom transformers are written against the plain estimator protocol
(``fit`` / ``transform`` / ``get_params`` / ``set_params``), so they can be
unit-tested without scikit-learn installed and still drop into a
``Pipeline``. :func:`build_preprocessor` imports scikit-learn lazily.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "OutOfFoldTargetEncoder",
    "FrequencyEncoder",
    "CyclicalEncoder",
    "build_preprocessor",
    "split_column_types",
]

LOGGER = logging.getLogger(__name__)


def split_column_types(
    frame: pd.DataFrame, *, max_cardinality: int = 50
) -> tuple[list[str], list[str], list[str]]:
    """Partition columns into numeric, low-cardinality and high-cardinality.

    The cardinality cut decides the encoding strategy: one-hot below it (cheap,
    interpretable, no target involvement), out-of-fold target encoding above it
    (one column instead of hundreds, at the cost of needing leak protection).

    Args:
        frame: Feature frame.
        max_cardinality: Boundary between the two categorical treatments.

    Returns:
        ``(numeric, low_cardinality_categorical, high_cardinality_categorical)``.
    """
    numeric: list[str] = []
    low: list[str] = []
    high: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            numeric.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            continue  # datetimes are consumed by the feature builders
        elif series.nunique(dropna=True) <= max_cardinality:
            low.append(column)
        else:
            high.append(column)
    return numeric, low, high


# =============================================================================
# Out-of-fold target encoding
# =============================================================================
class OutOfFoldTargetEncoder:
    """Target encoding that cannot leak, with smoothing toward the prior.

    Two mechanisms, each solving a different problem:

    **Out-of-fold fitting** solves leakage. During ``fit_transform`` the
    training rows are split into ``n_splits`` internal folds; a row in fold *i*
    is encoded from the category means of folds *!= i*. No row contributes to
    its own encoding. At ``transform`` time (validation, test, production) the
    full-training-set means are used, which is correct because those rows are
    genuinely unseen.

    **Smoothing** solves variance. A category with three rows has a target mean
    that is mostly noise. The encoding blends the category mean with the global
    prior:

    .. math::
        \\hat{y}_c = \\frac{n_c \\bar{y}_c + m \\bar{y}}{n_c + m}

    where :math:`m` is ``smoothing``. Large categories keep their own mean;
    small ones are pulled to the prior. ``m = 10`` means "trust a category once
    it has clearly more than ten observations".

    Args:
        columns: Categorical columns to encode. ``None`` encodes every object
            or category column present at fit time.
        n_splits: Internal folds used during ``fit_transform``.
        smoothing: Prior weight *m*.
        random_state: Seed for the internal fold assignment.

    Example:
        >>> encoder = OutOfFoldTargetEncoder(columns=["Merchant_Category"])
        >>> X_train_encoded = encoder.fit_transform(X_train, y_train)
        >>> X_test_encoded = encoder.transform(X_test)
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        *,
        n_splits: int = 5,
        smoothing: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.columns = list(columns) if columns is not None else None
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.mappings_: dict[str, pd.Series] = {}
        self.prior_: float = 0.0
        self.fitted_columns_: list[str] = []

    # -- estimator protocol (so this works inside a sklearn Pipeline) -------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters (scikit-learn clone protocol)."""
        return {
            "columns": self.columns,
            "n_splits": self.n_splits,
            "smoothing": self.smoothing,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "OutOfFoldTargetEncoder":
        """Set constructor parameters (scikit-learn clone protocol)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # -- internals ---------------------------------------------------------
    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in X.columns]
        return [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c])
            or isinstance(X[c].dtype, pd.CategoricalDtype)
        ]

    def _smoothed_means(self, series: pd.Series, y: pd.Series) -> pd.Series:
        grouped = y.groupby(series.astype("object"), observed=True)
        counts = grouped.count()
        means = grouped.mean()
        prior = float(y.mean())
        return (counts * means + self.smoothing * prior) / (counts + self.smoothing)

    # -- API ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OutOfFoldTargetEncoder":
        """Learn full-training-set category means (used by ``transform``)."""
        y = pd.Series(np.asarray(y), index=X.index, dtype="float64")
        self.prior_ = float(y.mean())
        self.fitted_columns_ = self._resolve_columns(X)
        self.mappings_ = {
            column: self._smoothed_means(X[column], y)
            for column in self.fitted_columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode using the fitted means; unseen categories get the prior.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self.fitted_columns_:
            raise RuntimeError("OutOfFoldTargetEncoder.transform called before fit.")
        out = X.copy()
        for column in self.fitted_columns_:
            if column not in out.columns:
                continue
            mapping = self.mappings_[column]
            out[column] = (
                out[column].astype("object").map(mapping).astype("float64")
                .fillna(self.prior_)
            )
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and encode the training set **out of fold**.

        This is the method that prevents the leak. It is deliberately NOT
        ``fit().transform()``: that would encode every training row with a mean
        it helped produce.
        """
        y = pd.Series(np.asarray(y), index=X.index, dtype="float64")
        self.fit(X, y)

        rng = np.random.default_rng(self.random_state)
        fold_of_row = rng.integers(0, self.n_splits, size=len(X))

        out = X.copy()
        for column in self.fitted_columns_:
            encoded = pd.Series(np.full(len(X), np.nan), index=X.index, dtype="float64")
            for fold in range(self.n_splits):
                holdout = fold_of_row == fold
                others = ~holdout
                if others.sum() == 0 or holdout.sum() == 0:
                    continue
                mapping = self._smoothed_means(
                    X.loc[others, column], y.loc[others]
                )
                fold_prior = float(y.loc[others].mean())
                encoded.loc[holdout] = (
                    X.loc[holdout, column].astype("object").map(mapping)
                    .astype("float64").fillna(fold_prior).to_numpy()
                )
            out[column] = encoded.fillna(self.prior_)

        LOGGER.debug(
            "Target-encoded %d column(s) out of fold across %d internal folds",
            len(self.fitted_columns_), self.n_splits,
        )
        return out


# =============================================================================
# Frequency encoding
# =============================================================================
class FrequencyEncoder:
    """Replace each category with its relative frequency in the TRAINING data.

    Useful where rarity itself is the signal - an unusual merchant category or
    an unusual device is more interesting than a common one - and it never
    touches the target, so it carries no target-leakage risk at all. It is
    still fitted on training rows only, because the frequencies themselves are
    learned parameters.
    """

    def __init__(self, columns: Sequence[str] | None = None) -> None:
        self.columns = list(columns) if columns is not None else None
        self.frequencies_: dict[str, pd.Series] = {}
        self.fitted_columns_: list[str] = []

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"columns": self.columns}

    def set_params(self, **params: Any) -> "FrequencyEncoder":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        self.fitted_columns_ = (
            [c for c in self.columns if c in X.columns]
            if self.columns is not None
            else [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        )
        self.frequencies_ = {
            column: X[column].astype("object").value_counts(normalize=True)
            for column in self.fitted_columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode; categories unseen in training map to 0.0 (maximally rare)."""
        out = X.copy()
        for column in self.fitted_columns_:
            if column in out.columns:
                out[column] = (
                    out[column].astype("object").map(self.frequencies_[column])
                    .astype("float64").fillna(0.0)
                )
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# =============================================================================
# Cyclical encoding
# =============================================================================
class CyclicalEncoder:
    """Encode a periodic integer as a (sin, cos) pair.

    Hour 23 and hour 0 are one hour apart, but as integers they are 23 apart.
    A tree can approximate the wrap-around with enough splits; a linear model
    cannot express it at all. Projecting onto the unit circle makes the
    adjacency exact for both.

    Stateless - ``fit`` exists only to satisfy the estimator protocol - and
    therefore incapable of leaking.

    Args:
        columns: Mapping of column name to its period, e.g.
            ``{"hour": 24, "day_of_week": 7, "month": 12}``.
    """

    def __init__(self, columns: dict[str, int] | None = None) -> None:
        self.columns = dict(columns or {})

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"columns": self.columns}

    def set_params(self, **params: Any) -> "CyclicalEncoder":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CyclicalEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column, period in self.columns.items():
            if column not in out.columns:
                continue
            angle = 2 * np.pi * out[column].astype("float64") / period
            out[f"{column}_sin"] = np.sin(angle)
            out[f"{column}_cos"] = np.cos(angle)
            out = out.drop(columns=[column])
        return out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        return self.transform(X)


# =============================================================================
# The preprocessor
# =============================================================================
def build_preprocessor(
    X: pd.DataFrame,
    *,
    scale: bool = True,
    max_cardinality: int = 50,
    numeric_impute: str = "median",
) -> Any:
    """Build a fold-safe ``ColumnTransformer`` for a feature frame.

    Numeric columns are imputed (median by default - robust to the heavy right
    tails throughout this data, where a mean would be dragged by the 99th
    percentile) and optionally standardised. Low-cardinality categoricals are
    one-hot encoded with ``handle_unknown="ignore"``, so a category that
    appears only in the test fold produces an all-zero row rather than an
    exception.

    **This object must be fitted inside the cross-validation loop**, never on
    the full frame. Returning an unfitted transformer rather than a transformed
    matrix is deliberate: it makes the leak-free usage the only usage.

    Args:
        X: Feature frame (used for its column types only, not its values).
        scale: Standardise numerics. Required for linear/SVM/neural models,
            irrelevant for trees.
        max_cardinality: One-hot below this many levels; above it the column is
            left for :class:`OutOfFoldTargetEncoder`.
        numeric_impute: ``median``, ``mean`` or ``most_frequent``.

    Returns:
        An unfitted ``sklearn.compose.ColumnTransformer``.

    Raises:
        ImportError: If scikit-learn is unavailable.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "build_preprocessor needs scikit-learn; install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    numeric, low_cardinality, high_cardinality = split_column_types(
        X, max_cardinality=max_cardinality
    )
    if high_cardinality:
        LOGGER.info(
            "High-cardinality column(s) left for OutOfFoldTargetEncoder: %s",
            high_cardinality,
        )

    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=numeric_impute))
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipeline, low_cardinality),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names_from(preprocessor: Any, fallback: Iterable[str]) -> list[str]:
    """Best-effort recovery of output feature names after one-hot expansion.

    Needed so a :class:`~novafin.utils.io.ModelBundle` records the *post*
    transformation contract, which is what the estimator actually consumed.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - older sklearn
        return list(fallback)
