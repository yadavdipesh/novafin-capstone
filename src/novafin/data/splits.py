"""
novafin-capstone/src/novafin/data/splits.py

Cross-validation schemes - one per module, because one scheme cannot serve all.

Why this module is not three lines of ``train_test_split``
----------------------------------------------------------
The Phase-0 audit established that these eight datasets have five genuinely
different structures: an i.i.d. cross-section of 180 rows, an i.i.d.
cross-section of 5,000, a timestamped event stream with entity repetition, a
balanced panel of 15 tickers x 1,457 days, and a 20-day intraday tape. A single
``StratifiedKFold`` applied to all of them would be defensible on exactly two
of the eight and would silently leak on the rest.

Three schemes are implemented natively here rather than delegated to
scikit-learn, because they are the leakage-critical ones and their correctness
is the argument:

* :class:`PurgedWalkForwardSplit` - splits on *unique dates*, so all 15 tickers
  of a date land in the same fold (otherwise the cross-section leaks), and
  inserts an embargo gap so a forward-looking target cannot straddle the
  boundary.
* :class:`WalkForwardByGroupSplit` - splits on whole trading days for the order
  book; an intraday random split would let 10:05 predict 10:04.
* :class:`ExpandingWindowSplit` - single-series forecasting with a fixed
  horizon gap between train end and validation start.

Every splitter yields ``(train_index, test_index)`` positional integer arrays,
matching the scikit-learn contract, so they drop into ``cross_val_score``,
``GridSearchCV`` and Optuna objectives unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from novafin.config import Config, load_config

__all__ = [
    "PurgedWalkForwardSplit",
    "WalkForwardByGroupSplit",
    "ExpandingWindowSplit",
    "TimeSeriesHoldout",
    "make_splitter",
    "holdout_by_time",
    "assert_no_temporal_overlap",
    "assert_no_group_overlap",
    "describe_splits",
]

LOGGER = logging.getLogger(__name__)

IndexPair = tuple[np.ndarray, np.ndarray]


# =============================================================================
# Native, leakage-critical splitters
# =============================================================================
@dataclass
class PurgedWalkForwardSplit:
    """Expanding-window CV over unique dates, with an embargo gap.

    Used for the equity panel (M5/M6).

    Two properties make it correct where ``KFold`` is not:

    1. **Date-level partitioning.** All rows sharing a date go to the same
       fold. Splitting on rows instead would put RELIANCE on 2023-04-11 in
       train and TCS on 2023-04-11 in test, so a model could exploit
       same-day cross-sectional information it would not have in production.
    2. **Embargo.** The target is a *forward* return spanning ``h`` days, so
       the last ``h`` training dates overlap the first test date's information
       set. Those dates are purged. The convention here is
       ``embargo >= forecast_horizon``; the config sets 5 against a 1-day
       horizon, which is deliberately conservative.

    Attributes:
        n_splits: Number of expanding folds.
        embargo: Dates dropped from the end of each training block.
        min_train_dates: Minimum dates in the first training block.

    Example:
        >>> splitter = PurgedWalkForwardSplit(n_splits=3, embargo=2)
        >>> for tr, te in splitter.split(X, dates=frame["Date"]):
        ...     ...
    """

    n_splits: int = 5
    embargo: int = 5
    min_train_dates: int = 60

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """scikit-learn compatibility."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None, *, dates: Sequence[Any] | None = None
    ) -> Iterator[IndexPair]:
        """Yield ``(train_idx, test_idx)`` for each expanding fold.

        Args:
            X: Feature matrix (used only for its length).
            y: Ignored; present for the sklearn signature.
            groups: Accepted as an alias for ``dates`` so the splitter can be
                passed to ``cross_validate(..., groups=frame["Date"])``.
            dates: The date of each row.

        Raises:
            ValueError: If no dates are supplied, or there are too few unique
                dates to honour ``min_train_dates`` and ``n_splits``.
        """
        date_values = dates if dates is not None else groups
        if date_values is None:
            raise ValueError(
                "PurgedWalkForwardSplit needs one date per row - pass dates=... "
                "or groups=frame['Date']."
            )
        series = pd.Series(pd.to_datetime(pd.Series(date_values).to_numpy()))
        unique = np.sort(series.unique())

        usable = len(unique) - self.min_train_dates - self.embargo
        if usable < self.n_splits:
            raise ValueError(
                f"{len(unique)} unique dates cannot support {self.n_splits} folds "
                f"with min_train_dates={self.min_train_dates} and embargo={self.embargo}."
            )

        fold_size = usable // self.n_splits
        positions = np.arange(len(series))

        for fold in range(self.n_splits):
            train_end = self.min_train_dates + fold * fold_size
            test_start = train_end + self.embargo
            test_end = test_start + fold_size if fold < self.n_splits - 1 else len(unique)

            train_dates = set(unique[:train_end])
            test_dates = set(unique[test_start:test_end])
            if not train_dates or not test_dates:
                continue

            train_mask = series.isin(train_dates).to_numpy()
            test_mask = series.isin(test_dates).to_numpy()
            yield positions[train_mask], positions[test_mask]


@dataclass
class WalkForwardByGroupSplit:
    """Walk forward over whole groups - trading days for the order book (M9).

    An intraday random split is the single most common way HFT results are
    inflated in student work: with 6,000 rows per day, a shuffled split puts
    10:04 and 10:06 in training and 10:05 in test, so the model interpolates
    rather than forecasts. Splitting on entire days removes that possibility.

    ``purge`` additionally drops the final rows of each training block, since
    their 100 ms forward label technically observes the first instants of the
    following block.
    """

    train_size: int = 12
    val_size: int = 4
    test_size: int = 4
    purge: int = 1
    mode: str = "single"  # "single" -> one train/val split; "rolling" -> several

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return 1 if self.mode == "single" else max(1, self.test_size)

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[IndexPair]:
        """Yield ``(train_idx, val_idx)``; the test days are held back.

        Args:
            X: Feature matrix (length only).
            y: Ignored.
            groups: The group (trading day) of each row - required.
        """
        if groups is None:
            raise ValueError("WalkForwardByGroupSplit requires groups=<trading day per row>.")

        series = pd.Series(np.asarray(groups))
        ordered = list(pd.unique(series.sort_values()))
        needed = self.train_size + self.val_size + self.test_size
        if len(ordered) < needed:
            raise ValueError(
                f"{len(ordered)} groups available but {needed} required "
                f"({self.train_size} train + {self.val_size} val + {self.test_size} test)."
            )

        positions = np.arange(len(series))
        train_groups = set(ordered[: self.train_size])
        val_groups = set(ordered[self.train_size : self.train_size + self.val_size])

        train_idx = positions[series.isin(train_groups).to_numpy()]
        if self.purge:
            train_idx = train_idx[: -self.purge] if len(train_idx) > self.purge else train_idx
        val_idx = positions[series.isin(val_groups).to_numpy()]
        yield train_idx, val_idx

    def test_index(self, groups: Sequence[Any]) -> np.ndarray:
        """Positional indices of the held-out test groups.

        Kept separate from :meth:`split` so the test days are physically
        impossible to touch during tuning - you have to ask for them.
        """
        series = pd.Series(np.asarray(groups))
        ordered = list(pd.unique(series.sort_values()))
        test_groups = set(ordered[self.train_size + self.val_size :][: self.test_size])
        return np.arange(len(series))[series.isin(test_groups).to_numpy()]


@dataclass
class ExpandingWindowSplit:
    """Expanding-window CV for a single time series (M7 liquidity).

    Differs from ``sklearn.model_selection.TimeSeriesSplit`` in one respect
    that matters here: a ``horizon`` gap is left between the end of training
    and the start of validation, matching the h-step-ahead forecast the model
    is actually asked to make. Without the gap, a 5-day-ahead forecaster is
    evaluated as if it were a 1-day-ahead forecaster.
    """

    n_splits: int = 5
    horizon: int = 5
    min_train: int = 250

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[IndexPair]:
        n = len(X)
        usable = n - self.min_train - self.horizon
        if usable < self.n_splits:
            raise ValueError(
                f"{n} observations cannot support {self.n_splits} folds with "
                f"min_train={self.min_train} and horizon={self.horizon}."
            )
        fold = usable // self.n_splits
        for i in range(self.n_splits):
            train_end = self.min_train + i * fold
            test_start = train_end + self.horizon
            test_end = test_start + fold if i < self.n_splits - 1 else n
            if test_start >= n:
                break
            yield np.arange(0, train_end), np.arange(test_start, min(test_end, n))


@dataclass
class TimeSeriesHoldout:
    """A single chronological cut - the final, untouched test period (M3).

    The fraud module holds out 2026-06-01 onwards. That block is scored exactly
    once, at the end, after every tuning decision has been made on earlier data.
    """

    cut: pd.Timestamp

    def split(self, dates: Sequence[Any]) -> IndexPair:
        """Return ``(train_idx, holdout_idx)`` around the cut date."""
        series = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy()))
        positions = np.arange(len(series))
        before = positions[(series < self.cut).to_numpy()]
        after = positions[(series >= self.cut).to_numpy()]
        if len(after) == 0:
            raise ValueError(f"No rows on or after the holdout cut {self.cut}.")
        return before, after


# =============================================================================
# Factory
# =============================================================================
def make_splitter(
    key: str,
    frame: pd.DataFrame | None = None,
    *,
    cfg: Config | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the configured splitter for a module.

    Reads ``validation.<key>`` from ``configs/config.yaml`` so the CV scheme is
    a configuration decision, never a hardcoded one.

    Args:
        key: Dataset registry key.
        frame: The loaded frame - needed to supply ``dates``/``groups``.
        cfg: Project config.

    Returns:
        ``(splitter, split_kwargs)``. Pass the kwargs straight to
        ``splitter.split(X, y, **split_kwargs)``.

    Raises:
        ValueError: For an unknown scheme name.
        ImportError: If a scikit-learn scheme is requested without sklearn
            installed (the three native schemes have no such dependency).
    """
    cfg = cfg or load_config()
    spec = cfg.dataset(key)
    recipe = cfg.splits(key)
    scheme = recipe.scheme
    kwargs: dict[str, Any] = {}

    if scheme == "purged_walk_forward":
        splitter: Any = PurgedWalkForwardSplit(
            n_splits=int(recipe.get("n_splits", 5)),
            embargo=int(recipe.get("embargo_days", 5)),
        )
        if frame is not None:
            kwargs["dates"] = frame[recipe.get("split_on", "Date")]

    elif scheme == "walk_forward_by_day":
        splitter = WalkForwardByGroupSplit(
            train_size=int(recipe.get("train_days", 12)),
            val_size=int(recipe.get("val_days", 4)),
            test_size=int(recipe.get("test_days", 4)),
            purge=int(recipe.get("purge_rows", 1)),
        )
        if frame is not None:
            kwargs["groups"] = frame[spec.group_column or "Trading_Day"]

    elif scheme == "expanding_window":
        splitter = ExpandingWindowSplit(
            n_splits=int(recipe.get("n_splits", 5)),
            horizon=int(recipe.get("horizon_days", 5)),
        )

    elif scheme == "time_series_split":
        from sklearn.model_selection import TimeSeriesSplit

        splitter = TimeSeriesSplit(n_splits=int(recipe.get("n_splits", 5)))

    elif scheme == "repeated_stratified_kfold":
        from sklearn.model_selection import RepeatedStratifiedKFold

        splitter = RepeatedStratifiedKFold(
            n_splits=int(recipe.get("n_splits", 5)),
            n_repeats=int(recipe.get("n_repeats", 10)),
            random_state=cfg.reproducibility.seed,
        )

    elif scheme in {"stratified_holdout_plus_kfold", "stratified_kfold"}:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(
            n_splits=int(recipe.get("n_splits", 5)),
            shuffle=True,
            random_state=cfg.reproducibility.seed,
        )

    elif scheme == "group_holdout":
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=int(recipe.get("n_splits", 5)))
        if frame is not None:
            group_on = recipe.get("group_on", ["Underlying"])
            column = group_on[0] if isinstance(group_on, list) else group_on
            if column in frame.columns:
                kwargs["groups"] = frame[column]

    else:
        raise ValueError(
            f"Unknown validation scheme {scheme!r} for '{key}'. "
            "Add it to make_splitter or fix configs/config.yaml."
        )

    LOGGER.info("Splitter for '%s': %s (%s)", key, type(splitter).__name__, scheme)
    return splitter, kwargs


def holdout_by_time(
    frame: pd.DataFrame, key: str, *, cfg: Config | None = None
) -> IndexPair:
    """Apply the configured chronological holdout for a module.

    Currently used by the fraud module, whose recipe carries ``holdout_from``.
    """
    cfg = cfg or load_config()
    spec = cfg.dataset(key)
    recipe = cfg.splits(key)
    cut = recipe.get("holdout_from")
    if cut is None:
        raise ValueError(f"validation.{key} has no 'holdout_from' date.")
    time_key = spec.datetime_columns[0]
    return TimeSeriesHoldout(pd.Timestamp(cut)).split(frame[time_key])


# =============================================================================
# Assertions - these are what the tests call
# =============================================================================
def assert_no_temporal_overlap(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dates: Sequence[Any],
    *,
    embargo: int = 0,
) -> None:
    """Fail if any training timestamp is at or after the first test timestamp.

    Also enforces the embargo when one is configured, by requiring a strictly
    positive gap between the last training date and the first test date.

    Raises:
        AssertionError: On overlap or an insufficient gap.
    """
    series = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy()))
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    last_train = series.iloc[train_idx].max()
    first_test = series.iloc[test_idx].min()
    if last_train >= first_test:
        raise AssertionError(
            f"Temporal leakage: last training date {last_train} is not before "
            f"the first test date {first_test}."
        )
    if embargo > 0 and (first_test - last_train) <= pd.Timedelta(0):
        raise AssertionError("Embargo gap is not positive.")


def assert_no_group_overlap(
    train_idx: np.ndarray, test_idx: np.ndarray, groups: Sequence[Any]
) -> None:
    """Fail if any group appears in both train and test.

    Raises:
        AssertionError: If the intersection is non-empty.
    """
    series = pd.Series(np.asarray(groups))
    shared = set(series.iloc[train_idx]) & set(series.iloc[test_idx])
    if shared:
        raise AssertionError(
            f"Group leakage: {len(shared)} group(s) appear in both folds, "
            f"e.g. {sorted(shared)[:5]}"
        )


def describe_splits(
    splitter: Any,
    X: pd.DataFrame,
    *,
    label_dates: Sequence[Any] | None = None,
    **split_kwargs: Any,
) -> pd.DataFrame:
    """Summarise a splitter's folds - sizes and date ranges.

    Printed in the EDA notebook and reproduced in the D7 guide, because a
    reviewer should be able to see the fold boundaries rather than take the
    scheme name on trust.

    Args:
        splitter: Any object with a scikit-learn-style ``split`` method.
        X: Feature matrix (length only).
        label_dates: Dates used **only to label** the folds. Kept separate
            from ``split_kwargs`` on purpose: ``ExpandingWindowSplit`` splits
            positionally and takes no ``dates`` argument, so forwarding the
            labelling series to every splitter would be a ``TypeError``.
        **split_kwargs: Forwarded verbatim to ``splitter.split``.

    Returns:
        One row per fold with sizes and, where dates are available, the
        training end, test start/end and the realised gap.
    """
    candidates = [label_dates, split_kwargs.get("dates"), split_kwargs.get("groups")]
    date_series = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            date_series = pd.Series(pd.to_datetime(pd.Series(candidate).to_numpy()))
            break
        except (ValueError, TypeError):
            continue

    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, **split_kwargs), start=1):
        row: dict[str, Any] = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        if date_series is not None:
            row["train_end"] = date_series.iloc[train_idx].max()
            row["test_start"] = date_series.iloc[test_idx].min()
            row["test_end"] = date_series.iloc[test_idx].max()
            row["gap"] = row["test_start"] - row["train_end"]
        rows.append(row)
    return pd.DataFrame(rows)
