"""
novafin-capstone/tests/test_splits.py

Tests for the cross-validation schemes.

A splitter is only worth having if it can be *shown* not to leak. Each test
below therefore constructs a frame with a known temporal or group structure and
asserts the property the scheme exists to guarantee - no training row at or
after a test row, no entity in two folds, a real gap where an embargo is
configured.

All tests run on synthetic data, so they execute in CI where the course CSVs
are absent, and none of them import scikit-learn: the three leakage-critical
splitters are implemented natively precisely so their correctness does not
depend on a third-party release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novafin.data.splits import (
    ExpandingWindowSplit,
    PurgedWalkForwardSplit,
    TimeSeriesHoldout,
    WalkForwardByGroupSplit,
    assert_no_group_overlap,
    assert_no_temporal_overlap,
    describe_splits,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """A balanced panel: 5 tickers x 400 business days, like the equity data."""
    dates = pd.bdate_range("2022-01-03", periods=400)
    rows = [
        {"Date": date, "Ticker": ticker, "x": float(i)}
        for i, date in enumerate(dates)
        for ticker in ("AAA", "BBB", "CCC", "DDD", "EEE")
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def tape() -> pd.DataFrame:
    """20 trading days x 300 events, like the order book."""
    rows = [
        {"Trading_Day": f"2026-01-{day:02d}", "seq": event, "x": float(event)}
        for day in range(1, 21)
        for event in range(300)
    ]
    return pd.DataFrame(rows)


# ==========================================================================
# Purged walk-forward (equity panel)
# ==========================================================================
def test_purged_walk_forward_never_trains_on_the_future(panel: pd.DataFrame) -> None:
    splitter = PurgedWalkForwardSplit(n_splits=4, embargo=5, min_train_dates=60)
    folds = list(splitter.split(panel, dates=panel["Date"]))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert_no_temporal_overlap(train_idx, test_idx, panel["Date"], embargo=5)


def test_purged_walk_forward_keeps_a_date_whole(panel: pd.DataFrame) -> None:
    """All five tickers of a date must land in the same fold.

    Splitting on rows instead of dates would let a model see four tickers'
    behaviour on a date while predicting the fifth - cross-sectional leakage
    that a plain KFold cannot avoid.
    """
    splitter = PurgedWalkForwardSplit(n_splits=3, embargo=5, min_train_dates=60)
    for train_idx, test_idx in splitter.split(panel, dates=panel["Date"]):
        assert_no_group_overlap(train_idx, test_idx, panel["Date"])
        test_dates = panel["Date"].iloc[test_idx]
        counts = test_dates.value_counts().unique()
        assert set(counts) == {5}, "a date was split across folds"


def test_purged_walk_forward_embargo_creates_a_real_gap(panel: pd.DataFrame) -> None:
    splitter = PurgedWalkForwardSplit(n_splits=3, embargo=10, min_train_dates=60)
    for train_idx, test_idx in splitter.split(panel, dates=panel["Date"]):
        last_train = panel["Date"].iloc[train_idx].max()
        first_test = panel["Date"].iloc[test_idx].min()
        assert (first_test - last_train).days >= 10


def test_purged_walk_forward_training_window_expands(panel: pd.DataFrame) -> None:
    splitter = PurgedWalkForwardSplit(n_splits=4, embargo=5, min_train_dates=60)
    sizes = [len(train) for train, _ in splitter.split(panel, dates=panel["Date"])]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_purged_walk_forward_requires_dates(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="needs one date per row"):
        list(PurgedWalkForwardSplit().split(panel))


def test_purged_walk_forward_rejects_impossible_configuration(panel: pd.DataFrame) -> None:
    splitter = PurgedWalkForwardSplit(n_splits=50, embargo=5, min_train_dates=380)
    with pytest.raises(ValueError, match="cannot support"):
        list(splitter.split(panel, dates=panel["Date"]))


# ==========================================================================
# Walk-forward by trading day (order book)
# ==========================================================================
def test_walk_forward_by_day_keeps_days_whole(tape: pd.DataFrame) -> None:
    splitter = WalkForwardByGroupSplit(train_size=12, val_size=4, test_size=4, purge=1)
    (train_idx, val_idx), = splitter.split(tape, groups=tape["Trading_Day"])
    assert_no_group_overlap(train_idx, val_idx, tape["Trading_Day"])
    assert tape["Trading_Day"].iloc[train_idx].nunique() == 12
    assert tape["Trading_Day"].iloc[val_idx].nunique() == 4


def test_walk_forward_by_day_test_set_is_separate(tape: pd.DataFrame) -> None:
    """Test days must be unreachable from ``split`` - you have to ask for them."""
    splitter = WalkForwardByGroupSplit(train_size=12, val_size=4, test_size=4)
    (train_idx, val_idx), = splitter.split(tape, groups=tape["Trading_Day"])
    test_idx = splitter.test_index(tape["Trading_Day"])

    days = tape["Trading_Day"]
    train_days = set(days.iloc[train_idx])
    val_days = set(days.iloc[val_idx])
    test_days = set(days.iloc[test_idx])
    assert not (train_days & val_days)
    assert not (train_days & test_days)
    assert not (val_days & test_days)
    assert len(test_days) == 4


def test_walk_forward_by_day_purges_the_boundary(tape: pd.DataFrame) -> None:
    unpurged, = WalkForwardByGroupSplit(purge=0).split(tape, groups=tape["Trading_Day"])
    purged, = WalkForwardByGroupSplit(purge=5).split(tape, groups=tape["Trading_Day"])
    assert len(purged[0]) == len(unpurged[0]) - 5


def test_walk_forward_by_day_needs_enough_days(tape: pd.DataFrame) -> None:
    splitter = WalkForwardByGroupSplit(train_size=30, val_size=5, test_size=5)
    with pytest.raises(ValueError, match="groups available"):
        list(splitter.split(tape, groups=tape["Trading_Day"]))


def test_walk_forward_by_day_requires_groups(tape: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="requires groups"):
        list(WalkForwardByGroupSplit().split(tape))


# ==========================================================================
# Expanding window (single series)
# ==========================================================================
def test_expanding_window_leaves_the_horizon_gap() -> None:
    """A 5-step-ahead forecaster must not be scored as a 1-step-ahead one."""
    series = pd.DataFrame({"y": np.arange(1000, dtype="float64")})
    splitter = ExpandingWindowSplit(n_splits=4, horizon=5, min_train=250)
    for train_idx, test_idx in splitter.split(series):
        assert test_idx.min() - train_idx.max() >= 5


def test_expanding_window_training_grows_and_tests_do_not_overlap() -> None:
    series = pd.DataFrame({"y": np.arange(1000, dtype="float64")})
    splitter = ExpandingWindowSplit(n_splits=4, horizon=5, min_train=250)
    folds = list(splitter.split(series))
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes)

    seen: set[int] = set()
    for _, test_idx in folds:
        assert not seen & set(test_idx.tolist()), "test blocks overlap"
        seen |= set(test_idx.tolist())


def test_expanding_window_rejects_a_series_that_is_too_short() -> None:
    series = pd.DataFrame({"y": np.arange(100, dtype="float64")})
    with pytest.raises(ValueError, match="cannot support"):
        list(ExpandingWindowSplit(n_splits=5, horizon=5, min_train=250).split(series))


# ==========================================================================
# Chronological holdout (fraud)
# ==========================================================================
def test_time_series_holdout_cuts_at_the_configured_date() -> None:
    dates = pd.date_range("2025-01-01", "2026-08-31", freq="D")
    train_idx, holdout_idx = TimeSeriesHoldout(pd.Timestamp("2026-06-01")).split(dates)
    assert pd.Series(dates).iloc[train_idx].max() < pd.Timestamp("2026-06-01")
    assert pd.Series(dates).iloc[holdout_idx].min() >= pd.Timestamp("2026-06-01")
    assert len(train_idx) + len(holdout_idx) == len(dates)


def test_time_series_holdout_rejects_an_empty_holdout() -> None:
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    with pytest.raises(ValueError, match="No rows on or after"):
        TimeSeriesHoldout(pd.Timestamp("2030-01-01")).split(dates)


# ==========================================================================
# Assertions and reporting
# ==========================================================================
def test_temporal_assertion_catches_a_shuffled_split(panel: pd.DataFrame) -> None:
    """A random split on time-series data must be detected as leakage."""
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(panel))
    train_idx, test_idx = shuffled[:1500], shuffled[1500:]
    with pytest.raises(AssertionError, match="Temporal leakage"):
        assert_no_temporal_overlap(train_idx, test_idx, panel["Date"])


def test_group_assertion_catches_a_shared_entity(tape: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="Group leakage"):
        assert_no_group_overlap(np.arange(0, 400), np.arange(200, 600), tape["Trading_Day"])


def test_describe_splits_reports_gaps(panel: pd.DataFrame) -> None:
    splitter = PurgedWalkForwardSplit(n_splits=3, embargo=7, min_train_dates=60)
    schedule = describe_splits(splitter, panel, dates=panel["Date"])
    assert len(schedule) == 3
    assert {"fold", "n_train", "n_test", "train_end", "test_start", "gap"} <= set(schedule.columns)
    assert (schedule["gap"] >= pd.Timedelta(days=7)).all()


def test_describe_splits_labels_without_forwarding(panel: pd.DataFrame) -> None:
    """``label_dates`` must not be passed to a splitter that takes no dates."""
    series = pd.DataFrame({"y": np.arange(len(panel), dtype="float64")})
    schedule = describe_splits(
        ExpandingWindowSplit(n_splits=3, horizon=5, min_train=250),
        series,
        label_dates=panel["Date"],
    )
    assert len(schedule) == 3
    assert "test_start" in schedule.columns
