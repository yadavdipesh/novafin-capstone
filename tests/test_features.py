"""
novafin-capstone/tests/test_features.py

Tests for the feature layer.

The headline tests are the **causality proofs**: for each module, build
features, corrupt only the future rows, rebuild, and assert every model feature
in the untouched prefix is bit-identical. A feature that peeked forward cannot
survive that.

Also tested, because each guards a specific and expensive mistake:

* ``causal_rolling`` really does exclude the current row (the ``shift``-then-
  ``roll`` order);
* ``forward_target`` does not bleed across entity boundaries;
* :class:`OutOfFoldTargetEncoder` produces a materially weaker - i.e. honest -
  signal than naive target encoding on a high-cardinality column;
* the correlation filter does not eliminate both ends of a redundancy chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novafin.config import load_config
from novafin.data.loader import load_dataset
from novafin.features.causal import (
    CausalityViolation,
    assert_causal,
    causal_expanding,
    causal_rolling,
    causal_shift,
    forward_target,
)
from novafin.features.encoders import (
    CyclicalEncoder,
    FrequencyEncoder,
    OutOfFoldTargetEncoder,
    split_column_types,
)
from novafin.features.engineer import (
    CUSTOMER_DECISION_COLUMNS,
    FEATURE_BUILDERS,
    black_scholes_price,
    build_features,
)
from novafin.features.selection import (
    correlation_filter,
    variance_filter,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def series_frame() -> pd.DataFrame:
    """Two entities, 50 ordered observations each, values 0..49."""
    return pd.DataFrame(
        {
            "entity": ["A"] * 50 + ["B"] * 50,
            "value": list(range(50)) + list(range(100, 150)),
        }
    )


# ==========================================================================
# Causal primitives
# ==========================================================================
def test_causal_shift_respects_group_boundaries(series_frame: pd.DataFrame) -> None:
    shifted = causal_shift(series_frame, "value", by="entity", periods=1)
    assert pd.isna(shifted.iloc[0]), "first row of a group has no history"
    assert pd.isna(shifted.iloc[50]), "B's first row must not see A's last row"
    assert shifted.iloc[1] == 0
    assert shifted.iloc[51] == 100


def test_causal_shift_rejects_zero_and_negative_periods(series_frame: pd.DataFrame) -> None:
    for periods in (0, -1):
        with pytest.raises(ValueError, match="periods >= 1"):
            causal_shift(series_frame, "value", by="entity", periods=periods)


def test_causal_rolling_excludes_the_current_row(series_frame: pd.DataFrame) -> None:
    """The whole point: shift THEN roll, not roll then shift."""
    rolled = causal_rolling(series_frame, "value", 3, "mean", by="entity")
    # Row 5 of entity A has value 5; the mean of rows 2,3,4 is 3.0.
    assert rolled.iloc[5] == pytest.approx(3.0)
    # The naive (leaky) version would include row 5 itself: mean(3,4,5) = 4.0.
    leaky = series_frame.groupby("entity")["value"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    assert leaky.iloc[5] == pytest.approx(4.0)
    assert rolled.iloc[5] != leaky.iloc[5]


def test_causal_expanding_excludes_the_current_row(series_frame: pd.DataFrame) -> None:
    expanded = causal_expanding(series_frame, "value", "mean", by="entity")
    assert pd.isna(expanded.iloc[0])
    assert expanded.iloc[4] == pytest.approx(np.mean([0, 1, 2, 3]))


def test_forward_target_does_not_bleed_across_entities(series_frame: pd.DataFrame) -> None:
    target = forward_target(series_frame, "value", by="entity", horizon=1)
    assert target.iloc[0] == 1
    assert pd.isna(target.iloc[49]), "A's last row has no future"
    assert target.iloc[50] == 101
    assert pd.isna(target.iloc[99])


def test_forward_target_rejects_non_positive_horizon(series_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="horizon >= 1"):
        forward_target(series_frame, "value", by="entity", horizon=0)


# ==========================================================================
# The causality proof itself
# ==========================================================================
def test_assert_causal_catches_a_deliberate_leak() -> None:
    """A test that never fails proves nothing - so prove it can fail."""
    frame = pd.DataFrame({"x": np.arange(200, dtype="float64")})

    def leaky_builder(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["peeks_forward"] = out["x"].shift(-1)     # looks ahead
        return out

    with pytest.raises(CausalityViolation, match="look ahead"):
        assert_causal(leaky_builder, frame, perturb_columns=["x"])


def test_assert_causal_passes_a_correct_builder() -> None:
    frame = pd.DataFrame({"x": np.arange(200, dtype="float64")})

    def causal_builder(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["lag_1"] = out["x"].shift(1)
        out["roll_5"] = out["x"].shift(1).rolling(5, min_periods=1).mean()
        return out

    report = assert_causal(causal_builder, frame, perturb_columns=["x"])
    assert report.empty


def test_assert_causal_ignores_the_declared_target() -> None:
    """A forward target is *supposed* to change - it must be exempt."""
    frame = pd.DataFrame({"x": np.arange(200, dtype="float64")})

    def builder(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["lag_1"] = out["x"].shift(1)
        out["target"] = out["x"].shift(-1)
        return out

    with pytest.raises(CausalityViolation):
        assert_causal(builder, frame, perturb_columns=["x"])
    assert assert_causal(
        builder, frame, perturb_columns=["x"], ignore_columns=["target"]
    ).empty


# ==========================================================================
# Builders - structure
# ==========================================================================
def test_every_module_has_a_builder(cfg) -> None:
    assert set(FEATURE_BUILDERS) == set(cfg.datasets)


def test_market_target_is_constructed_not_read(cfg) -> None:
    """Register L-01: the modelling target must be a forward shift."""
    rng = np.random.default_rng(42)
    frames = []
    for ticker in ("AAA", "BBB"):
        dates = pd.bdate_range("2023-01-02", periods=120)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        frames.append(
            pd.DataFrame(
                {
                    "Ticker": ticker, "Date": dates, "Sector": "Test",
                    "Open": close, "High": close * 1.01, "Low": close * 0.99,
                    "Close": close, "Volume": rng.integers(1e5, 1e6, 120),
                    "Market_Return": rng.normal(0, 0.008, 120),
                    "GDP_Growth": 6.0, "Interest_Rate": 6.5,
                    "VIX": rng.uniform(10, 25, 120),
                    "Return": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
                    "Momentum_20D": rng.normal(0, 0.05, 120),
                    "Volatility_20D": rng.uniform(0.01, 0.03, 120),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    result = build_features("market", panel, cfg)
    assert result.target == "fwd_return_1d"
    # One row per ticker is lost: the last observation has no future.
    assert result.dropped_rows == 2

    merged = result.frame
    for ticker in ("AAA", "BBB"):
        subset = merged[merged["Ticker"] == ticker].reset_index(drop=True)
        expected = subset["Return"].shift(-1).iloc[:-1]
        actual = subset["fwd_return_1d"].iloc[:-1]
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-5)


def test_customer_decision_columns_are_excluded_from_features(cfg) -> None:
    """Population ranks are a business ranking, never a model input."""
    forbidden = cfg.dataset("customers").forbidden_features
    for column in CUSTOMER_DECISION_COLUMNS:
        assert column in forbidden, (
            f"{column} is a dataset-wide percentile rank and leaks across the "
            "train/test split; it must be in drop_always"
        )


def test_black_scholes_matches_a_published_value() -> None:
    """S=100, K=100, T=1, sigma=0.2, r=0.05 -> call 10.4506, put 5.5735.

    A standard textbook reference point; independently checkable.
    """
    call = black_scholes_price(100.0, 100.0, 1.0, 0.2, 0.05, True)
    put = black_scholes_price(100.0, 100.0, 1.0, 0.2, 0.05, False)
    assert float(call) == pytest.approx(10.4506, abs=1e-3)
    assert float(put) == pytest.approx(5.5735, abs=1e-3)


def test_black_scholes_respects_put_call_parity() -> None:
    """C - P = S - K e^{-rT}. Parity is arbitrage, not approximation."""
    spot, strike, t, vol, rate = 120.0, 100.0, 0.75, 0.3, 0.06
    call = float(black_scholes_price(spot, strike, t, vol, rate, True))
    put = float(black_scholes_price(spot, strike, t, vol, rate, False))
    assert call - put == pytest.approx(spot - strike * np.exp(-rate * t), abs=1e-6)


# ==========================================================================
# Encoders
# ==========================================================================
def test_out_of_fold_encoding_removes_the_fake_signal() -> None:
    """The exhibit: a pure-noise category must not acquire predictive power."""
    rng = np.random.default_rng(42)
    n = 4000
    frame = pd.DataFrame({"cat": rng.integers(0, 400, n).astype(str)})
    y = pd.Series(rng.binomial(1, 0.05, n).astype(float))   # independent of cat

    naive = y.groupby(frame["cat"]).transform("mean")
    naive_corr = abs(float(np.corrcoef(naive, y)[0, 1]))

    encoder = OutOfFoldTargetEncoder(columns=["cat"], n_splits=5, smoothing=10.0)
    oof = encoder.fit_transform(frame, y)["cat"]
    oof_corr = abs(float(np.corrcoef(oof, y)[0, 1]))

    assert naive_corr > 0.15, "sanity: naive encoding should manufacture signal"
    assert oof_corr < 0.05, (
        f"out-of-fold encoding still leaks (|r| = {oof_corr:.3f}); a category "
        "independent of the target must not become predictive"
    )
    assert oof_corr < naive_corr / 3


def test_target_encoder_smoothing_pulls_small_categories_to_the_prior() -> None:
    frame = pd.DataFrame({"cat": ["rare"] + ["common"] * 999})
    y = pd.Series([1.0] + [0.0] * 999)
    encoder = OutOfFoldTargetEncoder(columns=["cat"], smoothing=10.0).fit(frame, y)
    encoded = encoder.transform(frame)["cat"]
    assert encoded.iloc[0] < 0.2, "a single observation must not dominate"
    assert encoder.prior_ == pytest.approx(0.001)


def test_target_encoder_maps_unseen_categories_to_the_prior() -> None:
    train = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    encoder = OutOfFoldTargetEncoder(columns=["cat"]).fit(train, y)
    encoded = encoder.transform(pd.DataFrame({"cat": ["zzz"]}))["cat"]
    assert float(encoded.iloc[0]) == pytest.approx(encoder.prior_)


def test_target_encoder_requires_fit_first() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        OutOfFoldTargetEncoder(columns=["cat"]).transform(pd.DataFrame({"cat": ["a"]}))


def test_frequency_encoder_is_fitted_on_training_only() -> None:
    train = pd.DataFrame({"cat": ["a"] * 90 + ["b"] * 10})
    encoder = FrequencyEncoder(columns=["cat"]).fit(train)
    encoded = encoder.transform(pd.DataFrame({"cat": ["a", "b", "unseen"]}))["cat"]
    assert float(encoded.iloc[0]) == pytest.approx(0.9)
    assert float(encoded.iloc[1]) == pytest.approx(0.1)
    assert float(encoded.iloc[2]) == 0.0, "unseen categories are maximally rare"


def test_cyclical_encoder_makes_hour_23_adjacent_to_hour_0() -> None:
    frame = pd.DataFrame({"hour": [0, 6, 12, 23]})
    out = CyclicalEncoder({"hour": 24}).fit_transform(frame)
    assert "hour" not in out.columns
    distance_23_to_0 = np.hypot(
        out["hour_sin"].iloc[3] - out["hour_sin"].iloc[0],
        out["hour_cos"].iloc[3] - out["hour_cos"].iloc[0],
    )
    distance_0_to_12 = np.hypot(
        out["hour_sin"].iloc[0] - out["hour_sin"].iloc[2],
        out["hour_cos"].iloc[0] - out["hour_cos"].iloc[2],
    )
    assert distance_23_to_0 < distance_0_to_12 / 4


def test_split_column_types_separates_by_cardinality() -> None:
    frame = pd.DataFrame(
        {
            "num": np.arange(100, dtype="float64"),
            "low": ["a", "b"] * 50,
            "high": [f"id_{i}" for i in range(100)],
        }
    )
    numeric, low, high = split_column_types(frame, max_cardinality=10)
    assert numeric == ["num"]
    assert low == ["low"]
    assert high == ["high"]


# ==========================================================================
# Selection
# ==========================================================================
def test_variance_filter_drops_constants_and_dominant_flags() -> None:
    frame = pd.DataFrame(
        {
            "constant": np.ones(1000),
            "dominant": [0] * 998 + [1, 1],
            "useful": np.random.default_rng(42).normal(size=1000),
        }
    )
    kept, dropped = variance_filter(frame)
    assert "useful" in kept
    assert "constant" in dropped
    assert "dominant" in dropped


def test_correlation_filter_keeps_one_of_a_redundant_pair() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(size=500)
    frame = pd.DataFrame(
        {"a": base, "b": base * 3 + 1, "c": rng.normal(size=500)}
    )
    kept, dropped = correlation_filter(frame, threshold=0.95)
    assert len(dropped) == 1
    assert "c" in kept
    assert len(kept) == 2


def test_correlation_filter_does_not_eat_a_whole_chain() -> None:
    """Regression test: x~y and y~z must not eliminate both x and y."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=800)
    frame = pd.DataFrame(
        {
            "x": base,
            "y": base + rng.normal(0, 0.01, 800),
            "z": base + rng.normal(0, 0.02, 800),
        }
    )
    kept, _ = correlation_filter(frame, threshold=0.95)
    assert len(kept) >= 1
    assert len(kept) == len(set(kept))


def test_correlation_filter_honours_priority() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(size=500)
    frame = pd.DataFrame({"interpretable": base, "derived": base * 2})
    kept, dropped = correlation_filter(frame, threshold=0.95, priority=["interpretable"])
    assert "interpretable" in kept
    assert "derived" in dropped


# ==========================================================================
# Against the real data - the module-level causality proofs
# ==========================================================================
_PERTURB = {
    "initiatives": ["Revenue_Year1", "Revenue_Year3", "Initial_Investment", "Expected_ROI"],
    "loans": ["Loan_Amount", "Collateral_Value", "Annual_Income", "Interest_Rate"],
    "transactions": ["Amount", "Historical_Avg_Transaction", "Fraud_Flag", "Timestamp"],
    "customers": ["Account_Balance", "Annual_Income", "Complaints", "Estimated_CLV"],
    "market": ["Return", "Close", "Volume", "VIX", "Momentum_20D", "Market_Return"],
    "liquidity": ["Expected_Outflows", "Expected_Inflows", "Deposits", "Market_Stress"],
    "options": ["Spot", "Strike", "Volatility", "Market_Price", "Black_Scholes_Price"],
    "hft": ["Mid_Price", "OBI_Level1", "OBI_3Level", "Trade_Volume", "Relative_Spread"],
}

#: Panel datasets are sliced to one entity so that row order equals time order.
#: Perturbing "the last 20% of rows" in a frame sorted by (entity, date) would
#: corrupt whole entities, not the future - a different test entirely.
_SINGLE_ENTITY = {"market": ("Ticker", "RELIANCE"), "hft": ("Stock", "NOVA01")}


@pytest.mark.needs_data
@pytest.mark.parametrize("key", list(_PERTURB))
def test_module_features_are_causal(cfg, key: str) -> None:
    frame = load_dataset(key, cfg=cfg, use_cache=False).frame
    if key in _SINGLE_ENTITY:
        column, value = _SINGLE_ENTITY[key]
        frame = frame[frame[column] == value].reset_index(drop=True)

    spec = cfg.dataset(key)
    result = build_features(key, frame, cfg)
    ignore = set(spec.forbidden_features) | {
        result.target, "fwd_return_5d", "fwd_inflows_5d",
    }

    report = assert_causal(
        lambda data: build_features(key, data, cfg).frame,
        frame,
        perturb_columns=_PERTURB[key],
        ignore_columns=ignore,
        tail_fraction=0.20,
    )
    assert report.empty


@pytest.mark.needs_data
def test_our_black_scholes_reproduces_the_supplied_column(cfg) -> None:
    """Validates our own maths against the dataset's own pricing column.

    Judged on RELATIVE error. The CSV stores Volatility, Interest_Rate and
    Time_to_Maturity rounded to 4 decimal places; a two-year option has rho in
    the thousands, so a 5e-5 input rounding moves the price by ~0.2 absolute
    while the implementation is correct to ~0.01%. An absolute-error threshold
    would fail a correct implementation - which is exactly what it did on the
    first run of this test.
    """
    frame = load_dataset("options", cfg=cfg, use_cache=False).frame
    result = build_features("options", frame, cfg)
    priced = result.frame[result.frame["Black_Scholes_Price"] > 1.0]
    relative = priced["bs_check_error_pct"].abs()
    assert float(relative.median()) < 1e-3, (
        f"median relative error {relative.median():.6f} - our Black-Scholes "
        "disagrees with the supplied column by more than input rounding explains"
    )
    assert float(relative.quantile(0.99)) < 1e-2
