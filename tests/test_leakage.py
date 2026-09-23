"""
novafin-capstone/tests/test_leakage.py

The leakage register, executable.

Every entry in ``docs/LEAKAGE_REGISTER.md`` has a test here. Two layers:

* **Synthetic tests** (always run, including in CI where the course data is
  absent) construct a frame with a known leak and assert the detector finds it.
  These prove the *detectors* work.
* **Data tests** (marked ``needs_data``) run the detectors against the real
  CSVs and assert the specific documented finding still holds. These prove the
  *register* is still accurate for the data in hand.

The distinction matters: CI cannot ship the datasets, but it can still prove
that a detector has not been quietly weakened - which is the failure mode that
would let a leak back in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novafin.config import load_config
from novafin.data.loader import load_dataset, make_feature_frame
from novafin.data.validate import (
    auc_standard_error,
    detect_constant_columns,
    detect_identity_columns,
    detect_label_reconstruction,
    detect_same_row_derivation,
    detect_target_aliases,
    minimum_positives_for_cv,
    validate_dataset,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ==========================================================================
# Synthetic: the detectors themselves
# ==========================================================================
def test_identity_detector_finds_a_difference() -> None:
    """L-03 pattern: one column is the difference of two others."""
    rng = _rng()
    inflow = rng.uniform(1500, 3500, 400)
    outflow = rng.uniform(1700, 4200, 400)
    frame = pd.DataFrame(
        {"Expected_Inflows": inflow, "Expected_Outflows": outflow,
         "Liquidity_Gap": outflow - inflow, "Noise": rng.normal(size=400)}
    )
    hits = detect_identity_columns(frame, prefer=["Liquidity_Gap"])
    subjects = {h[0] for h in hits}
    assert "Liquidity_Gap" in subjects, "identity detector missed an exact difference"


def test_identity_detector_reports_the_preferred_subject() -> None:
    """Every rearrangement of an identity is true; the register's subject wins.

    This is the bug the first implementation had: it only tested ``a - b`` and
    therefore reported ``Expected_Inflows = Expected_Outflows - Liquidity_Gap``
    instead of the form a reader needs.
    """
    rng = _rng()
    a, b = rng.uniform(10, 50, 200), rng.uniform(10, 50, 200)
    frame = pd.DataFrame({"A": a, "B": b, "Gap": b - a})
    hits = detect_identity_columns(frame, prefer=["Gap"])
    assert len(hits) == 1, "one relationship must yield exactly one finding"
    assert hits[0][0] == "Gap"


def test_identity_detector_ignores_near_misses() -> None:
    rng = _rng()
    a, b = rng.uniform(10, 50, 300), rng.uniform(10, 50, 300)
    frame = pd.DataFrame({"A": a, "B": b, "AlmostGap": b - a + rng.normal(0, 5, 300)})
    assert detect_identity_columns(frame) == []


def test_alias_detector_finds_a_perfect_predictor() -> None:
    """L-04 pattern: a benchmark column that nearly *is* the target."""
    rng = _rng()
    bs = rng.uniform(1, 500, 800)
    frame = pd.DataFrame(
        {"Black_Scholes_Price": bs,
         "Market_Price": bs + rng.normal(0, 0.5, 800),
         "Strike": rng.uniform(80, 400, 800)}
    )
    aliases = detect_target_aliases(frame, "Market_Price")
    assert "Black_Scholes_Price" in aliases
    assert abs(aliases["Black_Scholes_Price"]) > 0.95
    assert "Strike" not in aliases


def test_same_row_derivation_detector() -> None:
    """L-01 pattern: the target is the same-row percentage change of a price."""
    rng = _rng()
    frames = []
    for ticker in ("AAA", "BBB"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        frames.append(pd.DataFrame({"Ticker": ticker, "Close": close}))
    panel = pd.concat(frames, ignore_index=True)
    panel["Return"] = panel.groupby("Ticker")["Close"].pct_change()
    panel = panel.dropna().reset_index(drop=True)

    corr = detect_same_row_derivation(panel, "Return", group="Ticker")
    assert corr is not None and corr > 0.999

    panel["FwdReturn"] = panel.groupby("Ticker")["Return"].shift(-1)
    clean = panel.dropna()
    forward_corr = detect_same_row_derivation(clean, "FwdReturn", group="Ticker")
    assert forward_corr is not None and abs(forward_corr) < 0.2, (
        "a properly constructed forward target must NOT match the same-row return"
    )


def test_label_reconstruction_detector() -> None:
    """L-02 pattern: a continuous column that *is* the categorical label.

    Correlation alone under-detects this because the FLAT band compresses the
    middle of the range - which is exactly why this detector exists.
    """
    rng = _rng()
    future = rng.normal(0, 1e-4, 5000)
    label = np.where(future > 5e-5, "UP", np.where(future < -5e-5, "DOWN", "FLAT"))
    frame = pd.DataFrame(
        {"Future_Return": future, "Price_Move_Class": label,
         "Unrelated": rng.normal(size=5000)}
    )
    agreement = detect_label_reconstruction(frame, "Price_Move_Class", "Future_Return")
    assert agreement is not None and agreement > 0.95

    unrelated = detect_label_reconstruction(frame, "Price_Move_Class", "Unrelated")
    assert unrelated is not None and unrelated < 0.6


def test_constant_detector_uses_cv_not_std() -> None:
    """A large mean with a tiny spread is constant; a small mean is not."""
    rng = _rng()
    frame = pd.DataFrame(
        {"Buffer": rng.normal(11385, 11.4, 1000),      # CV ~0.001 -> constant
         "Probability": rng.uniform(0.0, 1.0, 1000)}   # CV ~0.58  -> fine
    )
    constants = detect_constant_columns(frame)
    assert "Buffer" in constants
    assert "Probability" not in constants


def test_constant_detector_excludes_identifier_columns() -> None:
    """Sequential IDs are a false positive for the variance test."""
    frame = pd.DataFrame({"Customer_ID": np.arange(200001, 205001)})
    assert "Customer_ID" in detect_constant_columns(frame)
    assert detect_constant_columns(frame, exclude=["Customer_ID"]) == {}


def test_auc_standard_error_matches_hanley_mcneil() -> None:
    """Sanity-check the closed form against its own monotonicity properties."""
    small = auc_standard_error(18, 982)
    large = auc_standard_error(200, 800)
    assert small > large, "fewer positives must widen the interval"
    assert 0.06 < small < 0.08
    assert 0.02 < large < 0.03
    assert auc_standard_error(1, 100) == float("inf")


def test_cv_power_passes_for_the_credit_module() -> None:
    ok, message = minimum_positives_for_cv(997, 5, n_total=5000)
    assert ok, message
    assert "standard error" in message


def test_cv_power_fails_for_the_churn_module() -> None:
    """N-01: 89 positives cannot support a distinguishable fold-level AUC."""
    ok, message = minimum_positives_for_cv(89, 5, n_total=5000)
    assert not ok
    assert "17.8 per fold" in message
    assert "noise" in message


def test_cv_power_fails_for_the_tiny_initiatives_module() -> None:
    """180 rows is the binding constraint - repeated CV is not optional there."""
    ok, _ = minimum_positives_for_cv(138, 5, n_total=180)
    assert not ok


# ==========================================================================
# The feature boundary
# ==========================================================================
def test_make_feature_frame_removes_every_forbidden_column(cfg) -> None:
    spec = cfg.dataset("market")
    frame = pd.DataFrame(
        {c: np.arange(10, dtype="float64")
         for c in ["Open", "High", "Low", "Close", "Return", "Momentum_20D",
                   "Volatility_20D", "VIX"]}
    )
    frame["Ticker"] = "AAA"
    X, y = make_feature_frame(frame, spec)
    for leaked in ("Open", "High", "Low", "Close", "Return"):
        assert leaked not in X.columns, f"{leaked} survived the feature boundary"
    assert "Momentum_20D" in X.columns, "trailing features must be retained"
    assert y is not None and y.name == "Return"


def test_make_feature_frame_keep_is_explicit(cfg) -> None:
    """The deliberately-leaked exhibit must be opt-in, never accidental."""
    spec = cfg.dataset("market")
    frame = pd.DataFrame({c: np.arange(10, dtype="float64")
                          for c in ["Open", "High", "Low", "Close", "Return", "VIX"]})
    X, _ = make_feature_frame(frame, spec, keep=["Close"])
    assert "Close" in X.columns
    assert "Open" not in X.columns


def test_make_feature_frame_extra_drop(cfg) -> None:
    """L-06: the rate-free credit model is one argument, not a separate path."""
    spec = cfg.dataset("loans")
    frame = pd.DataFrame({c: np.arange(10, dtype="float64")
                          for c in ["Credit_Score", "Interest_Rate", "Default_Flag",
                                    "Customer_ID", "Debt_to_Income"]})
    X, _ = make_feature_frame(frame, spec, extra_drop=["Interest_Rate"])
    assert "Interest_Rate" not in X.columns
    assert "Credit_Score" in X.columns


def test_forbidden_features_are_declared_for_every_dataset(cfg) -> None:
    for key in cfg.datasets:
        spec = cfg.dataset(key)
        assert spec.target in spec.forbidden_features


# ==========================================================================
# Against the real data
# ==========================================================================
@pytest.mark.needs_data
@pytest.mark.parametrize("key", [
    "initiatives", "loans", "transactions", "customers",
    "market", "liquidity", "options", "hft",
])
def test_no_undeclared_critical_leak(cfg, key: str) -> None:
    """The headline guarantee: every leak in the data is already accounted for."""
    result = load_dataset(key, cfg=cfg, use_cache=False)
    report = validate_dataset(key, result.frame, cfg=cfg)
    assert report.ok, f"undeclared CRITICAL finding(s):\n{report}"


@pytest.mark.needs_data
def test_l01_market_return_is_same_day(cfg) -> None:
    frame = load_dataset("market", cfg=cfg, use_cache=False).frame
    corr = detect_same_row_derivation(frame, "Return", group="Ticker")
    assert corr is not None and corr > 0.999, (
        "L-01 no longer reproduces: the Return column may have changed"
    )


@pytest.mark.needs_data
def test_l02_hft_future_return_reconstructs_the_label(cfg) -> None:
    frame = load_dataset("hft", cfg=cfg, use_cache=False).frame
    agreement = detect_label_reconstruction(
        frame, "Price_Move_Class", "Future_Return_100ms"
    )
    assert agreement is not None and agreement > 0.95


@pytest.mark.needs_data
def test_l03_liquidity_gap_is_an_identity(cfg) -> None:
    frame = load_dataset("liquidity", cfg=cfg, use_cache=False).frame
    residual = float(
        (frame["Liquidity_Gap"] - (frame["Expected_Outflows"] - frame["Expected_Inflows"]))
        .abs().max()
    )
    assert residual < 0.05, f"identity no longer holds (residual {residual})"


@pytest.mark.needs_data
def test_l04_black_scholes_nearly_is_the_market_price(cfg) -> None:
    frame = load_dataset("options", cfg=cfg, use_cache=False).frame
    corr = float(frame["Black_Scholes_Price"].corr(frame["Market_Price"]))
    assert corr > 0.99


@pytest.mark.needs_data
def test_n01_churn_has_no_learnable_signal(cfg) -> None:
    """The documented negative result. If this ever fails, the data changed."""
    frame = load_dataset("customers", cfg=cfg, use_cache=False).frame
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=["Churn_Flag", "Customer_ID"]
    )
    strongest = float(numeric.corrwith(frame["Churn_Flag"].astype("float64")).abs().max())
    assert strongest < 0.05, (
        f"churn now shows a correlation of {strongest:.4f}; the N-01 negative "
        "result must be re-examined before it is repeated in the report"
    )


@pytest.mark.needs_data
def test_n02_customer_ids_do_not_join(cfg) -> None:
    loans = load_dataset("loans", cfg=cfg, use_cache=False).frame
    customers = load_dataset("customers", cfg=cfg, use_cache=False).frame
    transactions = load_dataset("transactions", cfg=cfg, use_cache=False).frame
    assert len(set(loans["Customer_ID"]) & set(customers["Customer_ID"])) == 0
    assert len(set(loans["Customer_ID"]) & set(transactions["Customer_ID"])) == 2500
