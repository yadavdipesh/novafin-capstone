"""
novafin-capstone/tests/test_finance.py

Tests for the finance layer and the Module 11 allocation.

The finance functions are where a silent sign error turns into a wrong Board
recommendation, so every one is checked against a value that can be verified by
hand or against a textbook reference:

* NPV and IRR against a discounting calculation anyone can reproduce;
* Black-Scholes Greeks against the standard S=K=100, T=1, sigma=0.2, r=0.05
  reference point;
* put-call parity, which is an arbitrage relationship rather than an
  approximation - if it fails, the pricing is wrong;
* VaR ordering (99% must exceed 95%, and expected shortfall must exceed VaR),
  which no correct implementation can violate.

The allocation tests pin the properties the Board would actually check: the
weights sum to the capital available, every policy limit is respected, and the
constraint-dominance diagnostic fires when the limits - rather than the
optimiser - determine the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novafin.config import load_config
from novafin.finance import (
    allocate_capital,
    allocation_narrative,
    backtest_portfolio,
    black_scholes_greeks,
    build_bucket_inputs,
    credit_decision,
    customer_lifetime_value,
    equal_weight,
    expected_npv,
    expected_shortfall,
    fund_decision,
    historical_var,
    irr,
    markowitz_weights,
    monte_carlo_var,
    npv,
    parametric_var,
    portfolio_metrics,
    prioritise_customers,
    profitability_index,
    risk_band,
    sensitivity_analysis,
    stress_test_liquidity,
)
from novafin.finance.allocate import BucketInput


# ==========================================================================
# Corporate finance
# ==========================================================================
def test_npv_matches_a_hand_computation() -> None:
    """100 for 3 years at 10% discounts to 248.685; less 250 gives -1.315."""
    assert npv([100, 100, 100], 0.10, 250) == pytest.approx(-1.3148, abs=1e-3)


def test_npv_is_zero_at_the_irr() -> None:
    """The definition of IRR, used as a self-consistency check."""
    cash_flows = [100, 100, 100]
    rate = irr(cash_flows, 250)
    assert npv(cash_flows, rate, 250) == pytest.approx(0.0, abs=1e-5)
    assert 0.09 < rate < 0.10


def test_irr_returns_nan_for_a_project_that_never_pays_back() -> None:
    assert np.isnan(irr([1, 1, 1], 1_000_000))


def test_npv_falls_as_the_discount_rate_rises() -> None:
    rates = [0.05, 0.10, 0.20]
    values = [npv([100, 100, 100], r, 200) for r in rates]
    assert values == sorted(values, reverse=True)


def test_expected_npv_is_the_briefs_formula() -> None:
    assert float(expected_npv(100.0, 0.7)) == pytest.approx(70.0)
    assert float(expected_npv(100.0, 1.0)) == pytest.approx(100.0)
    assert float(expected_npv(100.0, 0.0)) == pytest.approx(0.0)


def test_expected_npv_handles_partial_recovery() -> None:
    assert float(expected_npv(100.0, 0.5, failure_recovery=-20.0)) == pytest.approx(40.0)


def test_profitability_index_crosses_one_at_break_even() -> None:
    assert profitability_index(120, 100) == pytest.approx(1.2)
    assert profitability_index(100, 100) == pytest.approx(1.0)


def test_fund_decision_maps_to_the_three_outcomes() -> None:
    assert fund_decision(50.0)[0] == "Fund"
    assert fund_decision(-5.0, review_band=(-10.0, 0.0))[0] == "Review"
    assert fund_decision(-50.0, review_band=(-10.0, 0.0))[0] == "Reject"


def test_high_risk_initiatives_face_a_higher_bar() -> None:
    """A marginal project should clear the bar at Low risk but not at High."""
    marginal = 1.0
    assert fund_decision(marginal, risk_rating=["Low"])[0] == "Fund"
    assert fund_decision(marginal, risk_rating=["High"])[0] != "Fund"


# ==========================================================================
# Credit
# ==========================================================================
def test_risk_bands_follow_the_brief() -> None:
    bands = risk_band([0.02, 0.10, 0.22, 0.55])
    assert bands.tolist() == ["low", "medium", "high", "very_high"]


def test_credit_decision_uses_more_than_pd() -> None:
    """A low PD on an enormous loan must not be waved through."""
    decisions = credit_decision(
        pd_values=[0.02, 0.02], ecl_values=[100, 60_000],
        exposure=[100_000, 100_000],
    )
    assert decisions["decision"].iloc[0] == "Approve"
    assert decisions["decision"].iloc[1] == "Reject", (
        "an ECL of 60% of exposure must be rejected regardless of PD"
    )


def test_collateral_can_upgrade_a_borderline_loan() -> None:
    """LGD is what turns a default into a loss - collateral changes the answer."""
    base = credit_decision([0.12], [5_000], [100_000])
    secured = credit_decision([0.12], [5_000], [100_000], collateral_ratio=[2.0])
    assert base["decision"].iloc[0] == "Review"
    assert secured["decision"].iloc[0] == "Approve"


# ==========================================================================
# Customer
# ==========================================================================
def test_clv_is_finite_and_rises_with_retention() -> None:
    low = float(customer_lifetime_value([100_000], retention_rate=0.7).iloc[0])
    high = float(customer_lifetime_value([100_000], retention_rate=0.95).iloc[0])
    assert np.isfinite(low) and np.isfinite(high)
    assert high > low


def test_clv_falls_as_the_discount_rate_rises() -> None:
    cheap = float(customer_lifetime_value([100_000], discount_rate=0.05).iloc[0])
    dear = float(customer_lifetime_value([100_000], discount_rate=0.25).iloc[0])
    assert cheap > dear


def test_prioritisation_produces_the_briefs_quadrants() -> None:
    rng = np.random.default_rng(42)
    frame = prioritise_customers(rng.uniform(0, 1e5, 500), rng.uniform(0, 1, 500),
                                 contact_budget=100)
    assert set(frame["segment"].unique()) <= {
        "Protect", "Priority retention", "Maintain", "Low-cost intervention"
    }
    assert frame["contact"].sum() == 100


def test_contact_ranking_uses_value_at_risk_not_probability_alone() -> None:
    """A high-probability, low-value customer must rank below the reverse."""
    frame = prioritise_customers(clv=[100.0, 10_000.0], churn_probability=[0.9, 0.2],
                                 contact_budget=1)
    contacted = frame[frame["contact"]]
    assert float(contacted["clv"].iloc[0]) == pytest.approx(10_000.0)


# ==========================================================================
# Portfolio
# ==========================================================================
def test_equal_weight_sums_to_one() -> None:
    weights = equal_weight(15)
    assert weights.sum() == pytest.approx(1.0)
    assert len(set(weights)) == 1


def test_markowitz_weights_are_valid() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=(300, 5))
    weights = markowitz_weights(returns.mean(axis=0), np.cov(returns.T))
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-9).all(), "long_only should prevent negative weights"


def test_markowitz_falls_back_on_a_singular_covariance() -> None:
    """Estimation error is why naive Markowitz underperforms 1/N - it must not crash."""
    singular = np.ones((4, 4))
    weights = markowitz_weights(np.array([0.01] * 4), singular)
    assert weights.sum() == pytest.approx(1.0)


def test_portfolio_metrics_on_a_known_series() -> None:
    returns = pd.Series([0.01] * 252)
    metrics = portfolio_metrics(returns, risk_free_rate=0.0)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert metrics["annual_return"] > 0


def test_max_drawdown_is_negative_after_a_fall() -> None:
    returns = pd.Series([0.05, 0.05, -0.30, 0.02])
    assert portfolio_metrics(returns)["max_drawdown"] < -0.25


def test_backtest_charges_costs_only_on_turnover() -> None:
    """Charging the full notional each period is a common backtest error."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    panel = pd.DataFrame(rng.normal(0.0004, 0.01, (300, 5)), index=dates,
                         columns=[f"T{i}" for i in range(5)])

    free, _ = backtest_portfolio(panel, strategy="equal", transaction_cost_bps=0, lookback=60)
    costly, _ = backtest_portfolio(panel, strategy="equal", transaction_cost_bps=50, lookback=60)
    assert free.sum() >= costly.sum()
    # Equal weight barely trades, so the drag must be small - proof that the
    # cost is applied to turnover rather than to the whole book.
    assert (free.sum() - costly.sum()) < 0.05


# ==========================================================================
# Risk
# ==========================================================================
def test_var_rises_with_confidence() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 2000)
    assert historical_var(returns, 0.99) > historical_var(returns, 0.95)
    assert parametric_var(returns, 0.99) > parametric_var(returns, 0.95)


def test_expected_shortfall_exceeds_var() -> None:
    """ES averages the tail BEYOND VaR, so it cannot be smaller."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 5000)
    for confidence in (0.95, 0.99):
        assert expected_shortfall(returns, confidence) > historical_var(returns, confidence)


def test_var_methods_agree_on_normal_data() -> None:
    """They should agree when the data IS normal - the gap on real data is the finding."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 20_000)
    historical = historical_var(returns, 0.99)
    parametric = parametric_var(returns, 0.99)
    assert abs(historical - parametric) / parametric < 0.10


def test_monte_carlo_var_is_reproducible() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 1000)
    assert monte_carlo_var(returns, 0.99, seed=1) == monte_carlo_var(returns, 0.99, seed=1)


def test_liquidity_stress_worsens_with_severity() -> None:
    mild = stress_test_liquidity(50_000, 500, 13_000, 3_000,
                                 deposit_runoff=0.02, funding_haircut=0.10)
    severe = stress_test_liquidity(50_000, 500, 13_000, 3_000,
                                   deposit_runoff=0.10, funding_haircut=0.50)
    assert severe["coverage_ratio"] < mild["coverage_ratio"]
    assert severe["stressed_outflows"] > mild["stressed_outflows"]


# ==========================================================================
# Derivatives
# ==========================================================================
def test_greeks_match_the_textbook_reference() -> None:
    """S=100, K=100, T=1, sigma=0.2, r=0.05 - a standard reference point."""
    greeks = black_scholes_greeks(100, 100, 1.0, 0.2, 0.05, is_call=True)
    assert greeks.delta == pytest.approx(0.6368, abs=1e-3)
    assert greeks.gamma == pytest.approx(0.01876, abs=1e-4)
    assert greeks.vega == pytest.approx(0.3752, abs=1e-3)      # per 1% vol
    assert greeks.rho == pytest.approx(0.5323, abs=1e-3)       # per 1% rate


def test_call_and_put_delta_differ_by_one() -> None:
    """Put-call parity in delta form: Delta_call - Delta_put = 1. Arbitrage, not approximation."""
    call = black_scholes_greeks(120, 100, 0.75, 0.3, 0.06, is_call=True)
    put = black_scholes_greeks(120, 100, 0.75, 0.3, 0.06, is_call=False)
    assert call.delta - put.delta == pytest.approx(1.0, abs=1e-6)


def test_gamma_and_vega_are_identical_for_calls_and_puts() -> None:
    """Both depend only on d1, so they cannot differ by option type."""
    call = black_scholes_greeks(100, 100, 1.0, 0.2, 0.05, is_call=True)
    put = black_scholes_greeks(100, 100, 1.0, 0.2, 0.05, is_call=False)
    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)


def test_deep_in_the_money_call_has_delta_near_one() -> None:
    greeks = black_scholes_greeks(200, 100, 0.25, 0.2, 0.05, is_call=True)
    assert greeks.delta > 0.98


def test_theta_is_negative_for_a_long_option() -> None:
    """Time decay works against the holder."""
    assert black_scholes_greeks(100, 100, 0.5, 0.25, 0.05, is_call=True).theta < 0


# ==========================================================================
# Module 11
# ==========================================================================
@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_allocation_spends_exactly_the_capital(cfg) -> None:
    result = allocate_capital(build_bucket_inputs(cfg), cfg=cfg)
    assert result.allocations["allocation_crore"].sum() == pytest.approx(1000.0, abs=0.5)
    assert result.allocations["allocation_pct"].sum() == pytest.approx(1.0, abs=1e-4)


def test_allocation_respects_every_policy_limit(cfg) -> None:
    buckets = build_bucket_inputs(cfg)
    result = allocate_capital(buckets, cfg=cfg)
    limits = {b.name: (b.min_allocation_pct, b.max_allocation_pct) for b in buckets}
    for _, row in result.allocations.iterrows():
        low, high = limits[row["bucket"]]
        assert row["allocation_pct"] >= low - 1e-4, f"{row['bucket']} below its minimum"
        assert row["allocation_pct"] <= high + 1e-4, f"{row['bucket']} above its maximum"


def test_liquidity_reserve_floor_is_honoured(cfg) -> None:
    """The floor comes from the stress test - it is a constraint, not a preference."""
    result = allocate_capital(build_bucket_inputs(cfg), cfg=cfg)
    reserve = result.allocations[result.allocations["bucket"] == "liquidity_reserve"]
    assert float(reserve["allocation_pct"].iloc[0]) >= 0.10 - 1e-4


def test_hft_capacity_cap_binds(cfg) -> None:
    """A high return on a small base - capacity, not conviction, sets the size."""
    result = allocate_capital(build_bucket_inputs(cfg), cfg=cfg)
    hft = result.allocations[result.allocations["bucket"] == "high_frequency_trading"]
    assert float(hft["allocation_crore"].iloc[0]) <= 50.0 + 1e-6


def test_infeasible_minimums_are_rejected(cfg) -> None:
    buckets = [
        BucketInput(name=f"b{i}", expected_return=0.1, volatility=0.1,
                    min_allocation_pct=0.5, max_allocation_pct=1.0)
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="exceeds the capital available"):
        allocate_capital(buckets, cfg=cfg)


def test_constraint_dominance_is_reported(cfg) -> None:
    """When the limits determine the answer, the result must SAY so."""
    result = allocate_capital(build_bucket_inputs(cfg), cfg=cfg)
    text = " ".join(result.limitations + result.assumptions)
    if len(result.constraints_binding) >= len(result.allocations) - 1:
        assert "CONSTRAINT-DOMINATED" in text, (
            "nearly every bucket is on a bound but the result does not say so"
        )


def test_placeholder_inputs_are_flagged(cfg) -> None:
    """A placeholder must never be silently mistaken for a measured value."""
    result = allocate_capital(build_bucket_inputs(cfg), cfg=cfg)
    assert any("PLACEHOLDER" in item for item in result.limitations)


def test_measured_inputs_clear_the_placeholder_warning(cfg) -> None:
    buckets = build_bucket_inputs(
        cfg,
        credit_metrics={"source": "M2 notebook 06"},
        equity_metrics={"source": "M5/M6 notebook 06"},
        fraud_metrics={"source": "M3 notebook 06"},
        hft_metrics={"source": "M9 notebook 06"},
        derivatives_metrics={"source": "M10 notebook 06"},
        initiative_metrics={"source": "M1 notebook 06"},
        customer_metrics={"source": "M4 notebook 06"},
        liquidity_metrics={"source": "M7 notebook 06"},
    )
    result = allocate_capital(buckets, cfg=cfg)
    placeholder_warnings = [item for item in result.limitations if "PLACEHOLDER" in item]
    assert not placeholder_warnings


def test_narrative_states_the_limitations(cfg) -> None:
    """The Board paper must carry the caveats, not just the numbers."""
    narrative = allocation_narrative(allocate_capital(build_bucket_inputs(cfg), cfg=cfg))
    for heading in ("Recommended deployment", "Binding constraints", "Assumptions",
                    "Risks and limitations", "How risk is controlled"):
        assert heading in narrative
    assert "N-02" in narrative, "the zero-overlap customer finding must be disclosed"


def test_sensitivity_analysis_covers_every_setting(cfg) -> None:
    frame = sensitivity_analysis(build_bucket_inputs(cfg), cfg=cfg,
                                 risk_aversions=(1.0, 4.0))
    assert len(frame) == 2
    assert "portfolio_return_pct" in frame.columns


def test_derivatives_hedging_is_modelled_as_a_cost(cfg) -> None:
    """It protects other buckets; it does not earn. The sign must reflect that."""
    buckets = {b.name: b for b in build_bucket_inputs(cfg)}
    assert buckets["derivatives_hedging"].expected_return < 0


def test_every_bucket_records_its_evidence(cfg) -> None:
    """A Board member must be able to ask where a number came from."""
    for bucket in build_bucket_inputs(cfg):
        assert bucket.source, f"{bucket.name} has no source"
        assert bucket.evidence or bucket.notes, f"{bucket.name} has no evidence"
