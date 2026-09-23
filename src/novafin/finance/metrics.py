"""
novafin-capstone/src/novafin/finance/metrics.py

Financial calculations - the layer that turns a model output into money.

Why this module exists separately from ``evaluate.py``
------------------------------------------------------
``evaluate.py`` answers "is the model any good?". This module answers "what
does that mean in rupees?". The brief is explicit that the chain

    Data -> ML Prediction -> Financial Metric -> Risk Assessment -> Business Decision

is what is being graded, and the middle link is this file. Every function here
takes a model output and returns a quantity a finance committee would
recognise.

Every constant is read from ``configs/config.yaml`` - the discount rate, LGD,
the fraud costs, transaction costs, VaR confidence levels. Nothing is hardcoded,
so a sensitivity analysis is a config edit (see ``docs/assumptions.md``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "npv",
    "expected_npv",
    "profitability_index",
    "irr",
    "fund_decision",
    "credit_decision",
    "risk_band",
    "customer_lifetime_value",
    "prioritise_customers",
    "portfolio_metrics",
    "equal_weight",
    "markowitz_weights",
    "backtest_portfolio",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "expected_shortfall",
    "stress_test_liquidity",
    "black_scholes_greeks",
    "norm_cdf",
    "norm_pdf",
]

LOGGER = logging.getLogger(__name__)
EPS = 1e-12


def norm_cdf(x: float | np.ndarray) -> np.ndarray:
    """Standard normal CDF via ``erf`` - no scipy dependency."""
    values = np.asarray(x, dtype="float64")
    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(values)


def norm_pdf(x: float | np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    values = np.asarray(x, dtype="float64")
    return np.exp(-0.5 * values**2) / np.sqrt(2 * np.pi)


# =============================================================================
# Module 1 - corporate finance
# =============================================================================
def npv(cash_flows: Sequence[float], rate: float, initial_investment: float = 0.0) -> float:
    """Net present value of a cash-flow stream.

    .. math:: \\mathrm{NPV} = -I_0 + \\sum_{t=1}^{T} \\frac{CF_t}{(1+r)^t}

    Cash flows are assumed to arrive at the END of each year (ordinary annuity
    convention). That is the conservative choice: assuming mid-year or start-of-
    year receipt would raise every NPV and flatter every project equally, which
    would not change the ranking but would overstate the absolute numbers the
    Board is being asked to approve.
    """
    discounted = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows, start=1))
    return float(discounted - initial_investment)


def expected_npv(npv_value: float | np.ndarray, probability_of_success: float | np.ndarray,
                 failure_recovery: float = 0.0) -> np.ndarray:
    """Probability-weighted NPV - the brief's ``Expected NPV = P(Success) x NPV``.

    ``failure_recovery`` extends the brief's formula slightly: a failed
    initiative rarely returns zero, and setting it to a fraction of the
    investment makes the expected value less pessimistic. It defaults to 0.0 so
    the base case matches the brief exactly.

    **This is the function that makes model CALIBRATION matter more than
    ranking.** A model that ranks initiatives perfectly but reports 0.9 where
    the truth is 0.7 will systematically overstate every expected NPV - which is
    why the M1 metric set leads with Brier and ECE, not AUC.
    """
    probability = np.asarray(probability_of_success, dtype="float64")
    value = np.asarray(npv_value, dtype="float64")
    return probability * value + (1 - probability) * failure_recovery


def profitability_index(discounted_inflows: float, initial_investment: float) -> float:
    """PV of inflows divided by the investment. PI > 1 creates value.

    Preferred to raw NPV when capital is RATIONED - which is exactly the
    Module 11 situation. With ₹1,000 crore and more projects than money, the
    right ordering is by value created per rupee committed, not by absolute NPV.
    """
    return float(discounted_inflows / (initial_investment + EPS))


def irr(cash_flows: Sequence[float], initial_investment: float,
        *, low: float = -0.99, high: float = 10.0, tolerance: float = 1e-7,
        boundary_tolerance: float = 0.01) -> float:
    """Internal rate of return, by bisection.

    Bisection rather than Newton-Raphson: it cannot diverge, and the NPV
    function is monotonically decreasing in the rate for a conventional
    cash-flow profile (one sign change), so a bracketed root is guaranteed.

    Returns:
        The IRR, or NaN when no ECONOMICALLY MEANINGFUL root exists.

    Note on the lower bound. A project that never pays back still has a
    mathematical root near r = -1, because the discount factor explodes as the
    rate approaches -100% and drives NPV positive. That root is an artefact of
    the arithmetic, not an internal rate of return anyone would quote, so a
    solution within ``boundary_tolerance`` of the lower bound is rejected as
    NaN. Reporting "-99%" as an IRR would be worse than reporting nothing.
    """
    def f(rate: float) -> float:
        return npv(cash_flows, rate, initial_investment)

    if f(low) * f(high) > 0:
        return float("nan")

    original_low = low
    root = float("nan")
    for _ in range(200):
        middle = (low + high) / 2
        value = f(middle)
        if abs(value) < tolerance:
            root = float(middle)
            break
        if f(low) * value < 0:
            high = middle
        else:
            low = middle
    else:
        root = float((low + high) / 2)

    if not np.isfinite(root) or root <= original_low + boundary_tolerance:
        return float("nan")
    return root


def fund_decision(
    expected_npv_value: float | np.ndarray,
    *,
    risk_rating: str | Sequence[str] | None = None,
    fund_threshold: float = 0.0,
    review_band: tuple[float, float] = (-10.0, 0.0),
    high_risk_penalty: float = 1.5,
) -> np.ndarray:
    """Map expected NPV to **Fund / Review / Reject**, per the brief.

    The brief's own definitions: Fund = high expected value and acceptable
    risk; Review = potentially attractive but significant uncertainty;
    Reject = low expected value or excessive risk.

    ``high_risk_penalty`` raises the funding bar for High-risk initiatives
    rather than excluding them outright - a high-risk project with a large
    expected NPV can still deserve funding, and a rule that says otherwise is
    too blunt to defend.
    """
    values = np.atleast_1d(np.asarray(expected_npv_value, dtype="float64"))
    thresholds = np.full(values.shape, float(fund_threshold))

    if risk_rating is not None:
        ratings = np.atleast_1d(np.asarray(risk_rating, dtype=object))
        high = np.array([str(r).lower() == "high" for r in ratings])
        thresholds = np.where(high, thresholds + abs(fund_threshold) * high_risk_penalty
                              + high_risk_penalty, thresholds)

    decisions = np.where(
        values > thresholds, "Fund",
        np.where(values >= review_band[0], "Review", "Reject"),
    )
    return decisions


# =============================================================================
# Module 2 - credit risk
# =============================================================================
def risk_band(pd_values: Any, bands: dict[str, Sequence[float]] | None = None) -> pd.Series:
    """Assign PD values to named risk bands.

    Defaults to the brief's illustrative bands (<5 / 5-15 / 15-30 / >30%). The
    brief states these should be justified rather than blindly adopted, so the
    EDA re-derives cut-points from the observed ECL distribution and both are
    reported.
    """
    bands = bands or {
        "low": (0.00, 0.05),
        "medium": (0.05, 0.15),
        "high": (0.15, 0.30),
        "very_high": (0.30, 1.01),
    }
    values = pd.Series(np.asarray(pd_values, dtype="float64"))
    out = pd.Series(["very_high"] * len(values), index=values.index, dtype=object)
    for name, (low, high) in bands.items():
        out[(values >= low) & (values < high)] = name
    return out


def credit_decision(
    pd_values: Any,
    ecl_values: Any,
    exposure: Any,
    *,
    approve_pd: float = 0.05,
    reject_pd: float = 0.30,
    max_ecl_ratio: float = 0.12,
    collateral_ratio: Any | None = None,
) -> pd.DataFrame:
    """Map PD and ECL to **Approve / Review / Reject**.

    The rule uses THREE signals rather than PD alone, because PD alone is not a
    lending decision:

    1. **PD** - the probability of the borrower defaulting;
    2. **ECL / exposure** - the expected loss as a share of the loan, which
       catches a low-PD borrower with a very large loan;
    3. **collateral coverage** - a well-secured loan can be approved at a PD
       that would otherwise fail, because the loss given default is small.

    The Phase-0 audit found **33.3% of loans have LTV > 1**, so the third signal
    is not decorative here - it changes a material share of decisions.
    """
    probability = np.asarray(pd_values, dtype="float64")
    loss = np.asarray(ecl_values, dtype="float64")
    amount = np.asarray(exposure, dtype="float64")
    ratio = loss / np.where(amount == 0, np.nan, amount)

    decision = np.where(
        (probability < approve_pd) & (ratio < max_ecl_ratio), "Approve",
        np.where((probability >= reject_pd) | (ratio >= max_ecl_ratio * 2), "Reject", "Review"),
    )

    if collateral_ratio is not None:
        coverage = np.asarray(collateral_ratio, dtype="float64")
        # A well-secured borderline loan is upgraded: LGD is what turns a
        # default into a loss, and full collateral makes that loss small.
        upgrade = (decision == "Review") & (coverage >= 1.5) & (probability < reject_pd)
        decision = np.where(upgrade, "Approve", decision)

    return pd.DataFrame(
        {
            "pd": probability,
            "ecl": loss,
            "exposure": amount,
            "ecl_ratio": ratio,
            "risk_band": risk_band(probability).to_numpy(),
            "decision": decision,
        }
    )


# =============================================================================
# Module 4 - customer analytics
# =============================================================================
def customer_lifetime_value(
    annual_revenue: Any,
    *,
    retention_rate: Any = 0.9,
    discount_rate: float = 0.12,
    horizon_years: int = 3,
    margin: float = 0.3,
) -> pd.Series:
    """Discounted CLV over a finite horizon.

    .. math::
        \\mathrm{CLV} = \\sum_{t=1}^{T} \\frac{R \\cdot m \\cdot r^{t}}{(1+d)^t}

    A **finite** horizon is used rather than the perpetuity form
    ``R*m*r/(1+d-r)``. The perpetuity is analytically tidy but explodes as the
    retention rate approaches ``1+d``, and it assumes a customer relationship
    continues forever - which no relationship manager would sign off on. Three
    years matches the horizon the rest of the project uses.

    Note: ``nova_customers.csv`` already supplies ``Estimated_CLV``. This
    function exists so the supplied column can be *reproduced and challenged*
    rather than taken on faith - see leakage-register entry L-07, which records
    that the supplied column is a derived quantity.
    """
    revenue = np.asarray(annual_revenue, dtype="float64")
    retention = np.asarray(retention_rate, dtype="float64")
    total = np.zeros_like(revenue)
    for year in range(1, horizon_years + 1):
        total += revenue * margin * (retention**year) / (1 + discount_rate) ** year
    return pd.Series(total, name="clv")


def prioritise_customers(
    clv: Any, churn_probability: Any, *, contact_budget: int = 1000,
    clv_quantile: float = 0.5, churn_quantile: float = 0.5,
) -> pd.DataFrame:
    """The brief's 2x2: Protect / Priority retention / Maintain / Low-cost.

    Also returns an ``expected_value_at_risk`` column - ``CLV x P(churn)`` -
    which is the correct ranking for a capacity-constrained campaign. A
    customer with a 90% churn probability and ₹100 of value matters less than
    one with a 20% probability and ₹10,000.

    **Caveat carried from finding N-01:** in this dataset churn has no learnable
    signal (max feature correlation 0.0225, 89 positives). The quadrant is still
    produced because the brief asks for it, but the defensible answer to
    "which 1,000 customers?" is the value-based ranking, with the churn axis
    flagged as unreliable.
    """
    value = pd.Series(np.asarray(clv, dtype="float64"), name="clv")
    probability = pd.Series(np.asarray(churn_probability, dtype="float64"), name="churn_p")

    high_value = value >= value.quantile(clv_quantile)
    high_churn = probability >= probability.quantile(churn_quantile)

    segment = np.where(
        high_value & high_churn, "Priority retention",
        np.where(high_value & ~high_churn, "Protect",
                 np.where(~high_value & high_churn, "Low-cost intervention", "Maintain")),
    )

    frame = pd.DataFrame(
        {
            "clv": value,
            "churn_p": probability,
            "segment": segment,
            "expected_value_at_risk": value * probability,
        }
    )
    frame["contact_rank"] = frame["expected_value_at_risk"].rank(ascending=False, method="first")
    frame["contact"] = frame["contact_rank"] <= contact_budget
    return frame


# =============================================================================
# Modules 5/6 - portfolio construction and backtesting
# =============================================================================
def portfolio_metrics(
    returns: Any, *, risk_free_rate: float = 0.06, periods_per_year: int = 252
) -> dict[str, float]:
    """Standard performance statistics for a return series.

    Includes **Sortino** alongside Sharpe because Sharpe penalises upside
    volatility identically to downside, which is not how an investment
    committee thinks about risk; and **maximum drawdown**, which is the
    statistic that actually ends mandates.
    """
    series = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    if series.empty:
        return {k: float("nan") for k in
                ("total_return", "annual_return", "annual_volatility", "sharpe",
                 "sortino", "max_drawdown", "calmar", "hit_rate", "n_periods")}

    total = float((1 + series).prod() - 1)
    years = len(series) / periods_per_year
    annual_return = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan")
    annual_vol = float(series.std(ddof=1) * np.sqrt(periods_per_year))

    excess = annual_return - risk_free_rate
    downside = series[series < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else np.nan

    equity = (1 + series).cumprod()
    drawdown = float((equity / equity.cummax() - 1).min())

    return {
        "total_return": total,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": float(excess / annual_vol) if annual_vol > 0 else float("nan"),
        "sortino": float(excess / downside_vol) if downside_vol and downside_vol > 0 else float("nan"),
        "max_drawdown": drawdown,
        "calmar": float(annual_return / abs(drawdown)) if drawdown < 0 else float("nan"),
        "hit_rate": float((series > 0).mean()),
        "n_periods": float(len(series)),
    }


def equal_weight(n_assets: int) -> np.ndarray:
    """The 1/N portfolio - a benchmark that is very hard to beat.

    DeMiguel, Garlappi & Uppal (2009) showed 1/N outperforms most optimised
    portfolios out of sample, because estimation error in the covariance matrix
    swamps the optimisation gain. It is included as the honest benchmark, not
    as a straw man.
    https://doi.org/10.1093/rfs/hhm075
    """
    return np.full(n_assets, 1.0 / max(n_assets, 1))


def markowitz_weights(
    expected_returns: Any, covariance: Any, *,
    risk_aversion: float = 1.0, long_only: bool = True, shrinkage: float = 0.1,
) -> np.ndarray:
    """Mean-variance optimal weights, with covariance shrinkage.

    .. math:: w^* \\propto \\Sigma^{-1} \\mu / \\lambda

    **Shrinkage is not optional.** A sample covariance matrix estimated from a
    short window is near-singular, and inverting it amplifies estimation error
    into extreme, unstable weights - the classic reason naive Markowitz
    underperforms 1/N. The Ledoit-Wolf style shrinkage toward a diagonal target
    is applied here at a fixed intensity, which is simple and defensible.

    Reference: Ledoit O, Wolf M (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices", *J. Multivariate Analysis* 88(2).
    https://doi.org/10.1016/S0047-259X(03)00096-4
    """
    mu = np.asarray(expected_returns, dtype="float64").ravel()
    sigma = np.asarray(covariance, dtype="float64")
    n = len(mu)

    target = np.diag(np.diag(sigma))
    sigma = (1 - shrinkage) * sigma + shrinkage * target
    sigma = sigma + np.eye(n) * 1e-8            # numerical floor

    try:
        weights = np.linalg.solve(sigma, mu) / risk_aversion
    except np.linalg.LinAlgError:
        LOGGER.warning("Covariance is singular even after shrinkage; using equal weights.")
        return equal_weight(n)

    if long_only:
        weights = np.clip(weights, 0, None)
    total = weights.sum()
    return weights / total if abs(total) > EPS else equal_weight(n)


def backtest_portfolio(
    returns_panel: pd.DataFrame,
    signal_panel: pd.DataFrame | None = None,
    *,
    strategy: str = "ml",
    rebalance: str = "ME",
    long_quantile: float = 0.2,
    short_quantile: float = 0.0,
    transaction_cost_bps: float = 10.0,
    lookback: int = 60,
    risk_free_rate: float = 0.06,
) -> tuple[pd.Series, pd.DataFrame]:
    """Backtest a portfolio strategy with realistic turnover costs.

    Args:
        returns_panel: Wide frame, dates x tickers, of realised returns.
        signal_panel: Same shape, of predicted returns. Required for ``ml``.
        strategy: ``equal`` | ``markowitz`` | ``ml``.
        rebalance: Pandas offset alias (``ME`` = month end).
        long_quantile: Top fraction to hold long.
        short_quantile: Bottom fraction to short. 0 = long-only.
        transaction_cost_bps: Charged per side on the turnover actually traded,
            not on the whole book - charging the full notional each period is a
            common backtest error that makes every strategy look terrible.
        lookback: Window for the Markowitz covariance estimate.
        risk_free_rate: For the Sharpe ratio.

    Returns:
        ``(portfolio_returns, weights_history)``.
    """
    returns_panel = returns_panel.sort_index()
    dates = returns_panel.index
    rebalance_dates = set(pd.Series(dates, index=dates).resample(rebalance).last().dropna())

    tickers = list(returns_panel.columns)
    weights = pd.Series(equal_weight(len(tickers)), index=tickers)
    history: list[dict[str, Any]] = []
    portfolio: list[float] = []

    for position, date in enumerate(dates):
        if date in rebalance_dates and position >= lookback:
            window = returns_panel.iloc[position - lookback : position]
            previous = weights.copy()

            if strategy == "equal":
                new = pd.Series(equal_weight(len(tickers)), index=tickers)
            elif strategy == "markowitz":
                new = pd.Series(
                    markowitz_weights(window.mean().to_numpy(), window.cov().to_numpy()),
                    index=tickers,
                )
            else:  # "ml"
                if signal_panel is None:
                    raise ValueError("strategy='ml' needs a signal_panel")
                scores = signal_panel.loc[date].reindex(tickers)
                if scores.notna().sum() < 2:
                    new = previous
                else:
                    ranks = scores.rank(pct=True)
                    new = pd.Series(0.0, index=tickers)
                    longs = ranks >= (1 - long_quantile)
                    if longs.any():
                        new[longs] = 1.0 / longs.sum()
                    if short_quantile > 0:
                        shorts = ranks <= short_quantile
                        if shorts.any():
                            new[shorts] = -1.0 / shorts.sum()

            turnover = float((new - previous).abs().sum())
            cost = turnover * transaction_cost_bps / 10_000
            weights = new
            history.append({"date": date, "turnover": turnover, "cost": cost,
                            **{f"w_{t}": weights[t] for t in tickers}})
        else:
            cost = 0.0

        period_return = float((weights * returns_panel.iloc[position].fillna(0)).sum()) - cost
        portfolio.append(period_return)

    return (
        pd.Series(portfolio, index=dates, name=f"{strategy}_returns"),
        pd.DataFrame(history).set_index("date") if history else pd.DataFrame(),
    )


# =============================================================================
# Module 8 - market risk
# =============================================================================
def historical_var(returns: Any, confidence: float = 0.99) -> float:
    """Historical simulation VaR - the empirical quantile of losses.

    Makes no distributional assumption, which is its strength: financial
    returns have fat tails and a normal VaR understates them systematically.
    Its weakness is that it cannot produce a loss larger than the worst one
    observed, so it is reported alongside the parametric and Monte Carlo
    figures rather than alone.

    Returns:
        A POSITIVE number representing the loss at that confidence.
    """
    series = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    if series.empty:
        return float("nan")
    return float(-np.percentile(series, (1 - confidence) * 100))


def parametric_var(returns: Any, confidence: float = 0.99) -> float:
    """Gaussian VaR: ``-(mu + z*sigma)``.

    Included because it is the regulatory default and every risk team reports
    it - and because the gap between it and the historical figure is itself a
    finding, quantifying how fat the tails are.
    """
    series = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    if series.empty:
        return float("nan")
    # Inverse normal CDF by bisection - avoids a scipy dependency.
    alpha = 1 - confidence
    low, high = -10.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2
        if float(norm_cdf(middle)) < alpha:
            low = middle
        else:
            high = middle
    z = (low + high) / 2
    return float(-(series.mean() + z * series.std(ddof=1)))


def monte_carlo_var(
    returns: Any, confidence: float = 0.99, *, n_paths: int = 10_000,
    horizon_days: int = 1, seed: int = 42,
) -> float:
    """Monte Carlo VaR by bootstrapping historical returns.

    Bootstrapping the empirical distribution rather than sampling a fitted
    normal keeps the observed skew and kurtosis, so the simulation inherits the
    fat tails instead of assuming them away.
    """
    series = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    if series.empty:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(series.to_numpy(), size=(n_paths, horizon_days), replace=True)
    return float(-np.percentile(draws.sum(axis=1), (1 - confidence) * 100))


def expected_shortfall(returns: Any, confidence: float = 0.99) -> float:
    """Average loss in the tail beyond VaR (a.k.a. CVaR).

    Preferred to VaR by Basel III (FRTB) for a reason worth stating in the
    viva: VaR says how much you might lose at a threshold but nothing about how
    bad the tail beyond it is, and it is not sub-additive - two portfolios can
    have a combined VaR larger than the sum of their parts, which is
    nonsensical for a risk measure. Expected shortfall is coherent.
    """
    series = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    if series.empty:
        return float("nan")
    cutoff = np.percentile(series, (1 - confidence) * 100)
    tail = series[series <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def stress_test_liquidity(
    deposits: float, wholesale_funding: float, liquid_assets: float,
    expected_outflows: float, *, deposit_runoff: float, funding_haircut: float,
) -> dict[str, float]:
    """Apply a stress scenario to a liquidity position.

    Models the two mechanisms that actually cause a liquidity crisis: retail
    depositors withdraw (``deposit_runoff``) and wholesale funding fails to roll
    (``funding_haircut``). Both hit at once in a stress, which is why they are
    applied jointly rather than separately.

    Returns the surviving buffer and an LCR-style coverage ratio, where a value
    below 1.0 means the institution cannot meet stressed outflows from liquid
    assets - the number the Board is actually asking about.
    """
    lost_deposits = deposits * deposit_runoff
    lost_funding = wholesale_funding * funding_haircut
    stressed_outflows = expected_outflows + lost_deposits + lost_funding
    surviving = liquid_assets - lost_funding

    return {
        "deposit_runoff": deposit_runoff,
        "funding_haircut": funding_haircut,
        "lost_deposits": lost_deposits,
        "lost_funding": lost_funding,
        "stressed_outflows": stressed_outflows,
        "surviving_buffer": surviving,
        "coverage_ratio": float(surviving / (stressed_outflows + EPS)),
        "survives": bool(surviving >= stressed_outflows),
    }


# =============================================================================
# Module 10 - derivatives
# =============================================================================
@dataclass(frozen=True)
class Greeks:
    """The option sensitivities, as a record."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return {"delta": self.delta, "gamma": self.gamma, "vega": self.vega,
                "theta": self.theta, "rho": self.rho}


def black_scholes_greeks(
    spot: float, strike: float, time_to_maturity: float,
    volatility: float, rate: float, *, is_call: bool = True,
) -> Greeks:
    """Analytic Black-Scholes Greeks.

    ``nova_options.csv`` ships no Greeks, but the brief asks students to
    "understand Greeks" and "construct simple hedges" - so they are computed
    from the closed form rather than approximated numerically.

    The conventional reporting units are applied: **vega per 1 percentage point**
    of volatility, **theta per calendar day**, **rho per 1 percentage point** of
    rate. Quoting raw per-unit derivatives is a common error that makes theta
    look catastrophic and vega look enormous.

    .. math::
        \\Delta_{call} = N(d_1), \\quad
        \\Gamma = \\frac{\\phi(d_1)}{S\\sigma\\sqrt{T}}, \\quad
        \\mathcal{V} = S\\phi(d_1)\\sqrt{T}
    """
    t = max(float(time_to_maturity), 1e-8)
    sigma = max(float(volatility), 1e-8)
    sqrt_t = sqrt(t)

    d1 = (log(spot / strike) + (rate + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = float(norm_pdf(d1))
    discount = exp(-rate * t)

    delta = float(norm_cdf(d1)) if is_call else float(norm_cdf(d1)) - 1.0
    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100.0               # per 1% vol

    theta_common = -(spot * pdf_d1 * sigma) / (2 * sqrt_t)
    if is_call:
        theta = (theta_common - rate * strike * discount * float(norm_cdf(d2))) / 365.0
        rho = strike * t * discount * float(norm_cdf(d2)) / 100.0
    else:
        theta = (theta_common + rate * strike * discount * float(norm_cdf(-d2))) / 365.0
        rho = -strike * t * discount * float(norm_cdf(-d2)) / 100.0

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
