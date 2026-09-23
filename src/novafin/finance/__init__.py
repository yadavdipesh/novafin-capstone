"""
novafin-capstone/src/novafin/finance/__init__.py

The layer that turns a model output into money.

``evaluate.py`` answers "is the model any good?". This package answers "what
does that mean in rupees?" - the middle link of the chain the brief grades:

    Data -> ML Prediction -> FINANCIAL METRIC -> Risk Assessment -> Business Decision

* :mod:`novafin.finance.metrics` - NPV/IRR, ECL decisions, CLV, portfolio
  construction and backtesting, VaR/ES, liquidity stress, Black-Scholes Greeks.
* :mod:`novafin.finance.allocate` - **Module 11**, the integrated ₹1,000 crore
  allocation that every other module feeds.
"""

from __future__ import annotations

from novafin.finance.allocate import (
    AllocationResult,
    BucketInput,
    allocate_capital,
    allocation_narrative,
    build_bucket_inputs,
    sensitivity_analysis,
)
from novafin.finance.metrics import (
    Greeks,
    backtest_portfolio,
    black_scholes_greeks,
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
    stress_test_liquidity,
)

__all__ = [
    # corporate finance
    "npv", "expected_npv", "profitability_index", "irr", "fund_decision",
    # credit
    "credit_decision", "risk_band",
    # customer
    "customer_lifetime_value", "prioritise_customers",
    # portfolio
    "portfolio_metrics", "equal_weight", "markowitz_weights", "backtest_portfolio",
    # risk
    "historical_var", "parametric_var", "monte_carlo_var", "expected_shortfall",
    "stress_test_liquidity",
    # derivatives
    "black_scholes_greeks", "Greeks",
    # module 11
    "BucketInput", "AllocationResult", "build_bucket_inputs", "allocate_capital",
    "allocation_narrative", "sensitivity_analysis",
]
