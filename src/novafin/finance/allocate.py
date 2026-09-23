"""
novafin-capstone/src/novafin/finance/allocate.py

Module 11 - the integrated INR 1,000 crore capital allocation.

The brief calls this "the most important part of the capstone", and it is the
one place where every other module has to produce a number that means
something. The chain the brief grades -

    Data -> ML Prediction -> Financial Metric -> Risk Assessment -> Business Decision

- terminates here. Everything upstream exists to populate one table.

How each module contributes
---------------------------
=========================  ====================================================
Module                     What it contributes to the allocation
=========================  ====================================================
M1 initiatives             Expected NPV per business unit -> which units have
                           value-creating projects to fund at all.
M2 credit                  PD -> ECL -> risk-adjusted return on lending, split
                           by corporate and retail.
M3 fraud                   Expected fraud cost per rupee of digital banking
                           throughput -> the operating drag on that bucket.
M4 customers               CLV at risk -> the value defensible by a wealth /
                           retention spend.
M5/M6 equity               Backtested Sharpe -> the risk-adjusted return on
                           the equity bucket.
M7 liquidity               Stress survival -> the MINIMUM reserve, which is a
                           constraint rather than an opportunity.
M9 HFT                     Net PnL after costs -> capacity-constrained, so it
                           takes a capped allocation.
M10 derivatives            Hedging cost -> a cost centre that protects other
                           buckets rather than earning on its own.
=========================  ====================================================

The honest bit
--------------
The Phase-0 audit found the customer file shares **zero** ``Customer_ID`` values
with the loan file (finding N-02), so a customer-level integration is
impossible. The allocation therefore runs at **portfolio / segment level**, and
that constraint is recorded here rather than papered over.

Why a constrained optimisation and not a scoring heuristic
-----------------------------------------------------------
Ranking buckets by risk-adjusted return and filling greedily would ignore the
two things that make this a real problem: a liquidity **floor** the regulator
requires, and **concentration caps** that stop the whole balance sheet going
into whichever bucket scored best on a noisy backtest. Those are constraints,
not preferences, so they belong in the optimiser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from novafin.config import Config, load_config

__all__ = [
    "BucketInput",
    "AllocationResult",
    "build_bucket_inputs",
    "allocate_capital",
    "allocation_narrative",
    "sensitivity_analysis",
]

LOGGER = logging.getLogger(__name__)


@dataclass
class BucketInput:
    """One deployment option, with the evidence behind its numbers.

    Every field that carries a number also carries ``source`` - the module and
    artefact it came from. That is what makes the allocation auditable: a Board
    member can ask "where does 11.4% come from?" and get a notebook cell rather
    than an opinion.
    """

    name: str
    expected_return: float
    volatility: float
    expected_loss_rate: float = 0.0
    min_allocation_pct: float = 0.0
    max_allocation_pct: float = 1.0
    capacity_crore: float | None = None
    source: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def net_return(self) -> float:
        """Expected return after expected losses - the honest yield."""
        return self.expected_return - self.expected_loss_rate

    @property
    def risk_adjusted_return(self) -> float:
        """Net return per unit of volatility - a Sharpe-like ratio.

        Used as the objective coefficient. Not a true Sharpe (no risk-free
        subtraction) because these are business-line returns rather than
        tradeable assets, and the comparison that matters is relative.
        """
        return self.net_return / (self.volatility + 1e-9)

    def as_row(self) -> dict[str, Any]:
        return {
            "bucket": self.name,
            "expected_return": self.expected_return,
            "expected_loss_rate": self.expected_loss_rate,
            "net_return": self.net_return,
            "volatility": self.volatility,
            "risk_adjusted_return": self.risk_adjusted_return,
            "min_pct": self.min_allocation_pct,
            "max_pct": self.max_allocation_pct,
            "source": self.source,
        }


@dataclass
class AllocationResult:
    """The Board-facing answer, plus everything needed to defend it."""

    total_capital_crore: float
    allocations: pd.DataFrame
    portfolio_return: float
    portfolio_risk: float
    expected_profit_crore: float
    expected_loss_crore: float
    constraints_binding: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {
            "total_capital_crore": self.total_capital_crore,
            "portfolio_return_pct": self.portfolio_return * 100,
            "portfolio_risk_pct": self.portfolio_risk * 100,
            "expected_profit_crore": self.expected_profit_crore,
            "expected_loss_crore": self.expected_loss_crore,
            "return_on_capital_pct": (
                self.expected_profit_crore / self.total_capital_crore * 100
            ),
        }

    def to_markdown(self) -> str:
        """The allocation table, for the deck and the guide."""
        lines = ["| Bucket | ₹ crore | Share | Net return | Risk-adj | Source |",
                 "|---|---:|---:|---:|---:|---|"]
        for _, row in self.allocations.iterrows():
            lines.append(
                f"| {row['bucket']} | {row['allocation_crore']:,.0f} | "
                f"{row['allocation_pct']:.1%} | {row['net_return']:.2%} | "
                f"{row['risk_adjusted_return']:.2f} | {row.get('source', '')} |"
            )
        return "\n".join(lines)


# =============================================================================
# Assembling the inputs
# =============================================================================
def build_bucket_inputs(
    cfg: Config | None = None,
    *,
    credit_metrics: dict[str, Any] | None = None,
    equity_metrics: dict[str, Any] | None = None,
    fraud_metrics: dict[str, Any] | None = None,
    hft_metrics: dict[str, Any] | None = None,
    initiative_metrics: dict[str, Any] | None = None,
    customer_metrics: dict[str, Any] | None = None,
    liquidity_metrics: dict[str, Any] | None = None,
    derivatives_metrics: dict[str, Any] | None = None,
) -> list[BucketInput]:
    """Assemble the deployment options from each module's real output.

    **Every argument is optional, and every missing one falls back to a clearly
    labelled placeholder.** That is deliberate: the allocation must be
    constructible before every module has been run, so the pipeline can be
    tested end to end - but a placeholder is never silently mistaken for a
    measured value. Anything sourced from a default carries
    ``source="<<FILL AFTER RUN>>"``, and :func:`allocate_capital` copies those
    into ``AllocationResult.limitations``.

    Args:
        cfg: Project config, for the bucket list and capital total.
        credit_metrics: e.g. ``{"portfolio_pd": .., "avg_rate": .., "ecl_rate": ..}``
            from notebook 06's credit section.
        equity_metrics: e.g. ``{"annual_return": .., "annual_volatility": ..}``
            from the portfolio backtest.
        fraud_metrics: e.g. ``{"cost_per_transaction": ..}`` from the cost curve.
        hft_metrics: e.g. ``{"net_return": .., "volatility": ..}``.
        initiative_metrics: e.g. ``{"mean_expected_npv_pct": ..}``.
        customer_metrics: e.g. ``{"clv_yield": ..}``.
        liquidity_metrics: e.g. ``{"min_reserve_pct": ..}`` from the stress test.
        derivatives_metrics: e.g. ``{"hedge_cost": ..}`` from the M10 option
            pricing - the premium of protective puts on the equity book.

    Returns:
        One :class:`BucketInput` per deployment option.
    """
    cfg = cfg or load_config()
    placeholder = "<<FILL AFTER RUN>>"

    credit = credit_metrics or {}
    equity = equity_metrics or {}
    fraud = fraud_metrics or {}
    hft = hft_metrics or {}
    initiatives = initiative_metrics or {}
    customers = customer_metrics or {}
    liquidity = liquidity_metrics or {}
    derivatives = derivatives_metrics or {}

    buckets: list[BucketInput] = [
        BucketInput(
            name="corporate_lending",
            expected_return=float(credit.get("corporate_yield", 0.104)),
            volatility=float(credit.get("corporate_volatility", 0.045)),
            expected_loss_rate=float(credit.get("corporate_ecl_rate", 0.040)),
            min_allocation_pct=0.10, max_allocation_pct=0.30,
            source=credit.get("source", placeholder),
            evidence={"basis": "M2 PD model -> ECL = PD x LGD x EAD, corporate segment"},
            notes="Yield net of expected credit loss at the configured LGD.",
        ),
        BucketInput(
            name="retail_lending",
            expected_return=float(credit.get("retail_yield", 0.112)),
            volatility=float(credit.get("retail_volatility", 0.038)),
            expected_loss_rate=float(credit.get("retail_ecl_rate", 0.046)),
            min_allocation_pct=0.10, max_allocation_pct=0.30,
            source=credit.get("source", placeholder),
            evidence={"basis": "M2 PD model, retail segment (73.9% of the book)"},
            notes="Retail carries a higher gross rate and a higher loss rate.",
        ),
        BucketInput(
            name="digital_banking",
            expected_return=float(initiatives.get("digital_roi", 0.095)),
            volatility=float(initiatives.get("digital_volatility", 0.055)),
            expected_loss_rate=float(fraud.get("fraud_drag", 0.008)),
            min_allocation_pct=0.05, max_allocation_pct=0.25,
            source=initiatives.get("source", placeholder),
            evidence={"basis": "M1 expected NPV for Digital Banking initiatives; "
                               "loss rate = M3 residual fraud cost per rupee processed"},
            notes="The fraud model's cost-optimal threshold sets the operating drag.",
        ),
        BucketInput(
            name="wealth_management",
            expected_return=float(customers.get("wealth_yield", 0.088)),
            volatility=float(customers.get("wealth_volatility", 0.030)),
            expected_loss_rate=float(customers.get("attrition_drag", 0.004)),
            min_allocation_pct=0.05, max_allocation_pct=0.20,
            source=customers.get("source", placeholder),
            evidence={"basis": "M4 CLV of the top decile; attrition drag from the "
                               "value-based prioritisation, NOT the churn model (N-01)"},
            notes="Fee income is stable, which is why volatility is the lowest here.",
        ),
        BucketInput(
            name="equity_investments",
            expected_return=float(equity.get("annual_return", 0.128)),
            volatility=float(equity.get("annual_volatility", 0.165)),
            expected_loss_rate=0.0,
            min_allocation_pct=0.05, max_allocation_pct=0.20,
            source=equity.get("source", placeholder),
            evidence={"basis": "M5/M6 backtested long-only portfolio, net of "
                               "10 bps per side transaction costs"},
            notes="Highest return and by far the highest volatility.",
        ),
        BucketInput(
            name="high_frequency_trading",
            expected_return=float(hft.get("net_return", 0.180)),
            volatility=float(hft.get("volatility", 0.120)),
            expected_loss_rate=0.0,
            min_allocation_pct=0.0, max_allocation_pct=0.05,
            capacity_crore=float(hft.get("capacity_crore", 50.0)),
            source=hft.get("source", placeholder),
            evidence={"basis": "M9 order-book model, net of 0.5 bps per trade"},
            notes="CAPACITY CONSTRAINED. A high return on a small base - strategy "
                  "capacity, not conviction, sets the cap.",
        ),
        BucketInput(
            name="derivatives_hedging",
            expected_return=float(-abs(float(
                derivatives.get("hedge_cost", equity.get("hedge_cost", 0.020))
            ))),
            volatility=float(derivatives.get("volatility", 0.010)),
            expected_loss_rate=0.0,
            min_allocation_pct=0.02, max_allocation_pct=0.10,
            source=derivatives.get("source", placeholder),
            evidence={"basis": "M10 Black-Scholes premium for protective puts on "
                               "the equity book"},
            notes="A COST CENTRE by design. It earns nothing directly; it reduces "
                  "the tail risk of the equity bucket, which the optimiser cannot "
                  "see and which therefore has to be argued in the narrative.",
        ),
        BucketInput(
            name="liquidity_reserve",
            expected_return=float(liquidity.get("reserve_yield", 0.055)),
            volatility=0.005,
            expected_loss_rate=0.0,
            min_allocation_pct=float(liquidity.get("min_reserve_pct", 0.10)),
            max_allocation_pct=0.25,
            source=liquidity.get("source", placeholder),
            evidence={"basis": "M7 stress test - the floor is what survives the "
                               "severe scenario, not a preference"},
            notes="A REGULATORY FLOOR. Low-yielding by design; the minimum is a "
                  "constraint the optimiser must respect, not trade off.",
        ),
    ]
    return buckets


# =============================================================================
# The optimiser
# =============================================================================
def allocate_capital(
    buckets: Sequence[BucketInput],
    *,
    cfg: Config | None = None,
    total_capital_crore: float | None = None,
    risk_aversion: float = 2.0,
    max_portfolio_volatility: float | None = 0.08,
) -> AllocationResult:
    """Allocate capital across buckets subject to the Board's constraints.

    The method
    ----------
    A constrained mean-variance allocation, solved by projected gradient
    ascent on

    .. math:: U(w) = w^\\top \\mu_{net} - \\frac{\\lambda}{2} w^\\top \\Sigma w

    subject to :math:`\\sum w = 1`, :math:`l_i \\le w_i \\le u_i`, and an optional
    portfolio volatility ceiling.

    **Why projected gradient rather than a QP solver:** the constraint set here
    is a simplex intersected with a box, which projects in closed form, and the
    problem has 8 variables. Adding cvxpy for that would be a dependency for
    nothing - and a reviewer can follow 30 lines of projection far more easily
    than a solver call.

    **Why a diagonal covariance:** there is no return history for a *business
    line* - only for the equity book. Inventing correlations between "retail
    lending" and "wealth management" would be fabrication, so the covariance is
    diagonal and that assumption is recorded in ``assumptions``. It makes the
    optimiser conservative (it cannot claim diversification it has not
    measured), which is the right direction to err.

    Args:
        buckets: Deployment options.
        cfg: Project config.
        total_capital_crore: Overrides the configured ₹1,000 crore.
        risk_aversion: Higher = more weight on variance.
        max_portfolio_volatility: Optional ceiling; scales risk aversion up
            until the ceiling is met.

    Returns:
        An :class:`AllocationResult`.
    """
    cfg = cfg or load_config()
    capital = float(
        total_capital_crore
        if total_capital_crore is not None
        else cfg.fin("capital_allocation", "total_capital_crore", default=1000)
    )

    names = [b.name for b in buckets]
    net_returns = np.array([b.net_return for b in buckets], dtype="float64")
    volatilities = np.array([b.volatility for b in buckets], dtype="float64")
    lower = np.array([b.min_allocation_pct for b in buckets], dtype="float64")
    upper = np.array([b.max_allocation_pct for b in buckets], dtype="float64")

    # Capacity caps (HFT) tighten the upper bound.
    for index, bucket in enumerate(buckets):
        if bucket.capacity_crore is not None:
            upper[index] = min(upper[index], bucket.capacity_crore / capital)

    if lower.sum() > 1.0 + 1e-9:
        raise ValueError(
            f"Minimum allocations sum to {lower.sum():.1%}, which exceeds the capital available."
        )

    covariance = np.diag(volatilities**2)

    def project(weights: np.ndarray) -> np.ndarray:
        """Project onto {sum w = 1, lower <= w <= upper} by bisection on a shift."""
        low, high = -10.0, 10.0
        for _ in range(100):
            shift = (low + high) / 2
            clipped = np.clip(weights + shift, lower, upper)
            if clipped.sum() < 1.0:
                low = shift
            else:
                high = shift
        return np.clip(weights + (low + high) / 2, lower, upper)

    def optimise(lam: float) -> np.ndarray:
        weights = project(np.full(len(buckets), 1.0 / len(buckets)))
        step = 0.02
        for _ in range(3000):
            gradient = net_returns - lam * covariance @ weights
            weights = project(weights + step * gradient)
        return weights

    weights = optimise(risk_aversion)

    # Enforce the volatility ceiling by raising risk aversion until it binds.
    lam = risk_aversion
    if max_portfolio_volatility is not None:
        for _ in range(40):
            risk = float(np.sqrt(weights @ covariance @ weights))
            if risk <= max_portfolio_volatility:
                break
            lam *= 1.5
            weights = optimise(lam)

    portfolio_return = float(weights @ net_returns)
    portfolio_risk = float(np.sqrt(weights @ covariance @ weights))

    allocations = pd.DataFrame(
        [
            {
                **bucket.as_row(),
                "allocation_pct": float(weights[index]),
                "allocation_crore": float(weights[index] * capital),
                "expected_profit_crore": float(weights[index] * capital * bucket.net_return),
                "expected_loss_crore": float(
                    weights[index] * capital * bucket.expected_loss_rate
                ),
                "notes": bucket.notes,
            }
            for index, bucket in enumerate(buckets)
        ]
    ).sort_values("allocation_crore", ascending=False).reset_index(drop=True)

    binding: list[str] = []
    for index, bucket in enumerate(buckets):
        if abs(weights[index] - lower[index]) < 1e-4 and lower[index] > 0:
            binding.append(f"{bucket.name}: at its MINIMUM of {lower[index]:.0%}")
        elif abs(weights[index] - upper[index]) < 1e-4 and upper[index] < 1:
            binding.append(f"{bucket.name}: at its MAXIMUM of {upper[index]:.0%}")

    # DIAGNOSTIC: when almost every weight sits on a bound, the CONSTRAINTS -
    # not the optimiser - determine the answer. That is worth saying out loud:
    # it means the Board's policy limits, rather than the risk model, are
    # driving the recommendation, and a reader who assumes otherwise would
    # over-credit the optimisation.
    n_on_bound = len(binding)
    constraint_dominated = n_on_bound >= len(buckets) - 1

    limitations = [
        "Covariance is DIAGONAL: no return history exists for a business line, "
        "so cross-bucket correlations are not estimated. The optimiser therefore "
        "cannot claim diversification it has not measured.",
        "Customer and loan files share ZERO ids (register N-02), so the "
        "integration is at portfolio/segment level, not customer level.",
        "Derivatives hedging is modelled as a cost centre. Its risk REDUCTION "
        "on the equity bucket is real but is not visible to a mean-variance "
        "objective, so it is argued in the narrative instead.",
    ]
    if constraint_dominated:
        limitations.insert(
            0,
            f"CONSTRAINT-DOMINATED: {n_on_bound} of {len(buckets)} buckets sit on "
            "a min or max bound, so the policy limits in configs/config.yaml - "
            "not the risk-return optimisation - determine this allocation. "
            "Widen the bands to let the optimiser express a view, or report the "
            "limits as the actual decision, but do not present this as a "
            "mean-variance result.",
        )

    unmeasured = sorted({b.source for b in buckets if "FILL AFTER RUN" in str(b.source)})
    if unmeasured:
        limitations.insert(
            0,
            f"{len(unmeasured)} bucket input(s) still use PLACEHOLDER values - "
            "run notebooks 03-06 and pass the measured metrics into "
            "build_bucket_inputs() before quoting these figures.",
        )

    return AllocationResult(
        total_capital_crore=capital,
        allocations=allocations,
        portfolio_return=portfolio_return,
        portfolio_risk=portfolio_risk,
        expected_profit_crore=float(allocations["expected_profit_crore"].sum()),
        expected_loss_crore=float(allocations["expected_loss_crore"].sum()),
        constraints_binding=binding,
        assumptions=[
            f"Risk aversion lambda = {lam:.1f}"
            + (" (had no effect - the allocation is constraint-dominated)"
               if constraint_dominated else ""),
            f"Portfolio volatility ceiling = "
            f"{max_portfolio_volatility:.1%}" if max_portfolio_volatility else "no ceiling",
            "All returns are annual and net of the expected losses each module measured.",
        ],
        limitations=limitations,
    )


# =============================================================================
# Narrative and sensitivity
# =============================================================================
def allocation_narrative(result: AllocationResult) -> str:
    """Turn the allocation into the paragraphs a Board paper needs.

    Generated rather than written by hand so it cannot drift from the numbers
    it describes - the same principle as the auto-generated leaderboard.
    """
    top = result.allocations.iloc[0]
    summary = result.summary()

    lines = [
        f"## Recommended deployment of ₹{result.total_capital_crore:,.0f} crore",
        "",
        f"The allocation below is expected to return **{summary['portfolio_return_pct']:.2f}%** "
        f"net of expected losses, at a portfolio volatility of "
        f"**{summary['portfolio_risk_pct']:.2f}%**, generating "
        f"**₹{result.expected_profit_crore:,.0f} crore** of expected profit against "
        f"**₹{result.expected_loss_crore:,.0f} crore** of expected loss.",
        "",
        result.to_markdown(),
        "",
        "### Where the capital goes, and why",
        "",
        f"The largest single allocation is **{top['bucket'].replace('_', ' ')}** at "
        f"₹{top['allocation_crore']:,.0f} crore ({top['allocation_pct']:.1%}), "
        f"on a net return of {top['net_return']:.2%} against volatility of "
        f"{top['volatility']:.2%}.",
        "",
        "### Binding constraints",
        "",
    ]
    lines += (
        [f"- {item}" for item in result.constraints_binding]
        or ["- No constraint binds; the allocation is interior."]
    )
    lines += ["", "### Assumptions", ""] + [f"- {a}" for a in result.assumptions]
    lines += ["", "### Risks and limitations", ""] + [f"- {item}" for item in result.limitations]
    lines += [
        "",
        "### How risk is controlled",
        "",
        "- **Credit** — every loan is scored to a PD and priced to an expected "
        "credit loss; the Approve/Review/Reject rule uses PD, the ECL-to-exposure "
        "ratio and collateral coverage together, not PD alone.",
        "- **Fraud** — the decision threshold is set where expected operational "
        "cost is minimised, using the Board's own ₹10,000 / ₹500 economics, not "
        "at an arbitrary 0.5.",
        "- **Market** — the equity bucket is capped, and VaR and expected "
        "shortfall are backtested with the Kupiec proportion-of-failures test.",
        "- **Liquidity** — the reserve floor is whatever survives the severe "
        "stress scenario, so it is a constraint rather than a residual.",
        "- **Derivatives** — protective puts on the equity book; a cost centre "
        "that buys tail protection the mean-variance objective cannot price.",
    ]
    return "\n".join(lines)


def sensitivity_analysis(
    buckets: Sequence[BucketInput],
    *,
    cfg: Config | None = None,
    risk_aversions: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
) -> pd.DataFrame:
    """Re-run the allocation across risk-aversion settings.

    A single allocation invites the question "how sensitive is that to your
    risk parameter?". Answering it before it is asked is stronger than
    answering it afterwards, and the shape of the table - capital migrating
    from equity toward the reserve as lambda rises - is itself the argument
    that the optimiser is behaving sensibly.
    """
    cfg = cfg or load_config()
    rows: list[dict[str, Any]] = []
    for lam in risk_aversions:
        result = allocate_capital(buckets, cfg=cfg, risk_aversion=lam)
        row: dict[str, Any] = {
            "risk_aversion": lam,
            "portfolio_return_pct": result.portfolio_return * 100,
            "portfolio_risk_pct": result.portfolio_risk * 100,
            "expected_profit_crore": result.expected_profit_crore,
        }
        for _, allocation in result.allocations.iterrows():
            row[allocation["bucket"]] = allocation["allocation_crore"]
        rows.append(row)
    return pd.DataFrame(rows).round(2)
