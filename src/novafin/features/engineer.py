"""
novafin-capstone/src/novafin/features/engineer.py

Per-module feature construction. Every feature here is causal by construction
and proved causal by ``tests/test_features.py``.

One builder per module, registered in :data:`FEATURE_BUILDERS` and dispatched
by :func:`build_features`, so a notebook never calls a builder by name and a
new module is one registry entry rather than a new code path.

Design rules that apply to all eight builders
---------------------------------------------
1. **Targets are built here, features never look forward.** The only forward
   reference in the whole codebase is
   :func:`novafin.features.causal.forward_target`.
2. **Ratios over differences.** ``Loan_Amount / Collateral_Value`` transfers
   across loan sizes; ``Loan_Amount - Collateral_Value`` does not. Financial
   features should be scale-free wherever the underlying quantity is.
3. **No feature may encode an entity key.** Entity history enters as a causal
   aggregate, never as the id (leakage register L-09).
4. **Domain first, then interactions.** Every engineered column below has a
   finance reason to exist that can be defended in one sentence. Automatic
   polynomial expansion is deliberately not used: it produces hundreds of
   uninterpretable columns and destroys the explainability that the brief
   explicitly grades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from novafin.config import Config, load_config
from novafin.features.causal import (
    causal_expanding,
    causal_rank,
    causal_rolling,
    causal_shift,
    forward_target,
)

__all__ = [
    "FeatureResult",
    "FEATURE_BUILDERS",
    "build_features",
    "build_initiative_features",
    "build_credit_features",
    "build_fraud_features",
    "build_customer_features",
    "build_market_features",
    "build_liquidity_features",
    "build_options_features",
    "build_hft_features",
    "CUSTOMER_DECISION_COLUMNS",
    "PRIMITIVE_FEATURES",
    "black_scholes_price",
]

LOGGER = logging.getLogger(__name__)

EPS = 1e-9


@dataclass
class FeatureResult:
    """A feature frame plus the contract describing it."""

    key: str
    frame: pd.DataFrame
    target: str
    engineered: list[str] = field(default_factory=list)
    dropped_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "module": self.key,
            "rows": len(self.frame),
            "rows_dropped": self.dropped_rows,
            "engineered_features": len(self.engineered),
            "target": self.target,
            "notes": "; ".join(self.notes) or "none",
        }


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN rather than inf on a zero.

    An infinity propagates silently through a scaler and poisons a whole
    feature; a NaN is caught by the imputer and logged.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# M1 - Strategic initiatives (corporate finance)
# =============================================================================
def build_initiative_features(
    frame: pd.DataFrame, cfg: Config | None = None
) -> FeatureResult:
    """Capital-budgeting features for ``strategic_initiatives.csv``.

    The finance content matters more than the ML here: the module's deliverable
    is an Expected NPV and a Fund / Review / Reject decision, so the features
    are the components of that calculation. NPV is computed at the configured
    discount rate with the three supplied revenue years, net of operating cost.

    Note on ``Expected_ROI``: leakage-register L-08 records that it correlates
    0.52 with a crude recomputed ROI. It is retained (it is the strongest
    single predictor at 0.121) but ``roi_gap`` below makes the redundancy
    explicit and measurable rather than hidden.
    """
    cfg = cfg or load_config()
    out = frame.copy()
    rate = float(cfg.fin("npv", "discount_rate", default=0.12))
    engineered: list[str] = []

    revenues = ["Revenue_Year1", "Revenue_Year2", "Revenue_Year3"]
    out["total_revenue"] = out[revenues].sum(axis=1)
    out["total_cost"] = out["Operating_Cost"] * 3 + out["Initial_Investment"]
    out["gross_profit"] = out["total_revenue"] - out["Operating_Cost"] * 3

    # Discounted cash flows: year-t net cash flow = revenue_t - operating cost.
    discounted = sum(
        (out[column] - out["Operating_Cost"]) / (1 + rate) ** year
        for year, column in enumerate(revenues, start=1)
    )
    out["npv"] = discounted - out["Initial_Investment"]
    out["profitability_index"] = _safe_divide(discounted, out["Initial_Investment"])
    out["npv_per_rupee"] = _safe_divide(out["npv"], out["Initial_Investment"])

    # Revenue trajectory: growing initiatives are worth more than front-loaded
    # ones of equal total, and the ratio is scale-free.
    out["revenue_cagr"] = (
        _safe_divide(out["Revenue_Year3"], out["Revenue_Year1"]) ** (1 / 2) - 1
    )
    out["revenue_growth_y2"] = _safe_divide(out["Revenue_Year2"], out["Revenue_Year1"]) - 1
    out["revenue_growth_y3"] = _safe_divide(out["Revenue_Year3"], out["Revenue_Year2"]) - 1
    out["cost_to_revenue"] = _safe_divide(out["Operating_Cost"] * 3, out["total_revenue"])
    out["margin"] = _safe_divide(out["gross_profit"], out["total_revenue"])

    # Simple payback in years, capped at the 3-year horizon the data supports.
    cumulative = out["Revenue_Year1"] - out["Operating_Cost"]
    payback = pd.Series(np.full(len(out), 4.0), index=out.index)
    payback = payback.mask(cumulative >= out["Initial_Investment"], 1.0)
    cumulative = cumulative + (out["Revenue_Year2"] - out["Operating_Cost"])
    payback = payback.mask((payback == 4.0) & (cumulative >= out["Initial_Investment"]), 2.0)
    cumulative = cumulative + (out["Revenue_Year3"] - out["Operating_Cost"])
    payback = payback.mask((payback == 4.0) & (cumulative >= out["Initial_Investment"]), 3.0)
    out["payback_years"] = payback

    # Macro and competitive context.
    out["growth_vs_gdp"] = out["Business_Growth"] - out["GDP_Growth"]
    out["roi_vs_rate"] = out["Expected_ROI"] - out["Interest_Rate"]
    out["roi_gap"] = out["Expected_ROI"] - _safe_divide(
        out["gross_profit"] - out["Initial_Investment"], out["Initial_Investment"]
    ) * 100
    out["competition_pressure"] = out["Competition_Index"] / 100.0
    out["risk_adjusted_roi"] = out["Expected_ROI"] * (1 - out["competition_pressure"])
    out["investment_log"] = np.log1p(out["Initial_Investment"])

    engineered = [
        "total_revenue", "total_cost", "gross_profit", "npv", "profitability_index",
        "npv_per_rupee", "revenue_cagr", "revenue_growth_y2", "revenue_growth_y3",
        "cost_to_revenue", "margin", "payback_years", "growth_vs_gdp", "roi_vs_rate",
        "roi_gap", "competition_pressure", "risk_adjusted_roi", "investment_log",
    ]
    return FeatureResult(
        key="initiatives", frame=out, target="Historical_Success",
        engineered=engineered,
        notes=[
            f"NPV discounted at {rate:.0%} over 3 years (configs/config.yaml)",
            "no time column - features are row-wise, no causality risk",
        ],
    )


# =============================================================================
# M2 - Credit risk
# =============================================================================
def build_credit_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Underwriting features for ``nova_loans.csv``.

    Every feature is a ratio a credit officer would actually compute. The file
    has no origination date, so all features are row-wise and no causality
    question arises - which is itself worth stating, because it is the reason
    out-of-time validation is impossible for this module (N-05).
    """
    cfg = cfg or load_config()
    out = frame.copy()

    # --- leverage and coverage -------------------------------------------
    out["ltv"] = _safe_divide(out["Loan_Amount"], out["Collateral_Value"])
    out["collateral_coverage"] = _safe_divide(out["Collateral_Value"], out["Loan_Amount"])
    out["under_collateralised"] = (out["ltv"] > 1).astype("int8")
    out["loan_to_income"] = _safe_divide(out["Loan_Amount"], out["Annual_Income"])
    out["existing_to_income"] = _safe_divide(out["Existing_Loan"], out["Annual_Income"])
    out["total_debt"] = out["Loan_Amount"] + out["Existing_Loan"]
    out["total_debt_to_income"] = _safe_divide(out["total_debt"], out["Annual_Income"])

    # --- affordability ----------------------------------------------------
    monthly_rate = out["Interest_Rate"] / 1200.0
    term = out["Loan_Term_Months"]
    factor = (1 + monthly_rate) ** term
    out["emi"] = out["Loan_Amount"] * monthly_rate * factor / (factor - 1).replace(0, np.nan)
    out["emi_to_monthly_income"] = _safe_divide(out["emi"] * 12, out["Annual_Income"])
    out["total_interest"] = out["emi"] * term - out["Loan_Amount"]
    out["interest_to_principal"] = _safe_divide(out["total_interest"], out["Loan_Amount"])

    # --- borrower quality -------------------------------------------------
    out["credit_score_norm"] = (out["Credit_Score"] - 300) / 550.0
    out["credit_band"] = pd.cut(
        out["Credit_Score"], bins=[0, 580, 670, 740, 800, 900],
        labels=["poor", "fair", "good", "very_good", "excellent"],
    ).astype(str)
    out["income_log"] = np.log1p(out["Annual_Income"])
    out["loan_log"] = np.log1p(out["Loan_Amount"])
    out["employment_stability"] = out["Employment_Length"] / 30.0
    out["dti_x_past_default"] = out["Debt_to_Income"] * out["Past_Default"]
    out["score_x_dti"] = out["credit_score_norm"] * (1 - out["Debt_to_Income"])

    # --- pricing signal (L-06) --------------------------------------------
    # NOTE: an earlier version computed `rate_spread = Interest_Rate - min(Interest_Rate)`.
    # The causality proof rejected it, and correctly: a minimum taken over the
    # WHOLE dataset is a fit-on-everything statistic, so test-set rows shape a
    # training feature. That is train/test leakage even though this file has no
    # time dimension at all - the leak is across the split, not across time.
    # Centring belongs in the preprocessor, which is fitted inside each fold
    # (see features/encoders.py). The raw rate is kept; trees are invariant to
    # a constant offset anyway, so nothing of value was lost.
    out["term_years"] = out["Loan_Term_Months"] / 12.0

    engineered = [
        "ltv", "collateral_coverage", "under_collateralised", "loan_to_income",
        "existing_to_income", "total_debt", "total_debt_to_income", "emi",
        "emi_to_monthly_income", "total_interest", "interest_to_principal",
        "credit_score_norm", "credit_band", "income_log", "loan_log",
        "employment_stability", "dti_x_past_default", "score_x_dti",
        "term_years",
    ]
    return FeatureResult(
        key="loans", frame=out, target="Default_Flag", engineered=engineered,
        notes=[
            "no origination date -> row-wise features only; out-of-time "
            "validation impossible (register N-05)",
            "every feature is computed from a SINGLE row - no dataset-wide "
            "statistic, so the split cannot leak",
        ],
    )


# =============================================================================
# M3 - Fraud
# =============================================================================
def build_fraud_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Behavioural features for ``nova_transactions.csv`` - the causal module.

    This is where causality is hardest and matters most. Every customer- and
    device-level aggregate is computed with ``shift(1)`` before the window, so
    transaction *t* sees that customer's history up to *t-1* and nothing later.

    ``amount_vs_history`` uses ``Historical_Avg_Transaction``, which is a
    same-row column supplied with the data and known at transaction time - it
    is not an aggregate we compute, so it carries no look-ahead. It was the
    strongest separator found in the Phase-0 audit.
    """
    cfg = cfg or load_config()
    out = frame.copy().sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    # --- same-row ratios (no history needed) ------------------------------
    out["amount_vs_history"] = _safe_divide(out["Amount"], out["Historical_Avg_Transaction"])
    out["amount_log"] = np.log1p(out["Amount"])
    out["amount_excess"] = out["Amount"] - out["Historical_Avg_Transaction"]
    out["account_age_years"] = out["Account_Age_Months"] / 12.0
    out["new_account"] = (out["Account_Age_Months"] < 12).astype("int8")

    # --- calendar --------------------------------------------------------
    timestamp = out["Timestamp"]
    out["hour"] = timestamp.dt.hour.astype("int16")
    out["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 22)).astype("int8")
    # Cyclical encoding: hour 23 and hour 0 are adjacent, which an integer
    # column cannot express and a tree can only approximate with deep splits.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    # --- causal customer history -----------------------------------------
    out["cust_prev_amount"] = causal_shift(out, "Amount", by="Customer_ID")
    out["cust_mean_amount_5"] = causal_rolling(out, "Amount", 5, "mean", by="Customer_ID")
    out["cust_std_amount_5"] = causal_rolling(out, "Amount", 5, "std", by="Customer_ID")
    out["cust_max_amount_10"] = causal_rolling(out, "Amount", 10, "max", by="Customer_ID")
    out["cust_mean_amount_all"] = causal_expanding(out, "Amount", "mean", by="Customer_ID")
    out["cust_txn_count"] = out.groupby("Customer_ID", observed=True).cumcount()

    out["amount_vs_cust_mean"] = _safe_divide(out["Amount"], out["cust_mean_amount_5"])
    out["amount_z_vs_cust"] = _safe_divide(
        out["Amount"] - out["cust_mean_amount_5"], out["cust_std_amount_5"]
    )
    out["amount_vs_cust_max"] = _safe_divide(out["Amount"], out["cust_max_amount_10"])

    # --- velocity: seconds since this customer's previous transaction -----
    previous_time = out.groupby("Customer_ID", observed=True)["Timestamp"].shift(1)
    out["seconds_since_prev"] = (timestamp - previous_time).dt.total_seconds()
    out["hours_since_prev"] = out["seconds_since_prev"] / 3600.0
    out["rapid_repeat"] = (out["seconds_since_prev"] < 3600).astype("float32")

    # --- causal device history -------------------------------------------
    # 1,500 devices shared by ~20 customers each (Phase-0). A device suddenly
    # serving a new customer is a classic account-takeover signal - but the
    # count must be a running count, never a total over the whole file.
    out["device_txn_count"] = out.groupby("Device_ID", observed=True).cumcount()
    out["device_mean_amount"] = causal_expanding(out, "Amount", "mean", by="Device_ID")
    out["amount_vs_device_mean"] = _safe_divide(out["Amount"], out["device_mean_amount"])

    customer_codes = out["Customer_ID"].astype("category").cat.codes
    out["_cust_code"] = customer_codes
    out["device_prev_customer"] = out.groupby("Device_ID", observed=True)["_cust_code"].shift(1)
    out["device_customer_changed"] = (
        (out["device_prev_customer"].notna())
        & (out["device_prev_customer"] != out["_cust_code"])
    ).astype("int8")
    out = out.drop(columns=["_cust_code", "device_prev_customer"])

    out["txn_frequency_norm"] = out["Transaction_Frequency"] / 27.0

    engineered = [
        "amount_vs_history", "amount_log", "amount_excess", "account_age_years",
        "new_account", "hour", "day_of_week", "is_weekend", "is_night",
        "hour_sin", "hour_cos", "cust_prev_amount", "cust_mean_amount_5",
        "cust_std_amount_5", "cust_max_amount_10", "cust_mean_amount_all",
        "cust_txn_count", "amount_vs_cust_mean", "amount_z_vs_cust",
        "amount_vs_cust_max", "seconds_since_prev", "hours_since_prev",
        "rapid_repeat", "device_txn_count", "device_mean_amount",
        "amount_vs_device_mean", "device_customer_changed", "txn_frequency_norm",
    ]
    return FeatureResult(
        key="transactions", frame=out, target="Fraud_Flag", engineered=engineered,
        notes=[
            "all history aggregates use shift(1) before rolling/expanding",
            "Device_ID and Customer_ID enter only as causal aggregates, never "
            "as raw keys (register L-09)",
        ],
    )


# =============================================================================
# M4 - Customer analytics
# =============================================================================
def build_customer_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Relationship features for ``nova_customers.csv``.

    Built in full knowledge that churn carries no learnable signal (N-01). That
    is precisely why they are built: a documented negative result is only
    credible if the features given to the model were competent. Reporting
    "no signal" after fitting on raw columns proves nothing; reporting it after
    a fair attempt is a finding.

    The same features are genuinely predictive of ``Estimated_CLV`` and drive
    the value-based answer to the 1,000-customer question.
    """
    cfg = cfg or load_config()
    out = frame.copy()

    out["total_balance"] = (
        out["Account_Balance"] + out["Investment_Balance"] + out["Loan_Balance"]
    )
    out["net_worth_proxy"] = (
        out["Account_Balance"] + out["Investment_Balance"] - out["Loan_Balance"]
    )
    out["balance_to_income"] = _safe_divide(out["Account_Balance"], out["Annual_Income"])
    out["investment_share"] = _safe_divide(out["Investment_Balance"], out["total_balance"])
    out["loan_to_balance"] = _safe_divide(out["Loan_Balance"], out["Account_Balance"])
    out["leverage"] = _safe_divide(out["Loan_Balance"], out["total_balance"])

    out["complaints_per_year"] = _safe_divide(
        out["Complaints"], out["Relationship_Length_Years"]
    )
    out["has_complaints"] = (out["Complaints"] > 0).astype("int8")
    out["products_per_year"] = _safe_divide(
        out["Products_Held"], out["Relationship_Length_Years"]
    )
    out["txn_per_product"] = _safe_divide(out["Transactions_Per_Month"], out["Products_Held"])
    out["digital_engaged"] = (out["Digital_Usage"] > 50).astype("int8")
    out["engagement_score"] = (
        out["Digital_Usage"] / 100 * 0.4
        + (out["Transactions_Per_Month"] / 35) * 0.3
        + (out["Products_Held"] / 6) * 0.3
    )
    out["tenure_ratio"] = _safe_divide(
        out["Relationship_Length_Years"], (out["Age"] - 18).clip(lower=1)
    )
    out["income_log"] = np.log1p(out["Annual_Income"])
    out["balance_log"] = np.log1p(out["Account_Balance"])
    out["age_band"] = pd.cut(
        out["Age"], bins=[0, 30, 45, 60, 100],
        labels=["young", "mid", "senior", "retired"],
    ).astype(str)

    # --- DECISION LAYER, NOT MODEL FEATURES -------------------------------
    # These three are population percentile ranks, used to answer the
    # "which 1,000 customers do we contact?" question over the whole book at
    # decision time. They are deliberately prefixed `decision_` and listed in
    # `datasets.customers.drop_always`, so `make_feature_frame` removes them
    # from X automatically.
    #
    # Why they must not be features: a percentile rank is computed across every
    # row, so a test-set customer's position depends on the training customers
    # and vice versa. The causality proof flags exactly this, and it is right -
    # the leak is across the train/test split rather than across time. They are
    # legitimate as a *business ranking*, illegitimate as a *model input*.
    out["decision_value_rank"] = out["Estimated_CLV"].rank(pct=True)
    out["decision_attrition_proxy"] = (
        out["complaints_per_year"].rank(pct=True) * 0.6
        + (1 - out["engagement_score"].rank(pct=True)) * 0.4
    )
    out["decision_priority_score"] = (
        out["decision_value_rank"] * out["decision_attrition_proxy"]
    )

    engineered = [
        "total_balance", "net_worth_proxy", "balance_to_income", "investment_share",
        "loan_to_balance", "leverage", "complaints_per_year", "has_complaints",
        "products_per_year", "txn_per_product", "digital_engaged",
        "engagement_score", "tenure_ratio", "income_log", "balance_log",
        "age_band", "decision_value_rank", "decision_attrition_proxy",
        "decision_priority_score",
    ]
    return FeatureResult(
        key="customers", frame=out, target="Churn_Flag", engineered=engineered,
        notes=[
            "built as a fair attempt so the N-01 null result is credible",
            "decision_* columns are population ranks: business ranking only, "
            "never model inputs (removed by drop_always)",
        ],
    )


#: Columns produced by the customer builder that answer the business question
#: but must never enter a feature matrix. Mirrored in configs/config.yaml.
CUSTOMER_DECISION_COLUMNS = [
    "decision_value_rank",
    "decision_attrition_proxy",
    "decision_priority_score",
]


# =============================================================================
# M5/M6 - Equity panel
# =============================================================================
def build_market_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Cross-sectional equity features and the FORWARD target (register L-01).

    The target is constructed, never read: ``fwd_return_1d`` is
    ``Return.shift(-1)`` within each ticker. Same-row OHLC and the raw
    ``Return`` are dropped at the feature boundary by ``make_feature_frame``.

    Cross-sectional ranks are legitimate here (see
    :func:`~novafin.features.causal.causal_rank`) because the strategy
    rebalances after the close using that day's published prices for all names.
    """
    cfg = cfg or load_config()
    horizon = int(cfg.fin("equity", "forecast_horizon_days", default=1))
    out = frame.copy().sort_values(["Ticker", "Date"], kind="mergesort").reset_index(drop=True)

    # --- TARGET (the only forward reference) ------------------------------
    out[f"fwd_return_{horizon}d"] = forward_target(
        out, "Return", by="Ticker", horizon=horizon
    )
    out["fwd_return_5d"] = forward_target(out, "Return", by="Ticker", horizon=5)

    # --- trailing price features (all strictly backward-looking) ----------
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_lag_{lag}"] = causal_shift(out, "Return", by="Ticker", periods=lag)

    out["ret_mean_5"] = causal_rolling(out, "Return", 5, "mean", by="Ticker")
    out["ret_mean_20"] = causal_rolling(out, "Return", 20, "mean", by="Ticker")
    out["ret_std_5"] = causal_rolling(out, "Return", 5, "std", by="Ticker")
    out["ret_std_20"] = causal_rolling(out, "Return", 20, "std", by="Ticker")
    out["ret_skew_20"] = causal_rolling(out, "Return", 20, "median", by="Ticker")

    # Trend and risk-adjusted trend. NOTE: no `reversal_1d = -ret_lag_1` and no
    # `momentum_20d = Momentum_20D` alias. Both were present in a first draft;
    # they are pure re-encodings (|r| = 1.000) that add nothing and force the
    # correlation filter to clean up after the builder. Short-horizon reversal
    # is expressed where it is actually used, in the cross-sectional rank below.
    out["momentum_ratio"] = _safe_divide(out["ret_mean_5"], out["ret_std_20"] + EPS)
    out["vol_ratio"] = _safe_divide(out["ret_std_5"], out["ret_std_20"] + EPS)
    out["risk_adj_momentum"] = _safe_divide(out["Momentum_20D"], out["Volatility_20D"] + EPS)

    # --- volume ------------------------------------------------------------
    out["volume_log"] = np.log1p(out["Volume"])
    out["volume_mean_20"] = causal_rolling(out, "volume_log", 20, "mean", by="Ticker")
    out["volume_surprise"] = out["volume_log"] - out["volume_mean_20"]

    # --- market context -----------------------------------------------------
    out["excess_return_lag1"] = out["ret_lag_1"] - causal_shift(
        out, "Market_Return", by="Ticker", periods=1
    )
    out["vix_lag1"] = causal_shift(out, "VIX", by="Ticker", periods=1)
    out["vix_change"] = out["vix_lag1"] - causal_rolling(
        out, "VIX", 20, "mean", by="Ticker"
    )
    out["high_vol_regime"] = (out["vix_lag1"] > 20).astype("float32")

    # --- cross-sectional ranks (computed within a date) ---------------------
    out["rank_momentum"] = causal_rank(out, "Momentum_20D", within="Date")
    out["rank_volatility"] = causal_rank(out, "Volatility_20D", within="Date")
    out["_neg_ret_lag_1"] = -out["ret_lag_1"]
    out["rank_reversal"] = causal_rank(out, "_neg_ret_lag_1", within="Date")
    out = out.drop(columns=["_neg_ret_lag_1"])
    out["rank_volume_surprise"] = causal_rank(out, "volume_surprise", within="Date")

    # Sector-relative momentum, using only trailing inputs.
    sector_mean = out.groupby(["Date", "Sector"], observed=True)["Momentum_20D"].transform("mean")
    out["momentum_vs_sector"] = out["Momentum_20D"] - sector_mean

    before = len(out)
    out = out.dropna(subset=[f"fwd_return_{horizon}d"]).reset_index(drop=True)
    dropped = before - len(out)

    engineered = [
        f"fwd_return_{horizon}d", "fwd_return_5d",
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_10",
        "ret_mean_5", "ret_mean_20", "ret_std_5", "ret_std_20", "ret_skew_20",
        "momentum_ratio", "vol_ratio", "risk_adj_momentum",
        "volume_log", "volume_mean_20",
        "volume_surprise", "excess_return_lag1", "vix_lag1", "vix_change",
        "high_vol_regime", "rank_momentum", "rank_volatility", "rank_reversal",
        "rank_volume_surprise", "momentum_vs_sector",
    ]
    return FeatureResult(
        key="market", frame=out, target=f"fwd_return_{horizon}d",
        engineered=engineered, dropped_rows=dropped,
        notes=[
            f"TARGET constructed as Return.shift(-{horizon}) within Ticker (L-01)",
            f"{dropped} rows dropped - the last observation of each ticker has no future",
            "cross-sectional ranks are legitimate: the strategy rebalances after the close",
        ],
    )


# =============================================================================
# M7 - Liquidity
# =============================================================================
def build_liquidity_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Lagged liquidity features and an h-day-ahead outflow target (L-03).

    ``Liquidity_Gap`` is an arithmetic identity, so it is neither a feature nor
    a target. The forecastable quantity is ``Expected_Outflows``; the gap is
    derived afterwards from the forecast.
    """
    cfg = cfg or load_config()
    horizon = int(cfg.fin("liquidity_risk", "forecast_horizon_days", default=5))
    out = frame.copy().sort_values("Date", kind="mergesort").reset_index(drop=True)

    out[f"fwd_outflows_{horizon}d"] = forward_target(out, "Expected_Outflows", horizon=horizon)
    out[f"fwd_inflows_{horizon}d"] = forward_target(out, "Expected_Inflows", horizon=horizon)

    base_columns = [
        "Deposits", "Loan_Disbursements", "Loan_Repayments", "Wholesale_Funding",
        "Expected_Inflows", "Expected_Outflows", "Market_Stress", "Interest_Rate",
    ]
    for column in base_columns:
        for lag in (1, 2, 5):
            out[f"{column.lower()}_lag_{lag}"] = causal_shift(out, column, periods=lag)
        out[f"{column.lower()}_ma_5"] = causal_rolling(out, column, 5, "mean")
        out[f"{column.lower()}_ma_20"] = causal_rolling(out, column, 20, "mean")
        out[f"{column.lower()}_std_20"] = causal_rolling(out, column, 20, "std")

    out["net_loan_flow_lag1"] = out["loan_disbursements_lag_1"] - out["loan_repayments_lag_1"]
    out["funding_ratio_lag1"] = _safe_divide(out["wholesale_funding_lag_1"], out["deposits_lag_1"])
    out["coverage_lag1"] = _safe_divide(out["expected_inflows_lag_1"], out["expected_outflows_lag_1"])
    out["liquid_assets_lag1"] = causal_shift(out, "Cash", periods=1) + causal_shift(
        out, "Securities_Holdings", periods=1
    )
    out["lcr_proxy_lag1"] = _safe_divide(out["liquid_assets_lag1"], out["expected_outflows_lag_1"])
    out["stress_flag_lag1"] = (out["market_stress_lag_1"] > 1.5).astype("float32")

    out["day_of_week"] = out["Date"].dt.dayofweek.astype("int8")
    out["month"] = out["Date"].dt.month.astype("int8")
    out["is_month_end"] = out["Date"].dt.is_month_end.astype("int8")
    out["is_quarter_end"] = out["Date"].dt.is_quarter_end.astype("int8")

    before = len(out)
    out = out.dropna(subset=[f"fwd_outflows_{horizon}d"]).reset_index(drop=True)
    dropped = before - len(out)

    engineered = [c for c in out.columns if c not in frame.columns]
    return FeatureResult(
        key="liquidity", frame=out, target=f"fwd_outflows_{horizon}d",
        engineered=engineered, dropped_rows=dropped,
        notes=[
            f"target = Expected_Outflows.shift(-{horizon}); the gap is DERIVED "
            "from the forecast, never modelled directly (L-03)",
            "Cash and Securities_Holdings are near-constant, so only their "
            "lagged sum enters, as an LCR proxy",
        ],
    )


# =============================================================================
# M10 - Derivatives
# =============================================================================
def black_scholes_price(
    spot: np.ndarray | pd.Series,
    strike: np.ndarray | pd.Series,
    time_to_maturity: np.ndarray | pd.Series,
    volatility: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    is_call: np.ndarray | pd.Series,
) -> np.ndarray:
    """European option price under Black-Scholes-Merton, no dividends.

    .. math::
        C = S\\,N(d_1) - K e^{-rT} N(d_2), \\quad
        P = K e^{-rT} N(-d_2) - S\\,N(-d_1)

    with :math:`d_1 = \\frac{\\ln(S/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}`
    and :math:`d_2 = d_1 - \\sigma\\sqrt{T}`.

    Implemented with :func:`math.erf` via ``scipy``-free normal CDF so the
    module has no extra dependency. Used both to reproduce the supplied
    ``Black_Scholes_Price`` (a correctness check on our own maths) and to
    generate the Greeks the dataset does not ship.

    Reference: Black F, Scholes M (1973), *Journal of Political Economy* 81(3).
    https://doi.org/10.1086/260062
    """
    from math import erf, sqrt as _sqrt

    def norm_cdf(x: np.ndarray) -> np.ndarray:
        vectorised = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / _sqrt(2.0))))
        return vectorised(x)

    spot = np.asarray(spot, dtype="float64")
    strike = np.asarray(strike, dtype="float64")
    time_to_maturity = np.maximum(np.asarray(time_to_maturity, dtype="float64"), 1e-8)
    volatility = np.maximum(np.asarray(volatility, dtype="float64"), 1e-8)
    rate = np.asarray(rate, dtype="float64")
    is_call = np.asarray(is_call).astype(bool)

    sqrt_t = np.sqrt(time_to_maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    discounted_strike = strike * np.exp(-rate * time_to_maturity)

    call = spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    put = discounted_strike * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return np.where(is_call, call, put)


def build_options_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Pricing features and the mispricing target for ``nova_options.csv``.

    Two experiments are set up here, per leakage-register L-04:

    * ``PRIMITIVE_FEATURES`` - the fair fight. ML sees only S, K, T, sigma, r
      and the option type, exactly what Black-Scholes sees. Can it rediscover
      the formula?
    * ``mispricing`` - the residual target, ``Market_Price - Black_Scholes_Price``.
      This is what a trading desk actually models.
    """
    cfg = cfg or load_config()
    out = frame.copy()

    is_call = (out["Option_Type"] == "Call")
    out["is_call"] = is_call.astype("int8")

    # --- moneyness --------------------------------------------------------
    out["moneyness"] = _safe_divide(out["Spot"], out["Strike"])
    out["log_moneyness"] = np.log(out["moneyness"].replace(0, np.nan))
    out["in_the_money"] = np.where(
        is_call, (out["Spot"] > out["Strike"]), (out["Spot"] < out["Strike"])
    ).astype("int8")
    out["intrinsic_value"] = np.where(
        is_call,
        np.maximum(out["Spot"] - out["Strike"], 0.0),
        np.maximum(out["Strike"] - out["Spot"], 0.0),
    )

    # --- time and volatility ----------------------------------------------
    out["sqrt_time"] = np.sqrt(out["Time_to_Maturity"])
    out["total_variance"] = out["Volatility"] ** 2 * out["Time_to_Maturity"]
    out["vol_sqrt_t"] = out["Volatility"] * out["sqrt_time"]
    out["standardised_moneyness"] = _safe_divide(out["log_moneyness"], out["vol_sqrt_t"] + EPS)
    out["discount_factor"] = np.exp(-out["Interest_Rate"] * out["Time_to_Maturity"])
    out["carry"] = out["Interest_Rate"] * out["Time_to_Maturity"]

    # --- d1 / d2 and the Greeks the dataset does not ship -----------------
    d1 = _safe_divide(
        out["log_moneyness"] + (out["Interest_Rate"] + 0.5 * out["Volatility"] ** 2)
        * out["Time_to_Maturity"],
        out["vol_sqrt_t"] + EPS,
    )
    out["d1"] = d1
    out["d2"] = d1 - out["vol_sqrt_t"]
    normal_pdf = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    out["vega"] = out["Spot"] * normal_pdf * out["sqrt_time"]
    out["gamma"] = _safe_divide(normal_pdf, out["Spot"] * out["vol_sqrt_t"] + EPS)

    # --- our own BS, as a check on the supplied column ---------------------
    out["bs_recomputed"] = black_scholes_price(
        out["Spot"], out["Strike"], out["Time_to_Maturity"],
        out["Volatility"], out["Interest_Rate"], is_call,
    )
    out["bs_check_error"] = (out["bs_recomputed"] - out["Black_Scholes_Price"]).abs()
    # Agreement must be judged in RELATIVE terms. The CSV stores Volatility,
    # Interest_Rate and Time_to_Maturity rounded to 4 decimal places, and a
    # long-dated option has vega and rho in the thousands - so a 5e-5 rounding
    # of an input moves the price by ~0.2 in absolute terms while remaining
    # correct to ~0.01%. Judging this check on absolute error would report a
    # correct implementation as broken.
    out["bs_check_error_pct"] = _safe_divide(
        out["bs_check_error"], out["Black_Scholes_Price"]
    )

    # --- the residual target ------------------------------------------------
    out["mispricing"] = out["Market_Price"] - out["Black_Scholes_Price"]
    out["mispricing_pct"] = _safe_divide(out["mispricing"], out["Black_Scholes_Price"])
    out["time_value"] = out["Market_Price"] - out["intrinsic_value"]

    out["maturity_bucket"] = pd.cut(
        out["Time_to_Maturity"], bins=[0, 0.25, 0.5, 1.0, 2.01],
        labels=["front", "near", "mid", "long"],
    ).astype(str)
    out["expiry_bucket"] = out["Expiry"].dt.to_period("Q").astype(str)

    engineered = [
        "is_call", "moneyness", "log_moneyness", "in_the_money", "intrinsic_value",
        "sqrt_time", "total_variance", "vol_sqrt_t", "standardised_moneyness",
        "discount_factor", "carry", "d1", "d2", "vega", "gamma",
        "bs_recomputed", "bs_check_error", "bs_check_error_pct", "mispricing", "mispricing_pct",
        "time_value", "maturity_bucket", "expiry_bucket",
    ]
    return FeatureResult(
        key="options", frame=out, target="Market_Price", engineered=engineered,
        notes=[
            "mispricing = Market_Price - Black_Scholes_Price is the residual target (L-04)",
            "bs_check_error validates our own implementation against the supplied column",
            "expiry_bucket is the grouping key for the leak-safe split",
        ],
    )


#: The fair-fight feature set: exactly the inputs Black-Scholes itself uses.
PRIMITIVE_FEATURES = [
    "Spot", "Strike", "Time_to_Maturity", "Volatility", "Interest_Rate", "is_call",
]


# =============================================================================
# M9 - High-frequency order book
# =============================================================================
def build_hft_features(frame: pd.DataFrame, cfg: Config | None = None) -> FeatureResult:
    """Microstructure features for ``nova_hft_orderbook.csv``.

    Aggregates are computed within ``(Stock, Trading_Day)`` and shifted, so a
    feature never sees a later tick - and never crosses a day boundary, which
    would be meaningless across an overnight gap.

    No cross-sectional ranking is used here: unlike the daily equity panel,
    another stock's later tick is genuinely not observable at decision time.
    """
    cfg = cfg or load_config()
    out = frame.copy()
    group = ["Stock", "Trading_Day"]

    # --- static book shape (same-tick, therefore observable) ---------------
    out["spread_bps"] = _safe_divide(out["Spread"], out["Mid_Price"]) * 10_000
    out["depth_imbalance_1"] = _safe_divide(
        out["Bid_Size_1"] - out["Ask_Size_1"], out["Bid_Size_1"] + out["Ask_Size_1"]
    )
    out["depth_imbalance_all"] = _safe_divide(
        (out["Bid_Size_1"] + out["Bid_Size_2"] + out["Bid_Size_3"])
        - (out["Ask_Size_1"] + out["Ask_Size_2"] + out["Ask_Size_3"]),
        out["Total_Depth"],
    )
    out["book_slope_bid"] = _safe_divide(
        out["Bid_Price_1"] - out["Bid_Price_3"], out["Bid_Size_1"] + out["Bid_Size_3"]
    )
    out["book_slope_ask"] = _safe_divide(
        out["Ask_Price_3"] - out["Ask_Price_1"], out["Ask_Size_1"] + out["Ask_Size_3"]
    )
    out["microprice_dev_bps"] = _safe_divide(
        out["Microprice_Minus_Mid"], out["Mid_Price"]
    ) * 10_000
    out["flow_pressure"] = _safe_divide(
        out["Order_Arrival_Rate"], out["Cancellation_Rate"] + EPS
    )
    out["log_depth"] = np.log1p(out["Total_Depth"])
    out["trade_volume_log"] = np.log1p(out["Trade_Volume"])
    out["is_buy"] = (out["Trade_Side"] == "BUY").astype("int8")
    out["is_market_order"] = (out["Event_Type"] == "MARKET_ORDER").astype("int8")
    out["is_cancel"] = (out["Event_Type"] == "CANCEL").astype("int8")

    # --- causal intraday history -------------------------------------------
    out["obi_lag_1"] = causal_shift(out, "OBI_Level1", by=group, periods=1)
    out["obi_mean_10"] = causal_rolling(out, "OBI_Level1", 10, "mean", by=group)
    out["obi_mean_50"] = causal_rolling(out, "OBI_3Level", 50, "mean", by=group)
    out["obi_change"] = out["OBI_Level1"] - out["obi_lag_1"]
    out["obi_vs_mean"] = out["OBI_Level1"] - out["obi_mean_10"]

    out["mid_lag_1"] = causal_shift(out, "Mid_Price", by=group, periods=1)
    out["mid_return_bps"] = _safe_divide(
        out["Mid_Price"] - out["mid_lag_1"], out["mid_lag_1"]
    ) * 10_000
    out["mid_mean_20"] = causal_rolling(out, "Mid_Price", 20, "mean", by=group)
    out["mid_vs_mean_bps"] = _safe_divide(
        out["Mid_Price"] - out["mid_mean_20"], out["mid_mean_20"]
    ) * 10_000

    out["spread_mean_20"] = causal_rolling(out, "Relative_Spread", 20, "mean", by=group)
    out["spread_vs_mean"] = out["Relative_Spread"] - out["spread_mean_20"]
    out["volume_mean_20"] = causal_rolling(out, "trade_volume_log", 20, "mean", by=group)
    out["volume_surprise"] = out["trade_volume_log"] - out["volume_mean_20"]
    out["buy_pressure_20"] = causal_rolling(out, "is_buy", 20, "mean", by=group)
    out["cancel_rate_20"] = causal_rolling(out, "is_cancel", 20, "mean", by=group)
    out["tick_index"] = out.groupby(group, observed=True).cumcount()

    engineered = [
        "spread_bps", "depth_imbalance_1", "depth_imbalance_all", "book_slope_bid",
        "book_slope_ask", "microprice_dev_bps", "flow_pressure", "log_depth",
        "trade_volume_log", "is_buy", "is_market_order", "is_cancel",
        "obi_lag_1", "obi_mean_10", "obi_mean_50", "obi_change", "obi_vs_mean",
        "mid_lag_1", "mid_return_bps", "mid_mean_20", "mid_vs_mean_bps",
        "spread_mean_20", "spread_vs_mean", "volume_mean_20", "volume_surprise",
        "buy_pressure_20", "cancel_rate_20", "tick_index",
    ]
    return FeatureResult(
        key="hft", frame=out, target="Price_Move_Class", engineered=engineered,
        notes=[
            "history computed within (Stock, Trading_Day) - never across the overnight gap",
            "no cross-sectional ranking: other stocks' later ticks are not observable",
        ],
    )


# =============================================================================
# Registry
# =============================================================================
FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame, Config | None], FeatureResult]] = {
    "initiatives": build_initiative_features,
    "loans": build_credit_features,
    "transactions": build_fraud_features,
    "customers": build_customer_features,
    "market": build_market_features,
    "liquidity": build_liquidity_features,
    "options": build_options_features,
    "hft": build_hft_features,
}


def build_features(
    key: str, frame: pd.DataFrame, cfg: Config | None = None
) -> FeatureResult:
    """Dispatch to the registered builder for ``key``.

    Raises:
        KeyError: If no builder is registered, listing the ones that are.
    """
    if key not in FEATURE_BUILDERS:
        raise KeyError(
            f"No feature builder for {key!r}. Registered: {sorted(FEATURE_BUILDERS)}"
        )
    result = FEATURE_BUILDERS[key](frame, cfg)
    LOGGER.info("Built features for '%s': %s", key, result.summary())
    return result
