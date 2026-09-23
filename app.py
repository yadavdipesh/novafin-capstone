"""
novafin-capstone/app.py

Gradio demo - the interactive front end for the NovaFin decision engine.

Run locally:      python app.py
Run in Colab:     !python app.py     (a public share link is printed)

Why Gradio rather than Streamlit
---------------------------------
``share=True`` produces a public URL with **no account and no API key**, which
is what the capstone's "no paid APIs, no keys" constraint requires. Streamlit's
equivalent needs a Community Cloud account. Gradio is Apache-2.0 and the share
tunnel is free.

What the demo shows
-------------------
Four tabs, one per decision the Board actually asked about:

1. **Credit** — a loan application scored to PD, ECL and Approve/Review/Reject.
2. **Fraud** — a transaction scored at the COST-OPTIMAL threshold, with the
   expected-cost arithmetic shown rather than hidden.
3. **Capital allocation** — the Module 11 ₹1,000 crore split, re-optimised live
   as the risk-aversion slider moves.
4. **Model governance** — the leakage register and model cards, because
   "explainability is important" is in the brief and a demo that hides the
   caveats is the wrong demo.

Graceful degradation is deliberate: if no trained bundle exists yet, the app
still runs and says so. A demo that crashes because the examiner has not run
notebook 03 is worse than one that explains what to run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from novafin.config import load_config
from novafin.evaluate import expected_credit_loss
from novafin.finance.allocate import (
    allocate_capital,
    allocation_narrative,
    build_bucket_inputs,
)
from novafin.finance.metrics import credit_decision, risk_band
from novafin.utils.logging_utils import setup_logging
from novafin.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)

cfg = load_config()
setup_logging("INFO")
seed_everything(cfg.reproducibility.seed)

THEME_TEAL = "#00B2A9"
THEME_NAVY = "#002B49"


def _load_bundle(name: str) -> Any | None:
    """Load a bundle if it exists; return None rather than raising."""
    try:
        from novafin.predict import load_bundle

        path = cfg.paths.artifacts / name
        return load_bundle(path) if path.exists() else None
    except Exception as exc:
        LOGGER.warning("Could not load %s: %s", name, exc)
        return None


CREDIT_BUNDLE = _load_bundle("loans_baseline.pkl")
FRAUD_BUNDLE = _load_bundle("transactions_baseline.pkl")


# =============================================================================
# Tab 1 - credit
# =============================================================================
def score_loan(
    annual_income: float, loan_amount: float, collateral_value: float,
    credit_score: int, debt_to_income: float, interest_rate: float,
    employment_length: int, loan_term: int, past_default: bool, customer_type: str,
) -> tuple[str, str]:
    """Score one application and explain the recommendation."""
    lgd = float(cfg.fin("credit", "lgd", default=0.40))
    ltv = loan_amount / max(collateral_value, 1.0)

    if CREDIT_BUNDLE is not None:
        row = pd.DataFrame([{
            "Annual_Income": annual_income, "Loan_Amount": loan_amount,
            "Collateral_Value": collateral_value, "Credit_Score": credit_score,
            "Debt_to_Income": debt_to_income, "Interest_Rate": interest_rate,
            "Employment_Length": employment_length, "Loan_Term_Months": loan_term,
            "Past_Default": int(past_default), "Customer_Type": customer_type,
            "Existing_Loan": annual_income * 0.5, "Sector_Risk": 50.0,
        }])
        try:
            from novafin.predict import predict

            for feature in CREDIT_BUNDLE.feature_names:
                if feature not in row.columns:
                    row[feature] = 0.0
            probability = float(predict(CREDIT_BUNDLE, row)[0])
            basis = "trained model (`loans_baseline.pkl`)"
        except Exception as exc:
            probability, basis = _heuristic_pd(credit_score, debt_to_income, ltv, past_default), \
                f"heuristic fallback — the model could not score this row ({exc})"
    else:
        probability = _heuristic_pd(credit_score, debt_to_income, ltv, past_default)
        basis = ("**heuristic fallback** — no trained bundle found. "
                 "Run `notebooks/03_baseline_models.ipynb` for real model scores.")

    ecl_flat = float(expected_credit_loss([probability], [loan_amount], lgd=lgd).iloc[0])
    ecl_secured = float(
        expected_credit_loss([probability], [loan_amount], collateral=[collateral_value]).iloc[0]
    )
    decision = credit_decision(
        [probability], [ecl_flat], [loan_amount],
        collateral_ratio=[collateral_value / max(loan_amount, 1.0)],
    ).iloc[0]

    colour = {"Approve": THEME_TEAL, "Review": "#FFC72C", "Reject": "#B88F20"}[decision["decision"]]
    verdict = f"""
<div style="border-left:6px solid {colour};padding:14px 18px;background:#F4F5F6;
            font-family:Tajawal,Segoe UI,sans-serif">
  <div style="font-size:26px;font-weight:700;color:{THEME_NAVY}">{decision['decision']}</div>
  <div style="font-size:15px;color:#333F48;margin-top:6px">
    Probability of default <b>{probability:.2%}</b> &nbsp;·&nbsp;
    risk band <b>{decision['risk_band'].replace('_',' ')}</b>
  </div>
</div>"""

    detail = f"""
### The arithmetic

| Quantity | Value |
|---|---:|
| Probability of default (PD) | **{probability:.2%}** |
| Exposure at default (EAD) | ₹{loan_amount:,.0f} |
| Loss given default (LGD), flat | {lgd:.0%} |
| **ECL = PD × LGD × EAD** | **₹{ecl_flat:,.0f}** |
| ECL, collateral-aware | ₹{ecl_secured:,.0f} |
| ECL as a share of exposure | {ecl_flat / max(loan_amount, 1):.2%} |
| Loan-to-value | {ltv:.2f} |

**Why this decision.** The rule uses three signals, not PD alone: the
probability itself, the ECL-to-exposure ratio (which catches a low-PD borrower
with a very large loan), and collateral coverage (a well-secured loan can be
approved at a PD that would otherwise fail).

**Score basis:** {basis}

> ⚠️ 33.3% of loans in this book have LTV > 1, so the flat 40% LGD understates
> loss on the tail — which is why the collateral-aware figure is shown beside it.
> This is decision **support**: a human underwriter decides.
"""
    return verdict, detail


def _heuristic_pd(credit_score: int, dti: float, ltv: float, past_default: bool) -> float:
    """A transparent stand-in when no bundle exists.

    Clearly labelled everywhere it is used. It exists so the demo runs before
    notebook 03 has been executed - never to be mistaken for a model.
    """
    logit = (
        -1.4
        - 3.0 * ((credit_score - 300) / 550 - 0.5)
        + 2.0 * (dti - 0.35)
        + 0.8 * (ltv - 0.8)
        + 1.2 * float(past_default)
    )
    return float(1 / (1 + np.exp(-logit)))


# =============================================================================
# Tab 2 - fraud
# =============================================================================
def score_transaction(
    amount: float, historical_average: float, transaction_type: str,
    merchant_category: str, account_age_months: int, hour: int, threshold: float,
) -> tuple[str, str]:
    """Score a transaction and show the cost arithmetic behind the decision."""
    cost_fn = float(cfg.fin("fraud", "cost_missed_fraud_inr", default=10000))
    cost_fp = float(cfg.fin("fraud", "cost_false_positive_inr", default=500))
    ratio = amount / max(historical_average, 1.0)

    if FRAUD_BUNDLE is not None:
        try:
            from novafin.predict import predict

            row = pd.DataFrame([{f: 0.0 for f in FRAUD_BUNDLE.feature_names}])
            for name, value in {
                "Amount": amount, "amount_vs_history": ratio,
                "Historical_Avg_Transaction": historical_average,
                "Account_Age_Months": account_age_months, "hour": hour,
                "amount_log": np.log1p(amount),
            }.items():
                if name in row.columns:
                    row[name] = value
            probability = float(predict(FRAUD_BUNDLE, row)[0])
            basis = "trained model (`transactions_baseline.pkl`)"
        except Exception as exc:
            probability = _heuristic_fraud(ratio, amount, account_age_months, hour)
            basis = f"heuristic fallback ({exc})"
    else:
        probability = _heuristic_fraud(ratio, amount, account_age_months, hour)
        basis = ("**heuristic fallback** — no trained bundle found. "
                 "Run `notebooks/03_baseline_models.ipynb` for real model scores.")

    investigate = probability >= threshold
    expected_loss = probability * cost_fn
    net_benefit = expected_loss - cost_fp

    colour = "#B88F20" if investigate else THEME_TEAL
    verdict = f"""
<div style="border-left:6px solid {colour};padding:14px 18px;background:#F4F5F6;
            font-family:Tajawal,Segoe UI,sans-serif">
  <div style="font-size:26px;font-weight:700;color:{THEME_NAVY}">
    {'INVESTIGATE' if investigate else 'ALLOW'}</div>
  <div style="font-size:15px;color:#333F48;margin-top:6px">
    Fraud probability <b>{probability:.2%}</b> &nbsp;·&nbsp;
    threshold <b>{threshold:.3f}</b>
  </div>
</div>"""

    detail = f"""
### The cost arithmetic

| Quantity | Value |
|---|---:|
| Fraud probability | **{probability:.2%}** |
| Amount ÷ customer's historical average | **{ratio:.2f}×** |
| Cost if this is fraud and we ignore it | ₹{expected_loss:,.0f} |
| Cost of investigating | ₹{cost_fp:,.0f} |
| **Net benefit of investigating** | **₹{net_benefit:,.0f}** |

**The decision variable is the threshold, not the model.** At a 2.28% base rate
the cost-minimising operating point is nowhere near 0.5 — it is wherever
expected cost is lowest given the Board's ₹{cost_fn:,.0f} / ₹{cost_fp:,.0f}
economics. Move the slider to see the trade-off.

**Score basis:** {basis}

> Flagged transactions are reviewed by an analyst, never auto-blocked.
"""
    return verdict, detail


def _heuristic_fraud(ratio: float, amount: float, account_age: int, hour: int) -> float:
    """Transparent stand-in, shaped by the Phase-0 decile-lift findings."""
    logit = (
        -4.2
        + 0.55 * np.log1p(max(ratio, 0.01))
        + 0.30 * (np.log1p(amount) - 9.0)
        - 0.010 * account_age
        + 0.45 * float(hour < 6 or hour >= 22)
    )
    return float(1 / (1 + np.exp(-logit)))


# =============================================================================
# Tab 3 - capital allocation
# =============================================================================
def run_allocation(total_capital: float, risk_aversion: float,
                   max_volatility: float) -> tuple[pd.DataFrame, str]:
    """Re-run the Module 11 allocation live."""
    buckets = build_bucket_inputs(cfg)
    result = allocate_capital(
        buckets, cfg=cfg, total_capital_crore=total_capital,
        risk_aversion=risk_aversion, max_portfolio_volatility=max_volatility,
    )
    table = result.allocations[
        ["bucket", "allocation_crore", "allocation_pct", "net_return",
         "volatility", "risk_adjusted_return", "expected_profit_crore"]
    ].copy()
    table.columns = ["Bucket", "₹ crore", "Share", "Net return",
                     "Volatility", "Risk-adjusted", "Expected profit (₹ cr)"]
    table["Share"] = (table["Share"] * 100).round(1).astype(str) + "%"
    for column in ("Net return", "Volatility"):
        table[column] = (table[column] * 100).round(2).astype(str) + "%"
    table = table.round(2)
    return table, allocation_narrative(result)


# =============================================================================
# Build the interface
# =============================================================================
def build_interface() -> Any:
    """Assemble the Gradio app."""
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.blue,
        font=[gr.themes.GoogleFont("Tajawal"), "Segoe UI", "sans-serif"],
    )

    with gr.Blocks(theme=theme, title="NovaFin — ML Decision Engine") as demo:
        gr.Markdown(
            f"""
# NovaFin Group — ML Decision Engine
### ML-Driven Enterprise Financial Strategy, Risk & Decision Support

Capstone demo · ePGD in AI & Data Science, IIIT Bombay · Group 2 ·
Dipesh Kumar Yadav · config `{cfg.fingerprint()}`

> Every number below comes from the models and the finance layer in this
> repository. Where a trained bundle is absent, the app says so rather than
> pretending.
"""
        )

        with gr.Tab("Credit risk"):
            gr.Markdown("### Score a loan application → PD → ECL → Approve / Review / Reject")
            with gr.Row():
                with gr.Column(scale=1):
                    income = gr.Number(value=1_000_000, label="Annual income (₹)")
                    amount = gr.Number(value=1_200_000, label="Loan amount (₹)")
                    collateral = gr.Number(value=1_500_000, label="Collateral value (₹)")
                    score = gr.Slider(400, 850, 690, step=1, label="Credit score")
                    dti = gr.Slider(0.03, 0.85, 0.38, step=0.01, label="Debt-to-income")
                    rate = gr.Slider(6.0, 19.4, 9.5, step=0.1, label="Interest rate (%)")
                    employment = gr.Slider(1, 30, 8, step=1, label="Employment length (years)")
                    term = gr.Dropdown([12, 24, 36, 48, 60, 72, 84], value=48, label="Term (months)")
                    default_flag = gr.Checkbox(label="Past default")
                    customer_type = gr.Radio(["Retail", "Corporate"], value="Retail", label="Segment")
                    credit_button = gr.Button("Score application", variant="primary")
                with gr.Column(scale=1):
                    credit_verdict = gr.HTML()
                    credit_detail = gr.Markdown()

            credit_button.click(
                score_loan,
                [income, amount, collateral, score, dti, rate, employment, term,
                 default_flag, customer_type],
                [credit_verdict, credit_detail],
            )

        with gr.Tab("Fraud detection"):
            gr.Markdown("### Score a transaction at the cost-optimal threshold")
            with gr.Row():
                with gr.Column(scale=1):
                    txn_amount = gr.Number(value=45_000, label="Transaction amount (₹)")
                    historical = gr.Number(value=7_000, label="Customer's historical average (₹)")
                    txn_type = gr.Radio(["Payment", "Transfer", "Withdrawal"],
                                        value="Transfer", label="Type")
                    merchant = gr.Dropdown(
                        ["Online", "Retail", "Travel", "Healthcare", "Utilities",
                         "Investment", "Other"], value="Online", label="Merchant category")
                    age = gr.Slider(3, 179, 24, step=1, label="Account age (months)")
                    hour = gr.Slider(0, 23, 2, step=1, label="Hour of day")
                    threshold = gr.Slider(0.01, 0.99, 0.15, step=0.01,
                                          label="Decision threshold (set from the cost curve)")
                    fraud_button = gr.Button("Score transaction", variant="primary")
                with gr.Column(scale=1):
                    fraud_verdict = gr.HTML()
                    fraud_detail = gr.Markdown()

            fraud_button.click(
                score_transaction,
                [txn_amount, historical, txn_type, merchant, age, hour, threshold],
                [fraud_verdict, fraud_detail],
            )

        with gr.Tab("Capital allocation"):
            gr.Markdown(
                "### Module 11 — deploy ₹1,000 crore across NovaFin's businesses\n"
                "Move the sliders to see how the Board's risk appetite changes "
                "the recommendation."
            )
            with gr.Row():
                capital = gr.Slider(100, 5000, 1000, step=50, label="Capital (₹ crore)")
                aversion = gr.Slider(0.5, 10.0, 2.0, step=0.5, label="Risk aversion (λ)")
                volatility = gr.Slider(0.02, 0.20, 0.08, step=0.01,
                                       label="Portfolio volatility ceiling")
            allocate_button = gr.Button("Recommend allocation", variant="primary")
            allocation_table = gr.Dataframe(label="Recommended deployment")
            allocation_text = gr.Markdown()

            allocate_button.click(
                run_allocation, [capital, aversion, volatility],
                [allocation_table, allocation_text],
            )

        with gr.Tab("Model governance"):
            register = Path("docs/LEAKAGE_REGISTER.md")
            gr.Markdown(
                "### Leakage register and validation controls\n\n"
                "Every model in this project is accompanied by the controls that "
                "make its numbers trustworthy. This tab is part of the demo on "
                "purpose: a decision engine that hides its caveats is the wrong "
                "engine.\n\n"
                + (register.read_text(encoding="utf-8")[:6000] + "\n\n*(truncated — "
                   "see `docs/LEAKAGE_REGISTER.md` for the full register)*"
                   if register.exists() else "_Register not found._")
            )

        gr.Markdown(
            "---\n"
            "*Synthetic course data, educational use only. Decision support — "
            "a human makes every final decision.*"
        )

    return demo


def main() -> None:
    """Launch the demo. ``share=True`` needs no account or key."""
    demo = build_interface()
    demo.launch(share=True, show_error=True)


if __name__ == "__main__":  # pragma: no cover
    main()
