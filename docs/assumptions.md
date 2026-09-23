# Assumption Register

Every number an examiner could challenge lives in `configs/config.yaml`, not in
code. This page explains *why* each default was chosen. Changing a value here
means editing YAML and re-running — which is the point.

| # | Assumption | Default | Config key | Justification |
|---|---|---|---|---|
| A-01 | Master seed | 42 | `reproducibility.seed` | Matches the dataset generator's seed, stated in the course `README.txt` |
| A-02 | Reporting currency | ₹ crore (data in lakh) | `finance.report_units` | Board question is framed in crore; data README states lakh |
| A-03 | Cost of capital | 12% | `finance.npv.discount_rate` | Plausible INR corporate hurdle rate; NPV is reported as a sensitivity band, not a point |
| A-04 | NPV horizon | 3 years | `finance.npv.horizon_years` | The dataset supplies exactly `Revenue_Year1..3` |
| A-05 | Loss given default | 40% | `finance.credit.lgd` | The brief's own worked example |
| A-06 | Collateral-aware LGD | enabled as sensitivity | `finance.credit.lgd_collateral_aware` | Median LTV 0.80 but **max 2.00**, so a flat 40% understates loss on the tail |
| A-07 | Exposure at default | `Loan_Amount` | `finance.credit.ead_column` | The brief's worked example |
| A-08 | PD bands | <5 / 5–15 / 15–30 / >30% | `finance.credit.pd_bands` | The brief calls these illustrative and says they should be justified — we report them **and** bands re-derived from the observed ECL distribution |
| A-09 | Missed-fraud cost | ₹10,000 | `finance.fraud.cost_missed_fraud_inr` | Given in the brief |
| A-10 | False-positive cost | ₹500 | `finance.fraud.cost_false_positive_inr` | Given in the brief |
| A-11 | Retention contact budget | 1,000 customers | `finance.churn.contact_budget` | The brief's executive question |
| A-12 | Equity forecast horizon | t+1 (t+5 robustness) | `finance.equity.forecast_horizon_days` | Shortest horizon the daily panel supports without overlapping-return autocorrelation |
| A-13 | Rebalance frequency | monthly | `finance.equity.rebalance_frequency` | Keeps turnover and transaction costs realistic on a 1,457-day panel |
| A-14 | Transaction cost | 10 bps per side | `finance.equity.transaction_cost_bps` | Conservative Indian large-cap round-trip estimate; swept in sensitivity |
| A-15 | Risk-free rate | 6% annual | `finance.equity.risk_free_rate_annual` | Used only for Sharpe; the dataset's own `Interest_Rate` averages ~6.8% |
| A-16 | VaR confidence | 95% and 99% | `finance.market_risk.var_confidence` | Regulatory convention; both reported with Kupiec/Christoffersen backtests |
| A-17 | Liquidity horizon | 5 business days | `finance.liquidity_risk.forecast_horizon_days` | Short enough to be forecastable, long enough to be operationally useful |
| A-18 | Stress scenarios | mild / severe / extreme | `finance.liquidity_risk.stress_scenarios` | Three-point ladder so the stress result is a curve, not a single claim |
| A-19 | HFT cost | 0.5 bps per trade | `finance.hft.cost_per_trade_bps` | Below the observed median relative spread (1.34 bps), so the strategy is tested against a realistic but not punitive cost |
| A-20 | Loan key | `Customer_ID` | `datasets.loans.id_columns` | The brief asks for `Loan_ID`, which does not exist in the file; `Customer_ID` is unique per row |
| A-21 | M11 integration level | portfolio / segment | — | Forced by the zero-overlap customer ID ranges (see `docs/DATA.md`) |
| A-22 | GPU usage | `auto` | `compute.use_gpu` | Falls back to a LightGBM-only path on a CPU-only Colab session with no code change |
