# Leakage Register — NovaFin Capstone

> **Status:** living document. Every entry is (a) reproduced verbatim in the D7 Project Guide §Limitations & §Validation, (b) encoded as a machine-readable `drop_always` rule in `configs/config.yaml`, (c) enforced by a failing test in `tests/test_leakage.py` from Phase 2, and (d) turned into a viva answer in D10.
>
> **Why this file exists.** Every leak below produces a *spectacular* metric. A model that reports 99.8% R² because it was handed the answer is worse than a model that reports 0.58 AUC honestly, because the first one cannot be defended. Discovering these before writing a single model — and being able to show the leaked-vs-clean comparison — is the strongest evidence of the "rigorous validation" criterion.
>
> All evidence numbers below were **computed on the actual supplied CSVs** during Phase 0, not asserted.

---

## L-01 · `nova_market_data.csv` — `Return` is contemporaneous ⚠ CRITICAL

| | |
|---|---|
| **Dataset** | `nova_market_data.csv` (M5/M6) |
| **Leaking columns** | `Return`, `Close`, `Open`, `High`, `Low` (same row) |
| **Evidence** | `corr(Return, same-day close-to-close) = 1.000`; `corr(Return, next-day close-to-close) = 0.0045`; `corr(Return, (Close−Open)/Open) = 0.949` |
| **Why it leaks** | `Return` is computed *from the `Close` on its own row*. Regressing `Return` on same-row OHLC is algebra, not prediction. |
| **Trap severity** | Maximum. The brief's wording — "predict future returns" — reads naturally as "model the `Return` column", which is exactly the wrong thing. |
| **Control** | `datasets.market.drop_always: [Open, High, Low, Close, Return]`. Target is constructed: `fwd_return_1d = Return.groupby(Ticker).shift(-1)` (plus a `t+5` robustness variant). |
| **Deliberate exhibit** | Train the leaked model *on purpose*, report its R², then report the clean model. The gap is a D8 slide and a D10 answer. |
| **Guide / viva ref** | D7 §5.2, D10 Q07 ("how did you prevent leakage?") |

**Cleared as safe (verified, not assumed):** `Momentum_20D` reproduces the trailing 20-day return at `corr = 1.000` and `Volatility_20D` reproduces the trailing 20-day return standard deviation at `corr = 1.000`. Both are strictly backward-looking and are **retained as features**.

---

## L-02 · `nova_hft_orderbook.csv` — three columns, one answer ⚠ CRITICAL

| | |
|---|---|
| **Dataset** | `nova_hft_orderbook.csv` (M9) |
| **Leaking columns** | `Future_Return_100ms`, `Future_Price_100ms` (when predicting `Price_Move_Class`) |
| **Evidence** | `corr(Future_Return_100ms, Future_Price_100ms/Mid_Price − 1) = 1.000`; a ±0.5 bp rule on `Future_Return_100ms` reproduces **90.5%** of `Price_Move_Class` labels |
| **Why it leaks** | All three are the same future observation at different encodings. Any one of them in `X` gives near-perfect accuracy. |
| **Control** | `datasets.hft.drop_always: [Future_Return_100ms, Future_Price_100ms]`; the multiclass target is `Price_Move_Class` only. |
| **Cleared as safe** | `Mid_Price`, `Microprice`, `Microprice_Minus_Mid`, `OBI_Level1`, `OBI_3Level`, `Spread`, `Relative_Spread`, `Total_Depth`, `Short_Term_Volatility` — all computed from the **current** book state. `OBI_3Level` correlates **+0.513** with `Future_Return_100ms`: that is the genuine microstructure signal, and it is the result the module should reproduce. |
| **Guide / viva ref** | D7 §9.3, D10 Q13 |

---

## L-03 · `nova_liquidity.csv` — `Liquidity_Gap` is an arithmetic identity ⚠ CRITICAL

| | |
|---|---|
| **Dataset** | `nova_liquidity.csv` (M7) |
| **Leaking columns** | `Expected_Inflows`, `Expected_Outflows` (when predicting `Liquidity_Gap`) |
| **Evidence** | `max abs( Liquidity_Gap − (Expected_Outflows − Expected_Inflows) ) = 0.01`; `corr = 1.000` |
| **Why it leaks** | `Liquidity_Gap` is not a quantity to be predicted; it is a subtraction of two columns sitting on the same row. |
| **Control** | `datasets.liquidity.drop_always: [Liquidity_Gap]`. Reframed as a genuine forecasting problem: predict `Expected_Outflows` (and inflows) **h = 5 business days ahead** from lagged history, then *derive* the gap. |
| **Secondary finding** | `Liquidity_Buffer` is near-constant — std **11.36** on a mean of **11,385** (CV ≈ 0.1%). It carries no variance and is unusable as a target. Recorded as a data limitation, not modelled. |
| **Guide / viva ref** | D7 §7.1, D10 Q15 |

---

## L-04 · `nova_options.csv` — Black-Scholes is a near-perfect feature ⚠ HIGH

| | |
|---|---|
| **Dataset** | `nova_options.csv` (M10) |
| **Leaking column** | `Black_Scholes_Price` (when predicting `Market_Price` and then claiming to "beat Black-Scholes") |
| **Evidence** | BS alone explains **R² = 0.9978** of `Market_Price` (`corr = 0.9988`); mean absolute residual **5.84**; median relative residual **−0.02%** |
| **Why it leaks** | `Market_Price` ≈ `Black_Scholes_Price` + noise. Feeding BS as a feature manufactures a "win over BS" that demonstrates nothing. |
| **Control — two experiments, both reported** | **(a) Fair fight:** ML on primitives only (`Spot`, `Strike`, `Time_to_Maturity`, `Volatility`, `Interest_Rate`, `Option_Type`) versus the analytic BS formula — *can ML rediscover Black-Scholes?* **(b) Residual model:** target `mispricing = Market_Price − Black_Scholes_Price`, which is what a real desk actually models. |
| **Metric hazard** | **85 rows** have `Black_Scholes_Price < 0.01`. MAPE explodes on these. Use RMSE/MAE; if MAPE is quoted at all, state the filter explicitly. |
| **Guide / viva ref** | D7 §10.2, D10 Q17 |

---

## L-05 · `nova_transactions.csv` — temporal **and** entity leakage ⚠ HIGH

| | |
|---|---|
| **Dataset** | `nova_transactions.csv` (M3) |
| **Vector A — temporal** | `Timestamp` spans 2025-01-01 → 2026-08-31 at uniform 29 m 08 s intervals. A random split trains on the future and tests on the past. |
| **Vector B — entity** | 2,500 customers across 30,000 rows, and **~19.9 customers share each of the 1,500 `Device_ID`s**. A random split lets the model memorise entities rather than learn behaviour. |
| **Vector C — feature construction** | Any rolling/aggregate feature (customer mean amount, device fraud rate) computed over the full frame embeds the future into the past. |
| **Control** | Primary: `TimeSeriesSplit` (expanding window) with **2026-06-01 → 2026-08-31 held out untouched**. Cross-check: `GroupKFold(groups=Customer_ID)` — the gap between the two quantifies entity memorisation and is itself a reportable result. All aggregates must be `shift(1)` **before** `rolling`. |
| **Cleared as safe** | `Amount / Historical_Avg_Transaction` uses only same-row values, both known at transaction time. It is the single strongest engineered feature (fraud rate **0.83% → 10.30%** across its deciles) and is legitimate. |
| **Guide / viva ref** | D7 §3.4, D10 Q07, Q09 |

---

## L-06 · `nova_loans.csv` — `Interest_Rate` encodes the incumbent scorecard ⚠ MEDIUM (judgement call)

| | |
|---|---|
| **Dataset** | `nova_loans.csv` (M2) |
| **Suspect column** | `Interest_Rate` — the single strongest feature, `corr(Default_Flag) = +0.308` (next: `Credit_Score` −0.187, `Debt_to_Income` +0.174) |
| **Is it leakage?** | **Not strictly.** The rate is set at origination and is genuinely known before default occurs, so it is *available* at scoring time. But under risk-based pricing it is a function of the bank's own internal risk assessment — so a model leaning on it is partly reading NovaFin's existing scorecard rather than the borrower. |
| **Control** | Train **dual models**: (i) full feature set, (ii) `Interest_Rate` excluded. Report both. Use the rate-free model for the business question *"can we out-predict our current pricing?"*, and the full model as the operational PD. |
| **Why this is the right answer** | It converts an ambiguity into a measured quantity instead of an assumption. The delta between (i) and (ii) *is* the value of the incumbent scorecard. |
| **Guide / viva ref** | D7 §2.3, D10 Q06 ("why this model?"), Q08 |

---

## L-07 · `nova_customers.csv` — `Estimated_CLV` is a derived column ⚠ MEDIUM

| | |
|---|---|
| **Dataset** | `nova_customers.csv` (M4) |
| **Issue** | `Estimated_CLV` correlates `Account_Balance` **0.697**, `Investment_Balance` **0.684**, `Annual_Income` **0.654**, `Loan_Balance` **0.607**. It is a *formula* over its own predictors, so "predicting" it from them is close to tautological. |
| **Control** | Do not present CLV regression as forecasting. Frame it as **attribution/imputation**: use SHAP to recover the generator's implied weights, and treat `Estimated_CLV` as a *supplied business quantity* in the M11 allocation. State the framing explicitly in the report. |
| **Secondary** | 17 rows have `Estimated_CLV == 0` — inspect before they distort MAPE or a log transform. |
| **Guide / viva ref** | D7 §4.3, D10 Q11 |

---

## L-08 · `strategic_initiatives.csv` — `Expected_ROI` is partially derived ⚠ LOW

| | |
|---|---|
| **Dataset** | `strategic_initiatives.csv` (M1) |
| **Issue** | `Expected_ROI` correlates **0.52** with a crude three-year ROI recomputed from `Revenue_Year1..3`, `Operating_Cost` and `Initial_Investment`. It is a derived field, not a forward-looking leak — and it is the strongest predictor of the target (`corr = 0.121`). |
| **Verdict** | **Keep**, but document the derivation so no examiner can claim it was smuggled in. Redundancy with the revenue columns is a *multicollinearity* issue for linear models, handled in `features/selection.py`. |
| **Guide / viva ref** | D7 §1.3 |

---

## Non-leakage findings that must travel with the register

These are not leaks, but they will be challenged in the viva and belong in the same audit trail.

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| **N-01** | **Churn has no learnable signal.** | 89 positives out of 5,000 (1.78%); **largest absolute correlation with any feature = 0.0225** (`Digital_Usage`); `Complaints` 0.011 | Report a documented **negative result** — permutation test against the null, power analysis, learning curve. Answer the "contact 1,000 customers" question on CLV × complaints × product depth instead. Do **not** tune until a fold looks good. |
| **N-02** | **`nova_customers.csv` does not join to anything.** | `Customer_ID` 300001–305000 vs loans 200001–205000 → **intersection = 0**. Loans ∩ transactions = **2,500**. | M11 integration happens at **portfolio/segment** level, not customer level. This is a data constraint and must be stated, not worked around silently. |
| **N-03** | **Macro variables are not shared across the cross-section.** | `nova_market_data.csv` has **15 distinct `GDP_Growth` values on every single date** — one per ticker | Real macro series are common factors. Treat as row-level noise; flag as a synthetic-data realism limitation. |
| **N-04** | **`Sector_Risk` is noise.** | `corr(Default_Flag) = +0.029`, 3,978 distinct values over 0.03–99.97 | Useful as a *demonstration* target for `features/selection.py` — a feature that should be dropped, and can be shown to be dropped. |
| **N-05** | **No origination date in `nova_loans.csv`.** | No datetime column exists | **Out-of-time credit validation is impossible.** Stated as a limitation; in production a PD model must be validated out-of-time. |
| **N-06** | **HFT timestamps are minute-resolution for a 100 ms horizon.** | 7,506 unique timestamps over 120,000 rows (~16 rows/minute) | Intra-minute ordering is ambiguous. Sort by `(Trading_Day, Timestamp, original_index)` and document the assumption. |
| **N-07** | **Under-collateralised loans are common.** | `Loan_Amount / Collateral_Value`: median 0.80, **max 2.00** | Justifies the collateral-aware LGD sensitivity alongside the brief's flat LGD = 40%. |

---

## How each control is enforced

| Layer | Mechanism | Phase |
|---|---|---|
| Declaration | `datasets.*.drop_always` in `configs/config.yaml` | 1 ✅ |
| Contract | `DatasetConfig.forbidden_features` (ids ∪ **entity key** ∪ targets ∪ audited leaks) | 1 ✅ |
| Automated detection | `data/validate.py` — four detector families rediscover every entry from the data itself | 2 ✅ |
| Feature boundary | `data/loader.make_feature_frame` — the only sanctioned way to build X | 2 ✅ |
| Split safety | `data/splits.py` + `assert_no_temporal_overlap` / `assert_no_group_overlap` | 2 ✅ |
| Artefact guard | `ModelBundle.__post_init__` refuses to save a bundle whose `feature_names` contains the target | 1 ✅ |
| Automated test | `tests/test_leakage.py` — one failing case per register entry | 2 ✅ |
| Build gate | GitHub Actions runs the suite on every push | 1 ✅ |
| Narrative | D7 §Validation + §Limitations; D8 "Leakage & Validation" slide | 9–10 |
| Defence | D10 viva answers Q06–Q09, Q11, Q13, Q15, Q17 | 10 |

---

*Every figure on this page was computed from the supplied CSVs during the Phase-0 audit. Nothing here is estimated or assumed.*


---

## Phase-2 addendum — what the automated scanners found

Running `data/validate.py` over the supplied CSVs reproduced **every** entry above with
no manual input, and added two refinements.

| Entry | Automated evidence |
|---|---|
| L-01 | `same_row_derivation` corr **+1.0000**; a constructed `shift(-1)` target scores **+0.0045** |
| L-02 | thresholds on `Future_Return_100ms` reproduce **100.0%** of `Price_Move_Class` labels |
| L-03 | `Liquidity_Gap == Expected_Outflows − Expected_Inflows`, max residual **0.01025** |
| L-04 | `corr(Black_Scholes_Price, Market_Price) = +0.9988` |
| N-01 | strongest admissible \|corr\| with churn **0.0225**; per-fold AUC SE **0.070** |
| N-02 | loans ∩ customers = **0**; loans ∩ transactions = **2,500** |

### L-09 · Entity keys must not be model features ⚠ MEDIUM (added in Phase 2)

The per-module signal chart exposed `Customer_ID` sitting in the fraud feature matrix.
A raw entity key is an invitation to memorise entities rather than learn behaviour —
the same failure L-05 guards against at split time. `DatasetConfig.forbidden_features`
now includes `group_column`, so `Customer_ID` (fraud) and `Trading_Day` (order book)
are removed automatically. Entity information enters only as an explicitly engineered,
causally computed aggregate.

### N-08 · Statistical power is the binding constraint on two modules

Using the Hanley & McNeil (1982) closed form for the standard error of an AUC, computed
before any model is fitted:

| Module | Positives | Per fold | AUC std. error | 95% interval |
|---|---:|---:|---:|---|
| **initiatives** | 138 / 180 | 27.6 | **0.096** | ±0.188 |
| **customers** | 89 / 5,000 | 17.8 | **0.070** | ±0.137 |
| loans | 997 / 5,000 | 199.4 | 0.022 | ±0.044 |
| transactions | 684 / 30,000 | 136.8 | 0.025 | ±0.050 |

The initiatives module has the **worst** per-fold precision of the four — worse than
churn — because only 42 negatives exist in total. This is the quantitative
justification for the 10-repeat cross-validation configured for both modules, and it
means any single-fold number from either is noise and must not be quoted.


---

## Phase-3 addendum — causality proved, and a new leak class found

### The proof

Every module's features are now verified by **perturbation**, not by review:
build features, corrupt only the final 20% of rows (sign-flip, x1000, plus
noise), rebuild, and assert every model feature in the untouched prefix is
**bit-identical**. A feature that looked forward would move.

| Module | Rows tested | Model features proved causal |
|---|---:|---:|
| initiatives | 180 | 18 |
| loans | 5,000 | 19 |
| transactions | 30,000 | 28 |
| customers | 5,000 | 16 |
| market (RELIANCE only) | 1,457 | 28 |
| liquidity | 1,216 | 58 |
| options | 10,000 | 22 |
| hft (NOVA01 only) | 12,023 | 28 |

Panel modules are sliced to a single entity so that row order equals time
order; otherwise "the last 20% of rows" would mean "three whole tickers", which
tests something else entirely.

`tests/test_features.py::test_assert_causal_catches_a_deliberate_leak` runs a
knowingly leaky builder and requires the proof to reject it — a test that can
never fail proves nothing.

### L-10 · Dataset-wide statistics leak across the split ⚠ HIGH (new)

The causality proof rejected two features that have **no time dimension at
all**, which is what makes this entry worth its own number:

| Feature | Why it leaked |
|---|---|
| `rate_spread = Interest_Rate − min(Interest_Rate)` (M2) | the minimum is taken over **every** row, so test rows shape a training feature |
| `value_rank`, `attrition_proxy`, `priority_score` (M4) | percentile ranks computed across the whole book |

The leak is **across the train/test split, not across time**. Any statistic
learned from data — a minimum, a mean, a rank, a category list, a scaler's
standard deviation, even a selected feature list — is a *fitted parameter* and
must be estimated on training rows only.

**Controls applied:**

* `rate_spread` deleted. Centring belongs in the preprocessor, which is fitted
  inside each fold; trees are invariant to a constant offset, so nothing was
  lost.
* The M4 ranks renamed `decision_*` and added to
  `datasets.customers.drop_always`. They remain the answer to the
  "contact 1,000 customers" question — a legitimate whole-book business
  ranking — but they are no longer model inputs.
* `features/encoders.py` returns an **unfitted** `ColumnTransformer` by design,
  so the leak-free usage is the only usage.

### L-11 · Naive target encoding ⚠ CRITICAL (prevented by construction)

Replacing a category with the mean target of its rows lets every row see its
own label. Measured on this data, using `Device_ID` (1,500 levels) against
`Fraud_Flag`:

| Encoding | \|corr\| with target |
|---|---:|
| Naive, fitted on all rows | **0.2304** — entirely manufactured |
| Out-of-fold (`OutOfFoldTargetEncoder`) | **0.0002** — the truth |

Device identity carries no real fraud signal in this dataset. The naive figure
is pure leakage, and a model trained on it would post an excellent CV score and
fail completely out of sample. `OutOfFoldTargetEncoder` encodes each training
row from the *other* folds only, with smoothing toward the prior so a
three-row category cannot dictate its own encoding.

Reference: Micci-Barreca D (2001), *ACM SIGKDD Explorations* 3(1).
https://doi.org/10.1145/507533.507538


---

## Phase-5 addendum — tuning cannot become a leak

Hyper-parameter search is a place where leakage re-enters a clean project,
because a tuning loop repeats a mistake hundreds of times and *rewards* it.
Three controls, all enforced in code:

### T-01 · The objective scores out-of-fold predictions only

Every Optuna trial runs the same `cross_validate_model` loop as Level 1, so the
whole pipeline — imputer, scaler, encoder — is fitted inside each fold.
`_assert_objective_is_oof` additionally checks the returned mask on every trial,
so a future edit cannot quietly start scoring training rows. A tuning study is
exactly where such a bug would go unnoticed: every trial would simply improve.

### T-02 · Pruning must not act on a single fold

Optuna's `MedianPruner` compares a trial's value at step *k* against other
trials at step *k*. If step *k* is fold 1, a trial dies on one fold's evidence.
The Phase-2 power analysis measured per-fold AUC standard errors of **0.096**
(initiatives) and **0.070** (churn) — wider than the gap between a good and a
bad configuration.

`SafeMedianPruner` therefore enforces a minimum number of reported folds before
any prune is permitted:

| Module | `min_folds_before_prune` | Per-fold AUC std error |
|---|---:|---:|
| initiatives | **3** | 0.096 |
| customers (churn) | **3** | 0.070 |
| loans, transactions, market, liquidity, options | 2 | 0.022 – 0.025 |
| hft | 1 | single train/val split by design |

Verified behaviour on a clearly-bad trial (0.40 against a median of 0.705):

| `min_folds` | 1 fold reported | 2 folds | 3 folds |
|---|---|---|---|
| 1 | prune | prune | prune |
| 2 | **no prune** | prune | prune |
| 3 | **no prune** | **no prune** | prune |

Pruning still kills a hopeless trial after 3 of 50 fits. It simply cannot kill
it on noise.

### T-03 · The search itself can overfit — so the churn study is kept short

With 89 positives, running 500 trials would eventually surface a configuration
with a flattering cross-validated score. That score would be **selection
noise**, not signal — the tuner would be searching for a lucky fold assignment.
The churn study is capped at 30 trials, and the reason is recorded in
`configs/search_spaces.yaml` as a `note` field, which
`tests/test_tune.py::test_churn_study_is_deliberately_short` asserts is present.

The honest measurement for that module remains the permutation test in notebook
`03`, not the tuned score.
