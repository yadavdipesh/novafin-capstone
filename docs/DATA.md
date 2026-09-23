# Data Card — NovaFin Synthetic Datasets

## Provenance and licence

Eight CSV files supplied by the course as part of the *NovaFin Group Capstone
Project* package. Per the accompanying `README.txt`:

* all monetary variables are synthetic and expressed in **INR lakh** unless the
  column context indicates otherwise;
* the datasets are designed for **educational use, not real-world financial
  decisions**;
* the generator used **random seed 42** — which is why this project uses the
  same seed.

**They are NOT redistributed in this repository.** `data/raw/` is git-ignored
and `tests/test_theme_and_hygiene.py::test_course_data_is_not_committed`
fails the build if a CSV ever appears inside the tree.

## Obtaining the data

Place the eight CSVs in a folder and point the config at it:

```bash
export NOVAFIN_DATA_RAW="/path/to/C5 ML In Finanace/Data"
```

In Colab, mount Drive and set the same variable (see README quickstart).

## Contents (verified by the Phase-0 audit, not assumed)

| File | Rows | Cols | Module | Target | Positive rate |
|---|---:|---:|---|---|---|
| `strategic_initiatives.csv` | 180 | 14 | M1 Capital allocation | `Historical_Success` | 76.67% |
| `nova_loans.csv` | 5,000 | 14 | M2 Credit risk | `Default_Flag` | 19.94% |
| `nova_transactions.csv` | 30,000 | 12 | M3 Fraud | `Fraud_Flag` | 2.28% |
| `nova_customers.csv` | 5,000 | 13 | M4 Churn & CLV | `Churn_Flag` | 1.78% |
| `nova_market_data.csv` | 21,855 | 15 | M5/M6 Equity & portfolio | forward return | n/a |
| `nova_liquidity.csv` | 1,216 | 13 | M7 Liquidity (inferred) | `Expected_Outflows` | n/a |
| `nova_options.csv` | 10,000 | 11 | M10 Derivatives | `Market_Price` | n/a |
| `nova_hft_orderbook.csv` | 120,000 | 33 | M9 HFT | `Price_Move_Class` | UP 41.70 / DOWN 41.91 / FLAT 16.38% |

**Missingness is 0.00% and duplicate rows are 0 in all eight files.** Total
in-memory footprint approximately 82 MB — comfortably inside the Colab free-tier
envelope of ~12 GB RAM.

## Coverage windows

| File | Span |
|---|---|
| `nova_market_data.csv` | 2021-01-29 → 2026-08-31 (15 tickers × 1,457 days, balanced panel) |
| `nova_liquidity.csv` | 2022-01-03 → 2026-08-31 (business days only) |
| `nova_transactions.csv` | 2025-01-01 → 2026-08-31 (uniform 29 m 08 s intervals) |
| `nova_hft_orderbook.csv` | 20 trading days, January 2026, 10 synthetic stocks |
| `nova_options.csv` | expiries 2026-09-13 → 2028-08-31 |

## Known limitations

See `docs/LEAKAGE_REGISTER.md` for the full audit. In summary:

1. **No entity join across customers.** `nova_customers.csv` uses IDs
   300001–305000; `nova_loans.csv` uses 200001–205000 — **intersection zero**.
   Loans ∩ transactions = 2,500. Integration is therefore at portfolio level.
2. **No loan origination date**, so out-of-time credit validation is impossible.
3. **Churn carries no learnable signal** (89 positives, max |corr| 0.0225).
4. **Macro variables are not common factors** — `GDP_Growth` takes 15 distinct
   values on every date in the market panel, one per ticker.
5. **HFT timestamps are minute-resolution** for a 100 ms prediction horizon.
6. **Three targets are arithmetic identities or same-row derivations** — see
   L-01, L-02, L-03.
