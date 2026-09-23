# NovaFin Group Capstone — ML-Driven Enterprise Financial Strategy, Risk & Decision Support

[![CI](https://github.com/yadavdipesh/novafin-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/yadavdipesh/novafin-capstone/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-informational.svg)](https://github.com/astral-sh/ruff)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yadavdipesh/novafin-capstone/blob/main/notebooks/01_setup_and_eda.ipynb)

> **ePGD in AI & Data Science, IIIT Bombay — C5 Machine Learning in Finance**
> Capstone, Group 2 · Author: Dipesh Kumar Yadav
> **Status: COMPLETE.** All ten phases delivered — from repository scaffold to the
> Board recommendation, the interactive demo, and the written deliverables.

---

## The one question this project answers

> NovaFin's Board grants **₹1,000 crore of incremental capital**.
> **Where should it be deployed, what risks does that take, and how are those risks controlled?**

Nine analytical modules each produce one number that feeds a single integrated allocation. The models are the means; the allocation is the deliverable.

```
Data ──▶ ML Prediction ──▶ Financial Metric ──▶ Risk Assessment ──▶ Business Decision
```

---

## Architecture

```
                         configs/config.yaml   ← single source of truth
                                  │                (paths, seeds, costs,
                                  │                 leakage rules, CV recipes)
                                  ▼
  data/raw/*.csv ──▶ data.loader ──▶ data.validate ──▶ features.engineer
   (never committed)      │               │                    │
                          │         schema + leakage      causal features only
                          │         assertions fail loud        │
                          ▼                                     ▼
                    utils.seed ─────────────────────────▶ models.build
                    utils.io   (ModelBundle:                    │
                    utils.theme  model+preprocessor      ┌──────┴───────┐
                                 +feature list)          │  L1 baseline │
                                                         │  L2 Optuna   │
                                                         │  L3 deep FT  │
                                                         │  L4 campaign │
                                                         └──────┬───────┘
                                                                ▼
   MLflow (local file store) ◀── evaluate ── explain(SHAP) ── predict ── app.py
                 │                                                        │
                 └────────────▶ reports/tables/leaderboard.csv ◀──────────┘
                                          │
                                          ▼
                        Module 11 — integrated ₹1,000 cr allocation
```

---

## Quickstart

### Google Colab (free tier, T4)

```python
# Cell 1 — bootstrap. PYTHONHASHSEED must be set before anything else imports.
import os
os.environ["PYTHONHASHSEED"] = "42"

!git clone https://github.com/yadavdipesh/novafin-capstone.git
%cd novafin-capstone
!pip install -q -r requirements.txt
!pip install -q -e .

# Cell 2 — mount Drive and point the config at the course data.
from google.colab import drive
drive.mount("/content/drive")
os.environ["NOVAFIN_DATA_RAW"] = "/content/drive/MyDrive/ePGD/C5 ML In Finanace/Data"

# Cell 3 — standard project preamble, used by every notebook.
from novafin.config import load_config
from novafin.utils.seed import seed_everything
from novafin.utils.theme import apply_theme
from novafin.utils.logging_utils import setup_logging

cfg = load_config()
setup_logging("INFO", log_file=cfg.paths.logs / "session.log")
seed_everything(cfg.reproducibility.seed)
apply_theme()
print("config fingerprint:", cfg.fingerprint())
```

### Local

```bash
git clone https://github.com/yadavdipesh/novafin-capstone.git
cd novafin-capstone
make setup      # pinned deps + editable install
make verify     # config loads, seed pins, fingerprint prints
make test       # full pytest suite
```

`make help` lists every target.

---

## Repository layout

| Path | Contents |
|---|---|
| `configs/config.yaml` | **Single source of truth** — paths, seed, dataset registry, leakage rules, CV recipes, every finance assumption |
| `configs/theme.yaml` | Mubadala theme tokens (colours + fonts only) |
| `configs/models.yaml` | **Model declarations** — a new model is a config entry, never a code change |
| `configs/search_spaces.yaml` | **85 tunable parameters, each with a written rationale** (enforced by tests) |
| `configs/campaigns/` | **D4 Level 4** — new tuning campaigns as YAML, zero code changes |
| `configs/tuning/` | D4 Level-4 campaign YAMLs — new tuning runs with **zero code changes** |
| `src/novafin/` | All pipeline logic. Notebooks orchestrate; they never contain the pipeline |
| `src/novafin/finance/` | NPV · ECL · CLV · VaR/ES · portfolio · Greeks · **Module 11 allocation** |
| `app.py` | Gradio demo — credit, fraud, live allocation, model governance |
| `notebooks/` | `00_environment_check` → `07_inference_demo`, eight notebooks in order |
| `docs/LEAKAGE_REGISTER.md` | **Every leak found in the data, with evidence and control** |
| `tests/` | Contract tests — including tests that fail if a leakage rule is deleted |
| `data/raw/` | Course CSVs — **git-ignored, never redistributed** |
| `artifacts/`, `mlruns/`, `reports/` | Generated outputs (git-ignored) |

---

## Results

Generated by `python -m novafin.models.leaderboard` from the MLflow run store —
never typed. Run `notebooks/03_baseline_models.ipynb` to populate it.

<!-- Populated from reports/tables/leaderboard.csv as each phase completes.
     No number appears here until it has been produced by code on the real data. -->

| Module | Model | Primary metric | Value | Produced by |
|---|---|---|---|---|
| M1 Capital allocation | — | PR-AUC / Brier | `<<FILL AFTER RUN>>` | `notebooks/03_baseline_models.ipynb` |
| M2 Credit risk | — | ROC-AUC / KS / Brier | `<<FILL AFTER RUN>>` | `notebooks/03_baseline_models.ipynb` |
| M3 Fraud | — | PR-AUC / expected cost | `<<FILL AFTER RUN>>` | `notebooks/03_baseline_models.ipynb` |
| M4 Churn & CLV | — | PR-AUC / lift@1000, MAE | `<<FILL AFTER RUN>>` | `notebooks/03_baseline_models.ipynb` |
| M5–M6 Equity & portfolio | — | Rank IC / Sharpe | `<<FILL AFTER RUN>>` | `notebooks/06_evaluation_and_explainability.ipynb` |
| M7–M8 Liquidity & VaR | — | MAE / Kupiec | `<<FILL AFTER RUN>>` | `notebooks/06_evaluation_and_explainability.ipynb` |
| M9 HFT | — | Macro-F1 / net PnL | `<<FILL AFTER RUN>>` | `notebooks/06_evaluation_and_explainability.ipynb` |
| M10 Derivatives | — | RMSE vs Black-Scholes | `<<FILL AFTER RUN>>` | `notebooks/06_evaluation_and_explainability.ipynb` |
| M11 Allocation | — | Risk-adjusted return | `<<FILL AFTER RUN>>` | `notebooks/06_evaluation_and_explainability.ipynb` |

---

## The fine-tuning ladder (D4)

| Level | Technique | Where |
|---|---|---|
| 1 | baselines + cross-validation | `models/train.py`, notebook `03` |
| 2 | Optuna: pruning, rationale, **resumable** SQLite studies | `models/tune.py`, notebook `04` |
| 3 | staged/warm-start boosting · self-supervised FT-Transformer + **LoRA** · layer-wise unfreezing with discriminative LRs | `models/finetune.py`, notebook `05` |
| 4 | **EXTRA FINE-TUNING hook** — new search space, more trials, stacking, threshold optimisation, all from YAML | `models/campaign.py`, `configs/campaigns/` |

```bash
make campaigns                                                    # list
make campaign-dry CAMPAIGN=configs/campaigns/example_extra_tuning.yaml   # validate
make campaign     CAMPAIGN=configs/campaigns/example_extra_tuning.yaml   # run
```

> **On PEFT/LoRA:** there is no text or image data in this package, so adapting a
> pretrained language model would be decorative. LoRA is instead applied to an
> FT-Transformer pre-trained **self-supervised** on the pooled tabular data —
> one frozen backbone, one cheap adapter per module. That is the genuine PEFT
> pattern applied to the data that exists.

## Deliverables

| ID | Deliverable | Where |
|---|---|---|
| D1 | Repo tree | this repository |
| D2 | Source modules (config, data, features, models, finance, evaluate, explain, predict, app) | `src/novafin/`, `app.py` |
| D3 | Eight Colab notebooks, in order | `notebooks/` |
| D4 | **Fine-tuning ladder, Levels 1–4** | `models/{train,tune,finetune,campaign}.py` |
| D5 | MLflow tracking + generated leaderboard | `tracking.py`, `models/leaderboard.py` |
| D6 | Reproducibility: pinned deps, seeds, fingerprints, versioned bundles | `requirements.txt`, `utils/` |
| D7 | **Project Guide** | `NovaFin_Project_Guide.docx` |
| D8 | **Board deck** | `NovaFin_Board_Deck.pptx` |
| D9 | README, LICENCE, CI, pytest suite (**287 tests**) | this file, `.github/`, `tests/` |
| D10 | **Viva prep pack** | `NovaFin_Viva_Prep.docx` |

## Running the demo

```bash
python app.py          # prints a public share link; no account, no API key
```

Four tabs: credit scoring → PD → ECL → Approve/Review/Reject; fraud scoring at
the cost-optimal threshold; the ₹1,000 crore allocation re-optimised live; and
model governance, because a decision engine that hides its caveats is the wrong
engine.

## Validation & leakage

The Phase-0 audit of the supplied CSVs found **eight** leakage vectors, three of them critical. Read [`docs/LEAKAGE_REGISTER.md`](docs/LEAKAGE_REGISTER.md) before running anything. Headlines:

* **`Return` in `nova_market_data.csv` is the *same-day* return** — `corr = 1.000` with same-day close-to-close. Modelling it directly is algebra, not prediction.
* **`Liquidity_Gap` is an exact identity** — `Expected_Outflows − Expected_Inflows`, max deviation 0.01.
* **`Future_Price_100ms` and `Future_Return_100ms` are the HFT label** in two other encodings.

Each is declared in `configs/config.yaml` as a `drop_always` rule and asserted by `tests/`. Deleting a rule breaks the build.

---

## Reproducibility

| Guarantee | Mechanism |
|---|---|
| Same seed everywhere | `seed_everything()` pins `random`, `numpy`, `torch`, CUDA, cuDNN, cuBLAS, DataLoader workers |
| `PYTHONHASHSEED` genuinely effective | Exported by the `Makefile`, CI `env:` and notebook cell 1 — **before** the interpreter starts |
| Pinned dependencies | `requirements.txt`, exact `==` pins, **licence stated for every package** |
| Traceable numbers | `Config.fingerprint()` + SHA-256 of every raw CSV, recorded in a `RunManifest` beside every result |
| Artefacts that cannot drift | `ModelBundle` refuses to save a model without its preprocessor and feature list, and re-validates the contract at predict time |

---

## Licences

* **Code** — MIT (see [`LICENSE`](LICENSE)).
* **Libraries** — 100% free and open-source; each licence is annotated inline in [`requirements.txt`](requirements.txt).
* **Data** — synthetic, course-supplied, educational use only. **Not redistributed here.**
* **Fonts** — Archivo and Tajawal under SIL OFL 1.1, downloaded at runtime. Interstate is proprietary and is referenced by *name only* in generated Office documents; no font data is embedded or committed.

---

## Academic integrity

Original work. Every external idea, paper and code reference is cited with a link in the D7 Project Guide and the D8 deck. No result, metric or figure appears in any deliverable until it has been produced by code in this repository running on the supplied data — placeholders read `<<FILL AFTER RUN>>` until then.
