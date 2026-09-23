# Campaigns — the D4 Level 4 "EXTRA FINE-TUNING" hook

> **The contract:** launch a new tuning campaign — new search space, more
> trials, ensembling/stacking, threshold optimisation — **by editing YAML
> only, with zero code changes.**

## Running one

```bash
make campaign CAMPAIGN=configs/campaigns/example_extra_tuning.yaml
make campaign-dry CAMPAIGN=configs/campaigns/example_extra_tuning.yaml   # validate only
```

or in a notebook:

```python
from novafin.models.campaign import run_campaign
result = run_campaign("configs/campaigns/example_extra_tuning.yaml")
```

Results land in `reports/tables/campaign_<name>.csv` and, if a tracker is
passed, in MLflow.

## Anatomy of a campaign file

```yaml
name: my_campaign               # used for the output filename and study suffix
description: what and why
author: Dipesh Kumar Yadav

steps:
  - module: loans               # a dataset key from configs/config.yaml
    operation: tune             # tune | stack | threshold | finetune
    models: [lightgbm]          # optional; defaults to the first non-baseline
    enabled: true               # switch a step off without deleting it
    note: why this step exists  # travels into the report
    params:
      n_trials: 200
```

## The four operations

| Operation | What it does | Required `params` |
|---|---|---|
| `tune` | New or extended Optuna study | — (`n_trials`, `search_spaces`, `study_suffix` optional) |
| `stack` | Meta-learner over base models' **out-of-fold** predictions | `meta_model` (+ ≥2 `models`) |
| `threshold` | Re-optimise the decision threshold against a cost matrix — **retrains nothing** | `cost_false_negative`, `cost_false_positive` |
| `finetune` | Level-3 staged/warm-start boosting | `schedule` |

## Why no code change is needed

Each earlier phase removed one hard-coded thing:

* **Phase 4** — estimators became a dotted path in `configs/models.yaml`,
  resolved with `importlib`.
* **Phase 5** — search spaces became YAML, and a campaign may point `tune` at
  **its own** `search_spaces` file.
* **Phase 4** also exposed `pipeline_factory` and `fold_callback` on the CV
  loop, so a campaign can substitute a stacked or calibrated pipeline.

Level 4 is the payoff for those decisions, not a new mechanism.

## Safety

* `validate_campaign` runs **before any model is fitted** — a campaign may run
  unattended for an hour, so a typo in step 4 must fail before step 1 starts.
* A campaign `tune` step writes to a **separate Optuna study** (suffixed with
  the campaign name), so it can never overwrite the Level-2 study.
* A failing step is recorded and skipped; the rest of the campaign continues.
