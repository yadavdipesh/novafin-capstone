"""
novafin-capstone/tests/test_tune.py

Tests for D4 Level 2 - the Optuna tuning layer.

Four families, each guarding something specific:

1. **Search-space validation.** Every declared parameter must carry a ``why``.
   That is what turns the brief's "search-space rationale" requirement into an
   enforced property of the repository rather than a claim - an undocumented
   range fails the build.
2. **Bound sanity.** Log-scaled parameters need a positive lower bound;
   ``low < high``; capacity parameters must respect the sample size. These
   catch the copy-a-blog-post failure mode.
3. **The pruning guard.** ``SafeMedianPruner`` must refuse to act before
   ``min_folds`` folds have reported. Phase 2 measured per-fold AUC standard
   errors of 0.096 (initiatives) and 0.070 (churn); pruning on one fold there
   selects for noise.
4. **Resumability.** A study is killed mid-run and restarted; the trial count
   must continue rather than reset. On Colab free tier this is the difference
   between finishing a 60-trial study and never finishing one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from novafin.config import load_config
from novafin.models.build import available_modules, load_model_specs
from novafin.models.tune import (
    DEFAULT_SPACES_PATH,
    MINIMISE_METRICS,
    SearchSpace,
    StudySettings,
    TuningResult,
    load_search_spaces,
    study_summary,
    validate_search_space,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _declared_modules() -> list[str]:
    raw = yaml.safe_load(DEFAULT_SPACES_PATH.read_text(encoding="utf-8")) or {}
    return [k for k in raw if k != "defaults"]


def _all_spaces() -> list[tuple[str, str, SearchSpace]]:
    out: list[tuple[str, str, SearchSpace]] = []
    for module in _declared_modules():
        spaces, _ = load_search_spaces(module)
        out.extend((module, name, space) for name, space in spaces.items())
    return out


# ==========================================================================
# 1 - Declarations are complete and documented
# ==========================================================================
def test_search_space_file_exists() -> None:
    assert DEFAULT_SPACES_PATH.exists(), "configs/search_spaces.yaml is a graded artefact"


@pytest.mark.parametrize("module", _declared_modules())
def test_every_declared_space_validates(module: str) -> None:
    spaces, _ = load_search_spaces(module)
    assert spaces, f"'{module}' declares no models"
    for name, space in spaces.items():
        problems = validate_search_space(space)
        assert not problems, f"{module}/{name}: " + "; ".join(problems)


def test_every_parameter_has_a_rationale() -> None:
    """The brief requires a search-space rationale. This enforces it."""
    undocumented: list[str] = []
    for module, name, space in _all_spaces():
        for parameter, spec in space.params.items():
            if len(str(spec.get("why", "")).strip()) < 20:
                undocumented.append(f"{module}/{name}.{parameter}")
    assert not undocumented, (
        "parameters without a substantive rationale: " + ", ".join(undocumented)
    )


def test_tunable_parameter_count_is_substantial() -> None:
    total = sum(len(space.params) for _, _, space in _all_spaces())
    assert total >= 50, f"only {total} tunable parameters declared"


def test_every_tuned_model_exists_in_models_yaml() -> None:
    """A search space for a model that is not declared would never run."""
    aliases = {"options_residual": "options"}
    for module, name, _ in _all_spaces():
        catalog_key = aliases.get(module, module)
        if catalog_key not in available_modules():
            continue
        assert name in load_model_specs(catalog_key).names(), (
            f"'{name}' has a search space under '{module}' but is not declared "
            "in configs/models.yaml"
        )


def test_unknown_module_raises_helpfully() -> None:
    with pytest.raises(KeyError, match="No search space declared"):
        load_search_spaces("not_a_module")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_search_spaces("loans", path=tmp_path / "absent.yaml")


# ==========================================================================
# 2 - Bounds are sane and data-appropriate
# ==========================================================================
def test_log_scaled_parameters_have_positive_lower_bounds() -> None:
    """log(0) is undefined - a silent source of broken studies."""
    for module, name, space in _all_spaces():
        for parameter, spec in space.params.items():
            if spec.get("log"):
                assert spec["low"] > 0, f"{module}/{name}.{parameter} is log-scaled with low <= 0"


def test_all_numeric_bounds_are_ordered() -> None:
    for module, name, space in _all_spaces():
        for parameter, spec in space.params.items():
            if spec.get("type") in {"float", "int"}:
                assert spec["low"] < spec["high"], f"{module}/{name}.{parameter}"


def test_multiplicative_parameters_use_log_scale() -> None:
    """Learning rates and penalties act multiplicatively; uniform wastes trials."""
    should_be_log = {"learning_rate", "reg_alpha", "reg_lambda", "alpha", "C",
                     "gamma", "min_child_weight", "learning_rate_init",
                     "scale_pos_weight"}
    offenders: list[str] = []
    for module, name, space in _all_spaces():
        for parameter, spec in space.params.items():
            if parameter in should_be_log and spec.get("type") == "float" and not spec.get("log"):
                offenders.append(f"{module}/{name}.{parameter}")
    assert not offenders, "multiplicative parameters sampled uniformly: " + ", ".join(offenders)


def test_capacity_is_bounded_by_sample_size_on_small_modules() -> None:
    """180 rows cannot fill 31 leaves. The ceiling must reflect the data."""
    cfg = load_config()
    spaces, _ = load_search_spaces("initiatives")
    n_rows = cfg.dataset("initiatives").expected_rows or 180

    leaves = spaces["lightgbm"].params["num_leaves"]
    assert leaves["high"] <= 20, (
        f"num_leaves high={leaves['high']} on {n_rows} rows leaves "
        f"{n_rows / leaves['high']:.0f} rows per leaf"
    )
    depth = spaces["random_forest"].params["max_depth"]
    assert depth["high"] <= 8, "a deep forest on 180 rows memorises individual initiatives"


def test_fraud_min_child_samples_reflects_the_base_rate() -> None:
    """At a 2.28% base rate a 20-row leaf holds 0.46 positives on average."""
    spaces, _ = load_search_spaces("transactions")
    floor = spaces["lightgbm"].params["min_child_samples"]["low"]
    assert floor >= 20, f"min_child_samples floor of {floor} fits individual frauds"


def test_scale_pos_weight_range_brackets_full_inverse_balance() -> None:
    """The optimum must be interior, not pinned to a boundary."""
    cfg = load_config()
    spaces, _ = load_search_spaces("transactions")
    spec = spaces["lightgbm"].params["scale_pos_weight"]
    positive_rate = cfg.dataset("transactions").positive_rate or 0.0228
    full_balance = (1 - positive_rate) / positive_rate       # ~43
    assert spec["low"] <= 1.0
    assert spec["high"] > full_balance, (
        f"high={spec['high']} does not exceed full inverse balance {full_balance:.1f}"
    )


# ==========================================================================
# 3 - Study settings
# ==========================================================================
def test_minimisation_modules_declare_the_right_direction() -> None:
    for module in ("liquidity", "options"):
        _, settings = load_search_spaces(module)
        assert settings.minimise, f"'{module}' optimises {settings.objective}, which is lower-is-better"


def test_maximisation_modules_declare_the_right_direction() -> None:
    for module in ("loans", "transactions", "market", "hft"):
        _, settings = load_search_spaces(module)
        assert not settings.minimise


def test_objectives_match_the_phase0_metric_choices() -> None:
    """The metric per module follows the Phase-0 audit, not convenience."""
    expected = {
        "initiatives": "pr_auc",      # calibration feeds Expected NPV
        "loans": "ks",                # credit-industry standard
        "transactions": "pr_auc",     # 2.28% base rate; ROC-AUC flatters
        "customers": "pr_auc",        # 1.78% base rate
        "market": "mean_ic",          # ordering, not level
        "liquidity": "mae",
        "options": "rmse",
        "hft": "macro_f1",            # FLAT is only 16.4%
    }
    for module, objective in expected.items():
        _, settings = load_search_spaces(module)
        assert settings.objective == objective, f"'{module}' optimises {settings.objective}"


def test_low_power_modules_require_more_folds_before_pruning() -> None:
    """The measured justification: fold noise of 0.096 and 0.070."""
    for module in ("initiatives", "customers"):
        _, settings = load_search_spaces(module)
        assert settings.min_folds_before_prune >= 3, (
            f"'{module}' has high per-fold variance; pruning on 2 folds selects noise"
        )


def test_churn_study_is_deliberately_short() -> None:
    """With 89 positives, a long search finds a lucky seed, not a model."""
    _, settings = load_search_spaces("customers")
    assert settings.n_trials <= 40
    assert settings.note, "the short-study decision must be documented in the config"


def test_settings_merge_defaults_with_module_overrides() -> None:
    _, loans = load_search_spaces("loans")
    _, churn = load_search_spaces("customers")
    assert loans.seed == 42 and churn.seed == 42          # from defaults
    assert loans.n_trials != churn.n_trials               # module overrides


# ==========================================================================
# 4 - The pruning guard
# ==========================================================================
def _fake_study(direction: str, values: list[float], steps: int = 3):
    optuna = pytest.importorskip("optuna")

    class _Trial:
        def __init__(self, value):
            self.state = type("S", (), {"name": "COMPLETE"})()
            self.value = value
            self.intermediate_values = {i: value for i in range(1, steps + 1)}

    class _Study:
        def __init__(self):
            self.direction = direction
            self.trials = [_Trial(v) for v in values]

    return _Study()


class _ReportingTrial:
    def __init__(self, values: list[float]) -> None:
        self.intermediate_values = dict(enumerate(values, start=1))


@pytest.mark.parametrize(("min_folds", "reported", "expected"), [
    (1, 1, True),    # no guard - prunes on a single fold
    (2, 1, False),   # guard blocks a one-fold prune
    (2, 2, True),
    (3, 1, False),
    (3, 2, False),   # the low-power setting: two folds is still not enough
    (3, 3, True),
])
def test_safe_median_pruner_respects_min_folds(min_folds: int, reported: int, expected: bool) -> None:
    pytest.importorskip("optuna")
    from novafin.models.tune import SafeMedianPruner

    study = _fake_study("maximize", [0.70, 0.71, 0.69, 0.72, 0.70, 0.71, 0.70, 0.69, 0.72, 0.71, 0.70, 0.71])
    pruner = SafeMedianPruner(n_startup_trials=5, n_warmup_steps=1, min_folds=min_folds)
    trial = _ReportingTrial([0.40] * reported)             # clearly below median
    assert pruner.prune(study, trial) is expected


def test_safe_median_pruner_keeps_a_good_trial() -> None:
    pytest.importorskip("optuna")
    from novafin.models.tune import SafeMedianPruner

    study = _fake_study("maximize", [0.70] * 12)
    pruner = SafeMedianPruner(n_startup_trials=5, n_warmup_steps=1, min_folds=2)
    assert pruner.prune(study, _ReportingTrial([0.85, 0.86, 0.84])) is False


# ==========================================================================
# 5 - Result bookkeeping
# ==========================================================================
def test_improvement_sign_flips_for_minimisation_metrics() -> None:
    """A lower RMSE is an improvement; a lower KS is not."""
    maximising = TuningResult(module="loans", model_name="m", objective="ks",
                              baseline_value=0.30, best_value=0.38)
    assert maximising.improvement == pytest.approx(0.08)

    minimising = TuningResult(module="options", model_name="m", objective="rmse",
                              baseline_value=8.0, best_value=5.0)
    assert minimising.improvement == pytest.approx(3.0)


def test_minimise_metric_set_covers_the_error_measures() -> None:
    for metric in ("rmse", "mae", "brier", "log_loss", "expected_cost"):
        assert metric in MINIMISE_METRICS


def test_study_settings_infers_direction_from_the_metric() -> None:
    """A config that says 'maximize rmse' is a typo, and is corrected."""
    settings = StudySettings(module="x", objective="rmse", direction="maximize")
    assert settings.minimise, "rmse is lower-is-better regardless of the declared direction"


def test_study_summary_is_empty_without_results() -> None:
    assert study_summary([]).empty


def test_study_summary_reports_pruning_share() -> None:
    result = TuningResult(module="loans", model_name="lightgbm", objective="ks",
                          baseline_value=0.30, best_value=0.35,
                          n_trials=50, n_complete=30, n_pruned=18, n_failed=2)
    row = study_summary([result]).iloc[0]
    assert row["pruned_pct"] == pytest.approx(36.0)
    assert row["improvement"] == pytest.approx(0.05)


def test_rationale_table_is_readable() -> None:
    """The rationale table is a D7 guide artefact, so its shape is tested."""
    spaces, _ = load_search_spaces("loans")
    table = spaces["lightgbm"].rationale_table()
    assert set(table.columns) == {"parameter", "type", "domain", "rationale"}
    assert len(table) == len(spaces["lightgbm"].params)
    assert table["rationale"].str.len().min() > 20


# ==========================================================================
# 6 - Persistence and resumability
# ==========================================================================
@pytest.mark.slow
def test_study_resumes_after_an_interruption(tmp_path: Path) -> None:
    """The Colab-disconnect scenario, simulated.

    A free-tier session dies at ~12 h or ~90 min idle. A 60-trial study that
    restarts from zero is a study that never finishes.
    """
    optuna = pytest.importorskip("optuna")

    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    name = "resume_test"

    def objective(trial):
        x = trial.suggest_float("x", -5.0, 5.0)
        return -(x**2)

    first = optuna.create_study(study_name=name, storage=storage,
                                direction="maximize", load_if_exists=True)
    first.optimize(objective, n_trials=6)
    assert len(first.trials) == 6

    # A fresh handle to the same storage - as a new Colab session would create.
    second = optuna.load_study(study_name=name, storage=storage)
    assert len(second.trials) == 6, "trials did not survive reopening"

    second.optimize(objective, n_trials=4)
    assert len(second.trials) == 10, "the study restarted instead of resuming"


def test_optuna_storage_path_is_configured() -> None:
    cfg = load_config()
    assert cfg.paths.optuna_storage.name.endswith(".db")
    assert "optuna" in str(cfg.paths.optuna_storage)
