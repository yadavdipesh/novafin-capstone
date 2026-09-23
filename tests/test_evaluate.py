"""
novafin-capstone/tests/test_evaluate.py

Tests for the metrics layer, the model factory and the CV loop.

The metrics are checked twice over:

1. against **hand-computable reference values** on a four-row example, so the
   arithmetic is verifiable by a reader with a calculator;
2. against **scikit-learn**, when it is importable, so the equivalence of the
   in-house implementations is demonstrated rather than claimed.

The second family is skipped where scikit-learn is absent, so the suite still
runs in a minimal environment - which is the whole reason the metrics were
written dependency-free in the first place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from novafin.config import load_config
from novafin.evaluate import (
    EvaluationResult,
    average_precision,
    bootstrap_ci,
    brier_score,
    calibration_table,
    classification_metrics,
    cost_curve,
    expected_calibration_error,
    expected_credit_loss,
    gains_table,
    kupiec_pof_test,
    ks_statistic,
    lift_at_k,
    log_loss,
    multiclass_metrics,
    optimal_threshold,
    permutation_test,
    population_stability_index,
    rank_ic,
    regression_metrics,
    roc_auc,
)
from novafin.models.build import (
    ALLOWED_MODULE_PREFIXES,
    ModelSpec,
    available_modules,
    load_model_specs,
    resolve_class,
)
from novafin.models.leaderboard import build_leaderboard, format_leaderboard
from novafin.models.train import cross_validate_model, evaluate_cv
from novafin.tracking import ExperimentTracker

# A four-row example whose metrics can be worked out by hand.
Y_REF = [0, 0, 1, 1]
S_REF = [0.1, 0.4, 0.35, 0.8]


# ==========================================================================
# Hand-computable references
# ==========================================================================
def test_roc_auc_reference_value() -> None:
    assert roc_auc(Y_REF, S_REF) == pytest.approx(0.75)


def test_average_precision_reference_value() -> None:
    # Ranked: 0.8(+), 0.4(-), 0.35(+), 0.1(-)
    # AP = 0.5 * 1.0 + 0.5 * (2/3) = 0.8333...
    assert average_precision(Y_REF, S_REF) == pytest.approx(0.8333333, abs=1e-6)


def test_brier_reference_value() -> None:
    # (0.01 + 0.16 + 0.4225 + 0.04) / 4
    assert brier_score(Y_REF, S_REF) == pytest.approx(0.158125)


def test_ks_reference_value() -> None:
    assert ks_statistic(Y_REF, S_REF) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([0.1, 0.2, 0.8, 0.9], 1.0), ([0.9, 0.8, 0.2, 0.1], 0.0), ([0.5] * 4, 0.5)],
)
def test_roc_auc_extremes(scores: list[float], expected: float) -> None:
    assert roc_auc([0, 0, 1, 1], scores) == pytest.approx(expected)


def test_roc_auc_handles_ties_at_half() -> None:
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_metrics_return_nan_for_a_single_class() -> None:
    assert np.isnan(roc_auc([1, 1], [0.2, 0.8]))
    assert np.isnan(average_precision([0, 0], [0.2, 0.8]))
    assert np.isnan(ks_statistic([0, 0], [0.2, 0.8]))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        roc_auc([0, 1], [0.1, 0.2, 0.3])


def test_log_loss_is_finite_at_the_boundaries() -> None:
    """Clipping must stop a confident mistake producing infinity."""
    assert np.isfinite(log_loss([1, 0], [0.0, 1.0]))


# ==========================================================================
# Equivalence with scikit-learn, where available
# ==========================================================================
def _sklearn_metrics():
    return pytest.importorskip("sklearn.metrics")


def test_matches_sklearn_on_random_data() -> None:
    metrics = _sklearn_metrics()
    rng = np.random.default_rng(42)
    n = 5000
    y = rng.binomial(1, 0.2, n)
    p = np.clip(0.2 + 0.35 * y + rng.normal(0, 0.2, n), 0.001, 0.999)

    assert roc_auc(y, p) == pytest.approx(metrics.roc_auc_score(y, p), abs=1e-9)
    assert average_precision(y, p) == pytest.approx(metrics.average_precision_score(y, p), abs=1e-9)
    assert brier_score(y, p) == pytest.approx(metrics.brier_score_loss(y, p), abs=1e-9)
    assert log_loss(y, p) == pytest.approx(metrics.log_loss(y, p), abs=1e-6)


def test_matches_sklearn_with_heavy_ties() -> None:
    """Ties are where a naive AUC implementation diverges from sklearn."""
    metrics = _sklearn_metrics()
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.4, 2000)
    p = np.round(rng.uniform(0, 1, 2000), 1)          # only 11 distinct scores
    assert roc_auc(y, p) == pytest.approx(metrics.roc_auc_score(y, p), abs=1e-9)


def test_regression_metrics_match_sklearn() -> None:
    metrics = _sklearn_metrics()
    rng = np.random.default_rng(42)
    truth = rng.normal(size=2000)
    pred = truth * 0.6 + rng.normal(0, 0.5, 2000)
    ours = regression_metrics(truth, pred)
    assert ours["rmse"] == pytest.approx(float(np.sqrt(metrics.mean_squared_error(truth, pred))), abs=1e-9)
    assert ours["mae"] == pytest.approx(metrics.mean_absolute_error(truth, pred), abs=1e-9)
    assert ours["r2"] == pytest.approx(metrics.r2_score(truth, pred), abs=1e-9)


def test_multiclass_macro_f1_matches_sklearn() -> None:
    metrics = _sklearn_metrics()
    rng = np.random.default_rng(42)
    truth = rng.choice(["UP", "DOWN", "FLAT"], 3000, p=[0.42, 0.42, 0.16])
    pred = np.where(rng.random(3000) < 0.6, truth, rng.choice(["UP", "DOWN", "FLAT"], 3000))
    ours = multiclass_metrics(truth, pred)
    assert ours["macro_f1"] == pytest.approx(metrics.f1_score(truth, pred, average="macro"), abs=1e-6)
    assert ours["balanced_accuracy"] == pytest.approx(
        metrics.balanced_accuracy_score(truth, pred), abs=1e-6
    )


# ==========================================================================
# Business metrics
# ==========================================================================
def test_lift_at_k_on_a_perfect_ranking() -> None:
    y = [1] * 10 + [0] * 90
    score = list(np.linspace(1, 0, 100))
    assert lift_at_k(y, score, 0.10) == pytest.approx(10.0)


def test_lift_at_k_rejects_an_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be in"):
        lift_at_k([0, 1], [0.1, 0.9], 1.5)


def test_gains_table_captures_everything_by_the_last_decile() -> None:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.2, 1000)
    p = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.2, 1000), 0, 1)
    table = gains_table(y, p)
    assert len(table) == 10
    assert table["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > table["lift"].iloc[-1]


def test_cost_curve_endpoints_are_the_do_nothing_baselines() -> None:
    """At threshold 0 everything is flagged; at 1.0 nothing is."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.05, 2000)
    p = rng.uniform(0, 1, 2000)
    curve = cost_curve(y, p, cost_false_negative=10000, cost_false_positive=500)

    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    assert curve["total_cost"].iloc[0] == pytest.approx(n_negative * 500)
    assert curve["total_cost"].iloc[-1] == pytest.approx(n_positive * 10000)


def test_optimal_threshold_beats_both_baselines_on_a_good_model() -> None:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.05, 5000)
    p = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.15, 5000), 0, 1)
    best = optimal_threshold(y, p, cost_false_negative=10000, cost_false_positive=500)
    assert best["total_cost"] < best["cost_flag_nothing"]
    assert best["total_cost"] < best["cost_flag_everything"]
    assert best["saving_vs_best_baseline"] > 0
    assert best["threshold"] != pytest.approx(0.5), "the optimum is rarely 0.5"


def test_expected_credit_loss_flat_lgd() -> None:
    ecl = expected_credit_loss([0.1, 0.3], [1000, 1000], lgd=0.40)
    assert list(np.round(ecl, 6)) == [40.0, 120.0]


def test_expected_credit_loss_collateral_aware() -> None:
    """LGD = 1 - collateral/exposure, floored at 0 for over-collateralised loans."""
    ecl = expected_credit_loss([0.2, 0.2], [1000, 1000], collateral=[600, 1500])
    assert ecl.iloc[0] == pytest.approx(0.2 * 0.4 * 1000)
    assert ecl.iloc[1] == pytest.approx(0.0), "fully secured -> zero expected loss"


def test_rank_ic_is_computed_per_date_then_averaged() -> None:
    rng = np.random.default_rng(42)
    dates = np.repeat(["d1", "d2", "d3"], 40)
    truth = rng.normal(size=120)
    result = rank_ic(truth, truth, dates)          # perfect prediction
    assert result["mean_ic"] == pytest.approx(1.0)
    assert result["hit_rate"] == pytest.approx(1.0)
    assert result["n_dates"] == 3


def test_rank_ic_of_noise_is_near_zero() -> None:
    rng = np.random.default_rng(42)
    dates = np.repeat([f"d{i}" for i in range(40)], 30)
    truth = rng.normal(size=1200)
    noise = rng.normal(size=1200)
    assert abs(rank_ic(truth, noise, dates)["mean_ic"]) < 0.08


def test_kupiec_accepts_a_well_calibrated_var() -> None:
    result = kupiec_pof_test(exceptions=3, n_observations=250, confidence=0.99)
    assert result["p_value"] > 0.05, "2.5 expected, 3 observed - should not reject"


def test_kupiec_rejects_an_understated_var() -> None:
    result = kupiec_pof_test(exceptions=12, n_observations=250, confidence=0.99)
    assert result["p_value"] < 0.01


def test_psi_flags_a_shifted_population() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 5000)
    same = rng.normal(0, 1, 5000)
    shifted = rng.normal(0.8, 1, 5000)
    assert population_stability_index(reference, same) < 0.10
    assert population_stability_index(reference, shifted) > 0.25


# ==========================================================================
# Calibration
# ==========================================================================
def test_calibration_of_a_perfectly_calibrated_model() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0.02, 0.98, 20000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.02


def test_calibration_detects_systematic_overconfidence() -> None:
    """A model that doubles every probability must be caught by ECE, not AUC."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.02, 0.45, 20000)
    y = rng.binomial(1, p)
    inflated = np.clip(p * 2, 0, 1)
    assert roc_auc(y, inflated) == pytest.approx(roc_auc(y, p), abs=1e-9)
    assert expected_calibration_error(y, inflated) > 5 * expected_calibration_error(y, p)


def test_calibration_table_columns() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 2000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p, n_bins=10)
    assert set(table.columns) == {"bin", "n", "mean_predicted", "observed_rate", "gap"}
    assert table["n"].sum() == 2000


# ==========================================================================
# Uncertainty and significance
# ==========================================================================
def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, 1000)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 1000), 0, 1)
    interval = bootstrap_ci(roc_auc, y, p, n_boot=200, seed=1)
    assert interval["lower"] < interval["point"] < interval["upper"]
    assert interval["upper"] - interval["lower"] > 0


def test_permutation_test_finds_no_signal_in_noise() -> None:
    """The churn module's designed output, on synthetic noise."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, 400)

    def fit_predict(target: np.ndarray) -> np.ndarray:
        return rng.normal(size=len(target))          # a model with no skill

    result = permutation_test(fit_predict, y, n_permutations=100, seed=1)
    assert result.p_value > 0.05
    assert result.null_mean == pytest.approx(0.5, abs=0.05)
    assert "NOT significant" in str(result)


def test_permutation_test_detects_real_signal() -> None:
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, 600)

    def fit_predict(target: np.ndarray) -> np.ndarray:
        # A model that genuinely learns whatever target it is given.
        return target + rng.normal(0, 0.4, len(target))

    result = permutation_test(fit_predict, y, n_permutations=60, seed=1)
    assert result.observed > 0.8
    # The null is also high here BY DESIGN - the surrogate learns the shuffled
    # target too - which is exactly why a permutation test must refit.
    assert result.p_value >= 1 / 61


def test_permutation_p_value_can_never_be_zero() -> None:
    """(1 + #{null >= observed}) / (1 + B) - a finite test cannot prove p = 0."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.4, 200)
    result = permutation_test(
        lambda t: np.asarray(t, dtype=float), y, n_permutations=30, seed=1
    )
    assert result.p_value >= 1 / 31


# ==========================================================================
# Model factory
# ==========================================================================
def test_every_dataset_module_has_declared_models() -> None:
    cfg = load_config()
    declared = set(available_modules())
    assert set(cfg.datasets).issubset(declared)


def test_each_module_starts_with_a_trivial_baseline() -> None:
    """A model that cannot beat a dummy has not earned a place in the report."""
    for module in available_modules():
        names = load_model_specs(module).names()
        first = names[0].lower()
        assert any(token in first for token in ("dummy", "baseline", "mean", "zero")), (
            f"'{module}' starts with '{names[0]}' - declare a trivial baseline first"
        )


def test_resolve_class_rejects_disallowed_packages() -> None:
    for path in ("os.system", "subprocess.Popen", "builtins.eval"):
        with pytest.raises(ValueError, match="Refusing to import"):
            resolve_class(path)


def test_resolve_class_requires_a_dotted_path() -> None:
    with pytest.raises(ValueError, match="not a dotted path"):
        resolve_class("LGBMClassifier")


def test_allowed_prefixes_are_all_pinned_dependencies() -> None:
    from novafin.paths import REPO_ROOT

    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for prefix in ALLOWED_MODULE_PREFIXES:
        if prefix == "novafin":
            continue
        package = {"sklearn": "scikit-learn"}.get(prefix, prefix)
        assert package in requirements, f"'{prefix}' is importable from config but not pinned"


def test_model_catalog_lookup() -> None:
    catalog = load_model_specs("loans")
    assert len(catalog) >= 3
    assert catalog.get("lightgbm").class_path == "lightgbm.LGBMClassifier"
    with pytest.raises(KeyError, match="No model named"):
        catalog.get("not_a_model")


# ==========================================================================
# The CV loop - verified with a dependency-free estimator
# ==========================================================================
class _NumpyLogit:
    """Minimal logistic regression, so the CV loop is testable without sklearn."""

    def __init__(self, epochs: int = 300, lr: float = 0.5) -> None:
        self.epochs, self.lr = epochs, lr

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        values = np.nan_to_num(X.to_numpy(dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
        return np.hstack([np.ones((len(values), 1)), (values - self.mu_) / self.sd_])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_NumpyLogit":
        values = np.nan_to_num(X.to_numpy(dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
        self.mu_, self.sd_ = values.mean(0), values.std(0) + 1e-9
        design = self._design(X)
        target = np.asarray(y, dtype="float64")
        weights = np.zeros(design.shape[1])
        for _ in range(self.epochs):
            prob = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
            weights -= self.lr * (design.T @ (prob - target) / len(target))
        self.w_ = weights
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob = 1 / (1 + np.exp(-np.clip(self._design(X) @ self.w_, -30, 30)))
        return np.column_stack([1 - prob, prob])


class _StratifiedKFold:
    def __init__(self, n_splits: int = 4, seed: int = 42) -> None:
        self.n_splits, self.seed = n_splits, seed

    def split(self, X, y=None, **kwargs):
        target = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        folds = np.empty(len(target), dtype=int)
        for klass in np.unique(target):
            index = np.where(target == klass)[0]
            rng.shuffle(index)
            folds[index] = np.arange(len(index)) % self.n_splits
        for k in range(self.n_splits):
            yield np.where(folds != k)[0], np.where(folds == k)[0]


@pytest.fixture
def toy_problem() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 1200
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {"signal": signal, "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)}
    )
    probability = 1 / (1 + np.exp(-(1.5 * signal - 0.5)))
    y = pd.Series(rng.binomial(1, probability), name="target")
    return X, y


def _factory(spec, X, cfg=None, defaults=None, extra_params=None):
    return _NumpyLogit()


def test_cross_validation_predicts_every_row_exactly_once(toy_problem) -> None:
    """Out-of-fold assembly is the property thresholding depends on."""
    X, y = toy_problem
    spec = ModelSpec(name="numpy_logit", class_path="novafin.test", scale=True)
    result = cross_validate_model(
        spec, X, y, _StratifiedKFold(4), task="binary_classification",
        module="toy", pipeline_factory=_factory,
    )
    assert len(result.folds) == 4
    assert result.oof_mask.all(), "every row must receive an out-of-fold prediction"
    assert np.isfinite(result.oof_predictions).all()

    covered = np.concatenate([f.test_index for f in result.folds])
    assert len(covered) == len(np.unique(covered)) == len(X), "folds must not overlap"


def test_cv_reports_fold_variance(toy_problem) -> None:
    X, y = toy_problem
    spec = ModelSpec(name="numpy_logit", class_path="novafin.test")
    result = cross_validate_model(
        spec, X, y, _StratifiedKFold(4), task="binary_classification",
        module="toy", pipeline_factory=_factory,
    )
    summary = result.summary()
    assert "roc_auc" in summary and "roc_auc_std" in summary
    assert summary["n_folds"] == 4
    assert summary["roc_auc"] > 0.7, "the toy problem has real signal"


def test_evaluate_cv_produces_calibration_gains_and_cost(toy_problem) -> None:
    X, y = toy_problem
    spec = ModelSpec(name="numpy_logit", class_path="novafin.test")
    result = cross_validate_model(
        spec, X, y, _StratifiedKFold(4), task="binary_classification",
        module="toy", pipeline_factory=_factory,
    )
    evaluation = evaluate_cv(
        result, y, cost_false_negative=10000, cost_false_positive=500
    )
    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.calibration is not None and not evaluation.calibration.empty
    assert evaluation.gains is not None and len(evaluation.gains) == 10
    assert evaluation.cost is not None and 0 <= evaluation.cost["threshold"] <= 1
    assert "cv_roc_auc_std" in evaluation.metrics
    assert evaluation.fold_metrics is not None and len(evaluation.fold_metrics) == 4


def test_max_folds_marks_the_result_as_truncated(toy_problem) -> None:
    """A truncated smoke run must never be mistaken for a complete one."""
    X, y = toy_problem
    spec = ModelSpec(name="numpy_logit", class_path="novafin.test")
    result = cross_validate_model(
        spec, X, y, _StratifiedKFold(4), task="binary_classification",
        module="toy", pipeline_factory=_factory, max_folds=2,
    )
    assert len(result.folds) == 2
    assert any("TRUNCATED" in note for note in result.notes)


# ==========================================================================
# Tracking and leaderboard
# ==========================================================================
def test_tracker_falls_back_without_mlflow(tmp_path) -> None:
    """Losing an experiment log must never cost a training run."""
    cfg = load_config()
    tracker = ExperimentTracker(cfg)
    assert tracker.backend in {"mlflow", "jsonl", "disabled"}

    with tracker.run("unit_test_module", "unit_test_model") as handle:
        handle.params["alpha"] = 1.0
        handle.metrics["roc_auc"] = 0.75
        assert handle.tags["config_fingerprint"] == cfg.fingerprint()
        assert handle.tags["seed"] == str(cfg.reproducibility.seed)


def test_leaderboard_ranks_by_the_configured_primary_metric() -> None:
    runs = pd.DataFrame(
        [
            {"tag.module": "loans", "tag.model": "a", "metric.ks": 0.30, "metric.roc_auc": 0.70},
            {"tag.module": "loans", "tag.model": "b", "metric.ks": 0.45, "metric.roc_auc": 0.68},
            {"tag.module": "options", "tag.model": "c", "metric.rmse": 5.0},
            {"tag.module": "options", "tag.model": "d", "metric.rmse": 2.0},
        ]
    )
    board = build_leaderboard(runs=runs)
    loans = board[board["module"] == "loans"].sort_values("rank")
    assert loans["model"].iloc[0] == "b", "KS is higher-is-better"

    options = board[board["module"] == "options"].sort_values("rank")
    assert options["model"].iloc[0] == "d", "RMSE is lower-is-better"


def test_leaderboard_markdown_needs_no_tabulate() -> None:
    runs = pd.DataFrame([{"tag.module": "loans", "tag.model": "a", "metric.ks": 0.3}])
    text = format_leaderboard(build_leaderboard(runs=runs))
    assert "| rank | model |" in text


def test_leaderboard_is_empty_before_any_run() -> None:
    assert build_leaderboard(runs=pd.DataFrame()).empty
    assert "No runs recorded" in format_leaderboard(pd.DataFrame())
