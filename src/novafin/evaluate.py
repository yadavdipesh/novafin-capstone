"""
novafin-capstone/src/novafin/evaluate.py

Metrics, thresholds and statistical tests.

Why the core metrics are implemented here rather than imported
--------------------------------------------------------------
Three reasons, and they are worth defending explicitly:

1. **The cost curve needs the full threshold sweep anyway.** Once you have
   sorted scores and cumulative true/false positive counts, ROC-AUC, PR-AUC,
   KS and the expected-cost curve all fall out of the same two arrays. Calling
   a library for three of them and hand-rolling the fourth is more code, not
   less.
2. **ROC-AUC has an exact closed form.** It is the Mann-Whitney U statistic
   divided by ``n_pos * n_neg`` - the probability that a random positive
   outranks a random negative. That is four lines, exact including ties, and
   it makes the number explainable in a viva rather than delegated.
3. **Metrics are the last thing that should differ between environments.** A
   dependency-free metrics layer runs identically in Colab, in CI and on a
   marker's laptop.

``tests/test_evaluate.py`` cross-checks every metric against scikit-learn when
it is importable, so the equivalence is demonstrated rather than claimed.

What is *not* reimplemented: anything requiring a model. Cross-validation,
estimators and calibration wrappers come from scikit-learn.

Metric choice per module (see the Phase-0 audit for why)
--------------------------------------------------------
=====================  ==================================================
Module                 Primary metric and reason
=====================  ==================================================
M1 initiatives         PR-AUC + Brier. Expected NPV = P(success) x NPV, so
                       *calibration* is the deliverable, not ranking.
M2 credit              KS + Brier. PD feeds ECL, so a well-ranked but
                       badly-calibrated model produces wrong money.
M3 fraud               PR-AUC then expected cost at the optimal threshold.
                       At a 2.28% base rate ROC-AUC flatters.
M4 churn               PR-AUC + permutation p-value. The designed output is
                       a null result with a significance test attached.
M5/M6 equity           Rank IC, not RMSE. Nobody trades a return forecast;
                       they trade its cross-sectional ordering.
M7 liquidity           MAE vs a seasonal-naive baseline.
M9 HFT                 Macro-F1 (the FLAT class is 16%), then net PnL.
M10 derivatives        RMSE against the Black-Scholes benchmark.
=====================  ==================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "ks_statistic",
    "lift_at_k",
    "gains_table",
    "classification_metrics",
    "regression_metrics",
    "multiclass_metrics",
    "calibration_table",
    "expected_calibration_error",
    "cost_curve",
    "optimal_threshold",
    "expected_credit_loss",
    "rank_ic",
    "kupiec_pof_test",
    "population_stability_index",
    "bootstrap_ci",
    "permutation_test",
    "PermutationResult",
    "EvaluationResult",
]

LOGGER = logging.getLogger(__name__)
EPS = 1e-15


# =============================================================================
# Ranking primitives
# =============================================================================
def _as_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays and drop rows where either side is NaN."""
    truth = np.asarray(y_true, dtype="float64").ravel()
    score = np.asarray(y_score, dtype="float64").ravel()
    if truth.shape != score.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {score.shape}")
    keep = ~(np.isnan(truth) | np.isnan(score))
    return truth[keep], score[keep]


def roc_auc(y_true: Any, y_score: Any) -> float:
    """Area under the ROC curve, via the Mann-Whitney U statistic.

    .. math::
        \\mathrm{AUC} = \\frac{U}{n_+ n_-}
        = \\Pr(\\text{score of a random positive} > \\text{score of a random negative})

    Ties are handled correctly because ``scipy``-style average ranks are used;
    a tied pair contributes 0.5. Returns NaN when one class is absent, which is
    the honest answer rather than 0.5.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = truth > 0.5
    n_pos = int(positives.sum())
    n_neg = int(len(truth) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype="float64")
    sorted_scores = score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0     # 1-based, averaged over the tie
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    rank_sum_positive = ranks[positives].sum()
    u_statistic = rank_sum_positive - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def average_precision(y_true: Any, y_score: Any) -> float:
    """Average precision - the step-wise area under the precision-recall curve.

    .. math::
        \\mathrm{AP} = \\sum_n (R_n - R_{n-1}) P_n

    Preferred over the trapezoidal PR-AUC because trapezoidal interpolation
    between PR points is not achievable by any classifier and is optimistic at
    low base rates - which is precisely the regime the fraud (2.28%) and churn
    (1.78%) modules live in.
    """
    truth, score = _as_arrays(y_true, y_score)
    n_pos = int((truth > 0.5).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    labels = truth[order] > 0.5
    true_positives = np.cumsum(labels)
    predicted_positives = np.arange(1, len(labels) + 1)
    precision = true_positives / predicted_positives
    recall = true_positives / n_pos

    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * recall_delta))


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Mean squared error of a probability forecast - a *calibration* metric.

    A model can rank perfectly (AUC 1.0) and still be useless for Expected NPV
    or ECL if its probabilities are systematically too high. Brier catches that
    where AUC cannot.
    """
    truth, prob = _as_arrays(y_true, y_prob)
    return float(np.mean((prob - truth) ** 2))


def log_loss(y_true: Any, y_prob: Any) -> float:
    """Binary cross-entropy, clipped to avoid an infinite penalty on a 0 or 1."""
    truth, prob = _as_arrays(y_true, y_prob)
    prob = np.clip(prob, EPS, 1 - EPS)
    return float(-np.mean(truth * np.log(prob) + (1 - truth) * np.log(1 - prob)))


def ks_statistic(y_true: Any, y_score: Any) -> float:
    """Kolmogorov-Smirnov separation - the standard credit-scoring metric.

    The maximum vertical distance between the cumulative distributions of
    scores for defaulters and non-defaulters. Reported because every credit
    risk team expects it, and because it localises *where* on the score
    distribution the model separates, which AUC averages away.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = truth > 0.5
    if positives.sum() == 0 or (~positives).sum() == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    labels = positives[order]
    cumulative_pos = np.cumsum(labels) / labels.sum()
    cumulative_neg = np.cumsum(~labels) / (~labels).sum()
    return float(np.max(np.abs(cumulative_pos - cumulative_neg)))


def lift_at_k(y_true: Any, y_score: Any, k: float = 0.10) -> float:
    """Event rate in the top ``k`` fraction, divided by the base rate.

    The operational metric for any capacity-constrained action: an
    investigations team that can review 10% of transactions, or relationship
    managers who can call 1,000 customers.
    """
    truth, score = _as_arrays(y_true, y_score)
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1]; got {k}")
    base_rate = truth.mean()
    if base_rate == 0:
        return float("nan")

    n_top = max(1, int(round(len(truth) * k)))
    order = np.argsort(-score, kind="mergesort")[:n_top]
    return float(truth[order].mean() / base_rate)


def gains_table(y_true: Any, y_score: Any, n_bins: int = 10) -> pd.DataFrame:
    """Decile gains table - the artefact a credit or fraud committee reads.

    Columns: ``decile, n, n_events, event_rate, cumulative_events,
    cumulative_capture, lift, cumulative_lift``.
    """
    truth, score = _as_arrays(y_true, y_score)
    order = np.argsort(-score, kind="mergesort")
    truth = truth[order]
    total_events = truth.sum()
    base_rate = truth.mean()

    splits = np.array_split(np.arange(len(truth)), n_bins)
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for index, chunk in enumerate(splits, start=1):
        events = float(truth[chunk].sum())
        cumulative += events
        rows.append(
            {
                "decile": index,
                "n": len(chunk),
                "n_events": int(events),
                "event_rate": events / len(chunk) if len(chunk) else np.nan,
                "cumulative_events": int(cumulative),
                "cumulative_capture": cumulative / total_events if total_events else np.nan,
                "lift": (events / len(chunk)) / base_rate if base_rate else np.nan,
                "cumulative_lift": (
                    (cumulative / (index * len(splits[0]))) / base_rate
                    if base_rate and len(splits[0]) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Metric bundles
# =============================================================================
def classification_metrics(
    y_true: Any, y_prob: Any, *, threshold: float = 0.5, k_values: Sequence[float] = (0.05, 0.10, 0.20)
) -> dict[str, float]:
    """Every binary-classification metric this project reports, in one dict."""
    truth, prob = _as_arrays(y_true, y_prob)
    predicted = (prob >= threshold).astype("float64")

    true_positive = float(((predicted == 1) & (truth == 1)).sum())
    false_positive = float(((predicted == 1) & (truth == 0)).sum())
    false_negative = float(((predicted == 0) & (truth == 1)).sum())
    true_negative = float(((predicted == 0) & (truth == 0)).sum())

    precision = true_positive / (true_positive + false_positive + EPS)
    recall = true_positive / (true_positive + false_negative + EPS)

    metrics = {
        "roc_auc": roc_auc(truth, prob),
        "pr_auc": average_precision(truth, prob),
        "brier": brier_score(truth, prob),
        "log_loss": log_loss(truth, prob),
        "ks": ks_statistic(truth, prob),
        "accuracy": float((predicted == truth).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall + EPS)),
        "specificity": float(true_negative / (true_negative + false_positive + EPS)),
        "base_rate": float(truth.mean()),
        "n": float(len(truth)),
        "n_positive": float(truth.sum()),
        "threshold": float(threshold),
    }
    for k in k_values:
        metrics[f"lift@{int(k * 100)}pct"] = lift_at_k(truth, prob, k)
    return metrics


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """RMSE, MAE, R^2, median absolute error and bias."""
    truth, pred = _as_arrays(y_true, y_pred)
    residual = pred - truth
    total_sum_squares = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "medae": float(np.median(np.abs(residual))),
        "r2": float(1 - np.sum(residual**2) / total_sum_squares) if total_sum_squares > 0 else float("nan"),
        "bias": float(np.mean(residual)),
        "n": float(len(truth)),
    }


def multiclass_metrics(y_true: Any, y_pred: Any, labels: Sequence[Any] | None = None) -> dict[str, float]:
    """Accuracy, balanced accuracy, macro-F1 and per-class recall.

    Macro-F1 is the headline for the order book because the FLAT class is only
    16% of rows: plain accuracy is dominated by UP/DOWN and would hide a model
    that never predicts FLAT at all.
    """
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    classes = list(labels) if labels is not None else sorted(set(truth.tolist()) | set(pred.tolist()))

    metrics: dict[str, float] = {"accuracy": float((truth == pred).mean()), "n": float(len(truth))}
    f1_scores: list[float] = []
    recalls: list[float] = []
    for klass in classes:
        true_positive = float(((pred == klass) & (truth == klass)).sum())
        false_positive = float(((pred == klass) & (truth != klass)).sum())
        false_negative = float(((pred != klass) & (truth == klass)).sum())
        precision = true_positive / (true_positive + false_positive + EPS)
        recall = true_positive / (true_positive + false_negative + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        metrics[f"recall_{klass}"] = float(recall)
        metrics[f"precision_{klass}"] = float(precision)
        metrics[f"f1_{klass}"] = float(f1)
        f1_scores.append(f1)
        recalls.append(recall)

    metrics["macro_f1"] = float(np.mean(f1_scores))
    metrics["balanced_accuracy"] = float(np.mean(recalls))
    return metrics


# =============================================================================
# Calibration
# =============================================================================
def calibration_table(y_true: Any, y_prob: Any, n_bins: int = 10, *, strategy: str = "quantile") -> pd.DataFrame:
    """Predicted vs observed event rate per probability bin.

    ``strategy="quantile"`` is the default because equal-width bins are almost
    empty at the top of the range when the base rate is 2%, producing a
    reliability curve made of noise.
    """
    truth, prob = _as_arrays(y_true, y_prob)
    if strategy == "quantile":
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(prob.min(), prob.max(), n_bins + 1)
    if len(edges) < 2:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "observed_rate", "gap"])

    indices = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    rows: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        mask = indices == index
        if mask.sum() == 0:
            continue
        mean_predicted = float(prob[mask].mean())
        observed = float(truth[mask].mean())
        rows.append(
            {
                "bin": index + 1,
                "n": int(mask.sum()),
                "mean_predicted": mean_predicted,
                "observed_rate": observed,
                "gap": observed - mean_predicted,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Sample-weighted mean absolute gap between predicted and observed rates.

    The number to quote whenever a probability is multiplied by money -
    Expected NPV in M1, ECL in M2.
    """
    table = calibration_table(y_true, y_prob, n_bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


# =============================================================================
# Cost-based decisions
# =============================================================================
def cost_curve(
    y_true: Any,
    y_prob: Any,
    *,
    cost_false_negative: float,
    cost_false_positive: float,
    n_points: int = 501,
) -> pd.DataFrame:
    """Expected operational cost across the full threshold range.

    Implements the brief's fraud economics directly: a missed fraud costs
    ``cost_false_negative``, an unnecessary investigation costs
    ``cost_false_positive``. The decision variable is the threshold, not the
    model - and the optimum is emphatically not 0.5.

    Returns:
        ``threshold, n_flagged, true_positive, false_positive, false_negative,
        precision, recall, total_cost, cost_per_transaction``.
    """
    truth, prob = _as_arrays(y_true, y_prob)
    thresholds = np.linspace(0.0, 1.0, n_points)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        flagged = prob >= threshold
        true_positive = float((flagged & (truth == 1)).sum())
        false_positive = float((flagged & (truth == 0)).sum())
        false_negative = float((~flagged & (truth == 1)).sum())
        total = false_negative * cost_false_negative + false_positive * cost_false_positive
        rows.append(
            {
                "threshold": float(threshold),
                "n_flagged": int(flagged.sum()),
                "true_positive": int(true_positive),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "precision": float(true_positive / (true_positive + false_positive + EPS)),
                "recall": float(true_positive / (true_positive + false_negative + EPS)),
                "total_cost": float(total),
                "cost_per_transaction": float(total / len(truth)),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: Any,
    y_prob: Any,
    *,
    cost_false_negative: float,
    cost_false_positive: float,
    n_points: int = 501,
) -> dict[str, float]:
    """Threshold minimising expected cost, with the do-nothing baselines.

    Both baselines are reported deliberately. "Flag nothing" costs every fraud;
    "flag everything" costs every investigation. A model earns its keep only by
    beating both, and quoting the saving against them is how the result becomes
    a business statement rather than a metric.
    """
    curve = cost_curve(
        y_true, y_prob,
        cost_false_negative=cost_false_negative,
        cost_false_positive=cost_false_positive,
        n_points=n_points,
    )
    best = curve.loc[curve["total_cost"].idxmin()]
    truth, _ = _as_arrays(y_true, y_prob)
    n_positive = float(truth.sum())
    n_total = float(len(truth))

    cost_flag_nothing = n_positive * cost_false_negative
    cost_flag_everything = (n_total - n_positive) * cost_false_positive

    return {
        "threshold": float(best["threshold"]),
        "total_cost": float(best["total_cost"]),
        "cost_per_transaction": float(best["cost_per_transaction"]),
        "n_flagged": float(best["n_flagged"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "cost_flag_nothing": float(cost_flag_nothing),
        "cost_flag_everything": float(cost_flag_everything),
        "saving_vs_flag_nothing": float(cost_flag_nothing - best["total_cost"]),
        "saving_vs_best_baseline": float(
            min(cost_flag_nothing, cost_flag_everything) - best["total_cost"]
        ),
    }


def expected_credit_loss(
    pd_estimate: Any,
    exposure: Any,
    *,
    lgd: float = 0.40,
    collateral: Any | None = None,
) -> pd.Series:
    """ECL = PD x LGD x EAD, with an optional collateral-aware LGD.

    The brief's worked example uses a flat LGD of 40%. The Phase-0 audit found
    the median loan-to-value is 0.797 but **33.3% of loans exceed an LTV of
    1.0**, so a flat rate understates loss on exactly the exposures that matter.
    Passing ``collateral`` switches to
    ``LGD = clip(1 - collateral / exposure, 0, 1)``, which is reported as a
    sensitivity beside the flat case.
    """
    probability = np.asarray(pd_estimate, dtype="float64").ravel()
    ead = np.asarray(exposure, dtype="float64").ravel()

    if collateral is not None:
        secured = np.asarray(collateral, dtype="float64").ravel()
        loss_given_default = np.clip(1.0 - secured / np.where(ead == 0, np.nan, ead), 0.0, 1.0)
    else:
        loss_given_default = np.full_like(probability, lgd)

    return pd.Series(probability * loss_given_default * ead, name="ecl")


# =============================================================================
# Finance-specific
# =============================================================================
def rank_ic(
    y_true: Any, y_pred: Any, groups: Any, *, method: str = "spearman"
) -> dict[str, float]:
    """Cross-sectional information coefficient, computed per group then averaged.

    The correct metric for M5/M6. Nobody trades a return *forecast*; they trade
    its cross-sectional *ordering*, so the quantity that matters is the rank
    correlation between prediction and outcome **within each rebalance date**.

    Pooling all dates into one correlation would be wrong: it mixes
    cross-sectional skill with the time-series level of returns and can look
    strong purely because the market trended.

    Returns:
        ``mean_ic``, ``std_ic``, ``ic_ir`` (mean/std - the information ratio of
        the signal), ``hit_rate`` (share of dates with positive IC), ``n_dates``.
    """
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype="float64").ravel(),
            "p": np.asarray(y_pred, dtype="float64").ravel(),
            "g": np.asarray(groups).ravel(),
        }
    ).dropna()

    coefficients: list[float] = []
    for _, chunk in frame.groupby("g", observed=True):
        if len(chunk) < 3 or chunk["p"].nunique() < 2 or chunk["y"].nunique() < 2:
            continue
        if method == "spearman":
            # Spearman IS Pearson on ranks. Computing it directly avoids a
            # scipy dependency inside a loop that runs once per rebalance date
            # (1,457 times on the equity panel), and is materially faster.
            coefficient = chunk["y"].rank().corr(chunk["p"].rank())
        else:
            coefficient = chunk["y"].corr(chunk["p"])
        coefficients.append(float(coefficient))

    values = np.asarray([c for c in coefficients if np.isfinite(c)], dtype="float64")
    if len(values) == 0:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"),
                "hit_rate": float("nan"), "n_dates": 0.0}

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": float(mean / std) if std and std > 0 else float("nan"),
        "hit_rate": float((values > 0).mean()),
        "n_dates": float(len(values)),
    }


def kupiec_pof_test(exceptions: int, n_observations: int, confidence: float = 0.99) -> dict[str, float]:
    """Kupiec proportion-of-failures test for VaR backtesting.

    A 99% VaR should be breached on about 1% of days. Too few breaches means
    the model is over-conservative and the desk is holding idle capital; too
    many means it is understating risk. The likelihood-ratio statistic is
    chi-squared with one degree of freedom.

    Reference: Kupiec PH (1995), "Techniques for verifying the accuracy of risk
    measurement models", *Journal of Derivatives* 3(2).
    https://doi.org/10.3905/jod.1995.407942
    """
    expected_rate = 1.0 - confidence
    n = int(n_observations)
    x = int(exceptions)
    if n == 0:
        return {"lr_statistic": float("nan"), "p_value": float("nan"),
                "observed_rate": float("nan"), "expected_rate": expected_rate}

    observed_rate = x / n
    if x == 0:
        lr = -2.0 * (n * np.log(1 - expected_rate))
    elif x == n:
        lr = -2.0 * (n * np.log(expected_rate))
    else:
        log_null = x * np.log(expected_rate) + (n - x) * np.log(1 - expected_rate)
        log_alt = x * np.log(observed_rate) + (n - x) * np.log(1 - observed_rate)
        lr = -2.0 * (log_null - log_alt)

    # Survival function of chi-squared with 1 df, without scipy:
    # P(X > x) = erfc(sqrt(x/2)).
    from math import erfc, sqrt

    p_value = float(erfc(sqrt(max(lr, 0.0) / 2.0)))
    return {
        "lr_statistic": float(lr),
        "p_value": p_value,
        "observed_rate": float(observed_rate),
        "expected_rate": float(expected_rate),
        "exceptions": float(x),
        "n_observations": float(n),
    }


def population_stability_index(expected: Any, actual: Any, n_bins: int = 10) -> float:
    """PSI between a reference and a current score distribution.

    The industry drift check: below 0.10 is stable, 0.10-0.25 warrants
    monitoring, above 0.25 means the population has moved and the model should
    be revalidated. Used to compare the fraud training window against the
    untouched 2026-06 to 2026-08 holdout.
    """
    reference = np.asarray(expected, dtype="float64").ravel()
    current = np.asarray(actual, dtype="float64").ravel()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)
    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


# =============================================================================
# Uncertainty and significance
# =============================================================================
def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: Any,
    y_score: Any,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for any metric.

    Every headline number in the report is quoted with this interval. The
    Phase-2 power analysis showed a per-fold AUC standard error of 0.096 on the
    initiatives module and 0.070 on churn - a point estimate without an
    interval would be actively misleading there.
    """
    truth, score = _as_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    n = len(truth)
    samples = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        index = rng.integers(0, n, n)
        try:
            samples[i] = metric(truth[index], score[index])
        except Exception:
            samples[i] = np.nan

    valid = samples[np.isfinite(samples)]
    if len(valid) == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_boot": 0.0}
    return {
        "point": float(metric(truth, score)),
        "lower": float(np.percentile(valid, 100 * alpha / 2)),
        "upper": float(np.percentile(valid, 100 * (1 - alpha / 2))),
        "std": float(valid.std(ddof=1)),
        "n_boot": float(len(valid)),
    }


@dataclass
class PermutationResult:
    """Outcome of a permutation test against the null of no signal."""

    observed: float
    null_scores: np.ndarray
    p_value: float
    n_permutations: int
    metric_name: str = "roc_auc"

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null_scores))

    @property
    def null_p95(self) -> float:
        return float(np.percentile(self.null_scores, 95))

    def summary(self) -> dict[str, float]:
        return {
            "metric": self.metric_name,
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_p95": self.null_p95,
            "p_value": self.p_value,
            "n_permutations": float(self.n_permutations),
            "significant_at_5pct": float(self.p_value < 0.05),
        }

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.p_value < 0.05 else "NOT significant"
        return (
            f"{self.metric_name}: observed {self.observed:.4f}, "
            f"null mean {self.null_mean:.4f} (p95 {self.null_p95:.4f}), "
            f"p = {self.p_value:.4f} -> {verdict} at 5%"
        )


def permutation_test(
    fit_predict: Callable[[np.ndarray], np.ndarray],
    y: Any,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc,
    n_permutations: int = 200,
    seed: int = 42,
    metric_name: str = "roc_auc",
) -> PermutationResult:
    """Test a model's score against the null hypothesis of no signal.

    This is the designed output of the churn module. Phase 0 found 89 positives
    and a maximum feature correlation of 0.0225; Phase 2 found a per-fold AUC
    standard error of 0.070. A cross-validated AUC of, say, 0.56 in that
    setting is **not** evidence of signal - it is well within what a model
    trained on a shuffled target achieves.

    The p-value uses the standard ``(1 + #{null >= observed}) / (1 + B)``
    correction, so it can never be reported as exactly zero - which would be an
    impossible claim from a finite number of permutations.

    Reference: Ojala M, Garriga GC (2010), "Permutation tests for studying
    classifier performance", *JMLR* 11. https://jmlr.org/papers/v11/ojala10a.html

    Args:
        fit_predict: Callable taking a target vector and returning
            out-of-fold predictions for it. Must refit the model internally -
            reusing predictions fitted on the true target would invalidate the
            test.
        y: The true target.
        metric: Scoring function, higher is better.
        n_permutations: Number of shuffles. 200 supports p-values down to ~0.005.
        seed: Seed.
        metric_name: Label for reporting.

    Returns:
        A :class:`PermutationResult`.
    """
    truth = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)

    observed = float(metric(truth, np.asarray(fit_predict(truth)).ravel()))

    null_scores = np.empty(n_permutations, dtype="float64")
    for i in range(n_permutations):
        shuffled = rng.permutation(truth)
        null_scores[i] = float(metric(shuffled, np.asarray(fit_predict(shuffled)).ravel()))
        if (i + 1) % 50 == 0:
            LOGGER.info("permutation %d/%d", i + 1, n_permutations)

    finite = null_scores[np.isfinite(null_scores)]
    p_value = float((1 + np.sum(finite >= observed)) / (1 + len(finite)))
    return PermutationResult(
        observed=observed, null_scores=finite, p_value=p_value,
        n_permutations=len(finite), metric_name=metric_name,
    )


# =============================================================================
# Container
# =============================================================================
@dataclass
class EvaluationResult:
    """All evaluation output for one model on one module."""

    module: str
    model_name: str
    task: str
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metrics: pd.DataFrame | None = None
    calibration: pd.DataFrame | None = None
    gains: pd.DataFrame | None = None
    cost: dict[str, float] | None = None
    permutation: PermutationResult | None = None
    notes: list[str] = field(default_factory=list)

    def headline(self, keys: Sequence[str] | None = None) -> dict[str, float]:
        """The two or three metrics that belong on a leaderboard row."""
        default_keys = {
            "binary_classification": ("roc_auc", "pr_auc", "brier"),
            "multiclass_classification": ("macro_f1", "balanced_accuracy"),
            "regression": ("rmse", "mae", "r2"),
            "panel_regression": ("mean_ic", "ic_ir"),
            "time_series_forecast": ("mae", "rmse"),
        }
        selected = keys or default_keys.get(self.task, tuple(self.metrics)[:3])
        return {k: self.metrics[k] for k in selected if k in self.metrics}

    def to_row(self) -> dict[str, Any]:
        """One flat leaderboard row."""
        row: dict[str, Any] = {"module": self.module, "model": self.model_name, "task": self.task}
        row.update({k: round(v, 6) if isinstance(v, float) else v for k, v in self.metrics.items()})
        if self.fold_metrics is not None and not self.fold_metrics.empty:
            row["n_folds"] = len(self.fold_metrics)
        if self.permutation is not None:
            row["permutation_p"] = round(self.permutation.p_value, 4)
        return row
