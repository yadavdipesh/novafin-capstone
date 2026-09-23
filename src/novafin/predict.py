"""
novafin-capstone/src/novafin/predict.py

Inference - scoring new rows with a saved artefact.

The contract this module enforces
----------------------------------
A :class:`~novafin.utils.io.ModelBundle` carries the model, its preprocessor
AND its feature list together. :func:`predict` re-validates that contract on
every call, so a frame with the right columns in the wrong ORDER, or with a
column silently renamed upstream, fails loudly instead of returning confident
nonsense.

That check is the difference between a demo and something a bank could run. The
classic production failure is not a bad model; it is a good model scored on a
frame whose column 7 changed meaning six months after training.

Business decisions, not just scores
------------------------------------
Each ``score_*`` function returns the DECISION the brief asks for, not a bare
probability: Approve / Review / Reject for credit, an investigate flag at the
cost-optimal threshold for fraud, a retention segment for customers. A
probability is a model output; a decision is a deliverable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from novafin.config import Config, load_config
from novafin.finance.metrics import credit_decision, prioritise_customers, risk_band
from novafin.utils.io import ModelBundle

__all__ = [
    "load_bundle",
    "predict",
    "score_credit_applications",
    "score_transactions",
    "score_customers",
    "available_bundles",
]

LOGGER = logging.getLogger(__name__)


def available_bundles(cfg: Config | None = None) -> list[Path]:
    """Every saved model bundle, newest first."""
    cfg = cfg or load_config()
    return sorted(cfg.paths.artifacts.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_bundle(path: Path | str, cfg: Config | None = None) -> ModelBundle:
    """Load a bundle and warn if the config has moved since training.

    A fingerprint mismatch is a warning rather than an error: the config may
    have changed for reasons unrelated to this model (a new module added, a
    threshold edited elsewhere). But it is surfaced, because a silent mismatch
    is how a stale artefact ends up scoring production traffic.
    """
    cfg = cfg or load_config()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cfg.paths.artifacts / candidate.name

    bundle = ModelBundle.load(candidate)
    if bundle.config_fingerprint and bundle.config_fingerprint != cfg.fingerprint():
        LOGGER.warning(
            "Bundle '%s' was trained under config %s but the current config is %s. "
            "Metrics recorded in the bundle may not correspond to the current settings.",
            candidate.name, bundle.config_fingerprint, cfg.fingerprint(),
        )
    return bundle


def predict(bundle: ModelBundle, frame: pd.DataFrame, *, strict: bool = False) -> np.ndarray:
    """Score a frame, re-validating the feature contract first.

    Args:
        bundle: A loaded bundle.
        frame: Rows to score. May carry extra columns (ids, for instance)
            unless ``strict``.
        strict: Also reject unexpected columns.

    Returns:
        Probabilities for a classifier, point predictions for a regressor.

    Raises:
        ValueError: If a training feature is missing from ``frame``.
    """
    columns = bundle.validate_frame(frame.columns, strict=strict)
    X = frame.loc[:, columns]          # reindexed to the TRAINING order

    if bundle.preprocessor is not None:
        try:
            from sklearn.pipeline import Pipeline

            model = Pipeline([("preprocess", bundle.preprocessor), ("model", bundle.model)])
        except ImportError:  # pragma: no cover
            model = bundle.model
    else:
        model = bundle.model

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X))


# =============================================================================
# Per-module scoring, returning DECISIONS
# =============================================================================
def score_credit_applications(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    *,
    cfg: Config | None = None,
    id_column: str = "Customer_ID",
    exposure_column: str = "Loan_Amount",
    collateral_column: str = "Collateral_Value",
) -> pd.DataFrame:
    """Score loans to PD, ECL and an Approve / Review / Reject recommendation.

    Returns both the flat-LGD ECL (the brief's worked example) and the
    collateral-aware figure, because the Phase-0 audit found 33.3% of loans
    have LTV > 1 and a flat rate understates the tail.
    """
    from novafin.evaluate import expected_credit_loss

    cfg = cfg or load_config()
    lgd = float(cfg.fin("credit", "lgd", default=0.40))

    probability = predict(bundle, frame)
    exposure = frame[exposure_column].to_numpy(dtype="float64")
    collateral = (
        frame[collateral_column].to_numpy(dtype="float64")
        if collateral_column in frame.columns else None
    )

    ecl_flat = expected_credit_loss(probability, exposure, lgd=lgd)
    ecl_secured = (
        expected_credit_loss(probability, exposure, collateral=collateral)
        if collateral is not None else ecl_flat
    )

    decisions = credit_decision(
        probability, ecl_flat, exposure,
        collateral_ratio=(collateral / np.where(exposure == 0, np.nan, exposure))
        if collateral is not None else None,
    )
    if id_column in frame.columns:
        decisions.insert(0, "Loan_ID", frame[id_column].to_numpy())
    decisions["ecl_collateral_aware"] = ecl_secured.to_numpy()
    decisions["lgd_assumption"] = lgd
    return decisions


def score_transactions(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    *,
    cfg: Config | None = None,
    threshold: float | None = None,
    id_column: str = "Transaction_ID",
) -> pd.DataFrame:
    """Score transactions and flag those worth investigating.

    ``threshold`` should be the COST-OPTIMAL value found on out-of-fold
    predictions, not 0.5. When omitted it falls back to the threshold stored in
    the bundle's metadata, and only then to 0.5 - with a warning, because 0.5
    is almost never the right operating point at a 2.28% base rate.
    """
    cfg = cfg or load_config()
    if threshold is None:
        threshold = bundle.metadata.get("optimal_threshold")
    if threshold is None:
        threshold = 0.5
        LOGGER.warning(
            "No cost-optimal threshold supplied or stored; defaulting to 0.5, "
            "which is almost certainly not the cost-minimising operating point."
        )

    probability = predict(bundle, frame)
    cost_fn = float(cfg.fin("fraud", "cost_missed_fraud_inr", default=10000))
    cost_fp = float(cfg.fin("fraud", "cost_false_positive_inr", default=500))

    out = pd.DataFrame(
        {
            "fraud_probability": probability,
            "investigate": probability >= threshold,
            "threshold": threshold,
            "expected_cost_if_ignored": probability * cost_fn,
            "investigation_cost": cost_fp,
        }
    )
    out["net_benefit_of_investigating"] = out["expected_cost_if_ignored"] - cost_fp
    out["priority"] = pd.cut(
        out["fraud_probability"],
        bins=[-0.01, threshold, min(threshold * 2, 0.99), 1.01],
        labels=["monitor", "review", "urgent"],
    ).astype(str)
    if id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return out


def score_customers(
    frame: pd.DataFrame,
    bundle: ModelBundle | None = None,
    *,
    cfg: Config | None = None,
    clv_column: str = "Estimated_CLV",
    id_column: str = "Customer_ID",
) -> pd.DataFrame:
    """Prioritise customers for retention contact.

    ``bundle`` is OPTIONAL and that is deliberate. Finding N-01 established
    that churn has no learnable signal in this dataset (89 positives, maximum
    feature correlation 0.0225). When no bundle is supplied the ranking is
    value-based - CLV, complaints and product depth - which is the defensible
    answer to "which 1,000 customers do we contact?".

    When a bundle IS supplied the churn score is included, and the output
    carries an explicit warning column so the caveat travels with the numbers.
    """
    cfg = cfg or load_config()
    budget = int(cfg.fin("churn", "contact_budget", default=1000))
    clv = frame[clv_column].to_numpy(dtype="float64")

    if bundle is not None:
        churn_probability = predict(bundle, frame)
        caveat = "churn score included - see finding N-01, signal is unreliable"
    else:
        # Value-based proxy: complaints and disengagement, no churn model.
        complaints = frame.get("Complaints", pd.Series(np.zeros(len(frame)))).to_numpy("float64")
        digital = frame.get("Digital_Usage", pd.Series(np.full(len(frame), 50.0))).to_numpy("float64")
        risk = (
            pd.Series(complaints).rank(pct=True) * 0.6
            + (1 - pd.Series(digital).rank(pct=True)) * 0.4
        ).to_numpy()
        churn_probability = risk
        caveat = "VALUE-BASED ranking (no churn model) - the defensible answer per N-01"

    out = prioritise_customers(clv, churn_probability, contact_budget=budget)
    out["basis"] = caveat
    if id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return out.sort_values("expected_value_at_risk", ascending=False).reset_index(drop=True)
