"""
novafin-capstone/src/novafin/models/campaign.py

D4 Level 4 - the EXTRA FINE-TUNING hook.

The contract this module exists to satisfy
-------------------------------------------
From the brief:

    "an 'EXTRA FINE-TUNING' hook - a documented, config-driven entry point so I
    can launch new tuning campaigns (new search space, more trials,
    ensembling/stacking, threshold optimisation) by editing YAML only, with
    zero code changes."

So: **every campaign is a YAML file in ``configs/campaigns/``**, and launching
one is

.. code-block:: bash

    make campaign CAMPAIGN=configs/campaigns/my_new_campaign.yaml

or, in a notebook, ``run_campaign("configs/campaigns/my_new_campaign.yaml")``.
Nothing in ``src/`` changes.

Why this is achievable at all
------------------------------
It only works because the earlier phases never named a model class or a
parameter in code:

* Phase 4 made estimators a dotted path in ``configs/models.yaml``, resolved by
  ``importlib`` (``models/build.py``);
* Phase 5 made search spaces a YAML declaration with a resolved-at-runtime
  study (``models/tune.py``), and ``Config.extra`` preserves unknown top-level
  blocks so a campaign can add sections;
* Phase 4 also exposed ``pipeline_factory`` and ``fold_callback`` on the CV
  loop, which is what lets a campaign substitute a stacked or calibrated
  pipeline without touching the loop.

Level 4 is the payoff for those decisions rather than a new mechanism.

The four campaign operations
-----------------------------
=====================  =====================================================
``tune``               A new/extended Optuna study - new space, more trials.
``stack``              Train a meta-learner over several base models'
                       out-of-fold predictions.
``threshold``          Re-optimise the decision threshold against a cost
                       matrix, without retraining anything.
``finetune``           A Level-3 staged-boosting schedule.
=====================  =====================================================

Each is declared per module in the campaign file; a campaign may mix them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import yaml

from novafin.config import Config, load_config
from novafin.evaluate import optimal_threshold
from novafin.models.build import ModelSpec, load_model_specs
from novafin.models.train import cross_validate_model, evaluate_cv
from novafin.paths import REPO_ROOT, resolve

__all__ = [
    "CampaignStep",
    "CampaignSpec",
    "CampaignResult",
    "load_campaign",
    "validate_campaign",
    "run_campaign",
    "list_campaigns",
    "SUPPORTED_OPERATIONS",
]

LOGGER = logging.getLogger(__name__)

CAMPAIGN_DIR = REPO_ROOT / "configs" / "campaigns"

SUPPORTED_OPERATIONS = ("tune", "stack", "threshold", "finetune")


# =============================================================================
# Declarations
# =============================================================================
@dataclass(frozen=True)
class CampaignStep:
    """One operation on one module within a campaign."""

    module: str
    operation: str
    models: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "operation": self.operation,
            "models": ", ".join(self.models) or "(all declared)",
            "note": " ".join(self.note.split())[:120],
        }


@dataclass(frozen=True)
class CampaignSpec:
    """A whole campaign, as loaded from YAML."""

    name: str
    description: str = ""
    author: str = ""
    steps: tuple[CampaignStep, ...] = ()
    source_path: Path | None = None

    def enabled_steps(self) -> list[CampaignStep]:
        return [step for step in self.steps if step.enabled]

    def plan(self) -> pd.DataFrame:
        """The campaign plan, as a table - printed before anything runs."""
        return pd.DataFrame([step.describe() for step in self.enabled_steps()])


@dataclass
class CampaignResult:
    """What a campaign produced."""

    name: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    artefacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def ok(self) -> bool:
        return not self.errors


# =============================================================================
# Loading and validation
# =============================================================================
def list_campaigns(directory: Path | str | None = None) -> list[Path]:
    """Every campaign file available to run.

    Files prefixed ``_`` or ``spaces_`` are excluded: a campaign may ship its
    own search-space YAML in the same folder, and offering that as a runnable
    campaign would be confusing (it has no ``steps`` and would fail validation).
    """
    target = Path(directory) if directory else CAMPAIGN_DIR
    if not target.exists():
        return []
    return sorted(
        p for p in target.glob("*.yaml")
        if not p.name.startswith(("_", "spaces_"))
    )


def load_campaign(path: Path | str) -> CampaignSpec:
    """Read a campaign YAML into a :class:`CampaignSpec`.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the campaign is malformed - with every problem listed at
            once, so a user fixes the file in one pass rather than four.
    """
    campaign_path = resolve(path)
    if not campaign_path.exists():
        available = [p.name for p in list_campaigns()]
        raise FileNotFoundError(
            f"Campaign not found: {campaign_path}. Available: {available}"
        )

    raw = yaml.safe_load(campaign_path.read_text(encoding="utf-8")) or {}
    steps: list[CampaignStep] = []
    for entry in raw.get("steps", []) or []:
        steps.append(
            CampaignStep(
                module=str(entry.get("module", "")),
                operation=str(entry.get("operation", "")),
                models=tuple(entry.get("models", []) or []),
                params=dict(entry.get("params", {}) or {}),
                enabled=bool(entry.get("enabled", True)),
                note=str(entry.get("note", "")),
            )
        )

    spec = CampaignSpec(
        name=str(raw.get("name", campaign_path.stem)),
        description=str(raw.get("description", "")),
        author=str(raw.get("author", "")),
        steps=tuple(steps),
        source_path=campaign_path,
    )

    problems = validate_campaign(spec)
    if problems:
        raise ValueError(
            f"Campaign '{spec.name}' is invalid:\n  " + "\n  ".join(problems)
        )
    return spec


def validate_campaign(spec: CampaignSpec, cfg: Config | None = None) -> list[str]:
    """Check a campaign before running a single model.

    Validating up front matters more here than elsewhere: a campaign is
    unattended and may run for an hour, so a typo in step 4 must be caught
    before step 1 starts rather than after.

    Returns:
        A list of problems. Empty means valid.
    """
    cfg = cfg or load_config()
    problems: list[str] = []

    if not spec.steps:
        problems.append("campaign declares no steps")

    known_modules = set(cfg.datasets) | {"options_residual", "customers_clv"}
    for index, step in enumerate(spec.steps, start=1):
        label = f"step {index} ({step.module}/{step.operation})"

        if step.operation not in SUPPORTED_OPERATIONS:
            problems.append(
                f"{label}: unknown operation {step.operation!r}; "
                f"supported: {SUPPORTED_OPERATIONS}"
            )
        if step.module not in known_modules:
            problems.append(f"{label}: unknown module {step.module!r}")

        if step.operation == "stack":
            if len(step.models) < 2:
                problems.append(f"{label}: stacking needs at least two base models")
            if not step.params.get("meta_model"):
                problems.append(f"{label}: stacking needs params.meta_model")
        if step.operation == "threshold":
            for required in ("cost_false_negative", "cost_false_positive"):
                if required not in step.params:
                    problems.append(f"{label}: threshold needs params.{required}")
        if step.operation == "finetune" and not step.params.get("schedule"):
            problems.append(f"{label}: finetune needs params.schedule")

    return problems


# =============================================================================
# Operations
# =============================================================================
def _op_tune(step: CampaignStep, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run (or extend) an Optuna study with campaign overrides."""
    from novafin.models.tune import tune_model

    cfg: Config = context["cfg"]
    X, y = context["X"], context["y"]
    rows: list[dict[str, Any]] = []

    models = step.models or (context["catalog"].names()[1:2])
    for model_name in models:
        result = tune_model(
            step.module, model_name, X, y, context["splitter"],
            task=context["task"], cfg=cfg,
            split_kwargs=context.get("split_kwargs"),
            ic_groups=context.get("ic_groups"),
            baseline_value=context.get("baselines", {}).get(model_name, float("nan")),
            n_trials=step.params.get("n_trials"),
            timeout=step.params.get("timeout_seconds"),
            # A campaign may supply its OWN search-space file - this is what
            # "new search space, zero code changes" actually means.
            spaces_path=step.params.get("search_spaces"),
            # ...and its own study suffix, so a campaign never collides with
            # the Level-2 study of the same model.
            study_suffix=step.params.get("study_suffix", context["campaign_name"]),
            pipeline_factory=context.get("pipeline_factory"),
        )
        rows.append({"operation": "tune", **result.summary()})
    return rows


def _op_threshold(step: CampaignStep, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-optimise the decision threshold against a cost matrix.

    Retrains nothing. The threshold is a *decision* parameter, not a model
    parameter, so sweeping the cost assumptions is cheap - which is exactly why
    it belongs in a campaign: an examiner asking "what if a missed fraud costs
    25,000 instead?" should be answered by editing YAML, not by retraining.
    """
    cfg: Config = context["cfg"]
    X, y = context["X"], context["y"]
    catalog = context["catalog"]
    rows: list[dict[str, Any]] = []

    cost_fn = float(step.params["cost_false_negative"])
    cost_fp = float(step.params["cost_false_positive"])
    models = step.models or catalog.names()[1:2]

    for model_name in models:
        spec = catalog.get(model_name)
        overrides = step.params.get("params", {}).get(model_name, {})
        if overrides:
            spec = ModelSpec(
                name=spec.name, class_path=spec.class_path,
                params={**spec.params, **overrides}, scale=spec.scale,
            )
        cv = cross_validate_model(
            spec, X, y, context["splitter"], task=context["task"],
            module=step.module, cfg=cfg,
            split_kwargs=context.get("split_kwargs"),
            defaults=catalog.defaults,
            pipeline_factory=context.get("pipeline_factory"),
        )
        mask = cv.oof_mask
        best = optimal_threshold(
            np.asarray(y)[mask], np.asarray(cv.oof_predictions)[mask],
            cost_false_negative=cost_fn, cost_false_positive=cost_fp,
        )
        rows.append(
            {
                "operation": "threshold", "module": step.module, "model": model_name,
                "cost_false_negative": cost_fn, "cost_false_positive": cost_fp,
                **{k: round(v, 4) for k, v in best.items()},
            }
        )
    return rows


def _op_stack(step: CampaignStep, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Train a meta-learner on the base models' out-of-fold predictions.

    The only correct way to stack: the meta-learner is fitted on **out-of-fold**
    predictions, because in-sample predictions from a fitted base model are
    over-confident and the meta-learner would simply learn to trust whichever
    base model overfits hardest.

    Reference: Wolpert DH (1992), "Stacked generalization", *Neural Networks*
    5(2). https://doi.org/10.1016/S0893-6080(05)80023-1
    """
    cfg: Config = context["cfg"]
    X, y = context["X"], context["y"]
    catalog = context["catalog"]

    oof_columns: dict[str, np.ndarray] = {}
    coverage: np.ndarray | None = None

    for model_name in step.models:
        spec = catalog.get(model_name)
        cv = cross_validate_model(
            spec, X, y, context["splitter"], task=context["task"],
            module=step.module, cfg=cfg,
            split_kwargs=context.get("split_kwargs"),
            defaults=catalog.defaults,
            pipeline_factory=context.get("pipeline_factory"),
        )
        oof_columns[model_name] = np.asarray(cv.oof_predictions, dtype="float64")
        coverage = cv.oof_mask if coverage is None else (coverage & cv.oof_mask)

    meta_X = pd.DataFrame(oof_columns)[coverage]
    meta_y = pd.Series(np.asarray(y)[coverage], name=str(y.name))

    meta_spec = ModelSpec(
        name=f"stack_{step.params['meta_model'].split('.')[-1]}",
        class_path=step.params["meta_model"],
        params=dict(step.params.get("meta_params", {})),
        scale=bool(step.params.get("meta_scale", True)),
    )
    meta_cv = cross_validate_model(
        meta_spec, meta_X, meta_y, context["stack_splitter"],
        task=context["task"], module=f"{step.module}_stack", cfg=cfg,
        pipeline_factory=context.get("pipeline_factory"),
    )
    evaluation = evaluate_cv(meta_cv, meta_y, cfg=cfg)

    rows = [
        {
            "operation": "stack", "module": step.module,
            "model": f"stack({'+'.join(step.models)})",
            "n_base_models": len(step.models),
            **{k: round(v, 5) for k, v in evaluation.headline().items()},
        }
    ]
    # Base-model scores alongside, so the stack is judged against what it stacks.
    for model_name, predictions in oof_columns.items():
        base_metrics = evaluate_cv(
            type(meta_cv)(
                module=step.module, model_name=model_name, task=context["task"],
                folds=meta_cv.folds, oof_predictions=predictions, oof_mask=coverage,
            ),
            pd.Series(np.asarray(y), name=str(y.name)), cfg=cfg,
        )
        rows.append(
            {
                "operation": "stack_base", "module": step.module, "model": model_name,
                **{k: round(v, 5) for k, v in base_metrics.headline().items()},
            }
        )
    return rows


def _op_finetune(step: CampaignStep, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a Level-3 staged-boosting schedule declared in the campaign."""
    from novafin.evaluate import average_precision, ks_statistic, roc_auc
    from novafin.models.finetune import staged_boosting

    cfg: Config = context["cfg"]
    scorers: dict[str, Callable[..., float]] = {
        "roc_auc": roc_auc, "pr_auc": average_precision, "ks": ks_statistic,
    }
    objective = str(step.params.get("objective", "roc_auc"))
    scorer = scorers.get(objective, roc_auc)

    result = staged_boosting(
        context["X"], context["y"], context["splitter"],
        schedule=step.params["schedule"],
        base_params=step.params.get("base_params"),
        scorer=scorer, module=step.module, cfg=cfg,
        split_kwargs=context.get("split_kwargs"),
        objective_name=objective, task=context["task"],
        patience=int(step.params.get("patience", 2)),
    )
    rows = [
        {
            "operation": "finetune", "module": step.module,
            "model": result.model_name, "objective": objective,
            "stage_1_score": result.baseline_score, "best_score": result.best_score,
            "improvement": result.improvement, "best_stage": result.best_stage,
            "n_stages_run": len(result.stages),
        }
    ]
    context.setdefault("stage_tables", {})[step.module] = result.to_frame()
    return rows


_OPERATIONS: dict[str, Callable[[CampaignStep, dict[str, Any]], list[dict[str, Any]]]] = {
    "tune": _op_tune,
    "stack": _op_stack,
    "threshold": _op_threshold,
    "finetune": _op_finetune,
}


# =============================================================================
# The runner
# =============================================================================
def run_campaign(
    path: Path | str,
    *,
    cfg: Config | None = None,
    context_builder: Callable[[str, Config], dict[str, Any]] | None = None,
    tracker: Any | None = None,
    dry_run: bool = False,
) -> CampaignResult:
    """Execute a campaign declared entirely in YAML.

    Args:
        path: Campaign file.
        cfg: Project config.
        context_builder: ``(module, cfg) -> context`` supplying ``X``, ``y``,
            ``splitter``, ``task``, ``catalog`` and friends. Defaults to
            :func:`default_context_builder`. Exposed so a notebook can reuse
            already-loaded matrices instead of rebuilding them per step - and
            so this runner is testable without scikit-learn.
        tracker: Optional :class:`~novafin.tracking.ExperimentTracker`.
        dry_run: Validate and print the plan without executing.

    Returns:
        A :class:`CampaignResult`.
    """
    cfg = cfg or load_config()
    spec = load_campaign(path)
    builder = context_builder or default_context_builder

    LOGGER.info("Campaign '%s': %d enabled step(s)", spec.name, len(spec.enabled_steps()))
    result = CampaignResult(name=spec.name)

    if dry_run:
        LOGGER.info("Dry run - validated, nothing executed.")
        result.rows = [step.describe() for step in spec.enabled_steps()]
        return result

    context_cache: dict[str, dict[str, Any]] = {}
    for step in spec.enabled_steps():
        try:
            if step.module not in context_cache:
                context_cache[step.module] = builder(step.module, cfg)
            context = dict(context_cache[step.module])
            context["cfg"] = cfg
            context["campaign_name"] = spec.name

            rows = _OPERATIONS[step.operation](step, context)
            for row in rows:
                row.setdefault("campaign", spec.name)
                row.setdefault("module", step.module)
            result.rows.extend(rows)

            if tracker is not None:
                with tracker.run(step.module, f"campaign__{spec.name}__{step.operation}") as handle:
                    handle.params["campaign"] = spec.name
                    handle.params["operation"] = step.operation
                    handle.params["level"] = "L4_campaign"
                    for row in rows:
                        handle.metrics.update(
                            {k: v for k, v in row.items() if isinstance(v, (int, float))}
                        )
        except Exception as exc:
            message = f"{step.module}/{step.operation}: {exc}"
            LOGGER.error("Campaign step failed - %s", message)
            result.errors.append(message)

    if result.rows:
        target = cfg.paths.tables / f"campaign_{spec.name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        result.to_frame().to_csv(target, index=False)
        result.artefacts.append(str(target))
        LOGGER.info("Campaign results written to %s", target)

    return result


def default_context_builder(module: str, cfg: Config) -> dict[str, Any]:
    """Build everything a campaign step needs for one module.

    Loads, engineers features, applies the leakage boundary and constructs the
    module's configured splitter - i.e. exactly the Phase 2-4 pipeline, so a
    campaign inherits every guarantee those phases established rather than
    re-implementing a shortcut.
    """
    from novafin.data import load_dataset, make_feature_frame, make_splitter
    from novafin.features import build_features

    base_module = {"options_residual": "options", "customers_clv": "customers"}.get(module, module)
    loaded = load_dataset(base_module, cfg=cfg)
    built = build_features(base_module, loaded.frame, cfg)
    spec = cfg.dataset(base_module)

    extra = [c for c in ("fwd_return_5d", "fwd_inflows_5d") if c in built.frame.columns]
    X, y = make_feature_frame(built.frame, spec, extra_drop=extra)
    X = X.drop(
        columns=[c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])],
        errors="ignore",
    )

    splitter, split_kwargs = make_splitter(base_module, built.frame, cfg=cfg)
    stack_splitter, _ = make_splitter(base_module, built.frame, cfg=cfg)

    task_map = {
        "binary_classification": "binary_classification",
        "multiclass_classification": "multiclass_classification",
        "regression": "regression",
        "panel_regression": "panel_regression",
        "time_series_forecast": "time_series_forecast",
    }

    return {
        "X": X,
        "y": built.frame[built.target],
        "splitter": splitter,
        "stack_splitter": stack_splitter,
        "split_kwargs": split_kwargs,
        "task": task_map.get(spec.task, "binary_classification"),
        "catalog": load_model_specs(module if module == "customers_clv" else base_module),
        "ic_groups": built.frame["Date"] if base_module == "market" else None,
        "frame": built.frame,
    }
