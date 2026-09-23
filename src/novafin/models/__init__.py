"""
novafin-capstone/src/novafin/models/__init__.py

Model layer: declare in YAML, fit inside the fold, rank from the run store.

* :mod:`novafin.models.build` - estimators resolved from ``configs/models.yaml``
  by dotted path, so adding a model is a config edit. This is the mechanism the
  D4 Level-4 hook depends on.
* :mod:`novafin.models.train` - the single cross-validation loop. Fits the
  whole pipeline inside each fold and assembles out-of-fold predictions.
* :mod:`novafin.models.leaderboard` - D5, generated from logged runs.
"""

from __future__ import annotations

from novafin.models.build import (
    ALLOWED_MODULE_PREFIXES,
    ModelCatalog,
    ModelSpec,
    available_modules,
    build_estimator,
    build_pipeline,
    load_model_specs,
    resolve_class,
)
from novafin.models.campaign import (
    CampaignResult,
    CampaignSpec,
    CampaignStep,
    list_campaigns,
    load_campaign,
    run_campaign,
    validate_campaign,
)
from novafin.models.finetune import (
    FTTransformerConfig,
    LoRAConfig,
    StagedBoostingResult,
    attach_lora,
    build_ft_transformer,
    default_schedule,
    discriminative_learning_rates,
    layerwise_unfreeze_schedule,
    lora_parameter_report,
    staged_boosting,
)
from novafin.models.leaderboard import build_leaderboard, format_leaderboard
from novafin.models.train import (
    CVResult,
    FoldResult,
    cross_validate_model,
    evaluate_cv,
    make_bundle,
    train_module,
)

__all__ = [
    "ModelSpec",
    "ModelCatalog",
    "load_model_specs",
    "build_estimator",
    "build_pipeline",
    "resolve_class",
    "available_modules",
    "ALLOWED_MODULE_PREFIXES",
    "FoldResult",
    "CVResult",
    "cross_validate_model",
    "evaluate_cv",
    "make_bundle",
    "train_module",
    "build_leaderboard",
    "format_leaderboard",
    # L3 fine-tuning
    "staged_boosting",
    "StagedBoostingResult",
    "default_schedule",
    "LoRAConfig",
    "attach_lora",
    "lora_parameter_report",
    "FTTransformerConfig",
    "build_ft_transformer",
    "layerwise_unfreeze_schedule",
    "discriminative_learning_rates",
    # L4 campaigns
    "CampaignSpec",
    "CampaignStep",
    "CampaignResult",
    "load_campaign",
    "validate_campaign",
    "run_campaign",
    "list_campaigns",
]
