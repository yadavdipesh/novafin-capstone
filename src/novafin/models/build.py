"""
novafin-capstone/src/novafin/models/build.py

Model factory - estimators are declared in YAML, never constructed in code.

The design decision, and why it matters beyond tidiness
-------------------------------------------------------
``configs/models.yaml`` gives each model a dotted import path and a parameter
dict; :func:`build_estimator` resolves the path with ``importlib`` and
instantiates it. Adding LightGBM to a module, changing ``num_leaves``, or
swapping in CatBoost is a config edit.

That is the mechanism the **D4 Level-4 "EXTRA FINE-TUNING" hook** depends on.
The contract there is that a new tuning campaign - a new search space, more
trials, a different ensemble - must be launchable by editing YAML with zero
code changes. That is only achievable if nothing downstream ever names a model
class. So nothing does.

Security note: ``importlib`` on a config-supplied string is code execution by
another name. :data:`ALLOWED_MODULE_PREFIXES` restricts resolution to the
modelling libraries pinned in ``requirements.txt``, so a malformed or hostile
config cannot import arbitrary code.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from novafin.config import Config, load_config
from novafin.paths import REPO_ROOT

__all__ = [
    "ModelSpec",
    "ModelCatalog",
    "load_model_specs",
    "build_estimator",
    "build_pipeline",
    "resolve_class",
    "ALLOWED_MODULE_PREFIXES",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = REPO_ROOT / "configs" / "models.yaml"

#: Only these top-level packages may be imported from config. Everything here
#: is pinned in requirements.txt with a stated licence.
ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "sklearn",
    "lightgbm",
    "xgboost",
    "catboost",
    "novafin",
)


@dataclass(frozen=True)
class ModelSpec:
    """One model declaration from ``configs/models.yaml``."""

    name: str
    class_path: str
    params: dict[str, Any] = field(default_factory=dict)
    scale: bool = False
    unsupervised: bool = False
    enabled: bool = True
    notes: str = ""

    @property
    def library(self) -> str:
        """Top-level package, used for MLflow tags and the leaderboard."""
        return self.class_path.split(".")[0]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "class": self.class_path,
            "library": self.library,
            "scale": self.scale,
            "unsupervised": self.unsupervised,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class ModelCatalog:
    """The model declarations for one module, plus the global defaults.

    A small container rather than a bare tuple: `specs, defaults = load(...)`
    is easy to get backwards, and a returned tuple annotated as `list` was
    exactly that kind of latent bug in the first draft of this file.
    """

    module: str
    specs: list[ModelSpec]
    defaults: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def get(self, name: str) -> ModelSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"No model named {name!r} for '{self.module}'. Have: {self.names()}")


def resolve_class(class_path: str) -> type:
    """Import a class from a dotted path, restricted to allowed packages.

    Args:
        class_path: e.g. ``"lightgbm.LGBMClassifier"``.

    Returns:
        The class object.

    Raises:
        ValueError: If the path is outside :data:`ALLOWED_MODULE_PREFIXES`.
        ImportError: If the module cannot be imported - with a message naming
            the library to install, since the usual cause is a missing pin.
        AttributeError: If the module has no such attribute.
    """
    if "." not in class_path:
        raise ValueError(f"'{class_path}' is not a dotted path, e.g. 'sklearn.dummy.DummyClassifier'")

    root = class_path.split(".")[0]
    if root not in ALLOWED_MODULE_PREFIXES:
        raise ValueError(
            f"Refusing to import from '{root}'. Allowed packages: "
            f"{ALLOWED_MODULE_PREFIXES}. Config-driven imports are restricted "
            "to the pinned modelling libraries."
        )

    module_path, _, class_name = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import '{module_path}' for model '{class_path}'. "
            f"Install it: pip install -r requirements.txt  ({exc})"
        ) from exc

    if not hasattr(module, class_name):
        raise AttributeError(f"'{module_path}' has no attribute '{class_name}'")
    return getattr(module, class_name)


def load_model_specs(
    module_key: str,
    *,
    path: Path | str | None = None,
    include_disabled: bool = False,
) -> ModelCatalog:
    """Read the model declarations for one module.

    Args:
        module_key: Key in ``models.yaml`` (usually the dataset key; ``M4``
            also has ``customers_clv`` for its regression half).
        path: Override the YAML path - used by Level-4 campaigns.
        include_disabled: Return entries marked ``enabled: false`` too.

    Returns:
        A :class:`ModelCatalog`.

    Raises:
        FileNotFoundError: If the YAML is missing.
        KeyError: If the module has no entry, listing the keys that exist.
    """
    yaml_path = Path(path) if path is not None else DEFAULT_MODELS_PATH
    if not yaml_path.exists():
        raise FileNotFoundError(f"Model config not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    defaults = dict(raw.get("defaults", {}))

    if module_key not in raw:
        available = [k for k in raw if k != "defaults"]
        raise KeyError(f"No models declared for '{module_key}'. Available: {available}")

    specs: list[ModelSpec] = []
    for entry in raw[module_key]:
        params = dict(entry.get("params", {}))
        # Global defaults are applied only where the estimator accepts them;
        # that check happens in build_estimator, which has the class in hand.
        spec = ModelSpec(
            name=entry["name"],
            class_path=entry["class"],
            params=params,
            scale=bool(entry.get("scale", False)),
            unsupervised=bool(entry.get("unsupervised", False)),
            enabled=bool(entry.get("enabled", True)),
            notes=entry.get("notes", ""),
        )
        if spec.enabled or include_disabled:
            specs.append(spec)

    LOGGER.info("Loaded %d model spec(s) for '%s' from %s", len(specs), module_key, yaml_path.name)
    return ModelCatalog(module=module_key, specs=specs, defaults=defaults)


def build_estimator(
    spec: ModelSpec,
    *,
    cfg: Config | None = None,
    defaults: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    """Instantiate the estimator described by ``spec``.

    Two conveniences that prevent real bugs:

    * ``random_state`` is injected from the project config whenever the
      estimator accepts it, so no model can be accidentally unseeded;
    * global defaults (``n_jobs``, ``verbose``) are filtered against the
      constructor signature, so a default meant for LightGBM does not raise a
      ``TypeError`` on a scikit-learn estimator that has no such argument.

    Args:
        spec: The declaration.
        cfg: Project config, for the seed.
        defaults: Global defaults from ``models.yaml``.
        extra_params: Overrides applied last - used by tuning to inject a
            trial's hyper-parameters.

    Returns:
        An unfitted estimator.
    """
    import inspect

    cfg = cfg or load_config()
    estimator_class = resolve_class(spec.class_path)

    try:
        signature = inspect.signature(estimator_class.__init__)
        accepted = set(signature.parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic estimators
        accepted = set()

    params: dict[str, Any] = {}
    for key, value in (defaults or {}).items():
        if not accepted or key in accepted:
            params[key] = value
    params.update(spec.params)
    params.update(extra_params or {})

    for seed_name in ("random_state", "seed"):
        if (not accepted or seed_name in accepted) and seed_name not in params:
            params[seed_name] = cfg.reproducibility.seed
            break

    params = {k: v for k, v in params.items() if not accepted or k in accepted}

    LOGGER.debug("Building %s with %s", spec.class_path, params)
    return estimator_class(**params)


def build_pipeline(
    spec: ModelSpec,
    X: Any,
    *,
    cfg: Config | None = None,
    defaults: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    """Wrap the estimator in a preprocessing pipeline.

    Returns an **unfitted** ``Pipeline``. This is the object handed to the
    cross-validation loop, and it is what makes the whole thing leak-free: the
    imputer, the scaler and the one-hot category list are all fitted inside
    ``pipeline.fit(X_train)``, never on the full dataset.

    Args:
        spec: The declaration.
        X: Feature frame, used only for its column types.
        cfg: Project config.
        defaults: Global defaults.
        extra_params: Estimator overrides.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.
    """
    from sklearn.pipeline import Pipeline

    from novafin.features.encoders import build_preprocessor

    preprocessor = build_preprocessor(X, scale=spec.scale)
    estimator = build_estimator(spec, cfg=cfg, defaults=defaults, extra_params=extra_params)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def available_modules(path: Path | str | None = None) -> list[str]:
    """List the module keys declared in ``models.yaml``."""
    yaml_path = Path(path) if path is not None else DEFAULT_MODELS_PATH
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return [k for k in raw if k != "defaults"]
