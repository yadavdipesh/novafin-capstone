"""
novafin-capstone/src/novafin/config.py

Typed configuration loader - the ONLY way any module obtains settings.

Design decisions (all of these are viva questions waiting to happen)
--------------------------------------------------------------------
1.  **Frozen dataclasses over a raw dict.**
    A dict gives you ``cfg["finance"]["credit"]["lgd"]`` with no completion, no
    type checking and a silent ``KeyError`` at 3 a.m. Frozen dataclasses give
    ``cfg.finance.credit.lgd``, fail at load time if a key is missing, and
    cannot be mutated halfway through a pipeline - so a metric logged to MLflow
    is guaranteed to correspond to the config hash recorded beside it.

2.  **Plain YAML + dataclasses instead of Hydra / OmegaConf.**
    Hydra is excellent but it rewrites the working directory, owns ``sys.argv``
    and adds magic that is awkward inside Colab notebooks. The grading
    criterion is transparency, not framework sophistication. This is ~200 lines
    a reviewer can read end to end.

3.  **Environment-variable overrides.**
    ``NOVAFIN_DATA_RAW=/content/drive/MyDrive/Data`` lets a Colab session point
    at Drive without editing a tracked file - which keeps ``git status`` clean
    and stops students from committing machine-specific paths.

4.  **`extra` escape hatch.**
    The D4 Level-4 "EXTRA FINE-TUNING" hook (Phase 6) must let you launch a new
    tuning campaign by editing YAML only. Unknown top-level keys are preserved
    in ``Config.extra`` rather than rejected, so a new campaign block needs zero
    code changes here.

Example
-------
>>> from novafin.config import load_config
>>> cfg = load_config()
>>> cfg.reproducibility.seed
42
>>> cfg.dataset("loans").target
'Default_Flag'
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from novafin.paths import ENV_PREFIX, REPO_ROOT, resolve

__all__ = [
    "Config",
    "ProjectConfig",
    "ReproducibilityConfig",
    "PathsConfig",
    "DatasetConfig",
    "ValidationConfig",
    "ComputeConfig",
    "MLflowConfig",
    "load_config",
    "load_theme",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
DEFAULT_THEME_PATH = REPO_ROOT / "configs" / "theme.yaml"


# =============================================================================
# Leaf sections
# =============================================================================
@dataclass(frozen=True)
class ProjectConfig:
    """Identity block used in report headers, MLflow tags and model cards."""

    name: str
    version: str
    author: str
    group: str
    description: str = ""


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Everything that must be identical between two runs of ``make all``."""

    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    n_jobs: int = -1
    float_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.deterministic and self.cudnn_benchmark:
            # cuDNN autotuning picks a different algorithm per run, which
            # breaks bit-for-bit reproducibility. Refuse the contradiction
            # loudly rather than silently producing non-reproducible results.
            raise ValueError(
                "reproducibility.deterministic=true is incompatible with "
                "cudnn_benchmark=true; set cudnn_benchmark: false."
            )
        if self.float_dtype not in {"float32", "float64"}:
            raise ValueError(f"Unsupported float_dtype: {self.float_dtype!r}")


@dataclass(frozen=True)
class PathsConfig:
    """Repo-relative paths, resolved to absolute at load time."""

    data_raw: Path
    data_interim: Path
    data_processed: Path
    artifacts: Path
    reports: Path
    figures: Path
    tables: Path
    mlruns: Path
    optuna_storage: Path
    logs: Path

    def ensure(self) -> None:
        """Create every writable directory. Idempotent; safe to call on import.

        ``data_raw`` is deliberately NOT created: if it is missing we want a
        loud failure telling the user to mount Drive, not an empty folder that
        silently yields "0 rows".
        """
        for f in fields(self):
            if f.name == "data_raw":
                continue
            target: Path = getattr(self, f.name)
            directory = target.parent if target.suffix else target
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """One entry of the dataset registry.

    ``drop_always`` is the machine-readable form of the Phase-0 leakage audit:
    columns listed here are removed from the feature matrix by
    ``features/engineer.py`` before any model sees them, and
    ``tests/test_leakage.py`` (Phase 2) asserts they never reappear.
    """

    key: str
    filename: str
    module: str
    task: str
    target: str
    id_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    drop_always: list[str] = field(default_factory=list)
    #: Columns that leak but are deliberately RETAINED in the raw frame because
    #: the experiment design needs them (e.g. Black_Scholes_Price, which is the
    #: benchmark for the fair fight and the base of the residual target). They
    #: are excluded from X per-experiment via make_feature_frame(extra_drop=...),
    #: never silently. Declaring them here stops the validator raising CRITICAL
    #: on a leak that has, in fact, already been reasoned about.
    acknowledged_leaks: list[str] = field(default_factory=list)
    group_column: str | None = None
    secondary_target: str | None = None
    engineered_target: str | None = None
    derived_target: str | None = None
    residual_target: str | None = None
    expected_rows: int | None = None
    expected_cols: int | None = None
    positive_rate: float | None = None
    notes: str = ""

    def path(self, paths: PathsConfig) -> Path:
        """Absolute path to the raw CSV for this dataset."""
        return paths.data_raw / self.filename

    @property
    def forbidden_features(self) -> set[str]:
        """Columns that must never enter X: ids, keys, targets and audited leaks.

        The **group column is included deliberately**. It is the entity key used
        to detect entity leakage (``Customer_ID`` for fraud, ``Trading_Day`` for
        the order book), and handing it to the model as a raw numeric feature is
        an invitation to memorise entities instead of learning behaviour - the
        exact failure mode leakage-register entry L-05 exists to prevent. Entity
        information enters the model only as an explicitly engineered, causally
        computed aggregate, never as the bare key.
        """
        banned = set(self.drop_always) | set(self.id_columns) | {self.target}
        if self.group_column:
            banned.add(self.group_column)
        for optional in (
            self.secondary_target,
            self.derived_target,
            self.engineered_target,
        ):
            if optional:
                banned.add(optional)
        return banned


@dataclass(frozen=True)
class ValidationConfig:
    """Per-module cross-validation recipe.

    Kept as a permissive container because the eight modules genuinely need
    eight different schemes (see the Phase-0 validation table); forcing them
    into one rigid dataclass would mean nullable fields everywhere.
    """

    key: str
    scheme: str
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """Read a scheme parameter with a default."""
        return self.params.get(name, default)


@dataclass(frozen=True)
class MLflowConfig:
    """Experiment-tracking settings (D5)."""

    enabled: bool = True
    tracking_uri: str | None = None
    experiment_prefix: str = "novafin"
    log_models: bool = True
    log_figures: bool = True
    leaderboard_path: str = "reports/tables/leaderboard.csv"

    def resolved_uri(self, paths: PathsConfig) -> str:
        """File-store URI when none is configured.

        A local file store needs no server, no account and no network - which
        is exactly what the "no paid APIs, no keys" constraint requires.
        """
        if self.tracking_uri:
            return self.tracking_uri
        return paths.mlruns.as_uri()


@dataclass(frozen=True)
class ComputeConfig:
    """Colab free-tier envelope and the guards that enforce it."""

    max_dataframe_mb: int = 512
    use_gpu: str | bool = "auto"
    colab_session_hours: float = 11.5
    checkpoint_every_n_trials: int = 10

    def gpu_enabled(self) -> bool:
        """Resolve ``use_gpu: auto`` against the actual runtime.

        Returns False (LightGBM-only fallback path) when torch is absent or no
        CUDA device is visible, so the repo runs on a CPU-only Colab session
        without edits.
        """
        if isinstance(self.use_gpu, bool):
            return self.use_gpu
        if str(self.use_gpu).lower() in {"false", "no", "0"}:
            return False
        if str(self.use_gpu).lower() in {"true", "yes", "1"}:
            return True
        try:  # pragma: no cover - depends on runtime hardware
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # torch not installed, or driver missing
            return False


# =============================================================================
# Root config
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Root configuration object handed to every entry point."""

    project: ProjectConfig
    reproducibility: ReproducibilityConfig
    paths: PathsConfig
    datasets: dict[str, DatasetConfig]
    validation: dict[str, ValidationConfig]
    finance: dict[str, Any]
    mlflow: MLflowConfig
    compute: ComputeConfig
    academic_integrity: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    # -- accessors ---------------------------------------------------------
    def dataset(self, key: str) -> DatasetConfig:
        """Look up a dataset by registry key, with a helpful error message."""
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dataset {key!r}. Known keys: {sorted(self.datasets)}"
            ) from exc

    def splits(self, key: str) -> ValidationConfig:
        """Look up the validation recipe for a module."""
        try:
            return self.validation[key]
        except KeyError as exc:
            raise KeyError(
                f"No validation recipe for {key!r}. "
                f"Known keys: {sorted(self.validation)}"
            ) from exc

    def fin(self, *path: str, default: Any = None) -> Any:
        """Read a nested value from the ``finance`` block.

        >>> cfg.fin("credit", "lgd")
        0.4
        """
        node: Any = self.finance
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    # -- provenance --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe nested dict - used for MLflow params and run manifests."""
        return _as_plain(self)

    def fingerprint(self) -> str:
        """Stable SHA-256 of the effective config.

        Logged with every run so a number in the report can always be traced to
        the exact configuration that produced it. Two runs with the same
        fingerprint and the same data hash must produce identical metrics.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Loading
# =============================================================================
def _as_plain(obj: Any) -> Any:
    """Recursively convert dataclasses/Paths into JSON-safe primitives."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _as_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_as_plain(v) for v in obj]
    return obj


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file, failing loudly with the offending path."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)  # safe_load: never execute arbitrary tags
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return data


def _apply_env_overrides(paths: dict[str, Any]) -> dict[str, Any]:
    """Override any ``paths.*`` entry from ``NOVAFIN_<UPPER_KEY>``."""
    out = dict(paths)
    for key in list(out):
        env_key = f"{ENV_PREFIX}{key.upper()}"
        if env_key in os.environ:
            LOGGER.info("Path override from %s -> %s", env_key, os.environ[env_key])
            out[key] = os.environ[env_key]
    return out


def _build_paths(raw: dict[str, Any]) -> PathsConfig:
    resolved = {k: resolve(v) for k, v in _apply_env_overrides(raw).items()}
    return PathsConfig(**resolved)


def _build_datasets(raw: dict[str, Any]) -> dict[str, DatasetConfig]:
    known = {f.name for f in fields(DatasetConfig)}
    out: dict[str, DatasetConfig] = {}
    for key, block in (raw or {}).items():
        payload = {k: v for k, v in block.items() if k in known}
        unknown = set(block) - known
        if unknown:
            # Warn rather than fail: a new descriptive key in YAML should not
            # break a pipeline, but silence would hide typos like "targt".
            LOGGER.warning("datasets.%s: ignoring unknown keys %s", key, sorted(unknown))
        out[key] = DatasetConfig(key=key, **payload)
    return out


def _build_validation(raw: dict[str, Any]) -> dict[str, ValidationConfig]:
    out: dict[str, ValidationConfig] = {}
    for key, block in (raw or {}).items():
        block = dict(block or {})
        scheme = block.pop("scheme", "kfold")
        out[key] = ValidationConfig(key=key, scheme=scheme, params=block)
    return out


def load_config(
    path: Path | str | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    ensure_dirs: bool = True,
) -> Config:
    """Load, validate and freeze the project configuration.

    Args:
        path: YAML file to read. Defaults to ``configs/config.yaml``.
        overrides: Shallow top-level overrides applied after the file is read.
            Used by tests and by the Level-4 tuning campaigns, which merge a
            campaign YAML over the base config.
        ensure_dirs: Create writable output directories.

    Returns:
        A frozen :class:`Config`.

    Raises:
        FileNotFoundError: If the YAML file is missing.
        ValueError: If a required section is absent or internally contradictory.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(cfg_path)

    if overrides:
        raw = {**raw, **dict(overrides)}

    required = ("project", "reproducibility", "paths", "datasets")
    missing = [section for section in required if section not in raw]
    if missing:
        raise ValueError(f"{cfg_path} is missing required section(s): {missing}")

    consumed = {
        "project",
        "reproducibility",
        "paths",
        "datasets",
        "validation",
        "finance",
        "mlflow",
        "compute",
        "academic_integrity",
    }

    paths = _build_paths(raw["paths"])
    if ensure_dirs:
        paths.ensure()

    cfg = Config(
        project=ProjectConfig(**raw["project"]),
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        paths=paths,
        datasets=_build_datasets(raw["datasets"]),
        validation=_build_validation(raw.get("validation", {})),
        finance=raw.get("finance", {}),
        mlflow=MLflowConfig(**raw.get("mlflow", {})),
        compute=ComputeConfig(**raw.get("compute", {})),
        academic_integrity=raw.get("academic_integrity", {}),
        # Unknown top-level blocks survive here so a Level-4 campaign can add
        # sections without touching this file.
        extra={k: v for k, v in raw.items() if k not in consumed},
        source_path=cfg_path,
    )
    LOGGER.debug("Loaded config %s (fingerprint %s)", cfg_path, cfg.fingerprint())
    return cfg


def load_theme(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``configs/theme.yaml`` as a plain dict.

    Deliberately NOT a dataclass: the theme is consumed by matplotlib,
    python-docx, python-pptx and Gradio, each of which wants a different slice.
    A dict keeps the token file free to grow without a code change - the same
    "edit YAML only" principle as the tuning campaigns.
    """
    return _read_yaml(Path(path) if path is not None else DEFAULT_THEME_PATH)
