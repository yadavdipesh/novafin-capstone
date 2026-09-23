"""
novafin-capstone/src/novafin/tracking.py

D5 - experiment tracking on a local MLflow file store.

Why a local file store and not a server
---------------------------------------
The capstone constraint is 100% free and open source, no paid APIs, no keys.
MLflow's file store satisfies that completely: ``mlruns/`` is a directory, runs
are folders, and nothing needs a network service or an account.

The UI is the part that does not survive Colab. ``mlflow ui`` binds to
localhost, and exposing it from a Colab VM needs a tunnel - ngrok is free but
requires an account token, which the constraints forbid. The fallback is built
in rather than bolted on: :func:`load_runs` reads the run store directly with
``mlflow.search_runs`` into a pandas frame, and ``models/leaderboard.py`` turns
that into the D5 leaderboard table. No server, no port, no token.

Graceful degradation
--------------------
If MLflow is not installed - or the store cannot be written, which happens on
some read-only Colab mounts - tracking downgrades to a JSON-lines file and the
pipeline continues. Losing an experiment log must never cost you a training
run; the metrics are still returned to the caller and still written to
``reports/tables/``.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from novafin.config import Config, load_config
from novafin.utils.io import atomic_write_text, timestamp_slug

__all__ = ["ExperimentTracker", "RunHandle", "load_runs", "mlflow_available"]

LOGGER = logging.getLogger(__name__)


def mlflow_available() -> bool:
    """True when MLflow can be imported."""
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class RunHandle:
    """A single logged run, usable whether or not MLflow is present."""

    run_id: str
    module: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    backend: str = "mlflow"

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "module": self.module,
            "model": self.model,
            "backend": self.backend,
        }
        record.update({f"param.{k}": v for k, v in self.params.items()})
        record.update({f"metric.{k}": v for k, v in self.metrics.items()})
        record.update({f"tag.{k}": v for k, v in self.tags.items()})
        return record


class ExperimentTracker:
    """Thin wrapper over MLflow with a JSON-lines fallback.

    Example:
        >>> tracker = ExperimentTracker(cfg)
        >>> with tracker.run("loans", "lightgbm") as run:
        ...     run.params.update(spec.params)
        ...     run.metrics.update(metrics)
        ...     tracker.log_table(run, fold_metrics, "folds.csv")
    """

    def __init__(self, cfg: Config | None = None, *, enabled: bool | None = None) -> None:
        self.cfg = cfg or load_config()
        self.enabled = self.cfg.mlflow.enabled if enabled is None else enabled
        self.backend = "disabled"
        self._mlflow: Any = None

        fallback_dir = self.cfg.paths.mlruns
        fallback_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_path = fallback_dir / "runs.jsonl"

        if not self.enabled:
            LOGGER.info("Experiment tracking disabled by config.")
            return

        if not mlflow_available():
            self.backend = "jsonl"
            LOGGER.warning("MLflow not installed; falling back to %s", self.fallback_path)
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(self.cfg.mlflow.resolved_uri(self.cfg.paths))
            self._mlflow = mlflow
            self.backend = "mlflow"
            LOGGER.info("MLflow tracking at %s", mlflow.get_tracking_uri())
        except Exception as exc:  # read-only mount, permissions, version skew
            self.backend = "jsonl"
            LOGGER.warning("MLflow unavailable (%s); falling back to JSON lines.", exc)

    # -- naming ------------------------------------------------------------
    def experiment_name(self, module: str) -> str:
        return f"{self.cfg.mlflow.experiment_prefix}_{module}"

    # -- run context -------------------------------------------------------
    @contextmanager
    def run(self, module: str, model: str, *, tags: dict[str, str] | None = None) -> Iterator[RunHandle]:
        """Open a run; params/metrics added to the handle are flushed on exit.

        Every run is tagged with the config fingerprint and the seed, so a
        metric in the leaderboard can always be traced to the configuration
        that produced it.
        """
        handle = RunHandle(
            run_id=f"{module}-{model}-{timestamp_slug()}",
            module=module,
            model=model,
            backend=self.backend,
            tags={
                "module": module,
                "model": model,
                "config_fingerprint": self.cfg.fingerprint(),
                "seed": str(self.cfg.reproducibility.seed),
                "project_version": self.cfg.project.version,
                **(tags or {}),
            },
        )

        if self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.set_experiment(self.experiment_name(module))
            with self._mlflow.start_run(run_name=f"{model}") as active:
                handle.run_id = active.info.run_id
                try:
                    yield handle
                finally:
                    self._flush_mlflow(handle)
        else:
            try:
                yield handle
            finally:
                self._flush_jsonl(handle)

    # -- flushing ----------------------------------------------------------
    def _flush_mlflow(self, handle: RunHandle) -> None:
        try:
            self._mlflow.set_tags(handle.tags)
            # MLflow rejects non-scalar params; stringify anything structured.
            self._mlflow.log_params(
                {k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))
                 for k, v in handle.params.items()}
            )
            clean = {
                k: float(v) for k, v in handle.metrics.items()
                if isinstance(v, (int, float)) and pd.notna(v)
            }
            if clean:
                self._mlflow.log_metrics(clean)
        except Exception as exc:  # pragma: no cover - never lose a run to logging
            LOGGER.warning("MLflow logging failed for %s: %s", handle.run_id, exc)

    def _flush_jsonl(self, handle: RunHandle) -> None:
        try:
            line = json.dumps(handle.as_record(), default=str)
            with self.fallback_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Fallback logging failed for %s: %s", handle.run_id, exc)

    # -- artefacts ---------------------------------------------------------
    def log_table(self, handle: RunHandle, frame: pd.DataFrame, filename: str) -> Path | None:
        """Log a dataframe as a CSV artefact.

        Always written to ``reports/tables/`` as well, so the guide and the deck
        can read it from a predictable path without an MLflow lookup.
        """
        if frame is None or frame.empty:
            return None
        target = self.cfg.paths.tables / f"{handle.module}__{handle.model}__{filename}"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, frame.to_csv(index=False))
        handle.artifacts.append(str(target))

        if self.backend == "mlflow" and self._mlflow is not None:
            try:
                self._mlflow.log_artifact(str(target))
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Artifact logging skipped: %s", exc)
        return target

    def log_figure(self, handle: RunHandle, path: Path | str) -> None:
        """Attach an already-saved figure to the run."""
        handle.artifacts.append(str(path))
        if self.backend == "mlflow" and self._mlflow is not None and self.cfg.mlflow.log_figures:
            try:
                self._mlflow.log_artifact(str(path))
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Figure logging skipped: %s", exc)


def load_runs(cfg: Config | None = None) -> pd.DataFrame:
    """Read every logged run into a dataframe.

    This is the no-server path to the D5 leaderboard: it reads the run store
    directly rather than requiring ``mlflow ui``, which cannot be reached from
    a Colab VM without a tunnel account.

    Returns:
        A frame with ``module``, ``model`` and the logged metric columns.
        Empty if nothing has been logged yet.
    """
    cfg = cfg or load_config()
    frames: list[pd.DataFrame] = []

    if mlflow_available():
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.mlflow.resolved_uri(cfg.paths))
            client = mlflow.tracking.MlflowClient()
            experiments = [
                e for e in client.search_experiments()
                if e.name.startswith(cfg.mlflow.experiment_prefix)
            ]
            if experiments:
                runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in experiments])
                if len(runs):
                    frames.append(runs)
        except Exception as exc:
            LOGGER.warning("Could not read MLflow runs: %s", exc)

    fallback = cfg.paths.mlruns / "runs.jsonl"
    if fallback.exists():
        records = [
            json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
