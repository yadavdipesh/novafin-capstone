"""
novafin-capstone/src/novafin/utils/seed.py

``seed_everything()`` - one call that pins every RNG the project touches.

Why a capstone needs this
-------------------------
Examiners re-run submissions. If the AUC in your report is 0.7412 and their run
gives 0.7389, the first question is "which number is real?" - and there is no
good answer. Pinning every generator makes the report's numbers falsifiable,
which is the whole point of the "rigorous validation" criterion.

What is actually seeded
-----------------------
=========================  ====================================================
Generator                  How it is pinned
=========================  ====================================================
``random``                 ``random.seed(seed)``
``numpy``                  ``np.random.seed`` (legacy global) **and** a returned
                           ``default_rng(seed)`` for modern code
``PYTHONHASHSEED``         ``os.environ`` - see the honest caveat below
``torch`` (CPU + CUDA)     ``manual_seed`` / ``cuda.manual_seed_all``
``cuDNN``                  ``deterministic=True``, ``benchmark=False``
``cuBLAS``                 ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
scikit-learn / LightGBM /  consume the numpy global RNG, or take
XGBoost / Optuna           ``random_state=cfg.reproducibility.seed`` explicitly
=========================  ====================================================

Honest caveat (say this in the viva, do not hide it)
----------------------------------------------------
``PYTHONHASHSEED`` is read by CPython **at interpreter start-up**. Setting it
from inside a running process does not retroactively change string hashing in
that process; it only affects child processes (e.g. ``joblib`` workers spawned
later). For a fully deterministic session the variable must be exported before
Python launches - which is exactly what the ``Makefile`` and the first cell of
every notebook do. This function still sets it, because that genuinely fixes
the multiprocessing workers, and it warns when it was not already set.

Determinism also costs speed: ``torch.use_deterministic_algorithms(True)``
disables faster non-deterministic kernels. That trade is correct here - the
datasets are small (largest is 120k x 33) and reproducibility is graded.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["seed_everything", "SeedReport", "worker_init_fn"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedReport:
    """What was actually pinned - logged to MLflow so a run is auditable."""

    seed: int
    deterministic: bool
    pythonhashseed_was_preset: bool
    torch_seeded: bool
    cuda_seeded: bool
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Flat dict suitable for ``mlflow.log_params``."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "pythonhashseed_preset": self.pythonhashseed_was_preset,
            "torch_seeded": self.torch_seeded,
            "cuda_seeded": self.cuda_seeded,
            "seed_warnings": "; ".join(self.warnings) if self.warnings else "none",
        }


def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> SeedReport:
    """Pin every random number generator used by the project.

    Args:
        seed: The master seed. The dataset README specifies 42, and the project
            config carries the same value.
        deterministic: Request deterministic kernels from torch/cuDNN. Slower,
            but required for bit-for-bit reproducibility.
        cudnn_benchmark: cuDNN autotuning. Must be ``False`` when
            ``deterministic`` is ``True``; the two are contradictory.

    Returns:
        A :class:`SeedReport` describing exactly what was pinned, including any
        caveat that applies to this process.

    Raises:
        ValueError: If ``deterministic`` and ``cudnn_benchmark`` are both True.
    """
    if deterministic and cudnn_benchmark:
        raise ValueError(
            "deterministic=True is incompatible with cudnn_benchmark=True."
        )

    warnings: list[str] = []

    preset = os.environ.get("PYTHONHASHSEED")
    pythonhashseed_was_preset = preset == str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not pythonhashseed_was_preset:
        warnings.append(
            "PYTHONHASHSEED was not exported before interpreter start; "
            "string hashing in THIS process is not pinned (child processes are). "
            "Use `make run` or the notebook bootstrap cell for full determinism."
        )

    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG - what sklearn/LightGBM read

    torch_seeded = False
    cuda_seeded = False
    try:
        import torch
    except ImportError:
        warnings.append("torch not installed; torch RNGs not seeded.")
    else:
        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
            # cuBLAS needs a fixed workspace for deterministic GEMMs.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # pragma: no cover - version dependent
                warnings.append(f"use_deterministic_algorithms unavailable: {exc}")

    report = SeedReport(
        seed=seed,
        deterministic=deterministic,
        pythonhashseed_was_preset=pythonhashseed_was_preset,
        torch_seeded=torch_seeded,
        cuda_seeded=cuda_seeded,
        warnings=tuple(warnings),
    )
    LOGGER.info("seed_everything(%s) -> %s", seed, report.as_dict())
    for message in warnings:
        LOGGER.warning(message)
    return report


def make_rng(seed: int = 42) -> np.random.Generator:
    """Return a modern, explicitly-seeded numpy generator.

    Preferred over the legacy global RNG in new code: passing a
    :class:`numpy.random.Generator` around makes randomness an explicit
    dependency instead of hidden global state.
    """
    return np.random.default_rng(seed)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Seed a ``torch.utils.data.DataLoader`` worker process.

    Each worker is a separate process with its own RNG state. Without this,
    ``num_workers > 0`` silently reintroduces nondeterminism - a classic and
    very hard-to-spot reproducibility bug.

    Args:
        worker_id: Index supplied by the DataLoader.
        base_seed: Master seed; the worker seed is ``base_seed + worker_id``.
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except ImportError:  # pragma: no cover
        pass
