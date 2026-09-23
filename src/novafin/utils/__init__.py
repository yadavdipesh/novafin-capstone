"""
novafin-capstone/src/novafin/utils/__init__.py

Cross-cutting utilities: reproducibility, artefact I/O, logging and theming.

These four modules have no dependency on any other part of the package, so they
can be imported from anywhere (including notebook cell 1) without circularity.
"""

from __future__ import annotations

from novafin.utils.io import ModelBundle, RunManifest, sha256_file
from novafin.utils.logging_utils import get_logger, setup_logging
from novafin.utils.seed import SeedReport, make_rng, seed_everything, worker_init_fn
from novafin.utils.theme import apply_theme, color, save_figure, semantic_color

__all__ = [
    "ModelBundle",
    "RunManifest",
    "sha256_file",
    "get_logger",
    "setup_logging",
    "SeedReport",
    "make_rng",
    "seed_everything",
    "worker_init_fn",
    "apply_theme",
    "color",
    "save_figure",
    "semantic_color",
]
