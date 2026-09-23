"""
novafin-capstone/src/novafin/__init__.py

NovaFin Group Capstone - ML-Driven Enterprise Financial Strategy, Risk &
Decision Support.

Package layout
--------------
``novafin.config``    typed configuration loader (single source of truth)
``novafin.paths``     repository-root discovery
``novafin.data``      loading + schema/leakage validation        (Phase 2)
``novafin.features``  engineering, encoding, selection           (Phase 3)
``novafin.models``    build / train / tune / finetune            (Phases 4-6)
``novafin.finance``   NPV, ECL, CLV, VaR, Black-Scholes, Greeks  (Phase 7)
``novafin.utils``     seed, io, logging, theme                   (Phase 1)

Import convention: notebooks orchestrate and visualise; all logic lives here.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Dipesh Kumar Yadav"
__licence__ = "MIT"

from novafin.config import Config, load_config, load_theme
from novafin.paths import REPO_ROOT, find_repo_root, resolve

__all__ = [
    "__version__",
    "__author__",
    "__licence__",
    "Config",
    "load_config",
    "load_theme",
    "REPO_ROOT",
    "find_repo_root",
    "resolve",
]
