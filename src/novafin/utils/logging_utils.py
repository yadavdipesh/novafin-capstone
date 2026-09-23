"""
novafin-capstone/src/novafin/utils/logging_utils.py

Project-wide logging setup.

Why logging and not ``print``
-----------------------------
``print`` cannot be filtered, cannot be redirected to a file, carries no
timestamp and no module name, and floods a notebook during a 200-trial Optuna
study. The D2 contract asks for logging in every module; this is the single
place it is configured, so a notebook gets one readable line per event and a
CI run gets a full ``reports/logs/*.log`` transcript.

Idempotence matters in notebooks: re-running the setup cell must not attach a
second handler (which is how you end up with every line printed three times).
:func:`setup_logging` therefore clears the project logger's handlers first.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

__all__ = ["setup_logging", "get_logger", "PROJECT_LOGGER_NAME"]

PROJECT_LOGGER_NAME = "novafin"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

LevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LevelName | int = "INFO",
    *,
    log_file: Path | str | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure and return the project logger.

    Args:
        level: Console verbosity. The file handler always records DEBUG, so a
            quiet notebook still leaves a full transcript on disk.
        log_file: Optional path for the file handler. Parent directories are
            created.
        quiet_libraries: Raise third-party loggers to WARNING. Without this,
            matplotlib's font manager and MLflow's tracking client bury your
            own messages.

    Returns:
        The configured ``novafin`` logger. Child loggers created with
        :func:`get_logger` inherit from it.
    """
    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)          # handlers do the filtering
    logger.handlers.clear()                 # idempotent across notebook re-runs
    logger.propagate = False                # avoid duplicate root output

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level if isinstance(level, int) else getattr(logging, level))
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console)

    if log_file is not None:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)
        logger.debug("Logging to %s", target)

    if quiet_libraries:
        for noisy in (
            "matplotlib",
            "matplotlib.font_manager",
            "PIL",
            "mlflow",
            "optuna",
            "urllib3",
            "git",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the project logger.

    Call as ``get_logger(__name__)``; the ``novafin.`` prefix is added when the
    caller is outside the package so every line is attributable.
    """
    if name.startswith(PROJECT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{PROJECT_LOGGER_NAME}.{name}")
