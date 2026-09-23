"""
novafin-capstone/src/novafin/paths.py

Repository-root discovery.

Why this module exists
----------------------
The project must run identically in three places: a local checkout, a GitHub
Actions runner, and a Google Colab session created by ``!git clone``. In all
three the *absolute* path differs, so no module may ever hardcode one.

``find_repo_root()`` walks upward from this file looking for a marker that only
the repository root carries (``pyproject.toml`` plus ``configs/``). Everything
else in the codebase resolves relative paths through :func:`resolve`.

Alternative considered and rejected
-----------------------------------
``os.getcwd()`` - rejected because Colab notebooks run from ``/content`` while
the package lives in ``/content/novafin-capstone``, so the CWD is wrong roughly
half the time. Walking up from ``__file__`` is invariant to the caller's CWD.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = ["find_repo_root", "resolve", "REPO_ROOT", "ENV_PREFIX"]

#: Environment-variable prefix used for config overrides, e.g. NOVAFIN_DATA_RAW.
ENV_PREFIX: Final[str] = "NOVAFIN_"

#: Files/directories that together identify the repository root unambiguously.
_ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "configs")


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing all root markers.

    Args:
        start: Directory or file to start from. Defaults to this module's
            location, which makes the result independent of the caller's CWD.

    Returns:
        Absolute path to the repository root.

    Raises:
        FileNotFoundError: If no ancestor directory carries the root markers.
            This is deliberate: silently falling back to the CWD produces
            "file not found" errors far from their cause.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if here.is_file():
        here = here.parent

    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate

    raise FileNotFoundError(
        f"Could not locate the repository root above {here}. "
        f"Expected an ancestor directory containing {_ROOT_MARKERS}."
    )


#: Resolved once at import time. Cheap, and makes the value inspectable.
REPO_ROOT: Final[Path] = find_repo_root()


def resolve(path: Path | str, root: Path | None = None) -> Path:
    """Resolve ``path`` against the repository root unless it is absolute.

    Also expands ``~`` and ``$VAR`` so a Colab user can point
    ``paths.data_raw`` at ``$HOME/drive/MyDrive/...`` in the YAML.

    Args:
        path: Absolute or repo-relative path.
        root: Override for the repository root (used by tests).

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    expanded = Path(os.path.expandvars(str(path))).expanduser()
    if expanded.is_absolute():
        return expanded
    return (root or REPO_ROOT) / expanded
