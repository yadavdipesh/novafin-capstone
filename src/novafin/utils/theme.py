"""
novafin-capstone/src/novafin/utils/theme.py

Mubadala theme application - colours and fonts only.

One call, ``apply_theme()``, in the first cell of every notebook makes every
figure in the D7 guide and the D8 deck identical by construction. Nothing else
in the repository may reference a hex code.

Font licensing (this is a graded constraint, so it is enforced in code)
-----------------------------------------------------------------------
Charts use **Archivo** and **Tajawal**, both SIL Open Font License 1.1, fetched
at runtime from Google Fonts. **Interstate is proprietary and is never used for
charts** - it appears only as a font *name* inside .docx/.pptx, rendered by a
locally installed copy. ``install_chart_fonts()`` downloads nothing if the
fonts are already present and degrades silently to matplotlib's bundled DejaVu
Sans when there is no network, so the pipeline never fails because of a font.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from novafin.config import load_theme
from novafin.paths import REPO_ROOT

__all__ = [
    "apply_theme",
    "get_palette",
    "color",
    "semantic_color",
    "decision_color",
    "categorical_cycle",
    "sequential_cmap",
    "diverging_cmap",
    "install_chart_fonts",
    "save_figure",
]

LOGGER = logging.getLogger(__name__)

_THEME_CACHE: dict[str, Any] | None = None
_FONT_DIR = REPO_ROOT / "artifacts" / "fonts"

#: Direct, stable Google Fonts static TTF URLs (SIL OFL 1.1).
_FONT_URLS: dict[str, str] = {
    "Archivo": (
        "https://github.com/google/fonts/raw/main/ofl/archivo/"
        "Archivo%5Bwdth%2Cwght%5D.ttf"
    ),
    "Tajawal": (
        "https://github.com/google/fonts/raw/main/ofl/tajawal/Tajawal-Regular.ttf"
    ),
}


def _theme() -> dict[str, Any]:
    """Load and memoise ``configs/theme.yaml``."""
    global _THEME_CACHE
    if _THEME_CACHE is None:
        _THEME_CACHE = load_theme()
    return _THEME_CACHE


# =============================================================================
# Token accessors
# =============================================================================
def get_palette() -> dict[str, str]:
    """Return the full ``{token: hex}`` palette."""
    return dict(_theme()["palette"])


def color(token: str) -> str:
    """Resolve a palette token to its hex value.

    Args:
        token: A key of ``theme.palette`` (e.g. ``"teal"``), or a literal hex
            string, which is returned unchanged so callers can pass either.

    Raises:
        KeyError: If the token is unknown - deliberately loud, because a typo
            that silently returns grey would quietly break brand consistency.
    """
    if token.startswith("#"):
        return token
    palette = _theme()["palette"]
    if token not in palette:
        raise KeyError(f"Unknown palette token {token!r}. Known: {sorted(palette)}")
    return palette[token]


def semantic_color(role: str) -> str:
    """Resolve a semantic role (``good``/``warning``/``critical``/...) to hex."""
    semantic = _theme()["semantic"]
    if role not in semantic:
        raise KeyError(f"Unknown semantic role {role!r}. Known: {sorted(semantic)}")
    return color(semantic[role])


def decision_color(label: str) -> str:
    """Resolve a business decision label (``Approve``/``Review``/...) to hex.

    Falls back to the neutral token so a new decision label renders sensibly
    instead of crashing a report build.
    """
    mapping = _theme().get("decisions", {})
    return color(mapping.get(label, _theme()["semantic"]["neutral"]))


def categorical_cycle() -> list[str]:
    """Ordered hex list for categorical series."""
    return [color(token) for token in _theme()["cycles"]["categorical"]]


def _build_cmap(spec: dict[str, Any]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(spec["name"], list(spec["stops"]))


def sequential_cmap() -> LinearSegmentedColormap:
    """White -> teal -> navy ramp for heatmaps and correlation matrices."""
    return _build_cmap(_theme()["colormaps"]["sequential"])


def diverging_cmap() -> LinearSegmentedColormap:
    """Gold <- white -> teal ramp for SHAP values, residuals and +/- returns."""
    return _build_cmap(_theme()["colormaps"]["diverging"])


# =============================================================================
# Fonts
# =============================================================================
def install_chart_fonts(timeout: int = 20) -> list[str]:
    """Download the OFL chart fonts into ``artifacts/fonts`` and register them.

    Never raises: a missing font must not break a pipeline run. On any failure
    the function logs a warning and returns the families it did manage to
    register, so :func:`apply_theme` falls back to DejaVu Sans.

    Args:
        timeout: Per-file download timeout in seconds.

    Returns:
        Font family names now available to matplotlib.
    """
    _FONT_DIR.mkdir(parents=True, exist_ok=True)
    registered: list[str] = []

    for family, url in _FONT_URLS.items():
        destination = _FONT_DIR / f"{family}.ttf"
        try:
            if not destination.exists():
                LOGGER.info("Downloading %s (SIL OFL 1.1)...", family)
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    destination.write_bytes(response.read())
            mpl.font_manager.fontManager.addfont(str(destination))
            registered.append(family)
        except Exception as exc:  # network off, proxy, 404 - all non-fatal
            LOGGER.warning("Could not install font %s (%s); using fallback.", family, exc)
            if destination.exists() and destination.stat().st_size == 0:
                destination.unlink(missing_ok=True)

    return registered


def _available_families() -> set[str]:
    return {f.name for f in mpl.font_manager.fontManager.ttflist}


# =============================================================================
# The one call that matters
# =============================================================================
def apply_theme(*, download_fonts: bool = True, verbose: bool = False) -> dict[str, Any]:
    """Apply the Mubadala theme to matplotlib globally.

    Call once in the first cell of every notebook, immediately after
    ``seed_everything()``.

    Args:
        download_fonts: Attempt to fetch the OFL chart fonts. Set ``False`` in
            CI or on an air-gapped machine.
        verbose: Log the resolved font stack and palette size.

    Returns:
        The loaded theme dict, so a caller can reach tokens without a second
        YAML read.
    """
    theme = _theme()
    mplcfg = theme["matplotlib"]

    families: list[str] = []
    if download_fonts:
        families = install_chart_fonts()

    available = _available_families()
    chart_fonts = theme["fonts"]["charts"]
    body = chart_fonts["body"]["family"]
    heading = chart_fonts["heading"]["family"]
    fallback = chart_fonts["fallback"]["family"]

    body_stack = [f for f in (body, heading, fallback) if f in available or f == fallback]
    if not body_stack:
        body_stack = [fallback]

    palette = theme["palette"]
    text = color(theme["semantic"]["text"])
    heading_color = color(theme["semantic"]["heading"])

    mpl.rcParams.update(
        {
            # --- fonts -----------------------------------------------------
            "font.family": "sans-serif",
            "font.sans-serif": body_stack + ["DejaVu Sans"],
            "font.size": mplcfg["font_size_base"],
            # --- figure ----------------------------------------------------
            "figure.figsize": tuple(mplcfg["figsize_default"]),
            "figure.dpi": mplcfg["figure_dpi"],
            "figure.facecolor": palette["card"],
            "figure.titlesize": mplcfg["font_size_title"],
            "figure.titleweight": "semibold",
            "savefig.dpi": mplcfg["savefig_dpi"],
            "savefig.facecolor": palette["card"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            # --- axes ------------------------------------------------------
            "axes.facecolor": palette["card"],
            "axes.edgecolor": palette["line"],
            "axes.linewidth": mplcfg["spine_linewidth"],
            "axes.labelcolor": text,
            "axes.labelsize": mplcfg["font_size_label"],
            "axes.titlesize": mplcfg["font_size_title"],
            "axes.titlecolor": heading_color,
            "axes.titleweight": "semibold",
            "axes.titlepad": mplcfg["axes_titlepad"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=categorical_cycle()),
            # --- grid / ticks ---------------------------------------------
            "grid.color": palette["line"],
            "grid.alpha": mplcfg["grid_alpha"],
            "grid.linewidth": mplcfg["grid_linewidth"],
            "xtick.color": text,
            "ytick.color": text,
            "xtick.labelsize": mplcfg["font_size_tick"],
            "ytick.labelsize": mplcfg["font_size_tick"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            # --- legend / text --------------------------------------------
            "legend.frameon": mplcfg["legend_frameon"],
            "legend.fontsize": mplcfg["font_size_tick"],
            "text.color": text,
            # --- lines -----------------------------------------------------
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "patch.edgecolor": palette["card"],
        }
    )

    # Register the two colormaps under their theme names so any library that
    # accepts a cmap *string* (seaborn, SHAP) can use the brand ramps.
    for cmap in (sequential_cmap(), diverging_cmap()):
        try:
            mpl.colormaps.register(cmap, force=True)
        except Exception as exc:  # pragma: no cover - older matplotlib
            LOGGER.debug("Colormap %s not registered: %s", cmap.name, exc)

    if verbose:
        LOGGER.info(
            "Theme applied | fonts=%s | downloaded=%s | palette tokens=%d",
            body_stack, families or "none", len(palette),
        )
    return theme


def save_figure(
    fig: "plt.Figure",
    name: str,
    *,
    directory: Path | str | None = None,
    formats: Sequence[str] = ("png",),
    close: bool = True,
) -> list[Path]:
    """Save a themed figure to ``reports/figures`` with consistent naming.

    Centralised so every figure in the guide and the deck has the same DPI,
    background and padding - and so a figure referenced by D7/D8 always exists
    at a predictable path.

    Args:
        fig: The matplotlib figure.
        name: Base filename without extension, e.g. ``m02_pd_calibration``.
        directory: Override the output directory.
        formats: Extensions to write (``png`` for decks, add ``pdf`` for print).
        close: Close the figure afterwards to free memory in long notebooks.

    Returns:
        The paths written.
    """
    out_dir = Path(directory) if directory else REPO_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        target = out_dir / f"{name}.{extension}"
        fig.savefig(target)
        written.append(target)
    if close:
        plt.close(fig)
    LOGGER.debug("Saved figure(s): %s", [str(p) for p in written])
    return written


def swatch(tokens: Iterable[str] | None = None) -> "plt.Figure":
    """Render the palette as a swatch strip.

    Used once in ``01_setup_and_eda`` to evidence the theme in the guide, and
    as a quick visual check that the tokens loaded.
    """
    names = list(tokens) if tokens else list(get_palette())
    fig, ax = plt.subplots(figsize=(len(names) * 0.9, 1.9))
    for index, token in enumerate(names):
        ax.add_patch(plt.Rectangle((index, 0), 1, 1, color=color(token)))
        ax.text(
            index + 0.5, -0.16, token,
            ha="center", va="top", fontsize=7, rotation=45,
        )
    ax.set_xlim(0, len(names))
    ax.set_ylim(-0.6, 1)
    ax.axis("off")
    ax.set_title("Mubadala theme - palette tokens")
    return fig
