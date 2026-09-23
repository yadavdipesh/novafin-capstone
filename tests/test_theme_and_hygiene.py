"""
novafin-capstone/tests/test_theme_and_hygiene.py

Two families of guard rails.

**Theme tests** keep the brand honest: every semantic role resolves to a real
hex value, the categorical cycle has no duplicates, the colormaps build, and
the WCAG contrast rule that bans gold-as-text-on-white is actually enforced
rather than merely written down in ``theme.yaml``.

**Repo-hygiene tests** keep the submission honest: the course datasets are
never committed, the licence exists, and every requirement carries a licence
annotation - which is a stated capstone constraint, not a nicety.
"""

from __future__ import annotations

import re

import pytest

from novafin.config import load_theme
from novafin.paths import REPO_ROOT

matplotlib = pytest.importorskip("matplotlib")

from novafin.utils.theme import (  # noqa: E402  (import after importorskip)
    apply_theme,
    categorical_cycle,
    color,
    decision_color,
    diverging_cmap,
    get_palette,
    semantic_color,
    sequential_cmap,
)

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


# ==========================================================================
# Theme
# ==========================================================================
def test_every_palette_token_is_a_valid_hex() -> None:
    for token, value in get_palette().items():
        assert HEX_RE.match(value), f"{token} -> {value!r} is not a 6-digit hex"


def test_core_mubadala_tokens_present() -> None:
    palette = get_palette()
    assert palette["navy"] == "#002B49"
    assert palette["teal"] == "#00B2A9"
    assert palette["gold"] == "#FFC72C"
    assert palette["charcoal"] == "#333F48"


def test_semantic_roles_resolve() -> None:
    for role in (
        "good", "warning", "critical", "neutral", "info", "text", "heading",
        "good_text", "warning_text", "critical_text",
    ):
        assert HEX_RE.match(semantic_color(role))


def test_unknown_token_raises() -> None:
    with pytest.raises(KeyError, match="Unknown palette token"):
        color("mauve")
    with pytest.raises(KeyError, match="Unknown semantic role"):
        semantic_color("fabulous")


def test_hex_passthrough() -> None:
    assert color("#123456") == "#123456"


def test_decision_colors_cover_every_business_label() -> None:
    for label in ("Approve", "Review", "Reject", "Fund", "Protect", "Maintain"):
        assert HEX_RE.match(decision_color(label))
    # Unknown labels degrade to neutral rather than crashing a report build.
    assert decision_color("Brand New Label") == semantic_color("neutral")


def test_categorical_cycle_is_distinct_and_long_enough() -> None:
    cycle = categorical_cycle()
    assert len(cycle) >= 6, "need at least 6 series colours for the panel charts"
    assert len(set(cycle)) == len(cycle), "duplicate colours in the categorical cycle"


def test_colormaps_build_and_are_continuous() -> None:
    for cmap in (sequential_cmap(), diverging_cmap()):
        assert cmap.N > 1
        assert cmap(0.0) != cmap(1.0)


def test_apply_theme_sets_rcparams_without_network() -> None:
    """CI runs offline; the theme must still apply and must not raise."""
    apply_theme(download_fonts=False)
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.grid"] is True
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle[0] == get_palette()["teal"]


def _relative_luminance(hex_color: str) -> float:
    """WCAG 2.1 relative luminance."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    channels = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_body_text_meets_wcag_aa_on_white() -> None:
    minimum = load_theme()["accessibility"]["min_contrast_body"]
    assert _contrast(semantic_color("text"), "#FFFFFF") >= minimum
    assert _contrast(semantic_color("heading"), "#FFFFFF") >= minimum


def test_gold_is_never_used_as_text_on_white() -> None:
    """theme.yaml claims this rule; this test is what makes it true.

    Measured ratios on white: gold #FFC72C = 1.56:1 (fills/markers only),
    dgold #B88F20 = 3.01:1 (WCAG AA *Large* text only), gold_text #7A5B00 =
    6.32:1 (safe for body copy). The first draft of this theme wrongly claimed
    dgold was AA-safe for small text - this test caught it.
    """
    rules = load_theme()["accessibility"]
    palette = get_palette()

    assert _contrast(palette["gold"], "#FFFFFF") < rules["min_contrast_body"]
    assert _contrast(palette["dgold"], "#FFFFFF") < rules["min_contrast_body"], (
        "dgold must not be advertised as safe for small text"
    )
    assert _contrast(palette["dgold"], "#FFFFFF") >= rules["min_contrast_large"], (
        "dgold is the AA-Large risk colour and must clear 3:1"
    )
    assert _contrast(palette["gold_text"], "#FFFFFF") >= rules["min_contrast_body"], (
        "gold_text is the small-text risk colour and must clear 4.5:1"
    )


@pytest.mark.parametrize("role", ["good_text", "warning_text", "critical_text"])
def test_text_safe_semantic_roles_meet_wcag_aa(role: str) -> None:
    """Any role whose name ends in _text must be legible as body copy."""
    minimum = load_theme()["accessibility"]["min_contrast_body"]
    assert _contrast(semantic_color(role), "#FFFFFF") >= minimum, (
        f"semantic role {role!r} is used for text and fails WCAG AA on white"
    )


def test_chart_fonts_are_open_licensed() -> None:
    """Hard constraint: no proprietary font may be used for chart rendering."""
    charts = load_theme()["fonts"]["charts"]
    for role in ("heading", "body"):
        assert "OFL" in charts[role]["licence"], f"chart {role} font must be OFL"
    families = {charts[role]["family"] for role in ("heading", "body", "fallback")}
    assert "Interstate" not in families, (
        "Interstate is proprietary - it may appear only as a font NAME in "
        "Office documents, never in matplotlib"
    )


# ==========================================================================
# Repo hygiene
# ==========================================================================
def test_course_data_is_not_committed() -> None:
    """Academic-integrity guard: the synthetic CSVs must never enter git."""
    committed = [
        p for p in (REPO_ROOT / "data").rglob("*.csv") if ".gitkeep" not in p.name
    ]
    assert not committed, (
        f"Course data found inside the repo: {[p.name for p in committed]}. "
        "These files are licensed for educational use and are not redistributed."
    )


def test_gitignore_blocks_raw_data() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/*" in text
    assert "!data/raw/.gitkeep" in text


def test_licence_file_exists() -> None:
    licence = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in licence
    assert "SIL Open Font License" in licence


def test_every_requirement_is_pinned_and_licensed() -> None:
    """Capstone constraint: state the licence of every library used."""
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [
        line for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    assert requirements, "requirements.txt appears to be empty"
    for line in requirements:
        assert "==" in line, f"unpinned requirement: {line!r}"
        assert "#" in line, f"requirement without a licence annotation: {line!r}"


def test_leakage_register_exists_and_covers_every_critical_finding() -> None:
    """The register is a graded artefact - it must not drift or disappear."""
    text = (REPO_ROOT / "docs" / "LEAKAGE_REGISTER.md").read_text(encoding="utf-8")
    for entry in ("L-01", "L-02", "L-03", "L-04", "L-05", "L-06", "L-07", "L-08"):
        assert entry in text, f"leakage register is missing {entry}"
    for finding in ("N-01", "N-02"):
        assert finding in text, f"leakage register is missing {finding}"
