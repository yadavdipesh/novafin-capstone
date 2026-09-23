"""
novafin-capstone/tests/test_config.py

Contract tests for the configuration layer.

These are not decorative. Each one pins a property the rest of the project
silently relies on:

* the YAML actually parses and every required section exists;
* the Phase-0 audit findings survive in machine-readable form, so a future
  edit cannot quietly delete a `drop_always` rule;
* the config fingerprint is stable (same input -> same hash) and sensitive
  (changed input -> different hash), which is what makes a reported metric
  traceable to a configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from novafin.config import Config, DatasetConfig, load_config, load_theme
from novafin.paths import REPO_ROOT, find_repo_root, resolve


@pytest.fixture(scope="module")
def cfg() -> Config:
    """The real project configuration, loaded once per module."""
    return load_config()


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def test_config_loads_and_is_frozen(cfg: Config) -> None:
    assert isinstance(cfg, Config)
    with pytest.raises(Exception):
        cfg.reproducibility.seed = 7  # type: ignore[misc]


def test_seed_is_forty_two(cfg: Config) -> None:
    """The dataset README specifies random seed 42; we match it deliberately."""
    assert cfg.reproducibility.seed == 42


def test_all_eight_datasets_registered(cfg: Config) -> None:
    expected = {
        "initiatives", "loans", "transactions", "customers",
        "market", "liquidity", "options", "hft",
    }
    assert set(cfg.datasets) == expected


def test_every_dataset_has_a_validation_recipe(cfg: Config) -> None:
    """No module may fall back to a default split - each needs its own scheme."""
    for key in cfg.datasets:
        recipe = cfg.splits(key)
        assert recipe.scheme, f"{key} has no validation scheme"


def test_paths_are_absolute(cfg: Config) -> None:
    assert cfg.paths.data_raw.is_absolute()
    assert cfg.paths.artifacts.is_absolute()


def test_unknown_dataset_raises_helpfully(cfg: Config) -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        cfg.dataset("does_not_exist")


# --------------------------------------------------------------------------
# Phase-0 audit findings must survive in config
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dataset_key", "must_drop"),
    [
        ("market", {"Open", "High", "Low", "Close", "Return"}),   # L-01
        ("hft", {"Future_Return_100ms", "Future_Price_100ms"}),   # L-02
        ("liquidity", {"Liquidity_Gap"}),                         # L-03
    ],
)
def test_leakage_rules_are_declared(
    cfg: Config, dataset_key: str, must_drop: set[str]
) -> None:
    """Leakage register entries L-01..L-03 are encoded, not just documented."""
    declared = set(cfg.dataset(dataset_key).drop_always)
    assert must_drop.issubset(declared), (
        f"{dataset_key}: leakage register requires {sorted(must_drop - declared)} "
        "in drop_always - see docs/LEAKAGE_REGISTER.md"
    )


def test_forbidden_features_include_target_and_ids(cfg: Config) -> None:
    ds: DatasetConfig = cfg.dataset("loans")
    forbidden = ds.forbidden_features
    assert ds.target in forbidden
    assert "Customer_ID" in forbidden


def test_audited_shapes_are_recorded(cfg: Config) -> None:
    """Row/column counts from the Phase-0 audit, used as load-time assertions."""
    assert cfg.dataset("initiatives").expected_rows == 180
    assert cfg.dataset("loans").expected_rows == 5000
    assert cfg.dataset("transactions").expected_rows == 30000
    assert cfg.dataset("customers").expected_rows == 5000
    assert cfg.dataset("market").expected_rows == 21855
    assert cfg.dataset("liquidity").expected_rows == 1216
    assert cfg.dataset("options").expected_rows == 10000
    assert cfg.dataset("hft").expected_rows == 120000


def test_finance_assumptions_present(cfg: Config) -> None:
    """Every number an examiner may challenge must live in config, not in code."""
    assert cfg.fin("credit", "lgd") == 0.40
    assert cfg.fin("fraud", "cost_missed_fraud_inr") == 10000
    assert cfg.fin("fraud", "cost_false_positive_inr") == 500
    assert cfg.fin("capital_allocation", "total_capital_crore") == 1000
    assert cfg.fin("churn", "contact_budget") == 1000
    assert cfg.fin("nope", "missing", default="fallback") == "fallback"


# --------------------------------------------------------------------------
# Fingerprint behaviour
# --------------------------------------------------------------------------
def test_fingerprint_is_stable_and_sensitive(cfg: Config) -> None:
    baseline = cfg.fingerprint()
    assert baseline == load_config().fingerprint()
    assert len(baseline) == 16

    changed = load_config(overrides={"reproducibility": {"seed": 1234}})
    assert changed.fingerprint() != baseline


def test_contradictory_determinism_flags_are_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        load_config(
            overrides={
                "reproducibility": {
                    "seed": 42,
                    "deterministic": True,
                    "cudnn_benchmark": True,
                }
            }
        )


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------
# Paths helper
# --------------------------------------------------------------------------
def test_repo_root_discovery_is_cwd_independent() -> None:
    assert find_repo_root() == REPO_ROOT
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert (REPO_ROOT / "configs" / "config.yaml").exists()


def test_resolve_passes_absolute_through(tmp_path: Path) -> None:
    assert resolve(tmp_path) == tmp_path
    assert resolve("configs") == REPO_ROOT / "configs"


# --------------------------------------------------------------------------
# Theme file
# --------------------------------------------------------------------------
def test_theme_yaml_has_required_blocks() -> None:
    theme = load_theme()
    for block in ("palette", "semantic", "cycles", "colormaps", "fonts", "matplotlib"):
        assert block in theme, f"theme.yaml missing '{block}'"
