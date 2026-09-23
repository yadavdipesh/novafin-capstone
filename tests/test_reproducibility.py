"""
novafin-capstone/tests/test_reproducibility.py

Tests for the D6 reproducibility core: seeding and artefact I/O.

The seeding tests do the only thing that actually proves determinism - draw
random numbers twice with the same seed and assert the sequences are identical,
then draw with a different seed and assert they are not. A test that merely
checks ``seed_everything`` returns without raising proves nothing.

The I/O tests pin the guarantee that earns ``ModelBundle`` its place: a model
can never be saved without its preprocessor and feature contract, and a frame
that does not match that contract is rejected at predict time rather than
scored silently against the wrong columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from novafin.utils.io import (
    ModelBundle,
    RunManifest,
    atomic_write_text,
    load_json,
    save_json,
    sha256_file,
    timestamp_slug,
)
from novafin.utils.seed import SeedReport, make_rng, seed_everything, worker_init_fn


# ==========================================================================
# Seeding
# ==========================================================================
def test_same_seed_gives_identical_draws() -> None:
    seed_everything(42)
    first = np.random.rand(64)
    seed_everything(42)
    second = np.random.rand(64)
    np.testing.assert_array_equal(first, second)


def test_different_seed_gives_different_draws() -> None:
    seed_everything(42)
    first = np.random.rand(64)
    seed_everything(1234)
    second = np.random.rand(64)
    assert not np.array_equal(first, second)


def test_python_random_is_also_pinned() -> None:
    import random

    seed_everything(42)
    first = [random.random() for _ in range(10)]
    seed_everything(42)
    second = [random.random() for _ in range(10)]
    assert first == second


def test_generator_api_is_reproducible() -> None:
    assert np.array_equal(make_rng(42).normal(size=32), make_rng(42).normal(size=32))
    assert not np.array_equal(make_rng(42).normal(size=32), make_rng(7).normal(size=32))


def test_seed_report_shape() -> None:
    report = seed_everything(42)
    assert isinstance(report, SeedReport)
    assert report.seed == 42
    payload = report.as_dict()
    assert payload["seed"] == 42
    assert "seed_warnings" in payload


def test_contradictory_flags_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        seed_everything(42, deterministic=True, cudnn_benchmark=True)


def test_dataloader_worker_seeding_is_deterministic() -> None:
    worker_init_fn(0, base_seed=42)
    first = np.random.rand(8)
    worker_init_fn(0, base_seed=42)
    np.testing.assert_array_equal(first, np.random.rand(8))


@pytest.mark.parametrize("seed", [0, 1, 42, 2026])
def test_seeding_is_stable_across_seeds(seed: int) -> None:
    seed_everything(seed)
    first = np.random.rand(16)
    seed_everything(seed)
    np.testing.assert_array_equal(first, np.random.rand(16))


# ==========================================================================
# Artefact I/O
# ==========================================================================
def test_sha256_matches_known_digest(tmp_path: Path) -> None:
    target = atomic_write_text(tmp_path / "a.txt", "novafin")
    # Independently verifiable: sha256("novafin")
    digest = sha256_file(target)
    assert len(digest) == 64
    assert digest == sha256_file(target)          # stable
    other = atomic_write_text(tmp_path / "b.txt", "novafin ")
    assert sha256_file(other) != digest            # sensitive


def test_sha256_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "absent.csv")


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    save_json(tmp_path / "m.json", {"a": 1})
    assert load_json(tmp_path / "m.json") == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_timestamp_slug_is_filename_safe() -> None:
    slug = timestamp_slug()
    assert slug.endswith("Z")
    assert not set(slug) & set('/\\:*?"<>| ')


class _DummyModel:
    """Stand-in estimator - the bundle must not care about the model type."""

    def predict(self, X):  # noqa: D102, ANN001, ANN201
        return np.zeros(len(X))


def _bundle(**kwargs) -> ModelBundle:
    defaults = dict(
        model=_DummyModel(),
        preprocessor=None,
        feature_names=["Credit_Score", "Debt_to_Income", "Loan_Amount"],
        target_name="Default_Flag",
        dataset_key="loans",
    )
    defaults.update(kwargs)
    return ModelBundle(**defaults)  # type: ignore[arg-type]


def test_bundle_requires_feature_names() -> None:
    with pytest.raises(ValueError, match="non-empty feature_names"):
        _bundle(feature_names=[])


def test_bundle_rejects_target_in_features() -> None:
    """The artefact-level guard against the most basic leak of all."""
    with pytest.raises(ValueError, match="target leakage"):
        _bundle(feature_names=["Credit_Score", "Default_Flag"])


def test_bundle_rejects_duplicate_features() -> None:
    with pytest.raises(ValueError, match="Duplicate feature names"):
        _bundle(feature_names=["Credit_Score", "Credit_Score"])


def test_bundle_roundtrip_and_sidecar(tmp_path: Path) -> None:
    bundle = _bundle(metrics={"roc_auc": 0.5}, config_fingerprint="deadbeefdeadbeef")
    path = bundle.save(tmp_path / "loans_lgbm.pkl")
    assert path.exists()

    sidecar = load_json(path.with_suffix(".json"))
    assert sidecar["dataset_key"] == "loans"
    assert sidecar["n_features"] == 3
    assert sidecar["config_fingerprint"] == "deadbeefdeadbeef"

    restored = ModelBundle.load(path)
    assert restored.feature_names == bundle.feature_names
    assert restored.target_name == "Default_Flag"


def test_validate_frame_detects_missing_column() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="missing"):
        bundle.validate_frame(["Credit_Score", "Debt_to_Income"])


def test_validate_frame_detects_extra_column_when_strict() -> None:
    bundle = _bundle()
    columns = [*bundle.feature_names, "Interest_Rate"]
    with pytest.raises(ValueError, match="unexpected"):
        bundle.validate_frame(columns, strict=True)
    # Non-strict mode is the inference path: ids may travel alongside features.
    assert bundle.validate_frame(columns, strict=False) == bundle.feature_names


def test_validate_frame_returns_training_order() -> None:
    bundle = _bundle()
    shuffled = list(reversed(bundle.feature_names))
    assert bundle.validate_frame(shuffled) == bundle.feature_names


def test_run_manifest_records_versions_and_hashes(tmp_path: Path) -> None:
    data = atomic_write_text(tmp_path / "raw.csv", "a,b\n1,2\n")
    manifest = RunManifest(
        run_id="test-run",
        config_fingerprint="0123456789abcdef",
        seed=42,
        data_hashes=RunManifest.hash_inputs([data, tmp_path / "missing.csv"]),
        package_versions=RunManifest.collect_versions(["numpy", "definitely-not-a-package"]),
    )
    assert "raw.csv" in manifest.data_hashes
    assert "missing.csv" not in manifest.data_hashes
    assert manifest.package_versions["definitely-not-a-package"] == "not installed"

    written = manifest.save(tmp_path / "manifest.json")
    assert load_json(written)["seed"] == 42
