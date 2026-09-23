"""
novafin-capstone/src/novafin/utils/io.py

Artefact I/O with provenance.

The D6 requirement that drives this module
------------------------------------------
"saved artifacts (model + preprocessor + feature list versioned together)".

That parenthesis is the whole point. The classic failure mode of a student
project is ``model.pkl`` saved on its own: six weeks later nobody can say which
preprocessor it expects, which column order it was trained on, or which config
produced it - so the demo silently scores garbage because column 7 is now
``Interest_Rate`` instead of ``Credit_Score``. :class:`ModelBundle` makes that
impossible by refusing to exist without all three parts, and
:meth:`ModelBundle.validate_frame` re-checks the feature contract at predict
time.

Other things this module guarantees
-----------------------------------
* **Atomic writes** - write to ``<name>.tmp`` then ``os.replace``. A Colab
  session that dies mid-save leaves the previous good artefact intact instead
  of a truncated file that raises ``EOFError`` a week later.
* **SHA-256 data provenance** - every raw CSV is hashed into the run manifest,
  so "which version of the data produced this number?" always has an answer.
* **joblib with a stdlib fallback** - joblib is faster for large numpy arrays;
  ``pickle`` guarantees the code still runs in a minimal environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "sha256_file",
    "atomic_write_bytes",
    "atomic_write_text",
    "save_json",
    "load_json",
    "dump_pickle",
    "load_pickle",
    "ModelBundle",
    "RunManifest",
    "timestamp_slug",
]

LOGGER = logging.getLogger(__name__)

#: Bundle schema version. Bump when the ModelBundle fields change so an old
#: artefact fails fast with a clear message instead of an AttributeError.
BUNDLE_SCHEMA_VERSION = 1

try:  # pragma: no cover - environment dependent
    import joblib

    _HAVE_JOBLIB = True
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]
    _HAVE_JOBLIB = False


# =============================================================================
# Primitives
# =============================================================================
def timestamp_slug(now: datetime | None = None) -> str:
    """UTC timestamp usable in a filename, e.g. ``20260921T103000Z``."""
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256.

    Chunked so the 34 MB order-book CSV (and anything larger later) never has
    to be held in memory twice.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes (default 1 MiB).

    Returns:
        Lower-case hex digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path | str, payload: bytes) -> Path:
    """Write bytes atomically: temp file in the same directory, then replace.

    ``os.replace`` is atomic on POSIX and on Windows for same-volume moves, so
    a reader never observes a half-written artefact.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    """Atomically write text. See :func:`atomic_write_bytes`."""
    return atomic_write_bytes(path, text.encode(encoding))


def save_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Serialise ``payload`` to JSON atomically, coercing Paths to strings."""
    return atomic_write_text(path, json.dumps(payload, indent=indent, default=str))


def load_json(path: Path | str) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_pickle(path: Path | str, obj: Any) -> Path:
    """Persist a Python object, preferring joblib, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if _HAVE_JOBLIB:
        joblib.dump(obj, tmp)
    else:  # pragma: no cover - fallback path
        with tmp.open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, target)
    return target


def load_pickle(path: Path | str) -> Any:
    """Load an object written by :func:`dump_pickle`.

    Security note: pickle executes arbitrary code on load. Only ever load
    artefacts this repository produced - never a file from an untrusted source.
    """
    target = Path(path)
    if _HAVE_JOBLIB:
        return joblib.load(target)
    with target.open("rb") as handle:  # pragma: no cover - fallback path
        return pickle.load(handle)


# =============================================================================
# The versioned artefact bundle
# =============================================================================
@dataclass
class ModelBundle:
    """A trained model, its preprocessor and its feature contract - together.

    Attributes:
        model: The fitted estimator.
        preprocessor: The fitted transformer that produced the training matrix.
            May be ``None`` only for models that genuinely consume raw frames
            (e.g. LightGBM with native categoricals) - but the feature list is
            still mandatory.
        feature_names: Exact column order the model was trained on.
        target_name: Target column, recorded so a bundle can never be pointed
            at the wrong problem.
        dataset_key: Registry key from ``configs/config.yaml``.
        metrics: Validation metrics achieved by this artefact.
        params: Hyper-parameters used.
        config_fingerprint: ``Config.fingerprint()`` at training time.
        data_hashes: ``{filename: sha256}`` of the raw inputs.
        metadata: Free-form extras (threshold, calibration method, CV scheme).
    """

    model: Any
    preprocessor: Any
    feature_names: list[str]
    target_name: str
    dataset_key: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    data_hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = BUNDLE_SCHEMA_VERSION
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform_info: str = field(default_factory=platform.platform)

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError(
                "ModelBundle requires a non-empty feature_names list - an "
                "artefact without its feature contract is not reproducible."
            )
        duplicates = [
            name for name in set(self.feature_names)
            if self.feature_names.count(name) > 1
        ]
        if duplicates:
            raise ValueError(f"Duplicate feature names in bundle: {sorted(duplicates)}")
        if self.target_name in self.feature_names:
            raise ValueError(
                f"Target {self.target_name!r} appears in feature_names - "
                "this is target leakage, refusing to save."
            )

    # -- persistence -------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        """Write the bundle plus a human-readable JSON sidecar.

        The sidecar exists so a reviewer (or a future you) can read the
        metrics, feature list and data hashes without unpickling anything.
        """
        target = Path(path)
        dump_pickle(target, self)
        save_json(target.with_suffix(".json"), self.describe())
        LOGGER.info("Saved bundle %s (%d features)", target, len(self.feature_names))
        return target

    @classmethod
    def load(cls, path: Path | str) -> "ModelBundle":
        """Load a bundle, refusing incompatible schema versions."""
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a ModelBundle (got {type(obj)}).")
        if obj.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Bundle schema {obj.schema_version} != expected "
                f"{BUNDLE_SCHEMA_VERSION}. Retrain, or pin the old code."
            )
        return obj

    # -- inspection --------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """JSON-safe summary (no estimator objects) for the sidecar and MLflow."""
        return {
            "schema_version": self.schema_version,
            "dataset_key": self.dataset_key,
            "target_name": self.target_name,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "params": self.params,
            "config_fingerprint": self.config_fingerprint,
            "data_hashes": self.data_hashes,
            "metadata": self.metadata,
            "model_class": type(self.model).__name__,
            "preprocessor_class": type(self.preprocessor).__name__
            if self.preprocessor is not None
            else None,
            "created_utc": self.created_utc,
            "python_version": self.python_version,
            "platform": self.platform_info,
        }

    # -- the guard that earns the module its keep --------------------------
    def validate_frame(self, columns: Iterable[str], *, strict: bool = True) -> list[str]:
        """Check an inference frame against the training feature contract.

        Args:
            columns: Columns of the frame about to be scored.
            strict: If True, extra columns are also an error. Set False when
                scoring a frame that legitimately carries ids alongside
                features (they are selected out by ``feature_names`` anyway).

        Returns:
            ``self.feature_names`` - the exact order to reindex to.

        Raises:
            ValueError: If any training feature is missing, or (when strict)
                unexpected columns are present.
        """
        present = list(columns)
        missing = [name for name in self.feature_names if name not in present]
        if missing:
            raise ValueError(
                f"Inference frame is missing {len(missing)} training feature(s): "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        if strict:
            extra = [name for name in present if name not in self.feature_names]
            if extra:
                raise ValueError(
                    f"Inference frame has {len(extra)} unexpected column(s): "
                    f"{extra[:10]}{'...' if len(extra) > 10 else ''}"
                )
        return list(self.feature_names)


# =============================================================================
# Run manifest
# =============================================================================
@dataclass
class RunManifest:
    """Everything needed to reproduce one execution of the pipeline.

    Written next to every set of results. Together with the config fingerprint
    and the data hashes it answers the examiner's question "can you show me
    this number came from this data and this code?".
    """

    run_id: str
    config_fingerprint: str
    seed: int
    data_hashes: dict[str, str] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    notes: str = ""
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def collect_versions(packages: Sequence[str]) -> dict[str, str]:
        """Record installed versions of the packages that affect results.

        Uses ``importlib.metadata`` so a package need not be imported (and so a
        missing optional dependency is reported as "not installed" rather than
        raising).
        """
        from importlib import metadata

        out: dict[str, str] = {}
        for name in packages:
            try:
                out[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                out[name] = "not installed"
        return out

    @staticmethod
    def hash_inputs(paths: Iterable[Path | str]) -> dict[str, str]:
        """SHA-256 every existing input file, keyed by filename."""
        out: dict[str, str] = {}
        for item in paths:
            target = Path(item)
            if target.exists() and target.is_file():
                out[target.name] = sha256_file(target)
            else:
                LOGGER.warning("Manifest input not found, skipping: %s", target)
        return out

    def save(self, path: Path | str) -> Path:
        """Write the manifest as JSON."""
        return save_json(path, self.__dict__)
