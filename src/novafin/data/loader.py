"""
novafin-capstone/src/novafin/data/loader.py

The single entry point for reading NovaFin data.

Design rule: **no notebook and no model module ever calls ``pd.read_csv``.**
Everything goes through :func:`load_dataset`, because that is the only place
where the four things that protect the project can be enforced together:

1. **Schema assertion.** Row and column counts are compared against the
   Phase-0 audit values recorded in ``configs/config.yaml``. If the file you
   loaded is not the file that was audited, you find out on line one rather
   than in the results section.
2. **Provenance.** Every load records the file's SHA-256, so a number in the
   report is traceable to a specific byte-for-byte version of the input.
3. **Dtype discipline.** Explicit downcasting to ``float32`` keeps the
   120,000 x 33 order book at roughly half its default footprint, which is what
   keeps the project inside the Colab free-tier envelope.
4. **Causal datetime handling.** Date columns are parsed and the frame is
   sorted by its time key on load, so a downstream ``shift`` or ``rolling`` is
   meaningful rather than accidentally scrambled.

What this module deliberately does NOT do
-----------------------------------------
It does not drop the leaking columns. ``load_dataset`` returns the frame
*exactly as supplied*, because the exploratory notebook must be able to SHOW
the leak (that is a graded exhibit), and because the leaked-vs-clean comparison
in the market and options modules needs both versions. Dropping happens at the
feature-matrix boundary, in :func:`make_feature_frame`, which is the function
every model path calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from novafin.config import Config, DatasetConfig, load_config
from novafin.utils.io import sha256_file

__all__ = [
    "LoadResult",
    "load_dataset",
    "load_all",
    "make_feature_frame",
    "downcast_numeric",
    "memory_mb",
]

LOGGER = logging.getLogger(__name__)

#: Datasets whose ``Timestamp`` column is not ISO-8601. The HFT file uses
#: day-first ``DD-MM-YYYY HH:MM``; parsing it without an explicit format either
#: warns or silently swaps day and month, which would scramble the walk-forward
#: split by trading day. Audited in Phase 0.
_EXPLICIT_DATE_FORMATS: dict[tuple[str, str], str] = {
    ("hft", "Timestamp"): "%d-%m-%Y %H:%M",
    ("hft", "Trading_Day"): "%d-%m-%Y",
}


@dataclass
class LoadResult:
    """A loaded dataset together with everything needed to trust it."""

    key: str
    frame: pd.DataFrame
    path: Path
    sha256: str
    n_rows: int
    n_cols: int
    memory_mb: float
    schema_ok: bool
    schema_notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Flat dict for logging, MLflow params and the run manifest."""
        return {
            "dataset": self.key,
            "rows": self.n_rows,
            "cols": self.n_cols,
            "memory_mb": round(self.memory_mb, 2),
            "sha256_16": self.sha256[:16],
            "schema_ok": self.schema_ok,
            "schema_notes": "; ".join(self.schema_notes) or "none",
        }


# =============================================================================
# Memory helpers
# =============================================================================
def memory_mb(frame: pd.DataFrame) -> float:
    """Deep memory footprint of a frame in MiB (object columns included)."""
    return float(frame.memory_usage(deep=True).sum()) / 1024**2


def downcast_numeric(frame: pd.DataFrame, float_dtype: str = "float32") -> pd.DataFrame:
    """Shrink numeric columns without changing their meaning.

    Floats become ``float32`` (roughly 7 significant digits - ample for prices,
    rates and probabilities here, and the order book's smallest magnitude is
    ~1.5e-4, nowhere near the float32 subnormal range). Integers are narrowed
    to the smallest type that holds their observed range.

    Deliberately **not** applied to identifier columns before they are used as
    keys - ``Customer_ID`` values up to 305,000 fit ``int32`` safely, but the
    caller controls that via the id-column list.

    Args:
        frame: Frame to convert (not modified in place).
        float_dtype: ``"float32"`` or ``"float64"``.

    Returns:
        A new frame with narrowed numeric dtypes.
    """
    if float_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unsupported float_dtype {float_dtype!r}")

    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_float_dtype(series):
            out[column] = series.astype(float_dtype)
        elif pd.api.types.is_integer_dtype(series):
            out[column] = pd.to_numeric(series, downcast="integer")
    return out


# =============================================================================
# Loading
# =============================================================================
def _parse_datetimes(
    frame: pd.DataFrame, spec: DatasetConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Parse declared datetime columns, using an explicit format where needed."""
    notes: list[str] = []
    for column in spec.datetime_columns:
        if column not in frame.columns:
            notes.append(f"declared datetime column '{column}' absent")
            continue
        fmt = _EXPLICIT_DATE_FORMATS.get((spec.key, column))
        try:
            frame[column] = pd.to_datetime(frame[column], format=fmt)
        except (ValueError, TypeError) as exc:
            notes.append(f"could not parse '{column}' as datetime: {exc}")
    return frame, notes


def _sort_causally(
    frame: pd.DataFrame, spec: DatasetConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Sort by the time key (and entity key) so shifts and rollings are valid.

    For the panel datasets the sort is ``(entity, date)`` because every
    trailing feature must be computed *within* a ticker, never across the
    cross-section. For the HFT file, ties on the minute-resolution timestamp
    are broken by the original file order, which is the only ordering
    information the data actually carries (see leakage register N-06).
    """
    notes: list[str] = []
    if not spec.datetime_columns:
        return frame, notes

    time_key = spec.datetime_columns[0]
    if time_key not in frame.columns:
        return frame, notes

    entity_keys = [c for c in spec.id_columns if c in frame.columns]
    if spec.key == "market":
        entity_keys = ["Ticker"]
    elif spec.key == "hft":
        entity_keys = []

    sort_by = entity_keys + [time_key]
    frame = frame.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    notes.append(f"sorted by {sort_by} (stable)")
    return frame, notes


def _check_schema(
    frame: pd.DataFrame, spec: DatasetConfig, strict: bool
) -> tuple[bool, list[str]]:
    """Compare the loaded frame against the Phase-0 audit values."""
    notes: list[str] = []
    ok = True

    if spec.expected_rows is not None and len(frame) != spec.expected_rows:
        ok = False
        notes.append(f"row count {len(frame)} != audited {spec.expected_rows}")
    if spec.expected_cols is not None and frame.shape[1] != spec.expected_cols:
        ok = False
        notes.append(f"column count {frame.shape[1]} != audited {spec.expected_cols}")
    if spec.target and spec.target not in frame.columns:
        ok = False
        notes.append(f"target column '{spec.target}' is absent")

    if spec.positive_rate is not None and spec.target in frame.columns:
        observed = float(pd.to_numeric(frame[spec.target], errors="coerce").mean())
        if not np.isclose(observed, spec.positive_rate, atol=5e-4):
            ok = False
            notes.append(
                f"positive rate {observed:.4f} != audited {spec.positive_rate:.4f}"
            )

    if not ok and strict:
        raise ValueError(
            f"Schema check failed for '{spec.key}' ({spec.filename}): "
            + "; ".join(notes)
            + ". The file on disk is not the file that was audited in Phase 0. "
              "Either restore the original CSV or re-run the audit and update "
              "configs/config.yaml before trusting any result."
        )
    return ok, notes


def load_dataset(
    key: str,
    *,
    cfg: Config | None = None,
    downcast: bool = True,
    strict_schema: bool = True,
    use_cache: bool = True,
    nrows: int | None = None,
) -> LoadResult:
    """Load one registered dataset, validated and provenance-stamped.

    Args:
        key: Registry key from ``configs/config.yaml`` (e.g. ``"loans"``).
        cfg: Project config. Loaded if omitted.
        downcast: Narrow numeric dtypes (see :func:`downcast_numeric`).
        strict_schema: Raise if the file does not match the audited shape.
            Set ``False`` only when deliberately exploring a modified file.
        use_cache: Read/write a parquet copy in ``data/interim``. The order
            book takes several seconds to parse from CSV on a Colab CPU and is
            read by four notebooks; parquet makes reloads effectively free.
        nrows: Read only the first N rows (development convenience). Disables
            caching and schema strictness, since the shape will not match.

    Returns:
        A :class:`LoadResult`.

    Raises:
        FileNotFoundError: If the CSV is absent - with a message pointing at
            ``NOVAFIN_DATA_RAW``.
        ValueError: If ``strict_schema`` and the audit check fails.
    """
    cfg = cfg or load_config()
    spec = cfg.dataset(key)
    path = spec.path(cfg.paths)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{key}' not found at {path}. Set NOVAFIN_DATA_RAW to the "
            "folder holding the NovaFin CSVs (see README quickstart)."
        )

    if nrows is not None:
        use_cache = False
        strict_schema = False

    digest = sha256_file(path)
    cache_path = cfg.paths.data_interim / f"{key}__{digest[:16]}.parquet"

    frame: pd.DataFrame | None = None
    if use_cache and cache_path.exists():
        try:
            frame = pd.read_parquet(cache_path)
            LOGGER.debug("Loaded '%s' from cache %s", key, cache_path.name)
        except Exception as exc:  # corrupt cache must never block a run
            LOGGER.warning("Cache unreadable (%s); re-reading CSV.", exc)
            frame = None

    notes: list[str] = []
    if frame is None:
        frame = pd.read_csv(path, nrows=nrows)
        frame, parse_notes = _parse_datetimes(frame, spec)
        notes.extend(parse_notes)
        if downcast:
            frame = downcast_numeric(frame, cfg.reproducibility.float_dtype)
        frame, sort_notes = _sort_causally(frame, spec)
        notes.extend(sort_notes)
        if use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(cache_path, index=False)
            except Exception as exc:  # pyarrow missing, read-only fs, etc.
                LOGGER.warning("Could not write parquet cache: %s", exc)

    schema_ok, schema_notes = _check_schema(frame, spec, strict_schema)
    notes.extend(schema_notes)

    result = LoadResult(
        key=key,
        frame=frame,
        path=path,
        sha256=digest,
        n_rows=len(frame),
        n_cols=frame.shape[1],
        memory_mb=memory_mb(frame),
        schema_ok=schema_ok,
        schema_notes=notes,
    )

    budget = cfg.compute.max_dataframe_mb
    if result.memory_mb > budget:
        LOGGER.warning(
            "'%s' occupies %.1f MB, above the configured budget of %d MB. "
            "Consider chunking or dropping unused columns.",
            key, result.memory_mb, budget,
        )

    LOGGER.info("Loaded %s", result.summary())
    return result


def load_all(
    keys: Iterable[str] | None = None,
    *,
    cfg: Config | None = None,
    **kwargs: Any,
) -> dict[str, LoadResult]:
    """Load several datasets at once.

    Used by the EDA notebook and by the Module 11 integration step. Failures
    are reported and skipped rather than aborting the batch, so one missing
    file does not hide the status of the other seven.
    """
    cfg = cfg or load_config()
    selected = list(keys) if keys is not None else list(cfg.datasets)
    out: dict[str, LoadResult] = {}
    for key in selected:
        try:
            out[key] = load_dataset(key, cfg=cfg, **kwargs)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.error("Skipping '%s': %s", key, exc)
    return out


# =============================================================================
# The leakage boundary
# =============================================================================
def make_feature_frame(
    frame: pd.DataFrame,
    spec: DatasetConfig,
    *,
    extra_drop: Iterable[str] = (),
    keep: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split a raw frame into a leak-free feature matrix and its target.

    **This is the only sanctioned way to build X.** It removes every column in
    ``DatasetConfig.forbidden_features`` - identifiers, the target, secondary
    and derived targets, and the audited leakage columns - so a model cannot
    see them even if a notebook forgets.

    Args:
        frame: Raw frame from :func:`load_dataset`.
        spec: The dataset's registry entry.
        extra_drop: Additional columns to exclude (e.g. ``Interest_Rate`` for
            the rate-free credit model of leakage-register entry L-06).
        keep: Columns to retain even though they are normally forbidden. Use
            only to build the *deliberately leaked* comparison model, and say
            so in the report - this argument exists to make that exhibit
            explicit rather than accidental.

    Returns:
        ``(X, y)``. ``y`` is ``None`` when the target column is absent, which
        is the inference case.

    Raises:
        ValueError: If the resulting feature matrix would be empty.
    """
    forbidden = set(spec.forbidden_features) | set(extra_drop)
    forbidden -= set(keep)

    feature_columns = [c for c in frame.columns if c not in forbidden]
    if not feature_columns:
        raise ValueError(
            f"No features left for '{spec.key}' after dropping {sorted(forbidden)}."
        )

    X = frame.loc[:, feature_columns].copy()
    y = frame[spec.target].copy() if spec.target in frame.columns else None

    LOGGER.debug(
        "make_feature_frame('%s'): kept %d of %d columns; dropped %s",
        spec.key, len(feature_columns), frame.shape[1], sorted(forbidden & set(frame.columns)),
    )
    return X, y
