"""
novafin-capstone/src/novafin/features/causal.py

Causality primitives - and the proof that they work.

The rule every feature in this project obeys
--------------------------------------------
A feature for row *t* may use information from rows *< t* only. Never row *t*'s
own future, never row *t+1*, and - in a panel - never another entity's row on
the same date unless that information would genuinely have been published.

In pandas this reduces to one habit: **shift before you roll.**

    # WRONG - the window includes row t itself
    df.groupby("Customer_ID")["Amount"].rolling(10).mean()

    # RIGHT - row t sees only rows strictly before it
    df.groupby("Customer_ID")["Amount"].shift(1).rolling(10).mean()

The wrong version is the single most common leak in applied tabular work, and
it is invisible: the code looks reasonable, the metrics look excellent, and the
model collapses in production.

Why a perturbation test rather than code review
-----------------------------------------------
Reading code cannot prove absence of look-ahead. :func:`assert_causal` proves
it empirically and is the strongest validation artefact in this repository:

1. build features on a frame,
2. **corrupt the last k rows of the raw inputs** (multiply by 1000, negate,
   shuffle),
3. rebuild features,
4. assert every feature value in the untouched prefix is *bit-identical*.

If any feature peeked forward, its earlier values would move. Nothing survives
that test by accident, and it catches leaks that no amount of reading would.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "causal_shift",
    "causal_rolling",
    "causal_expanding",
    "causal_rank",
    "forward_target",
    "assert_causal",
    "CausalityViolation",
]

LOGGER = logging.getLogger(__name__)


class CausalityViolation(AssertionError):
    """Raised when a feature builder is shown to use future information."""


# =============================================================================
# Primitives
# =============================================================================
def causal_shift(
    frame: pd.DataFrame, column: str, *, by: str | Sequence[str] | None = None, periods: int = 1
) -> pd.Series:
    """Value of ``column`` ``periods`` rows earlier, within each group.

    Args:
        frame: Source frame, already sorted causally by the loader.
        column: Column to shift.
        by: Group key(s). ``None`` treats the frame as one series.
        periods: How far back to look. Must be >= 1; a zero or negative shift
            would expose the present or the future.

    Raises:
        ValueError: If ``periods < 1``.
    """
    if periods < 1:
        raise ValueError(
            f"causal_shift requires periods >= 1 (got {periods}); a shift of 0 "
            "exposes row t to itself and a negative shift exposes the future."
        )
    if by is None:
        return frame[column].shift(periods)
    return frame.groupby(by, observed=True)[column].shift(periods)


def causal_rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    stat: str = "mean",
    *,
    by: str | Sequence[str] | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """Rolling statistic over the ``window`` rows **strictly before** each row.

    Implemented as ``shift(1)`` then ``rolling(window)`` - in that order. The
    reverse order silently includes the current row.

    Args:
        frame: Source frame, causally sorted.
        column: Column to aggregate.
        window: Window length in rows.
        stat: ``mean``, ``std``, ``sum``, ``min``, ``max``, ``median``, ``count``.
        by: Group key(s) to compute within.
        min_periods: Minimum observations before a value is produced. 1 keeps
            early rows usable; the resulting warm-up bias is documented rather
            than hidden by dropping rows.

    Returns:
        A series aligned to ``frame.index``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1 (got {window})")

    def _apply(series: pd.Series) -> pd.Series:
        rolled = series.shift(1).rolling(window, min_periods=min_periods)
        if not hasattr(rolled, stat):
            raise ValueError(f"Unsupported rolling statistic {stat!r}")
        return getattr(rolled, stat)()

    if by is None:
        return _apply(frame[column])
    return frame.groupby(by, observed=True)[column].transform(_apply)


def causal_expanding(
    frame: pd.DataFrame,
    column: str,
    stat: str = "mean",
    *,
    by: str | Sequence[str] | None = None,
    min_periods: int = 1,
) -> pd.Series:
    """Expanding statistic over all rows **strictly before** each row.

    The natural choice for customer-level history: at transaction *t* the bank
    knows everything that customer has done up to *t-1* and nothing after.
    """

    def _apply(series: pd.Series) -> pd.Series:
        expanded = series.shift(1).expanding(min_periods=min_periods)
        if not hasattr(expanded, stat):
            raise ValueError(f"Unsupported expanding statistic {stat!r}")
        return getattr(expanded, stat)()

    if by is None:
        return _apply(frame[column])
    return frame.groupby(by, observed=True)[column].transform(_apply)


def causal_rank(frame: pd.DataFrame, column: str, *, within: str) -> pd.Series:
    """Cross-sectional percentile rank of ``column`` within each ``within`` group.

    This one is causal **by convention, not by construction**, and the
    distinction matters enough to state plainly: ranking all 15 tickers on a
    date uses contemporaneous information from the other 14. That is legitimate
    here because the equity strategy rebalances *after* the close using that
    day's published prices for every name - the information is genuinely
    available at decision time.

    It would NOT be legitimate for the intraday order book, where other stocks'
    later ticks are not observable. No cross-sectional rank is used there.
    """
    return frame.groupby(within, observed=True)[column].rank(pct=True)


def forward_target(
    frame: pd.DataFrame, column: str, *, by: str | Sequence[str] | None = None, horizon: int = 1
) -> pd.Series:
    """The value of ``column`` ``horizon`` rows ahead - the prediction target.

    This is the ONLY function in the project permitted to look forward, and it
    is used exclusively to construct *targets*, never features. Keeping the
    forward shift in one named place means a reviewer can grep for it and
    confirm there is no other forward reference in the codebase.

    Args:
        frame: Causally sorted frame.
        column: Column to look ahead in.
        by: Group key(s) - essential for a panel, or the target bleeds across
            entities at their boundaries.
        horizon: Steps ahead. Must be >= 1.

    Returns:
        The forward series, with ``horizon`` NaNs at the end of each group.
    """
    if horizon < 1:
        raise ValueError(f"forward_target requires horizon >= 1 (got {horizon})")
    if by is None:
        return frame[column].shift(-horizon)
    return frame.groupby(by, observed=True)[column].shift(-horizon)


# =============================================================================
# The proof
# =============================================================================
def assert_causal(
    builder: Callable[[pd.DataFrame], pd.DataFrame],
    frame: pd.DataFrame,
    *,
    perturb_columns: Iterable[str],
    tail_fraction: float = 0.20,
    feature_columns: Sequence[str] | None = None,
    ignore_columns: Iterable[str] = (),
    seed: int = 42,
) -> pd.DataFrame:
    """Prove empirically that ``builder`` uses no future information.

    Builds features twice - once on the original frame, once on a frame whose
    final ``tail_fraction`` of rows has been corrupted - and asserts that every
    feature value in the untouched prefix is identical.

    Corruption is aggressive on purpose (sign flip, x1000, plus noise), so any
    dependence on future rows produces a visible difference rather than a
    rounding-level one.

    Args:
        builder: Function mapping a raw frame to a feature frame. Must preserve
            row order and the original index for the prefix.
        frame: A causally sorted raw frame.
        perturb_columns: Raw columns to corrupt. Choose the ones the features
            are actually derived from.
        tail_fraction: Share of trailing rows to corrupt.
        feature_columns: Restrict the comparison to these columns. Defaults to
            every column the builder added.
        ignore_columns: Columns exempt from the check - the forward *target*
            belongs here, since it is supposed to look ahead.
        seed: Seed for the corruption noise.

    Returns:
        A frame of offending columns and the first index where they differ.
        Empty when the builder is causal.

    Raises:
        CausalityViolation: If any feature in the prefix changed.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    cut = int(n * (1 - tail_fraction))
    if cut < 10:
        raise ValueError("Frame too small for a meaningful causality test.")

    corrupted = frame.copy()
    for column in perturb_columns:
        if column not in corrupted.columns:
            continue

        if pd.api.types.is_datetime64_any_dtype(corrupted[column]):
            # Shifting timestamps far into the future is the natural corruption
            # for a time column and keeps the dtype valid.
            values = corrupted[column].to_numpy(copy=True)
            values[cut:] = values[cut:] + np.timedelta64(365 * 5, "D")
            corrupted[column] = values
        elif pd.api.types.is_float_dtype(corrupted[column]):
            # The corrupted column MUST keep its original dtype. The loader
            # downcasts to float32; if the perturbed frame carried float64
            # instead, every downstream arithmetic op would be evaluated at a
            # different precision and the prefix would differ in the last bits
            # - which the exact comparison below would report as look-ahead.
            # That is a false positive, and chasing it would mask a real one.
            original_dtype = corrupted[column].dtype
            values = corrupted[column].to_numpy(dtype="float64", copy=True)
            tail = values[cut:]
            scale = float(abs(np.nanmean(tail))) + 1.0
            values[cut:] = -tail * 1000.0 + rng.normal(0, scale, size=len(tail))
            corrupted[column] = values.astype(original_dtype)

        elif pd.api.types.is_integer_dtype(corrupted[column]):
            # Integers are permuted rather than rescaled, which changes the
            # future without overflowing a narrow dtype such as int8.
            values = corrupted[column].to_numpy(copy=True)
            tail = values[cut:].copy()
            rng.shuffle(tail)
            values[cut:] = tail[::-1]
            corrupted[column] = values
        else:
            values = corrupted[column].to_numpy(copy=True)
            tail = values[cut:].copy()
            rng.shuffle(tail)
            values[cut:] = tail
            corrupted[column] = values

    original = builder(frame)
    perturbed = builder(corrupted)

    added = [c for c in original.columns if c not in frame.columns]
    columns = list(feature_columns) if feature_columns else added
    columns = [c for c in columns if c not in set(ignore_columns) and c in perturbed.columns]

    offenders: list[dict[str, object]] = []
    for column in columns:
        left = original[column].iloc[:cut]
        right = perturbed[column].iloc[:cut]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            same = np.isclose(
                left.to_numpy(dtype="float64"),
                right.to_numpy(dtype="float64"),
                rtol=0, atol=0, equal_nan=True,
            )
        else:
            same = (left.to_numpy() == right.to_numpy()) | (
                left.isna().to_numpy() & right.isna().to_numpy()
            )
        if not same.all():
            first_bad = int(np.argmax(~same))
            offenders.append(
                {
                    "feature": column,
                    "n_differing": int((~same).sum()),
                    "first_differing_row": first_bad,
                    "original": left.iloc[first_bad],
                    "perturbed": right.iloc[first_bad],
                }
            )

    report = pd.DataFrame(offenders)
    if not report.empty:
        raise CausalityViolation(
            f"{len(report)} feature(s) changed when only FUTURE rows were "
            f"corrupted - they look ahead:\n{report.to_string(index=False)}"
        )

    LOGGER.info(
        "Causality proof passed: %d feature(s) unaffected by corrupting the "
        "final %.0f%% of rows.", len(columns), tail_fraction * 100,
    )
    return report
