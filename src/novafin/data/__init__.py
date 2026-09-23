"""
novafin-capstone/src/novafin/data/__init__.py

Data access layer: load -> validate -> split, leak-safe by construction.

The intended call order, and the guarantee each step adds:

1. :func:`~novafin.data.loader.load_dataset` - schema assertion against the
   Phase-0 audit, SHA-256 provenance, dtype downcasting, causal sorting.
2. :func:`~novafin.data.validate.validate_dataset` - rediscovers the leakage
   register from the data itself and will not pass silently on a CRITICAL
   finding.
3. :func:`~novafin.data.loader.make_feature_frame` - the leakage boundary; the
   only sanctioned way to build X.
4. :func:`~novafin.data.splits.make_splitter` - the module's configured
   cross-validation scheme, never a default.

Nothing downstream of step 3 can see a forbidden column, and nothing downstream
of step 4 can shuffle a time series.
"""

from __future__ import annotations

from novafin.data.loader import (
    LoadResult,
    downcast_numeric,
    load_all,
    load_dataset,
    make_feature_frame,
    memory_mb,
)
from novafin.data.profile import (
    class_balance,
    column_profile,
    dataset_overview,
    decile_lift,
    group_rate,
    numeric_summary,
    temporal_rate,
)
from novafin.data.splits import (
    ExpandingWindowSplit,
    PurgedWalkForwardSplit,
    TimeSeriesHoldout,
    WalkForwardByGroupSplit,
    assert_no_group_overlap,
    assert_no_temporal_overlap,
    describe_splits,
    holdout_by_time,
    make_splitter,
)
from novafin.data.validate import (
    Finding,
    ValidationReport,
    check_entity_overlap,
    correlation_with_target,
    detect_constant_columns,
    detect_identity_columns,
    detect_same_row_derivation,
    detect_target_aliases,
    validate_all,
    validate_dataset,
)

__all__ = [
    # loader
    "LoadResult",
    "load_dataset",
    "load_all",
    "make_feature_frame",
    "downcast_numeric",
    "memory_mb",
    # validate
    "Finding",
    "ValidationReport",
    "validate_dataset",
    "validate_all",
    "detect_identity_columns",
    "detect_target_aliases",
    "detect_constant_columns",
    "detect_same_row_derivation",
    "correlation_with_target",
    "check_entity_overlap",
    # splits
    "PurgedWalkForwardSplit",
    "WalkForwardByGroupSplit",
    "ExpandingWindowSplit",
    "TimeSeriesHoldout",
    "make_splitter",
    "holdout_by_time",
    "assert_no_temporal_overlap",
    "assert_no_group_overlap",
    "describe_splits",
    # profile
    "column_profile",
    "dataset_overview",
    "class_balance",
    "decile_lift",
    "group_rate",
    "temporal_rate",
    "numeric_summary",
]
