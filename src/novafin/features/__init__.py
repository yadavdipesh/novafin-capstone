"""
novafin-capstone/src/novafin/features/__init__.py

Feature layer: build causally, encode without leaking, select honestly.

Order of use, and the guarantee each step adds:

1. :func:`~novafin.features.engineer.build_features` - per-module features,
   every one causal by construction. Forward references exist only in
   :func:`~novafin.features.causal.forward_target`, which builds *targets*.
2. :func:`~novafin.features.causal.assert_causal` - the empirical proof:
   corrupt the future, assert the past is bit-identical.
3. :func:`~novafin.features.encoders.build_preprocessor` and
   :class:`~novafin.features.encoders.OutOfFoldTargetEncoder` - transformations
   whose parameters are fitted on training rows only.
4. :func:`~novafin.features.selection.select_features` - variance, redundancy
   and null-importance filtering, also fitted on training rows only.

The recurring theme: any statistic learned from data - a mean, a rank, a
minimum, a category list, a selected feature set - is a fitted parameter and
must never see the test set.
"""

from __future__ import annotations

from novafin.features.causal import (
    CausalityViolation,
    assert_causal,
    causal_expanding,
    causal_rank,
    causal_rolling,
    causal_shift,
    forward_target,
)
from novafin.features.encoders import (
    CyclicalEncoder,
    FrequencyEncoder,
    OutOfFoldTargetEncoder,
    build_preprocessor,
    split_column_types,
)
from novafin.features.engineer import (
    CUSTOMER_DECISION_COLUMNS,
    FEATURE_BUILDERS,
    PRIMITIVE_FEATURES,
    FeatureResult,
    black_scholes_price,
    build_features,
)
from novafin.features.selection import (
    SelectionReport,
    correlation_filter,
    null_importance_filter,
    select_features,
    variance_filter,
)

__all__ = [
    "CausalityViolation",
    "assert_causal",
    "causal_shift",
    "causal_rolling",
    "causal_expanding",
    "causal_rank",
    "forward_target",
    "FeatureResult",
    "FEATURE_BUILDERS",
    "build_features",
    "black_scholes_price",
    "PRIMITIVE_FEATURES",
    "CUSTOMER_DECISION_COLUMNS",
    "OutOfFoldTargetEncoder",
    "FrequencyEncoder",
    "CyclicalEncoder",
    "build_preprocessor",
    "split_column_types",
    "SelectionReport",
    "select_features",
    "variance_filter",
    "correlation_filter",
    "null_importance_filter",
]
