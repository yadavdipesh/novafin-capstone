"""
novafin-capstone/src/novafin/data/validate.py

Automated data validation and leakage detection.

The point of this module
------------------------
``docs/LEAKAGE_REGISTER.md`` records what a human found in Phase 0. This module
**rediscovers those findings from the data itself**, which is a much stronger
claim: the controls are not a story told about the dataset, they are the output
of a repeatable procedure. If the course reissues the CSVs with different
generator settings, running :func:`validate_dataset` tells you immediately
whether the register is still accurate.

Four families of check
----------------------
1. **Schema** - shape, dtypes, missingness, duplicates, constant columns.
2. **Identity leakage** - is a column an exact arithmetic function of two
   others? (Catches ``Liquidity_Gap = Expected_Outflows - Expected_Inflows``.)
3. **Target-alias leakage** - is a column a near-perfect predictor, a monotone
   re-encoding, or a same-row derivation of the target? (Catches ``Return``,
   ``Future_Return_100ms``, ``Future_Price_100ms``, ``Black_Scholes_Price``.)
4. **Structural** - class balance, minimum positives for cross-validation,
   temporal ordering, and entity-key overlap between datasets.

Severity vocabulary
-------------------
``CRITICAL`` the column makes the problem trivial; it must be dropped.
``HIGH``     strongly suspicious; requires a documented decision.
``MEDIUM``   a judgement call to be argued in the report (e.g. risk-based
             pricing in ``Interest_Rate``).
``INFO``     a fact worth recording, not a defect.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from novafin.config import Config, DatasetConfig, load_config

__all__ = [
    "Finding",
    "ValidationReport",
    "validate_dataset",
    "validate_all",
    "detect_identity_columns",
    "detect_target_aliases",
    "detect_constant_columns",
    "detect_same_row_derivation",
    "check_entity_overlap",
    "minimum_positives_for_cv",
    "auc_standard_error",
]

LOGGER = logging.getLogger(__name__)

Severity = Literal["CRITICAL", "HIGH", "MEDIUM", "INFO"]

#: |corr| above this against the target is treated as an alias, not a feature.
#: 0.95 is deliberately conservative: the strongest *legitimate* predictor
#: found anywhere in this data is OBI_3Level at 0.513, so nothing honest comes
#: close to the threshold.
ALIAS_CORRELATION_THRESHOLD = 0.95

#: Tolerance for declaring an exact arithmetic identity. Chosen relative to the
#: magnitudes involved: the liquidity columns are in the thousands, so a
#: residual under 0.05 cannot be coincidence.
IDENTITY_ATOL = 0.05

#: A column whose coefficient of variation is below this is effectively a
#: constant and cannot be a useful target (catches Liquidity_Buffer at 0.001).
CONSTANT_CV_THRESHOLD = 0.01


# =============================================================================
# Report objects
# =============================================================================
@dataclass(frozen=True)
class Finding:
    """One validation result."""

    dataset: str
    check: str
    severity: Severity
    message: str
    columns: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    register_id: str | None = None  # e.g. "L-03"

    def __str__(self) -> str:
        tag = f"[{self.register_id}] " if self.register_id else ""
        return f"{self.severity:<8} {self.dataset:<13} {tag}{self.message}"


@dataclass
class ValidationReport:
    """All findings for one dataset."""

    dataset: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "CRITICAL"]

    @property
    def ok(self) -> bool:
        """True when nothing CRITICAL was found."""
        return not self.critical

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def to_frame(self) -> pd.DataFrame:
        """Tabular view for the EDA notebook and the D7 guide appendix."""
        if not self.findings:
            return pd.DataFrame(
                columns=["dataset", "check", "severity", "register_id", "columns", "message"]
            )
        return pd.DataFrame(
            [
                {
                    "dataset": f.dataset,
                    "check": f.check,
                    "severity": f.severity,
                    "register_id": f.register_id or "",
                    "columns": ", ".join(f.columns),
                    "message": f.message,
                }
                for f in self.findings
            ]
        )

    def raise_if_critical(self) -> None:
        """Abort a pipeline run when a CRITICAL leak is present."""
        if self.critical:
            lines = "\n  ".join(str(f) for f in self.critical)
            raise ValueError(
                f"CRITICAL leakage findings for '{self.dataset}':\n  {lines}"
            )

    def __str__(self) -> str:
        if not self.findings:
            return f"{self.dataset}: no findings"
        return "\n".join(str(f) for f in self.findings)


# =============================================================================
# Detectors
# =============================================================================
def detect_constant_columns(
    frame: pd.DataFrame,
    *,
    cv_threshold: float = CONSTANT_CV_THRESHOLD,
    exclude: Iterable[str] = (),
) -> dict[str, float]:
    """Find numeric columns with almost no variance.

    Uses the coefficient of variation (std / |mean|) rather than raw std,
    because raw std is scale-dependent: a std of 11 is enormous for a
    probability and negligible for a balance of 11,385.

    ``exclude`` must be given the identifier columns. Sequential IDs are a
    false positive for this test by construction - ``Customer_ID`` runs
    200001-205000, so its CV is 0.007 purely because the values are large and
    the range is narrow, which says nothing about information content.

    Returns:
        ``{column: coefficient_of_variation}`` for offending columns.
    """
    banned = set(exclude)
    out: dict[str, float] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in banned:
            continue
        series = frame[column].astype("float64")
        mean = abs(series.mean())
        if mean == 0:
            continue
        cv = float(series.std() / mean)
        if cv < cv_threshold:
            out[column] = cv
    return out


def detect_identity_columns(
    frame: pd.DataFrame,
    *,
    atol: float = IDENTITY_ATOL,
    max_columns: int = 20,
    exclude: Iterable[str] = (),
    prefer: Iterable[str] = (),
) -> list[tuple[str, str, str, str, float]]:
    """Find columns that are an exact sum or difference of two others.

    Brute force over ordered pairs, O(k^2) in the number of numeric columns -
    fine for k <= 20 and skipped above that.

    Both orderings of the difference are tested (``a - b`` **and** ``b - a``).
    Missing the reversed form is not a cosmetic bug: on the liquidity file it
    reported ``Expected_Inflows = Expected_Outflows - Liquidity_Gap`` and
    missed the form a reader actually needs,
    ``Liquidity_Gap = Expected_Outflows - Expected_Inflows``. Since every
    rearrangement of one identity is true, ``prefer`` decides which column is
    named as the subject - pass the declared leak columns so the report reads
    the way the register does.

    Args:
        frame: Data to scan.
        atol: Maximum absolute residual to call it an identity.
        max_columns: Skip the scan above this many numeric columns.
        exclude: Columns to ignore entirely (identifiers).
        prefer: Columns to report as the subject when several forms match.

    Returns:
        Tuples ``(subject, operand_a, op, operand_b, max_abs_residual)``,
        de-duplicated so one relationship yields one finding.
    """
    banned = set(exclude)
    preferred = set(prefer)
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=[c for c in banned if c in frame.columns], errors="ignore"
    )
    columns = list(numeric.columns)
    if len(columns) > max_columns:
        LOGGER.debug("Skipping identity scan: %d numeric columns", len(columns))
        return []

    values = {c: numeric[c].astype("float64").to_numpy() for c in columns}
    raw: list[tuple[str, str, str, str, float]] = []

    for subject in columns:
        others = [c for c in columns if c != subject]
        for a, b in itertools.combinations(others, 2):
            candidates = (
                ("-", a, b, values[a] - values[b]),
                ("-", b, a, values[b] - values[a]),
                ("+", a, b, values[a] + values[b]),
            )
            for op, left, right, computed in candidates:
                residual = float(np.nanmax(np.abs(values[subject] - computed)))
                if residual <= atol:
                    raw.append((subject, left, op, right, residual))
                    break

    # One relationship produces several algebraic rearrangements. Keep one per
    # {subject, a, b} set, preferring a subject the register already names.
    best: dict[frozenset[str], tuple[str, str, str, str, float]] = {}
    for hit in raw:
        signature = frozenset({hit[0], hit[1], hit[3]})
        incumbent = best.get(signature)
        if incumbent is None:
            best[signature] = hit
        elif hit[0] in preferred and incumbent[0] not in preferred:
            best[signature] = hit
    return list(best.values())


def detect_label_reconstruction(
    frame: pd.DataFrame,
    target: str,
    column: str,
    *,
    min_agreement: float = 0.85,
) -> float | None:
    """Test whether a numeric column reconstructs a categorical label by cuts.

    Correlation alone under-detects this case. In the order book,
    ``Future_Return_100ms`` correlates only 0.85 with the ordinal encoding of
    ``Price_Move_Class`` - below the 0.95 alias threshold - purely because the
    FLAT band compresses the middle. But two thresholds on that column
    reproduce the label almost exactly, which is the property that matters.

    The cut points are placed at the observed class proportions, which is
    precisely the rule a generator would have used, so no fitting is involved
    and there is nothing to overfit.

    Args:
        frame: Data.
        target: Categorical target column.
        column: Numeric candidate.
        min_agreement: Agreement above which the column is an alias.

    Returns:
        Agreement in ``[0, 1]``, or ``None`` if the test does not apply.
    """
    if target not in frame.columns or column not in frame.columns:
        return None
    if not pd.api.types.is_numeric_dtype(frame[column]):
        return None

    labels = frame[target]
    classes = list(pd.unique(labels.dropna()))
    if not 2 <= len(classes) <= 10:
        return None

    values = frame[column].astype("float64")
    proportions = labels.value_counts(normalize=True).sort_index()
    quantiles = proportions.cumsum().to_numpy()[:-1]
    if len(quantiles) == 0:
        return None

    edges = np.quantile(values.dropna(), quantiles)
    predicted_bin = np.searchsorted(edges, values.to_numpy(), side="right")

    # Map each bin to its most common true label, then measure agreement.
    frame_bins = pd.DataFrame({"bin": predicted_bin, "label": labels.to_numpy()})
    mapping = frame_bins.groupby("bin", observed=True)["label"].agg(
        lambda s: s.value_counts().idxmax()
    )
    reconstructed = frame_bins["bin"].map(mapping)
    if not len(reconstructed):
        return None
    return float((reconstructed.to_numpy() == labels.to_numpy()).mean())


def detect_same_row_derivation(
    frame: pd.DataFrame,
    target: str,
    *,
    group: str | None = None,
    price_column: str = "Close",
) -> float | None:
    """Test whether ``target`` is the same-row percentage change of a price.

    This is the specific check that catches leakage-register entry **L-01**:
    ``Return`` in the market panel is the close-to-close return computed from
    the ``Close`` value on its own row, so a model given same-row OHLC is
    solving an algebra problem.

    Args:
        frame: The panel, already sorted causally by ``(group, date)``.
        target: Candidate target column (``"Return"``).
        group: Entity column to compute within (``"Ticker"``).
        price_column: Price used for the comparison.

    Returns:
        Pearson correlation between ``target`` and the within-group same-row
        percentage change, or ``None`` if the columns are absent.
    """
    if target not in frame.columns or price_column not in frame.columns:
        return None

    prices = frame[price_column].astype("float64")
    if group and group in frame.columns:
        same_row = frame.groupby(group, observed=True)[price_column].pct_change()
    else:
        same_row = prices.pct_change()

    mask = same_row.notna() & frame[target].notna()
    if mask.sum() < 10:
        return None
    return float(np.corrcoef(frame.loc[mask, target].astype("float64"), same_row[mask])[0, 1])


def detect_target_aliases(
    frame: pd.DataFrame,
    target: str,
    *,
    threshold: float = ALIAS_CORRELATION_THRESHOLD,
    exclude: Iterable[str] = (),
) -> dict[str, float]:
    """Find columns whose correlation with the target is implausibly high.

    For a categorical target the column is first encoded by its category codes,
    which is enough to expose an identical re-encoding (the HFT case, where
    ``Price_Move_Class`` is a discretisation of ``Future_Return_100ms``).

    Returns:
        ``{column: correlation}`` sorted by absolute value, descending.
    """
    if target not in frame.columns:
        return {}

    y = frame[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Categorical(y).codes
    y = pd.Series(y, index=frame.index).astype("float64")

    banned = set(exclude) | {target}
    hits: dict[str, float] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in banned:
            continue
        series = frame[column].astype("float64")
        if series.std() == 0:
            continue
        corr = float(np.corrcoef(series, y)[0, 1])
        if np.isfinite(corr) and abs(corr) >= threshold:
            hits[column] = corr
    return dict(sorted(hits.items(), key=lambda kv: abs(kv[1]), reverse=True))


def correlation_with_target(
    frame: pd.DataFrame, target: str, *, top: int = 15
) -> pd.Series:
    """Rank numeric columns by absolute correlation with the target.

    Reported in the EDA notebook for every module: a maximum around 0.1-0.5 is
    a normal, learnable problem; a maximum near 1.0 is a leak; a maximum near
    0.0 is a null result (which is exactly what the churn module shows).
    """
    if target not in frame.columns:
        return pd.Series(dtype="float64")

    y = frame[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.Categorical(y).codes, index=frame.index)

    numeric = frame.select_dtypes(include=[np.number]).astype("float64")
    numeric = numeric.drop(columns=[c for c in [target] if c in numeric.columns])
    corr = numeric.corrwith(y.astype("float64"))
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top)


def auc_standard_error(
    n_positive: float, n_negative: float, *, auc: float = 0.70
) -> float:
    """Approximate standard error of an ROC-AUC estimate.

    Uses the Hanley & McNeil (1982) closed form, which treats the AUC as the
    probability that a random positive outranks a random negative and derives
    its variance from that:

    .. math::

        SE = \\sqrt{\\frac{A(1-A) + (n_p-1)(Q_1 - A^2) + (n_n-1)(Q_2 - A^2)}
                        {n_p \\, n_n}}

    with :math:`Q_1 = A/(2-A)` and :math:`Q_2 = 2A^2/(1+A)`.

    Reference: Hanley JA, McNeil BJ, "The meaning and use of the area under a
    receiver operating characteristic (ROC) curve", *Radiology* 143(1), 1982.
    https://doi.org/10.1148/radiology.143.1.7063747

    Args:
        n_positive: Positives available for scoring (per fold).
        n_negative: Negatives available for scoring (per fold).
        auc: The AUC the estimate is evaluated at. 0.70 is used as a neutral
            reference because the variance is largest near 0.5 and shrinks as
            the AUC approaches 1, so assuming a *perfect* model would flatter
            the power calculation.

    Returns:
        Standard error, or infinity when a fold has fewer than two of a class.
    """
    if n_positive <= 1 or n_negative <= 1:
        return float("inf")
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    numerator = (
        auc * (1 - auc)
        + (n_positive - 1) * (q1 - auc**2)
        + (n_negative - 1) * (q2 - auc**2)
    )
    return float(np.sqrt(numerator / (n_positive * n_negative)))


def minimum_positives_for_cv(
    n_positive: int,
    n_splits: int,
    *,
    n_total: int | None = None,
    max_se: float = 0.05,
) -> tuple[bool, str]:
    """Decide whether a binary target can support the requested folds.

    An earlier version of this function used a "ten positives per fold" rule of
    thumb. That was too lenient to support the conclusion it was being asked to
    justify: with 89 churn positives over 5 folds it returned *pass* at 17.8
    per fold, while the actual standard error of a fold-level AUC at that size
    is about 0.07 - a 95% interval of roughly +/- 0.14, wide enough that two
    genuinely different models are indistinguishable.

    The check is therefore based on :func:`auc_standard_error` instead, which
    is a published closed form rather than a convention, and which correctly
    also flags the 180-row initiatives module.

    Note on repeated CV: repeating the split reduces the variance of the
    *averaged* estimate, but the repeats are correlated (they reuse the same
    rows), so the reduction is less than the naive ``1/sqrt(n_repeats)``. No
    credit is taken for it here - the figure reported is the honest per-fold
    precision, and repeated CV is what makes the *mean* trustworthy, not the
    individual fold.

    Args:
        n_positive: Total positives in the dataset.
        n_splits: Number of cross-validation folds.
        n_total: Total rows, used to derive the negatives per fold. If omitted,
            a 50/50 split is assumed, which is conservative.
        max_se: Largest acceptable per-fold standard error.

    Returns:
        ``(passes, human_readable_message)``.
    """
    if n_splits <= 0:
        return False, "n_splits must be positive"

    positives_per_fold = n_positive / n_splits
    if n_total is not None:
        negatives_per_fold = (n_total - n_positive) / n_splits
    else:
        negatives_per_fold = positives_per_fold

    se = auc_standard_error(positives_per_fold, negatives_per_fold)
    interval = 1.96 * se

    if se > max_se:
        return False, (
            f"{n_positive} positives over {n_splits} folds is "
            f"{positives_per_fold:.1f} per fold; the per-fold AUC standard "
            f"error is {se:.3f} (95% interval +/-{interval:.3f}), above the "
            f"{max_se:.2f} tolerance - single-fold numbers are noise, so "
            "report mean +/- std over repeated CV and state the interval"
        )
    return True, (
        f"{positives_per_fold:.1f} positives per fold; per-fold AUC standard "
        f"error {se:.3f} (95% interval +/-{interval:.3f})"
    )


def check_entity_overlap(
    frames: dict[str, pd.DataFrame], column: str = "Customer_ID"
) -> pd.DataFrame:
    """Cross-tabulate how many entity keys each pair of datasets shares.

    This is what surfaced non-leakage finding **N-02**: the customer file uses
    a disjoint ID range, so no customer-level join is possible and the Module
    11 integration has to happen at portfolio level.
    """
    present = {k: set(v[column].unique()) for k, v in frames.items() if column in v.columns}
    keys = sorted(present)
    matrix = pd.DataFrame(index=keys, columns=keys, dtype="int64")
    for a in keys:
        for b in keys:
            matrix.loc[a, b] = len(present[a] & present[b])
    return matrix


# =============================================================================
# Orchestration
# =============================================================================
def _schema_checks(frame: pd.DataFrame, spec: DatasetConfig, report: ValidationReport) -> None:
    missing = frame.isna().sum()
    offenders = missing[missing > 0]
    if len(offenders):
        report.add(Finding(
            spec.key, "missingness", "HIGH",
            f"{len(offenders)} column(s) contain missing values",
            tuple(offenders.index), {"counts": offenders.to_dict()},
        ))
    else:
        report.add(Finding(spec.key, "missingness", "INFO", "no missing values"))

    duplicates = int(frame.duplicated().sum())
    report.add(Finding(
        spec.key, "duplicates",
        "HIGH" if duplicates else "INFO",
        f"{duplicates} duplicate row(s)",
        evidence={"n_duplicates": duplicates},
    ))

    for column in spec.id_columns:
        if column in frame.columns and frame[column].is_unique:
            report.add(Finding(
                spec.key, "identifier", "INFO",
                f"'{column}' is unique per row and usable as the row key",
                (column,),
            ))


#: Register IDs attached to findings, keyed by dataset.
_REGISTER_BY_DATASET = {
    "market": "L-01",
    "hft": "L-02",
    "liquidity": "L-03",
    "options": "L-04",
    "transactions": "L-05",
    "loans": "L-06",
    "customers": "L-07",
    "initiatives": "L-08",
}


def _handled(spec: DatasetConfig, column: str) -> bool:
    """True when a leak in ``column`` is already accounted for by design.

    Two ways a leak counts as handled: it is in ``drop_always`` (removed from X
    automatically), or it is in ``acknowledged_leaks`` (retained on purpose and
    controlled by the experiment design - the Black-Scholes benchmark is the
    example, since the fair-fight and residual models both need it present in
    the raw frame).
    """
    acknowledged = set(spec.acknowledged_leaks or [])
    return column in set(spec.drop_always) or column in acknowledged


def _key_columns(spec: DatasetConfig) -> list[str]:
    """Identifier-like columns: row ids plus the entity/group key.

    Excluded from the variance and identity scans. A sequential key is a false
    positive for both tests by construction - it has low CV because its values
    are large and narrow, and it can trivially appear in arithmetic identities
    with other keys.
    """
    keys = list(spec.id_columns)
    if spec.group_column:
        keys.append(spec.group_column)
    return keys


def _constant_checks(frame: pd.DataFrame, spec: DatasetConfig, report: ValidationReport) -> None:
    constants = detect_constant_columns(frame, exclude=_key_columns(spec))
    for column, cv in constants.items():
        severity: Severity = (
            "HIGH" if column in {spec.target, spec.derived_target} else "INFO"
        )
        report.add(Finding(
            spec.key, "near_constant", severity,
            f"'{column}' has coefficient of variation {cv:.5f} - effectively constant"
            + (" and therefore unusable as a target"
               if column in {spec.target, spec.derived_target} else ""),
            (column,), {"cv": cv},
            register_id=_REGISTER_BY_DATASET.get(spec.key),
        ))


def _identity_checks(frame: pd.DataFrame, spec: DatasetConfig, report: ValidationReport) -> None:
    hits = detect_identity_columns(
        frame, exclude=_key_columns(spec), prefer=spec.drop_always
    )
    for subject, a, op, b, residual in hits:
        handled = _handled(spec, subject)
        report.add(Finding(
            spec.key, "arithmetic_identity",
            "INFO" if handled else "CRITICAL",
            f"'{subject}' == '{a}' {op} '{b}' (max residual {residual:.4g})"
            + (" - declared, removed from X automatically" if handled else
               " - NOT declared; add it to drop_always before modelling"),
            (subject, a, b), {"max_residual": residual},
            register_id=_REGISTER_BY_DATASET.get(spec.key),
        ))


def _alias_checks(frame: pd.DataFrame, spec: DatasetConfig, report: ValidationReport) -> None:
    register = _REGISTER_BY_DATASET.get(spec.key)

    # --- linear aliases ---------------------------------------------------
    for column, corr in detect_target_aliases(
        frame, spec.target, exclude=spec.id_columns
    ).items():
        handled = _handled(spec, column)
        report.add(Finding(
            spec.key, "target_alias",
            "INFO" if handled else "CRITICAL",
            f"'{column}' correlates {corr:+.4f} with target '{spec.target}'"
            + (" - declared, handled by design" if handled else
               " - NOT declared; this makes the problem trivial"),
            (column,), {"corr": corr}, register_id=register,
        ))

    # --- threshold-rule aliases (catches the categorical case) ------------
    if spec.task == "multiclass_classification" or not pd.api.types.is_numeric_dtype(
        frame.get(spec.target, pd.Series(dtype="float64"))
    ):
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column in set(spec.id_columns):
                continue
            agreement = detect_label_reconstruction(frame, spec.target, column)
            if agreement is None or agreement < 0.85:
                continue
            handled = _handled(spec, column)
            report.add(Finding(
                spec.key, "label_reconstruction",
                "INFO" if handled else "CRITICAL",
                f"thresholds on '{column}' reproduce {agreement:.1%} of "
                f"'{spec.target}' labels"
                + (" - declared, removed from X automatically" if handled else
                   " - NOT declared; this IS the label in another encoding"),
                (column,), {"agreement": agreement}, register_id=register,
            ))

    # --- honest signal strength, excluding everything already removed -----
    ranked = correlation_with_target(
        frame.drop(columns=[c for c in spec.forbidden_features if c in frame.columns],
                   errors="ignore").assign(**{spec.target: frame[spec.target]})
        if spec.target in frame.columns else frame,
        spec.target, top=5,
    )
    if len(ranked):
        strongest = float(ranked.abs().max())
        # When the raw target is itself a declared leak (the market panel), the
        # ranking below is against that leaked column and is NOT the number to
        # quote. Say so, rather than let a plausible-looking figure travel.
        caveat = (
            " - NOTE: measured against the RAW target, which is itself declared "
            "leaky; the figure to quote is computed against the engineered "
            "forward target in Phase 3"
            if spec.target in set(spec.drop_always) else ""
        )
        report.add(Finding(
            spec.key, "signal_strength",
            "HIGH" if strongest < 0.05 else "INFO",
            f"strongest |corr| with target among ADMISSIBLE features is "
            f"{strongest:.4f} ({ranked.abs().idxmax()})"
            + (" - no learnable linear signal; expect a null result"
               if strongest < 0.05 else "")
            + caveat,
            tuple(ranked.index), {"top": ranked.round(4).to_dict()},
            register_id="N-01" if spec.key == "customers" else register,
        ))


def _balance_checks(frame: pd.DataFrame, spec: DatasetConfig, cfg: Config, report: ValidationReport) -> None:
    if spec.task not in {"binary_classification", "multiclass_classification"}:
        return
    if spec.target not in frame.columns:
        return

    counts = frame[spec.target].value_counts()
    report.add(Finding(
        spec.key, "class_balance", "INFO",
        "class distribution " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        (spec.target,), {"counts": counts.to_dict()},
    ))

    if spec.task == "binary_classification":
        n_positive = int(pd.to_numeric(frame[spec.target], errors="coerce").sum())
        n_splits = int(cfg.splits(spec.key).get("n_splits", 5) or 5)
        ok, message = minimum_positives_for_cv(
            n_positive, n_splits, n_total=len(frame)
        )
        report.add(Finding(
            spec.key, "cv_power", "INFO" if ok else "HIGH", message,
            (spec.target,),
            {"n_positive": n_positive, "n_splits": n_splits, "n_total": len(frame)},
            register_id="N-01" if spec.key == "customers" else None,
        ))


def _temporal_checks(frame: pd.DataFrame, spec: DatasetConfig, report: ValidationReport) -> None:
    if not spec.datetime_columns:
        report.add(Finding(
            spec.key, "temporal_structure", "INFO",
            "no date column - out-of-time validation is impossible for this module",
            register_id="N-05" if spec.key == "loans" else None,
        ))
        return

    time_key = spec.datetime_columns[0]
    if time_key not in frame.columns or not pd.api.types.is_datetime64_any_dtype(frame[time_key]):
        return

    series = frame[time_key]
    report.add(Finding(
        spec.key, "temporal_structure", "INFO",
        f"'{time_key}' spans {series.min()} to {series.max()} "
        f"({series.nunique()} unique values over {len(series)} rows)",
        (time_key,),
        {"min": str(series.min()), "max": str(series.max()), "nunique": int(series.nunique())},
        register_id="N-06" if spec.key == "hft" else None,
    ))

    if spec.key == "market":
        corr = detect_same_row_derivation(frame, spec.target, group="Ticker")
        if corr is not None and abs(corr) > ALIAS_CORRELATION_THRESHOLD:
            handled = _handled(spec, spec.target)
            report.add(Finding(
                spec.key, "same_row_derivation",
                "INFO" if handled else "CRITICAL",
                f"'{spec.target}' reproduces the SAME-DAY close-to-close return "
                f"(corr {corr:+.4f}) - it is not a forecastable quantity; the "
                "modelling target is constructed as groupby(Ticker).shift(-1)"
                + (" - declared, removed from X automatically" if handled else
                   " - NOT declared; add it to drop_always before modelling"),
                (spec.target, "Close"), {"corr": corr}, register_id="L-01",
            ))


def validate_dataset(
    key: str,
    frame: pd.DataFrame,
    *,
    cfg: Config | None = None,
) -> ValidationReport:
    """Run every check against one dataset.

    Args:
        key: Registry key.
        frame: The raw frame as returned by ``load_dataset``.
        cfg: Project config.

    Returns:
        A :class:`ValidationReport`. Findings already declared in
        ``drop_always`` are downgraded to ``INFO`` - the register is the
        record of decisions already taken; ``CRITICAL`` is reserved for leaks
        nobody has accounted for yet.
    """
    cfg = cfg or load_config()
    spec = cfg.dataset(key)
    report = ValidationReport(dataset=key)

    _schema_checks(frame, spec, report)
    _constant_checks(frame, spec, report)
    _identity_checks(frame, spec, report)
    _alias_checks(frame, spec, report)
    _balance_checks(frame, spec, cfg, report)
    _temporal_checks(frame, spec, report)

    LOGGER.info(
        "Validated '%s': %d findings (%d critical)",
        key, len(report.findings), len(report.critical),
    )
    return report


def validate_all(
    frames: dict[str, pd.DataFrame], *, cfg: Config | None = None
) -> dict[str, ValidationReport]:
    """Validate several datasets and return their reports."""
    cfg = cfg or load_config()
    return {key: validate_dataset(key, frame, cfg=cfg) for key, frame in frames.items()}


def combined_report_frame(reports: Sequence[ValidationReport]) -> pd.DataFrame:
    """Stack several reports into one table for the guide appendix."""
    parts = [r.to_frame() for r in reports if len(r.findings)]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
