"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for SEAL (Semantic Envelope for
    Admissible Conditioning) — Distribution-Constrained Data Conditioning.
    Searches for minimal, contract-feasible transformations that move the
    empirical distribution into a user-declared admissible envelope Q(theta),
    returning an admissible family of K conditioned DataFrame variants.

Scientific Context:
    SEAL treats conditioning as an epistemic intervention. It searches for
    minimal, contract-feasible transformations T that move the empirical
    distribution into an admissible envelope Q(theta):
      Q(theta) := {P : r_j(P) <= theta_j for all j}
    Distortion is measured by IQR-normalised L1 displacement:
      Omega(T;X) := (1/(n*F)) sum_col sum_i |T(x_i,col) - x_i,col| / IQR(x_col)
    EVI (Envelope Violation Influence) measures per-sample distributional
    influence on the constraint violations via leave-one-out analysis.
    Constraint families: tail_lo, tail_hi, q05_dev, q95_dev.
    Transform families: winsorize (hard clamp), soft_clamp (tanh-smooth).

Invariants:
    - compute_seal always returns a SEALResult with is_valid set correctly.
    - All computation is pure: no Qt calls, no I/O, no side effects.
    - Every numeric literal is a named Final constant with a unit suffix.
    - EVI scores are indexed 0..n-1 matching the original df row positions.
    - transformed_dfs[k] is df.copy() with selected_cols columns replaced.

Assumptions:
    - df contains at least MIN_SEAL_ROWS rows after NaN removal.
    - selected_cols are all present and numeric in df.
    - bounds_list may be empty or partially cover selected_cols.
    - scipy and sklearn are available in the dnv conda environment.

Failure Modes:
    - Fewer than MIN_SEAL_ROWS valid rows: returns SEALResult with is_valid=False.
    - Non-numeric selected_cols: dropped with a logged warning.
    - Zero IQR for a column: distortion for that column treated as zero.
    - All grid candidates violate budget: returns best available (lowest d_after).

Provenance:
    - This module emits no transformation metadata; all provenance is managed
      by the calling GUI layer via DataCleaningWindow.apply_cleaning_operation.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Numeric constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum valid rows after NaN removal [dimensionless].
MIN_SEAL_ROWS: Final[int] = 10

#: Default number of admissible variants to return [dimensionless].
SEAL_N_VARIANTS_DEFAULT: Final[int] = 5

#: Default total distortion budget tau [dimensionless IQR-normalised].
SEAL_DISTORTION_BUDGET_DEFAULT: Final[float] = 0.30

#: Default tail mass fraction tolerance theta_tail [fraction].
SEAL_TAIL_TOL_DEFAULT: Final[float] = 0.05

#: Default quantile deviation tolerance in IQR units [dimensionless].
SEAL_Q_TOL_IQR_DEFAULT: Final[float] = 0.20

#: Default tanh smoothness parameter for soft_clamp [dimensionless].
SEAL_SOFT_SMOOTHNESS_DEFAULT: Final[float] = 0.15

#: Grid of lower percentile clipping points for transform search [percent].
SEAL_GRID_LO_PCT: Final[Tuple[float, ...]] = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)

#: Grid of upper percentile clipping points for transform search [percent].
SEAL_GRID_HI_PCT: Final[Tuple[float, ...]] = (90.0, 92.5, 95.0, 97.0, 98.0, 99.0, 99.5)

#: Row count threshold for exact LOO EVI vs kernel-density approximation [dimensionless].
SEAL_LOO_EXACT_THRESHOLD: Final[int] = 300

#: Weight for upper-tail constraint in distortion measure [dimensionless].
SEAL_WEIGHT_TAIL_HI: Final[float] = 1.0

#: Weight for lower-tail constraint in distortion measure [dimensionless].
SEAL_WEIGHT_TAIL_LO: Final[float] = 1.0

#: Weight for 5th-percentile deviation constraint [dimensionless].
SEAL_WEIGHT_Q05_DEV: Final[float] = 0.5

#: Weight for 95th-percentile deviation constraint [dimensionless].
SEAL_WEIGHT_Q95_DEV: Final[float] = 0.5

#: Minimum IQR value to avoid division by zero [dimensionless].
_MIN_IQR_GUARD: Final[float] = 1e-12

#: Fraction of a column's full range used as a lower floor for its IQR
#: when Q1≈Q3 (near-constant columns). Keeps the scale in _compute_omega
#: physically comparable to the data spread instead of collapsing to
#: _MIN_IQR_GUARD and producing astronomical per-column distortion.
_IQR_RANGE_FLOOR_FRAC: Final[float] = 0.01

#: Minimum kernel density estimate value to avoid division by zero [dimensionless].
_MIN_KDE_GUARD: Final[float] = 1e-12

#: Total number of grid candidates (computed from grid dimensions).
_N_GRID_CANDIDATES: int = len(SEAL_GRID_LO_PCT) * len(SEAL_GRID_HI_PCT)

#: Per-column distortion budget as fraction of global tau [dimensionless].
#: Rejects any candidate where a single column absorbs more than this
#: fraction of the total budget — prevents one extreme column from
#: consuming all admissible distortion.
SEAL_COL_BUDGET_RATIO: Final[float] = 0.60

#: Per-sample displacement cap as fraction of column IQR [dimensionless].
#: Rejects candidates where the 90th-percentile per-sample displacement
#: exceeds this fraction of IQR — prevents feasibility achieved by
#: rewriting a large fraction of observations, while permitting genuine
#: extreme outlier clipping on a small fraction of samples.
SEAL_SAMPLE_BUDGET_IQR_RATIO: Final[float] = 3.0

#: Percentile of per-sample displacements to check against budget [percent].
#: Using the 90th percentile means the top 10% most-displaced samples
#: (typically genuine outliers) are exempt from the per-sample cap.
SEAL_SAMPLE_BUDGET_PERCENTILE_PCT: Final[float] = 90.0

#: Absolute tolerance for hard contract bound enforcement [dimensionless].
#: After transformation, values must satisfy lo - ε ≤ T(x) ≤ hi + ε
#: where ε accounts for floating-point rounding in soft_clamp.
SEAL_BOUND_EPS: Final[float] = 1e-9


# ---------------------------------------------------------------------------
# Data contracts / result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnBoundSpec:
    """Physical bounds declaration for a single column.

    Attributes
    ----------
    col : str
        Column name in the DataFrame.
    lo : float or None
        Physical lower bound; None means unconstrained.
    hi : float or None
        Physical upper bound; None means unconstrained.
    """
    col: str
    lo: Optional[float]  # physical lower bound, None = unconstrained
    hi: Optional[float]  # physical upper bound, None = unconstrained


@dataclass(frozen=True)
class ConstraintResidual:
    """Per-column, per-constraint violation summary before and after best transform.

    Attributes
    ----------
    col : str
        Column name.
    constraint : str
        One of "tail_lo", "tail_hi", "q05_dev", "q95_dev".
    weight : float
        Constraint weight in distortion measure.
    residual_before : float
        Raw residual r_j(P_X) before transform.
    tolerance : float
        Declared tolerance theta_j.
    excess_before : float
        max(0, residual_before - tolerance).
    residual_after_best : float
        Raw residual r_j(P_{T(X)}) after best-variant transform.
    excess_after_best : float
        max(0, residual_after_best - tolerance).
    """
    col: str
    constraint: str       # "tail_lo"|"tail_hi"|"q05_dev"|"q95_dev"
    weight: float
    residual_before: float
    tolerance: float
    excess_before: float       # max(0, residual_before - tolerance)
    residual_after_best: float
    excess_after_best: float   # max(0, residual_after_best - tolerance)


@dataclass(frozen=True)
class SEALVariant:
    """A single admissible transform variant.

    Attributes
    ----------
    variant_id : int
        Zero-based index in the admissible family.
    distortion : float
        Omega(T;X) — IQR-normalised L1 displacement.
    d_after : float
        d(P_{T(X)}, Q(theta)) — weighted constraint violation after transform.
    transform_params : dict
        Per-column transform parameters:
        {col: {"lo_val": float, "hi_val": float, "lo_pct": float, "hi_pct": float}}.
    """
    variant_id: int
    distortion: float           # Omega(T;X)
    d_after: float              # d(P_{T(X)}, Q(theta))
    transform_params: dict      # {col: {"lo_val":float,"hi_val":float,"lo_pct":float,"hi_pct":float}}


@dataclass(frozen=True)
class SEALResult:
    """Complete result record from compute_seal.

    Attributes
    ----------
    selected_cols : tuple of str
        Column names that were conditioned (after dropping non-numeric).
    n_rows : int
        Number of rows after NaN removal used for statistics.
    n_variants : int
        Number of admissible variants returned.
    variants : tuple of SEALVariant
        Sorted by distortion ascending.
    best_variant_idx : int
        Index into variants of the lowest-d_after variant.
    evi_scores : pd.Series
        Per-sample total EVI scores; index 0..original_n_rows-1.
    evi_df : pd.DataFrame
        Per-sample, per-column EVI contributions.
    constraint_residuals : tuple of ConstraintResidual
        One entry per (col, constraint) combination.
    distortion_before : float
        Always 0.0 (identity transform).
    distortion_per_variant : tuple of float
        Omega values for each variant in output order.
    transformed_dfs : tuple of pd.DataFrame
        Full df copies with selected_cols replaced by conditioned values.
    is_valid : bool
        True if at least MIN_SEAL_ROWS rows were available.
    message : str
        Human-readable status or error message.
    """
    selected_cols: Tuple[str, ...]
    n_rows: int
    n_variants: int
    variants: Tuple[SEALVariant, ...]
    best_variant_idx: int        # index of lowest-d_after variant in variants
    evi_scores: object           # pd.Series index=0..n-1
    evi_df: object               # pd.DataFrame rows=samples, cols=selected_cols
    constraint_residuals: Tuple[ConstraintResidual, ...]
    d_before: float              # d(P_X, Q(theta)) — admissibility distance of original data
    distortion_before: float     # always 0 (identity transform has zero displacement)
    distortion_per_variant: Tuple[float, ...]
    transformed_dfs: Tuple[object, ...]  # len=n_variants, each is full df with conditioned cols replaced
    is_valid: bool
    message: str


# ---------------------------------------------------------------------------
# Internal helpers — transforms
# ---------------------------------------------------------------------------

def _apply_winsorize(x: np.ndarray, lo_val: float, hi_val: float) -> np.ndarray:
    """Hard winsorization: clip(x, lo_val, hi_val)."""
    return np.clip(x, lo_val, hi_val)


def _apply_soft_clamp(
    x: np.ndarray,
    lo_val: float,
    hi_val: float,
    smoothness: float,
) -> np.ndarray:
    """Smooth tanh-based clamping that maps all of R -> (lo_val, hi_val).

    Interior values are approximately preserved; exterior values are smoothly
    compressed. If lo_val == hi_val the input is returned unchanged.
    """
    rng = hi_val - lo_val
    if abs(rng) < _MIN_IQR_GUARD:
        return x.copy()
    mid = (lo_val + hi_val) / 2.0
    denom = smoothness * rng
    if abs(denom) < _MIN_IQR_GUARD:
        denom = _MIN_IQR_GUARD
    result = lo_val + rng * (0.5 + 0.5 * np.tanh((x - mid) / denom))
    return result


# ---------------------------------------------------------------------------
# Internal helpers — constraint residuals
# ---------------------------------------------------------------------------

def _compute_residuals_for_col(
    x: np.ndarray,
    lo_bound: Optional[float],
    hi_bound: Optional[float],
    q05_ref: float,
    q95_ref: float,
    iqr: float,
    tail_tol: float,
    q_tol_iqr: float,
) -> Dict[str, float]:
    """Compute the four constraint residuals for a single column array.

    Returns dict with keys 'tail_lo', 'tail_hi', 'q05_dev', 'q95_dev'.
    """
    n = len(x)
    residuals: Dict[str, float] = {}

    # tail_lo: fraction below declared lo_bound
    if lo_bound is not None:
        frac_below = float(np.mean(x < lo_bound))
        residuals["tail_lo"] = frac_below
    else:
        residuals["tail_lo"] = 0.0

    # tail_hi: fraction above declared hi_bound
    if hi_bound is not None:
        frac_above = float(np.mean(x > hi_bound))
        residuals["tail_hi"] = frac_above
    else:
        residuals["tail_hi"] = 0.0

    # q05_dev: |q_0.05(x) - q05_ref| / IQR
    q05_now = float(np.percentile(x, 5.0))
    iqr_safe = max(iqr, _MIN_IQR_GUARD)
    residuals["q05_dev"] = abs(q05_now - q05_ref) / iqr_safe

    # q95_dev: |q_0.95(x) - q95_ref| / IQR
    q95_now = float(np.percentile(x, 95.0))
    residuals["q95_dev"] = abs(q95_now - q95_ref) / iqr_safe

    return residuals


def _compute_d(
    residuals_by_col: Dict[str, Dict[str, float]],
    tail_tol: float,
    q_tol_iqr: float,
) -> float:
    """Weighted L1 distortion: sum_j w_j * max(0, r_j - theta_j)."""
    d = 0.0
    for _col, res in residuals_by_col.items():
        d += SEAL_WEIGHT_TAIL_LO * max(0.0, res["tail_lo"] - tail_tol)
        d += SEAL_WEIGHT_TAIL_HI * max(0.0, res["tail_hi"] - tail_tol)
        d += SEAL_WEIGHT_Q05_DEV * max(0.0, res["q05_dev"] - q_tol_iqr)
        d += SEAL_WEIGHT_Q95_DEV * max(0.0, res["q95_dev"] - q_tol_iqr)
    return d


def _compute_omega(
    x_orig: np.ndarray,
    x_trans: np.ndarray,
    iqr: float,
) -> float:
    """IQR-normalised L1 displacement for a single column."""
    iqr_safe = max(iqr, _MIN_IQR_GUARD)
    return float(np.mean(np.abs(x_trans - x_orig)) / iqr_safe)


def _check_hard_feasibility(
    x_trans: np.ndarray,
    lo_bound: Optional[float],
    hi_bound: Optional[float],
) -> bool:
    """C(T;X) = 0 hard contract check: all values within declared bounds.

    Returns True if the transform satisfies the hard contract, False if
    any value violates declared physical bounds beyond SEAL_BOUND_EPS.
    """
    if lo_bound is not None:
        if float(np.nanmin(x_trans)) < lo_bound - SEAL_BOUND_EPS:
            return False
    if hi_bound is not None:
        if float(np.nanmax(x_trans)) > hi_bound + SEAL_BOUND_EPS:
            return False
    return True


def _check_sample_budget(
    x_orig: np.ndarray,
    x_trans: np.ndarray,
    iqr: float,
) -> bool:
    """Per-sample budget: 90th-percentile displacement ≤ SEAL_SAMPLE_BUDGET_IQR_RATIO * IQR.

    Checks that the bulk of samples (90th percentile of displacement
    distribution) stays within budget.  Genuine extreme outliers in the
    top 10% are permitted to exceed the cap — their clipping is the
    intended SEAL behaviour.  Returns True if budget is satisfied.
    """
    iqr_safe = max(iqr, _MIN_IQR_GUARD)
    displacements = np.abs(x_trans - x_orig)
    p90_disp = float(np.percentile(displacements, SEAL_SAMPLE_BUDGET_PERCENTILE_PCT))
    return p90_disp <= SEAL_SAMPLE_BUDGET_IQR_RATIO * iqr_safe


# ---------------------------------------------------------------------------
# Internal helpers — EVI computation
# ---------------------------------------------------------------------------

def _evi_tail_lo_col(
    x: np.ndarray,
    lo_bound: float,
    tail_tol: float,
) -> np.ndarray:
    """Per-sample EVI contribution for tail_lo constraint on one column.

    Uses exact O(n) LOO formula.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    k_lo = int(np.sum(x < lo_bound))
    r_lo = k_lo / n
    excess_full = max(0.0, r_lo - tail_tol)
    delta = np.zeros(n)
    for i in range(n):
        indicator = 1 if x[i] < lo_bound else 0
        r_loo = (k_lo - indicator) / (n - 1)
        excess_loo = max(0.0, r_loo - tail_tol)
        delta[i] = SEAL_WEIGHT_TAIL_LO * (excess_full - excess_loo)
    return delta


def _evi_tail_hi_col(
    x: np.ndarray,
    hi_bound: float,
    tail_tol: float,
) -> np.ndarray:
    """Per-sample EVI contribution for tail_hi constraint on one column.

    Uses exact O(n) LOO formula.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    k_hi = int(np.sum(x > hi_bound))
    r_hi = k_hi / n
    excess_full = max(0.0, r_hi - tail_tol)
    delta = np.zeros(n)
    for i in range(n):
        indicator = 1 if x[i] > hi_bound else 0
        r_loo = (k_hi - indicator) / (n - 1)
        excess_loo = max(0.0, r_loo - tail_tol)
        delta[i] = SEAL_WEIGHT_TAIL_HI * (excess_full - excess_loo)
    return delta


def _evi_quantile_col_exact(
    x: np.ndarray,
    alpha: float,
    q_ref: float,
    iqr: float,
    q_tol_iqr: float,
    weight: float,
) -> np.ndarray:
    """Exact LOO EVI for a quantile deviation constraint.

    For each sample i, recomputes np.percentile on x without sample i.
    O(n^2) — only used when n <= SEAL_LOO_EXACT_THRESHOLD.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    iqr_safe = max(iqr, _MIN_IQR_GUARD)
    q_now = float(np.percentile(x, alpha * 100.0))
    excess_full = max(0.0, abs(q_now - q_ref) / iqr_safe - q_tol_iqr)
    delta = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        q_loo = float(np.percentile(x[mask], alpha * 100.0))
        excess_loo = max(0.0, abs(q_loo - q_ref) / iqr_safe - q_tol_iqr)
        delta[i] = weight * (excess_full - excess_loo)
        mask[i] = True
    return delta


def _evi_quantile_col_approx(
    x: np.ndarray,
    alpha: float,
    q_ref: float,
    iqr: float,
    q_tol_iqr: float,
    weight: float,
) -> np.ndarray:
    """Jacobian-approximation LOO EVI for a quantile deviation constraint.

    Uses: dq_alpha/dx_i ≈ (I(x_i <= q_alpha) - alpha) / (n * f_hat(q_alpha))
    where f_hat is Gaussian KDE evaluated at q_alpha.
    O(n log n).
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    iqr_safe = max(iqr, _MIN_IQR_GUARD)
    q_now = float(np.percentile(x, alpha * 100.0))
    excess_full = max(0.0, abs(q_now - q_ref) / iqr_safe - q_tol_iqr)

    # Estimate f_hat(q_alpha) via Gaussian KDE
    try:
        kde = gaussian_kde(x)
        f_hat = float(kde(q_now)[0])
    except Exception:
        f_hat = _MIN_KDE_GUARD
    f_hat = max(f_hat, _MIN_KDE_GUARD)

    dq_dxi = (((x <= q_now).astype(float) - alpha) / (n * f_hat))

    # LOO quantile: q_loo ≈ q_now - dq_dxi * (1 contribution removed)
    # The leave-one-out effect on sum is: remove x_i contribution => dq ≈ -dq_dxi * 1
    # q_loo_i ≈ q_now + (q_now - (q_now + dq_dxi * (-1))) for LOO
    # More precisely: q_loo_i ≈ q_now - dq_dxi_i / (1 - 1/n)
    q_loo = q_now - dq_dxi / max(1.0 - 1.0 / n, _MIN_IQR_GUARD)

    sign_full = np.sign(q_now - q_ref) if abs(q_now - q_ref) > _MIN_IQR_GUARD else 0.0
    excess_loo = np.maximum(0.0, np.abs(q_loo - q_ref) / iqr_safe - q_tol_iqr)
    delta = weight * (excess_full - excess_loo)
    return delta


def _compute_evi(
    df_clean: pd.DataFrame,
    selected_cols: List[str],
    bounds_map: Dict[str, ColumnBoundSpec],
    q05_refs: Dict[str, float],
    q95_refs: Dict[str, float],
    iqrs: Dict[str, float],
    tail_tol: float,
    q_tol_iqr: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute per-sample EVI scores for all columns and constraints.

    Returns (evi_scores: pd.Series, evi_df: pd.DataFrame).
    evi_scores is indexed 0..n-1 (aligned to df_clean.index).
    evi_df has columns=selected_cols, rows aligned to df_clean.index.
    """
    n = len(df_clean)
    evi_matrix = np.zeros((n, len(selected_cols)))

    for col_idx, col in enumerate(selected_cols):
        x = df_clean[col].values.astype(float)
        spec = bounds_map.get(col)
        lo_bound = spec.lo if spec is not None else None
        hi_bound = spec.hi if spec is not None else None
        q05_ref = q05_refs[col]
        q95_ref = q95_refs[col]
        iqr = iqrs[col]

        col_delta = np.zeros(n)

        # tail_lo EVI (exact)
        if lo_bound is not None:
            col_delta += _evi_tail_lo_col(x, lo_bound, tail_tol)

        # tail_hi EVI (exact)
        if hi_bound is not None:
            col_delta += _evi_tail_hi_col(x, hi_bound, tail_tol)

        # q05_dev EVI
        if n <= SEAL_LOO_EXACT_THRESHOLD:
            col_delta += _evi_quantile_col_exact(
                x, 0.05, q05_ref, iqr, q_tol_iqr, SEAL_WEIGHT_Q05_DEV
            )
        else:
            col_delta += _evi_quantile_col_approx(
                x, 0.05, q05_ref, iqr, q_tol_iqr, SEAL_WEIGHT_Q05_DEV
            )

        # q95_dev EVI
        if n <= SEAL_LOO_EXACT_THRESHOLD:
            col_delta += _evi_quantile_col_exact(
                x, 0.95, q95_ref, iqr, q_tol_iqr, SEAL_WEIGHT_Q95_DEV
            )
        else:
            col_delta += _evi_quantile_col_approx(
                x, 0.95, q95_ref, iqr, q_tol_iqr, SEAL_WEIGHT_Q95_DEV
            )

        evi_matrix[:, col_idx] = col_delta

    evi_df = pd.DataFrame(evi_matrix, index=df_clean.index, columns=selected_cols)
    evi_scores = evi_df.sum(axis=1)
    evi_scores.name = "evi_total"
    return evi_scores, evi_df


# ---------------------------------------------------------------------------
# Grid search and Pareto-diversity variant selection
# ---------------------------------------------------------------------------

def _apply_transform_col(
    x: np.ndarray,
    lo_pct: float,
    hi_pct: float,
    transform_type: str,
    smoothness: float,
) -> Tuple[np.ndarray, float, float]:
    """Apply a (lo_pct, hi_pct) transform to column values.

    Returns (x_transformed, lo_val, hi_val).
    """
    lo_val = float(np.percentile(x, lo_pct))
    hi_val = float(np.percentile(x, hi_pct))
    if transform_type == "soft_clamp":
        x_t = _apply_soft_clamp(x, lo_val, hi_val, smoothness)
    else:
        x_t = _apply_winsorize(x, lo_val, hi_val)
    return x_t, lo_val, hi_val


def _grid_search(
    df_clean: pd.DataFrame,
    selected_cols: List[str],
    bounds_map: Dict[str, ColumnBoundSpec],
    q05_refs: Dict[str, float],
    q95_refs: Dict[str, float],
    iqrs: Dict[str, float],
    tail_tol: float,
    q_tol_iqr: float,
    distortion_budget: float,
    transform_type: str,
    smoothness: float,
    progress_cb,
) -> List[dict]:
    """Search 7x7=49 (lo_pct, hi_pct) grid candidates with feasibility checks.

    Each candidate is checked against:
    1. C(T;X) = 0  — hard physical-bound feasibility per column
    2. Per-column distortion budget — no single column exceeds
       SEAL_COL_BUDGET_RATIO * tau
    3. Per-sample displacement budget — no single observation displaced
       more than SEAL_SAMPLE_BUDGET_IQR_RATIO * IQR

    Infeasible candidates are included but sorted after all feasible ones.
    Returns list of candidate dicts sorted by (infeasible, d_after, omega).
    Each dict: {lo_pct, hi_pct, d_after, omega, feasible, transform_params}.
    """
    candidates: List[dict] = []
    total = len(SEAL_GRID_LO_PCT) * len(SEAL_GRID_HI_PCT)
    call_idx = 0
    col_budget = distortion_budget * SEAL_COL_BUDGET_RATIO

    x_orig: Dict[str, np.ndarray] = {
        col: df_clean[col].values.astype(float) for col in selected_cols
    }

    for lo_pct in SEAL_GRID_LO_PCT:
        for hi_pct in SEAL_GRID_HI_PCT:
            if progress_cb is not None:
                try:
                    progress_cb(call_idx, total)
                except Exception:
                    pass
            call_idx += 1

            # Apply same (lo_pct, hi_pct) to all selected columns using
            # column-specific percentile values
            residuals_by_col: Dict[str, Dict[str, float]] = {}
            col_omega_vals: List[float] = []
            tparams: Dict[str, dict] = {}
            feasible = True

            for col in selected_cols:
                x = x_orig[col]
                x_t, lo_val, hi_val = _apply_transform_col(
                    x, lo_pct, hi_pct, transform_type, smoothness
                )
                spec = bounds_map.get(col)
                lo_bound = spec.lo if spec is not None else None
                hi_bound = spec.hi if spec is not None else None

                # ── C(T;X) = 0: hard physical-bound check ────────────
                if not _check_hard_feasibility(x_t, lo_bound, hi_bound):
                    feasible = False

                # ── Per-sample displacement budget ────────────────────
                if not _check_sample_budget(x, x_t, iqrs[col]):
                    feasible = False

                res = _compute_residuals_for_col(
                    x_t, lo_bound, hi_bound,
                    q05_refs[col], q95_refs[col], iqrs[col],
                    tail_tol, q_tol_iqr,
                )
                residuals_by_col[col] = res

                omega_col = _compute_omega(x, x_t, iqrs[col])
                col_omega_vals.append(omega_col)

                # ── Per-column distortion budget ──────────────────────
                if omega_col > col_budget:
                    feasible = False

                tparams[col] = {
                    "lo_val": lo_val,
                    "hi_val": hi_val,
                    "lo_pct": lo_pct,
                    "hi_pct": hi_pct,
                }

            d_after = _compute_d(residuals_by_col, tail_tol, q_tol_iqr)
            n_cols = len(selected_cols)
            omega = sum(col_omega_vals) / max(n_cols, 1)

            candidates.append({
                "lo_pct": lo_pct,
                "hi_pct": hi_pct,
                "d_after": d_after,
                "omega": omega,
                "feasible": feasible,
                "transform_params": tparams,
            })

    if progress_cb is not None:
        try:
            progress_cb(total, total)
        except Exception:
            pass

    # Sort: feasible first, then d_after ascending, then omega ascending
    candidates.sort(key=lambda c: (not c["feasible"], c["d_after"], c["omega"]))
    return candidates


def _greedy_diversity_select(
    candidates: List[dict],
    n_variants: int,
) -> List[dict]:
    """Greedy max-coverage selection in (lo_pct, hi_pct) parameter space.

    Among all candidates with the minimum d_after (plus a small tolerance),
    selects K diverse variants by maximising minimum pairwise distance in
    (lo_pct, hi_pct) space. Preferentially includes lowest-d_after.
    """
    if not candidates:
        return []

    # Always include the best candidate (lowest d_after)
    selected = [candidates[0]]
    remaining = list(candidates[1:])

    while len(selected) < n_variants and remaining:
        # For each remaining candidate, compute min distance to any selected
        best_idx = 0
        best_min_dist = -1.0
        for i, cand in enumerate(remaining):
            min_d = min(
                (cand["lo_pct"] - s["lo_pct"]) ** 2 + (cand["hi_pct"] - s["hi_pct"]) ** 2
                for s in selected
            )
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_seal(
    df: pd.DataFrame,
    selected_cols: List[str],
    bounds_list: List[ColumnBoundSpec],
    tail_tol: float = SEAL_TAIL_TOL_DEFAULT,
    q_tol_iqr: float = SEAL_Q_TOL_IQR_DEFAULT,
    distortion_budget: float = SEAL_DISTORTION_BUDGET_DEFAULT,
    transform_type: str = "winsorize",
    soft_smoothness: float = SEAL_SOFT_SMOOTHNESS_DEFAULT,
    n_variants: int = SEAL_N_VARIANTS_DEFAULT,
    rs: int = 42,
    progress_cb=None,
) -> SEALResult:
    """Compute an admissible family of distribution-conditioned DataFrame variants.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Non-selected columns are passed through unchanged.
    selected_cols : list of str
        Columns to condition. Non-numeric entries are dropped with a warning.
    bounds_list : list of ColumnBoundSpec
        Physical bounds declarations; may be empty or partial.
    tail_tol : float
        Fraction tolerance for tail-mass constraints (theta_tail).
    q_tol_iqr : float
        IQR-unit tolerance for quantile-deviation constraints (theta_q).
    distortion_budget : float
        Total distortion budget tau (used as reference for variant selection).
    transform_type : str
        "winsorize" for hard clamp; "soft_clamp" for tanh-smooth.
    soft_smoothness : float
        Smoothness parameter for soft_clamp (ignored for winsorize).
    n_variants : int
        Number of admissible variants to return (K).
    rs : int
        Random seed (reserved for future stochastic extensions).
    progress_cb : callable or None
        Optional callback(current: int, total: int) called for each grid candidate.

    Returns
    -------
    SEALResult
        Complete result record; is_valid=False if fewer than MIN_SEAL_ROWS rows.
    """
    # ── 1. Validate and filter selected_cols ──────────────────────────────
    numeric_cols = set(df.select_dtypes(include="number").columns)
    clean_selected: List[str] = []
    for col in selected_cols:
        if col not in df.columns:
            log.warning("SEAL: column %r not in DataFrame — skipped.", col)
            continue
        if col not in numeric_cols:
            log.warning("SEAL: column %r is non-numeric — skipped.", col)
            continue
        clean_selected.append(col)

    if not clean_selected:
        return SEALResult(
            selected_cols=tuple(selected_cols),
            n_rows=0,
            n_variants=0,
            variants=(),
            best_variant_idx=0,
            evi_scores=pd.Series(dtype=float),
            evi_df=pd.DataFrame(),
            constraint_residuals=(),
            d_before=0.0,
            distortion_before=0.0,
            distortion_per_variant=(),
            transformed_dfs=(),
            is_valid=False,
            message="No valid numeric columns selected.",
        )

    # ── 2. Drop NaN rows in selected_cols for statistics ──────────────────
    df_clean = df.dropna(subset=clean_selected).copy()
    n = len(df_clean)

    if n < MIN_SEAL_ROWS:
        return SEALResult(
            selected_cols=tuple(clean_selected),
            n_rows=n,
            n_variants=0,
            variants=(),
            best_variant_idx=0,
            evi_scores=pd.Series(dtype=float),
            evi_df=pd.DataFrame(),
            constraint_residuals=(),
            d_before=0.0,
            distortion_before=0.0,
            distortion_per_variant=(),
            transformed_dfs=(),
            is_valid=False,
            message=(
                f"Insufficient rows after NaN removal: {n} < {MIN_SEAL_ROWS}."
            ),
        )

    # ── 3. Build bounds map and compute column statistics ─────────────────
    bounds_map: Dict[str, ColumnBoundSpec] = {b.col: b for b in bounds_list}

    q05_refs: Dict[str, float] = {}
    q95_refs: Dict[str, float] = {}
    iqrs: Dict[str, float] = {}

    for col in clean_selected:
        x = df_clean[col].values.astype(float)
        q05_refs[col] = float(np.percentile(x, 5.0))
        q95_refs[col] = float(np.percentile(x, 95.0))
        q25 = float(np.percentile(x, 25.0))
        q75 = float(np.percentile(x, 75.0))
        # Near-constant columns can have IQR≈0 (t_ratio in the spinel set
        # has 35 unique values but Q1=Q3=1.0). _MIN_IQR_GUARD=1e-12 lets
        # _compute_omega explode to 1e8+. Floor by 1% of the full range
        # so the resulting scale is physically meaningful.
        rng_col = float(x.max() - x.min())
        iqrs[col] = max(q75 - q25, rng_col * _IQR_RANGE_FLOOR_FRAC, _MIN_IQR_GUARD)

    # ── 4. Compute baseline residuals (d_before) ──────────────────────────
    residuals_before_by_col: Dict[str, Dict[str, float]] = {}
    for col in clean_selected:
        x = df_clean[col].values.astype(float)
        spec = bounds_map.get(col)
        lo_bound = spec.lo if spec is not None else None
        hi_bound = spec.hi if spec is not None else None
        residuals_before_by_col[col] = _compute_residuals_for_col(
            x, lo_bound, hi_bound,
            q05_refs[col], q95_refs[col], iqrs[col],
            tail_tol, q_tol_iqr,
        )

    d_before = _compute_d(residuals_before_by_col, tail_tol, q_tol_iqr)

    # ── 5. Grid search ────────────────────────────────────────────────────
    candidates = _grid_search(
        df_clean=df_clean,
        selected_cols=clean_selected,
        bounds_map=bounds_map,
        q05_refs=q05_refs,
        q95_refs=q95_refs,
        iqrs=iqrs,
        tail_tol=tail_tol,
        q_tol_iqr=q_tol_iqr,
        distortion_budget=distortion_budget,
        transform_type=transform_type,
        smoothness=soft_smoothness,
        progress_cb=progress_cb,
    )

    # ── 6. Pareto-diverse variant selection ───────────────────────────────
    n_request = max(1, min(n_variants, len(candidates)))
    selected_cands = _greedy_diversity_select(candidates, n_request)

    # ── 7. Build SEALVariant records (sorted by omega) ───────────────────
    selected_cands.sort(key=lambda c: c["omega"])
    seal_variants: List[SEALVariant] = []
    for vid, cand in enumerate(selected_cands):
        seal_variants.append(SEALVariant(
            variant_id=vid,
            distortion=cand["omega"],
            d_after=cand["d_after"],
            transform_params=cand["transform_params"],
        ))

    # ── 8. Find best variant (lowest d_after) ─────────────────────────────
    best_variant_idx = int(
        min(range(len(seal_variants)), key=lambda i: seal_variants[i].d_after)
    )

    # ── 9. Build transformed DataFrames ──────────────────────────────────
    transformed_dfs: List[pd.DataFrame] = []
    for cand in selected_cands:
        df_out = df.copy()
        for col in clean_selected:
            tparam = cand["transform_params"][col]
            lo_val = tparam["lo_val"]
            hi_val = tparam["hi_val"]
            x_orig = df_out[col].values.astype(float)
            nan_mask = np.isnan(x_orig)
            x_valid = x_orig.copy()
            not_nan = ~nan_mask
            if not_nan.any():
                if transform_type == "soft_clamp":
                    x_valid[not_nan] = _apply_soft_clamp(
                        x_orig[not_nan], lo_val, hi_val, soft_smoothness
                    )
                else:
                    x_valid[not_nan] = _apply_winsorize(
                        x_orig[not_nan], lo_val, hi_val
                    )
            x_valid[nan_mask] = np.nan
            df_out[col] = x_valid
        transformed_dfs.append(df_out)

    # ── 10. Compute best-variant residuals for ConstraintResidual records ──
    best_cand = selected_cands[best_variant_idx]
    residuals_after_best_by_col: Dict[str, Dict[str, float]] = {}
    for col in clean_selected:
        tparam = best_cand["transform_params"][col]
        x_orig = df_clean[col].values.astype(float)
        lo_val = tparam["lo_val"]
        hi_val = tparam["hi_val"]
        if transform_type == "soft_clamp":
            x_t = _apply_soft_clamp(x_orig, lo_val, hi_val, soft_smoothness)
        else:
            x_t = _apply_winsorize(x_orig, lo_val, hi_val)
        spec = bounds_map.get(col)
        lo_bound = spec.lo if spec is not None else None
        hi_bound = spec.hi if spec is not None else None
        residuals_after_best_by_col[col] = _compute_residuals_for_col(
            x_t, lo_bound, hi_bound,
            q05_refs[col], q95_refs[col], iqrs[col],
            tail_tol, q_tol_iqr,
        )

    constraint_records: List[ConstraintResidual] = []
    _constraint_names = [
        ("tail_lo",  SEAL_WEIGHT_TAIL_LO, tail_tol),
        ("tail_hi",  SEAL_WEIGHT_TAIL_HI, tail_tol),
        ("q05_dev",  SEAL_WEIGHT_Q05_DEV, q_tol_iqr),
        ("q95_dev",  SEAL_WEIGHT_Q95_DEV, q_tol_iqr),
    ]
    for col in clean_selected:
        for cname, wt, tol in _constraint_names:
            r_before = residuals_before_by_col[col][cname]
            r_after  = residuals_after_best_by_col[col][cname]
            constraint_records.append(ConstraintResidual(
                col=col,
                constraint=cname,
                weight=wt,
                residual_before=r_before,
                tolerance=tol,
                excess_before=max(0.0, r_before - tol),
                residual_after_best=r_after,
                excess_after_best=max(0.0, r_after - tol),
            ))

    # ── 11. Compute EVI ───────────────────────────────────────────────────
    evi_scores, evi_df = _compute_evi(
        df_clean=df_clean,
        selected_cols=clean_selected,
        bounds_map=bounds_map,
        q05_refs=q05_refs,
        q95_refs=q95_refs,
        iqrs=iqrs,
        tail_tol=tail_tol,
        q_tol_iqr=q_tol_iqr,
    )

    # Re-index EVI to sequential 0..n-1 integer index
    evi_scores = evi_scores.reset_index(drop=True)
    evi_df = evi_df.reset_index(drop=True)

    # ── 12. Count feasible variants ─────────────────────────────────────
    n_feasible = sum(1 for c in selected_cands if c.get("feasible", True))

    # ── 13. Assemble SEALResult ───────────────────────────────────────────
    distortion_per_variant = tuple(v.distortion for v in seal_variants)

    message = (
        f"SEAL completed: {len(seal_variants)} variants from {_N_GRID_CANDIDATES} "
        f"candidates ({n_feasible} contract-feasible). "
        f"Best d_after={seal_variants[best_variant_idx].d_after:.4f}, "
        f"d_before={d_before:.4f}."
    )

    return SEALResult(
        selected_cols=tuple(clean_selected),
        n_rows=n,
        n_variants=len(seal_variants),
        variants=tuple(seal_variants),
        best_variant_idx=best_variant_idx,
        evi_scores=evi_scores,
        evi_df=evi_df,
        constraint_residuals=tuple(constraint_records),
        d_before=d_before,
        distortion_before=0.0,
        distortion_per_variant=distortion_per_variant,
        transformed_dfs=tuple(transformed_dfs),
        is_valid=True,
        message=message,
    )
