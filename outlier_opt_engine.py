"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for moment-optimized outlier
    conditioning — finding the minimal sample exclusion mask that brings
    the empirical kurtosis and skewness of selected columns closest to
    declared target values.

Scientific Context:
    Standard outlier methods (IQR, z-score) apply fixed-threshold rules.
    This module turns the threshold into a decision variable: given a base
    outlier method, it optimizes the threshold (Layer 1) or the individual
    sample inclusion mask (Layer 2) to minimize a distributional objective:

        L(mask) = w_k · (κ(x[mask]) − κ_target)²
                + w_s · (γ(x[mask]) − γ_target)²
                + λ   · (n_removed / n)

    Three optimizers are available:
      - Bisection on threshold (Layer 1: 1D, deterministic)
      - Simulated annealing on mask (Layer 2: combinatorial, stochastic)
      - Hill climbing with random mutation (Layer 2: local refinement)

    The retention penalty λ prevents aggressive data gutting.  A minimum
    retention ratio enforces that at least MIN_RETENTION_RATIO of samples
    survive.

Invariants:
    - All computation is pure: no Qt calls, no I/O, no side effects.
    - Every numeric literal is a named Final constant with a unit suffix.
    - optimize_outliers always returns an OutlierOptResult with the mask,
      objective trace, and before/after moment statistics.
    - The input DataFrame is never mutated.

Assumptions:
    - Input columns are numeric and contain at least MIN_SAMPLES_FOR_MOMENTS
      non-NaN values.
    - scipy is available in the dnv conda environment.

Failure Modes:
    - Fewer than MIN_SAMPLES_FOR_MOMENTS valid values: raises ValueError.
    - All optimizers converge to removing 0 rows if distribution already
      meets targets.
    - SA/hill-climb may not find global optimum in allocated iterations.

Provenance:
    - This module emits no transformation metadata.  Provenance is recorded
      by the calling DataCleaningWindow.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numeric constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Target excess kurtosis (0 = mesokurtic / Gaussian) [dimensionless].
TARGET_KURTOSIS_DEFAULT: Final[float] = 0.0

#: Target skewness (0 = symmetric) [dimensionless].
TARGET_SKEWNESS_DEFAULT: Final[float] = 0.0

#: Weight for kurtosis term in objective [dimensionless].
WEIGHT_KURTOSIS_DEFAULT: Final[float] = 1.0

#: Weight for skewness term in objective [dimensionless].
WEIGHT_SKEWNESS_DEFAULT: Final[float] = 1.0

#: Weight for retention penalty in objective [dimensionless].
WEIGHT_RETENTION_DEFAULT: Final[float] = 0.5

#: Minimum fraction of samples that must survive optimization [dimensionless].
MIN_RETENTION_RATIO: Final[float] = 0.50

#: Minimum samples required for reliable moment estimation [dimensionless].
MIN_SAMPLES_FOR_MOMENTS: Final[int] = 10

#: Bisection tolerance on threshold parameter [dimensionless].
BISECT_TOL_DEFAULT: Final[float] = 0.01

#: Maximum bisection iterations [dimensionless].
BISECT_MAX_ITER: Final[int] = 50

#: Default simulated annealing iterations [dimensionless].
SA_MAX_ITER_DEFAULT: Final[int] = 2000

#: SA initial temperature [dimensionless].
SA_TEMP_INIT_DEFAULT: Final[float] = 1.0

#: SA cooling rate (multiplicative per iteration) [dimensionless].
SA_COOLING_RATE_DEFAULT: Final[float] = 0.995

#: Default hill climbing iterations [dimensionless].
HC_MAX_ITER_DEFAULT: Final[int] = 1000

#: Default random state seed [dimensionless].
RANDOM_STATE_DEFAULT: Final[int] = 42


# ---------------------------------------------------------------------------
# Result record — frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutlierOptResult:
    """Result of moment-optimized outlier conditioning.

    Attributes
    ----------
    column : str
        Column name that was optimized.
    mask : np.ndarray
        Boolean array aligned to input DataFrame; True = keep, False = remove.
    n_total : int
        Total number of rows in input.
    n_removed : int
        Number of rows removed.
    kurtosis_before : float
        Excess kurtosis of original data.
    kurtosis_after : float
        Excess kurtosis after optimization.
    skewness_before : float
        Skewness of original data.
    skewness_after : float
        Skewness after optimization.
    target_kurtosis : float
        Target excess kurtosis.
    target_skewness : float
        Target skewness.
    objective_before : float
        Objective value before optimization.
    objective_after : float
        Objective value after optimization.
    optimizer : str
        Optimizer used ("bisection", "simulated_annealing", "hill_climbing").
    base_method : str
        Base outlier method used for initial mask ("iqr", "zscore", "none").
    objective_trace : Tuple[float, ...]
        Objective value at each iteration (for convergence monitoring).
    params : Dict[str, Any]
        Full optimizer parameters for provenance.
    """
    column:            str
    mask:              np.ndarray
    n_total:           int
    n_removed:         int
    kurtosis_before:   float
    kurtosis_after:    float
    skewness_before:   float
    skewness_after:    float
    target_kurtosis:   float
    target_skewness:   float
    objective_before:  float
    objective_after:   float
    optimizer:         str
    base_method:       str
    objective_trace:   Tuple[float, ...]
    params:            Dict[str, Any]


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _compute_objective(
    x: np.ndarray,
    target_kurt: float,
    target_skew: float,
    w_kurt: float,
    w_skew: float,
    w_retention: float,
    n_total: int,
) -> float:
    """Compute the moment-optimization objective.

    L = w_k·(κ − κ_target)² + w_s·(γ − γ_target)² + λ·(n_removed/n)
    """
    n = len(x)
    if n < MIN_SAMPLES_FOR_MOMENTS:
        return float("inf")
    kurt = float(_scipy_kurtosis(x, fisher=True, nan_policy="omit"))
    skew = float(_scipy_skew(x, nan_policy="omit"))
    n_removed = n_total - n
    return (
        w_kurt * (kurt - target_kurt) ** 2
        + w_skew * (skew - target_skew) ** 2
        + w_retention * (n_removed / max(n_total, 1))
    )


def _compute_moments(x: np.ndarray) -> Tuple[float, float]:
    """Return (excess_kurtosis, skewness) of array x."""
    if len(x) < MIN_SAMPLES_FOR_MOMENTS:
        return (float("nan"), float("nan"))
    kurt = float(_scipy_kurtosis(x, fisher=True, nan_policy="omit"))
    skew = float(_scipy_skew(x, nan_policy="omit"))
    return (kurt, skew)


# ---------------------------------------------------------------------------
# Layer 1: Bisection on threshold
# ---------------------------------------------------------------------------

def optimize_threshold_bisection(
    x: np.ndarray,
    *,
    base_method: str = "iqr",
    target_kurtosis: float = TARGET_KURTOSIS_DEFAULT,
    target_skewness: float = TARGET_SKEWNESS_DEFAULT,
    w_kurtosis: float = WEIGHT_KURTOSIS_DEFAULT,
    w_skewness: float = WEIGHT_SKEWNESS_DEFAULT,
    w_retention: float = WEIGHT_RETENTION_DEFAULT,
    min_retention: float = MIN_RETENTION_RATIO,
    tol: float = BISECT_TOL_DEFAULT,
    max_iter: int = BISECT_MAX_ITER,
) -> Tuple[np.ndarray, float, List[float]]:
    """Optimize the IQR/z-score threshold via bisection to minimize L.

    Parameters
    ----------
    x : np.ndarray
        1D array of column values (NaN-free).
    base_method : str
        "iqr" or "zscore".
    target_kurtosis, target_skewness : float
        Target moment values.
    w_kurtosis, w_skewness, w_retention : float
        Objective weights.
    min_retention : float
        Minimum fraction of samples to retain.
    tol : float
        Convergence tolerance on threshold.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    (mask, best_threshold, trace)
        mask: bool array True=keep; best_threshold: optimal value; trace: objective per iter.
    """
    n = len(x)
    min_keep = max(MIN_SAMPLES_FOR_MOMENTS, int(n * min_retention))
    trace: List[float] = []

    def _mask_at_theta(theta: float) -> np.ndarray:
        if base_method == "zscore":
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            if sigma < 1e-12:
                return np.ones(n, dtype=bool)
            z = np.abs((x - mu) / sigma)
            return z <= theta
        else:  # IQR
            q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
            iqr = q75 - q25
            if iqr < 1e-12:
                return np.ones(n, dtype=bool)
            lo = q25 - theta * iqr
            hi = q75 + theta * iqr
            return (x >= lo) & (x <= hi)

    def _obj_at_theta(theta: float) -> float:
        m = _mask_at_theta(theta)
        if m.sum() < min_keep:
            return float("inf")
        return _compute_objective(
            x[m], target_kurtosis, target_skewness,
            w_kurtosis, w_skewness, w_retention, n,
        )

    # Search range: theta_lo (aggressive) to theta_hi (permissive)
    if base_method == "zscore":
        theta_lo, theta_hi = 0.5, 6.0
    else:
        theta_lo, theta_hi = 0.1, 5.0

    # Golden-section search (more robust than bisection for non-monotone)
    _GOLDEN_RATIO: float = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = theta_lo, theta_hi
    c = b - _GOLDEN_RATIO * (b - a)
    d = a + _GOLDEN_RATIO * (b - a)

    best_theta = (a + b) / 2.0
    best_obj = _obj_at_theta(best_theta)
    trace.append(best_obj)

    for _ in range(max_iter):
        fc = _obj_at_theta(c)
        fd = _obj_at_theta(d)
        trace.append(min(fc, fd))

        if fc < fd:
            b = d
            if fc < best_obj:
                best_obj = fc
                best_theta = c
        else:
            a = c
            if fd < best_obj:
                best_obj = fd
                best_theta = d

        c = b - _GOLDEN_RATIO * (b - a)
        d = a + _GOLDEN_RATIO * (b - a)

        if abs(b - a) < tol:
            break

    return _mask_at_theta(best_theta), best_theta, trace


# ---------------------------------------------------------------------------
# Layer 2a: Simulated annealing on mask
# ---------------------------------------------------------------------------

def optimize_mask_sa(
    x: np.ndarray,
    initial_mask: np.ndarray,
    *,
    target_kurtosis: float = TARGET_KURTOSIS_DEFAULT,
    target_skewness: float = TARGET_SKEWNESS_DEFAULT,
    w_kurtosis: float = WEIGHT_KURTOSIS_DEFAULT,
    w_skewness: float = WEIGHT_SKEWNESS_DEFAULT,
    w_retention: float = WEIGHT_RETENTION_DEFAULT,
    min_retention: float = MIN_RETENTION_RATIO,
    max_iter: int = SA_MAX_ITER_DEFAULT,
    temp_init: float = SA_TEMP_INIT_DEFAULT,
    cooling_rate: float = SA_COOLING_RATE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
) -> Tuple[np.ndarray, List[float]]:
    """Refine an inclusion mask via simulated annealing.

    Each move flips one sample's inclusion bit.  Moves that violate
    min_retention are rejected outright.

    Parameters
    ----------
    x : np.ndarray
        1D array of column values.
    initial_mask : np.ndarray
        Starting boolean mask (True=keep).
    target_kurtosis, target_skewness : float
        Target moment values.
    w_kurtosis, w_skewness, w_retention : float
        Objective weights.
    min_retention : float
        Minimum fraction of samples to retain.
    max_iter : int
        Number of SA iterations.
    temp_init : float
        Initial temperature.
    cooling_rate : float
        Multiplicative cooling per iteration.
    rs : int
        Random seed.

    Returns
    -------
    (best_mask, trace)
    """
    rng = np.random.default_rng(rs)
    n = len(x)
    min_keep = max(MIN_SAMPLES_FOR_MOMENTS, int(n * min_retention))

    mask = initial_mask.copy()
    current_obj = _compute_objective(
        x[mask], target_kurtosis, target_skewness,
        w_kurtosis, w_skewness, w_retention, n,
    )
    best_mask = mask.copy()
    best_obj = current_obj
    trace: List[float] = [current_obj]
    temp = temp_init

    for i in range(max_iter):
        # Pick a random index to flip
        idx = rng.integers(0, n)
        candidate = mask.copy()
        candidate[idx] = not candidate[idx]

        # Check retention constraint
        n_keep = int(candidate.sum())
        if n_keep < min_keep:
            trace.append(best_obj)
            temp *= cooling_rate
            continue

        cand_obj = _compute_objective(
            x[candidate], target_kurtosis, target_skewness,
            w_kurtosis, w_skewness, w_retention, n,
        )

        delta = cand_obj - current_obj
        if delta < 0 or (temp > 1e-12 and rng.random() < math.exp(-delta / temp)):
            mask = candidate
            current_obj = cand_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_mask = mask.copy()

        trace.append(best_obj)
        temp *= cooling_rate

    return best_mask, trace


# ---------------------------------------------------------------------------
# Layer 2b: Hill climbing with random mutation
# ---------------------------------------------------------------------------

def optimize_mask_hillclimb(
    x: np.ndarray,
    initial_mask: np.ndarray,
    *,
    target_kurtosis: float = TARGET_KURTOSIS_DEFAULT,
    target_skewness: float = TARGET_SKEWNESS_DEFAULT,
    w_kurtosis: float = WEIGHT_KURTOSIS_DEFAULT,
    w_skewness: float = WEIGHT_SKEWNESS_DEFAULT,
    w_retention: float = WEIGHT_RETENTION_DEFAULT,
    min_retention: float = MIN_RETENTION_RATIO,
    max_iter: int = HC_MAX_ITER_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
) -> Tuple[np.ndarray, List[float]]:
    """Refine an inclusion mask via greedy hill climbing with random restarts.

    Each iteration flips a random bit.  Only improvements are accepted.
    Every n/4 steps without improvement, a random multi-flip perturbation
    is applied to escape plateaus.

    Parameters
    ----------
    x : np.ndarray
        1D array of column values.
    initial_mask : np.ndarray
        Starting boolean mask (True=keep).
    (remaining params same as SA)

    Returns
    -------
    (best_mask, trace)
    """
    rng = np.random.default_rng(rs)
    n = len(x)
    min_keep = max(MIN_SAMPLES_FOR_MOMENTS, int(n * min_retention))
    stale_limit = max(n // 4, 20)

    mask = initial_mask.copy()
    current_obj = _compute_objective(
        x[mask], target_kurtosis, target_skewness,
        w_kurtosis, w_skewness, w_retention, n,
    )
    best_mask = mask.copy()
    best_obj = current_obj
    trace: List[float] = [current_obj]
    stale_count = 0

    for i in range(max_iter):
        # Random single-bit flip
        idx = rng.integers(0, n)
        candidate = mask.copy()
        candidate[idx] = not candidate[idx]

        n_keep = int(candidate.sum())
        if n_keep < min_keep:
            stale_count += 1
            trace.append(best_obj)
            if stale_count >= stale_limit:
                # Multi-flip perturbation to escape plateau
                n_flips = rng.integers(2, max(3, n // 20))
                flip_idx = rng.choice(n, size=n_flips, replace=False)
                for fi in flip_idx:
                    mask[fi] = not mask[fi]
                if mask.sum() < min_keep:
                    mask = best_mask.copy()
                else:
                    current_obj = _compute_objective(
                        x[mask], target_kurtosis, target_skewness,
                        w_kurtosis, w_skewness, w_retention, n,
                    )
                stale_count = 0
            continue

        cand_obj = _compute_objective(
            x[candidate], target_kurtosis, target_skewness,
            w_kurtosis, w_skewness, w_retention, n,
        )

        if cand_obj < current_obj:
            mask = candidate
            current_obj = cand_obj
            stale_count = 0
            if current_obj < best_obj:
                best_obj = current_obj
                best_mask = mask.copy()
        else:
            stale_count += 1

        trace.append(best_obj)

        # Multi-flip perturbation on stale
        if stale_count >= stale_limit:
            n_flips = rng.integers(2, max(3, n // 20))
            flip_idx = rng.choice(n, size=n_flips, replace=False)
            for fi in flip_idx:
                mask[fi] = not mask[fi]
            if mask.sum() < min_keep:
                mask = best_mask.copy()
            else:
                current_obj = _compute_objective(
                    x[mask], target_kurtosis, target_skewness,
                    w_kurtosis, w_skewness, w_retention, n,
                )
            stale_count = 0

    return best_mask, trace


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def optimize_outliers(
    df: pd.DataFrame,
    column: str,
    *,
    optimizer: str = "bisection",
    base_method: str = "iqr",
    target_kurtosis: float = TARGET_KURTOSIS_DEFAULT,
    target_skewness: float = TARGET_SKEWNESS_DEFAULT,
    w_kurtosis: float = WEIGHT_KURTOSIS_DEFAULT,
    w_skewness: float = WEIGHT_SKEWNESS_DEFAULT,
    w_retention: float = WEIGHT_RETENTION_DEFAULT,
    min_retention: float = MIN_RETENTION_RATIO,
    sa_max_iter: int = SA_MAX_ITER_DEFAULT,
    sa_temp_init: float = SA_TEMP_INIT_DEFAULT,
    sa_cooling_rate: float = SA_COOLING_RATE_DEFAULT,
    hc_max_iter: int = HC_MAX_ITER_DEFAULT,
    bisect_tol: float = BISECT_TOL_DEFAULT,
    bisect_max_iter: int = BISECT_MAX_ITER,
    rs: int = RANDOM_STATE_DEFAULT,
) -> OutlierOptResult:
    """Orchestrate moment-optimized outlier conditioning.

    Workflow:
      1. Extract the target column as a clean (NaN-free) 1D array.
      2. Run Layer 1 (bisection on threshold) to get an initial mask.
      3. If optimizer is "simulated_annealing" or "hill_climbing", refine
         the mask with the chosen Layer 2 optimizer.
      4. Return the full OutlierOptResult with before/after statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to optimize.
    optimizer : str
        "bisection", "simulated_annealing", or "hill_climbing".
    base_method : str
        "iqr" or "zscore" — used for Layer 1 threshold seeding.
    target_kurtosis : float
        Target excess kurtosis (Fisher definition; 0 = Gaussian).
    target_skewness : float
        Target skewness (0 = symmetric).
    w_kurtosis, w_skewness, w_retention : float
        Objective weights.
    min_retention : float
        Minimum fraction of samples to retain (0.0–1.0).
    sa_max_iter, sa_temp_init, sa_cooling_rate : SA params.
    hc_max_iter : hill climbing iterations.
    bisect_tol, bisect_max_iter : bisection params.
    rs : int
        Random seed.

    Returns
    -------
    OutlierOptResult
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    s = df[column].dropna()
    if len(s) < MIN_SAMPLES_FOR_MOMENTS:
        raise ValueError(
            f"Column '{column}' has {len(s)} non-NaN values; need at least "
            f"{MIN_SAMPLES_FOR_MOMENTS} for reliable moment estimation."
        )

    x = s.values.astype(float)
    n = len(x)
    # Map back to DataFrame index
    valid_idx = s.index

    # Before stats
    kurt_before, skew_before = _compute_moments(x)
    obj_before = _compute_objective(
        x, target_kurtosis, target_skewness,
        w_kurtosis, w_skewness, w_retention, n,
    )

    # Layer 1: bisection on threshold
    bisect_mask, _best_theta, bisect_trace = optimize_threshold_bisection(
        x,
        base_method=base_method,
        target_kurtosis=target_kurtosis,
        target_skewness=target_skewness,
        w_kurtosis=w_kurtosis,
        w_skewness=w_skewness,
        w_retention=w_retention,
        min_retention=min_retention,
        tol=bisect_tol,
        max_iter=bisect_max_iter,
    )

    if optimizer == "bisection":
        final_mask_local = bisect_mask
        trace = bisect_trace
    elif optimizer == "simulated_annealing":
        final_mask_local, sa_trace = optimize_mask_sa(
            x, bisect_mask,
            target_kurtosis=target_kurtosis,
            target_skewness=target_skewness,
            w_kurtosis=w_kurtosis,
            w_skewness=w_skewness,
            w_retention=w_retention,
            min_retention=min_retention,
            max_iter=sa_max_iter,
            temp_init=sa_temp_init,
            cooling_rate=sa_cooling_rate,
            rs=rs,
        )
        trace = bisect_trace + sa_trace
    elif optimizer == "hill_climbing":
        final_mask_local, hc_trace = optimize_mask_hillclimb(
            x, bisect_mask,
            target_kurtosis=target_kurtosis,
            target_skewness=target_skewness,
            w_kurtosis=w_kurtosis,
            w_skewness=w_skewness,
            w_retention=w_retention,
            min_retention=min_retention,
            max_iter=hc_max_iter,
            rs=rs,
        )
        trace = bisect_trace + hc_trace
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. "
            "Use 'bisection', 'simulated_annealing', or 'hill_climbing'."
        )

    # Map local mask back to full DataFrame index
    full_mask = np.ones(len(df), dtype=bool)
    # NaN rows are kept (not subject to outlier removal)
    nan_positions = df[column].isna()
    removed_positions = valid_idx[~final_mask_local]
    full_mask[df.index.isin(removed_positions)] = False

    # After stats
    x_after = x[final_mask_local]
    kurt_after, skew_after = _compute_moments(x_after)
    obj_after = _compute_objective(
        x_after, target_kurtosis, target_skewness,
        w_kurtosis, w_skewness, w_retention, n,
    )
    n_removed = int((~final_mask_local).sum())

    params_dict: Dict[str, Any] = {
        "optimizer": optimizer,
        "base_method": base_method,
        "target_kurtosis": target_kurtosis,
        "target_skewness": target_skewness,
        "w_kurtosis": w_kurtosis,
        "w_skewness": w_skewness,
        "w_retention": w_retention,
        "min_retention": min_retention,
        "random_state": rs,
    }
    if optimizer == "bisection":
        params_dict["bisect_tol"] = bisect_tol
        params_dict["best_threshold"] = float(_best_theta)
    elif optimizer == "simulated_annealing":
        params_dict["sa_max_iter"] = sa_max_iter
        params_dict["sa_temp_init"] = sa_temp_init
        params_dict["sa_cooling_rate"] = sa_cooling_rate
    elif optimizer == "hill_climbing":
        params_dict["hc_max_iter"] = hc_max_iter

    return OutlierOptResult(
        column=column,
        mask=full_mask,
        n_total=n,
        n_removed=n_removed,
        kurtosis_before=kurt_before,
        kurtosis_after=kurt_after,
        skewness_before=skew_before,
        skewness_after=skew_after,
        target_kurtosis=target_kurtosis,
        target_skewness=target_skewness,
        objective_before=obj_before,
        objective_after=obj_after,
        optimizer=optimizer,
        base_method=base_method,
        objective_trace=tuple(trace),
        params=params_dict,
    )
