"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for cross-variant invariance testing.
    Runs any analysis function across the admissible family produced by SEAL,
    computes stability diagnostics (feature importance variance, Kendall tau,
    top-k overlap, sign consistency, prediction variance), assigns stability
    labels ("family_stable" vs "contingent"), and wraps results as citable
    records.

Scientific Context:
    DNV-2.0 thesis: "Interpretability is an invariance claim.  Stable fit does
    not imply stable meaning."  SEAL produces K admissible variant DataFrames;
    this engine consumes them.  For each variant T_k(X) it re-runs a caller-
    supplied analysis function and collects {Result_k}.  Stability diagnostics
    follow DNV-1.0 Paper Eq 5:
      sigma^2_f = Var_k(phi_f^{T_k})   — per-feature importance variance
      Delta_pred = Var_k(L^{T_k})       — prediction loss variance
    Rank-based diagnostics: pairwise Kendall tau, top-k overlap, sign
    consistency.  Each feature/metric is labelled "family_stable" if its
    normalised variance (CV^2) is below a threshold, else "contingent".

Invariants:
    - All computation is pure: no Qt calls, no I/O, no side effects.
    - Every numeric literal is a named Final constant with a unit suffix.
    - run_across_variants never mutates the input SEALResult or DataFrames.
    - Stability labels are always one of LABEL_FAMILY_STABLE or LABEL_CONTINGENT.
    - Results are frozen dataclasses.

Assumptions:
    - seal_result.is_valid is True and seal_result.transformed_dfs is non-empty.
    - Analysis functions fn(df, ...) return objects with a .scores (pd.Series)
      attribute (for feature stability) or a .metrics (dict) attribute (for
      prediction stability).
    - scipy is available in the dnv conda environment.

Failure Modes:
    - All K variant analyses raise: RuntimeError with diagnostic message.
    - Fewer than MIN_VARIANTS_FOR_STABILITY successful variants: stability
      diagnostics are trivially stable (all variance = 0).
    - Mismatched feature sets across variants: union-aligned with fill value 0.

Provenance:
    - CitableRecord captures variant_id, dataset SHA-256, contract context,
      method configuration, timestamp, and stability label.
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ing_provenance import dataframe_sha256 as _dataframe_sha256

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numeric constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Default k for top-k overlap computation [dimensionless].
TOP_K_OVERLAP_DEFAULT: Final[int] = 10

#: Normalised variance threshold below which a feature is "family_stable" [dimensionless].
FEATURE_VARIANCE_THRESHOLD_DEFAULT: Final[float] = 0.05

#: Metric standard deviation threshold below which prediction is "family_stable" [dimensionless].
METRIC_STD_THRESHOLD_DEFAULT: Final[float] = 0.02

#: Minimum successful variant analyses required for meaningful stability [dimensionless].
MIN_VARIANTS_FOR_STABILITY: Final[int] = 2

#: Fill value for missing features when union-aligning score vectors [dimensionless].
MISSING_SCORE_FILL: Final[float] = 0.0

#: Stability label for features/metrics that persist across all variants.
LABEL_FAMILY_STABLE: Final[str] = "family_stable"

#: Stability label for features/metrics that do not persist across variants.
LABEL_CONTINGENT: Final[str] = "contingent"

#: Stability label for features with uniformly weak effects (stable by triviality).
LABEL_TRIVIALLY_STABLE: Final[str] = "trivially_stable"

#: Minimum mean absolute importance below which stability is trivial [dimensionless].
#: Features below this floor are labelled "trivially_stable" rather than
#: "family_stable" to flag that stability arises because all effects are
#: uniformly weak (DNV-1.0 §5: "avoid stability by triviality").
MIN_EFFECT_SIZE_THRESHOLD: Final[float] = 1e-4


# ---------------------------------------------------------------------------
# Result records — frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CitableRecord:
    """Wraps any analysis result with full provenance for scientific citation.

    Attributes
    ----------
    analysis_result : Any
        The wrapped result (CorrelationResult, FeatureRankingResult, MLResult, etc.).
    variant_id : int or None
        Which SEAL variant this was computed on (None = base dataset).
    dataset_sha256 : str
        SHA-256 hex digest of the exact DataFrame analysed.
    contract_context : dict
        Active contracts at analysis time.
    method_config : dict
        Full method configuration / hyperparameters.
    model_state_hash : str or None
        SHA-256 hex digest of serialised model state (if ML), else None.
    timestamp_utc : str
        ISO-8601 UTC timestamp of record creation.
    stability_label : str or None
        LABEL_FAMILY_STABLE, LABEL_CONTINGENT, or None if not yet assessed.
    """
    analysis_result:  Any
    variant_id:       Optional[int]
    dataset_sha256:   str
    contract_context: Dict[str, Any]
    method_config:    Dict[str, Any]
    model_state_hash: Optional[str]
    timestamp_utc:    str
    stability_label:  Optional[str] = None


@dataclass(frozen=True)
class FeatureStabilityResult:
    """Stability diagnostics for feature importance across K variants.

    Attributes
    ----------
    sigma2_f : pd.Series
        Per-feature variance of importance scores across variants.
        Index = feature name, values = Var_k(phi_f^{T_k}).
    mean_importance : pd.Series
        Per-feature mean importance across variants.
    kendall_tau_matrix : np.ndarray
        (K x K) pairwise Kendall tau correlation matrix of rankings.
    mean_kendall_tau : float
        Mean of off-diagonal Kendall tau values.
    top_k_overlap : float
        Fraction of top-k features present in ALL variants (0.0–1.0).
    sign_consistency : pd.Series
        Per-feature fraction of variants agreeing on sign (0.0–1.0).
    stability_labels : Dict[str, str]
        {feature_name: LABEL_FAMILY_STABLE | LABEL_CONTINGENT}.
    per_variant_scores : Tuple[pd.Series, ...]
        Raw importance scores from each successful variant analysis.
    n_variants_used : int
        Number of variants that produced valid results.
    """
    sigma2_f:           pd.Series
    mean_importance:    pd.Series
    kendall_tau_matrix: np.ndarray
    mean_kendall_tau:   float
    top_k_overlap:      float
    sign_consistency:   pd.Series
    stability_labels:   Dict[str, str]
    per_variant_scores: Tuple[pd.Series, ...]
    n_variants_used:    int


@dataclass(frozen=True)
class PredictionStabilityResult:
    """Stability diagnostics for prediction metrics across K variants.

    Attributes
    ----------
    delta_pred : float
        Var_k(primary_metric_k) — variance of the primary metric across variants.
    mean_metrics : Dict[str, float]
        Mean of each metric across variants.
    metric_std : Dict[str, float]
        Standard deviation of each metric across variants.
    per_variant_metrics : Tuple[Dict[str, float], ...]
        Raw metrics dict from each successful variant analysis.
    stability_labels : Dict[str, str]
        {metric_name: LABEL_FAMILY_STABLE | LABEL_CONTINGENT}.
    n_variants_used : int
        Number of variants that produced valid results.
    """
    delta_pred:          float
    mean_metrics:        Dict[str, float]
    metric_std:          Dict[str, float]
    per_variant_metrics: Tuple[Dict[str, float], ...]
    stability_labels:    Dict[str, str]
    n_variants_used:     int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_scores(scores_list: List[pd.Series]) -> pd.DataFrame:
    """Union-align K score vectors into a (K x F) DataFrame.

    Missing features in any variant are filled with MISSING_SCORE_FILL.

    Parameters
    ----------
    scores_list : list of pd.Series
        Each Series is indexed by feature name with importance values.

    Returns
    -------
    pd.DataFrame
        Rows = variants (0..K-1), columns = union of all feature names.
    """
    all_features = pd.Index(
        sorted(set().union(*(s.index for s in scores_list)))
    )
    aligned = pd.DataFrame(
        MISSING_SCORE_FILL,
        index=range(len(scores_list)),
        columns=all_features,
        dtype=float,
    )
    for k, s in enumerate(scores_list):
        aligned.loc[k, s.index] = s.values.astype(float)
    return aligned


def _pairwise_kendall_tau(score_matrix: pd.DataFrame) -> np.ndarray:
    """Compute K x K pairwise Kendall tau correlation matrix.

    Parameters
    ----------
    score_matrix : pd.DataFrame
        (K x F) matrix from _align_scores.

    Returns
    -------
    np.ndarray
        (K x K) symmetric matrix of Kendall tau values.
    """
    K = score_matrix.shape[0]
    tau_matrix = np.eye(K, dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tau_val, _ = kendalltau(
                    score_matrix.iloc[i].values,
                    score_matrix.iloc[j].values,
                )
            if np.isnan(tau_val):
                tau_val = 0.0
            tau_matrix[i, j] = tau_val
            tau_matrix[j, i] = tau_val
    return tau_matrix


def _top_k_overlap(score_matrix: pd.DataFrame, k: int) -> float:
    """Fraction of top-k features present in ALL K variants.

    Parameters
    ----------
    score_matrix : pd.DataFrame
        (K x F) matrix from _align_scores.
    k : int
        Number of top features to consider per variant.

    Returns
    -------
    float
        |intersection of top-k sets| / k.  Range [0.0, 1.0].
    """
    K = score_matrix.shape[0]
    effective_k = min(k, score_matrix.shape[1])
    if effective_k == 0:
        return 1.0
    top_k_sets = []
    for i in range(K):
        row = score_matrix.iloc[i].abs().sort_values(ascending=False)
        top_k_sets.append(set(row.index[:effective_k]))
    intersection = top_k_sets[0]
    for s in top_k_sets[1:]:
        intersection = intersection & s
    return len(intersection) / effective_k


def _sign_consistency(score_matrix: pd.DataFrame) -> pd.Series:
    """Per-feature fraction of variants agreeing on the majority sign.

    Parameters
    ----------
    score_matrix : pd.DataFrame
        (K x F) matrix from _align_scores.

    Returns
    -------
    pd.Series
        Index = feature names, values = fraction [0.0, 1.0].
    """
    K = score_matrix.shape[0]
    if K <= 1:
        return pd.Series(1.0, index=score_matrix.columns)
    signs = np.sign(score_matrix.values)  # K x F
    # For each feature, count how many variants have the majority sign
    pos_count = (signs > 0).sum(axis=0)
    neg_count = (signs < 0).sum(axis=0)
    majority = np.maximum(pos_count, neg_count)
    # Features with all-zero scores get consistency 1.0
    all_zero = (signs == 0).all(axis=0)
    consistency = majority / K
    consistency[all_zero] = 1.0
    return pd.Series(consistency, index=score_matrix.columns)


def _stability_labels_from_variance(
    variance: pd.Series,
    mean_abs: pd.Series,
    threshold: float,
    effect_floor: float = MIN_EFFECT_SIZE_THRESHOLD,
) -> Dict[str, str]:
    """Assign stability labels based on normalised variance (CV^2).

    A feature is:
    - "trivially_stable" if mean_abs < effect_floor (DNV-1.0 p.5:
      "avoid stability by triviality — flag cases where stability arises
      because all effects are uniformly weak").
    - "family_stable" if sigma^2_f / (mean_abs_f + eps)^2 < threshold.
    - "contingent" otherwise.

    Parameters
    ----------
    variance : pd.Series
        Per-feature variance of importance scores.
    mean_abs : pd.Series
        Per-feature mean absolute importance.
    threshold : float
        Normalised variance threshold.
    effect_floor : float
        Minimum mean absolute importance; features below this are
        labelled "trivially_stable" regardless of CV^2.

    Returns
    -------
    dict
        {feature_name: LABEL_FAMILY_STABLE | LABEL_CONTINGENT | LABEL_TRIVIALLY_STABLE}.
    """
    eps = np.finfo(float).eps
    cv_sq = variance / (mean_abs + eps) ** 2
    labels: Dict[str, str] = {}
    for feat in variance.index:
        if mean_abs[feat] < effect_floor:
            labels[feat] = LABEL_TRIVIALLY_STABLE
        elif cv_sq[feat] < threshold:
            labels[feat] = LABEL_FAMILY_STABLE
        else:
            labels[feat] = LABEL_CONTINGENT
    return labels


def _metric_stability_labels(
    metric_std: Dict[str, float],
    threshold: float,
) -> Dict[str, str]:
    """Assign stability labels to prediction metrics based on std deviation.

    Parameters
    ----------
    metric_std : dict
        {metric_name: standard_deviation_across_variants}.
    threshold : float
        Threshold on std below which the metric is "family_stable".

    Returns
    -------
    dict
        {metric_name: LABEL_FAMILY_STABLE | LABEL_CONTINGENT}.
    """
    return {
        name: LABEL_FAMILY_STABLE if std < threshold else LABEL_CONTINGENT
        for name, std in metric_std.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_across_variants(
    seal_result: Any,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, ...]:
    """Run an analysis function across all SEAL variants.

    Calls ``fn(df_k, *args, **kwargs)`` for each variant DataFrame in
    ``seal_result.transformed_dfs``.  Variants that raise are skipped
    with a logged warning.  If ALL variants fail, raises RuntimeError.

    Parameters
    ----------
    seal_result : SEALResult
        Must have ``transformed_dfs`` (tuple of DataFrames) and ``is_valid=True``.
    fn : callable
        Analysis function with signature ``fn(df, *args, **kwargs) -> Result``.
    *args, **kwargs
        Forwarded to fn after the DataFrame argument.

    Returns
    -------
    tuple
        Results from each successful variant, in variant order.
        Failed variants are omitted (so len may be < K).

    Raises
    ------
    RuntimeError
        If all K variant analyses fail.
    ValueError
        If seal_result is invalid or has no variants.
    """
    if not getattr(seal_result, "is_valid", False):
        raise ValueError("seal_result.is_valid is False; cannot run variants.")
    dfs = seal_result.transformed_dfs
    if not dfs:
        raise ValueError("seal_result.transformed_dfs is empty.")

    results: List[Any] = []
    for k, df_k in enumerate(dfs):
        try:
            r = fn(df_k, *args, **kwargs)
            results.append(r)
        except Exception as exc:
            log.warning(
                "run_across_variants: variant %d/%d failed: %s",
                k, len(dfs), exc,
            )
    if not results:
        raise RuntimeError(
            f"All {len(dfs)} variant analyses failed. "
            "Check logs for per-variant errors."
        )
    return tuple(results)


def compute_feature_stability(
    seal_result: Any,
    fi_fn: Callable,
    target_col: str,
    *,
    top_k: int = TOP_K_OVERLAP_DEFAULT,
    variance_threshold: float = FEATURE_VARIANCE_THRESHOLD_DEFAULT,
    **fi_kwargs: Any,
) -> FeatureStabilityResult:
    """Compute feature importance stability across SEAL variants.

    Runs ``fi_fn(df_k, feature_cols, target_col, **fi_kwargs)`` on each
    variant, then computes per-feature variance, Kendall tau, top-k
    overlap, sign consistency, and stability labels.

    Parameters
    ----------
    seal_result : SEALResult
        Valid SEAL result with transformed_dfs.
    fi_fn : callable
        Feature importance function with signature
        ``fi_fn(df, target_col, **kwargs) -> FeatureRankingResult``.
        Must return an object with a ``.scores`` attribute (pd.Series).
        This matches the fi_engine convention where feature columns are
        derived internally from all numeric columns minus target.
    target_col : str
        Target column name.
    top_k : int
        Number of top features for overlap computation.
    variance_threshold : float
        Normalised variance threshold for stability labels.
    **fi_kwargs
        Additional keyword arguments forwarded to fi_fn.

    Returns
    -------
    FeatureStabilityResult
    """
    dfs = seal_result.transformed_dfs

    scores_list: List[pd.Series] = []
    for k, df_k in enumerate(dfs):
        try:
            result = fi_fn(df_k, target_col, **fi_kwargs)
            scores_list.append(result.scores)
        except Exception as exc:
            log.warning(
                "compute_feature_stability: variant %d/%d failed: %s",
                k, len(dfs), exc,
            )

    if not scores_list:
        raise RuntimeError(
            f"All {len(dfs)} feature importance analyses failed. "
            "Check logs for per-variant errors."
        )

    K = len(scores_list)
    score_matrix = _align_scores(scores_list)

    # Per-feature variance and mean (Eq 5)
    sigma2_f = score_matrix.var(axis=0, ddof=1) if K > 1 else pd.Series(
        0.0, index=score_matrix.columns
    )
    mean_importance = score_matrix.mean(axis=0)
    mean_abs = score_matrix.abs().mean(axis=0)

    # Kendall tau
    tau_matrix = _pairwise_kendall_tau(score_matrix) if K > 1 else np.array([[1.0]])
    off_diag = tau_matrix[np.triu_indices_from(tau_matrix, k=1)]
    mean_tau = float(off_diag.mean()) if off_diag.size > 0 else 1.0

    # Top-k overlap
    overlap = _top_k_overlap(score_matrix, top_k)

    # Sign consistency
    sign_cons = _sign_consistency(score_matrix)

    # Stability labels
    labels = _stability_labels_from_variance(sigma2_f, mean_abs, variance_threshold)

    return FeatureStabilityResult(
        sigma2_f=sigma2_f,
        mean_importance=mean_importance,
        kendall_tau_matrix=tau_matrix,
        mean_kendall_tau=mean_tau,
        top_k_overlap=overlap,
        sign_consistency=sign_cons,
        stability_labels=labels,
        per_variant_scores=tuple(scores_list),
        n_variants_used=K,
    )


def compute_prediction_stability(
    seal_result: Any,
    ml_fn: Callable,
    feature_cols: List[str],
    target_col: str,
    *,
    primary_metric: Optional[str] = None,
    metric_std_threshold: float = METRIC_STD_THRESHOLD_DEFAULT,
    **ml_kwargs: Any,
) -> PredictionStabilityResult:
    """Compute prediction stability across SEAL variants.

    Runs ``ml_fn(df_k, feature_cols, target_col, **ml_kwargs)`` on each
    variant, then computes metric variance, mean, std, and stability
    labels.

    Parameters
    ----------
    seal_result : SEALResult
        Valid SEAL result with transformed_dfs.
    ml_fn : callable
        ML function with signature
        ``ml_fn(df, feature_cols, target_col, **kwargs) -> MLResult``.
        Must return an object with a ``.metrics`` attribute (dict).
    feature_cols : list of str
        Feature column names for the ML model.
    target_col : str
        Target column name.
    primary_metric : str, optional
        Name of the metric used for Δ_pred (Eq. 5).  If None, defaults
        to ``"r2"`` for regression, ``"accuracy"`` for classification
        (falls back to first metric if the chosen name is absent).
    metric_std_threshold : float
        Threshold on metric std for stability labels.
    **ml_kwargs
        Additional keyword arguments forwarded to ml_fn.

    Returns
    -------
    PredictionStabilityResult
    """
    dfs = seal_result.transformed_dfs

    metrics_list: List[Dict[str, float]] = []
    task_types: List[str] = []
    for k, df_k in enumerate(dfs):
        try:
            result = ml_fn(df_k, feature_cols, target_col, **ml_kwargs)
            metrics_list.append(result.metrics)
            task_types.append(getattr(result, "task_type", ""))
        except Exception as exc:
            log.warning(
                "compute_prediction_stability: variant %d/%d failed: %s",
                k, len(dfs), exc,
            )

    if not metrics_list:
        raise RuntimeError(
            f"All {len(dfs)} ML analyses failed. "
            "Check logs for per-variant errors."
        )

    K = len(metrics_list)

    # Collect all metric names
    all_metric_names = sorted(
        set().union(*(m.keys() for m in metrics_list))
    )

    # Build (K x M) array
    metric_array = np.zeros((K, len(all_metric_names)), dtype=float)
    for k, m in enumerate(metrics_list):
        for j, name in enumerate(all_metric_names):
            metric_array[k, j] = m.get(name, np.nan)

    # Mean and std per metric
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_vals = np.nanmean(metric_array, axis=0)
        std_vals = np.nanstd(metric_array, axis=0, ddof=1) if K > 1 else np.zeros(
            len(all_metric_names)
        )

    mean_metrics = {
        name: float(mean_vals[j]) for j, name in enumerate(all_metric_names)
    }
    metric_std = {
        name: float(std_vals[j]) if not np.isnan(std_vals[j]) else 0.0
        for j, name in enumerate(all_metric_names)
    }

    # Delta_pred = Var_k of the designated primary metric (Eq. 5).
    # Resolve primary_metric: explicit > task-based default > first available.
    _pm = primary_metric
    if _pm is None:
        is_clf = any(t == "classification" for t in task_types)
        _pm = "accuracy" if is_clf else "r2"
    if _pm in all_metric_names:
        primary_idx = all_metric_names.index(_pm)
    else:
        primary_idx = 0  # fallback to first metric
    primary_vals = metric_array[:, primary_idx]
    delta_pred = float(np.nanvar(primary_vals, ddof=1)) if K > 1 else 0.0
    if np.isnan(delta_pred):
        delta_pred = 0.0

    # Stability labels
    labels = _metric_stability_labels(metric_std, metric_std_threshold)

    return PredictionStabilityResult(
        delta_pred=delta_pred,
        mean_metrics=mean_metrics,
        metric_std=metric_std,
        per_variant_metrics=tuple(metrics_list),
        stability_labels=labels,
        n_variants_used=K,
    )


def make_citable(
    result: Any,
    variant_id: Optional[int],
    df: pd.DataFrame,
    *,
    contract_context: Optional[Dict[str, Any]] = None,
    method_config: Optional[Dict[str, Any]] = None,
    model_state_hash: Optional[str] = None,
    stability_label: Optional[str] = None,
) -> CitableRecord:
    """Wrap any analysis result as a CitableRecord with full provenance.

    Parameters
    ----------
    result : Any
        The analysis result to wrap.
    variant_id : int or None
        SEAL variant index, or None for base dataset.
    df : pd.DataFrame
        The exact DataFrame that was analysed.
    contract_context : dict, optional
        Active contracts at analysis time.
    method_config : dict, optional
        Full method configuration / hyperparameters.
    model_state_hash : str, optional
        SHA-256 of serialised model (for ML results).
    stability_label : str, optional
        LABEL_FAMILY_STABLE, LABEL_CONTINGENT, or None.

    Returns
    -------
    CitableRecord
    """
    return CitableRecord(
        analysis_result=result,
        variant_id=variant_id,
        dataset_sha256=_dataframe_sha256(df),
        contract_context=contract_context or {},
        method_config=method_config or {},
        model_state_hash=model_state_hash,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        stability_label=stability_label,
    )


# ---------------------------------------------------------------------------
# Model state hashing (DNV-2.0 Slide 11)
# ---------------------------------------------------------------------------

def compute_model_state_hash(model: Any) -> str:
    """Compute SHA-256 hex digest of a fitted model's serialised state.

    Used to make ML results citable: the model_state_hash field in
    CitableRecord ensures that the exact model weights are recorded.

    Parameters
    ----------
    model : Any
        A fitted sklearn / xgboost / lightgbm / PyTorch model.

    Returns
    -------
    str
        64-character hex SHA-256 digest.
    """
    import pickle
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


# ---------------------------------------------------------------------------
# Proxy substitution testing (DNV-2.0 Slide 7, Gap 8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProxyStabilityResult:
    """Stability diagnostics for proxy substitution within a proxy group.

    Attributes
    ----------
    proxy_group : str
        Name of the proxy equivalence group (e.g. "ionic_radius").
    members : tuple of str
        Column names in this group.
    per_substitution_scores : Tuple[pd.Series, ...]
        Raw importance scores from each substitution variant.
    sigma2_f : pd.Series
        Per-feature variance across substitutions.
    mean_importance : pd.Series
        Per-feature mean importance across substitutions.
    kendall_tau_matrix : np.ndarray
        Pairwise Kendall tau between substitution variants.
    mean_kendall_tau : float
        Mean off-diagonal Kendall tau.
    stability_labels : Dict[str, str]
        {feature: LABEL_FAMILY_STABLE | LABEL_CONTINGENT}.
    n_substitutions : int
        Number of substitutions performed.
    """
    proxy_group:              str
    members:                  Tuple[str, ...]
    per_substitution_scores:  Tuple[pd.Series, ...]
    sigma2_f:                 pd.Series
    mean_importance:          pd.Series
    kendall_tau_matrix:       np.ndarray
    mean_kendall_tau:         float
    stability_labels:         Dict[str, str]
    n_substitutions:          int


def find_proxy_groups(
    contracts: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Group columns by proxy_group from their contracts.

    Parameters
    ----------
    contracts : list of dict
        Observable contracts (each with optional "proxy_group" field).

    Returns
    -------
    dict
        {group_name: [col1, col2, ...]} for groups with >= 2 members.
    """
    groups: Dict[str, List[str]] = {}
    for c in contracts:
        obs = str(c.get("observable", "")).strip()
        pg = str(c.get("proxy_group", "")).strip()
        if obs and pg:
            groups.setdefault(pg, []).append(obs)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def substitute_proxy(
    df: pd.DataFrame,
    col_out: str,
    col_in: str,
    contracts: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Create a DataFrame variant with col_out replaced by standardised col_in.

    Both columns are z-scored so that the substitution is scale-invariant.
    The result has col_out replaced by the standardised values of col_in,
    and col_in is dropped to avoid duplication.

    If contracts are provided, a unit compatibility check is performed and
    a warning is logged if the two proxies have declared incompatible units
    (DNV-1.0 §1: affine shifts/scalings permitted only when they correspond
    to declared unit conversions).

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame containing both columns.
    col_out : str
        Column to replace (the "active" proxy).
    col_in : str
        Column to substitute in (the "alternative" proxy).
    contracts : list of dict, optional
        Observable contracts for unit compatibility checking.

    Returns
    -------
    pd.DataFrame
        Copy with col_out values replaced and col_in dropped.
    """
    if contracts:
        from ing_contracts import check_unit_compatibility
        msg = check_unit_compatibility(col_out, col_in, contracts)
        if msg:
            log.warning("substitute_proxy: %s", msg)
    df_new = df.copy()
    s_in = df_new[col_in].astype(float)
    mu, sigma = s_in.mean(), s_in.std()
    if sigma > 0:
        standardised = (s_in - mu) / sigma
    else:
        standardised = s_in - mu

    # Match the scale of the original column
    s_out = df_new[col_out].astype(float)
    mu_out, sigma_out = s_out.mean(), s_out.std()
    if sigma_out > 0:
        rescaled = standardised * sigma_out + mu_out
    else:
        rescaled = standardised + mu_out

    df_new[col_out] = rescaled
    df_new = df_new.drop(columns=[col_in])
    return df_new


def compute_proxy_stability(
    df: pd.DataFrame,
    proxy_groups: Dict[str, List[str]],
    analysis_fn: Callable,
    *args: Any,
    variance_threshold: float = FEATURE_VARIANCE_THRESHOLD_DEFAULT,
    **kwargs: Any,
) -> Dict[str, ProxyStabilityResult]:
    """Test interpretation stability under proxy substitution.

    For each proxy group with >= 2 members, holds one member as the
    "active" proxy and substitutes each alternative in turn.  Re-runs
    analysis_fn on each substitution variant and computes stability
    diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    proxy_groups : dict
        {group_name: [col1, col2, ...]} from find_proxy_groups().
    analysis_fn : callable
        ``analysis_fn(df, *args, **kwargs) -> result`` where result has
        a ``.scores`` attribute (pd.Series).
    *args, **kwargs
        Forwarded to analysis_fn after the DataFrame.
    variance_threshold : float
        Threshold for stability labels.

    Returns
    -------
    dict
        {group_name: ProxyStabilityResult}.
    """
    results: Dict[str, ProxyStabilityResult] = {}

    for group_name, members in proxy_groups.items():
        if len(members) < 2:
            continue

        scores_list: List[pd.Series] = []
        # Base case: use first member as-is (drop others in group)
        active = members[0]
        others = members[1:]

        # Variant 0: original df with other proxies dropped
        df_base = df.drop(columns=[c for c in others if c in df.columns], errors="ignore")
        try:
            r = analysis_fn(df_base, *args, **kwargs)
            scores_list.append(r.scores)
        except Exception as exc:
            log.warning("compute_proxy_stability: base variant for '%s' failed: %s", group_name, exc)

        # Variant i: substitute member[i] for active
        for alt in others:
            if alt not in df.columns or active not in df.columns:
                continue
            df_sub = substitute_proxy(df, active, alt)
            try:
                r = analysis_fn(df_sub, *args, **kwargs)
                scores_list.append(r.scores)
            except Exception as exc:
                log.warning(
                    "compute_proxy_stability: substitution %s→%s failed: %s",
                    active, alt, exc,
                )

        if len(scores_list) < 2:
            log.warning("compute_proxy_stability: group '%s' — fewer than 2 successful; skipping.", group_name)
            continue

        K = len(scores_list)
        score_matrix = _align_scores(scores_list)
        sigma2_f = score_matrix.var(axis=0, ddof=1)
        mean_imp = score_matrix.mean(axis=0)
        mean_abs = score_matrix.abs().mean(axis=0)
        tau_matrix = _pairwise_kendall_tau(score_matrix)
        off_diag = tau_matrix[np.triu_indices_from(tau_matrix, k=1)]
        mean_tau = float(off_diag.mean()) if off_diag.size > 0 else 1.0
        labels = _stability_labels_from_variance(sigma2_f, mean_abs, variance_threshold)

        results[group_name] = ProxyStabilityResult(
            proxy_group=group_name,
            members=tuple(members),
            per_substitution_scores=tuple(scores_list),
            sigma2_f=sigma2_f,
            mean_importance=mean_imp,
            kendall_tau_matrix=tau_matrix,
            mean_kendall_tau=mean_tau,
            stability_labels=labels,
            n_substitutions=K,
        )

    return results


# ---------------------------------------------------------------------------
# SHAP stability (DNV-2.0 Slide 11)
# ---------------------------------------------------------------------------

def compute_shap_stability(
    seal_result: Any,
    target_col: str,
    *,
    top_k: int = TOP_K_OVERLAP_DEFAULT,
    variance_threshold: float = FEATURE_VARIANCE_THRESHOLD_DEFAULT,
    **shap_kwargs: Any,
) -> FeatureStabilityResult:
    """Run SHAP across all SEAL variants and compute stability diagnostics.

    Thin wrapper around compute_feature_stability specialised for SHAP.
    Uses fi_engine.compute_shap as the analysis function.

    Parameters
    ----------
    seal_result : SEALResult
        Valid SEAL result with transformed_dfs.
    target_col : str
        Target column name.
    top_k : int
        Number of top features for overlap computation.
    variance_threshold : float
        Normalised variance threshold for stability labels.
    **shap_kwargs
        Forwarded to fi_engine.compute_shap (n_bg, n_coal, rs).

    Returns
    -------
    FeatureStabilityResult
        With per-feature SHAP variance, Kendall tau, and stability labels.
    """
    import fi_engine as _fi
    return compute_feature_stability(
        seal_result,
        _fi.compute_shap,
        target_col,
        top_k=top_k,
        variance_threshold=variance_threshold,
        **shap_kwargs,
    )


# ---------------------------------------------------------------------------
# Ensemble stability — nested variance decomposition (NPJ paper contribution)
# ---------------------------------------------------------------------------
#
# Existing compute_feature_stability handles K variants × 1 fi_fn.  The NPJ
# paper's novelty is a nested decomposition across K variants × M models:
#
#   sigma2_within_f  = E_m [ Var_k ( phi_kmf ) ]   (preprocessing-driven)
#   sigma2_between_f = Var_m [ E_k ( phi_kmf ) ]   (model-driven)
#
# The helpers below build the (K, M, F) tensor by delegating one call to
# compute_feature_stability per model, then apply the decomposition.


#: Threshold below which sigma2_total_f is flagged "trivially stable" [fraction].
#: Applied to L1-normalised importance, so the scale is comparable across
#: feature-importance families (|coef|, impurity, gain, etc.).
ENSEMBLE_TRIVIAL_STABLE_VAR: Final[float] = 1e-8


@dataclass(frozen=True)
class EnsembleModelSpec:
    """A single (fi_fn, title) entry in the stability ensemble.

    Attributes
    ----------
    key     : machine identifier, e.g. "lasso", "rf", "xgb"
    title   : human-readable label for figures
    fi_fn   : feature-importance callable with signature
              ``fi_fn(df, target_col, **kwargs) -> FeatureRankingResult``
              (matches fi_engine compute_* convention)
    fi_kwargs : dict forwarded verbatim to fi_fn on every variant
    """
    key:       str
    title:     str
    fi_fn:     Callable
    fi_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleStabilityResult:
    """Nested (K × M × F) stability decomposition across a SEAL variant
    family and a multi-model feature-importance ensemble.

    Attributes
    ----------
    model_keys        : tuple of str, len=M
    feature_cols      : tuple of str, len=F (union across all M models)
    phi_tensor        : ndarray shape (K, M, F), L1-normalised per cell
    sigma2_within_f   : pd.Series  — preprocessing-driven variance
    sigma2_between_f  : pd.Series  — model-driven variance
    sigma2_total_f    : pd.Series  — total variance over the K*M grid
    mean_phi_f        : pd.Series  — grand mean (model+variant averaged)
    per_model_results : tuple of FeatureStabilityResult — one per model,
                        re-using the K×1 output of compute_feature_stability
    mean_kendall_tau  : float — mean of per-model mean Kendall tau
    topk_jaccard      : float — top-k Jaccard on the model-averaged ranking
    trivially_stable_flag : pd.Series[bool]
    stability_labels  : Dict[str, str] — family_stable / contingent / trivial
                        derived from sigma2_total_f and mean_phi_f
    n_variants_used   : int — K effective (min across per-model runs)
    is_valid          : bool
    message           : str
    """
    model_keys:            Tuple[str, ...]
    feature_cols:          Tuple[str, ...]
    phi_tensor:            np.ndarray
    sigma2_within_f:       pd.Series
    sigma2_between_f:      pd.Series
    sigma2_total_f:        pd.Series
    mean_phi_f:            pd.Series
    per_model_results:     Tuple[FeatureStabilityResult, ...]
    mean_kendall_tau:      float
    topk_jaccard:          float
    trivially_stable_flag: pd.Series
    stability_labels:      Dict[str, str]
    n_variants_used:       int
    is_valid:              bool
    message:               str


def _l1_normalise_row(v: np.ndarray) -> np.ndarray:
    """|v| / sum(|v|); all-zero rows stay all-zero."""
    abs_v = np.abs(v)
    s = float(abs_v.sum())
    if s <= 0.0:
        return np.zeros_like(abs_v)
    return abs_v / s


def _pairwise_jaccard_topk(matrix_K_by_F: np.ndarray, k: int) -> float:
    """Mean pairwise Jaccard overlap of the top-k feature index sets
    across the K rows."""
    K, F = matrix_K_by_F.shape
    if K <= 1:
        return 1.0
    k_eff = min(k, F)
    sets: List[set] = []
    for i in range(K):
        top_idx = np.argsort(-matrix_K_by_F[i], kind="mergesort")[:k_eff]
        sets.append(set(int(x) for x in top_idx))
    total = 0.0
    pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            total += (inter / union) if union > 0 else 1.0
            pairs += 1
    return float(total / pairs) if pairs > 0 else 1.0


def compute_ensemble_stability(
    seal_result:          Any,
    target_col:           str,
    model_specs:          Sequence[EnsembleModelSpec],
    *,
    top_k:                int   = TOP_K_OVERLAP_DEFAULT,
    variance_threshold:   float = FEATURE_VARIANCE_THRESHOLD_DEFAULT,
) -> EnsembleStabilityResult:
    """Run the nested (K × M × F) stability decomposition.

    For each model spec, delegates to compute_feature_stability to obtain
    the per-variant importance series, then assembles a (K, M, F) tensor
    with L1-normalised rows and performs the nested-variance split.

    Parameters
    ----------
    seal_result : SEALResult
    target_col  : name of the regression target column
    model_specs : ensemble of EnsembleModelSpec instances
    top_k       : depth for Jaccard overlap shortlist
    variance_threshold : forwarded to each compute_feature_stability call

    Returns
    -------
    EnsembleStabilityResult
    """
    specs = tuple(model_specs)
    M = len(specs)
    if M == 0:
        raise ValueError("model_specs must be non-empty")

    per_model: List[FeatureStabilityResult] = []
    per_model_matrices: List[pd.DataFrame] = []  # each (K × F)

    for spec in specs:
        res_m = compute_feature_stability(
            seal_result,
            spec.fi_fn,
            target_col,
            top_k=top_k,
            variance_threshold=variance_threshold,
            **spec.fi_kwargs,
        )
        per_model.append(res_m)
        mat = _align_scores(list(res_m.per_variant_scores))
        per_model_matrices.append(mat)

    # Union of features over all models
    all_feats: List[str] = []
    seen = set()
    for mat in per_model_matrices:
        for c in mat.columns:
            if c not in seen:
                seen.add(c)
                all_feats.append(c)
    feat = tuple(all_feats)
    F = len(feat)

    K_eff = min(mat.shape[0] for mat in per_model_matrices)
    if K_eff <= 0 or F == 0:
        raise RuntimeError(
            f"compute_ensemble_stability: insufficient data (K={K_eff}, F={F})"
        )

    # Build (K, M, F) tensor, L1-normalised per (k, m) row
    phi_tensor = np.zeros((K_eff, M, F), dtype=float)
    for m_idx, mat in enumerate(per_model_matrices):
        mat_aligned = mat.reindex(columns=feat, fill_value=MISSING_SCORE_FILL)
        arr = mat_aligned.values[:K_eff, :].astype(float)
        for k in range(K_eff):
            phi_tensor[k, m_idx, :] = _l1_normalise_row(arr[k])

    # Nested variance decomposition
    var_k_phi      = np.var(phi_tensor, axis=0, ddof=0)          # (M, F)
    sigma2_within  = np.mean(var_k_phi, axis=0)                  # (F,)
    mean_k_phi     = np.mean(phi_tensor, axis=0)                 # (M, F)
    sigma2_between = np.var(mean_k_phi, axis=0, ddof=0)          # (F,)
    flat           = phi_tensor.reshape(K_eff * M, F)
    sigma2_total   = np.var(flat, axis=0, ddof=0)                # (F,)
    mean_phi       = np.mean(flat, axis=0)                       # (F,)

    # Top-k Jaccard on the model-averaged, per-variant ranking
    ranking_K_by_F = np.mean(phi_tensor, axis=1)                 # (K, F)
    topk_jac = _pairwise_jaccard_topk(ranking_K_by_F, top_k)

    # Aggregate Kendall tau: mean across per-model mean_kendall_tau
    mean_tau = float(np.mean([r.mean_kendall_tau for r in per_model]))

    # Stability labels per feature (reuse existing threshold logic)
    sigma2_total_s = pd.Series(sigma2_total, index=feat)
    mean_phi_s     = pd.Series(mean_phi,     index=feat)
    labels = _stability_labels_from_variance(
        sigma2_total_s, mean_phi_s.abs(), variance_threshold,
    )

    trivial_flag = pd.Series(
        sigma2_total < ENSEMBLE_TRIVIAL_STABLE_VAR,
        index=feat, name="trivially_stable_flag",
    )

    msg = (
        f"ensemble stability: K={K_eff}, M={M}, F={F} | "
        f"mean Kendall tau={mean_tau:.3f} | "
        f"top-{top_k} Jaccard={topk_jac:.3f}"
    )

    return EnsembleStabilityResult(
        model_keys            = tuple(s.key for s in specs),
        feature_cols          = feat,
        phi_tensor            = phi_tensor,
        sigma2_within_f       = pd.Series(sigma2_within,  index=feat, name="sigma2_within_f"),
        sigma2_between_f      = pd.Series(sigma2_between, index=feat, name="sigma2_between_f"),
        sigma2_total_f        = sigma2_total_s.rename("sigma2_total_f"),
        mean_phi_f            = mean_phi_s.rename("mean_phi_f"),
        per_model_results     = tuple(per_model),
        mean_kendall_tau      = mean_tau,
        topk_jaccard          = topk_jac,
        trivially_stable_flag = trivial_flag,
        stability_labels      = labels,
        n_variants_used       = K_eff,
        is_valid              = True,
        message               = msg,
    )


def default_ensemble_specs() -> Tuple[EnsembleModelSpec, ...]:
    """Return the NPJ paper's default ensemble: LASSO + RF + XGBoost.

    Falls back to (LASSO, RF) if xgboost cannot be imported.  Uses
    fi_engine.compute_* functions so the ensemble is backed by the
    already-wired feature-importance registry.
    """
    import fi_engine as _fi
    specs: List[EnsembleModelSpec] = [
        EnsembleModelSpec("lasso", "LASSO",          _fi.compute_lasso,         {}),
        EnsembleModelSpec("rf",    "Random Forest",  _fi.compute_random_forest, {}),
    ]
    try:
        import xgboost  # noqa: F401
        if hasattr(_fi, "compute_xgboost"):
            specs.append(EnsembleModelSpec(
                "xgb", "XGBoost", _fi.compute_xgboost, {},
            ))
        else:
            log.warning("fi_engine.compute_xgboost missing; ensemble = (LASSO, RF)")
    except ImportError:
        log.warning("xgboost not installed; ensemble = (LASSO, RF)")
    return tuple(specs)
