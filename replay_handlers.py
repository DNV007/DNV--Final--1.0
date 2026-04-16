"""
DNV Scientific Module
---------------------
Role:
    Registers provenance replay handlers for all pipeline operations.
    Importing this module populates ing_provenance._REPLAY_REGISTRY so
    that replay_pipeline() can re-execute recorded operation sequences.

Scientific Context:
    DNV-2.0 Slide 3: "Provenance: replayable logs."  Each operation
    recorded in pipeline_history is mapped to a pure function that
    re-applies it given the stored params dict.

Invariants:
    - Every handler has signature fn(df, params) -> df.
    - Handlers never modify the input DataFrame in place.
    - Handlers use only the params dict stored in ProvenanceRecord;
      no external state.

Assumptions:
    - params dicts match the schema written by the GUI at apply time.
    - numpy, pandas, sklearn are available.

Failure Modes:
    - Missing or malformed params: handler raises, replay_pipeline skips
      with a warning.
    - SEAL conditioning: not replayable (stochastic search); logged as skip.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Final, List

import numpy as np
import pandas as pd

from ing_provenance import register_replay

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay defaults — fallback values matching the schema written at apply time
# ---------------------------------------------------------------------------

#: Default IQR multiplier for outlier detection [dimensionless].
DEFAULT_IQR_FACTOR: Final[float] = 1.5

#: Default z-score threshold for outlier detection [dimensionless].
DEFAULT_ZSCORE_THRESHOLD: Final[float] = 3.0

#: Default SEAL soft-clamp smoothness [dimensionless fraction of range].
DEFAULT_SOFT_SMOOTHNESS: Final[float] = 0.05


# ---------------------------------------------------------------------------
# Replay handlers — each takes (df, params) -> df
# ---------------------------------------------------------------------------

def _replay_handle_missing_values(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay missing-value imputation."""
    method = str(params.get("method", "drop_rows")).strip().lower()
    df = df.copy()

    if method == "drop_rows":
        return df.dropna()
    elif method == "drop_columns":
        thresh = params.get("threshold")
        if thresh is not None:
            return df.dropna(axis=1, thresh=int(thresh))
        return df.dropna(axis=1)
    elif method in ("mean", "median", "mode"):
        num = df.select_dtypes(include="number")
        if method == "mean":
            df[num.columns] = num.fillna(num.mean())
        elif method == "median":
            df[num.columns] = num.fillna(num.median())
        elif method == "mode":
            modes = num.mode().iloc[0] if not num.mode().empty else num.mean()
            df[num.columns] = num.fillna(modes)
        return df
    elif method == "constant":
        fill_value = params.get("fill_value", 0)
        return df.fillna(fill_value)
    elif method in ("linear", "polynomial", "spline", "akima",
                     "pchip", "barycentric", "krogh"):
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].interpolate(method=method)
        return df
    else:
        log.warning("_replay_handle_missing_values: unknown method '%s'", method)
        return df


def _replay_manage_columns(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay column selection / reordering."""
    columns_final = params.get("columns_final", [])
    if columns_final:
        available = [c for c in columns_final if c in df.columns]
        return df[available].copy()
    return df.copy()


def _replay_remove_outliers(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay outlier removal (IQR or z-score method)."""
    method = str(params.get("method", "iqr")).strip().lower()
    columns = params.get("columns", [])
    df = df.copy()

    if not columns:
        columns = df.select_dtypes(include="number").columns.tolist()

    if method == "iqr":
        factor = float(params.get("factor", DEFAULT_IQR_FACTOR))
        for col in columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (df[col] >= q1 - factor * iqr) & (df[col] <= q3 + factor * iqr)
            df = df[mask | df[col].isna()]
    elif method == "zscore":
        threshold = float(params.get("threshold", DEFAULT_ZSCORE_THRESHOLD))
        for col in columns:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            z = (s - s.mean()) / s.std()
            mask = z.abs() <= threshold
            full_mask = pd.Series(True, index=df.index)
            full_mask.loc[s.index] = mask
            df = df[full_mask | df[col].isna()]

    return df


def _replay_normalize_scale(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay normalization / scaling."""
    method = str(params.get("method", "standardize")).strip().lower()
    columns = params.get("columns", [])
    df = df.copy()

    if not columns:
        columns = df.select_dtypes(include="number").columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        if method == "standardize":
            mean, std = s.mean(), s.std()
            if std > 0:
                df[col] = (s - mean) / std
        elif method == "minmax":
            lo, hi = s.min(), s.max()
            rng = hi - lo
            if rng > 0:
                df[col] = (s - lo) / rng
        elif method == "robust":
            median = s.median()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[col] = (s - median) / iqr
        elif method == "log":
            df[col] = np.log1p(s.clip(lower=0))
        elif method == "sqrt":
            df[col] = np.sqrt(s.clip(lower=0))

    return df


def _replay_create_derived_column(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay derived column creation via expression evaluation."""
    expression = str(params.get("expression", "")).strip()
    new_name = str(params.get("new_column_name", "derived")).strip()
    if not expression:
        return df.copy()
    df = df.copy()
    try:
        df[new_name] = df.eval(expression)
    except Exception as exc:
        log.warning("_replay_create_derived_column: eval failed: %s", exc)
    return df


def _replay_transform_columns(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay column transformation (log, sqrt, etc.)."""
    transform_type = str(params.get("transform_type", "")).strip().lower()
    columns = params.get("columns", [])
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        if transform_type == "log":
            df[col] = np.log(s)
        elif transform_type == "log1p":
            df[col] = np.log1p(s)
        elif transform_type == "sqrt":
            df[col] = np.sqrt(s)
        elif transform_type == "square":
            df[col] = s ** 2
        elif transform_type == "reciprocal":
            df[col] = 1.0 / s.replace(0, np.nan)
        elif transform_type == "standardize":
            mean, std = s.mean(), s.std()
            if std > 0:
                df[col] = (s - mean) / std
        elif transform_type == "minmax":
            lo, hi = s.min(), s.max()
            rng = hi - lo
            if rng > 0:
                df[col] = (s - lo) / rng
        else:
            log.warning("_replay_transform_columns: unknown type '%s'", transform_type)

    return df


def _replay_seal_conditioning(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay SEAL conditioning deterministically from stored transform_params.

    The stored params contain the variant's per-column (lo_val, hi_val) and
    the transform_type used.  For "winsorize" we apply hard clip; for
    "soft_clamp" we apply the same tanh-based smooth clamping used by
    seal_engine._apply_soft_clamp.
    """
    transform_params = params.get("transform_params", {})
    transform_type = str(params.get("transform_type", "winsorize")).strip().lower()
    smoothness = float(params.get("soft_smoothness", DEFAULT_SOFT_SMOOTHNESS))
    df = df.copy()

    for col, col_params in transform_params.items():
        if col not in df.columns:
            continue
        lo_val = col_params.get("lo_val")
        hi_val = col_params.get("hi_val")
        if lo_val is not None and hi_val is not None:
            x = df[col].astype(float).values
            nan_mask = np.isnan(x)
            if transform_type == "soft_clamp":
                rng = float(hi_val) - float(lo_val)
                if abs(rng) < 1e-12:
                    x_t = x.copy()
                else:
                    mid = (float(lo_val) + float(hi_val)) / 2.0
                    denom = smoothness * rng
                    if abs(denom) < 1e-12:
                        denom = 1e-12
                    x_t = float(lo_val) + rng * (
                        0.5 + 0.5 * np.tanh((x - mid) / denom)
                    )
            else:
                x_t = np.clip(x, float(lo_val), float(hi_val))
            # Preserve NaN positions
            x_t[nan_mask] = np.nan
            df[col] = x_t

    return df


# ---------------------------------------------------------------------------
# Register all handlers on import
# ---------------------------------------------------------------------------

register_replay("handle_missing_values", _replay_handle_missing_values)
register_replay("manage_columns", _replay_manage_columns)
register_replay("remove_outliers", _replay_remove_outliers)
register_replay("normalize_scale", _replay_normalize_scale)
register_replay("create_derived_column", _replay_create_derived_column)
register_replay("transform_columns", _replay_transform_columns)
register_replay("seal_conditioning", _replay_seal_conditioning)


def _replay_moment_opt_outliers(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Replay moment-optimized outlier removal.

    Re-runs the optimizer with stored params so the same (or comparable)
    mask is produced deterministically.
    """
    import outlier_opt_engine as _opt

    column = str(params.get("column", "")).strip()
    if not column or column not in df.columns:
        log.warning("_replay_moment_opt_outliers: column %r not found; skipping.", column)
        return df.copy()

    result = _opt.optimize_outliers(
        df, column,
        optimizer=str(params.get("optimizer", "bisection")),
        base_method=str(params.get("base_method", "iqr")),
        target_kurtosis=float(params.get("target_kurtosis", 0.0)),
        target_skewness=float(params.get("target_skewness", 0.0)),
        w_kurtosis=float(params.get("w_kurtosis", 1.0)),
        w_skewness=float(params.get("w_skewness", 1.0)),
        w_retention=float(params.get("w_retention", 0.5)),
        min_retention=float(params.get("min_retention", 0.5)),
        sa_max_iter=int(params.get("sa_hc_iterations", 2000)),
        hc_max_iter=int(params.get("sa_hc_iterations", 1000)),
    )
    return df[result.mask].reset_index(drop=True)


register_replay("moment_opt_outliers", _replay_moment_opt_outliers)
