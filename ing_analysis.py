"""
DNV Scientific Module
---------------------
Role:
    Provides schema inference, basic statistics, and full integrity-report
    assembly for a loaded DataFrame.

Scientific Context:
    Operates on raw DataFrames to produce typed schema records, descriptive
    statistics, and an integrity report covering missing values, outliers,
    duplicates, mixed types, and near-constant columns.

Invariants:
    - infer_schema always returns a dict with 'rows', 'cols', and 'columns'
      keys.
    - outlier_counts_iqr reports per-column counts only for numeric columns
      with IQR > 0.

Assumptions:
    - Input DataFrame has at least one column.
    - IngestionConfig outlier_iqr_factor is positive.

Failure Modes:
    - All-NaN column: statistics are omitted for that column without error.
    - Zero IQR column: outlier detection is skipped for that column.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ing_config import IngestionConfig
from ing_provenance import (
    SCHEMA_VERSION, dataframe_sha256, file_bytes,
    json_safe, new_analysis_id, sha256_of_file, utc_now_iso,
)


# ---------------------------------------------------------------------------
# Named constants — all values carry explicit units or semantics in name
# ---------------------------------------------------------------------------

#: Conversion factor: bytes → megabytes.
BYTES_PER_MB: Final[float] = 1e6

#: Minimum non-null values required for IQR outlier detection.
OUTLIER_MIN_NONNULL: Final[int] = 4

#: Fraction of total cells above which a missingness warning fires.
HIGH_MISSINGNESS_FRAC: Final[float] = 0.05

#: Fraction of rows above which a duplicate-rows warning fires.
HIGH_DUPLICATE_FRAC: Final[float] = 0.01

#: Maximum unique-value count considered "constant" (inclusive).
CONSTANT_NUNIQUE_THRESHOLD: Final[int] = 1

#: Minimum sample size for reporting skewness and kurtosis.
MOMENT_MIN_SAMPLE_SIZE: Final[int] = 8

#: Robust quantile breakpoints used for admissibility envelope diagnostics.
ROBUST_QUANTILES: Final[Tuple[float, ...]] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

#: Bessel correction parameter for sample standard deviation (ddof=1).
SAMPLE_STD_DDOF: Final[int] = 1


def infer_schema(df: pd.DataFrame,
                 cfg: Optional[IngestionConfig] = None) -> Dict[str, Any]:
    cfg = cfg or IngestionConfig()
    n   = len(df)
    sample = df.head(min(cfg.sample_rows_for_inference, n))
    cols: Dict[str, Any] = {}
    for c in df.columns:
        col  = df[c];  scol = sample[c]
        missing = int(col.isna().sum())
        entry: Dict[str, Any] = {
            "dtype":            str(col.dtype),
            "non_null":         int(col.notna().sum()),
            "missing":          missing,
            "missing_fraction": min(1.0, max(0.0, float(missing / max(n, 1)))),
            "unique_non_null":  int(col.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(col):
            nn = scol.dropna()
            if len(nn) > 0:
                entry.update(sample_min=json_safe(nn.min()),
                             sample_max=json_safe(nn.max()),
                             sample_mean=json_safe(float(nn.mean())),
                             sample_std=json_safe(float(nn.std(ddof=SAMPLE_STD_DDOF))),
                             sample_std_ddof=SAMPLE_STD_DDOF)
        else:
            vc = col.dropna().value_counts(dropna=True)
            top_n = min(cfg.max_categories_per_column, len(vc))
            if top_n > 0:
                entry["top_categories"] = {str(k): int(v) for k, v in vc.head(top_n).items()}
        cols[str(c)] = entry
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": cols}


def basic_statistics(df: pd.DataFrame,
                     cfg: Optional[IngestionConfig] = None) -> Dict[str, Any]:
    cfg = cfg or IngestionConfig()
    num = df.select_dtypes(include=["number"])
    n_rows = int(df.shape[0])
    out: Dict[str, Any] = {
        "rows": n_rows, "columns": int(df.shape[1]),
        "memory_mb": float(round(df.memory_usage(deep=True).sum() / BYTES_PER_MB, 3)),
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "numeric_columns": int(num.shape[1]),
        "non_numeric_columns": int(df.shape[1] - num.shape[1]),
    }
    if not num.empty:
        n = num.iloc[:, :cfg.max_numeric_describe_cols]
        out["numeric_describe"] = {
            str(k): {str(kk): json_safe(vv) for kk, vv in v.items()}
            for k, v in n.describe(percentiles=list(ROBUST_QUANTILES)).to_dict().items()
        }
        # Skewness and kurtosis (trimmed-moment bounds, DNV-1.0 §1 p.3)
        if n_rows >= MOMENT_MIN_SAMPLE_SIZE:
            out["skewness"] = {
                str(c): json_safe(float(n[c].dropna().skew()))
                for c in n.columns if n[c].dropna().shape[0] >= MOMENT_MIN_SAMPLE_SIZE
            }
            out["kurtosis"] = {
                str(c): json_safe(float(n[c].dropna().kurtosis()))
                for c in n.columns if n[c].dropna().shape[0] >= MOMENT_MIN_SAMPLE_SIZE
            }
    return out


# ── integrity-report helpers ─────────────────────────────────────────────────

def outlier_counts_iqr(df: pd.DataFrame, factor: float, max_cols: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for c in df.select_dtypes(include=["number"]).columns[:max_cols]:
        col = df[c].dropna()
        if len(col) < OUTLIER_MIN_NONNULL: continue
        q1, q3 = float(col.quantile(0.25)), float(col.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0.0: continue
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        low, high = int((col < lo).sum()), int((col > hi).sum())
        if low + high > 0:
            result[str(c)] = {"low": low, "high": high, "total": low + high,
                              "fraction_of_nonnull": float((low + high) / max(len(col), 1)),
                              "fence_lo": json_safe(lo), "fence_hi": json_safe(hi)}
    return result


def _mixed_type_columns(df: pd.DataFrame) -> List[Tuple[str, int, int]]:
    mixed = []
    for c in df.select_dtypes(include=["object", "string"]).columns:
        nn = df[c].dropna().astype(str)
        if len(nn) == 0: continue
        n_num = int(pd.to_numeric(nn, errors="coerce").notna().sum())
        n_non = int(len(nn) - n_num)
        if n_num > 0 and n_non > 0:
            mixed.append((str(c), n_num, n_non))
    return mixed


def integrity_report(df: pd.DataFrame,
                     path: Optional[Union[str, Path]] = None,
                     cfg:  Optional[IngestionConfig]  = None) -> Dict[str, Any]:
    cfg = cfg or IngestionConfig()
    n   = max(len(df), 1)

    missing_cells   = int(df.isna().sum().sum())
    missing_frac    = float(missing_cells / (n * max(df.shape[1], 1)))
    rows_w_missing  = int(df.isna().any(axis=1).sum())
    mbc_raw         = df.isna().sum()
    cols_w_missing  = sorted([str(c) for c in mbc_raw[mbc_raw > 0].index], key=lambda c: -int(mbc_raw[c]))
    missing_by_col  = {str(c): int(mbc_raw[c]) for c in cols_w_missing}

    dup_rows = int(df.duplicated().sum())
    dup_frac = float(dup_rows / n)
    dup_cols = sorted([str(c) for c in df.columns[df.columns.duplicated()].tolist()])

    all_null  = [str(c) for c in df.columns[df.isna().all()].tolist()]
    constant  = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= CONSTANT_NUNIQUE_THRESHOLD]
    obj_cols  = df.select_dtypes(include=["object", "string"]).columns.tolist()
    hi_card   = [str(c) for c in obj_cols[:cfg.max_report_hicard_cols]
                 if len(df) > 0 and int(df[c].nunique(dropna=True)) > cfg.high_cardinality_frac * len(df)]
    cand_keys = [str(c) for c in df.columns
                 if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
                     or pd.api.types.is_integer_dtype(df[c]))
                 and (lambda s: len(s) > 0 and s.nunique(dropna=True) / len(s) >= cfg.high_cardinality_frac)(df[c].dropna())]

    key_coll: Dict[str, Any] = {}
    for c in cand_keys[:10]:
        nn = df[c].dropna()
        if len(nn) == 0: continue
        nun = int(nn.nunique(dropna=True)); coll = int(len(nn) - nun)
        key_coll[str(c)] = {"nonnull": int(len(nn)), "unique": nun, "collisions": coll,
                            "collision_rate_over_nonnull": float(coll / max(len(nn), 1)),
                            "nonnull_rate": float(len(nn) / n)}

    schema  = infer_schema(df, cfg=cfg)
    numeric = df.select_dtypes(include=["number"])
    num_qc: Optional[Dict[str, Any]] = None
    if not numeric.empty:
        num = numeric.iloc[:, :cfg.max_numeric_describe_cols]
        num_qc = {
            "describe": {str(k): {str(kk): json_safe(vv) for kk, vv in v.items()}
                         for k, v in num.describe().to_dict().items()},
            "has_infinite": {str(c): bool(np.isinf(num[c].to_numpy(dtype=float, copy=False)).any())
                             for c in num.columns},
        }

    outliers     = outlier_counts_iqr(df, cfg.outlier_iqr_factor, cfg.max_numeric_describe_cols)
    empty_strs   = {str(c): int((df[c].notna() & (df[c].astype(str).str.strip() == "")).sum())
                    for c in obj_cols if int((df[c].notna() & (df[c].astype(str).str.strip() == "")).sum()) > 0}
    mixed_raw    = _mixed_type_columns(df)
    mixed_cols   = {col: {"n_numeric": nm, "n_non_numeric": nn} for col, nm, nn in mixed_raw}

    import dataclasses as _dc
    meta: Dict[str, Any] = {
        "ingested_utc": utc_now_iso(), "analysis_id": new_analysis_id(),
        "schema_version": dict(SCHEMA_VERSION), "dataframe_sha256": dataframe_sha256(df),
        "config": _dc.asdict(cfg),
        "file": None, "file_bytes": None, "sha256": None, "pipeline_history": [],
    }
    if path is not None:
        p = Path(path); meta["file"] = p.name
        try: meta["file_bytes"] = int(file_bytes(p))
        except OSError: pass
        try: meta["sha256"] = sha256_of_file(p)
        except OSError: pass

    warnings: List[str] = []
    notes:    List[str] = []
    if missing_frac > HIGH_MISSINGNESS_FRAC:
        warnings.append(f"High missingness: {missing_frac:.2%} of all cells are NA.")
    if len(df) > 0 and dup_rows / len(df) > HIGH_DUPLICATE_FRAC:
        warnings.append(f"Non-trivial duplicate rows: {dup_frac:.2%}.")
    if dup_cols:
        warnings.append("Duplicate column names detected.")
    if all_null:
        warnings.append(f"All-null columns ({len(all_null)}): {', '.join(all_null[:10])}" +
                        (" …" if len(all_null) > 10 else "") + ".")
    if constant:
        notes.append("Constant columns present — verify before dropping.")
    if cand_keys:
        notes.append("Candidate key columns detected (near-unique values).")
    if outliers:
        notes.append(f"Tukey-fence outliers in {len(outliers)} numeric column(s) (IQR × {cfg.outlier_iqr_factor}).")
    if empty_strs:
        warnings.append(f"Empty-string cells in {len(empty_strs)} object column(s). Not captured as NA by default.")
    if mixed_cols:
        warnings.append("Mixed-type columns (numeric + string): " +
                        ", ".join(list(mixed_cols.keys())[:10]) + ".")

    suspect_re = re.compile(
        r"(target|label|y$|voltage|capacity|energy|formation|hull|stability|bandgap|e[_ ]?above[_ ]?hull)", re.I)
    leakage = sorted({str(c) for c in df.columns if suspect_re.search(str(c))})
    if leakage:
        notes.append("Leakage suspects are heuristic — confirm before modelling.")

    return {
        "meta": meta,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "schema": schema,
        "missingness": {
            "missing_cells": int(missing_cells), "missing_fraction": float(missing_frac),
            "rows_with_missing": int(rows_w_missing),
            "columns_with_missing": cols_w_missing[:cfg.max_report_missing_cols],
            "missing_by_column": missing_by_col,
        },
        "duplicates": {"duplicate_rows": dup_rows, "duplicate_fraction": dup_frac, "duplicate_columns": dup_cols},
        "structure": {
            "all_null_columns": all_null, "constant_columns": constant[:cfg.max_report_constant_cols],
            "high_cardinality_object_columns": hi_card,
            "candidate_key_columns": cand_keys[:25], "candidate_key_collision_stats": key_coll,
        },
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "numeric_quickcheck": num_qc,
        "outliers": outliers,
        "empty_strings": empty_strs,
        "mixed_type_columns": mixed_cols,
        "leakage_suspects": leakage[:50],
        "warnings": warnings,
        "notes": notes,
    }
