"""
DNV Scientific Module
---------------------
Role:
    Provides DataIngestorV2 and format_table — plain-text rendering of
    ingestion diagnostics for the DataIngestionWindow report panel.

Scientific Context:
    Formats DataFrame previews, schema summaries, statistical tables, and
    outlier reports as fixed-width plain text for display in the
    ReportTextPanel widget; no scientific computation.

Invariants:
    - format_table always returns a non-empty string even for empty
      DataFrames.
    - DataIngestorV2.render always returns a complete multi-section report
      string.

Assumptions:
    - outlier_counts_iqr is available from ing_analysis.
    - DataAsset has non-None dataframe, metadata, and statistics fields.

Failure Modes:
    - Column value exceeds col_width: truncated with ellipsis suffix.
    - DataFrame larger than preview_max_bytes: truncated at max_rows with
      a notice appended.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Final, List, Optional

import numpy as np
import pandas as pd

from ing_config   import IngestionConfig
from ing_analysis import outlier_counts_iqr, MOMENT_MIN_SAMPLE_SIZE, ROBUST_QUANTILES

# ---------------------------------------------------------------------------
# Named constants for float formatting thresholds
# ---------------------------------------------------------------------------

#: Below this absolute value, use scientific notation for display.
FMT_SCI_LO: Final[float] = 1e-4

#: At or above this absolute value, use scientific notation for display.
FMT_SCI_HI: Final[float] = 1e4

#: Number of significant digits for general float display.
FMT_SIGFIGS: Final[int] = 6

#: Number of decimal places for scientific notation.
FMT_SCI_DECIMALS: Final[int] = 3


def format_table(df: pd.DataFrame, max_rows: int = 50, max_cols: int = 20,
                 col_width: int = 24, preview_max_bytes: int = 2_000_000) -> str:
    if df is None or df.empty:
        return "(empty)"
    view = df.copy()
    tr_cols = view.shape[1] > max_cols
    tr_rows = view.shape[0] > max_rows
    if tr_cols: view = view.iloc[:, :max_cols]
    if tr_rows: view = view.head(max_rows)

    def _fmt(x: Any) -> str:
        try:
            if x is None or x is pd.NA: return ""
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return str(x)
            if isinstance(x, (int, np.integer)):   return f"{int(x):,}"
            if isinstance(x, (float, np.floating)):
                ax = abs(float(x))
                if ax == 0: return "0"
                return (f"{float(x):.{FMT_SCI_DECIMALS}e}"
                        if ax < FMT_SCI_LO or ax >= FMT_SCI_HI
                        else f"{float(x):.{FMT_SIGFIGS}g}")
            return str(x)
        except Exception:
            return str(x)

    for c in view.columns:
        view[c] = view[c].map(_fmt)
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_cols,
                           "display.width", 240, "display.colheader_justify", "left",
                           "display.expand_frame_repr", False, "display.max_colwidth", col_width):
        s = view.to_string(index=True)
    if tr_cols: s += "\n… (columns truncated)"
    if tr_rows: s += "\n… (rows truncated)"
    if len(s.encode("utf-8")) > preview_max_bytes:
        s = s[:max(1_000, int(preview_max_bytes * 0.8))] + "\n… (table truncated by byte limit)"
    return s


def _section(title: str) -> str:
    return f"{title}\n{'=' * 78}"


class DataIngestorV2:
    """Stateless convenience wrapper for GUI consumers — no Qt objects held."""

    def __init__(self, cfg: Optional[IngestionConfig] = None):
        self.cfg = cfg or IngestionConfig()

    def basic_stats_text(self, df: pd.DataFrame) -> str:
        cfg = self.cfg
        n   = df.shape[0]

        overview = pd.DataFrame(
            {"value": [n, int(df.shape[1]),
                       float(df.memory_usage(deep=True).sum() / 1e6),
                       int(df.select_dtypes(include=["number"]).shape[1]),
                       int(df.select_dtypes(exclude=["number"]).shape[1])]},
            index=["rows", "columns", "memory_mb", "numeric_columns", "non_numeric_columns"],
        )
        dtype_tbl = df.dtypes.astype(str).value_counts().to_frame(name="count")

        miss     = df.isna().sum().sort_values(ascending=False)
        miss     = miss[miss > 0]
        miss_tbl = pd.DataFrame({"missing_count": miss.astype(int),
                                  "missing_frac": (miss / max(n, 1)).astype(float)})
        miss_tbl.index.name = "column"

        dup_mask = df.duplicated()
        dup_tbl  = pd.DataFrame(
            {"value": [int(dup_mask.sum()), float(dup_mask.mean()) if n else 0.0,
                       int(df.columns.duplicated().sum())]},
            index=["duplicate_rows", "duplicate_rows_frac", "duplicate_column_names"],
        )
        num = df.select_dtypes(include=["number"])
        numeric_block = "(no numeric columns)"
        skew_kurt_block = ""
        if not num.empty:
            num2  = num.iloc[:, :cfg.max_numeric_describe_cols]
            desc  = num2.describe(percentiles=list(ROBUST_QUANTILES)).T
            keep  = [c for c in ["count","mean","std","min","1%","5%","25%","50%","75%","95%","99%","max"]
                     if c in desc.columns]
            desc  = desc[keep].replace([np.inf, -np.inf], np.nan).round(FMT_SIGFIGS)
            numeric_block = format_table(desc, max_rows=25, max_cols=14,
                                         col_width=22, preview_max_bytes=cfg.preview_max_bytes)
            # Skewness and kurtosis (trimmed-moment bounds, DNV-1.0 §1 p.3)
            if n >= MOMENT_MIN_SAMPLE_SIZE:
                moment_cols = [c for c in num2.columns if num2[c].dropna().shape[0] >= MOMENT_MIN_SAMPLE_SIZE]
                if moment_cols:
                    sk = pd.DataFrame({
                        "skewness": {str(c): round(float(num2[c].dropna().skew()), 4) for c in moment_cols},
                        "kurtosis": {str(c): round(float(num2[c].dropna().kurtosis()), 4) for c in moment_cols},
                    })
                    sk.index.name = "column"
                    skew_kurt_block = format_table(sk, max_rows=25, max_cols=4,
                                                    col_width=22, preview_max_bytes=cfg.preview_max_bytes)

        outlier_data = outlier_counts_iqr(df, cfg.outlier_iqr_factor, cfg.max_numeric_describe_cols)
        outlier_block = "(no outliers detected)"
        if outlier_data:
            otbl = pd.DataFrame([
                {"column": c, "low": v["low"], "high": v["high"], "total": v["total"],
                 "frac_nonnull": round(v["fraction_of_nonnull"], 4)}
                for c, v in outlier_data.items()
            ]).set_index("column")
            outlier_block = format_table(otbl, max_rows=30, max_cols=6,
                                         col_width=20, preview_max_bytes=cfg.preview_max_bytes)

        n_num_shown = min(cfg.max_numeric_describe_cols, num.shape[1] if not num.empty else 0)
        parts: List[str] = [
            _section("Basic Statistics"),
            "\nOverview:",
            format_table(overview, max_rows=10, max_cols=2, col_width=24, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nDTypes:",
            format_table(dtype_tbl, max_rows=50, max_cols=3, col_width=24, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nMissingness (top 25):",
            format_table(miss_tbl.head(25), max_rows=25, max_cols=4, col_width=32, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nDuplicates:",
            format_table(dup_tbl, max_rows=10, max_cols=2, col_width=32, preview_max_bytes=cfg.preview_max_bytes),
            f"\n\nNumeric summary — robust quantiles (first {n_num_shown} columns):",
            numeric_block,
        ]
        if skew_kurt_block:
            parts += [
                f"\n\nSkewness & Kurtosis (n ≥ {MOMENT_MIN_SAMPLE_SIZE}):",
                skew_kurt_block,
            ]
        parts += [
            f"\n\nOutliers — Tukey fence (IQR × {cfg.outlier_iqr_factor}):",
            outlier_block,
        ]
        return "\n".join(parts)

    def integrity_text(self, rep: Dict[str, Any]) -> str:
        import json as _json
        cfg    = self.cfg
        meta   = rep.get("meta", {})
        shape  = rep.get("shape", {})
        miss   = rep.get("missingness", {})
        dups   = rep.get("duplicates", {})
        struct = rep.get("structure", {})

        meta_tbl = pd.DataFrame(
            {"value": [meta.get("file"), meta.get("ingested_utc"), meta.get("analysis_id"),
                       _json.dumps(meta.get("schema_version"), ensure_ascii=False),
                       meta.get("dataframe_sha256"), meta.get("file_bytes"), meta.get("sha256")]},
            index=["file","ingested_utc","analysis_id","schema_version","dataframe_sha256","file_bytes","sha256"],
        )
        miss_tbl = pd.DataFrame(
            {"value": [miss.get("missing_cells",0), miss.get("missing_fraction",0.0),
                       miss.get("rows_with_missing",0), len(miss.get("columns_with_missing",[]) or [])]},
            index=["missing_cells","missing_fraction","rows_with_missing","cols_with_missing"],
        )
        mbc = miss.get("missing_by_column", {})
        miss_top = pd.DataFrame()
        if isinstance(mbc, dict):
            ms = pd.Series(mbc).astype(int).sort_values(ascending=False)
            miss_top = ms[ms > 0].head(25).to_frame(name="missing_count")
            miss_top.index.name = "column"

        dup_tbl = pd.DataFrame(
            {"value": [dups.get("duplicate_rows",0), dups.get("duplicate_fraction",0.0),
                       len(dups.get("duplicate_columns",[]) or [])]},
            index=["duplicate_rows","duplicate_fraction","duplicate_column_names"],
        )
        struct_tbl = pd.DataFrame(
            {"value": [len(struct.get("all_null_columns",[]) or []), len(struct.get("constant_columns",[]) or []),
                       len(struct.get("high_cardinality_object_columns",[]) or []),
                       len(struct.get("candidate_key_columns",[]) or [])]},
            index=["all_null_cols","constant_cols","high_cardinality_object_cols","candidate_key_cols"],
        )
        kstats = struct.get("candidate_key_collision_stats", {}) or {}
        kdf = pd.DataFrame()
        if isinstance(kstats, dict) and kstats:
            kdf  = pd.DataFrame(kstats).T
            keep = [c for c in ["nonnull","unique","collisions","collision_rate_over_nonnull","nonnull_rate"] if c in kdf.columns]
            kdf  = kdf[keep].copy()
            for c in ["collision_rate_over_nonnull","nonnull_rate"]:
                if c in kdf.columns:
                    kdf[c] = pd.to_numeric(kdf[c], errors="coerce").round(6)

        outlier_data  = rep.get("outliers", {}) or {}
        outlier_block = "(none detected)"
        if outlier_data:
            otbl = pd.DataFrame([
                {"column": c, "low": v["low"], "high": v["high"], "total": v["total"],
                 "frac_nonnull": round(v.get("fraction_of_nonnull", 0.0), 4)}
                for c, v in outlier_data.items()
            ]).set_index("column")
            outlier_block = format_table(otbl, max_rows=30, max_cols=6,
                                         col_width=20, preview_max_bytes=cfg.preview_max_bytes)

        parts: List[str] = [
            _section("Integrity Report"),
            "\nMeta:",
            format_table(meta_tbl, max_rows=10, max_cols=2, col_width=48, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nShape:",
            format_table(pd.DataFrame({"value": [shape.get("rows",0), shape.get("cols",0)]},
                                       index=["rows","cols"]), max_rows=5, max_cols=2, col_width=20,
                         preview_max_bytes=cfg.preview_max_bytes),
            "\n\nMissingness summary:",
            format_table(miss_tbl, max_rows=10, max_cols=2, col_width=26, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nMissing by column (top 25):",
            format_table(miss_top, max_rows=25, max_cols=3, col_width=36, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nDuplicates:",
            format_table(dup_tbl, max_rows=10, max_cols=2, col_width=28, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nStructure summary:",
            format_table(struct_tbl, max_rows=10, max_cols=2, col_width=34, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nCandidate key collision statistics:",
            format_table(kdf, max_rows=15, max_cols=8, col_width=24, preview_max_bytes=cfg.preview_max_bytes),
            "\n\nOutliers — Tukey fence:", outlier_block,
        ]
        for label, key, max_r, max_c, cw in [
            ("Empty-string cells (object columns):", "empty_strings", 25, 2, 36),
            ("Mixed-type columns:",                  "mixed_type_columns", 25, 3, 32),
        ]:
            data = rep.get(key, {}) or {}
            if isinstance(data, dict) and data:
                rows = [{"column": col, **counts} for col, counts in data.items()][:25]
                tbl  = pd.DataFrame(rows).set_index("column")
                parts += [f"\n\n{label}", format_table(tbl, max_rows=max_r, max_cols=max_c,
                                                        col_width=cw, preview_max_bytes=cfg.preview_max_bytes)]

        leakage = rep.get("leakage_suspects", []) or []
        if leakage:
            parts += ["\n\nLeakage suspects (heuristic):",
                      format_table(pd.DataFrame({"leakage_suspect": leakage[:25]}),
                                   max_rows=25, max_cols=1, col_width=52, preview_max_bytes=cfg.preview_max_bytes)]

        if rep.get("warnings"): parts += ["\n\nWarnings:"] + [f"  [!] {w}" for w in rep["warnings"]]
        if rep.get("notes"):    parts += ["\n\nNotes:"]    + [f"  [-] {n}" for n in rep["notes"]]
        return "\n".join(parts)
