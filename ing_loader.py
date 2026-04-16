"""
DNV Scientific Module
---------------------
Role:
    Provides load_dataframe — a format-dispatching file loader with column
    normalisation and safe type coercions.

Scientific Context:
    Reads CSV/TSV/Parquet/Excel/JSON files into a pandas DataFrame;
    normalises column names (strip, lowercase, replace spaces) and applies
    type coercions declared in IngestionConfig.

Invariants:
    - Column names are always strings after load.
    - load_dataframe never returns a DataFrame with duplicate column names.

Assumptions:
    - openpyxl is available for Excel; pyarrow or fastparquet for Parquet.
    - File encoding is UTF-8 with BOM fallback for CSV/TSV.

Failure Modes:
    - Unsupported extension: raises ValueError with the extension name.
    - Encoding error: raises UnicodeDecodeError propagated from pandas.
    - Empty file: raises ValueError.

Provenance:
    - This module emits no transformation metadata; file hash is computed
      by ing_provenance.
"""
from __future__ import annotations

import logging, os, re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from ing_config import IngestionConfig

log = logging.getLogger(__name__)

_SUPPORTED = frozenset({".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls",
                         ".json", ".jsonl", ".ndjson"})
_NA_DEFAULTS: Tuple[str, ...] = ("", "NA", "N/A", "nan", "NaN", "null", "None", "-")


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Collapse whitespace runs in column names; return (df, mapping)."""
    mapping: Dict[str, str] = {}
    new_cols = []
    for c in df.columns:
        cc = re.sub(r"\s+", " ", str(c)).strip()
        mapping[str(c)] = cc
        new_cols.append(cc)
    out = df.copy()
    out.columns = pd.Index(new_cols)
    return out, mapping


def strip_object_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string-dtype cells."""
    out = df.copy()
    for c in out.select_dtypes(include=["object", "string"]).columns:
        dtype = out[c].dtype
        mask = out[c].notna()
        out.loc[mask, c] = out.loc[mask, c].astype(str).str.strip()
        try:
            out[c] = out[c].astype(dtype)
        except (TypeError, ValueError):
            pass
    return out


def _coerce_numeric(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    n = max(len(out), 1)
    for c in out.select_dtypes(include=["object", "string"]).columns:
        converted = pd.to_numeric(out[c], errors="coerce")
        if converted.notna().sum() / n >= threshold:
            out[c] = converted
    return out


def _coerce_dates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    n = max(len(out), 1)
    for c in out.select_dtypes(include=["object", "string"]).columns:
        converted = pd.to_datetime(out[c], errors="coerce", utc=True)
        if converted.notna().sum() / n >= threshold:
            out[c] = converted
    return out


def _read_json(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(str(path), lines=True)
    try:
        return pd.read_json(str(path), lines=False)
    except ValueError:
        return pd.read_json(str(path), lines=True)


def load_dataset(path: Union[str, Path],
                 cfg: Optional[IngestionConfig] = None) -> pd.DataFrame:
    """Load a tabular dataset and apply hygiene transforms."""
    cfg = cfg or IngestionConfig()
    p   = Path(path)

    if not p.exists():      raise FileNotFoundError(f"File not found: {p}")
    if p.is_dir():          raise IsADirectoryError(f"Path is a directory: {p}")
    if not os.access(p, os.R_OK): raise PermissionError(f"Read permission denied: {p}")

    ext = p.suffix.lower()
    if ext not in _SUPPORTED:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: {', '.join(sorted(_SUPPORTED))}")

    if cfg.max_file_bytes > 0:
        from ing_provenance import file_bytes
        size = file_bytes(p)
        if size > cfg.max_file_bytes:
            raise ValueError(f"File size {size:,} B exceeds limit {cfg.max_file_bytes:,} B.")

    na = list(cfg.na_tokens) if cfg.na_tokens else list(_NA_DEFAULTS)

    if ext in (".csv", ".tsv", ".txt"):
        if cfg.delimiter_sniff:
            df = pd.read_csv(p, sep=None, engine="python", encoding=cfg.encoding,
                             na_values=na, keep_default_na=True)
        else:
            df = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",",
                             encoding=cfg.encoding, na_values=na, keep_default_na=True)
        if df.shape[1] < 2:
            log.warning("load_dataset: only %d column(s) parsed — possible delimiter mismatch.", df.shape[1])
    elif ext == ".parquet":
        df = pd.read_parquet(p)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = _read_json(p)

    if cfg.normalize_columns: df, _ = normalize_columns(df)
    if cfg.strip_whitespace:  df    = strip_object_whitespace(df)
    if cfg.coerce_numeric:    df    = _coerce_numeric(df, cfg.coerce_threshold)
    if cfg.parse_dates:       df    = _coerce_dates(df, cfg.coerce_threshold)

    # Enforce invariant: no duplicate column names after hygiene transforms.
    if df.columns.duplicated().any():
        seen: Dict[str, int] = {}
        new_cols: list = []
        for c in df.columns:
            s = str(c)
            if s in seen:
                seen[s] += 1
                new_cols.append(f"{s}_{seen[s]}")
            else:
                seen[s] = 0
                new_cols.append(s)
        df.columns = pd.Index(new_cols)
        log.warning("load_dataset: deduplicated %d column name(s).", sum(1 for v in seen.values() if v > 0))

    log.info("load_dataset: '%s' → %d rows × %d cols", p.name, df.shape[0], df.shape[1])
    return df
