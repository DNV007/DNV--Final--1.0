"""
DNV Scientific Module
---------------------
Role:
    Provides build_data_asset and derive_data_asset — the top-level
    orchestration functions for the DNV ingestion pipeline.

Scientific Context:
    Coordinates file loading (ing_loader), schema inference (ing_analysis),
    contract binding (ing_contracts), provenance recording (ing_provenance),
    and statistics assembly into a single immutable DataAsset.

Invariants:
    - build_data_asset always returns a DataAsset or raises; it never
      returns None.
    - derive_data_asset preserves the parent DataAsset's hash as
      parent_hash in the new ProvenanceRecord.

Assumptions:
    - The file path is readable and in a supported format.
    - IngestionConfig is valid (all fields within declared domains).

Failure Modes:
    - File read error: raises IOError propagated from ing_loader.
    - Schema inference error: raises ValueError propagated from ing_analysis.
    - Missing-value helpers silently coerce NaN-only columns to object dtype.

Provenance:
    - Emits one ProvenanceRecord per build_data_asset call; parent_hash is
      None for root assets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ing_config     import IngestionConfig
from ing_provenance import (DataAsset, ProvenanceRecord, SCHEMA_VERSION,
                             dataframe_sha256, json_safe, utc_now_iso)
from ing_loader     import load_dataset
from ing_analysis   import basic_statistics, integrity_report

log = logging.getLogger(__name__)


def build_data_asset(source: Union[str, Path],
                     cfg: Optional[IngestionConfig] = None) -> DataAsset:
    """Run the full ingestion pipeline and return a provenance-bound DataAsset."""
    cfg    = cfg or IngestionConfig()
    df     = load_dataset(source, cfg=cfg)
    stats  = basic_statistics(df, cfg=cfg)
    report = integrity_report(df, path=source, cfg=cfg)

    record = ProvenanceRecord(
        operation="load_dataset", applied_utc=utc_now_iso(),
        rows=int(df.shape[0]), cols=int(df.shape[1]),
        params={"source": str(source)},
    )
    report["meta"]["pipeline_history"] = [record.to_dict()]
    metadata = dict(report["meta"])
    metadata["dataframe_sha256"] = dataframe_sha256(df)
    metadata["schema_version"]   = dict(SCHEMA_VERSION)
    report["meta"] = metadata

    return DataAsset(dataframe=df, metadata=metadata, statistics=stats,
                     integrity_report=report, source_path=str(source))


def derive_data_asset(dataframe:        pd.DataFrame,
                      source_path:      Optional[Union[str, Path]] = None,
                      cfg:              Optional[IngestionConfig]   = None,
                      prior_asset:      Optional[DataAsset]         = None,
                      operation:        str                         = "manual_edit",
                      operation_params: Optional[Dict[str, Any]]    = None) -> DataAsset:
    """Return a new DataAsset derived from an edited dataframe, with full provenance."""
    cfg    = cfg or IngestionConfig()
    df     = dataframe.copy()
    stats  = basic_statistics(df, cfg=cfg)
    report = integrity_report(df, path=source_path, cfg=cfg)

    history:     List[Dict[str, Any]] = []
    parent_id:   Optional[str]        = None
    parent_hash: Optional[str]        = None

    if prior_asset is not None:
        prior_meta = dict(prior_asset.metadata)
        history    = [ProvenanceRecord.from_dict(h).to_dict()
                      for h in (prior_meta.get("pipeline_history") or [])]
        parent_id   = prior_meta.get("analysis_id")
        parent_hash = prior_meta.get("dataframe_sha256")

    record = ProvenanceRecord(
        operation=str(operation), applied_utc=utc_now_iso(),
        rows=int(df.shape[0]), cols=int(df.shape[1]),
        params=json_safe(operation_params or {}),
        parent_analysis_id=parent_id,
        parent_dataframe_sha256=parent_hash,
    )
    history.append(record.to_dict())
    report["meta"]["pipeline_history"] = history

    resolved = (str(source_path) if source_path is not None
                else (prior_asset.source_path if prior_asset is not None else None))

    metadata = dict(report["meta"])
    metadata["dataframe_sha256"] = dataframe_sha256(df)
    metadata["schema_version"]   = dict(SCHEMA_VERSION)
    report["meta"] = metadata

    return DataAsset(dataframe=df, metadata=metadata, statistics=stats,
                     integrity_report=report, source_path=resolved)


def rows_with_any_missing(df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    return df.loc[df.isna().any(axis=1)].head(max_rows).copy()


def columns_with_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss.to_frame(name="missing_count")
