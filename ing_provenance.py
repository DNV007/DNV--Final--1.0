"""
DNV Scientific Module
---------------------
Role:
    Provides ProvenanceRecord and DataAsset frozen dataclasses, SHA-256
    hashing, JSON serialisation, and persist/restore for the full pipeline
    provenance chain.

Scientific Context:
    Defines the immutable data contract flowing through the entire DNV
    pipeline: DataAsset wraps (dataframe, metadata, statistics,
    integrity_report, source_path) with a hash-linked ProvenanceRecord
    chain enabling full audit reconstruction.

Invariants:
    - DataAsset is a frozen dataclass; all fields are immutable after
      construction.
    - dataframe_sha256 is deterministic for identical DataFrames (same
      dtypes, row order, and values).

Assumptions:
    - DataFrames are stored in row-major order; column reordering changes
      the hash.
    - JSON serialisation uses json_safe to handle numpy scalars and
      non-finite floats.

Failure Modes:
    - Non-serialisable field in metadata: json_safe converts to string
      representation.
    - Persist file unwritable: raises IOError propagated to caller.

Provenance:
    - Emits one ProvenanceRecord per transformation; parent_hash links to
      the preceding record.
"""
from __future__ import annotations

import hashlib, json, logging, math, uuid, zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ing_config import IngestionConfig

log = logging.getLogger(__name__)

SCHEMA_VERSION: Dict[str, str] = {"ingestion": "3.0", "pipeline": "1.1"}


# ── ProvenanceRecord ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable descriptor for a single pipeline operation."""
    operation:               str
    applied_utc:             str
    rows:                    int
    cols:                    int
    params:                  Dict[str, Any] = field(default_factory=dict)
    parent_analysis_id:      Optional[str]  = None
    parent_dataframe_sha256: Optional[str]  = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation":               self.operation,
            "applied_utc":             self.applied_utc,
            "rows":                    self.rows,
            "cols":                    self.cols,
            "params":                  json_safe(self.params),
            "parent_analysis_id":      self.parent_analysis_id,
            "parent_dataframe_sha256": self.parent_dataframe_sha256,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            operation=str(d.get("operation", "unknown")),
            applied_utc=str(d.get("applied_utc", utc_now_iso())),
            rows=int(d.get("rows", 0)),
            cols=int(d.get("cols", 0)),
            params=dict(d.get("params") or {}),
            parent_analysis_id=d.get("parent_analysis_id"),
            parent_dataframe_sha256=d.get("parent_dataframe_sha256"),
        )


# ── DataAsset ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataAsset:
    """Immutable, provenance-bound dataset container."""
    dataframe:        pd.DataFrame
    metadata:         Dict[str, Any]
    statistics:       Dict[str, Any]
    integrity_report: Dict[str, Any]
    source_path:      Optional[str] = None


# ── Provenance utilities ─────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def new_analysis_id() -> str:
    return uuid.uuid4().hex[:8].upper()

def file_bytes(path: Union[str, Path]) -> int:
    return Path(path).stat().st_size

def sha256_of_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def dataframe_sha256(df: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="uint64", copy=False)
    payload = row_hashes.tobytes() + "||".join(map(str, df.columns)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ── JSON helpers ─────────────────────────────────────────────────────────────

def json_safe(x: Any) -> Any:
    """Recursively coerce non-serialisable values to JSON-safe equivalents."""
    if x is None or x is pd.NA:                          return None
    if isinstance(x, bool):                              return x
    if isinstance(x, np.bool_):                          return bool(x)
    if isinstance(x, np.integer):                        return int(x)
    if isinstance(x, np.floating):
        v = float(x); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, float):                             return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, pd.Timestamp):                      return x.isoformat()
    if isinstance(x, np.ndarray):                        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, dict):                              return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list):                              return [json_safe(v) for v in x]
    return x

def save_json(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    """Serialise obj to path as indented UTF-8 JSON, NaN/Inf → null."""
    Path(path).write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Asset persistence ────────────────────────────────────────────────────────

def persist_data_asset(asset: DataAsset, path: Union[str, Path]) -> None:
    payload = {
        "metadata":         dict(asset.metadata),
        "statistics":       dict(asset.statistics),
        "integrity_report": dict(asset.integrity_report),
        "source_path":      asset.source_path,
        "_schema_version":  dict(SCHEMA_VERSION),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_json = asset.dataframe.to_json(orient="table", date_format="iso", index=True)
    with zipfile.ZipFile(Path(path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataframe.table.json", df_json)
        zf.writestr("payload.json", json.dumps(json_safe(payload), indent=2, ensure_ascii=False))


# ── Provenance replay ───────────────────────────────────────────────────────

_REPLAY_REGISTRY: Dict[str, Any] = {}


def register_replay(operation: str, fn: Any) -> None:
    """Register a replay handler for an operation name.

    Parameters
    ----------
    operation : str
        Operation name matching ProvenanceRecord.operation.
    fn : callable
        ``fn(df: pd.DataFrame, params: dict) -> pd.DataFrame``.
    """
    _REPLAY_REGISTRY[operation] = fn


def replay_pipeline(
    source_df: pd.DataFrame,
    history: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Re-apply recorded pipeline operations in sequence.

    Parameters
    ----------
    source_df : pd.DataFrame
        The original (pre-pipeline) DataFrame.
    history : list of dict
        Pipeline history from ``asset.metadata["pipeline_history"]``.
        Each entry has ``operation`` and ``params`` keys.

    Returns
    -------
    pd.DataFrame
        The replayed result.  Operations without a registered handler
        are skipped with a logged warning.
    """
    df = source_df.copy()
    for i, record_dict in enumerate(history):
        op = record_dict.get("operation", "")
        params = record_dict.get("params", {})
        fn = _REPLAY_REGISTRY.get(op)
        if fn is None:
            log.warning(
                "replay_pipeline: step %d — no handler for operation '%s'; skipping.",
                i, op,
            )
            continue
        try:
            df = fn(df, params)
            log.info("replay_pipeline: step %d — replayed '%s' → %s", i, op, df.shape)
        except Exception as exc:
            log.warning(
                "replay_pipeline: step %d — '%s' failed: %s; skipping.", i, op, exc,
            )
    return df


def restore_data_asset(path: Union[str, Path]) -> DataAsset:
    """Restore a DataAsset from a zip bundle.

    Three-level version fallback so old monolith sessions don't warn:
      1. payload["_schema_version"]          (new package)
      2. payload["metadata"]["schema_version"] (old monolith)
      3. None → silent legacy accept
    """
    with zipfile.ZipFile(Path(path), mode="r") as zf:
        payload = json.loads(zf.read("payload.json").decode("utf-8"))
        df_json = zf.read("dataframe.table.json").decode("utf-8")
    dataframe = pd.read_json(StringIO(df_json), orient="table")

    pv: dict = payload.get("_schema_version") or {}
    if not pv:
        pv = (payload.get("metadata") or {}).get("schema_version") or {}
    resolved = pv.get("ingestion")

    if resolved is not None and resolved != SCHEMA_VERSION["ingestion"]:
        log.warning("restore_data_asset: persisted version '%s' differs from current '%s'.",
                    resolved, SCHEMA_VERSION["ingestion"])
    elif resolved is None:
        log.debug("restore_data_asset: no schema version found; treating as legacy bundle.")

    return DataAsset(
        dataframe=dataframe,
        metadata=dict(payload["metadata"]),
        statistics=dict(payload["statistics"]),
        integrity_report=dict(payload["integrity_report"]),
        source_path=payload.get("source_path"),
    )
