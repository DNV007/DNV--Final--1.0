"""
DNV Scientific Module
---------------------
Role:
    Provides IngestionConfig — the immutable loader-and-audit configuration
    record for the DNV ingestion pipeline.

Scientific Context:
    Declares all tunable thresholds for file loading, schema inference,
    statistics, and integrity checks as a frozen dataclass with physically
    meaningful defaults for heterogeneous materials datasets.

Invariants:
    - IngestionConfig is a frozen dataclass; all fields are read-only after
      construction.
    - All numeric thresholds are positive and within their declared physical
      domains.

Assumptions:
    - Default values represent conservative, production-safe settings.
    - Fields are set once at application start and not mutated during a
      session.

Failure Modes:
    - Field set to a non-positive value: downstream functions may raise
      ZeroDivisionError or produce degenerate outputs.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class IngestionConfig:
    # Hygiene
    normalize_columns: bool = True
    strip_whitespace:  bool = True
    # Optional coercions (off by default)
    coerce_numeric:   bool  = False
    parse_dates:      bool  = False
    coerce_threshold: float = 0.80
    # I/O
    delimiter_sniff: bool                      = True
    encoding:        Optional[str]             = None
    na_tokens:       Optional[Tuple[str, ...]] = None
    # Memory / display safety
    sample_rows_for_inference:    int   = 5_000
    max_preview_rows:             int   = 50
    max_preview_cols:             int   = 80
    max_report_missing_cols:      int   = 50
    max_report_constant_cols:     int   = 50
    max_report_hicard_cols:       int   = 50
    high_cardinality_frac:        float = 0.90
    max_numeric_describe_cols:    int   = 80
    preview_max_bytes:            int   = 2_000_000
    max_file_bytes:               int   = 500 * 1024 * 1024
    # Integrity analysis
    max_categories_per_column: int   = 20
    outlier_iqr_factor:        float = 1.5

    def __post_init__(self) -> None:
        if not (0.0 < self.coerce_threshold <= 1.0):
            raise ValueError(f"coerce_threshold must be in (0.0, 1.0]; got {self.coerce_threshold!r}")
        if self.outlier_iqr_factor <= 0.0:
            raise ValueError(f"outlier_iqr_factor must be > 0; got {self.outlier_iqr_factor!r}")
