"""
DNV Scientific Module
---------------------
Role:
    Provides observable contract validation, binding, and cleaning-operation
    guards for the DNV ingestion pipeline.

Scientific Context:
    Validates declared physical bounds, units, and missingness contracts
    against the loaded DataFrame; attaches validated contracts to a DataAsset
    and reports violations as structured records.

Invariants:
    - attach_contracts_to_asset returns a new DataAsset with the contracts
      field populated.
    - validate_contracts always returns a ValidationReport with is_valid=True
      if no contracts are declared.

Assumptions:
    - Contracts are declared as dicts with 'min', 'max', 'unit', and
      'missing_ok' keys.
    - DataAsset is a frozen dataclass importable from ing_provenance.

Failure Modes:
    - Contract column absent from DataFrame: violation is recorded, not
      raised.
    - Malformed contract dict (missing keys): absent keys are treated as
      unconstrained.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd

from ing_provenance import DataAsset, json_safe

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observable Contract — frozen dataclass (DNV-2.0 Slide 5)
# ---------------------------------------------------------------------------

#: Operations recognised by the admissible_operations enforcement gate.
KNOWN_OPERATIONS: Final[Tuple[str, ...]] = (
    "view", "summarise", "scale", "shift", "log", "log1p", "sqrt",
    "reciprocal", "square", "normalize", "standardize", "minmax",
    "robust", "rank", "winsorize", "soft_clamp", "clip", "drop_na",
    "impute", "add", "subtract", "multiply", "divide",
)

#: Fraction of non-null values below which a bounds violation is downgraded
#: from FAIL to WARN (i.e. ≤ 1 % of data outside bounds → warning only).
BOUNDS_WARN_FRAC: Final[float] = 0.01


@dataclass(frozen=True)
class ObservableContract:
    """Executable semantic contract for a single observable column.

    Turns a feature column into a declared scientific object with executable
    semantic constraints (DNV-2.0 Slide 5).

    Attributes
    ----------
    observable : str
        Column name this contract governs.
    nullable : bool
        Whether NaN / missing values are admissible.
    kind : str
        Dtype category: "numeric", "continuous", "ratio", "interval",
        "datetime", "boolean", "categorical", or "".
    lower_bound : float or None
        Physical lower bound (hard constraint).
    upper_bound : float or None
        Physical upper bound (hard constraint).
    units : str
        Physical units (e.g. "eV", "Å", "K").
    admissible_operations : tuple of str
        Transforms allowed on this column. Empty tuple = unrestricted.
    derivation : str
        How this column was derived (formula or description).
    reference_convention : str
        Reference convention (e.g. "Shannon", "Pauling", "Goldschmidt").
    role : str
        Column role: "identifier", "feature", "target", or "".
    protocol : str
        Measurement/computation protocol (e.g. "DFT-PBE", "experimental").
    proxy_group : str
        Proxy equivalence group (e.g. "ionic_radius", "electronegativity").
        Columns in the same proxy_group represent the same physical quantity
        under different proxies and are interchangeable for substitution testing.
    """
    observable:            str
    nullable:              bool = True
    kind:                  str = ""
    lower_bound:           Optional[float] = None
    upper_bound:           Optional[float] = None
    units:                 str = ""
    admissible_operations: Tuple[str, ...] = ()
    derivation:            str = ""
    reference_convention:  str = ""
    role:                  str = ""
    protocol:              str = ""
    proxy_group:           str = ""


def contract_from_dict(d: Dict[str, Any]) -> ObservableContract:
    """Convert a legacy contract dict to an ObservableContract.

    Backward-compatible: missing keys use defaults.
    """
    nullable_raw = str(d.get("nullable", "true")).strip().lower()
    nullable = nullable_raw not in ("false", "no", "0")

    admissible_raw = d.get("admissible_operations", ())
    if isinstance(admissible_raw, str):
        admissible = tuple(s.strip() for s in admissible_raw.split(",") if s.strip())
    elif isinstance(admissible_raw, (list, tuple)):
        admissible = tuple(str(s).strip() for s in admissible_raw)
    else:
        admissible = ()

    lo = d.get("lower_bound")
    hi = d.get("upper_bound")
    try:
        lo = float(lo) if lo not in (None, "") else None
    except (ValueError, TypeError):
        lo = None
    try:
        hi = float(hi) if hi not in (None, "") else None
    except (ValueError, TypeError):
        hi = None

    return ObservableContract(
        observable=str(d.get("observable", "")).strip(),
        nullable=nullable,
        kind=str(d.get("kind", "")).strip().lower(),
        lower_bound=lo,
        upper_bound=hi,
        units=str(d.get("units", "")).strip(),
        admissible_operations=admissible,
        derivation=str(d.get("derivation", "")).strip(),
        reference_convention=str(d.get("reference_convention", "")).strip(),
        role=str(d.get("role", "")).strip().lower(),
        protocol=str(d.get("protocol", "")).strip(),
        proxy_group=str(d.get("proxy_group", "")).strip(),
    )


def contract_to_dict(c: ObservableContract) -> Dict[str, Any]:
    """Convert an ObservableContract to a dict for serialisation."""
    d: Dict[str, Any] = {"observable": c.observable}
    if not c.nullable:
        d["nullable"] = "false"
    if c.kind:
        d["kind"] = c.kind
    if c.lower_bound is not None:
        d["lower_bound"] = c.lower_bound
    if c.upper_bound is not None:
        d["upper_bound"] = c.upper_bound
    if c.units:
        d["units"] = c.units
    if c.admissible_operations:
        d["admissible_operations"] = list(c.admissible_operations)
    if c.derivation:
        d["derivation"] = c.derivation
    if c.reference_convention:
        d["reference_convention"] = c.reference_convention
    if c.role:
        d["role"] = c.role
    if c.protocol:
        d["protocol"] = c.protocol
    if c.proxy_group:
        d["proxy_group"] = c.proxy_group
    return d


def check_unit_compatibility(
    col_a: str,
    col_b: str,
    contracts: List[Dict[str, Any]],
) -> Optional[str]:
    """Return a warning string if col_a and col_b have incompatible units.

    Returns None if units are compatible or undeclared.
    """
    lookup = {str(c.get("observable", "")).strip(): c for c in contracts}
    unit_a = str(lookup.get(col_a, {}).get("units", "")).strip()
    unit_b = str(lookup.get(col_b, {}).get("units", "")).strip()
    if unit_a and unit_b and unit_a != unit_b:
        return (
            f"Unit mismatch: '{col_a}' has units '{unit_a}' but "
            f"'{col_b}' has units '{unit_b}'. Combining columns with "
            f"incompatible units may produce physically meaningless results."
        )
    return None


def validate_contracts(df: pd.DataFrame,
                       contracts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate df against a list of observable contract dicts.

    Returns list of dicts with keys: observable, status, violations, warnings, checks_run.
    Status: PASS | FAIL | WARN | SKIP.
    violations and warnings are always separate keys (never merged).
    """
    results = []

    for contract in contracts:
        obs = str(contract.get("observable", "")).strip()
        if not obs:
            continue
        if obs not in df.columns:
            results.append({"observable": obs, "status": "SKIP",
                            "violations": [f"Column '{obs}' not found in dataset."],
                            "warnings": [], "checks_run": 0})
            continue

        col        = df[obs]
        violations: List[str] = []
        warnings:   List[str] = []
        checks = 0

        # Nullability
        nullable = str(contract.get("nullable", "true")).strip().lower() not in ("false", "no", "0")
        checks += 1
        null_count = int(col.isna().sum())
        if not nullable and null_count > 0:
            violations.append(f"Nullability violation: {null_count} null value(s) in non-nullable column.")
        if not nullable and pd.api.types.is_object_dtype(col):
            checks += 1
            empty = int((col.notna() & (col.astype(str).str.strip() == "")).sum())
            if empty > 0:
                violations.append(f"Empty-string cells in non-nullable column: {empty} cell(s).")

        # Kind / dtype
        kind = str(contract.get("kind", "")).strip().lower()
        checks += 1
        if kind in ("numeric", "continuous", "ratio", "interval"):
            if not pd.api.types.is_numeric_dtype(col):
                violations.append(f"Declared kind '{kind}' but dtype is '{col.dtype}'.")
        elif kind in ("datetime", "date", "timestamp"):
            if not pd.api.types.is_datetime64_any_dtype(col):
                violations.append(f"Declared kind '{kind}' but dtype is '{col.dtype}'.")
        elif kind in ("boolean", "bool"):
            if not pd.api.types.is_bool_dtype(col):
                violations.append(f"Declared kind '{kind}' but dtype is '{col.dtype}'.")

        # Mixed-type check for object columns
        if pd.api.types.is_object_dtype(col):
            checks += 1
            nn_str = col.dropna().astype(str)
            n_num  = int(pd.to_numeric(nn_str, errors="coerce").notna().sum())
            if 0 < n_num < len(nn_str):
                warnings.append(f"Mixed types: {n_num} numeric-parseable and {len(nn_str) - n_num} non-numeric non-null values.")

        # Numeric bounds + infinite values
        if pd.api.types.is_numeric_dtype(col):
            non_null  = col.dropna()
            n_nonnull = max(len(non_null), 1)
            for bound_key, op, label in [("lower_bound", "__lt__", "Lower-bound"), ("upper_bound", "__gt__", "Upper-bound")]:
                raw = str(contract.get(bound_key, "")).strip()
                if raw:
                    checks += 1
                    try:
                        bound = float(raw)
                        count = int(getattr(non_null, op)(bound).sum())
                        if count > 0:
                            frac = count / n_nonnull
                            msg  = f"{label} violation: {count} value(s) ({frac:.2%} of non-null)."
                            (warnings if frac <= BOUNDS_WARN_FRAC else violations).append(msg)
                    except ValueError:
                        violations.append(f"Cannot parse {bound_key} '{raw}' as a number.")
            checks += 1
            inf_count = int(np.isinf(non_null.to_numpy(dtype=float, copy=False)).sum())
            if inf_count > 0:
                violations.append(f"Infinite values: {inf_count} ±Inf cell(s) detected.")

        # Admissible operations — validate declared ops against known set
        admissible_raw = contract.get("admissible_operations", "")
        if admissible_raw:
            checks += 1
            if isinstance(admissible_raw, str):
                declared_ops = [s.strip() for s in admissible_raw.split(",") if s.strip()]
            elif isinstance(admissible_raw, (list, tuple)):
                declared_ops = [str(s).strip() for s in admissible_raw if str(s).strip()]
            else:
                declared_ops = []
            unknown = [op for op in declared_ops if op not in KNOWN_OPERATIONS]
            if unknown:
                warnings.append(
                    f"Unknown admissible operation(s): {', '.join(unknown)}. "
                    f"Known: {', '.join(KNOWN_OPERATIONS[:8])}…"
                )

        status = "FAIL" if violations else ("WARN" if warnings else "PASS")
        results.append({"observable": obs, "status": status,
                        "violations": violations, "warnings": warnings, "checks_run": checks})
    return results


def attach_contracts_to_asset(asset: DataAsset,
                               contracts: List[Dict[str, Any]]) -> DataAsset:
    """Return asset with observable_contracts bound into metadata."""
    meta     = dict(asset.metadata)
    rep      = dict(asset.integrity_report)
    rep_meta = dict(rep.get("meta", {}))
    clean    = [json_safe(dict(c)) for c in (contracts or [])]
    meta["observable_contracts"]     = clean
    rep_meta["observable_contracts"] = clean
    rep["meta"] = rep_meta
    return DataAsset(dataframe=asset.dataframe, metadata=meta,
                     statistics=dict(asset.statistics),
                     integrity_report=rep, source_path=asset.source_path)


def assess_cleaning_operation(df: pd.DataFrame,
                               operation: str,
                               params:    Optional[Dict[str, Any]] = None,
                               contracts: Optional[List[Dict[str, Any]]] = None
                               ) -> Dict[str, Any]:
    """Check whether a cleaning operation is admissible for the current asset."""
    params    = dict(params or {})
    warnings: List[str] = []
    violations: List[str] = []

    lookup: Dict[str, Dict[str, Any]] = {}
    for c in (contracts or []):
        obs = str(c.get("observable", "")).strip()
        if obs:
            lookup[obs] = c

    def kind_of(col: str) -> str:
        k = str(lookup.get(col, {}).get("kind", "")).strip().lower()
        if k: return k
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):      return "numeric"
        if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
        if pd.api.types.is_bool_dtype(s):          return "boolean"
        return "categorical"

    if operation == "handle_missing_values":
        method = str(params.get("method", "")).strip().lower()
        interp = {"linear","polynomial","spline","akima","pchip","barycentric","krogh"}
        stat   = {"mean","median"}
        if method in interp | stat:
            non_num = [c for c in df.columns if kind_of(c) not in ("numeric","continuous","ratio","interval")]
            if non_num:
                warnings.append("Non-numeric observables excluded from numeric imputation: " + ", ".join(non_num[:8]))
        if method in interp:
            bad = [c for c in df.columns if kind_of(c) in ("categorical","boolean","bool")]
            if bad:
                warnings.append("Interpolation inadmissible for categorical/boolean observables; only numeric subset revised.")

    elif operation == "create_derived_column":
        if not str(params.get("expression", "")).strip():
            violations.append("Derived-column operation requires a non-empty expression.")
        # Check unit compatibility for binary operations
        expr_cols = params.get("source_columns", []) or []
        if len(expr_cols) >= 2:
            for i in range(len(expr_cols)):
                for j in range(i + 1, len(expr_cols)):
                    msg = check_unit_compatibility(expr_cols[i], expr_cols[j], contracts or [])
                    if msg:
                        warnings.append(msg)

    elif operation == "manage_columns":
        final = params.get("columns_final", []) or []
        id_cols = [c for c in df.columns if str(lookup.get(c, {}).get("role", "")).strip().lower() == "identifier"]
        dropped = [c for c in id_cols if c not in final]
        if dropped:
            warnings.append("Identifier observables removed: " + ", ".join(dropped[:8]))

    elif operation == "transform_columns":
        transform_type = str(params.get("transform_type", "")).strip().lower()
        affected_cols = params.get("columns", []) or []
        for col in affected_cols:
            contract = lookup.get(col)
            if contract:
                admissible = contract.get("admissible_operations")
                if admissible and transform_type:
                    if isinstance(admissible, str):
                        allowed = [s.strip() for s in admissible.split(",")]
                    else:
                        allowed = [str(s).strip() for s in admissible]
                    if transform_type not in allowed:
                        violations.append(
                            f"Transform '{transform_type}' not in admissible "
                            f"operations for '{col}': {allowed}"
                        )
                # Unit-aware: warn if combining with incompatible units
                units = str(contract.get("units", "")).strip()
                if units and transform_type in ("add", "subtract", "multiply", "divide"):
                    warnings.append(
                        f"Column '{col}' has declared units '{units}'; "
                        f"verify that '{transform_type}' preserves dimensional consistency."
                    )

    elif operation == "normalize_scale":
        method = str(params.get("method", "")).strip().lower()
        affected_cols = params.get("columns", []) or []
        # Map normalization methods to their admissible_operations equivalents
        _NORM_OP_MAP = {
            "standardize": "standardize", "minmax": "minmax", "robust": "robust",
            "log": "log", "sqrt": "sqrt",
        }
        op_name = _NORM_OP_MAP.get(method, method)
        for col in affected_cols:
            contract = lookup.get(col)
            if contract:
                admissible = contract.get("admissible_operations")
                if admissible and op_name:
                    if isinstance(admissible, str):
                        allowed = [s.strip() for s in admissible.split(",")]
                    else:
                        allowed = [str(s).strip() for s in admissible]
                    if op_name not in allowed:
                        violations.append(
                            f"Normalization '{method}' (maps to '{op_name}') not in "
                            f"admissible operations for '{col}': {allowed}"
                        )

    elif operation == "remove_outliers":
        method = str(params.get("method", "iqr")).strip().lower()
        affected_cols = params.get("columns", []) or []
        for col in affected_cols:
            contract = lookup.get(col)
            if contract:
                admissible = contract.get("admissible_operations")
                if admissible:
                    if isinstance(admissible, str):
                        allowed = [s.strip() for s in admissible.split(",")]
                    else:
                        allowed = [str(s).strip() for s in admissible]
                    # Outlier removal is admissible only if "clip" or "winsorize"
                    # is in allowed operations (row deletion is always allowed)
                    if method in ("iqr", "zscore") and "clip" not in allowed and "winsorize" not in allowed:
                        warnings.append(
                            f"Outlier removal ({method}) on '{col}' may alter "
                            f"distribution semantics; 'clip'/'winsorize' not in "
                            f"admissible operations: {allowed}"
                        )

    return {"status": "FAIL" if violations else ("WARN" if warnings else "PASS"),
            "violations": violations, "warnings": warnings}
