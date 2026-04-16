"""
DNV Scientific Module
---------------------
Role:
    Canonical spinel dataset loader for the DNV-2.0 NPJ paper case study.
    Assembles a contract-bound DataAsset of 249 AB2X4 spinel compositions
    with DFT-computed migration barriers and hull distances.

Scientific Context:
    The paper's case study relies on a consistent, contract-declared
    representation of 249 spinels (carriers: Li, Na, Mg, Ca, Zn, Al;
    anions: O, S, Se).  Two source files coexist on disk: the raw feature
    table (261 rows, clean Shannon radii, canonical chemistry columns) and
    the post-outlier-removal table (249 rows, contains E_hull and alpha
    but has a decimal-shift corruption affecting R_B and EA_B in 82 rows).
    This loader merges the two on compound name, taking clean chemistry
    from raw and taking only the non-corrupted columns (E_hull, alpha)
    from the outlier-removal table.

Invariants:
    - The returned DataFrame has exactly SPINEL_N_ROWS_EXPECTED rows.
    - All declared ObservableContracts in SPINEL_CONTRACTS are satisfied
      (or explicitly reported as violations by the caller).
    - Column names match the canonical set declared in SPINEL_CANONICAL_COLS.

Assumptions:
    - The two source CSVs live in the project root next to this module.
    - The raw file's radius columns are in angstroms and have not been
      mutated since upload.

Failure Modes:
    - Missing source file: loader raises FileNotFoundError with the
      missing path.
    - Compound-name mismatch: only compounds present in BOTH files survive
      the inner join; the dropped list is returned in the DataAsset
      metadata so callers can audit the attrition.

Provenance:
    - load_canonical_spinel_asset emits a ProvenanceRecord naming both
      source files, the merge strategy, and the resulting row count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Tuple

import numpy as np
import pandas as pd

from ing_contracts  import ObservableContract
from ing_provenance import DataAsset, ProvenanceRecord, utc_now_iso

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File locations and expected shape
# ---------------------------------------------------------------------------

#: Directory this module lives in (project root for DNV flat layout).
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent

#: Source file with raw (clean-radius) spinel features.  261 rows × 29 cols.
SPINEL_RAW_FILE: Final[Path] = _MODULE_DIR / "spinel_features_fixed_ox3_radii.csv"

#: Source file with post-outlier-removal table.  249 rows × 28 cols.
#: Semicolon-separated; loader auto-detects the delimiter.
#: NOTE: R_B and EA_B columns in this file are corrupted for 82 rows
#: (decimal shift moved Å values into pm×10).  Only E_hull and alpha
#: columns are read from this file — they are not part of the corruption.
SPINEL_CLEANED_FILE: Final[Path] = _MODULE_DIR / "spinel_features_fixed_ox3_radii_cleaned_outlierrm.csv"

#: Expected number of rows in the canonical merged dataset.
SPINEL_N_ROWS_EXPECTED: Final[int] = 249


# ---------------------------------------------------------------------------
# Column harmonisation map: raw column name → canonical name
# ---------------------------------------------------------------------------

#: Rename map applied to raw-file columns so they match the paper's canonical
#: set (as used in gui_dlg_desc_plot SHAP plot and DNV-2.0 slide 11).
SPINEL_RAW_RENAME: Final[dict] = {
    "a_lat":                "a",
    "b_lat":                "b",
    "c_lat":                "c",
    "avg_d_AX_excl":        "d_AX",
    "avg_d_BX":             "d_BX",
    "k64_m_excl":           "k_64m",
    "u_eff_excl":           "U",
    "tetragonality_ratio":  "t_ratio",
}

#: Columns taken from the cleaned file (only these two are uncorrupted).
SPINEL_CLEANED_EXTRA_COLS: Final[Tuple[str, ...]] = ("E_hull", "alpha")

#: Intermediate "_all" variants dropped after harmonisation (we keep
#: the `_excl` variants as the paper's default).
SPINEL_DROPPED_VARIANT_COLS: Final[Tuple[str, ...]] = (
    "avg_d_AX_all", "k64_m_all", "u_eff_all",
)

#: Canonical ordered column set of the merged dataset.
SPINEL_CANONICAL_COLS: Final[Tuple[str, ...]] = (
    "compound", "E_a", "E_hull",
    "A_site", "B_site", "X_site", "A_valence",
    "EN_A", "EN_B", "EN_X",
    "R_A",  "R_B",  "R_X",
    "Polarizability_A", "Polarizability_B", "Polarizability_X",
    "EA_A", "EA_B", "EA_X",
    "a", "b", "c", "alpha",
    "d_AX", "d_BX", "k_64m", "U", "t_ratio",
)


# ---------------------------------------------------------------------------
# Physical bounds — Shannon ionic radii, DFT barrier ranges, lattice
# ---------------------------------------------------------------------------

#: Shannon ionic radius lower bound applicable to all cations [Å].
SHANNON_RADIUS_MIN_A: Final[float] = 0.3
#: Shannon ionic radius upper bound applicable to all cations [Å].
SHANNON_RADIUS_MAX_A: Final[float] = 1.5
#: Anion radius lower bound (O, S, Se) [Å].
ANION_RADIUS_MIN_A:   Final[float] = 1.2
#: Anion radius upper bound [Å].
ANION_RADIUS_MAX_A:   Final[float] = 2.2

#: Migration barrier physical range for spinels [eV].
E_A_MIN_EV: Final[float] = 0.0
E_A_MAX_EV: Final[float] = 3.0

#: Convex hull distance physical range (stable → weakly metastable) [eV/atom].
E_HULL_MIN_EV: Final[float] = -0.5
E_HULL_MAX_EV: Final[float] = 1.0

#: Pauling electronegativity physical range [dimensionless].
EN_MIN: Final[float] = 0.5
EN_MAX: Final[float] = 4.0

#: Lattice constant physical range for AB2X4 spinels [Å].
#: Upper bound accommodates selenide spinels with large A+X radii
#: (observed max ≈12.57 Å in the canonical 249-spinel set).
LATTICE_MIN_A: Final[float] = 7.0
LATTICE_MAX_A: Final[float] = 13.0

#: Spinel lattice angle bounds (cubic ≈ 90°; Jahn-Teller distortion allowed) [°].
ALPHA_MIN_DEG: Final[float] = 85.0
ALPHA_MAX_DEG: Final[float] = 110.0

#: Bond-length physical range [Å].
BOND_MIN_A: Final[float] = 1.5
BOND_MAX_A: Final[float] = 3.5


# ---------------------------------------------------------------------------
# Declared observable contracts (DNV-2.0 Slide 5)
# ---------------------------------------------------------------------------

def _mk_contract(
    name: str, unit: str, lo: float, hi: float, *, nullable: bool = False,
    kind: str = "continuous",
) -> ObservableContract:
    """Factory producing a single ObservableContract with the standard
    DNV field set — keeps SPINEL_CONTRACTS declaration compact."""
    return ObservableContract(
        observable=name, units=unit, kind=kind,
        lower_bound=lo, upper_bound=hi, nullable=nullable,
        admissible_operations=("view", "summarise", "scale", "standardize",
                               "robust", "winsorize", "soft_clamp", "rank"),
    )


#: Executable observable contracts for every numeric column of the canonical
#: spinel dataset.  Callers run validate_contracts() on the merged DataAsset
#: and report any violations as a SEAL contract-feasibility failure.
SPINEL_CONTRACTS: Final[Tuple[ObservableContract, ...]] = (
    _mk_contract("E_a",    "eV",       E_A_MIN_EV,          E_A_MAX_EV),
    _mk_contract("E_hull", "eV/atom",  E_HULL_MIN_EV,       E_HULL_MAX_EV),
    _mk_contract("EN_A",   "",         EN_MIN,              EN_MAX),
    _mk_contract("EN_B",   "",         EN_MIN,              EN_MAX),
    _mk_contract("EN_X",   "",         EN_MIN,              EN_MAX),
    _mk_contract("R_A",    "angstrom", SHANNON_RADIUS_MIN_A, SHANNON_RADIUS_MAX_A),
    _mk_contract("R_B",    "angstrom", SHANNON_RADIUS_MIN_A, SHANNON_RADIUS_MAX_A),
    _mk_contract("R_X",    "angstrom", ANION_RADIUS_MIN_A,   ANION_RADIUS_MAX_A),
    _mk_contract("a",      "angstrom", LATTICE_MIN_A,        LATTICE_MAX_A),
    _mk_contract("b",      "angstrom", LATTICE_MIN_A,        LATTICE_MAX_A),
    _mk_contract("c",      "angstrom", LATTICE_MIN_A,        LATTICE_MAX_A),
    _mk_contract("alpha",  "degree",   ALPHA_MIN_DEG,        ALPHA_MAX_DEG),
    _mk_contract("d_AX",   "angstrom", BOND_MIN_A,           BOND_MAX_A),
    _mk_contract("d_BX",   "angstrom", BOND_MIN_A,           BOND_MAX_A),
    _mk_contract("k_64m",  "",         0.5,                  2.0),
    _mk_contract("U",      "",         0.2,                  0.5),
    _mk_contract("t_ratio","",         0.75,                 1.20),
    _mk_contract("EA_A", "eV", -1.0, 4.0, nullable=True),
    _mk_contract("EA_B", "eV", -1.0, 4.0, nullable=True),
    _mk_contract("EA_X", "eV", -1.0, 4.0, nullable=True),
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpinelLoadReport:
    """
    Diagnostic record emitted by load_canonical_spinel_asset.

    Fields
    ------
    n_raw            : rows in the raw source file before merge
    n_cleaned_source : rows in the cleaned source file before merge
    n_merged         : rows surviving the inner join on compound name
    dropped_from_raw : compounds present in raw but missing from cleaned
    radii_corruption_rows_in_cleaned : count of R_B values in the cleaned
        file that exceed the declared Shannon upper bound (demonstrates
        the pre-SEAL data-quality failure mode)
    """
    n_raw:            int
    n_cleaned_source: int
    n_merged:         int
    dropped_from_raw: Tuple[str, ...]
    radii_corruption_rows_in_cleaned: int


def _read_cleaned_source(path: Path) -> pd.DataFrame:
    """Auto-detect comma vs. semicolon delimiter."""
    head = path.read_text().splitlines()[0]
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, sep=sep)


def load_canonical_spinel_asset(
    raw_path:     Path = SPINEL_RAW_FILE,
    cleaned_path: Path = SPINEL_CLEANED_FILE,
) -> Tuple[DataAsset, SpinelLoadReport]:
    """
    Assemble the canonical 249-row spinel dataset used in the paper.

    Returns
    -------
    asset  : DataAsset with canonical columns, contracts attached.
    report : SpinelLoadReport with merge diagnostics and corruption count.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"raw spinel file not found: {raw_path}")
    if not cleaned_path.exists():
        raise FileNotFoundError(f"cleaned spinel file not found: {cleaned_path}")

    raw     = pd.read_csv(raw_path)
    cleaned = _read_cleaned_source(cleaned_path)

    # Corruption diagnostic before we touch anything
    r_b_in_cleaned = pd.to_numeric(cleaned["R_B"], errors="coerce")
    n_corrupt = int((r_b_in_cleaned > SHANNON_RADIUS_MAX_A).sum())

    # Harmonise raw column names
    raw_renamed = raw.rename(columns=SPINEL_RAW_RENAME)
    for col in SPINEL_DROPPED_VARIANT_COLS:
        if col in raw_renamed.columns:
            raw_renamed = raw_renamed.drop(columns=[col])

    # Inner-join to pick up uncorrupted E_hull and alpha from cleaned
    extra = cleaned[["compound", *SPINEL_CLEANED_EXTRA_COLS]]
    merged = raw_renamed.merge(extra, on="compound", how="inner")

    # Attrition audit
    raw_compounds   = set(raw_renamed["compound"])
    clean_compounds = set(cleaned["compound"])
    dropped         = tuple(sorted(raw_compounds - clean_compounds))

    # Reorder to canonical column order (tolerant of extras)
    ordered = [c for c in SPINEL_CANONICAL_COLS if c in merged.columns]
    extras  = [c for c in merged.columns if c not in SPINEL_CANONICAL_COLS]
    merged  = merged[ordered + extras]

    report = SpinelLoadReport(
        n_raw                             = len(raw),
        n_cleaned_source                  = len(cleaned),
        n_merged                          = len(merged),
        dropped_from_raw                  = dropped,
        radii_corruption_rows_in_cleaned  = n_corrupt,
    )

    prov = ProvenanceRecord(
        operation               = "load_canonical_spinel_asset",
        applied_utc             = utc_now_iso(),
        rows                    = len(merged),
        cols                    = merged.shape[1],
        params                  = {
            "raw_file":                           str(raw_path.name),
            "cleaned_file":                       str(cleaned_path.name),
            "merge_strategy":                     "inner_join_on_compound",
            "n_raw":                              len(raw),
            "n_cleaned_source":                   len(cleaned),
            "cleaned_file_corruption_rows_r_b":   n_corrupt,
            "dropped_raw_only_compounds":         list(dropped),
        },
        parent_analysis_id      = None,
        parent_dataframe_sha256 = None,
    )

    asset = DataAsset(
        dataframe        = merged,
        metadata         = {
            "dataset_name":     "spinel_canonical_249",
            "target_column":    "E_a",
            "contracts":        SPINEL_CONTRACTS,
            "load_report":      report,
            "provenance_chain": (prov,),
        },
        statistics       = {},
        integrity_report = {},
        source_path      = str(raw_path),
    )
    return asset, report


if __name__ == "__main__":
    asset, rep = load_canonical_spinel_asset()
    df = asset.dataframe
    print(f"loaded {len(df)} rows × {df.shape[1]} cols")
    print(f"  raw source:     {rep.n_raw} rows")
    print(f"  cleaned source: {rep.n_cleaned_source} rows")
    print(f"  merged (inner): {rep.n_merged} rows")
    print(f"  dropped raw-only compounds ({len(rep.dropped_from_raw)}): "
          f"{list(rep.dropped_from_raw)}")
    print(f"  R_B corruption in cleaned source: "
          f"{rep.radii_corruption_rows_in_cleaned} rows "
          f"(> {SHANNON_RADIUS_MAX_A} Å upper bound)")
    print()
    print("canonical columns present:")
    for c in SPINEL_CANONICAL_COLS:
        mark = "✓" if c in df.columns else "✗"
        print(f"  {mark} {c}")
    print()
    print("numeric summary of critical columns:")
    for c in ["E_a", "E_hull", "R_A", "R_B", "R_X", "alpha", "d_AX", "U"]:
        s = df[c]
        print(f"  {c:8s}  min={s.min():10.4f}  max={s.max():10.4f}  "
              f"median={s.median():10.4f}  n_nan={s.isna().sum()}")
