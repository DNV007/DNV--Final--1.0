"""
DNV Scientific Module
---------------------
Role:
    Pure computation of the three-step surrogate-shortcut audit described
    in DNV 2.0 Slide 10.  Tests whether a model's predictive success for a
    target observable y is carried by a genuine mechanism or by proxy
    reconstruction through cheap audit features.

Scientific Context:
    Let y be the regression target, p a mechanistically-meaningful proxy
    variable (e.g. a DFT-structural parameter), and X_aud a set of cheap
    tabulated audit features that we suspect may encode p.  The full
    feature set X_full = X_aud ∪ (other features).

    Three-step protocol:
      (1) Encoding test:   R²₁ = R²( p | X_aud )
          If X_aud already recovers p, a y-prediction built from X_aud is
          "predicting p and riding its correlation with y."
      (2) Baseline test:   R²₂ = R²( y | X_full )
          How well the full model actually predicts the target.
      (3) Residualisation: fit p̂ = f(X_aud), form y_res = y − g(p),
          then R²₃ = R²( y_res | X_aud ).  Ideally this should collapse
          to near zero if the audit features only reach y *through* p.
          An alternative (X_full \ p) ablation is also reported.

    Shortcut verdict:
      - "shortcut"        : R²₁ ≥ τ_proxy AND (R²₂ − R²₃) ≥ τ_drop
      - "partial_shortcut": R²₁ ≥ τ_proxy AND (R²₂ − R²₃) < τ_drop
      - "no_shortcut"     : R²₁ < τ_proxy  (audit features do not encode p)

Invariants:
    - compute_surrogate_audit returns a ShortcutAuditResult with is_valid
      set correctly; no exception propagates to the caller for ordinary
      data issues (empty frames, insufficient rows → is_valid=False).
    - All R² values are cross-validated with the same KFold shuffle.
    - Pure computation: no Qt, no I/O, no side effects.

Assumptions:
    - target_col, proxy_col, and all audit_features / full_features exist
      in df as numeric columns.
    - full_features typically contains proxy_col; if not, the ablation
      step is skipped with a note in the verdict.

Failure Modes:
    - Fewer than SHORTCUT_MIN_ROWS rows after NaN removal: is_valid=False.
    - An individual CV fit raising: that R² entry becomes NaN and the
      verdict is marked "inconclusive".

Provenance:
    - Emits no transformation metadata; caller binds a ProvenanceRecord
      referencing the (target, proxy, audit_features, full_features,
      model_class, cv_folds) tuple.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing   import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named constants — no bare literals
# ---------------------------------------------------------------------------

#: Minimum rows after NaN removal for a valid audit [count].
SHORTCUT_MIN_ROWS: Final[int] = 30

#: Default number of CV folds for every sub-task [count].
SHORTCUT_CV_FOLDS_DEFAULT: Final[int] = 5

#: Default random seed for KFold shuffling [dimensionless].
SHORTCUT_RANDOM_STATE_DEFAULT: Final[int] = 42

#: Threshold on R²₁ above which audit features are said to encode the
#: proxy. [dimensionless]
SHORTCUT_PROXY_R2_THRESHOLD: Final[float] = 0.75

#: Minimum drop (R²_full − R²_residual) that counts as a shortcut
#: collapse.  Smaller drops are classified "partial_shortcut". [dimensionless]
SHORTCUT_MIN_DROP: Final[float] = 0.25

#: Default tree count for the RandomForest surrogate used in every
#: sub-task when no explicit model_factory is supplied. [count]
SHORTCUT_RF_N_ESTIMATORS: Final[int] = 200

#: Default max depth for the RF surrogate (None → unrestricted). [count or None]
SHORTCUT_RF_MAX_DEPTH: Final[Optional[int]] = None

#: Human-readable verdict strings — single source of truth.
VERDICT_SHORTCUT:         Final[str] = "shortcut"
VERDICT_PARTIAL_SHORTCUT: Final[str] = "partial_shortcut"
VERDICT_NO_SHORTCUT:      Final[str] = "no_shortcut"
VERDICT_INCONCLUSIVE:     Final[str] = "inconclusive"


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShortcutAuditResult:
    """Three-step surrogate-shortcut audit output.

    Attributes
    ----------
    target_col        : name of the regression target y
    proxy_col         : name of the mechanistic proxy p
    audit_features    : tuple of str — the cheap/tabulated feature set
    full_features     : tuple of str — the full feature set (may include p)
    n_samples         : int — rows after NaN removal
    r2_proxy_from_audit   : float — R²( p | X_aud )               [step 1]
    r2_target_full        : float — R²( y | X_full )              [step 2]
    r2_target_residual    : float — R²( y_res | X_aud ) via residualisation
    r2_target_no_proxy    : float — R²( y | X_full \\ {proxy} ) ablation
                            (np.nan if proxy_col not in full_features)
    shortcut_drop         : float — r2_target_full − r2_target_residual
    ablation_drop         : float — r2_target_full − r2_target_no_proxy
                            (np.nan if ablation was skipped)
    proxy_r2_threshold    : float
    shortcut_min_drop     : float
    verdict               : one of the VERDICT_* constants
    message               : human-readable summary
    is_valid              : bool
    """
    target_col:            str
    proxy_col:             str
    audit_features:        Tuple[str, ...]
    full_features:         Tuple[str, ...]
    n_samples:             int
    r2_proxy_from_audit:   float
    r2_target_full:        float
    r2_target_residual:    float
    r2_target_no_proxy:    float
    shortcut_drop:         float
    ablation_drop:         float
    proxy_r2_threshold:    float
    shortcut_min_drop:     float
    verdict:               str
    message:               str
    is_valid:              bool


# ---------------------------------------------------------------------------
# Default model factory — shared across all three steps
# ---------------------------------------------------------------------------

def _default_rf_factory() -> Any:
    """Return a fresh RandomForestRegressor with fixed seed.

    Random forests give reasonable R² on both linear and non-linear
    signals with no feature scaling required, which matters because the
    three sub-tasks use different feature subsets.
    """
    return RandomForestRegressor(
        n_estimators = SHORTCUT_RF_N_ESTIMATORS,
        max_depth    = SHORTCUT_RF_MAX_DEPTH,
        random_state = SHORTCUT_RANDOM_STATE_DEFAULT,
        n_jobs       = -1,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    factory: Callable[[], Any],
    cv_folds: int,
    random_state: int,
) -> float:
    """K-fold cross_val_predict R² for one (X, y, model factory) triple.

    Returns np.nan on any sklearn exception so the caller can record an
    inconclusive step without the whole audit failing.
    """
    try:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(factory(), X, y, cv=kf)
        return float(r2_score(y, y_pred))
    except Exception as exc:
        log.warning("surrogate audit: CV fit failed: %s", exc)
        return float("nan")


def _cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    factory: Callable[[], Any],
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    """Return the KFold cross-validated predictions array, or y itself
    on failure (yielding zero-signal residuals, which collapse to the
    pure-baseline case and are handled by the verdict logic)."""
    try:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cross_val_predict(factory(), X, y, cv=kf)
    except Exception as exc:
        log.warning("surrogate audit: cross_val_predict failed: %s", exc)
        return y.copy()


def _standardise(X: np.ndarray) -> np.ndarray:
    """Column-standardise X (mean 0, std 1); constant columns stay 0."""
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_surrogate_audit(
    df:             pd.DataFrame,
    target_col:     str,
    proxy_col:      str,
    audit_features: Sequence[str],
    full_features:  Sequence[str],
    *,
    model_factory:  Optional[Callable[[], Any]] = None,
    cv_folds:       int = SHORTCUT_CV_FOLDS_DEFAULT,
    random_state:   int = SHORTCUT_RANDOM_STATE_DEFAULT,
    proxy_r2_threshold: float = SHORTCUT_PROXY_R2_THRESHOLD,
    shortcut_min_drop:  float = SHORTCUT_MIN_DROP,
) -> ShortcutAuditResult:
    """Run the three-step surrogate-shortcut audit on a single variant.

    Parameters
    ----------
    df             : DataFrame containing target, proxy, and all features
    target_col     : regression target y
    proxy_col      : mechanistic proxy variable p
    audit_features : tuple of suspected-shortcut features
    full_features  : full feature set (typically including proxy_col)
    model_factory  : zero-arg callable returning a fresh sklearn regressor.
                     Default: RandomForestRegressor.
    cv_folds       : KFold count for every sub-task
    random_state   : KFold shuffle seed
    proxy_r2_threshold : R² above which audit features are said to
                     encode the proxy
    shortcut_min_drop  : minimum (R²_full − R²_residual) that counts as
                     a shortcut collapse

    Returns
    -------
    ShortcutAuditResult
    """
    aud = tuple(audit_features)
    full = tuple(full_features)
    needed = list({target_col, proxy_col, *aud, *full})
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return _invalid(
            target_col, proxy_col, aud, full,
            message=f"columns missing from df: {missing}",
        )

    sub = df[needed].dropna()
    n = len(sub)
    if n < SHORTCUT_MIN_ROWS:
        return _invalid(
            target_col, proxy_col, aud, full,
            message=f"only {n} rows after NaN removal (need >= {SHORTCUT_MIN_ROWS})",
            n_samples=n,
        )

    factory = model_factory if model_factory is not None else _default_rf_factory

    y         = sub[target_col].values.astype(float)
    p         = sub[proxy_col].values.astype(float)
    X_aud_raw = sub[list(aud)].values.astype(float)
    X_full_raw = sub[list(full)].values.astype(float)

    X_aud = _standardise(X_aud_raw)
    X_full = _standardise(X_full_raw)

    # ── Step 1: R²( p | X_aud ) ──────────────────────────────────────────
    r2_proxy = _cv_r2(X_aud, p, factory, cv_folds, random_state)

    # ── Step 2: R²( y | X_full ) ─────────────────────────────────────────
    r2_full = _cv_r2(X_full, y, factory, cv_folds, random_state)

    # ── Step 3a: residualise y against the proxy, then R²( y_res | X_aud )
    # Simple linear residualisation: y_res = y − (a + b * p).
    # A non-linear residualisation (RF on p alone) collapses to the mean
    # when p has too little variance and biases the test; the linear form
    # is both scientifically conservative and interpretable.
    slope, intercept = np.polyfit(p, y, deg=1)
    y_res = y - (intercept + slope * p)
    r2_residual = _cv_r2(X_aud, y_res, factory, cv_folds, random_state)

    # ── Step 3b: ablation — R²( y | X_full \ {proxy} ) ───────────────────
    if proxy_col in full:
        full_no_proxy = [c for c in full if c != proxy_col]
        X_fnp = _standardise(sub[full_no_proxy].values.astype(float))
        r2_no_proxy = _cv_r2(X_fnp, y, factory, cv_folds, random_state)
    else:
        r2_no_proxy = float("nan")

    # ── Verdict ──────────────────────────────────────────────────────────
    any_nan = any(
        np.isnan(v) for v in (r2_proxy, r2_full, r2_residual)
    )
    if any_nan:
        verdict = VERDICT_INCONCLUSIVE
    else:
        shortcut_drop_val = r2_full - r2_residual
        if r2_proxy >= proxy_r2_threshold and shortcut_drop_val >= shortcut_min_drop:
            verdict = VERDICT_SHORTCUT
        elif r2_proxy >= proxy_r2_threshold:
            verdict = VERDICT_PARTIAL_SHORTCUT
        else:
            verdict = VERDICT_NO_SHORTCUT

    shortcut_drop = float(r2_full - r2_residual) if not any_nan else float("nan")
    ablation_drop = (
        float(r2_full - r2_no_proxy)
        if not (np.isnan(r2_no_proxy) or np.isnan(r2_full))
        else float("nan")
    )

    msg = (
        f"audit: R²(p|X_aud)={r2_proxy:.3f}  R²(y|X_full)={r2_full:.3f}  "
        f"R²(y_res|X_aud)={r2_residual:.3f}  Δ_shortcut={shortcut_drop:.3f}  "
        f"Δ_ablation={ablation_drop:.3f}  verdict={verdict}"
    )

    return ShortcutAuditResult(
        target_col          = target_col,
        proxy_col           = proxy_col,
        audit_features      = aud,
        full_features       = full,
        n_samples           = n,
        r2_proxy_from_audit = float(r2_proxy),
        r2_target_full      = float(r2_full),
        r2_target_residual  = float(r2_residual),
        r2_target_no_proxy  = float(r2_no_proxy),
        shortcut_drop       = shortcut_drop,
        ablation_drop       = ablation_drop,
        proxy_r2_threshold  = proxy_r2_threshold,
        shortcut_min_drop   = shortcut_min_drop,
        verdict             = verdict,
        message             = msg,
        is_valid            = not any_nan,
    )


def _invalid(
    target_col: str,
    proxy_col: str,
    audit_features: Tuple[str, ...],
    full_features: Tuple[str, ...],
    *,
    message: str,
    n_samples: int = 0,
) -> ShortcutAuditResult:
    return ShortcutAuditResult(
        target_col          = target_col,
        proxy_col           = proxy_col,
        audit_features      = audit_features,
        full_features       = full_features,
        n_samples           = n_samples,
        r2_proxy_from_audit = float("nan"),
        r2_target_full      = float("nan"),
        r2_target_residual  = float("nan"),
        r2_target_no_proxy  = float("nan"),
        shortcut_drop       = float("nan"),
        ablation_drop       = float("nan"),
        proxy_r2_threshold  = SHORTCUT_PROXY_R2_THRESHOLD,
        shortcut_min_drop   = SHORTCUT_MIN_DROP,
        verdict             = VERDICT_INCONCLUSIVE,
        message             = message,
        is_valid            = False,
    )
