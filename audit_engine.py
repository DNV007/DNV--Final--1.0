"""
DNV Scientific Module
---------------------
Role:
    Provides the surrogate audit workflow — a three-step diagnostic that tests
    whether predictive success for a target observable arises from genuine
    mechanistic signal or from proxy reconstruction.

Scientific Context:
    DNV-2.0 Slide 10: "Predictive success can arise from proxy reconstruction
    rather than the intended mechanism."  The audit fits a model from a
    declared set of proxy-only features to a structural parameter.  If the
    surrogate achieves high R², the target's accuracy may come from proxy
    reconstruction, not mechanism.  This separates "predicts well" from
    "targets the intended object."

    Three-step audit logic:
      1. Failure mode: tabulated-only E_a accuracy may come from proxy
         reconstruction.
      2. Audit task: predict structural parameter from proxy features only.
      3. Decision: if R² > threshold, flag as "proxy-recoverable."

Invariants:
    - All computation is pure: no Qt calls, no I/O, no side effects.
    - Every numeric literal is a named Final constant with a unit suffix.
    - compute_surrogate_audit always returns a SurrogateAuditResult with
      is_proxy_recoverable set correctly.

Assumptions:
    - df contains the target_col and all proxy_features as numeric columns.
    - At least MIN_AUDIT_ROWS rows remain after NaN removal.
    - sklearn is available in the dnv conda environment.

Failure Modes:
    - Fewer than MIN_AUDIT_ROWS valid rows after NaN removal: raises ValueError.
    - proxy_features empty: raises ValueError.
    - Model fitting fails: raises with diagnostic message.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numeric constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum valid rows after NaN removal for surrogate audit [dimensionless].
MIN_AUDIT_ROWS: Final[int] = 20

#: R² threshold above which a target is flagged as proxy-recoverable [dimensionless].
PROXY_R2_THRESHOLD_DEFAULT: Final[float] = 0.80

#: Default test fraction for train/test split [dimensionless].
AUDIT_TEST_SIZE_DEFAULT: Final[float] = 0.2

#: Default random state seed [dimensionless].
AUDIT_RANDOM_STATE_DEFAULT: Final[int] = 42

#: Number of cross-validation folds for surrogate model evaluation [dimensionless].
AUDIT_CV_FOLDS: Final[int] = 5

#: Default number of tree estimators for ensemble surrogate models [dimensionless].
AUDIT_N_ESTIMATORS_DEFAULT: Final[int] = 100

#: Default regularisation alpha for Ridge/Lasso surrogate models [dimensionless].
AUDIT_ALPHA_DEFAULT: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Surrogate model registry
# ---------------------------------------------------------------------------

#: Available surrogate model constructors: key → (title, factory).
_SURROGATE_MODELS: Final[Dict[str, Tuple[str, Callable]]] = {
    "linear": (
        "Linear Regression",
        lambda: LinearRegression(),
    ),
    "ridge": (
        "Ridge Regression",
        lambda: Ridge(alpha=AUDIT_ALPHA_DEFAULT),
    ),
    "lasso": (
        "Lasso Regression",
        lambda: Lasso(alpha=AUDIT_ALPHA_DEFAULT, max_iter=5000),
    ),
    "random_forest": (
        "Random Forest",
        lambda: RandomForestRegressor(
            n_estimators=AUDIT_N_ESTIMATORS_DEFAULT,
            random_state=AUDIT_RANDOM_STATE_DEFAULT,
        ),
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        lambda: GradientBoostingRegressor(
            n_estimators=AUDIT_N_ESTIMATORS_DEFAULT,
            random_state=AUDIT_RANDOM_STATE_DEFAULT,
        ),
    ),
}

#: Default surrogate model key.
DEFAULT_SURROGATE_MODEL: Final[str] = "ridge"


# ---------------------------------------------------------------------------
# Result record — frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateAuditResult:
    """Result of a surrogate audit: can proxy features recover the target?

    Attributes
    ----------
    target_col : str
        The structural parameter being predicted (e.g., anion parameter u).
    proxy_features : tuple of str
        Feature names used as proxy-only predictors.
    surrogate_model_key : str
        Key identifying the surrogate model type.
    surrogate_model_title : str
        Human-readable surrogate model name.
    r2 : float
        R² of the surrogate model (cross-validated).
    rmse : float
        RMSE of the surrogate model (cross-validated).
    mae : float
        MAE of the surrogate model (cross-validated).
    r2_threshold : float
        Threshold above which the target is flagged proxy-recoverable.
    is_proxy_recoverable : bool
        True if r2 >= r2_threshold — predictive success may be from proxy
        reconstruction, not mechanism.
    y_actual : np.ndarray
        Actual target values (for parity plot).
    y_predicted : np.ndarray
        Cross-validated predictions (for parity plot).
    n_samples : int
        Number of valid samples used.
    feature_importances : Optional[pd.Series]
        Per-proxy-feature importances (if model supports them).
    verdict : str
        Human-readable verdict string.
    """
    target_col:           str
    proxy_features:       Tuple[str, ...]
    surrogate_model_key:  str
    surrogate_model_title: str
    r2:                   float
    rmse:                 float
    mae:                  float
    r2_threshold:         float
    is_proxy_recoverable: bool
    y_actual:             np.ndarray
    y_predicted:          np.ndarray
    n_samples:            int
    feature_importances:  Optional[pd.Series] = None
    verdict:              str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_surrogate_audit(
    df: pd.DataFrame,
    target_col: str,
    proxy_features: List[str],
    *,
    surrogate_model: str = DEFAULT_SURROGATE_MODEL,
    r2_threshold: float = PROXY_R2_THRESHOLD_DEFAULT,
    cv_folds: int = AUDIT_CV_FOLDS,
    standardize: bool = True,
) -> SurrogateAuditResult:
    """Run the surrogate audit: predict target from proxy features only.

    If the surrogate model achieves R² >= threshold using only proxy features,
    the target is flagged as "proxy-recoverable" — meaning predictive success
    for the main model may arise from proxy reconstruction rather than genuine
    mechanistic signal.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing target and proxy feature columns.
    target_col : str
        The structural parameter to predict (e.g., anion parameter u).
    proxy_features : list of str
        Proxy-only features to use as predictors (e.g., ionic radii).
    surrogate_model : str
        Key from _SURROGATE_MODELS registry. Default "ridge".
    r2_threshold : float
        R² threshold for "proxy-recoverable" flag. Default 0.80.
    cv_folds : int
        Number of CV folds for evaluation. Default 5.
    standardize : bool
        Whether to standardise proxy features before fitting.

    Returns
    -------
    SurrogateAuditResult

    Raises
    ------
    ValueError
        If proxy_features is empty or too few rows after NaN removal.
    """
    if not proxy_features:
        raise ValueError("proxy_features must contain at least one feature.")

    missing = [c for c in proxy_features + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Subset and drop NaN
    sub = df[proxy_features + [target_col]].dropna()
    n = len(sub)
    if n < MIN_AUDIT_ROWS:
        raise ValueError(
            f"Only {n} valid rows after NaN removal (need >= {MIN_AUDIT_ROWS})."
        )

    X = sub[proxy_features].values.astype(float)
    y = sub[target_col].values.astype(float)

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Get surrogate model
    entry = _SURROGATE_MODELS.get(surrogate_model)
    if entry is None:
        raise ValueError(
            f"Unknown surrogate model: {surrogate_model!r}. "
            f"Available: {list(_SURROGATE_MODELS.keys())}"
        )
    model_title, model_factory = entry
    model = model_factory()

    # Cross-validated predictions
    effective_folds = min(cv_folds, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(model, X, y, cv=effective_folds)

    # Metrics from cross-validated predictions
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))

    # Fit the full model for importances
    model_full = model_factory()
    model_full.fit(X, y)
    importances = _extract_importances(model_full, proxy_features)

    is_recoverable = r2 >= r2_threshold

    if is_recoverable:
        verdict = (
            f"PROXY-RECOVERABLE: {target_col} can be predicted from proxy "
            f"features alone (R² = {r2:.4f} >= {r2_threshold}). Predictive "
            f"success for the main model may arise from proxy reconstruction, "
            f"not mechanism."
        )
    else:
        verdict = (
            f"NOT proxy-recoverable: {target_col} cannot be well predicted "
            f"from proxy features alone (R² = {r2:.4f} < {r2_threshold}). "
            f"Main model accuracy is less likely to be a proxy artefact."
        )

    return SurrogateAuditResult(
        target_col=target_col,
        proxy_features=tuple(proxy_features),
        surrogate_model_key=surrogate_model,
        surrogate_model_title=model_title,
        r2=r2,
        rmse=rmse,
        mae=mae,
        r2_threshold=r2_threshold,
        is_proxy_recoverable=is_recoverable,
        y_actual=y,
        y_predicted=y_pred,
        n_samples=n,
        feature_importances=importances,
        verdict=verdict,
    )


def available_surrogate_models() -> Dict[str, str]:
    """Return {key: title} for all available surrogate models."""
    return {k: v[0] for k, v in _SURROGATE_MODELS.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_importances(
    model: Any,
    feature_names: List[str],
) -> Optional[pd.Series]:
    """Extract feature importances or |coefficients| from a fitted model."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            model.feature_importances_,
            index=feature_names,
            name="importance",
        )
    if hasattr(model, "coef_"):
        coef = np.array(model.coef_).ravel()
        if len(coef) == len(feature_names):
            return pd.Series(
                np.abs(coef),
                index=feature_names,
                name="importance",
            )
    return None
