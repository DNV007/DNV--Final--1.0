"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for descriptor modelling — symbolic
    feature generation (unary and binary operations on base features) followed
    by linear screening or regularised regression to rank and select physically
    meaningful descriptors for a target property.

Scientific Context:
    Operates on (feature_matrix X, target_vector y) pairs derived from a
    DataAsset.  Applies element-wise unary transforms (x², x³, 1/x, |x|,
    √|x|, ∛x, log|x|, exp x) and pairwise binary operations (+, −, ×, ÷,
    |a−b|, harmonic mean) to generate a symbolic feature space up to
    FEATURE_TOTAL_CAP_DEFAULT columns.  Screening and regularisation are then
    applied to identify the most predictive descriptors.

Invariants:
    - Every compute_* function returns a DescriptorResult with a non-empty
      results_df and is_valid set appropriately.
    - generate_features always returns a DataFrame with all-NaN columns
      dropped and at most FEATURE_TOTAL_CAP_DEFAULT columns.
    - COEFF_SIGNIFICANCE_THRESHOLD is the only site that defines the
      "effectively zero coefficient" boundary.

Assumptions:
    - df contains at least 2 numeric columns with no all-NaN rows.
    - target_col is a column in df.
    - selected_ops is a dict with optional 'unary' and 'binary' list keys.
Failure Modes:
    - Fewer than MIN_VALID_ROWS rows after NaN-dropping: raises ValueError.
    - Singular covariance or degenerate data: is_valid=False in result.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import functools
import itertools
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warning suppression — scoped, not global
# ---------------------------------------------------------------------------
# sklearn emits UserWarning from LassoCV/ElasticNetCV (convergence, etc.).
# We silence them *only* inside compute_* bodies via @_quiet_user_warnings,
# never at module level — a module-level filterwarnings() pollutes every
# other module running in the same process.

@contextmanager
def _suppress_user_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        yield


def _quiet_user_warnings(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _suppress_user_warnings():
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Numeric constants — all values carry explicit units or domain in name
# ---------------------------------------------------------------------------

#: Maximum number of features to use in binary combination iteration [dimensionless].
FEATURE_COMBO_CAP_DEFAULT: Final[int] = 30

#: Hard cap on total generated features to prevent memory overrun [dimensionless].
FEATURE_TOTAL_CAP_DEFAULT: Final[int] = 5000

#: Minimum valid row count after NaN-dropping for model fitting [dimensionless].
MIN_VALID_ROWS: Final[int] = 10

#: Minimum row count for a single-feature pair fit [dimensionless].
MIN_VALID_PAIR: Final[int] = 2

#: Absolute-value threshold below which a regularised coefficient is treated as zero.
COEFF_SIGNIFICANCE_THRESHOLD: Final[float] = 1e-6

#: Default R² filter threshold for linear screening [dimensionless].
DEFAULT_R2_THRESHOLD: Final[float] = 0.0

#: Base-10 log of the smallest alpha in Ridge/ElasticNet alpha grid.
RIDGE_ALPHAS_LOG_START: Final[float] = -6.0

#: Base-10 log of the largest alpha in Ridge/ElasticNet alpha grid.
RIDGE_ALPHAS_LOG_STOP: Final[float] = 6.0

#: Number of alpha values in Ridge/ElasticNet log-spaced grid [dimensionless].
RIDGE_ALPHAS_COUNT: Final[int] = 20

#: Default maximum label length for formula strings in results [characters].
FORMULA_LABEL_MAX_CHARS: Final[int] = 60

#: n_features / n_samples ratio at or above which the overfitting warning fires.
OVERFIT_RATIO_WARN: Final[float] = 1.0

#: Minimum cross-validated R² below which a DescriptorResult is marked
#: is_valid=False. R² < MIN_VALID_R2 means the model predicts worse
#: than the y-mean baseline by more than this margin — scientifically,
#: that is not a fit worth reporting. Anything below 0 is already a
#: failed model; we use -1 to allow small negative CV fluctuations on
#: low-information feature subsets without flagging them as outright
#: broken.
MIN_VALID_R2: Final[float] = -1.0

#: Number of CV folds used by `_score_subset` to guide the combinatorial
#: search. Must match the fold count used downstream in `_finalise_search`
#: so the search objective and the reported metric live in the same space.
SEARCH_CV_FOLDS: Final[int] = 5

#: Fixed random seed for `_score_subset` CV splits. Fixing it means that
#: two evaluations of the same subset return the same score, which is a
#: hard requirement for any deterministic search (SA, CEM, PT, QA, ...).
SEARCH_CV_RANDOM_STATE: Final[int] = 0

#: Number of top features to preselect by univariate Pearson R² before search [dimensionless].
SEARCH_PRESELECT_DEFAULT: Final[int] = 100

#: Minimum number of cross-validation folds for parity computation [dimensionless].
CV_FOLDS_MIN: Final[int] = 2

#: Maximum number of cross-validation folds for parity computation [dimensionless].
CV_FOLDS_MAX: Final[int] = 5

#: Preselect cap for Greedy Backward (smaller pool for tractability) [dimensionless].
GREEDY_BACKWARD_PRESELECT_DEFAULT: Final[int] = 30

#: Maximum feature subset size in Greedy Forward [dimensionless].
GREEDY_MAX_FEATURES_DEFAULT: Final[int] = 20

#: Minimum feature subset size in Greedy Backward [dimensionless].
GREEDY_MIN_FEATURES_DEFAULT: Final[int] = 1

#: Default maximum iterations for stochastic search methods [dimensionless].
SEARCH_MAX_ITER_DEFAULT: Final[int] = 2000

#: Default initial subset size for stochastic search methods [dimensionless].
SEARCH_K_INIT_DEFAULT: Final[int] = 5

#: Default initial temperature for Simulated Annealing [dimensionless].
SA_T0_DEFAULT: Final[float] = 0.5

#: Default geometric cooling rate for SA (multiplied each step) [dimensionless ratio].
SA_COOLING_RATIO_DEFAULT: Final[float] = 0.995

#: Rolling window length for ARMHC acceptance-rate evaluation [dimensionless].
ARMHC_ADAPT_WINDOW: Final[int] = 100

#: Lower bound of healthy acceptance-rate band; below this ARMHC increases mutation [dimensionless ratio].
ARMHC_ADAPT_LO_THRESHOLD: Final[float] = 0.10

#: Upper bound of healthy acceptance-rate band; above this ARMHC decreases mutation [dimensionless ratio].
ARMHC_ADAPT_HI_THRESHOLD: Final[float] = 0.20

#: Starting fraction of features flipped per ARMHC mutation step [dimensionless ratio].
ARMHC_INIT_MUTATION_INTENSITY_RATIO: Final[float] = 0.10

#: Multiplicative scale-up factor when ARMHC acceptance rate is too low [dimensionless ratio].
ARMHC_INTENSITY_SCALE_UP_RATIO: Final[float] = 1.5

#: Maximum random fractional decrease applied when ARMHC acceptance rate is too high [dimensionless ratio].
ARMHC_INTENSITY_SCALE_DOWN_MAX_RATIO: Final[float] = 0.30

#: Minimum allowed ARMHC mutation intensity [dimensionless ratio].
ARMHC_INTENSITY_MIN_RATIO: Final[float] = 0.01

#: Maximum allowed ARMHC mutation intensity [dimensionless ratio].
ARMHC_INTENSITY_MAX_RATIO: Final[float] = 0.50

#: Reactive Tabu Search — initial tabu tenure (iterations a feature is forbidden) [dimensionless].
RTS_INIT_TABU_TENURE_DEFAULT: Final[int] = 5

#: Reactive Tabu Search — tenure increment on solution revisit [dimensionless].
RTS_TABU_ADAPT_STEP_DEFAULT: Final[int] = 2

#: Reactive Tabu Search — candidate moves evaluated per iteration [dimensionless].
RTS_CANDIDATES_PER_ITER_DEFAULT: Final[int] = 20

#: Cross-Entropy Method — samples drawn per iteration [dimensionless].
CEM_N_SAMPLES_DEFAULT: Final[int] = 50

#: Cross-Entropy Method — fraction of top samples used to update probabilities [dimensionless ratio].
CEM_ELITE_FRAC_DEFAULT: Final[float] = 0.20

#: Cross-Entropy Method — smoothing factor for probability update (0=full update, 1=no update) [dimensionless ratio].
CEM_SMOOTHING_DEFAULT: Final[float] = 0.70

#: Cross-Entropy Method — default outer iterations (each does n_samples evaluations) [dimensionless].
CEM_MAX_ITER_DEFAULT: Final[int] = 200

#: Parallel Tempering — number of replica chains [dimensionless].
PT_N_REPLICAS_DEFAULT: Final[int] = 4

#: Parallel Tempering — temperature of coldest replica [dimensionless].
PT_T_MIN_DEFAULT: Final[float] = 0.05

#: Parallel Tempering — temperature of hottest replica [dimensionless].
PT_T_MAX_DEFAULT: Final[float] = 2.0

#: Parallel Tempering — iterations between replica swap attempts [dimensionless].
PT_SWAP_INTERVAL_DEFAULT: Final[int] = 50

#: Parallel Tempering — default iterations (each does n_replicas SA steps) [dimensionless].
PT_MAX_ITER_DEFAULT: Final[int] = 500

#: Quantum Annealing — number of Trotter slices for PIMC representation [dimensionless].
QA_N_TROTTER_DEFAULT: Final[int] = 8

#: Quantum Annealing — initial transverse field strength [dimensionless].
QA_GAMMA_INIT_DEFAULT: Final[float] = 2.0

#: Quantum Annealing — final transverse field strength [dimensionless].
QA_GAMMA_FINAL_DEFAULT: Final[float] = 0.001

#: Quantum Annealing — fixed thermal temperature for PIMC sampling [dimensionless].
QA_T_FIXED_DEFAULT: Final[float] = 0.5

#: PSO — number of particles in the swarm [dimensionless].
PSO_N_PARTICLES_DEFAULT: Final[int] = 30

#: PSO — inertia weight [dimensionless ratio].
PSO_W_DEFAULT: Final[float] = 0.7

#: PSO — cognitive acceleration coefficient [dimensionless].
PSO_C1_DEFAULT: Final[float] = 1.5

#: PSO — social acceleration coefficient [dimensionless].
PSO_C2_DEFAULT: Final[float] = 1.5

#: PSO — velocity clamping maximum [dimensionless].
PSO_V_MAX_DEFAULT: Final[float] = 4.0

#: PSO — sigmoid threshold for converting continuous position to binary mask [dimensionless].
PSO_SIGMOID_THRESHOLD: Final[float] = 0.5

#: GP — tournament selection size [dimensionless].
GP_TOURNAMENT_SIZE_DEFAULT: Final[int] = 3

#: GP — population size [dimensionless].
GP_POP_SIZE_DEFAULT: Final[int] = 50

#: GP — maximum generations [dimensionless].
GP_MAX_GENS_DEFAULT: Final[int] = 40

#: GP — crossover probability [dimensionless ratio].
GP_CROSSOVER_PROB_DEFAULT: Final[float] = 0.7

#: GP — mutation probability per gene [dimensionless ratio].
GP_MUTATION_PROB_DEFAULT: Final[float] = 0.1

#: GP — elitism count [dimensionless].
GP_ELITISM_COUNT_DEFAULT: Final[int] = 2

#: GP — parsimony penalty per selected feature [dimensionless coefficient].
GP_PARSIMONY_COEFF_DEFAULT: Final[float] = 0.005


# ---------------------------------------------------------------------------
# Operator registries — single source of truth for all supported operations
# ---------------------------------------------------------------------------

#: Unary operation template strings.  'x' is replaced by the backtick-quoted feature.
_UNARY_OPS_MAP: Final[Dict[str, str]] = {
    "sq":   "(x)**2",
    "cb":   "(x)**3",
    "inv":  "1/(x+1e-6)",
    "abs":  "abs(x)",
    "sqrt": "sqrt(abs(x))",
    "cbrt": "(abs(x))**(1/3)",
    "log":  "log(abs(x)+1e-6)",
    "exp":  "exp(x)",
}

#: Binary operation template strings.  '{a}' and '{b}' are backtick-quoted features.
_BINARY_OPS_MAP: Final[Dict[str, str]] = {
    "add":          "({a}) + ({b})",
    "sub":          "({a}) - ({b})",
    "mul":          "({a}) * ({b})",
    "div":          "({a}) / ({b}+1e-6)",
    "abs_diff":     "abs({a} - {b})",
    "harmonic_mean":"2 / (1/({a}+1e-6) + 1/({b}+1e-6))",
}

#: Binary operators whose result is invariant under argument order.
#: For these, `itertools.combinations` is used instead of `permutations`
#: to avoid generating duplicate columns (e.g. `a + b` and `b + a` are
#: literally the same column). Emitting both duplicates makes the
#: design matrix rank-deficient and ruins any downstream OLS fit —
#: this is the single root cause of the descriptor-engine catastrophic
#: CV-R² documented in ENGINE_AUDIT_2026_04_15.md §4.
_COMMUTATIVE_BINARY_OPS: Final[frozenset[str]] = frozenset({
    "add", "mul", "abs_diff", "harmonic_mean",
})


# ---------------------------------------------------------------------------
# DescriptorResult — typed, immutable output record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DescriptorResult:
    """
    Typed, immutable output of a single descriptor-modelling computation.

    Fields
    ------
    method_key    : machine identifier matching _DISPATCH key
    method_title  : human-readable method name
    target_col    : name of the target column used
    results_df    : scored descriptor table (schema varies by method — see below)
    n_generated   : total number of symbolic features generated
    overall_r2    : R² of the full model (None for single-feature screening)
    overall_rmse  : RMSE of the full model (None for single-feature screening)
    extra         : additional data: y_true, y_pred, model_equation, etc.
    is_valid      : False if computation degraded

    results_df schema
    -----------------
    linear_screening  : ['formula', 'cv_r2']    (K-fold CV R² for single-feature models)
    lasso/ridge/enet  : ['formula', 'coefficient', 'r2_single', 'rmse_single']

    extra keys (regularised methods)
    ---------------------------------
    train_r2        : float  — in-sample R² (reported alongside CV R² for transparency)
    n_samples       : int    — number of rows used for fitting
    n_features      : int    — number of symbolic features passed to the model
    overfit_warning : bool   — True when n_features ≥ n_samples
    """
    method_key:    str
    method_title:  str
    target_col:    str
    results_df:    pd.DataFrame
    n_generated:   int
    overall_r2:    Optional[float]
    overall_rmse:  Optional[float]
    extra:         Dict[str, Any] = field(default_factory=dict)
    is_valid:      bool = True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def safe_eval_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """
    Evaluate a pandas formula string on df, returning a NaN-filled Series on
    any error.

    All infinities are replaced with NaN.  A formula that evaluates to all-zero
    or all-NaN is treated as uninformative and also returns a NaN Series.
    """
    try:
        with np.errstate(all="ignore"):
            result = df.eval(formula, engine="python")
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        result = result.replace([np.inf, -np.inf], np.nan)
        if result.isnull().all() or (result == 0).all():
            return pd.Series([np.nan] * len(df), index=df.index)
        return result
    except Exception:
        return pd.Series([np.nan] * len(df), index=df.index)


def generate_features(
    df: pd.DataFrame,
    base_features: List[str],
    selected_ops: Dict[str, List[str]],
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
) -> pd.DataFrame:
    """
    Generate a symbolic feature space from base_features via unary and binary
    operations.

    Parameters
    ----------
    df            : source DataFrame (must contain base_features columns)
    base_features : column names to use as seeds
    selected_ops  : {'unary': [...], 'binary': [...]} operation keys
    combo_cap     : maximum columns to iterate over for binary combinations
    total_cap     : hard cap on total output columns

    Returns
    -------
    pd.DataFrame with all-NaN columns dropped and at most total_cap columns.
    """
    feature_df = df[base_features].copy()

    # ── Unary transforms ─────────────────────────────────────────────────
    new_unary: List[pd.Series] = []
    for op in selected_ops.get("unary", []):
        template = _UNARY_OPS_MAP.get(op)
        if not template:
            continue
        for feat in base_features:
            col_name = f"{op}({feat})"
            formula  = template.replace("x", f"`{feat}`")
            s = safe_eval_formula(feature_df, formula)
            s.name = col_name
            new_unary.append(s)
    if new_unary:
        feature_df = pd.concat([feature_df] + new_unary, axis=1)

    # ── Binary combinations ───────────────────────────────────────────────
    cols_for_binary = list(feature_df.columns)
    if len(cols_for_binary) > combo_cap:
        log.warning(
            "generate_features: %d features after unary ops; "
            "capping binary iteration at %d.",
            len(cols_for_binary), combo_cap,
        )
        cols_for_binary = cols_for_binary[:combo_cap]

    new_binary: List[pd.Series] = []
    for op in selected_ops.get("binary", []):
        template = _BINARY_OPS_MAP.get(op)
        if not template:
            continue
        pair_iter = (
            itertools.combinations(cols_for_binary, 2)
            if op in _COMMUTATIVE_BINARY_OPS
            else itertools.permutations(cols_for_binary, 2)
        )
        for a, b in pair_iter:
            col_name = (f"{a} {op} {b}")[:FORMULA_LABEL_MAX_CHARS]
            formula  = template.format(a=f"`{a}`", b=f"`{b}`")
            s = safe_eval_formula(feature_df, formula)
            s.name = col_name
            new_binary.append(s)
    if new_binary:
        feature_df = pd.concat([feature_df] + new_binary, axis=1)

    log.debug("generate_features: produced %d raw features.", len(feature_df.columns))

    if len(feature_df.columns) > total_cap:
        log.warning(
            "generate_features: %d features; capping at %d.",
            len(feature_df.columns), total_cap,
        )
        feature_df = feature_df.iloc[:, :total_cap]

    return feature_df.dropna(axis=1, how="all")


# ---------------------------------------------------------------------------
# 1. Linear Screening — single-feature R² ranking
# ---------------------------------------------------------------------------

def compute_linear_screening(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
) -> DescriptorResult:
    """
    Score every generated feature by its single-feature linear R² with the
    target.  Returns all features with R² ≥ r2_threshold, sorted descending.

    extra keys
    ----------
    y_true      : np.ndarray — target values for the best-formula parity plot
    y_pred_best : np.ndarray — predictions for the best formula
    best_formula: str        — formula string of the highest-R² feature
    n_passing   : int        — number of features that passed the threshold
    """
    feature_df = generate_features(df, selected_columns, selected_ops, combo_cap, total_cap)
    n_generated = len(feature_df.columns)
    y_full = df[target_col]

    # Reported R² must be cross-validated, not in-sample, to avoid
    # winner's-curse inflation when screening thousands of candidate
    # formulas (DNV research-grade metrics rule).  Use K-fold CV with a
    # fixed shuffled splitter so all candidates are ranked on identical
    # partitions, making scores directly comparable.
    passing: List[Tuple[str, float]] = []
    for col in feature_df.columns:
        if col == target_col:
            continue
        combined = pd.concat(
            [feature_df[col].rename("feature"), y_full], axis=1
        ).dropna()
        n_c = len(combined)
        if n_c < MIN_VALID_PAIR:
            continue
        X_c = combined[["feature"]].values
        y_c = combined[target_col].values
        try:
            n_splits = max(CV_FOLDS_MIN, min(CV_FOLDS_MAX, n_c // CV_FOLDS_MIN))
            if n_splits < CV_FOLDS_MIN:
                continue
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            cv_r2 = float(cross_val_score(
                LinearRegression(), X_c, y_c, cv=kf, scoring="r2"
            ).mean())
            if cv_r2 >= r2_threshold:
                passing.append((col, cv_r2))
        except Exception:
            continue

    passing.sort(key=lambda t: t[1], reverse=True)
    log.info(
        "compute_linear_screening: %d / %d features passed CV R² ≥ %.3f.",
        len(passing), n_generated, r2_threshold,
    )

    results_df = pd.DataFrame(passing, columns=["formula", "cv_r2"])

    # Parity data for best formula
    y_true_best = np.array([])
    y_pred_best = np.array([])
    best_formula = ""
    if passing:
        best_formula = passing[0][0]
        combined_b = pd.concat(
            [feature_df[best_formula].rename("feature"), y_full], axis=1
        ).dropna()
        X_b = combined_b[["feature"]].values
        y_b = combined_b[target_col].values
        try:
            n_b  = len(y_b)
            cv_b = max(CV_FOLDS_MIN, min(CV_FOLDS_MAX, n_b // CV_FOLDS_MIN))
            y_pred_best = cross_val_predict(LinearRegression(), X_b, y_b, cv=cv_b)
            y_true_best = y_b
        except Exception as exc:
            # CV failed — parity data is unavailable; in-sample predictions are
            # forbidden by the research-grade metrics rule (DNV_STYLE_GUIDE §6).
            log.warning(
                "compute_linear_screening: cross_val_predict failed for best "
                "formula '%s' (%s). Parity data will be empty.",
                best_formula, exc,
            )

    best_cv_r2 = float(passing[0][1]) if passing else None

    return DescriptorResult(
        method_key="linear_screening",
        method_title="Linear Screening (CV R² per formula)",
        target_col=target_col,
        results_df=results_df,
        n_generated=n_generated,
        overall_r2=best_cv_r2,
        overall_rmse=None,
        extra={
            "y_true":          y_true_best,
            "y_pred_best":     y_pred_best,
            "best_formula":    best_formula,
            "n_passing":       len(passing),
            # Parity keys expected by gui_dlg_desc_plot.py overfit annotation.
            # For single-feature screening, n_features is always 1 (the best
            # formula is scalar) and overfit_warning is always False since
            # n_samples >> 1 for any usable dataset.
            "n_samples":       int(len(y_true_best)) if len(y_true_best) > 0 else 0,
            "n_features":      1 if passing else 0,
            "overfit_warning": False,
            "train_r2":        best_cv_r2,
        },
        is_valid=len(passing) > 0,
    )


# ---------------------------------------------------------------------------
# 2–4. Regularised CV models — shared implementation
# ---------------------------------------------------------------------------

def _run_regularised(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    method_key: str,
    method_title: str,
    model,
    combo_cap: int,
    total_cap: int,
) -> DescriptorResult:
    """
    Shared regularised-model kernel.

    1. generate_features → symbolic feature space
    2. Join with y, drop NaN rows
    3. StandardScaler on X
    4. Fit the supplied CV model
    5. For each selected feature (|coeff| > COEFF_SIGNIFICANCE_THRESHOLD):
       fit single LinearRegression → individual R² and RMSE
    6. Return DescriptorResult
    """
    feature_df = generate_features(df, selected_columns, selected_ops, combo_cap, total_cap)
    n_generated = len(feature_df.columns)

    combined = feature_df.join(df[target_col]).dropna()
    if len(combined) < MIN_VALID_ROWS:
        raise ValueError(
            f"Only {len(combined)} valid rows remain after feature generation "
            f"and NaN removal (need ≥ {MIN_VALID_ROWS})."
        )

    X_df = combined.drop(columns=[target_col])
    y    = combined[target_col].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    model.fit(X_scaled, y)
    coef   = np.array(model.coef_).ravel()
    n_samp = len(y)
    n_feat = X_scaled.shape[1]

    # ── Honest out-of-sample metrics via cross-validated predictions ─────────
    best_alpha = float(model.alpha_)
    cv_folds   = max(2, min(5, n_samp // 2))

    if method_key == "lasso":
        fixed_model: Any = Lasso(
            alpha=best_alpha,
            max_iter=getattr(model, "max_iter", 5000),
            random_state=getattr(model, "random_state", 42),
        )
    elif method_key == "ridge":
        fixed_model = Ridge(alpha=best_alpha)
    elif method_key == "elasticnet":
        fixed_model = ElasticNet(
            alpha=best_alpha,
            l1_ratio=float(getattr(model, "l1_ratio_", 0.5)),
            max_iter=getattr(model, "max_iter", 5000),
            random_state=getattr(model, "random_state", 42),
        )
    else:
        fixed_model = None

    # Training metrics (kept in extra for transparency)
    train_r2   = float(model.score(X_scaled, y))
    y_pred_all = model.predict(X_scaled).ravel()
    overall_r2   = train_r2
    overall_rmse = float(np.sqrt(mean_squared_error(y, y_pred_all)))

    if fixed_model is not None:
        try:
            y_cv         = cross_val_predict(fixed_model, X_scaled, y, cv=cv_folds)
            overall_r2   = float(r2_score(y, y_cv))
            overall_rmse = float(np.sqrt(mean_squared_error(y, y_cv)))
            y_pred_all   = y_cv  # parity plot shows OOS predictions
        except Exception as cv_exc:
            log.warning(
                "_run_regularised [%s]: cross_val_predict failed (%s); "
                "reporting training metrics.",
                method_key, cv_exc,
            )

    overfit_warn = n_feat >= n_samp
    if overfit_warn:
        log.warning(
            "_run_regularised [%s]: n_features=%d >= n_samples=%d — "
            "CV metrics are the honest estimate; training R²=%.4f may be inflated.",
            method_key, n_feat, n_samp, train_r2,
        )

    selected_idx = np.where(np.abs(coef) > COEFF_SIGNIFICANCE_THRESHOLD)[0]
    log.info(
        "_run_regularised [%s]: %d / %d features selected; CV R²=%.4f  train R²=%.4f",
        method_key, len(selected_idx), n_generated, overall_r2, train_r2,
    )

    rows: List[Tuple] = []
    for idx in selected_idx:
        feat = X_df.columns[idx]
        c    = float(coef[idx])
        col_vals = X_df.iloc[:, idx].values.reshape(-1, 1)
        combined_s = pd.DataFrame({"f": col_vals.ravel(), "y": y}).dropna()
        if len(combined_s) < MIN_VALID_PAIR:
            continue
        try:
            m_s = LinearRegression().fit(
                combined_s[["f"]].values, combined_s["y"].values
            )
            r2_s   = float(m_s.score(combined_s[["f"]].values, combined_s["y"].values))
            rmse_s = float(np.sqrt(mean_squared_error(
                combined_s["y"].values,
                m_s.predict(combined_s[["f"]].values),
            )))
        except Exception:
            r2_s = float("nan"); rmse_s = float("nan")
        rows.append((feat, c, r2_s, rmse_s))

    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    results_df = pd.DataFrame(rows, columns=["formula", "coefficient", "r2_single", "rmse_single"])

    # Build model equation string
    intercept   = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
    eq_lines    = [f"{target_col} = {intercept:.5f}"]
    for fname, coef_val, _, _ in rows:
        sign = "+" if coef_val >= 0 else "-"
        eq_lines.append(f"  {sign} ({abs(coef_val):.5f} × scale(`{fname}`))")
    model_equation = "\n".join(eq_lines)

    # Selected alpha
    alpha_val = float(getattr(model, "alpha_", float("nan")))
    l1_ratio  = float(getattr(model, "l1_ratio_", float("nan"))) if hasattr(model, "l1_ratio_") else None

    extra: Dict[str, Any] = {
        "y_true":          y,
        "y_pred":          y_pred_all,
        "model_equation":  model_equation,
        "selected_alpha":  alpha_val,
        "train_r2":        train_r2,
        "n_samples":       n_samp,
        "n_features":      n_feat,
        "overfit_warning": overfit_warn,
    }
    if l1_ratio is not None:
        extra["l1_ratio"] = l1_ratio

    return DescriptorResult(
        method_key=method_key,
        method_title=method_title,
        target_col=target_col,
        results_df=results_df,
        n_generated=n_generated,
        overall_r2=overall_r2,
        overall_rmse=overall_rmse,
        extra=extra,
        is_valid=len(rows) > 0,
    )


def compute_lasso(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    cv: int = 5,
    max_iter: int = 5000,
    rs: int = 42,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
) -> DescriptorResult:
    """Lasso CV descriptor selection."""
    model = LassoCV(cv=cv, max_iter=max_iter, random_state=rs, n_jobs=-1)
    return _run_regularised(
        df, selected_columns, target_col, selected_ops,
        "lasso", "Lasso CV", model, combo_cap, total_cap,
    )


def compute_ridge(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
) -> DescriptorResult:
    """Ridge CV descriptor selection (no random state — deterministic)."""
    alphas = np.logspace(RIDGE_ALPHAS_LOG_START, RIDGE_ALPHAS_LOG_STOP, RIDGE_ALPHAS_COUNT)
    model  = RidgeCV(alphas=alphas)
    return _run_regularised(
        df, selected_columns, target_col, selected_ops,
        "ridge", "Ridge CV", model, combo_cap, total_cap,
    )


def compute_elasticnet(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    cv: int = 5,
    max_iter: int = 5000,
    rs: int = 42,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
) -> DescriptorResult:
    """Elastic Net CV descriptor selection."""
    alphas = np.logspace(RIDGE_ALPHAS_LOG_START, RIDGE_ALPHAS_LOG_STOP, RIDGE_ALPHAS_COUNT)
    model  = ElasticNetCV(
        alphas=alphas, cv=cv, max_iter=max_iter, random_state=rs, n_jobs=-1,
    )
    return _run_regularised(
        df, selected_columns, target_col, selected_ops,
        "elasticnet", "Elastic Net CV", model, combo_cap, total_cap,
    )


# ---------------------------------------------------------------------------
# Combinatorial search — shared helpers
# ---------------------------------------------------------------------------

def _preselect_univariate(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized univariate Pearson R² preselection.

    Returns an index array of the top-k features by |r|².  k is clamped to
    X.shape[1] if larger.  Protects against zero-variance columns via 1e-12.
    """
    y_z  = (y - y.mean()) / (y.std() + 1e-12)
    X_z  = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    r    = (X_z * y_z[:, np.newaxis]).mean(axis=0)
    k_eff = min(k, X.shape[1])
    return np.argsort(r ** 2)[-k_eff:]


def _score_subset(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """
    SEARCH_CV_FOLDS-fold cross-validated R² for a boolean feature mask,
    clamped to [0.0, 1.0].

    The combinatorial search methods (SA, PT, CEM, QA, PSO, RTS, GP,
    greedy-forward/backward, ARMHC) all call this function to score a
    candidate subset. It must use the **same** metric the method will
    report downstream — otherwise the search objective and the reported
    result drift apart and the search optimises for overfit.

    Returns 0.0 for empty subsets or any numerical failure. Negative
    CV R² values are clamped to 0.0 so the search treats "worse than
    baseline" subsets as uniformly uninformative rather than creating a
    negative-R² gradient that biases the search toward one broken
    subset over another.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    X_sub = X[:, idx]
    n_samp = X_sub.shape[0]
    k = min(SEARCH_CV_FOLDS, max(2, n_samp // 2))
    try:
        kf = KFold(n_splits=k, shuffle=True, random_state=SEARCH_CV_RANDOM_STATE)
        s = float(cross_val_score(
            Ridge(alpha=1.0), X_sub, y, cv=kf, scoring="r2",
        ).mean())
        if not np.isfinite(s):
            return 0.0
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0


def _prepare_search(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    preselect_k: int,
    combo_cap: int,
    total_cap: int,
) -> Tuple["pd.DataFrame", np.ndarray, np.ndarray, int]:
    """
    Shared pre-processing kernel for all combinatorial search methods.

    Steps:
    1. generate_features → symbolic feature space
    2. Join with target, drop NaN rows
    3. Validate minimum row count
    4. StandardScaler on X
    5. _preselect_univariate → top-k subspace

    Returns (X_df_pre, X_pre, y, n_generated).
    """
    feature_df  = generate_features(df, selected_columns, selected_ops, combo_cap, total_cap)
    n_generated = len(feature_df.columns)

    combined = feature_df.join(df[target_col]).dropna()
    if len(combined) < MIN_VALID_ROWS:
        raise ValueError(
            f"Only {len(combined)} valid rows after feature generation "
            f"and NaN removal (need ≥ {MIN_VALID_ROWS})."
        )

    X_df = combined.drop(columns=[target_col])
    y    = combined[target_col].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    top_idx  = _preselect_univariate(X_scaled, y, preselect_k)
    X_pre    = X_scaled[:, top_idx]
    X_df_pre = X_df.iloc[:, top_idx]

    return X_df_pre, X_pre, y, n_generated


def _finalise_search(
    X_df_pre: "pd.DataFrame",
    X_pre: np.ndarray,
    y: np.ndarray,
    best_mask: np.ndarray,
    score_history: List[float],
    method_key: str,
    method_title: str,
    target_col: str,
    n_generated: int,
    extra_kv: Dict[str, Any],
) -> DescriptorResult:
    """
    Shared post-processing kernel for all combinatorial search methods.

    1. Final RidgeCV fit on the selected features (regularisation is
       mandatory: the symbolic feature space is near-collinear by
       construction, and plain OLS produces catastrophic CV-R² values
       on rank-deficient subsets — see ENGINE_AUDIT_2026_04_15.md §4).
       The alpha grid is log-spaced
       [10^RIDGE_ALPHAS_LOG_START, 10^RIDGE_ALPHAS_LOG_STOP] with
       RIDGE_ALPHAS_COUNT points; RidgeCV picks by leave-one-out R².
    2. CV predictions via cross_val_predict with a fixed-alpha Ridge
       (alpha = RidgeCV.alpha_) so train and eval see the same model.
    3. Per-feature results_df (formula, coefficient, r2_single, rmse_single).
    4. Return DescriptorResult with is_valid guarded by MIN_VALID_R2.

    extra_kv is merged into result.extra (e.g. temp_history).
    """
    sel_idx = np.where(best_mask)[0]
    n_samp  = len(y)
    n_feat  = len(sel_idx)

    if n_feat == 0:
        empty_df = pd.DataFrame(columns=["formula", "coefficient", "r2_single", "rmse_single"])
        return DescriptorResult(
            method_key=method_key, method_title=method_title, target_col=target_col,
            results_df=empty_df, n_generated=n_generated,
            overall_r2=None, overall_rmse=None,
            extra={**extra_kv, "score_history": score_history,
                   "y_true": y, "y_pred": np.array([])},
            is_valid=False,
        )

    alphas = np.logspace(RIDGE_ALPHAS_LOG_START, RIDGE_ALPHAS_LOG_STOP, RIDGE_ALPHAS_COUNT)
    X_sel  = X_pre[:, sel_idx]
    ridge  = RidgeCV(alphas=alphas)
    ridge.fit(X_sel, y)
    alpha_star = float(ridge.alpha_)
    coef       = np.array(ridge.coef_).ravel()
    train_r2   = float(ridge.score(X_sel, y))

    cv_folds = max(2, min(5, n_samp // 2))
    try:
        y_cv         = cross_val_predict(Ridge(alpha=alpha_star), X_sel, y, cv=cv_folds)
        overall_r2   = float(r2_score(y, y_cv))
        overall_rmse = float(np.sqrt(mean_squared_error(y, y_cv)))
        y_pred_all   = y_cv
    except Exception as exc:
        log.warning("_finalise_search [%s]: cross_val_predict failed (%s).", method_key, exc)
        y_pred_all   = ridge.predict(X_sel).ravel()
        overall_r2   = train_r2
        overall_rmse = float(np.sqrt(mean_squared_error(y, y_pred_all)))

    overfit_warn = n_feat >= n_samp

    rows: List[Tuple] = []
    for local_i, global_i in enumerate(sel_idx):
        feat     = X_df_pre.columns[global_i]
        c        = float(coef[local_i])
        col_vals = X_pre[:, global_i].reshape(-1, 1)
        combined_s = pd.DataFrame({"f": col_vals.ravel(), "y": y}).dropna()
        if len(combined_s) < MIN_VALID_PAIR:
            continue
        try:
            m_s    = LinearRegression().fit(combined_s[["f"]].values, combined_s["y"].values)
            r2_s   = float(m_s.score(combined_s[["f"]].values, combined_s["y"].values))
            rmse_s = float(np.sqrt(mean_squared_error(
                combined_s["y"].values, m_s.predict(combined_s[["f"]].values),
            )))
        except Exception:
            r2_s = float("nan"); rmse_s = float("nan")
        rows.append((feat, c, r2_s, rmse_s))

    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    results_df = pd.DataFrame(rows, columns=["formula", "coefficient", "r2_single", "rmse_single"])

    intercept = float(ridge.intercept_)
    eq_lines  = [f"{target_col} = {intercept:.5f}"]
    for fname, coef_val, _, _ in rows:
        sign = "+" if coef_val >= 0 else "-"
        eq_lines.append(f"  {sign} ({abs(coef_val):.5f} × scale(`{fname}`))")
    model_equation = "\n".join(eq_lines)

    extra: Dict[str, Any] = {
        "y_true":          y,
        "y_pred":          y_pred_all,
        "score_history":   score_history,
        "model_equation":  model_equation,
        "train_r2":        train_r2,
        "ridge_alpha":     alpha_star,
        "n_samples":       n_samp,
        "n_features":      n_feat,
        "n_selected":      n_feat,
        "overfit_warning": overfit_warn,
    }
    extra.update(extra_kv)

    is_valid = (
        len(rows) > 0
        and overall_r2 is not None
        and np.isfinite(overall_r2)
        and overall_r2 >= MIN_VALID_R2
    )
    return DescriptorResult(
        method_key=method_key, method_title=method_title, target_col=target_col,
        results_df=results_df, n_generated=n_generated,
        overall_r2=overall_r2, overall_rmse=overall_rmse,
        extra=extra,
        is_valid=is_valid,
    )


# ---------------------------------------------------------------------------
# Search algorithm implementations (operate on preselected feature matrix)
# ---------------------------------------------------------------------------

def _search_greedy_forward(
    X: np.ndarray,
    y: np.ndarray,
    max_features: int,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Greedy forward selection: sequentially add the feature that most improves R².

    Returns (mask, score_history).
    """
    n_features    = X.shape[1]
    mask          = np.zeros(n_features, dtype=bool)
    score_history: List[float] = []
    best_score    = 0.0
    total         = min(max_features, n_features)

    for step in range(total):
        best_gain = -np.inf
        best_idx  = -1
        for i in range(n_features):
            if mask[i]:
                continue
            mask[i] = True
            s = _score_subset(X, y, mask)
            mask[i] = False
            gain = s - best_score
            if gain > best_gain:
                best_gain = gain
                best_idx  = i
        if best_idx < 0 or best_gain <= 0.0:
            break
        mask[best_idx] = True
        best_score = _score_subset(X, y, mask)
        score_history.append(best_score)
        if progress_cb is not None:
            progress_cb(step + 1, total, best_score)

    return mask, score_history


def _search_greedy_backward(
    X: np.ndarray,
    y: np.ndarray,
    min_features: int,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Greedy backward elimination: sequentially remove the feature with least harm.

    Returns (mask, score_history).
    """
    n_features    = X.shape[1]
    mask          = np.ones(n_features, dtype=bool)
    score_history: List[float] = [_score_subset(X, y, mask)]
    total         = n_features - min_features
    step          = 0

    while mask.sum() > min_features:
        cur_score = _score_subset(X, y, mask)
        best_loss = np.inf
        best_idx  = -1
        for i in range(n_features):
            if not mask[i]:
                continue
            mask[i] = False
            s = _score_subset(X, y, mask)
            mask[i] = True
            loss = cur_score - s
            if loss < best_loss:
                best_loss = loss
                best_idx  = i
        if best_idx < 0:
            break
        mask[best_idx] = False
        step += 1
        score_history.append(_score_subset(X, y, mask))
        if progress_cb is not None:
            progress_cb(step, total, score_history[-1])

    return mask, score_history


def _search_armhc(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    init_mutation_intensity: float = ARMHC_INIT_MUTATION_INTENSITY_RATIO,
    adapt_lo: float = ARMHC_ADAPT_LO_THRESHOLD,
    adapt_hi: float = ARMHC_ADAPT_HI_THRESHOLD,
    adapt_window: int = ARMHC_ADAPT_WINDOW,
    intensity_scale_up: float = ARMHC_INTENSITY_SCALE_UP_RATIO,
    intensity_scale_down_max: float = ARMHC_INTENSITY_SCALE_DOWN_MAX_RATIO,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Adaptive Random Mutation Hill Climbing.

    Acceptance rule: strict improvement only.

    Mutation intensity (fraction of features flipped per step) adapts every
    ARMHC_ADAPT_WINDOW steps based on the recent acceptance rate:
      - rate > adapt_hi : too many accepts → decrease intensity by a random
                          fraction ∈ [0, ARMHC_INTENSITY_SCALE_DOWN_MAX_RATIO]
      - rate < adapt_lo : too few accepts  → increase intensity by
                          ARMHC_INTENSITY_SCALE_UP_RATIO
      - else            : within [adapt_lo, adapt_hi] → no change

    Returns (best_mask, score_history, intensity_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)

    mask = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, k_eff, replace=False)] = True

    best_mask          = mask.copy()
    best_score         = _score_subset(X, y, mask)
    score_history:     List[float] = [best_score]
    intensity_history: List[float] = []
    mutation_intensity = float(init_mutation_intensity)
    accept_window:     List[int] = []

    for iteration in range(max_iter):
        n_flip    = max(1, int(n_features * mutation_intensity))
        candidate = mask.copy()
        true_idx  = np.where(candidate)[0]
        false_idx = np.where(~candidate)[0]
        n_flip    = min(n_flip, len(true_idx), len(false_idx))
        if n_flip == 0:
            break
        flip_out = rng.choice(true_idx,  n_flip, replace=False)
        flip_in  = rng.choice(false_idx, n_flip, replace=False)
        candidate[flip_out] = False
        candidate[flip_in]  = True

        s = _score_subset(X, y, candidate)
        if s > best_score:
            mask       = candidate
            best_score = s
            best_mask  = mask.copy()
            accept_window.append(1)
        else:
            accept_window.append(0)

        score_history.append(best_score)
        intensity_history.append(mutation_intensity)

        # Adapt mutation intensity every adapt_window iterations
        if (iteration + 1) % adapt_window == 0:
            window_slice = accept_window[-adapt_window:]
            rate = sum(window_slice) / len(window_slice)
            if rate > adapt_hi:
                # Too many accepts — decrease intensity (random scale-down)
                factor = 1.0 - rng.random() * intensity_scale_down_max
                mutation_intensity = max(
                    ARMHC_INTENSITY_MIN_RATIO, mutation_intensity * factor,
                )
            elif rate < adapt_lo:
                # Too few accepts — increase intensity (fixed scale-up)
                mutation_intensity = min(
                    ARMHC_INTENSITY_MAX_RATIO,
                    mutation_intensity * intensity_scale_up,
                )
            # Within healthy band → no change

        if progress_cb is not None and (iteration + 1) % 50 == 0:
            progress_cb(iteration + 1, max_iter, best_score)

    return best_mask, score_history, intensity_history


def _search_sa(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    t0: float,
    cooling: float,
    rs: int,
    criterion: str,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Simulated Annealing with single-bit flip proposal.

    Metropolis acceptance : accept if Δ ≥ 0; else P = exp(Δ/T)
    Glauber acceptance    : P = 1 / (1 + exp(−Δ/T))   (sigmoid, always < 1)

    Returns (best_mask, score_history, temp_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)

    mask = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, k_eff, replace=False)] = True

    best_mask  = mask.copy()
    cur_score  = _score_subset(X, y, mask)
    best_score = cur_score
    score_history: List[float] = [best_score]
    temp_history:  List[float] = [t0]
    T = t0

    for iteration in range(max_iter):
        i             = int(rng.integers(n_features))
        candidate     = mask.copy()
        candidate[i]  = not candidate[i]

        new_score = _score_subset(X, y, candidate)
        delta     = new_score - cur_score

        if criterion == "metropolis":
            if delta >= 0.0:
                accept = True
            else:
                accept = rng.random() < float(np.exp(delta / max(T, 1e-12)))
        else:  # glauber / gibbs
            with np.errstate(over="ignore"):
                accept = rng.random() < 1.0 / (1.0 + float(np.exp(-delta / max(T, 1e-12))))

        if accept:
            mask      = candidate
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_mask  = mask.copy()

        score_history.append(best_score)
        T *= cooling
        temp_history.append(T)
        if progress_cb is not None and (iteration + 1) % 50 == 0:
            progress_cb(iteration + 1, max_iter, best_score)

    return best_mask, score_history, temp_history


# ---------------------------------------------------------------------------
# 6. Greedy Forward Selection
# ---------------------------------------------------------------------------

def compute_greedy_forward(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_features: int = GREEDY_MAX_FEATURES_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Greedy forward feature selection on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_greedy_forward(X_pre, y, max_features, progress_cb)
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "greedy_forward", "Greedy Forward Selection",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# 7. Greedy Backward Elimination
# ---------------------------------------------------------------------------

def compute_greedy_backward(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    min_features: int = GREEDY_MIN_FEATURES_DEFAULT,
    preselect_k: int = GREEDY_BACKWARD_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Greedy backward feature elimination on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_greedy_backward(X_pre, y, min_features, progress_cb)
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "greedy_backward", "Greedy Backward Elimination",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# 8. ARMHC — Adaptive Random Mutation Hill Climbing
# ---------------------------------------------------------------------------

def compute_armhc(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = SEARCH_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    init_mutation_intensity: float = ARMHC_INIT_MUTATION_INTENSITY_RATIO,
    adapt_lo: float = ARMHC_ADAPT_LO_THRESHOLD,
    adapt_hi: float = ARMHC_ADAPT_HI_THRESHOLD,
    adapt_window: int = ARMHC_ADAPT_WINDOW,
    intensity_scale_up: float = ARMHC_INTENSITY_SCALE_UP_RATIO,
    intensity_scale_down_max: float = ARMHC_INTENSITY_SCALE_DOWN_MAX_RATIO,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Adaptive Random Mutation Hill Climbing on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history, intensity_history = _search_armhc(
        X_pre, y, max_iter, k_init, rs,
        init_mutation_intensity, adapt_lo, adapt_hi,
        adapt_window, intensity_scale_up, intensity_scale_down_max, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "armhc", "ARMHC — Adaptive Hill Climbing",
        target_col, n_generated, {"intensity_history": intensity_history},
    )


# ---------------------------------------------------------------------------
# 9. SA — Metropolis acceptance
# ---------------------------------------------------------------------------

def compute_sa_metropolis(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = SEARCH_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    t0: float = SA_T0_DEFAULT,
    cooling: float = SA_COOLING_RATIO_DEFAULT,
    rs: int = 42,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Simulated Annealing with Metropolis acceptance criterion."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history, temp_history = _search_sa(
        X_pre, y, max_iter, k_init, t0, cooling, rs, "metropolis", progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "sa_metropolis", "SA — Metropolis Acceptance",
        target_col, n_generated, {"temp_history": temp_history},
    )


# ---------------------------------------------------------------------------
# 10. SA — Glauber acceptance
# ---------------------------------------------------------------------------

def compute_sa_glauber(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = SEARCH_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    t0: float = SA_T0_DEFAULT,
    cooling: float = SA_COOLING_RATIO_DEFAULT,
    rs: int = 42,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Simulated Annealing with Glauber (sigmoid / Gibbs) acceptance criterion."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history, temp_history = _search_sa(
        X_pre, y, max_iter, k_init, t0, cooling, rs, "glauber", progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "sa_glauber", "SA — Glauber Acceptance",
        target_col, n_generated, {"temp_history": temp_history},
    )


# ---------------------------------------------------------------------------
# 11. Reactive Tabu Search
# ---------------------------------------------------------------------------

def _search_rts(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    init_tabu_tenure: int,
    tabu_adapt_step: int,
    n_candidates: int,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Reactive Tabu Search.

    Maintains a tabu list of recently touched features.  If the same solution
    hash is revisited, tabu tenure increases by tabu_adapt_step (reactive
    memory), discouraging cycling.  Aspiration criterion: a tabu move is
    allowed if it improves the global best.

    Each iteration evaluates n_candidates randomly sampled single-bit flips
    and picks the best admissible one.

    Returns (best_mask, score_history, tenure_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)

    mask = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, k_eff, replace=False)] = True

    best_mask  = mask.copy()
    best_score = _score_subset(X, y, mask)
    score_history:  List[float] = [best_score]
    tenure_history: List[float] = [float(init_tabu_tenure)]

    tabu_remaining: Dict[int, int] = {}   # feature_idx → steps remaining
    tabu_tenure = init_tabu_tenure
    visited: Dict[bytes, int] = {}

    for iteration in range(max_iter):
        # Decrement tabu counters
        tabu_remaining = {k: v - 1 for k, v in tabu_remaining.items() if v > 1}

        # Evaluate n_candidates random single-bit flips
        candidate_idx = rng.choice(n_features, min(n_candidates, n_features), replace=False)
        best_cand_score = -np.inf
        best_cand_feat  = -1

        for i in candidate_idx:
            candidate    = mask.copy()
            candidate[i] = not candidate[i]
            s            = _score_subset(X, y, candidate)
            # Aspiration: accept tabu move if it beats global best
            if i in tabu_remaining and s <= best_score:
                continue
            if s > best_cand_score:
                best_cand_score = s
                best_cand_feat  = int(i)

        if best_cand_feat < 0:
            # All candidates tabu and none aspire → pick least-tabu
            remaining_sorted = sorted(tabu_remaining, key=lambda k: tabu_remaining[k])
            best_cand_feat = int(remaining_sorted[0]) if remaining_sorted else int(candidate_idx[0])

        mask[best_cand_feat] = not mask[best_cand_feat]
        tabu_remaining[best_cand_feat] = tabu_tenure

        cur_score = _score_subset(X, y, mask)
        if cur_score > best_score:
            best_score = cur_score
            best_mask  = mask.copy()

        # Reactive: detect revisit by mask hash
        mhash = mask.tobytes()
        if mhash in visited:
            tabu_tenure = min(tabu_tenure + tabu_adapt_step, n_features)
            visited[mhash] += 1
        else:
            visited[mhash] = 1

        score_history.append(best_score)
        tenure_history.append(float(tabu_tenure))

        if progress_cb is not None and (iteration + 1) % 50 == 0:
            progress_cb(iteration + 1, max_iter, best_score)

    return best_mask, score_history, tenure_history


def compute_rts(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = SEARCH_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    init_tabu_tenure: int = RTS_INIT_TABU_TENURE_DEFAULT,
    tabu_adapt_step: int = RTS_TABU_ADAPT_STEP_DEFAULT,
    n_candidates: int = RTS_CANDIDATES_PER_ITER_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Reactive Tabu Search on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history, tenure_history = _search_rts(
        X_pre, y, max_iter, k_init, rs,
        init_tabu_tenure, tabu_adapt_step, n_candidates, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "rts", "Reactive Tabu Search",
        target_col, n_generated, {"tenure_history": tenure_history},
    )


# ---------------------------------------------------------------------------
# 12. Cross-Entropy Method
# ---------------------------------------------------------------------------

def _search_cem(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    n_samples: int,
    elite_frac: float,
    smoothing: float,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Cross-Entropy Method for feature subset selection.

    Maintains a probability vector p[i] (probability feature i is included).
    Each iteration:
      1. Sample n_samples binary subsets via Bernoulli(p[i]).
      2. Evaluate each subset's R².
      3. Select the top elite_frac fraction.
      4. Update p as smoothed mixture of old p and elite mean.

    Returns (best_mask, score_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)

    # Initialise probability to k_eff / n_features
    p_init = k_eff / n_features
    p      = np.full(n_features, p_init, dtype=float)

    best_mask  = np.zeros(n_features, dtype=bool)
    best_score = 0.0
    score_history: List[float] = [best_score]
    n_elite = max(1, int(n_samples * elite_frac))

    for iteration in range(max_iter):
        # Sample subsets
        samples = rng.random((n_samples, n_features)) < p
        scores  = np.array([_score_subset(X, y, samples[i]) for i in range(n_samples)])

        # Select elite
        elite_idx = np.argsort(scores)[-n_elite:]

        # Track best
        top_idx = int(elite_idx[-1])
        if scores[top_idx] > best_score:
            best_score = float(scores[top_idx])
            best_mask  = samples[top_idx].copy()

        # Update p with smoothing: p = α·p + (1−α)·elite_mean
        elite_mean = samples[elite_idx].mean(axis=0)
        p = smoothing * p + (1.0 - smoothing) * elite_mean
        p = np.clip(p, 0.01, 0.99)

        score_history.append(best_score)

        if progress_cb is not None:
            progress_cb(iteration + 1, max_iter, best_score)

    return best_mask, score_history


def compute_cem(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = CEM_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    n_samples: int = CEM_N_SAMPLES_DEFAULT,
    elite_frac: float = CEM_ELITE_FRAC_DEFAULT,
    smoothing: float = CEM_SMOOTHING_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Cross-Entropy Method on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_cem(
        X_pre, y, max_iter, k_init, rs,
        n_samples, elite_frac, smoothing, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "cem", "Cross-Entropy Method",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# 13. Parallel Tempering
# ---------------------------------------------------------------------------

def _search_pt(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    n_replicas: int,
    t_min: float,
    t_max: float,
    swap_interval: int,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Parallel Tempering (Replica Exchange Monte Carlo).

    Runs n_replicas SA chains on a log-spaced temperature ladder
    [t_min, …, t_max].  Every swap_interval iterations, adjacent chains
    propose a configuration exchange accepted via the Metropolis criterion
    for the combined system.

    Returns (best_mask, score_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)

    # Log-spaced temperature ladder (coldest first)
    n_rep = max(1, n_replicas)
    if n_rep == 1:
        temps = [t_min]
    else:
        temps = np.exp(
            np.linspace(np.log(t_min), np.log(t_max), n_rep)
        ).tolist()

    # Initialise replicas
    masks  = []
    scores = []
    for _ in range(n_rep):
        m = np.zeros(n_features, dtype=bool)
        m[rng.choice(n_features, k_eff, replace=False)] = True
        masks.append(m)
        scores.append(_score_subset(X, y, m))

    best_score = float(max(scores))
    best_mask  = masks[int(np.argmax(scores))].copy()
    score_history: List[float] = [best_score]

    for iteration in range(max_iter):
        # SA step for each replica
        for r in range(n_rep):
            i            = int(rng.integers(n_features))
            candidate    = masks[r].copy()
            candidate[i] = not candidate[i]
            new_s        = _score_subset(X, y, candidate)
            delta        = new_s - scores[r]
            T            = temps[r]
            if delta >= 0 or rng.random() < float(np.exp(delta / max(T, 1e-12))):
                masks[r]  = candidate
                scores[r] = new_s
                if new_s > best_score:
                    best_score = new_s
                    best_mask  = masks[r].copy()

        # Replica exchange: attempt adjacent swaps
        if (iteration + 1) % swap_interval == 0:
            for r in range(n_rep - 1):
                # Swap acceptance: exp((β_r − β_{r+1}) * (E_{r+1} − E_r))
                # where β = 1/T and E = score (we maximise so E is reward)
                beta_r  = 1.0 / max(temps[r],   1e-12)
                beta_r1 = 1.0 / max(temps[r+1], 1e-12)
                delta_accept = (beta_r - beta_r1) * (scores[r+1] - scores[r])
                if rng.random() < min(1.0, float(np.exp(delta_accept))):
                    masks[r],  masks[r+1]  = masks[r+1],  masks[r]
                    scores[r], scores[r+1] = scores[r+1], scores[r]

        score_history.append(best_score)

        if progress_cb is not None and (iteration + 1) % 50 == 0:
            progress_cb(iteration + 1, max_iter, best_score)

    return best_mask, score_history


def compute_pt(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = PT_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    n_replicas: int = PT_N_REPLICAS_DEFAULT,
    t_min: float = PT_T_MIN_DEFAULT,
    t_max: float = PT_T_MAX_DEFAULT,
    swap_interval: int = PT_SWAP_INTERVAL_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Parallel Tempering (Replica Exchange Monte Carlo) on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_pt(
        X_pre, y, max_iter, k_init, rs,
        n_replicas, t_min, t_max, swap_interval, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "pt", "Parallel Tempering",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# 14. Quantum Annealing (Trotterized PIMC)
# ---------------------------------------------------------------------------

def _search_qa(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    n_trotter: int,
    gamma_init: float,
    gamma_final: float,
    t_fixed: float,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Quantum Annealing via Path Integral Monte Carlo (Trotterized PIMC).

    Approximates quantum tunneling using P Trotter slices.  The transverse
    field Γ decreases geometrically from gamma_init → gamma_final.

    Inter-slice quantum coupling:
        J = -(T/2) · ln(tanh(Γ/(P·T)))

    Each iteration selects a random slice and feature; the total energy
    change includes the classical score improvement plus quantum coupling
    to the two adjacent slices.  Metropolis acceptance at temperature T.

    Returns (best_mask, score_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    P          = max(1, n_trotter)
    k_eff      = min(k_init, n_features)
    T          = t_fixed

    # Initialise Trotter slices
    slices       = []
    slice_scores = []
    for _ in range(P):
        m = np.zeros(n_features, dtype=bool)
        m[rng.choice(n_features, k_eff, replace=False)] = True
        slices.append(m)
        slice_scores.append(_score_subset(X, y, m))

    best_score = float(max(slice_scores))
    best_mask  = slices[int(np.argmax(slice_scores))].copy()
    score_history: List[float] = [best_score]

    for iteration in range(max_iter):
        frac  = iteration / max(max_iter - 1, 1)
        gamma = gamma_init * (gamma_final / max(gamma_init, 1e-12)) ** frac

        # Quantum coupling strength (Trotter approximation)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            _arg = gamma / max(P * T, 1e-12)
            _th  = float(np.tanh(_arg))
            J    = -(T / 2.0) * float(np.log(max(_th, 1e-12)))
        J = J if np.isfinite(J) else 0.0

        # Select random slice and feature
        r_slice = int(rng.integers(P))
        i_feat  = int(rng.integers(n_features))

        candidate    = slices[r_slice].copy()
        candidate[i_feat] = not candidate[i_feat]

        # Classical energy change
        new_s           = _score_subset(X, y, candidate)
        delta_classical = new_s - slice_scores[r_slice]

        # Quantum coupling to neighbouring slices (periodic boundary)
        prev_s = slices[(r_slice - 1) % P]
        next_s = slices[(r_slice + 1) % P]
        cur_agree = int(np.sum(slices[r_slice] == prev_s)) + int(np.sum(slices[r_slice] == next_s))
        new_agree = int(np.sum(candidate     == prev_s)) + int(np.sum(candidate     == next_s))
        delta_q   = J * (new_agree - cur_agree) / max(n_features, 1)

        delta_total = delta_classical + delta_q
        if delta_total >= 0 or rng.random() < float(np.exp(delta_total / max(T, 1e-12))):
            slices[r_slice]       = candidate
            slice_scores[r_slice] = new_s
            if new_s > best_score:
                best_score = new_s
                best_mask  = candidate.copy()

        score_history.append(best_score)

        if progress_cb is not None and (iteration + 1) % 50 == 0:
            progress_cb(iteration + 1, max_iter, best_score)

    return best_mask, score_history


def compute_qa(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = SEARCH_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    n_trotter: int = QA_N_TROTTER_DEFAULT,
    gamma_init: float = QA_GAMMA_INIT_DEFAULT,
    gamma_final: float = QA_GAMMA_FINAL_DEFAULT,
    t_fixed: float = QA_T_FIXED_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Quantum Annealing (Trotterized PIMC) on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_qa(
        X_pre, y, max_iter, k_init, rs,
        n_trotter, gamma_init, gamma_final, t_fixed, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "qa", "Quantum Annealing (PIMC)",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# 15. Particle Swarm Optimization (Binary PSO)
# ---------------------------------------------------------------------------

def _search_pso(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    n_particles: int,
    w: float,
    c1: float,
    c2: float,
    v_max: float,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Binary Particle Swarm Optimization for feature subset selection.

    Each particle's position is a continuous vector in [0, 1]^d; the binary
    feature mask is obtained via a sigmoid transfer function:

        mask_i = 1  if  sigmoid(pos_i) > PSO_SIGMOID_THRESHOLD  else  0

    Velocity update (Kennedy & Eberhart, 1997):
        v(t+1) = w·v(t) + c1·r1·(pbest − pos) + c2·r2·(gbest − pos)

    v is clamped to [−v_max, +v_max].  Fitness = R²(selected subset).

    Returns (best_mask, score_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)
    n_part     = max(2, n_particles)

    def _sigmoid(v: np.ndarray) -> np.ndarray:
        v_clip = np.clip(v, -500, 500)
        return 1.0 / (1.0 + np.exp(-v_clip))

    def _to_mask(pos: np.ndarray) -> np.ndarray:
        return _sigmoid(pos) > PSO_SIGMOID_THRESHOLD

    # Initialise positions so that ~k_eff features are active per particle
    # logit(k_eff/n_features) gives the centre point
    p_init = k_eff / n_features
    logit_centre = float(np.log(p_init / (1 - p_init + 1e-12)))
    positions = rng.normal(logit_centre, 1.0, (n_part, n_features))
    velocities = rng.uniform(-1.0, 1.0, (n_part, n_features))

    # Evaluate initial fitness
    masks   = np.array([_to_mask(positions[i]) for i in range(n_part)])
    fitness = np.array([_score_subset(X, y, masks[i]) for i in range(n_part)])

    # Personal bests
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    # Global best
    gbest_idx  = int(np.argmax(pbest_fit))
    gbest_pos  = pbest_pos[gbest_idx].copy()
    gbest_fit  = float(pbest_fit[gbest_idx])
    gbest_mask = masks[gbest_idx].copy()

    score_history: List[float] = [gbest_fit]

    for iteration in range(max_iter):
        r1 = rng.random((n_part, n_features))
        r2 = rng.random((n_part, n_features))

        # Velocity update
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        # Position update
        positions = positions + velocities

        # Evaluate
        for i in range(n_part):
            m = _to_mask(positions[i])
            if not m.any():
                # Ensure at least one feature selected
                m[int(rng.integers(n_features))] = True
            s = _score_subset(X, y, m)

            if s > pbest_fit[i]:
                pbest_pos[i] = positions[i].copy()
                pbest_fit[i] = s

            if s > gbest_fit:
                gbest_pos  = positions[i].copy()
                gbest_fit  = s
                gbest_mask = m.copy()

        score_history.append(gbest_fit)

        if progress_cb is not None and (iteration + 1) % 50 == 0:
            progress_cb(iteration + 1, max_iter, gbest_fit)

    return gbest_mask, score_history


def compute_pso(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = SEARCH_MAX_ITER_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    n_particles: int = PSO_N_PARTICLES_DEFAULT,
    w: float = PSO_W_DEFAULT,
    c1: float = PSO_C1_DEFAULT,
    c2: float = PSO_C2_DEFAULT,
    v_max: float = PSO_V_MAX_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Binary PSO on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_pso(
        X_pre, y, max_iter, k_init, rs,
        n_particles, w, c1, c2, v_max, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "pso", "PSO — Particle Swarm",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# 16. Genetic Programming (variable-length chromosome)
# ---------------------------------------------------------------------------

def _search_gp(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    k_init: int,
    rs: int,
    pop_size: int,
    crossover_prob: float,
    mutation_prob: float,
    tournament_size: int,
    elitism: int,
    parsimony_coeff: float,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Genetic Programming for feature subset selection.

    Unlike GA with fixed-length binary chromosomes, GP uses variable-length
    chromosomes where the genome encodes both a feature mask and a selection
    weight.  Crossover exchanges feature subsets at a random cut point.
    Mutation randomly adds or removes features.

    Fitness = R²(subset) − λ · |subset| / n_features
    where λ = parsimony_coeff penalises complex solutions.

    Selection: tournament selection with configurable tournament size.
    Elitism: top-k individuals survive unchanged into the next generation.

    Returns (best_mask, score_history).
    """
    rng        = np.random.default_rng(rs)
    n_features = X.shape[1]
    k_eff      = min(k_init, n_features)
    n_pop      = max(4, pop_size)
    n_elite    = min(elitism, n_pop // 2)
    t_size     = min(tournament_size, n_pop)

    def _fitness(mask: np.ndarray) -> float:
        r2 = _score_subset(X, y, mask)
        complexity = float(mask.sum()) / n_features
        return r2 - parsimony_coeff * complexity

    # Initialise population — variable-length chromosomes
    population: List[np.ndarray] = []
    for _ in range(n_pop):
        m = np.zeros(n_features, dtype=bool)
        n_sel = max(1, int(rng.poisson(k_eff)))
        n_sel = min(n_sel, n_features)
        m[rng.choice(n_features, n_sel, replace=False)] = True
        population.append(m)

    fit = np.array([_fitness(p) for p in population])
    best_idx   = int(np.argmax(fit))
    best_mask  = population[best_idx].copy()
    best_score = float(fit[best_idx])
    score_history: List[float] = [best_score]

    for generation in range(max_iter):
        # Elitism — preserve top individuals
        elite_idx = np.argsort(fit)[-n_elite:]
        new_pop: List[np.ndarray] = [population[i].copy() for i in elite_idx]
        new_fit: List[float] = [float(fit[i]) for i in elite_idx]

        while len(new_pop) < n_pop:
            # Tournament selection (two parents)
            def _tournament() -> np.ndarray:
                candidates = rng.choice(n_pop, t_size, replace=False)
                winner = candidates[int(np.argmax(fit[candidates]))]
                return population[winner].copy()

            p1 = _tournament()
            p2 = _tournament()

            # Variable-length crossover: one-point crossover on the feature index
            if rng.random() < crossover_prob:
                cut = int(rng.integers(1, n_features))
                child = np.concatenate([p1[:cut], p2[cut:]])
            else:
                child = p1.copy()

            # Mutation: flip each gene with mutation_prob
            mut_mask = rng.random(n_features) < mutation_prob
            child = np.where(mut_mask, ~child, child)

            # Ensure at least one feature
            if not child.any():
                child[int(rng.integers(n_features))] = True

            f = _fitness(child)
            new_pop.append(child)
            new_fit.append(f)

        population = new_pop[:n_pop]
        fit = np.array(new_fit[:n_pop])

        gen_best_idx = int(np.argmax(fit))
        if fit[gen_best_idx] > best_score:
            best_score = float(fit[gen_best_idx])
            best_mask  = population[gen_best_idx].copy()

        score_history.append(best_score)

        if progress_cb is not None:
            progress_cb(generation + 1, max_iter, best_score)

    return best_mask, score_history


def compute_gp(
    df: pd.DataFrame,
    selected_columns: List[str],
    target_col: str,
    selected_ops: Dict[str, List[str]],
    max_iter: int = GP_MAX_GENS_DEFAULT,
    k_init: int = SEARCH_K_INIT_DEFAULT,
    rs: int = 42,
    pop_size: int = GP_POP_SIZE_DEFAULT,
    crossover_prob: float = GP_CROSSOVER_PROB_DEFAULT,
    mutation_prob: float = GP_MUTATION_PROB_DEFAULT,
    tournament_size: int = GP_TOURNAMENT_SIZE_DEFAULT,
    elitism: int = GP_ELITISM_COUNT_DEFAULT,
    parsimony_coeff: float = GP_PARSIMONY_COEFF_DEFAULT,
    preselect_k: int = SEARCH_PRESELECT_DEFAULT,
    combo_cap: int = FEATURE_COMBO_CAP_DEFAULT,
    total_cap: int = FEATURE_TOTAL_CAP_DEFAULT,
    progress_cb=None,
) -> DescriptorResult:
    """Genetic Programming on the symbolic feature space."""
    X_df_pre, X_pre, y, n_generated = _prepare_search(
        df, selected_columns, target_col, selected_ops,
        preselect_k, combo_cap, total_cap,
    )
    best_mask, score_history = _search_gp(
        X_pre, y, max_iter, k_init, rs,
        pop_size, crossover_prob, mutation_prob,
        tournament_size, elitism, parsimony_coeff, progress_cb,
    )
    return _finalise_search(
        X_df_pre, X_pre, y, best_mask, score_history,
        "gp", "GP — Genetic Programming",
        target_col, n_generated, {},
    )


# ---------------------------------------------------------------------------
# Apply scoped warning suppression to every public compute_* entry point.
# Must run *before* _DISPATCH is built so the dispatch stores wrapped refs.
# ---------------------------------------------------------------------------
for _fn_name in [n for n in list(globals()) if n.startswith("compute_")]:
    globals()[_fn_name] = _quiet_user_warnings(globals()[_fn_name])
del _fn_name


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: Dict[str, Tuple[str, str, object]] = {
    # key                  title                                   category             callable
    "linear_screening": ("Linear Screening (CV R² per formula)",   "BASIC",           compute_linear_screening),
    "lasso":            ("Lasso CV Descriptor Selection",        "REGULARISATION",    compute_lasso),
    "ridge":            ("Ridge CV Descriptor Selection",        "REGULARISATION",    compute_ridge),
    "elasticnet":       ("Elastic Net CV Descriptor Selection",  "REGULARISATION",    compute_elasticnet),
    "greedy_forward":   ("Greedy Forward Selection",             "COMBINATORIAL",     compute_greedy_forward),
    "greedy_backward":  ("Greedy Backward Elimination",          "COMBINATORIAL",     compute_greedy_backward),
    "armhc":            ("ARMHC — Adaptive Hill Climbing",       "COMBINATORIAL",     compute_armhc),
    "sa_metropolis":    ("SA — Metropolis Acceptance",           "COMBINATORIAL",     compute_sa_metropolis),
    "sa_glauber":       ("SA — Glauber Acceptance",              "COMBINATORIAL",     compute_sa_glauber),
    "rts":              ("Reactive Tabu Search",                  "COMBINATORIAL",     compute_rts),
    "cem":              ("Cross-Entropy Method",                  "COMBINATORIAL",     compute_cem),
    "pt":               ("Parallel Tempering",                    "COMBINATORIAL",     compute_pt),
    "qa":               ("Quantum Annealing (PIMC)",              "COMBINATORIAL",     compute_qa),
    "pso":              ("PSO — Particle Swarm",                  "COMBINATORIAL",     compute_pso),
    "gp":               ("GP — Genetic Programming",              "COMBINATORIAL",     compute_gp),
}
