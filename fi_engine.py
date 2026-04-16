"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for 32 feature-ranking and importance
    methods, returning typed, immutable FeatureRankingResult records.

Scientific Context:
    Operates on (feature_matrix X, target_vector y) pairs derived from a
    DataAsset.  Covers model-intrinsic, regularisation-based, statistical,
    model-agnostic, optimisation-based, and probabilistic methods.
    No UI calls, no side effects, no I/O.

Invariants:
    - Every function returns a FeatureRankingResult with a pd.Series
      indexed by feature name.  Extra visualisation data is placed in
      result.extra.
    - Scores are not globally normalised; callers normalise for display.
    - Classification vs regression is auto-detected from target cardinality.
    - All estimators receive random_state=42 unless overridden.

Assumptions:
    - X contains only float64 columns with no NaN (callers pre-filter).
    - y contains no NaN.
    - For classification methods, y contains integer or string class labels.

Failure Modes:
    - Target column cardinality = 1 (constant): raises ValueError with
      diagnostic message.
    - n_features < 2 for methods requiring multiple features: raises
      ValueError.
    - Missing optional dependency (torch for deep_attr): raises ImportError
      with install hint.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import functools
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import differential_entropy, kendalltau, mannwhitneyu, kruskal, pearsonr, rankdata
from scipy.special import comb as _comb

from sklearn.decomposition import SparsePCA, PCA
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE, chi2, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
)
from sklearn.inspection import permutation_importance as _pi
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warning suppression — scoped, not global
# ---------------------------------------------------------------------------
# sklearn emits UserWarning during .fit() for many estimators (convergence,
# feature-name mismatches, etc.).  We silence them *only* inside compute_*
# bodies via @_quiet_user_warnings, never at module level — a module-level
# filterwarnings() pollutes every other module running in the same process.

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
# Result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureRankingResult:
    """
    Typed, immutable output of a single feature-importance computation.

    Fields
    ------
    method_key   : machine identifier matching _DISPATCH key
    method_title : human-readable method name
    scores       : pd.Series indexed by feature name, values = importance
    target_col   : name of the target column used
    extra        : dict of additional data for non-standard visualisations
                   (e.g. shap_matrix, rfe_grid_scores, interaction_matrix)
    is_valid     : False if computation degraded (singular matrix, etc.)

    Variant-indexed attribution (NPJ paper, Task #28)
    -------------------------------------------------
    The four fields below bind every importance vector to the exact
    representation state under which it was computed, so that stability
    decompositions and cross-variant comparisons can be cited without
    ambiguity.  All four default to empty/None so existing call sites
    keep working unchanged.

    variant_id       : int or None — SEAL variant index (0..K-1), or None
                       if computed on the base DataFrame.
    dataset_sha256   : str — SHA-256 hex of the exact DataFrame analysed.
                       Empty string means "not recorded by this call site."
    contract_context : dict — snapshot of active ObservableContract records
                       at computation time ({col: {lo, hi, units}}).
    model_state_hash : str or None — SHA-256 hex of the serialised model
                       state (None if not recorded or not applicable).
    """
    method_key:       str
    method_title:     str
    scores:           pd.Series
    target_col:       str
    extra:            Dict[str, Any] = field(default_factory=dict)
    is_valid:         bool = True
    variant_id:       Optional[int] = None
    dataset_sha256:   str            = ""
    contract_context: Dict[str, Any] = field(default_factory=dict)
    model_state_hash: Optional[str]  = None


# ---------------------------------------------------------------------------
# Named defaults (DNV style: no bare numeric literals)
# ---------------------------------------------------------------------------

#: Default random state for reproducibility [dimensionless seed].
DEFAULT_RANDOM_STATE: Final[int] = 42

#: Default n_estimators for tree ensembles [count].
DEFAULT_N_ESTIMATORS: Final[int] = 100

#: Lightweight ensemble size for auxiliary models (SHAP, LIME, etc.) [count].
LIGHT_N_ESTIMATORS: Final[int] = 50

#: Minimal ensemble size for wrapper methods (GA, PSO, etc.) [count].
MINIMAL_N_ESTIMATORS: Final[int] = 30

#: Default max_iter for SVM / MLP solvers [iteration count].
DEFAULT_MAX_ITER_SVM: Final[int] = 2000

#: Default max_iter for lasso / ridge / elastic net solvers [iteration count].
DEFAULT_MAX_ITER_REGULARISED: Final[int] = 5000

#: Default max_iter for Sparse PCA solver [iteration count].
DEFAULT_MAX_ITER_SPCA: Final[int] = 500

#: Default max_iter for Bayesian / calibration solvers [iteration count].
DEFAULT_MAX_ITER_BAYES: Final[int] = 300

#: Default max_iter for logistic regression [iteration count].
DEFAULT_MAX_ITER_LOGISTIC: Final[int] = 1000

#: Default boosting max_depth [tree depth].
DEFAULT_GB_MAX_DEPTH: Final[int] = 3

#: Default boosting learning rate [dimensionless step size].
DEFAULT_GB_LEARNING_RATE: Final[float] = 0.1

#: Default number of permutation repeats [count].
DEFAULT_N_REPEATS: Final[int] = 10

#: Default number of MI / k-NN neighbours [count].
DEFAULT_N_NEIGHBORS: Final[int] = 5

#: Default Sparse PCA / mRMR component count [count].
DEFAULT_N_COMPONENTS: Final[int] = 5

#: Default SVM regularisation strength [dimensionless].
DEFAULT_SVM_C: Final[float] = 1.0

#: Default L1 penalty for lasso / elastic net [dimensionless].
DEFAULT_ALPHA_LASSO: Final[float] = 0.01

#: Default L2 penalty for ridge [dimensionless].
DEFAULT_ALPHA_RIDGE: Final[float] = 1.0

#: Default L1/L2 ratio for elastic net [dimensionless].
DEFAULT_L1_RATIO: Final[float] = 0.5

#: Default cross-validation fold count [count].
DEFAULT_CV_FOLDS: Final[int] = 5

#: Minimal cross-validation fold count for wrapper methods [count].
MINIMAL_CV_FOLDS: Final[int] = 3

#: Default mRMR feature selection count [count].
DEFAULT_MRMR_N_FEATURES: Final[int] = 10

#: Default Boruta trial count [count].
DEFAULT_BORUTA_TRIALS: Final[int] = 20

#: Default quantile discretisation bins for chi² [count].
DEFAULT_CHI2_BINS: Final[int] = 5

#: Default ReliefF neighbour count [count].
DEFAULT_RELIEFF_K: Final[int] = 10

#: Minimum samples per group for Mann-Whitney U / Kruskal-Wallis H tests.
#: Below this the asymptotic test approximations are unreliable and the
#: test is rejected with a clear ValueError rather than silently returning 0.
MW_KW_MIN_GROUP_SIZE: Final[int] = 2

#: Maximum number of unique categorical classes allowed as an MW/KW target.
#: Above this the column is almost certainly an ID column, not a class label,
#: and the test is rejected with a clear ValueError.
MW_KW_MAX_CATEGORICAL_CLASSES: Final[int] = 50

#: Default LIME samples [count].
DEFAULT_LIME_N_SAMPLES: Final[int] = 500

#: Default LIME display features [count].
DEFAULT_LIME_N_DISPLAY: Final[int] = 20

#: Default LIME kernel width [dimensionless].
DEFAULT_LIME_SIGMA: Final[float] = 1.0

#: Default SHAP background samples [count].
DEFAULT_SHAP_N_BG: Final[int] = 50

#: Default SHAP coalition samples [count].
DEFAULT_SHAP_N_COAL: Final[int] = 150

#: Default GA population size [count].
DEFAULT_GA_POP: Final[int] = 30

#: Default GA generations [count].
DEFAULT_GA_GENS: Final[int] = 20

#: Default GA mutation rate [dimensionless probability].
DEFAULT_GA_MUTATION_RATE: Final[float] = 0.05

#: Default PSO swarm size [count].
DEFAULT_PSO_SWARM: Final[int] = 20

#: Default PSO iterations [count].
DEFAULT_PSO_ITERS: Final[int] = 30

#: Default PSO inertia weight [dimensionless].
DEFAULT_PSO_W: Final[float] = 0.7

#: Default PSO cognitive coefficient [dimensionless].
DEFAULT_PSO_C1: Final[float] = 1.5

#: Default PSO social coefficient [dimensionless].
DEFAULT_PSO_C2: Final[float] = 1.5

#: Default autoencoder hidden size [neuron count].
DEFAULT_AE_HIDDEN: Final[int] = 32

#: Default deep-learning attribution epochs [count].
DEFAULT_DL_EPOCHS: Final[int] = 100

#: Default deep-learning attribution learning rate [dimensionless].
DEFAULT_DL_LR: Final[float] = 1e-3

#: Alpha threshold for conditional-independence tests [dimensionless p-value].
DEFAULT_CI_ALPHA: Final[float] = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_classification(y: np.ndarray) -> bool:
    """True if y looks like class labels (≤ CLASSIFICATION_MAX_UNIQUE unique values or non-numeric)."""
    from ml_engine import CLASSIFICATION_MAX_UNIQUE
    if y.dtype.kind in ("U", "S", "O"):
        return True
    return int(np.unique(y).size) <= CLASSIFICATION_MAX_UNIQUE


def _prepare(
    df: pd.DataFrame,
    target_col: str,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Index, bool]:
    """
    Drop NaN rows, split X / y, optionally standardise X.

    Returns (X, y, feature_names, is_clf).
    """
    sub = df.dropna()
    y = sub[target_col].values
    X_df = sub.drop(columns=[target_col]).select_dtypes(include="number")
    feature_names = X_df.columns
    X = X_df.values.astype(float)
    if standardize:
        X = StandardScaler().fit_transform(X)
    is_clf = _is_classification(y)
    if is_clf:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.astype(float)
    if len(np.unique(y)) < 2:
        raise ValueError(f"Target column '{target_col}' has < 2 unique values.")
    return X, y, feature_names, is_clf


def _make_estimator(is_clf: bool, n_estimators: int = DEFAULT_N_ESTIMATORS,
                    rs: int = DEFAULT_RANDOM_STATE):
    if is_clf:
        return RandomForestClassifier(n_estimators=n_estimators, random_state=rs)
    return RandomForestRegressor(n_estimators=n_estimators, random_state=rs)


def _result(key: str, scores: pd.Series, target_col: str,
            extra: dict = None, is_valid: bool = True) -> FeatureRankingResult:
    title = _DISPATCH.get(key, (key,))[0]
    return FeatureRankingResult(
        method_key=key, method_title=title,
        scores=scores, target_col=target_col,
        extra=extra or {}, is_valid=is_valid,
    )


def bind_variant_attribution(
    result:           FeatureRankingResult,
    *,
    variant_id:       Optional[int]      = None,
    dataset_sha256:   str                = "",
    contract_context: Optional[Dict[str, Any]] = None,
    model_state_hash: Optional[str]      = None,
) -> FeatureRankingResult:
    """Return a copy of ``result`` with variant-attribution fields set.

    Frozen dataclass semantics: we build a new record instead of mutating.
    Intended for use at the boundary between fi_engine and stability/SEAL
    orchestration code that knows *which* variant and *which* data hash
    a given importance vector came from.
    """
    return FeatureRankingResult(
        method_key       = result.method_key,
        method_title     = result.method_title,
        scores           = result.scores,
        target_col       = result.target_col,
        extra            = dict(result.extra),
        is_valid         = result.is_valid,
        variant_id       = variant_id,
        dataset_sha256   = dataset_sha256,
        contract_context = dict(contract_context) if contract_context else {},
        model_state_hash = model_state_hash,
    )


# ---------------------------------------------------------------------------
# 1. Decision Tree (Gini)
# ---------------------------------------------------------------------------

def compute_decision_tree(df, target_col, max_depth=None, criterion="squared_error",
                          rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    est = (DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
           if is_clf else
           DecisionTreeRegressor(max_depth=max_depth, criterion=criterion, random_state=rs))
    est.fit(X, y)
    return _result("decision_tree", pd.Series(est.feature_importances_, index=fn), target_col)


# ---------------------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------------------

def compute_random_forest(df, target_col, n_estimators=DEFAULT_N_ESTIMATORS,
                          max_depth=None, rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
    if max_depth:
        est.set_params(max_depth=max_depth)
    est.fit(X, y)
    return _result("random_forest", pd.Series(est.feature_importances_, index=fn), target_col)


def compute_xgboost(df, target_col, n_estimators=DEFAULT_N_ESTIMATORS,
                    max_depth=6, learning_rate=0.05,
                    rs=DEFAULT_RANDOM_STATE):
    """Gain-based feature importance from an XGBoost tree ensemble.

    Raises ImportError with install hint if xgboost is absent; the
    stability ensemble catches this and falls back to (LASSO, RF).
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "xgboost not installed; `pip install xgboost` to enable "
            "gain-based XGBoost importance."
        ) from exc
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        est = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=rs,
            verbosity=0, tree_method="hist", n_jobs=-1,
        )
    else:
        est = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=rs,
            verbosity=0, tree_method="hist", n_jobs=-1,
        )
    est.fit(X, y)
    return _result("xgboost", pd.Series(est.feature_importances_, index=fn), target_col)


# ---------------------------------------------------------------------------
# 3. Linear Regression Coefficients
# ---------------------------------------------------------------------------

def compute_linear_regression(df, target_col):
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=DEFAULT_MAX_ITER_LOGISTIC,
                                  random_state=DEFAULT_RANDOM_STATE)
        est.fit(X, y)
        coef = np.abs(est.coef_).mean(axis=0)
    else:
        est = LinearRegression()
        est.fit(X, y)
        coef = est.coef_
    return _result("linear_regression", pd.Series(coef, index=fn), target_col)


# ---------------------------------------------------------------------------
# 4. SVM Coefficients (Linear Kernel)
# ---------------------------------------------------------------------------

def compute_svm(df, target_col, C=DEFAULT_SVM_C, max_iter=DEFAULT_MAX_ITER_SVM):
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        est = LinearSVC(C=C, max_iter=max_iter, random_state=DEFAULT_RANDOM_STATE)
        est.fit(X, y)
        coef = np.abs(est.coef_).mean(axis=0)
    else:
        est = LinearSVR(C=C, max_iter=max_iter, random_state=DEFAULT_RANDOM_STATE)
        est.fit(X, y)
        coef = est.coef_
    return _result("svm", pd.Series(coef, index=fn), target_col)


# ---------------------------------------------------------------------------
# 5. Feature Maps (Feature × Target correlation heatmap)
# ---------------------------------------------------------------------------

def compute_feature_maps(df, target_col, method="pearson"):
    sub = df.dropna()
    num = sub.select_dtypes(include="number")
    corr = num.corr(method=method)
    target_corr = corr[target_col].drop(target_col).abs()
    return _result("feature_maps", target_corr, target_col,
                   extra={"full_corr": corr, "target_col": target_col})


# ---------------------------------------------------------------------------
# 6–8. Regularisation: Lasso / Ridge / ElasticNet
# ---------------------------------------------------------------------------

def compute_lasso(df, target_col, alpha=DEFAULT_ALPHA_LASSO,
                  max_iter=DEFAULT_MAX_ITER_REGULARISED):
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(penalty="l1", C=1/alpha, solver="saga",
                                  max_iter=max_iter, random_state=DEFAULT_RANDOM_STATE)
        est.fit(X, y)
        coef = np.abs(est.coef_).mean(axis=0)
    else:
        est = Lasso(alpha=alpha, max_iter=max_iter)
        est.fit(X, y)
        coef = est.coef_
    return _result("lasso", pd.Series(coef, index=fn), target_col)


def compute_ridge(df, target_col, alpha=DEFAULT_ALPHA_RIDGE,
                  max_iter=DEFAULT_MAX_ITER_REGULARISED):
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(penalty="l2", C=1/alpha,
                                  max_iter=max_iter, random_state=DEFAULT_RANDOM_STATE)
        est.fit(X, y)
        coef = np.abs(est.coef_).mean(axis=0)
    else:
        est = Ridge(alpha=alpha)
        est.fit(X, y)
        coef = est.coef_
    return _result("ridge", pd.Series(coef, index=fn), target_col)


def compute_elasticnet(df, target_col, alpha=DEFAULT_ALPHA_LASSO,
                       l1_ratio=DEFAULT_L1_RATIO,
                       max_iter=DEFAULT_MAX_ITER_REGULARISED):
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(penalty="elasticnet", C=1/alpha,
                                  l1_ratio=l1_ratio, solver="saga",
                                  max_iter=max_iter, random_state=DEFAULT_RANDOM_STATE)
        est.fit(X, y)
        coef = np.abs(est.coef_).mean(axis=0)
    else:
        est = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        est.fit(X, y)
        coef = est.coef_
    return _result("elasticnet", pd.Series(coef, index=fn), target_col)


# ---------------------------------------------------------------------------
# 9. Sparse PCA Importances
# ---------------------------------------------------------------------------

def compute_sparse_pca(df, target_col, n_components=DEFAULT_N_COMPONENTS,
                       alpha=DEFAULT_ALPHA_RIDGE, rs=DEFAULT_RANDOM_STATE):
    X, y, fn, _ = _prepare(df, target_col, standardize=False)
    X = StandardScaler().fit_transform(X)
    n_comp = min(n_components, X.shape[1])
    spca = SparsePCA(n_components=n_comp, alpha=alpha, random_state=rs,
                     max_iter=DEFAULT_MAX_ITER_SPCA)
    spca.fit(X)
    # Mean absolute loading per feature across components
    importances = np.abs(spca.components_).mean(axis=0)
    return _result("sparse_pca", pd.Series(importances, index=fn), target_col)


# ---------------------------------------------------------------------------
# 10. Gradient Boosting
# ---------------------------------------------------------------------------

def compute_gradient_boosting(df, target_col, n_estimators=DEFAULT_N_ESTIMATORS,
                               max_depth=DEFAULT_GB_MAX_DEPTH,
                               learning_rate=DEFAULT_GB_LEARNING_RATE,
                               rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        est = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                         learning_rate=learning_rate, random_state=rs)
    else:
        est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                        learning_rate=learning_rate, random_state=rs)
    est.fit(X, y)
    return _result("gradient_boosting", pd.Series(est.feature_importances_, index=fn), target_col)


# ---------------------------------------------------------------------------
# 11. RFE — Recursive Feature Elimination
# ---------------------------------------------------------------------------

def compute_rfe(df, target_col, n_features_to_select=None, step=1,
                n_estimators=LIGHT_N_ESTIMATORS, cv=DEFAULT_CV_FOLDS,
                rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
    n_sel = n_features_to_select or max(1, X.shape[1] // 2)
    from sklearn.feature_selection import RFECV
    rfe = RFECV(est, min_features_to_select=1, step=step, cv=min(cv, len(y)//2))
    rfe.fit(X, y)
    # Importance = 1/rank (higher rank = eliminated earlier = less important)
    importances = 1.0 / rfe.ranking_.astype(float)
    extra = {
        "ranking": rfe.ranking_,
        "grid_scores": rfe.cv_results_["mean_test_score"],
        "n_features_selected": rfe.n_features_,
        "support": rfe.support_,
        "feature_names": list(fn),
    }
    return _result("rfe", pd.Series(importances, index=fn), target_col, extra=extra)


# ---------------------------------------------------------------------------
# 12. Boruta (shadow-feature method, no external package)
# ---------------------------------------------------------------------------

def compute_boruta(df, target_col, n_trials=DEFAULT_BORUTA_TRIALS,
                   n_estimators=LIGHT_N_ESTIMATORS, rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    n_features = X.shape[1]
    confirmed = np.zeros(n_features)

    rng = np.random.RandomState(rs)
    for trial in range(n_trials):
        X_shadow = np.apply_along_axis(rng.permutation, 0, X)
        X_aug = np.hstack([X, X_shadow])
        est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs + trial)
        est.fit(X_aug, y)
        imp = est.feature_importances_
        real_imp   = imp[:n_features]
        shadow_max = imp[n_features:].max()
        confirmed += (real_imp > shadow_max).astype(float)

    scores = confirmed / n_trials   # fraction of trials beating max shadow
    return _result("boruta", pd.Series(scores, index=fn), target_col,
                   extra={"threshold_label": "fraction of trials beating max shadow"})


# ---------------------------------------------------------------------------
# 13. ANOVA F-value
# ---------------------------------------------------------------------------

def compute_anova_f(df, target_col):
    X, y, fn, is_clf = _prepare(df, target_col)
    fn_func = f_classif if is_clf else f_regression
    F, pvals = fn_func(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    return _result("anova_f", pd.Series(F, index=fn), target_col,
                   extra={"pvalues": pd.Series(pvals, index=fn)})


# ---------------------------------------------------------------------------
# 14. Chi-Square Scores
# ---------------------------------------------------------------------------

def compute_chi_square(df, target_col):
    X, y, fn, is_clf = _prepare(df, target_col, standardize=False)
    # Chi² requires non-negative features
    X_nn = MinMaxScaler().fit_transform(np.abs(X))
    if not is_clf:
        # Discretise target for chi²
        y_disc = pd.qcut(y, q=DEFAULT_CHI2_BINS, labels=False, duplicates="drop")
    else:
        y_disc = y
    chi2_scores, pvals = chi2(X_nn, y_disc)
    chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
    return _result("chi_square", pd.Series(chi2_scores, index=fn), target_col,
                   extra={"pvalues": pd.Series(pvals, index=fn)})


# ---------------------------------------------------------------------------
# 15. Kendall Tau Correlation
# ---------------------------------------------------------------------------

def compute_kendall_tau(df, target_col):
    sub = df.dropna()
    y = sub[target_col].values
    X_df = sub.drop(columns=[target_col]).select_dtypes(include="number")
    scores = {}
    for col in X_df.columns:
        tau, _ = kendalltau(X_df[col].values, y)
        scores[col] = float(np.nan_to_num(tau))
    return _result("kendall_tau", pd.Series(scores), target_col)


# ---------------------------------------------------------------------------
# 16. Mann-Whitney U Test
# ---------------------------------------------------------------------------

def compute_mann_whitney(df, target_col):
    """
    Computes per-feature Mann-Whitney U statistic.

    Requires a binary target (two classes).  For multi-class targets,
    the test is applied for each feature between the first two classes.
    For continuous regression targets the target is binarised at its
    median (``y > median(y)``) before the test is run; the coercion is
    recorded in ``result.extra['target_coercion']``.
    """
    sub = df.dropna()
    y_raw = sub[target_col].values
    X_df = sub.drop(columns=[target_col]).select_dtypes(include="number")
    if y_raw.dtype.kind in ("U", "S", "O"):
        uniq = np.unique(y_raw)
        if len(uniq) > MW_KW_MAX_CATEGORICAL_CLASSES:
            raise ValueError(
                f"Mann-Whitney U: target {target_col!r} has {len(uniq)} unique "
                f"categorical values (>{MW_KW_MAX_CATEGORICAL_CLASSES}). "
                f"This is not a classification target — it looks like an ID "
                f"column. Pick a numeric target or a low-cardinality class label."
            )
    coercion = None
    if not _is_classification(y_raw):
        threshold = float(np.median(y_raw.astype(float)))
        y = (y_raw.astype(float) > threshold).astype(int)
        coercion = f"regression→binary median-split @ {threshold:.6g}"
    else:
        y = y_raw
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Mann-Whitney U: target {target_col!r} has <2 distinct classes "
            f"after coercion — cannot run the test."
        )
    c0, c1 = classes[0], classes[1]
    n0, n1 = int((y == c0).sum()), int((y == c1).sum())
    if n0 < MW_KW_MIN_GROUP_SIZE or n1 < MW_KW_MIN_GROUP_SIZE:
        raise ValueError(
            f"Mann-Whitney U: target {target_col!r} produces groups of size "
            f"{n0} and {n1} (minimum {MW_KW_MIN_GROUP_SIZE}). Check target "
            f"choice — a per-row ID column cannot be used."
        )
    scores = {}
    for col in X_df.columns:
        g0 = X_df.loc[y == c0, col].values
        g1 = X_df.loc[y == c1, col].values
        try:
            stat, _ = mannwhitneyu(g0, g1, alternative="two-sided")
            # Normalise to [0,1]: rank-biserial correlation
            scores[col] = float(abs(stat / (n0 * n1) - 0.5) * 2)
        except Exception:
            scores[col] = 0.0
    extra = {"target_coercion": coercion} if coercion else {}
    return _result("mann_whitney", pd.Series(scores), target_col, extra=extra)


# ---------------------------------------------------------------------------
# 17. Kruskal-Wallis H Test
# ---------------------------------------------------------------------------

#: Default quantile bin count for non-parametric ANOVA on continuous targets.
KRUSKAL_REGRESSION_BINS: Final[int] = 5


def compute_kruskal_wallis(df, target_col):
    """
    Per-feature Kruskal-Wallis H statistic.

    For classification targets each class is an H-test group.  For
    continuous regression targets y is binned into
    ``KRUSKAL_REGRESSION_BINS`` equal-frequency quantiles before the
    test is run; the coercion is recorded in
    ``result.extra['target_coercion']``.  The previous behaviour
    (treating every unique float as its own class) produced
    length-1 groups and was silently invalid.
    """
    sub = df.dropna()
    y_raw = sub[target_col].values
    X_df = sub.drop(columns=[target_col]).select_dtypes(include="number")
    if y_raw.dtype.kind in ("U", "S", "O"):
        uniq = np.unique(y_raw)
        if len(uniq) > MW_KW_MAX_CATEGORICAL_CLASSES:
            raise ValueError(
                f"Kruskal-Wallis H: target {target_col!r} has {len(uniq)} unique "
                f"categorical values (>{MW_KW_MAX_CATEGORICAL_CLASSES}). "
                f"This is not a classification target — it looks like an ID "
                f"column. Pick a numeric target or a low-cardinality class label."
            )
    coercion = None
    if not _is_classification(y_raw):
        y_series = pd.Series(y_raw.astype(float))
        y = pd.qcut(y_series, q=KRUSKAL_REGRESSION_BINS, labels=False,
                    duplicates="drop").astype(float).values
        coercion = f"regression→{KRUSKAL_REGRESSION_BINS}-quantile bins"
    else:
        y = y_raw
    classes = np.unique(y[~pd.isna(y)]) if coercion else np.unique(y)
    valid_classes = [c for c in classes if int((y == c).sum()) >= MW_KW_MIN_GROUP_SIZE]
    if len(valid_classes) < 2:
        raise ValueError(
            f"Kruskal-Wallis H: target {target_col!r} produces <2 groups of "
            f"size ≥{MW_KW_MIN_GROUP_SIZE} after coercion. Check target "
            f"choice — a per-row ID column cannot be used."
        )
    scores = {}
    for col in X_df.columns:
        groups = [X_df.loc[y == c, col].values for c in valid_classes]
        try:
            H, _ = kruskal(*groups)
            scores[col] = float(np.nan_to_num(H))
        except Exception:
            scores[col] = 0.0
    extra = {"target_coercion": coercion} if coercion else {}
    return _result("kruskal_wallis", pd.Series(scores), target_col, extra=extra)


# ---------------------------------------------------------------------------
# 18. Mutual Information
# ---------------------------------------------------------------------------

def compute_mutual_information(df, target_col, n_neighbors=DEFAULT_N_NEIGHBORS,
                               rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    fn_func = mutual_info_classif if is_clf else mutual_info_regression
    scores = fn_func(X, y, n_neighbors=n_neighbors, random_state=rs)
    return _result("mutual_information", pd.Series(scores, index=fn), target_col)


# ---------------------------------------------------------------------------
# 19. Information Bottleneck (approximation via normalised MI)
# ---------------------------------------------------------------------------

#: Floor added to differential-entropy denominators to avoid division by
#: zero on (near-)constant features [nats, dimensionless].
IB_ENTROPY_FLOOR: Final[float] = 1e-6


def compute_information_bottleneck(df, target_col, n_neighbors=DEFAULT_N_NEIGHBORS,
                                   rs=DEFAULT_RANDOM_STATE):
    """
    IB-inspired score: I(X_i; Y) / H(X_i).

    The Information Bottleneck (Tishby–Pereira–Bialek 1999) seeks a
    compressed T that maximises I(T;Y) while minimising I(T;X).  For
    per-feature ranking the score I(X_i; Y) / H(X_i) measures how
    efficiently feature i encodes the target relative to its own entropy —
    the IB-efficient direction.

    H(X_i) is estimated via scipy's Vasicek–Ebrahimi nonparametric
    differential entropy estimator (scipy.stats.differential_entropy),
    which is consistent for continuous 1-D distributions.  This replaces
    an earlier self-MI-with-noise approximation that did not estimate
    entropy at all.
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    mi_func = mutual_info_classif if is_clf else mutual_info_regression
    mi = mi_func(X, y, n_neighbors=n_neighbors, random_state=rs)

    entropy_per_feature = np.array([
        max(float(differential_entropy(X[:, i])), IB_ENTROPY_FLOOR)
        for i in range(X.shape[1])
    ])
    scores = mi / entropy_per_feature
    return _result("information_bottleneck",
                   pd.Series(scores, index=fn), target_col,
                   extra={"entropy": pd.Series(entropy_per_feature, index=fn),
                          "mutual_info": pd.Series(mi, index=fn)})


# ---------------------------------------------------------------------------
# 20. MRMR — Min-Redundancy Max-Relevance
# ---------------------------------------------------------------------------

def compute_mrmr(df, target_col, n_features=DEFAULT_MRMR_N_FEATURES,
                 n_neighbors=DEFAULT_N_NEIGHBORS, rs=DEFAULT_RANDOM_STATE):
    """
    Peng–Ding–Long MRMR via mutual information.

    Iteratively selects features maximising:
        score(f) = MI(f, Y) - (1/|S|) Σ_{s∈S} MI(f, s)
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    mi_func = mutual_info_classif if is_clf else mutual_info_regression
    n_select = min(n_features, X.shape[1])

    relevance = mi_func(X, y, n_neighbors=n_neighbors, random_state=rs)
    selected: list = []
    remaining = list(range(X.shape[1]))
    mrmr_scores = np.zeros(X.shape[1])

    for step in range(n_select):
        if not remaining:
            break
        best_i, best_score = -1, -np.inf
        for i in remaining:
            if not selected:
                score = relevance[i]
            else:
                redundancy = np.mean([
                    mutual_info_regression(
                        X[:, i].reshape(-1, 1), X[:, j], random_state=rs
                    )[0]
                    for j in selected
                ])
                score = relevance[i] - redundancy
            if score > best_score:
                best_score, best_i = score, i
        mrmr_scores[best_i] = n_select - step   # rank-order score
        selected.append(best_i)
        remaining.remove(best_i)

    return _result("mrmr", pd.Series(mrmr_scores, index=fn), target_col,
                   extra={"selected_order": [fn[i] for i in selected]})


# ---------------------------------------------------------------------------
# 21. Permutation Importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(df, target_col, n_estimators=LIGHT_N_ESTIMATORS,
                                    n_repeats=DEFAULT_N_REPEATS,
                                    rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
    est.fit(X, y)
    result = _pi(est, X, y, n_repeats=n_repeats, random_state=rs)
    scores = result.importances_mean
    stds   = result.importances_std
    return _result("permutation_importance",
                   pd.Series(scores, index=fn), target_col,
                   extra={"stds": pd.Series(stds, index=fn)})


# ---------------------------------------------------------------------------
# 22. SHAP — KernelSHAP approximation (no shap package needed)
# ---------------------------------------------------------------------------

def compute_shap(df, target_col, n_estimators=LIGHT_N_ESTIMATORS,
                 n_bg=DEFAULT_SHAP_N_BG, n_coal=DEFAULT_SHAP_N_COAL,
                 rs=DEFAULT_RANDOM_STATE):
    """
    KernelSHAP approximation.

    Fits a RandomForest, then estimates SHAP values via a sampling-based
    weighted linear surrogate (Lundberg & Lee, 2017 — Algorithm 1).
    Background = random sample of n_bg rows.
    n_coal coalitions sampled per background point.

    Returns mean |SHAP| per feature as global importance, plus the full
    SHAP matrix (n_samples × n_features) in extra for beeswarm plotting.
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
    est.fit(X, y)
    predict = est.predict_proba if is_clf and hasattr(est, "predict_proba") else est.predict

    n_samples, n_features = X.shape
    rng = np.random.RandomState(rs)
    bg_idx = rng.choice(n_samples, size=min(n_bg, n_samples), replace=False)
    background = X[bg_idx]                # (n_bg, n_features)

    # Pre-compute kernel weights for coalitions of size s out of n_features
    def _kernel_weight(s, n):
        if s == 0 or s == n:
            return 1e6
        return (n - 1) / (_comb(n, s, exact=False) * s * (n - s))

    shap_matrix = np.zeros((n_samples, n_features))

    for idx in range(n_samples):
        x = X[idx]
        phi = np.zeros(n_features)

        # Sample coalitions
        z_arr = rng.randint(0, 2, (n_coal, n_features))
        weights = np.array([_kernel_weight(int(z.sum()), n_features) for z in z_arr])

        f_vals = np.zeros(n_coal)
        for k, z in enumerate(z_arr):
            # Average over background: f(z mask on x, background otherwise)
            x_masked = np.where(z[None, :], x[None, :], background)   # (n_bg, n_features)
            try:
                preds = predict(x_masked)
                if preds.ndim > 1:
                    preds = preds[:, 1]   # binary classif — P(class=1)
                f_vals[k] = preds.mean()
            except Exception:
                f_vals[k] = 0.0

        # Fit weighted least squares: z_arr @ phi ≈ f_vals
        try:
            W = np.sqrt(weights)
            Zw = z_arr * W[:, None]
            fw = f_vals * W
            phi, _, _, _ = np.linalg.lstsq(Zw, fw, rcond=None)
        except Exception:
            phi = np.zeros(n_features)

        shap_matrix[idx] = phi

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    return _result("shap",
                   pd.Series(mean_abs_shap, index=fn), target_col,
                   extra={"shap_matrix": shap_matrix,
                          "X_raw": X,
                          "feature_names": list(fn)})


# ---------------------------------------------------------------------------
# 23. LIME — Local Interpretable Model-Agnostic Explanations (approximation)
# ---------------------------------------------------------------------------

def compute_lime(df, target_col, n_estimators=LIGHT_N_ESTIMATORS,
                  n_samples=DEFAULT_LIME_N_SAMPLES,
                  sigma=DEFAULT_LIME_SIGMA, n_display=DEFAULT_LIME_N_DISPLAY,
                  rs=DEFAULT_RANDOM_STATE):
    """
    LIME approximation: perturb each query point with Gaussian noise,
    weight by similarity, fit local linear model, aggregate |coeff|.

    Aggregates over min(n_display, n_samples_data) query points and
    returns mean |local coefficient| as global feature importance.
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
    est.fit(X, y)
    predict = est.predict_proba if is_clf and hasattr(est, "predict_proba") else est.predict

    rng = np.random.RandomState(rs)
    n_queries = min(n_display, len(X))
    query_idx = rng.choice(len(X), size=n_queries, replace=False)

    global_coef = np.zeros(X.shape[1])

    for idx in query_idx:
        x = X[idx]
        perturb = x[None, :] + rng.normal(0, sigma, (n_samples, X.shape[1]))
        dist = np.sqrt(((perturb - x) ** 2).sum(axis=1))
        weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))

        try:
            preds = predict(perturb)
            if preds.ndim > 1:
                preds = preds[:, 1]
        except Exception:
            continue

        # Weighted least-squares local linear model
        W = np.diag(weights)
        XtW = perturb.T @ W
        try:
            coef = np.linalg.lstsq(XtW @ perturb, XtW @ preds, rcond=None)[0]
            global_coef += np.abs(coef)
        except Exception:
            pass

    global_coef /= max(n_queries, 1)
    return _result("lime", pd.Series(global_coef, index=fn), target_col)


# ---------------------------------------------------------------------------
# 24. Feature Interaction Networks (H-statistic approximation)
# ---------------------------------------------------------------------------

def compute_feature_interactions(df, target_col, n_estimators=LIGHT_N_ESTIMATORS,
                                 rs=DEFAULT_RANDOM_STATE):
    """
    Pairwise feature interaction strengths approximated via:
        H²(i,j) = Var(PD_ij - PD_i - PD_j) / Var(PD_ij)

    where PD functions are estimated by replacing feature values with
    their mean (marginal integration over a grid sample).
    Uses RF for faster computation.
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    n_f = X.shape[1]
    est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
    est.fit(X, y)
    predict = (lambda x: est.predict_proba(x)[:, 1]) if is_clf else est.predict

    # Marginalise feature i: replace X[:,i] with its mean
    def _pd_marginal(i):
        X_m = X.copy(); X_m[:, i] = X[:, i].mean()
        return predict(X_m)

    full_pred = predict(X)
    marginals = [_pd_marginal(i) for i in range(n_f)]

    interaction_matrix = np.zeros((n_f, n_f))
    for i in range(n_f):
        for j in range(i + 1, n_f):
            X_ij = X.copy()
            X_ij[:, i] = X[:, i].mean()
            X_ij[:, j] = X[:, j].mean()
            pd_ij = predict(X_ij)
            h2_num = np.var(full_pred - pd_ij - marginals[i] - marginals[j])
            h2_den = np.var(full_pred - pd_ij) + 1e-12
            interaction_matrix[i, j] = interaction_matrix[j, i] = h2_num / h2_den

    # Per-feature total interaction strength
    scores = interaction_matrix.sum(axis=1)
    return _result("feature_interactions",
                   pd.Series(scores, index=fn), target_col,
                   extra={"interaction_matrix": pd.DataFrame(
                       interaction_matrix, index=fn, columns=fn)})


# ---------------------------------------------------------------------------
# 25. Genetic Algorithm Feature Selection
# ---------------------------------------------------------------------------

def compute_genetic_algorithm(df, target_col, pop_size=DEFAULT_GA_POP, n_gen=40,
                               mutation_rate=DEFAULT_GA_MUTATION_RATE,
                               n_estimators=MINIMAL_N_ESTIMATORS,
                               cv=MINIMAL_CV_FOLDS, rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    n_features = X.shape[1]
    rng = np.random.RandomState(rs)

    def _fitness(mask):
        sel = np.where(mask)[0]
        if len(sel) == 0:
            return 0.0
        est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
        scoring = "accuracy" if is_clf else "r2"
        s = cross_val_score(est, X[:, sel], y, cv=min(cv, len(y)//5, 5), scoring=scoring)
        return float(s.mean()) - 0.005 * len(sel) / n_features

    # Initialise
    pop = rng.randint(0, 2, (pop_size, n_features)).astype(bool)
    for i in range(pop_size):
        if not pop[i].any():
            pop[i, rng.randint(0, n_features)] = True

    selection_counts = np.zeros(n_features)

    for _ in range(n_gen):
        fitness = np.array([_fitness(p) for p in pop])
        selection_counts += pop.sum(axis=0)

        # Tournament selection
        new_pop = []
        for _ in range(pop_size):
            a, b = rng.choice(pop_size, 2, replace=False)
            new_pop.append(pop[a].copy() if fitness[a] >= fitness[b] else pop[b].copy())
        pop = np.array(new_pop)

        # Uniform crossover
        for i in range(0, pop_size - 1, 2):
            mask = rng.randint(0, 2, n_features).astype(bool)
            child1 = np.where(mask, pop[i], pop[i + 1])
            child2 = np.where(mask, pop[i + 1], pop[i])
            pop[i], pop[i + 1] = child1, child2

        # Mutation
        mut = rng.uniform(0, 1, pop.shape) < mutation_rate
        pop = np.where(mut, ~pop, pop)
        for i in range(pop_size):
            if not pop[i].any():
                pop[i, rng.randint(0, n_features)] = True

    scores = selection_counts / (n_gen * pop_size)
    return _result("genetic_algorithm", pd.Series(scores, index=fn), target_col)


# ---------------------------------------------------------------------------
# 26. Particle Swarm Optimization
# ---------------------------------------------------------------------------

def compute_pso(df, target_col, n_particles=DEFAULT_PSO_SWARM, n_iter=40,
                w=DEFAULT_PSO_W, c1=DEFAULT_PSO_C1, c2=DEFAULT_PSO_C2,
                n_estimators=MINIMAL_N_ESTIMATORS, cv=MINIMAL_CV_FOLDS,
                rs=DEFAULT_RANDOM_STATE):
    X, y, fn, is_clf = _prepare(df, target_col)
    n_features = X.shape[1]
    rng = np.random.RandomState(rs)

    def _fitness(pos):
        sel = np.where(pos > 0.5)[0]
        if len(sel) == 0:
            return 0.0
        est = _make_estimator(is_clf, n_estimators=n_estimators, rs=rs)
        scoring = "accuracy" if is_clf else "r2"
        s = cross_val_score(est, X[:, sel], y, cv=min(cv, len(y)//5, 5), scoring=scoring)
        return float(s.mean()) - 0.005 * len(sel) / n_features

    pos = rng.uniform(0, 1, (n_particles, n_features))
    vel = rng.uniform(-0.5, 0.5, (n_particles, n_features))
    pbest = pos.copy()
    pbest_fit = np.array([_fitness(p) for p in pos])
    gbest = pbest[pbest_fit.argmax()].copy()
    gbest_fit = pbest_fit.max()

    for _ in range(n_iter):
        r1 = rng.uniform(0, 1, pos.shape)
        r2 = rng.uniform(0, 1, pos.shape)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0.0, 1.0)

        for i in range(n_particles):
            f = _fitness(pos[i])
            if f > pbest_fit[i]:
                pbest[i] = pos[i].copy(); pbest_fit[i] = f
                if f > gbest_fit:
                    gbest = pos[i].copy(); gbest_fit = f

    # Return gbest position as continuous importance [0, 1]
    return _result("pso", pd.Series(gbest, index=fn), target_col)


# ---------------------------------------------------------------------------
# 27. ReliefF (RReliefF for regression, ReliefF for classification)
# ---------------------------------------------------------------------------

def compute_relieff(df, target_col, k=DEFAULT_RELIEFF_K, n_iter=None,
                    rs=DEFAULT_RANDOM_STATE):
    """
    Kira–Rendell ReliefF (classification) and Robnik-Šikonja RReliefF (regression).

    Feature weight update:
      hit:  w[f] -= diff(f,xi,nh)^2 / (m*k)
      miss: w[f] += diff(f,xi,nm)^2 * P(class_miss) / (m*k)
    """
    X, y, fn, is_clf = _prepare(df, target_col, standardize=False)
    X = MinMaxScaler().fit_transform(X)
    n_samples, n_features = X.shape
    n_iter = n_iter or n_samples
    rng = np.random.RandomState(rs)
    weights = np.zeros(n_features)

    if is_clf:
        classes, counts = np.unique(y, return_counts=True)
        class_prior = {c: cnt / n_samples for c, cnt in zip(classes, counts)}

        for _ in range(n_iter):
            i = rng.randint(0, n_samples)
            xi, yi = X[i], y[i]

            same_class = np.where(y == yi)[0]
            same_class = same_class[same_class != i]
            diff_class = {c: np.where(y == c)[0] for c in classes if c != yi}

            if len(same_class) == 0:
                continue

            dists_same = np.sum((X[same_class] - xi) ** 2, axis=1)
            hits = same_class[np.argsort(dists_same)[:k]]
            weights -= np.abs(X[hits] - xi).mean(axis=0) / n_iter

            for c, cidx in diff_class.items():
                if len(cidx) == 0:
                    continue
                dists_miss = np.sum((X[cidx] - xi) ** 2, axis=1)
                misses = cidx[np.argsort(dists_miss)[:k]]
                p_c = class_prior[c] / (1 - class_prior[yi] + 1e-10)
                weights += p_c * np.abs(X[misses] - xi).mean(axis=0) / n_iter

    else:   # RReliefF (Robnik-Šikonja & Kononenko 1997)
        # Rank-normalise the target so weights are insensitive to outliers
        # in y_range (the previous raw-range implementation collapsed to
        # near-zero target_diff for typical neighbour pairs when y had
        # long-tailed outliers, e.g. E_a with max ≫ median).
        y_rank = (rankdata(y, method="average") - 1.0) / max(1, n_samples - 1)
        # Accumulators: N_dC scalar, N_dA and N_dC_dA vectors over features.
        n_dc = 0.0
        n_da = np.zeros(n_features)
        n_dc_da = np.zeros(n_features)
        for _ in range(n_iter):
            i = rng.randint(0, n_samples)
            xi, yri = X[i], y_rank[i]
            dists = np.sqrt(np.sum((X - xi) ** 2, axis=1))
            dists[i] = np.inf
            knn = np.argsort(dists)[:k]
            for j in knn:
                dC = abs(yri - y_rank[j])        # ∈ [0, 1]
                dA = np.abs(xi - X[j])           # features already in [0, 1]
                n_dc    += dC / k
                n_da    += dA / k
                n_dc_da += dA * dC / k
        denom_rel = n_dc + 1e-12
        denom_irr = (n_iter - n_dc) + 1e-12
        weights = n_dc_da / denom_rel - (n_da - n_dc_da) / denom_irr

    return _result("relieff", pd.Series(weights, index=fn), target_col)


# ---------------------------------------------------------------------------
# 28. Recursive Feature Clustering
# ---------------------------------------------------------------------------

def compute_recursive_feature_clustering(df, target_col, n_clusters=None,
                                          linkage="average",
                                          rs=DEFAULT_RANDOM_STATE):
    """
    Hierarchical clustering of features by |correlation|, then selects
    the feature per cluster with highest MI to the target as the
    representative.  Importance = 1/rank within cluster × cluster_mi.
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    n_features = X.shape[1]
    n_clust = n_clusters or max(1, n_features // 3)

    from sklearn.cluster import AgglomerativeClustering
    # Distance = 1 - |corr|
    corr_mat = np.corrcoef(X.T)
    dist_mat = 1 - np.abs(np.clip(corr_mat, -1, 1))
    np.fill_diagonal(dist_mat, 0)

    clust = AgglomerativeClustering(
        n_clusters=min(n_clust, n_features),
        metric="precomputed", linkage=linkage,
    )
    labels = clust.fit_predict(dist_mat)

    mi_func = mutual_info_classif if is_clf else mutual_info_regression
    mi = mi_func(X, y, random_state=rs)

    scores = np.zeros(n_features)
    for cid in np.unique(labels):
        members = np.where(labels == cid)[0]
        cluster_mi = mi[members]
        rank_in_cluster = np.argsort(-cluster_mi)   # 0=best
        for rank, member_idx in enumerate(members[np.argsort(-cluster_mi)]):
            scores[member_idx] = cluster_mi[rank_in_cluster[rank]] / (rank + 1)

    extra = {"cluster_labels": pd.Series(labels, index=fn),
             "mi_scores": pd.Series(mi, index=fn)}
    return _result("recursive_feature_clustering",
                   pd.Series(scores, index=fn), target_col, extra=extra)


# ---------------------------------------------------------------------------
# 29. Bayesian Variable Importance
# ---------------------------------------------------------------------------

#: Number of bootstrap resamples for posterior z-score in the classification
#: path of bayesian_variable_importance.  Draws from Rubin (1981): the
#: bootstrap distribution of a regression coefficient is an asymptotic
#: approximation to its posterior under a non-informative prior.
BAYESIAN_VI_N_BOOTSTRAP: Final[int] = 50
BAYESIAN_VI_EPS: Final[float] = 1e-10


def compute_bayesian_variable_importance(df, target_col,
                                         n_iter=DEFAULT_MAX_ITER_BAYES,
                                         rs=DEFAULT_RANDOM_STATE):
    """
    Posterior coefficient z-score as variable importance.

    Regression: BayesianRidge gives a closed-form posterior over coefficients.
        importance[i] = |mean(beta_i)| / sqrt(Sigma_ii)

    Classification: no closed-form posterior in sklearn.  We use the bootstrap
    posterior of LogisticRegression coefficients (Rubin 1981 — the nonparametric
    bootstrap of an MLE is asymptotically equivalent to the Bayesian posterior
    under a flat prior):
        importance[i] = |mean_b(beta_i)| / std_b(beta_i)
    where b indexes BAYESIAN_VI_N_BOOTSTRAP resampled fits.
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    if is_clf:
        from sklearn.linear_model import LogisticRegression
        n_samples, n_features = X.shape
        rng = np.random.RandomState(rs)
        boot_coefs = np.zeros((BAYESIAN_VI_N_BOOTSTRAP, n_features))
        for b in range(BAYESIAN_VI_N_BOOTSTRAP):
            idx = rng.randint(0, n_samples, size=n_samples)
            Xb, yb = X[idx], y[idx]
            # Degenerate resamples (one class only) skipped → coef row stays 0
            if len(np.unique(yb)) < 2:
                continue
            lr = LogisticRegression(
                C=DEFAULT_SVM_C, max_iter=n_iter, random_state=rs + b,
                solver="liblinear",
            )
            try:
                lr.fit(Xb, yb)
                # Multi-class: mean |coef| across classes (OvR rows)
                coef = np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1 else lr.coef_.ravel()
                boot_coefs[b] = coef
            except Exception as exc:   # singular design, convergence, etc.
                log.warning("bayesian_variable_importance: bootstrap %d failed: %s", b, exc)
        mean_b = boot_coefs.mean(axis=0)
        std_b  = boot_coefs.std(axis=0)
        scores = np.abs(mean_b) / (std_b + BAYESIAN_VI_EPS)
    else:
        est = BayesianRidge(max_iter=n_iter)
        est.fit(X, y)
        # posterior z-score: |mean| / sigma
        sigma_w = np.sqrt(np.diag(est.sigma_))
        scores = np.abs(est.coef_) / (sigma_w + BAYESIAN_VI_EPS)

    return _result("bayesian_variable_importance",
                   pd.Series(scores, index=fn), target_col)


# ---------------------------------------------------------------------------
# 30. Markov Blanket (IAMB algorithm)
# ---------------------------------------------------------------------------

#: Minimum residual-std ratio below which a Fisher-z transform is treated
#: as perfectly dependent — avoids atanh(±1) blowup.
PARTIAL_CORR_MAX_ABS: Final[float] = 1.0 - 1e-7


def _partial_corr_ci_test(X: np.ndarray, y: np.ndarray,
                          f_idx: int, cond_idx: list) -> Tuple[float, float]:
    """
    Fisher-z test for H₀: ρ(X_f, Y | X_cond) = 0.

    Under the Gaussian assumption (Kalisch & Bühlmann 2007, Algorithm 2),
    this is the CI test used by the PC and IAMB algorithms on continuous
    data.  Returns (partial_correlation, p_value_two_sided).

    For |cond| == 0 this degenerates to the ordinary Pearson correlation
    Fisher-z test.
    """
    from scipy.stats import norm as _norm
    n = len(y)
    yf = y.astype(float)
    if not cond_idx:
        r = float(np.corrcoef(X[:, f_idx], yf)[0, 1])
    else:
        # Residualise both X_f and y against the conditioning set via OLS.
        Z = np.column_stack([X[:, cond_idx], np.ones(n)])
        beta_f, *_ = np.linalg.lstsq(Z, X[:, f_idx], rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z, yf,          rcond=None)
        r_f = X[:, f_idx] - Z @ beta_f
        r_y = yf           - Z @ beta_y
        denom = float(np.std(r_f) * np.std(r_y))
        r = 0.0 if denom == 0.0 else float(np.mean(r_f * r_y) / denom)
    r = float(np.clip(r, -PARTIAL_CORR_MAX_ABS, PARTIAL_CORR_MAX_ABS))
    df = n - len(cond_idx) - 3
    if df <= 0:
        return r, 1.0
    z = 0.5 * np.log((1.0 + r) / (1.0 - r)) * np.sqrt(df)
    p = 2.0 * (1.0 - _norm.cdf(abs(z)))
    return r, float(p)


def compute_markov_blanket(df, target_col, n_neighbors=DEFAULT_N_NEIGHBORS,
                            alpha=DEFAULT_CI_ALPHA,
                            max_features=None, rs=DEFAULT_RANDOM_STATE):
    """
    Incremental Association Markov Blanket (IAMB, Tsamardinos et al. 2003)
    with a Fisher-z partial-correlation CI test.

    Phase 1 (grow): repeatedly adds the feature f with maximum |partial
    correlation| ρ(f, Y | MB) whose CI test rejects H₀ at level `alpha`.
    Phase 2 (shrink): removes features whose ρ(f, Y | MB∖{f}) cannot
    reject H₀ at level `alpha`.

    Assumption: jointly Gaussian (X, Y).  For non-Gaussian data the test
    is not a valid CI test and should be replaced with a nonparametric
    alternative (e.g. kernel-CMI).  Documented explicitly because earlier
    versions of this module used an MI-of-linear-residuals approximation
    that was silently wrong outside the Gaussian case.

    Returns binary (0/1) membership scores with the final blanket in
    extra["selected_features"] and per-feature p-values in extra["pvalues"].
    """
    X, y, fn, _ = _prepare(df, target_col)
    n_features = X.shape[1]
    max_f = max_features or n_features

    pvalues = np.ones(n_features)
    correls = np.zeros(n_features)

    # Grow phase
    mb: list = []
    remaining = list(range(n_features))
    for _ in range(max_f):
        if not remaining:
            break
        best_i, best_abs_r, best_p = -1, -1.0, 1.0
        for i in remaining:
            r, p = _partial_corr_ci_test(X, y, i, mb)
            if abs(r) > best_abs_r:
                best_i, best_abs_r, best_p = i, abs(r), p
                correls[i], pvalues[i] = r, p
        if best_i < 0 or best_p >= alpha:
            break
        mb.append(best_i)
        remaining.remove(best_i)

    # Shrink phase
    to_remove = []
    for m in list(mb):
        cond = [x for x in mb if x != m]
        r, p = _partial_corr_ci_test(X, y, m, cond)
        correls[m], pvalues[m] = r, p
        if p >= alpha:
            to_remove.append(m)
    for m in to_remove:
        mb.remove(m)

    scores = np.zeros(n_features)
    scores[mb] = 1.0
    return _result("markov_blanket",
                   pd.Series(scores, index=fn), target_col,
                   extra={"selected_features": [fn[i] for i in mb],
                          "pvalues":        pd.Series(pvalues, index=fn),
                          "partial_corr":   pd.Series(correls, index=fn)})


# ---------------------------------------------------------------------------
# 31. Autoencoder-Based Importance
# ---------------------------------------------------------------------------

#: Minimum usable bottleneck width (even when n_features is tiny).
AE_BOTTLENECK_MIN: Final[int] = 2
#: Fraction of the encoder hidden size used for the bottleneck.
AE_BOTTLENECK_RATIO: Final[float] = 0.5
#: Upper bound on hidden size relative to n_features (prevents runaway widths).
AE_HIDDEN_MAX_MULTIPLIER: Final[int] = 2
#: Ridge regularisation for the supervised head (prevents collinear blow-up
#: when [X ; X̂] has duplicate columns on datasets where the AE reconstructs
#: near-perfectly).
AE_HEAD_RIDGE_ALPHA: Final[float] = 1.0


def compute_autoencoder_importance(df, target_col, hidden_size=DEFAULT_AE_HIDDEN,
                                   n_iter=DEFAULT_MAX_ITER_BAYES,
                                   rs=DEFAULT_RANDOM_STATE):
    """
    Target-aware encoder + supervised head importance.

    Architecture:

        X  ──► MLPRegressor(h, bottleneck, h) ──► X̂         (autoencoder)
                                                   │
                                                   ▼
                           Ridge / LogisticRegression(y)    (supervised head)

    The autoencoder is trained unsupervised on X→X to learn a denoised
    representation X̂.  A supervised head is then fit on
    Z = [X ; X̂] → y.  Feature importance is the **end-to-end permutation
    drop**: for each feature j we permute column j in X, re-encode to X̂',
    rebuild Z' = [X_perm ; X̂'], score the head on (Z', y), and record the
    drop from the base score.

    This is the Robnik-Šikonja-style "fit a model then permute one column"
    recipe — but routed through an autoencoder first, so the contribution
    of each feature is measured after non-linear reconstruction, not just
    through the linear head.  Unlike the previous (unsupervised-only)
    implementation, the resulting scores reflect target relevance.

    The previous design trained `n_features + 1` autoencoders per call
    and never looked at `y`; this version trains a single autoencoder
    plus a single head and is O(n_features) slower than a plain model,
    not O(n_features²).
    """
    X, y, fn, is_clf = _prepare(df, target_col)
    n_samples, n_features = X.shape
    h = max(AE_BOTTLENECK_MIN * 2,
            min(hidden_size, n_features * AE_HIDDEN_MAX_MULTIPLIER))
    bottleneck = max(AE_BOTTLENECK_MIN, int(h * AE_BOTTLENECK_RATIO))

    # ── 1. Train the autoencoder X → X (single multi-output fit) ─────────
    ae = MLPRegressor(
        hidden_layer_sizes=(h, bottleneck, h),
        max_iter=n_iter, random_state=rs,
        early_stopping=True, validation_fraction=0.1,
    )
    ae.fit(X, X)
    X_hat = ae.predict(X)

    # ── 2. Fit the supervised head on Z = [X ; X̂] → y ────────────────────
    Z = np.hstack([X, X_hat])
    if is_clf:
        from sklearn.linear_model import LogisticRegression
        head = LogisticRegression(
            max_iter=DEFAULT_MAX_ITER_LOGISTIC, random_state=rs,
        )
    else:
        head = Ridge(alpha=AE_HEAD_RIDGE_ALPHA)
    head.fit(Z, y)
    base_score = float(head.score(Z, y))

    # ── 3. End-to-end permutation importance on original features ─────────
    rng = np.random.default_rng(rs)
    scores = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        Z_perm = np.hstack([X_perm, ae.predict(X_perm)])
        scores[i] = max(0.0, base_score - float(head.score(Z_perm, y)))

    return _result(
        "autoencoder_importance",
        pd.Series(scores, index=fn),
        target_col,
        extra={
            "base_score":      base_score,
            "bottleneck_size": bottleneck,
            "hidden_size":     h,
            "head_kind":       "logistic" if is_clf else "ridge",
        },
    )


# ---------------------------------------------------------------------------
# 32. Deep Learning Attribution (Gradient × Input via PyTorch)
# ---------------------------------------------------------------------------

def compute_deep_learning_attribution(df, target_col, hidden_size=64,
                                       n_epochs=DEFAULT_DL_EPOCHS,
                                       lr=DEFAULT_DL_LR,
                                       rs=DEFAULT_RANDOM_STATE):
    """
    Gradient × Input attribution using a PyTorch MLP.

    Trains a 2-layer MLP, then computes |∂ŷ/∂x_i| × |x_i| averaged
    over all samples.  This is the Simonyan–Vedaldi–Zisserman (2014)
    saliency map adapted to tabular regression/classification.

    Requires torch (available in dnv environment).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "Deep Learning Attribution requires PyTorch.\n"
            "Install with:  conda install pytorch  or  pip install torch"
        )

    X, y, fn, is_clf = _prepare(df, target_col)
    torch.manual_seed(rs)
    n_features = X.shape[1]
    h = max(16, min(hidden_size, n_features * 4))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y.astype(float), dtype=torch.float32).unsqueeze(1)

    n_out = 1
    loss_fn = nn.MSELoss()

    if is_clf:
        n_classes = int(y.max()) + 1
        y_t = torch.tensor(y.astype(int), dtype=torch.long)
        n_out = n_classes
        loss_fn = nn.CrossEntropyLoss()

    model = nn.Sequential(
        nn.Linear(n_features, h), nn.ReLU(),
        nn.Linear(h, h // 2),    nn.ReLU(),
        nn.Linear(h // 2, n_out),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()
        out = model(X_t)
        if is_clf:
            loss = loss_fn(out, y_t)
        else:
            loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()

    # Compute gradient × input, then take |·| *per sample* before averaging —
    # averaging signed contributions first allows positive/negative sample
    # contributions to cancel for sign-balanced features (Sundararajan et al.
    # 2017, "Axiomatic Attribution for Deep Networks" — attribution magnitude
    # is the per-sample absolute value, not the absolute of the mean).
    X_inp = X_t.clone().requires_grad_(True)
    out = model(X_inp)
    scalar_out = out.sum()
    scalar_out.backward()
    attr_per_sample = np.abs(X_inp.grad.detach().numpy() * X)   # (n_samples, n_features)
    importance = attr_per_sample.mean(axis=0)

    return _result("deep_learning_attribution",
                   pd.Series(importance, index=fn), target_col,
                   extra={"loss_final": float(loss.item())})


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
    # key                           title                                     category           callable
    "decision_tree":               ("Decision Tree Importances",              "BASIC",            compute_decision_tree),
    "random_forest":               ("Random Forest Importances",              "BASIC",            compute_random_forest),
    "linear_regression":           ("Linear Regression Coefficients",         "BASIC",            compute_linear_regression),
    "svm":                         ("SVM Coefficients (Linear)",              "BASIC",            compute_svm),
    "feature_maps":                ("Feature Maps (Corr Heatmap)",            "BASIC",            compute_feature_maps),
    "lasso":                       ("Lasso Importances",                      "REGULARISATION",   compute_lasso),
    "ridge":                       ("Ridge Importances",                      "REGULARISATION",   compute_ridge),
    "elasticnet":                  ("Elastic Net Importances",                "REGULARISATION",   compute_elasticnet),
    "sparse_pca":                  ("Sparse PCA Importances",                 "REGULARISATION",   compute_sparse_pca),
    "gradient_boosting":           ("Gradient Boosting Importances",          "ENSEMBLE",         compute_gradient_boosting),
    "xgboost":                     ("XGBoost Gain Importances",               "ENSEMBLE",         compute_xgboost),
    "rfe":                         ("Recursive Feature Elimination",          "ENSEMBLE",         compute_rfe),
    "boruta":                      ("Boruta Feature Selection",               "ENSEMBLE",         compute_boruta),
    "anova_f":                     ("ANOVA F-value",                          "STATISTICAL",      compute_anova_f),
    "chi_square":                  ("Chi-Square Scores",                      "STATISTICAL",      compute_chi_square),
    "kendall_tau":                 ("Kendall Tau Correlation",                "STATISTICAL",      compute_kendall_tau),
    "mann_whitney":                ("Mann-Whitney U Test",                    "STATISTICAL",      compute_mann_whitney),
    "kruskal_wallis":              ("Kruskal-Wallis H Test",                  "STATISTICAL",      compute_kruskal_wallis),
    "mutual_information":          ("Mutual Information",                     "STATISTICAL",      compute_mutual_information),
    "information_bottleneck":      ("Information Bottleneck",                 "STATISTICAL",      compute_information_bottleneck),
    "mrmr":                        ("MRMR (Min-Redundancy Max-Relevance)",    "STATISTICAL",      compute_mrmr),
    "permutation_importance":      ("Permutation Importance",                 "AGNOSTIC",         compute_permutation_importance),
    "shap":                        ("SHAP Values (KernelSHAP)",               "AGNOSTIC",         compute_shap),
    "lime":                        ("LIME (Local Explanations)",              "AGNOSTIC",         compute_lime),
    "feature_interactions":        ("Feature Interaction Network",            "AGNOSTIC",         compute_feature_interactions),
    "genetic_algorithm":           ("Genetic Algorithm Selection",            "OPTIMISATION",     compute_genetic_algorithm),
    "pso":                         ("Particle Swarm Optimization",            "OPTIMISATION",     compute_pso),
    "relieff":                     ("ReliefF Algorithm",                      "OPTIMISATION",     compute_relieff),
    "recursive_feature_clustering":("Recursive Feature Clustering",           "OPTIMISATION",     compute_recursive_feature_clustering),
    "bayesian_variable_importance":("Bayesian Variable Importance",           "PROBABILISTIC",    compute_bayesian_variable_importance),
    "markov_blanket":              ("Markov Blanket Selection",               "PROBABILISTIC",    compute_markov_blanket),
    "autoencoder_importance":      ("Autoencoder-Based Importance",           "PROBABILISTIC",    compute_autoencoder_importance),
    "deep_learning_attribution":   ("Deep Learning Attribution",              "PROBABILISTIC",    compute_deep_learning_attribution),
}
