"""
DNV Scientific Module
---------------------
Role:
    Provides pure computation functions for machine learning model training,
    evaluation, and result packaging across 17 methods in 7 families.

Scientific Context:
    Each compute_* function trains a specified ML model on (X, y) derived from
    a DataAsset, evaluates it on a held-out test split, and returns a typed
    MLResult record with metrics, predictions, and supplementary diagnostics
    (feature importances, confusion matrices, ROC data).  Classification vs
    regression is auto-detected from target cardinality or overridden via
    task_type.  StandardScaler is applied to X when standardize=True.

Invariants:
    - Every compute_* function returns an MLResult with is_valid set correctly.
    - _prepare always returns a _PreparedData with X_train/X_test as float64
      numpy arrays after optional standardisation.
    - CLASSIFICATION_MAX_UNIQUE is the single site defining the clf/reg boundary.
    - _fit_and_eval is the single site that calls model.fit, model.predict, and
      packages metrics, importances, ROC data, and loss curves.

Assumptions:
    - df contains at least MIN_VALID_ROWS rows after NaN removal.
    - feature_cols are all numeric columns present in df.
    - target_col is present in df.
    - PyTorch 2.x is available in the dnv conda environment.

Failure Modes:
    - Fewer than MIN_VALID_ROWS valid rows: raises ValueError.
    - XGBoost / LightGBM not installed: raises ImportError with install hint.
    - Task mismatch (e.g. linear_regression with clf target): raises ValueError.
    - PyTorch unavailable: raises ImportError with install hint.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import functools
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
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
# Numeric constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum valid rows after NaN removal for model fitting [dimensionless].
MIN_VALID_ROWS: Final[int] = 10

#: Target unique-value count at or below which classification is assumed [dimensionless].
CLASSIFICATION_MAX_UNIQUE: Final[int] = 20

#: Default train/test split test fraction [dimensionless].
TEST_SIZE_DEFAULT: Final[float] = 0.2

#: Default random state seed [dimensionless].
RANDOM_STATE_DEFAULT: Final[int] = 42

#: Default number of tree-based estimators [dimensionless].
N_ESTIMATORS_DEFAULT: Final[int] = 100

#: Sentinel value for unlimited tree depth (maps to sklearn max_depth=None) [dimensionless].
MAX_DEPTH_UNLIMITED_SENTINEL: Final[int] = 0

#: Default gradient boosting learning rate [dimensionless].
GB_LEARNING_RATE_DEFAULT: Final[float] = 0.1

#: Default k-NN neighbour count [dimensionless].
KNN_N_NEIGHBORS_DEFAULT: Final[int] = 5

#: Default MLP hidden layer sizes (comma-separated integers).
MLP_HIDDEN_DEFAULT: Final[str] = "100,50"

#: Default MLP max training iterations [dimensionless].
MLP_MAX_ITER_DEFAULT: Final[int] = 200

#: Default PyTorch MLP training epochs [dimensionless].
PYTORCH_EPOCHS_DEFAULT: Final[int] = 200

#: Default PyTorch MLP learning rate [dimensionless].
PYTORCH_LR_DEFAULT: Final[float] = 1e-3

#: Default top-N feature importances to retain for display [dimensionless].
IMPORTANCE_TOP_N_DEFAULT: Final[int] = 20


# ---------------------------------------------------------------------------
# _PreparedData — internal data container (not frozen: holds numpy arrays)
# ---------------------------------------------------------------------------

@dataclass
class _PreparedData:
    """Prepared train/test split ready for model fitting."""
    X_train:       np.ndarray
    X_test:        np.ndarray
    y_train:       np.ndarray
    y_test:        np.ndarray
    feature_names: Tuple[str, ...]
    is_clf:        bool
    n_classes:     int
    label_encoder: Optional[LabelEncoder]


# ---------------------------------------------------------------------------
# MLResult — typed, immutable output record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLResult:
    """
    Typed output of a single ML model training and evaluation run.

    Fields
    ------
    method_key    : machine identifier matching _DISPATCH key
    method_title  : human-readable method name
    task_type     : "regression" | "classification"
    target_col    : name of the target column used
    metrics       : regression: r2/rmse/mae; classification: accuracy/f1/precision/recall
    y_test        : held-out true target values
    y_pred        : model predictions on the test set
    feature_names : tuple of feature column names used
    extra         : importances (pd.Series), confusion_matrix, roc_fpr/tpr/auc,
                    loss_curve, class_names, loss_curve
    is_valid      : False if computation degraded
    """
    method_key:    str
    method_title:  str
    task_type:     str
    target_col:    str
    metrics:       Dict[str, float]
    y_test:        np.ndarray
    y_pred:        np.ndarray
    feature_names: Tuple[str, ...]
    extra:         Dict[str, Any] = field(default_factory=dict)
    is_valid:      bool = True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _prepare(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
) -> _PreparedData:
    """
    Shared data preparation: NaN removal, task detection, encoding, splitting, scaling.

    Parameters
    ----------
    task_type : "auto" | "classification" | "regression"
        "auto" detects from target cardinality (≤ CLASSIFICATION_MAX_UNIQUE unique
        integer-like values → classification).
    """
    sub = df[feature_cols + [target_col]].dropna()
    if len(sub) < MIN_VALID_ROWS:
        raise ValueError(
            f"Only {len(sub)} valid rows after NaN removal (need ≥ {MIN_VALID_ROWS})."
        )

    X      = sub[feature_cols].values.astype(float)
    y_raw  = sub[target_col]

    # ── Task-type detection ───────────────────────────────────────────────
    if task_type == "auto":
        is_numeric = pd.api.types.is_numeric_dtype(y_raw)
        is_int_like = is_numeric and np.all(
            y_raw.dropna().values == y_raw.dropna().values.astype(np.int64)
        )
        is_clf = (
            not is_numeric
            or (y_raw.nunique() <= CLASSIFICATION_MAX_UNIQUE and is_int_like)
        )
    elif task_type == "classification":
        is_clf = True
    else:
        is_clf = False

    # ── Target encoding ───────────────────────────────────────────────────
    label_encoder: Optional[LabelEncoder] = None
    if is_clf:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str)).astype(np.int64)
        n_classes = int(len(label_encoder.classes_))
    else:
        y = y_raw.values.astype(float)
        n_classes = 1

    # ── Train/test split ─────────────────────────────────────────────────
    stratify = y if is_clf else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=stratify,
        )
    except ValueError:
        # Fallback: no stratification when a class has too few samples
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs,
        )

    # ── Feature scaling ───────────────────────────────────────────────────
    if standardize:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    return _PreparedData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=tuple(feature_cols),
        is_clf=is_clf, n_classes=n_classes,
        label_encoder=label_encoder,
    )


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """R², RMSE, MAE for regression evaluation."""
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
    }


def _clf_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """Accuracy, F1, precision, recall for classification evaluation."""
    avg = "macro" if n_classes > 2 else "binary"
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def _extract_importances(
    model: Any,
    feature_names: Tuple[str, ...],
) -> Optional[pd.Series]:
    """Extract feature importances or |coefficients| from a fitted model."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            model.feature_importances_, index=list(feature_names), name="importance"
        )
    if hasattr(model, "coef_"):
        coef = np.array(model.coef_).ravel()
        if len(coef) == len(feature_names):
            return pd.Series(
                np.abs(coef), index=list(feature_names), name="importance"
            )
    return None


def _roc_data(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> Dict[str, Any]:
    """Compute ROC curve data for binary classifiers that support predict_proba."""
    if n_classes != 2:
        return {}
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            return {}
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = float(roc_auc_score(y_test, y_score))
        return {"roc_fpr": fpr, "roc_tpr": tpr, "roc_auc": auc}
    except Exception:
        return {}


def _fit_and_eval(
    pd_: _PreparedData,
    model: Any,
    method_key: str,
    method_title: str,
    target_col: str,
) -> MLResult:
    """
    Shared fit → predict → metrics → package kernel.

    Single site that calls model.fit, model.predict, computes all metrics,
    extracts importances, ROC data, and loss curves.
    """
    model.fit(pd_.X_train, pd_.y_train)
    y_pred = model.predict(pd_.X_test)

    imp  = _extract_importances(model, pd_.feature_names)
    _ts = getattr(model, "train_score_", None)
    _lc = getattr(model, "loss_curve_",  None)
    if _ts is not None:
        loss: Optional[List[float]] = list(_ts)
    elif _lc is not None:
        loss = list(_lc)
    else:
        loss = None

    # Compute model state hash for citable records (DNV-2.0 Slide 11).
    try:
        import hashlib as _hl, pickle as _pk
        _model_hash: Optional[str] = _hl.sha256(_pk.dumps(model)).hexdigest()
    except Exception:
        _model_hash = None

    if pd_.is_clf:
        metrics = _clf_metrics(pd_.y_test, y_pred, pd_.n_classes)
        cm      = confusion_matrix(pd_.y_test, y_pred)
        roc     = _roc_data(model, pd_.X_test, pd_.y_test, pd_.n_classes)
        extra: Dict[str, Any] = {
            "importances":      imp,
            "confusion_matrix": cm,
            "class_names":      (
                list(pd_.label_encoder.classes_) if pd_.label_encoder else None
            ),
            "loss_curve":       loss,
            "model_state_hash": _model_hash,
            **roc,
        }
        task_type = "classification"
    else:
        metrics = _reg_metrics(pd_.y_test, y_pred)
        extra = {
            "importances":  imp,
            "loss_curve":   loss,
            "model_state_hash": _model_hash,
        }
        task_type = "regression"

    log.info(
        "_fit_and_eval [%s]: task=%s  metrics=%s",
        method_key, task_type,
        {k: f"{v:.4f}" for k, v in metrics.items()},
    )

    return MLResult(
        method_key=method_key,
        method_title=method_title,
        task_type=task_type,
        target_col=target_col,
        metrics=metrics,
        y_test=pd_.y_test,
        y_pred=y_pred,
        feature_names=pd_.feature_names,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Sentinel helper
# ---------------------------------------------------------------------------

def _depth(max_depth: int) -> Optional[int]:
    """Convert MAX_DEPTH_UNLIMITED_SENTINEL (0) → None for sklearn."""
    return None if max_depth == MAX_DEPTH_UNLIMITED_SENTINEL else max_depth


# ---------------------------------------------------------------------------
# 1. Linear Regression (regression only)
# ---------------------------------------------------------------------------

def compute_linear_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """OLS linear regression — regression targets only."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    if pd_.is_clf:
        raise ValueError(
            "linear_regression requires a continuous target. "
            "Set task_type='regression' or use logistic_regression."
        )
    return _fit_and_eval(
        pd_, LinearRegression(),
        "linear_regression", "Linear Regression", target_col,
    )


# ---------------------------------------------------------------------------
# 2. Logistic Regression (classification only)
# ---------------------------------------------------------------------------

def compute_logistic_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Logistic regression — classification targets only."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    if not pd_.is_clf:
        raise ValueError(
            "logistic_regression requires a categorical target. "
            "Set task_type='classification' or use linear_regression."
        )
    model = LogisticRegression(
        C=C, max_iter=max_iter, solver=solver, random_state=rs,
    )
    return _fit_and_eval(pd_, model, "logistic_regression", "Logistic Regression", target_col)


# ---------------------------------------------------------------------------
# 3. Ridge (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_ridge(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    alpha: float = 1.0,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Ridge regression or Ridge classifier (auto-detected)."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    model = RidgeClassifier(alpha=alpha) if pd_.is_clf else Ridge(alpha=alpha)
    return _fit_and_eval(pd_, model, "ridge", "Ridge", target_col)


# ---------------------------------------------------------------------------
# 4. Lasso (regression only)
# ---------------------------------------------------------------------------

def compute_lasso(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    alpha: float = 0.01,
    max_iter: int = 1000,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Lasso regression — regression targets only."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    if pd_.is_clf:
        raise ValueError(
            "lasso requires a continuous target. Set task_type='regression'."
        )
    return _fit_and_eval(
        pd_, Lasso(alpha=alpha, max_iter=max_iter),
        "lasso", "Lasso", target_col,
    )


# ---------------------------------------------------------------------------
# 5. Decision Tree (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_decision_tree(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_depth: int = MAX_DEPTH_UNLIMITED_SENTINEL,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """CART decision tree — auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    d   = _depth(max_depth)
    model = (
        DecisionTreeClassifier(max_depth=d, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, random_state=rs)
        if pd_.is_clf else
        DecisionTreeRegressor(max_depth=d, min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf, random_state=rs)
    )
    return _fit_and_eval(pd_, model, "decision_tree", "Decision Tree", target_col)


# ---------------------------------------------------------------------------
# 6. Random Forest (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_random_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = N_ESTIMATORS_DEFAULT,
    max_depth: int = MAX_DEPTH_UNLIMITED_SENTINEL,
    min_samples_split: int = 2,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Random forest — bagged CART trees, auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    d   = _depth(max_depth)
    model = (
        RandomForestClassifier(n_estimators=n_estimators, max_depth=d,
                               min_samples_split=min_samples_split, random_state=rs, n_jobs=-1)
        if pd_.is_clf else
        RandomForestRegressor(n_estimators=n_estimators, max_depth=d,
                              min_samples_split=min_samples_split, random_state=rs, n_jobs=-1)
    )
    return _fit_and_eval(pd_, model, "random_forest", "Random Forest", target_col)


# ---------------------------------------------------------------------------
# 7. Extra Trees (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_extra_trees(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = N_ESTIMATORS_DEFAULT,
    max_depth: int = MAX_DEPTH_UNLIMITED_SENTINEL,
    min_samples_split: int = 2,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Extremely randomised trees — auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    d   = _depth(max_depth)
    model = (
        ExtraTreesClassifier(n_estimators=n_estimators, max_depth=d,
                             min_samples_split=min_samples_split, random_state=rs, n_jobs=-1)
        if pd_.is_clf else
        ExtraTreesRegressor(n_estimators=n_estimators, max_depth=d,
                            min_samples_split=min_samples_split, random_state=rs, n_jobs=-1)
    )
    return _fit_and_eval(pd_, model, "extra_trees", "Extra Trees", target_col)


# ---------------------------------------------------------------------------
# 8. Gradient Boosting (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_gradient_boosting(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = N_ESTIMATORS_DEFAULT,
    learning_rate: float = GB_LEARNING_RATE_DEFAULT,
    max_depth: int = 3,
    subsample: float = 1.0,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """sklearn Gradient Boosting — staged training, auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    model = (
        GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                   max_depth=max_depth, subsample=subsample, random_state=rs)
        if pd_.is_clf else
        GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                  max_depth=max_depth, subsample=subsample, random_state=rs)
    )
    return _fit_and_eval(pd_, model, "gradient_boosting", "Gradient Boosting", target_col)


# ---------------------------------------------------------------------------
# 9. AdaBoost (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_adaboost(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """AdaBoost — adaptive sample reweighting, auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    model = (
        AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                           random_state=rs, algorithm="SAMME")
        if pd_.is_clf else
        AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                          random_state=rs)
    )
    return _fit_and_eval(pd_, model, "adaboost", "AdaBoost", target_col)


# ---------------------------------------------------------------------------
# 10. SVM (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_svm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    epsilon: float = 0.1,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Support Vector Machine — RBF kernel by default, auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    # Parse gamma: accept "scale", "auto", or float string
    try:
        g: Any = float(gamma)
    except (ValueError, TypeError):
        g = gamma
    model = (
        SVC(C=C, kernel=kernel, gamma=g, probability=True, random_state=rs)
        if pd_.is_clf else
        SVR(C=C, kernel=kernel, gamma=g, epsilon=epsilon)
    )
    return _fit_and_eval(pd_, model, "svm", "SVM", target_col)


# ---------------------------------------------------------------------------
# 11. K-Nearest Neighbours (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_knn(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_neighbors: int = KNN_N_NEIGHBORS_DEFAULT,
    weights: str = "uniform",
    metric: str = "minkowski",
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """k-Nearest Neighbours — distance voting, auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    model = (
        KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        if pd_.is_clf else
        KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
    )
    return _fit_and_eval(pd_, model, "knn", "K-Nearest Neighbours", target_col)


# ---------------------------------------------------------------------------
# 12. MLP — sklearn (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_mlp(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    hidden_sizes: str = MLP_HIDDEN_DEFAULT,
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 1e-4,
    max_iter: int = MLP_MAX_ITER_DEFAULT,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """sklearn MLP neural network — Adam optimiser, auto-detected task type."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    try:
        h = tuple(int(x.strip()) for x in hidden_sizes.split(",") if x.strip())
    except ValueError:
        raise ValueError(
            f"Invalid hidden_sizes format: {hidden_sizes!r} — use comma-separated integers."
        )
    model = (
        MLPClassifier(hidden_layer_sizes=h, activation=activation, solver=solver,
                      alpha=alpha, max_iter=max_iter, random_state=rs)
        if pd_.is_clf else
        MLPRegressor(hidden_layer_sizes=h, activation=activation, solver=solver,
                     alpha=alpha, max_iter=max_iter, random_state=rs)
    )
    return _fit_and_eval(pd_, model, "mlp", "MLP (sklearn)", target_col)


# ---------------------------------------------------------------------------
# 13. PyTorch MLP (auto clf / reg)
# ---------------------------------------------------------------------------

def compute_pytorch_mlp(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    hidden_sizes: str = MLP_HIDDEN_DEFAULT,
    activation: str = "relu",
    n_epochs: int = PYTORCH_EPOCHS_DEFAULT,
    lr: float = PYTORCH_LR_DEFAULT,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """PyTorch MLP — configurable depth, GPU-capable, auto-detected task type."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for pytorch_mlp.\n"
            "Install: conda install pytorch -c pytorch"
        )

    torch.manual_seed(rs)
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)

    try:
        h_sizes = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
    except ValueError:
        raise ValueError(
            f"Invalid hidden_sizes format: {hidden_sizes!r} — use comma-separated integers."
        )

    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    act_cls = act_map.get(activation, nn.ReLU)

    n_in  = pd_.X_train.shape[1]
    n_out = pd_.n_classes if (pd_.is_clf and pd_.n_classes > 2) else 1
    sizes = [n_in] + h_sizes + [n_out]

    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act_cls())
    net = nn.Sequential(*layers).float()

    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    X_tr = torch.FloatTensor(pd_.X_train)
    y_tr = torch.FloatTensor(pd_.y_train)

    if pd_.is_clf:
        loss_fn: nn.Module = (
            nn.BCEWithLogitsLoss() if pd_.n_classes == 2 else nn.CrossEntropyLoss()
        )
        if pd_.n_classes > 2:
            y_tr = y_tr.long()
    else:
        loss_fn = nn.MSELoss()

    loss_hist: List[float] = []
    net.train()
    for _ in range(n_epochs):
        optimiser.zero_grad()
        out  = net(X_tr).squeeze()
        loss = loss_fn(out, y_tr)
        loss.backward()
        optimiser.step()
        loss_hist.append(float(loss.item()))

    net.eval()
    with torch.no_grad():
        out_te = net(torch.FloatTensor(pd_.X_test)).squeeze()
        if pd_.is_clf:
            if pd_.n_classes == 2:
                y_pred = (torch.sigmoid(out_te) > 0.5).long().numpy()
            else:
                y_pred = torch.argmax(out_te, dim=1).numpy()
        else:
            y_pred = out_te.numpy()

    if pd_.is_clf:
        metrics  = _clf_metrics(pd_.y_test, y_pred, pd_.n_classes)
        cm       = confusion_matrix(pd_.y_test, y_pred)
        extra: Dict[str, Any] = {
            "importances":      None,
            "confusion_matrix": cm,
            "class_names":      (
                list(pd_.label_encoder.classes_) if pd_.label_encoder else None
            ),
            "loss_curve":       loss_hist,
        }
        task_type_str = "classification"
    else:
        metrics = _reg_metrics(pd_.y_test, y_pred)
        extra = {"importances": None, "loss_curve": loss_hist}
        task_type_str = "regression"

    log.info(
        "compute_pytorch_mlp: task=%s  epochs=%d  final_loss=%.6f  metrics=%s",
        task_type_str, n_epochs, loss_hist[-1] if loss_hist else float("nan"),
        {k: f"{v:.4f}" for k, v in metrics.items()},
    )

    return MLResult(
        method_key="pytorch_mlp",
        method_title="PyTorch MLP",
        task_type=task_type_str,
        target_col=target_col,
        metrics=metrics,
        y_test=pd_.y_test,
        y_pred=y_pred,
        feature_names=pd_.feature_names,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 14. Gaussian Naive Bayes (classification only)
# ---------------------------------------------------------------------------

def compute_naive_bayes(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    var_smoothing: float = 1e-9,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Gaussian Naive Bayes — class-conditional Gaussian, classification only."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    if not pd_.is_clf:
        raise ValueError(
            "naive_bayes requires a categorical target. "
            "Set task_type='classification'."
        )
    return _fit_and_eval(
        pd_, GaussianNB(var_smoothing=var_smoothing),
        "naive_bayes", "Naive Bayes", target_col,
    )


# ---------------------------------------------------------------------------
# 15. Bayesian Ridge (regression only)
# ---------------------------------------------------------------------------

def compute_bayesian_ridge(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_iter: int = 300,
    tol: float = 1e-3,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """Bayesian Ridge — ARD prior, probabilistic weights, regression only."""
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    if pd_.is_clf:
        raise ValueError(
            "bayesian_ridge requires a continuous target. "
            "Set task_type='regression'."
        )
    return _fit_and_eval(
        pd_, BayesianRidge(max_iter=max_iter, tol=tol),
        "bayesian_ridge", "Bayesian Ridge", target_col,
    )


# ---------------------------------------------------------------------------
# 16. XGBoost (optional dependency)
# ---------------------------------------------------------------------------

def compute_xgboost(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = N_ESTIMATORS_DEFAULT,
    learning_rate: float = GB_LEARNING_RATE_DEFAULT,
    max_depth: int = 3,
    subsample: float = 1.0,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """XGBoost — extreme gradient boosting, auto-detected task type."""
    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "XGBoost is not installed.\n"
            "Install: conda install -c conda-forge xgboost"
        )
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    common = dict(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, subsample=subsample,
        random_state=rs, n_jobs=-1, verbosity=0,
    )
    model = XGBClassifier(**common) if pd_.is_clf else XGBRegressor(**common)
    return _fit_and_eval(pd_, model, "xgboost", "XGBoost", target_col)


# ---------------------------------------------------------------------------
# 17. LightGBM (optional dependency)
# ---------------------------------------------------------------------------

def compute_lightgbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_estimators: int = N_ESTIMATORS_DEFAULT,
    learning_rate: float = GB_LEARNING_RATE_DEFAULT,
    max_depth: int = MAX_DEPTH_UNLIMITED_SENTINEL,
    num_leaves: int = 31,
    standardize: bool = True,
    test_size: float = TEST_SIZE_DEFAULT,
    rs: int = RANDOM_STATE_DEFAULT,
    task_type: str = "auto",
    **_: Any,
) -> MLResult:
    """LightGBM — leaf-wise histogram boosting, auto-detected task type."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "LightGBM is not installed.\n"
            "Install: conda install -c conda-forge lightgbm"
        )
    pd_ = _prepare(df, feature_cols, target_col, standardize, test_size, rs, task_type)
    d = _depth(max_depth)
    common = dict(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=d if d is not None else -1,
        num_leaves=num_leaves, random_state=rs, n_jobs=-1, verbose=-1,
    )
    model = LGBMClassifier(**common) if pd_.is_clf else LGBMRegressor(**common)
    return _fit_and_eval(pd_, model, "lightgbm", "LightGBM", target_col)


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

_DISPATCH: Dict[str, Tuple[str, str, Any]] = {
    # key                    title                    category                 callable
    "linear_regression":  ("Linear Regression",      "LINEAR & REGULARISED",  compute_linear_regression),
    "logistic_regression":("Logistic Regression",    "LINEAR & REGULARISED",  compute_logistic_regression),
    "ridge":              ("Ridge",                  "LINEAR & REGULARISED",  compute_ridge),
    "lasso":              ("Lasso",                  "LINEAR & REGULARISED",  compute_lasso),
    "decision_tree":      ("Decision Tree",          "TREE-BASED",            compute_decision_tree),
    "random_forest":      ("Random Forest",          "TREE-BASED",            compute_random_forest),
    "extra_trees":        ("Extra Trees",            "TREE-BASED",            compute_extra_trees),
    "gradient_boosting":  ("Gradient Boosting",      "ENSEMBLE",              compute_gradient_boosting),
    "adaboost":           ("AdaBoost",               "ENSEMBLE",              compute_adaboost),
    "svm":                ("SVM",                    "KERNEL & NEIGHBOURS",   compute_svm),
    "knn":                ("K-Nearest Neighbours",   "KERNEL & NEIGHBOURS",   compute_knn),
    "mlp":                ("MLP (sklearn)",          "NEURAL NETWORK",        compute_mlp),
    "pytorch_mlp":        ("PyTorch MLP",            "NEURAL NETWORK",        compute_pytorch_mlp),
    "naive_bayes":        ("Naive Bayes",            "PROBABILISTIC",         compute_naive_bayes),
    "bayesian_ridge":     ("Bayesian Ridge",         "PROBABILISTIC",         compute_bayesian_ridge),
    "xgboost":            ("XGBoost",                "GRADIENT BOOSTING",     compute_xgboost),
    "lightgbm":           ("LightGBM",               "GRADIENT BOOSTING",     compute_lightgbm),
}
