"""
DNV Scientific Module
---------------------
Role:
    Provides the MLDialog — a fully configurable, canvas-embedded dialog for
    machine learning model training, evaluation, and visualisation, with
    exhaustive per-method options, a shared style panel, and a complete
    export/save panel.

Scientific Context:
    Accepts a method key and a SessionStore; collects feature selections and
    method hyperparameters from the operator; dispatches to ml_engine for
    model fitting and evaluation; renders the result as a three-panel figure
    (primary metric plot + secondary diagnostic + feature importances) with
    full style and export control.

Invariants:
    - No computation occurs at dialog construction time; only on Generate.
    - Every numeric constant is a named Final with a unit suffix.
    - _apply_style() is the single site that writes per-axes matplotlib state.
    - The single FigureCanvas is replaced atomically on each Generate call.

Assumptions:
    - store.asset is a valid DataAsset with a DataFrame containing ≥ 2
      numeric columns.
    - ml_engine functions are pure and raise ImportError for missing optional deps.
    - matplotlib backend is "Agg".

Failure Modes:
    - Fewer than MIN_VALID_ROWS rows after NaN removal: ValueError caught.
    - XGBoost / LightGBM not installed: ImportError caught and shown as hint.
    - Task mismatch (e.g. linear_regression on clf target): ValueError caught.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore    import Qt, Signal
from PySide6.QtGui     import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from gui_session import SessionStore
from gui_style import (
    make_style_panel, make_export_panel,
    apply_style, finalise_figure, rc_ctx,
)
import ml_engine as _eng
import stability_engine as _stab
import audit_engine as _audit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout and display constants
# ---------------------------------------------------------------------------

#: Fixed width of the left configuration panel [px].
LEFT_PANEL_WIDTH_PX: Final[int] = 340

#: Minimum height of the canvas area [px].
CANVAS_MIN_HEIGHT_PX: Final[int] = 500

#: Minimum width of the canvas area [px].
CANVAS_MIN_WIDTH_PX: Final[int] = 800

#: Default figure width [inches].
DEFAULT_FIG_WIDTH_IN: Final[float] = 18.0

#: Default figure height [inches].
DEFAULT_FIG_HEIGHT_IN: Final[float] = 6.0

#: Default export DPI.
DEFAULT_EXPORT_DPI: Final[int] = 150

#: Default screen rendering DPI.
DEFAULT_SCREEN_DPI: Final[int] = 100

#: Spacing after section separators in left panel [px].
SECTION_SPACING_PX: Final[int] = 4

#: Maximum feature name label length for importance bar chart [characters].
IMPORTANCE_LABEL_MAX_CHARS: Final[int] = 40

_COLORMAPS: Final[List[str]] = [
    "Blues", "YlOrRd", "viridis", "plasma", "coolwarm", "RdBu",
    "Greens", "Oranges", "Purples",
]
_FONT_FAMILIES: Final[List[str]] = [
    "sans-serif", "serif", "monospace",
    "Segoe UI", "Arial", "Helvetica", "Times New Roman", "DejaVu Sans",
]
_LEGEND_LOCS: Final[List[str]] = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
]
_SCALES: Final[List[str]] = ["linear", "log", "symlog"]
_LINESTYLES: Final[List[str]] = ["--", "-", ":", "-."]
_SVM_KERNELS: Final[List[str]] = ["rbf", "linear", "poly", "sigmoid"]
_KNN_WEIGHTS: Final[List[str]] = ["uniform", "distance"]
_KNN_METRICS: Final[List[str]] = ["minkowski", "euclidean", "manhattan", "chebyshev"]
_MLP_ACTIVATIONS: Final[List[str]] = ["relu", "tanh", "logistic", "identity"]
_MLP_SOLVERS: Final[List[str]] = ["adam", "sgd", "lbfgs"]
_LR_SOLVERS: Final[List[str]] = ["lbfgs", "saga", "newton-cg", "liblinear"]
_TASK_TYPES: Final[List[str]] = ["auto", "regression", "classification"]
_SECONDARY_MODES: Final[List[str]] = ["residuals / metrics", "loss curve"]

#: Stability bar color for family-stable metrics.
STAB_BAR_STABLE_COLOR: Final[str] = "#27ae60"

#: Stability bar color for contingent metrics.
STAB_BAR_CONTINGENT_COLOR: Final[str] = "#e74c3c"

#: Maximum height of the stability text report panel [px].
STAB_REPORT_MAX_HEIGHT_PX: Final[int] = 120

#: Default figure width for stability diagnostic plots [inches].
STAB_FIG_WIDTH_IN: Final[float] = 14.0

#: Default figure height for stability diagnostic plots [inches].
STAB_FIG_HEIGHT_IN: Final[float] = 6.0

#: Default figure width for surrogate audit parity plot [inches].
AUDIT_FIG_WIDTH_IN: Final[float] = 12.0

#: Default figure height for surrogate audit parity plot [inches].
AUDIT_FIG_HEIGHT_IN: Final[float] = 5.5

#: Parity scatter color when target is proxy-recoverable.
AUDIT_RECOVERABLE_COLOR: Final[str] = "#e74c3c"

#: Parity scatter color when target is NOT proxy-recoverable.
AUDIT_SAFE_COLOR: Final[str] = "#27ae60"

#: Minimum width for the surrogate audit dialog [px].
AUDIT_DIALOG_MIN_WIDTH_PX: Final[int] = 420

#: Minimum height for the surrogate audit dialog [px].
AUDIT_DIALOG_MIN_HEIGHT_PX: Final[int] = 360


# ---------------------------------------------------------------------------
# _ColorBtn
# ---------------------------------------------------------------------------

class _ColorBtn(QPushButton):
    """Compact colour-picker button.  Emits color_changed(str) on selection."""
    color_changed = Signal(str)

    def __init__(self, initial: str = "#3a7bd5", parent=None):
        super().__init__(parent)
        self._color = initial
        self._refresh()
        self.setFixedWidth(54)
        self.clicked.connect(self._pick)

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if c.isValid():
            self._color = c.name()
            self._refresh()
            self.color_changed.emit(self._color)

    def _refresh(self) -> None:
        self.setStyleSheet(
            f"background:{self._color};border:1px solid #555;border-radius:3px;"
        )

    def color(self) -> str:
        return self._color

    def set_color(self, c: str) -> None:
        self._color = c; self._refresh()


# ---------------------------------------------------------------------------
# Widget micro-helpers
# ---------------------------------------------------------------------------

def _combo(items: List[str], default: str = "") -> QComboBox:
    w = QComboBox(); w.addItems(items)
    idx = items.index(default) if default in items else 0
    w.setCurrentIndex(idx); return w

def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w

def _dspin(lo: float, hi: float, val: float, step: float = 0.05) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step); w.setDecimals(6)
    return w

def _check(state: bool, label: str = "") -> QCheckBox:
    w = QCheckBox(label); w.setChecked(state); return w

def _hrow(label: str, widget: QWidget) -> QWidget:
    row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label); lbl.setMinimumWidth(130)
    lay.addWidget(lbl); lay.addWidget(widget, 1); return row

def _eyebrow(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("ReportPanelEyebrow"); return lbl

def _section_sep() -> QFrame:
    f = QFrame(); f.setObjectName("Divider")
    f.setFixedHeight(1); f.setFrameShape(QFrame.HLine); return f

def _scrolled(inner: QWidget) -> QScrollArea:
    sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(inner)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); return sa

def _read(w: Any) -> Any:
    if isinstance(w, QComboBox):      return w.currentText()
    if isinstance(w, QCheckBox):      return w.isChecked()
    if isinstance(w, QSpinBox):       return w.value()
    if isinstance(w, QDoubleSpinBox): return w.value()
    if isinstance(w, _ColorBtn):      return w.color()
    if isinstance(w, QLineEdit):      return w.text().strip()
    return None


# ---------------------------------------------------------------------------
# _MultiColSelector
# ---------------------------------------------------------------------------

class _MultiColSelector(QListWidget):
    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.addItem(item)
        self.setMaximumHeight(160)

    def selected(self) -> List[str]:
        return [self.item(i).text() for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked]

    def select_all(self) -> None:
        for i in range(self.count()): self.item(i).setCheckState(Qt.Checked)

    def deselect_all(self) -> None:
        for i in range(self.count()): self.item(i).setCheckState(Qt.Unchecked)


# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------

class _MLDataPage(QWidget):
    """
    Data tab for MLDialog.

    Provides: target column selector, feature multi-selector, test size,
    random state, standardize flag, and task-type override.
    """

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(4)

        num_cols = list(df.select_dtypes(include="number").columns)
        all_cols = list(df.columns)

        lay.addWidget(_eyebrow("TARGET COLUMN"))
        self._target = _combo(all_cols)
        lay.addWidget(_hrow("Target:", self._target))

        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("FEATURE COLUMNS"))
        self._sel = _MultiColSelector(num_cols)
        lay.addWidget(self._sel)
        br = QHBoxLayout()
        ba = QPushButton("All"); bn = QPushButton("None")
        ba.clicked.connect(self._sel.select_all)
        bn.clicked.connect(self._sel.deselect_all)
        br.addWidget(ba); br.addWidget(bn); br.addStretch()
        lay.addLayout(br)

        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("TRAINING SETTINGS"))
        self._test_size  = _dspin(0.05, 0.5, 0.2, 0.05)
        self._rs         = _spin(0, 9999, 42)
        self._standardize = _check(True)
        self._task_type  = _combo(_TASK_TYPES, "auto")
        lay.addWidget(_hrow("Test size:",    self._test_size))
        lay.addWidget(_hrow("Random state:", self._rs))
        lay.addWidget(_hrow("Standardize X:", self._standardize))
        lay.addWidget(_hrow("Task type:",    self._task_type))

        lay.addStretch()

    def target(self) -> str:       return self._target.currentText()
    def test_size(self) -> float:  return float(self._test_size.value())
    def random_state(self) -> int: return int(self._rs.value())
    def standardize(self) -> bool: return self._standardize.isChecked()
    def task_type(self) -> str:    return self._task_type.currentText()

    def features(self) -> List[str]:
        return [c for c in self._sel.selected() if c != self.target()]


# ---------------------------------------------------------------------------
# Options panel — per-method controls
# ---------------------------------------------------------------------------

def _make_options(key: str) -> Tuple[QWidget, Callable[[], dict]]:
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
    controls: Dict[str, Any] = {}

    def _add(label: str, widget: Any, k: str) -> None:
        controls[k] = widget; lay.addWidget(_hrow(label, widget))

    def _sec(text: str) -> None:
        lay.addWidget(_section_sep()); lay.addWidget(_eyebrow(text))
        lay.addSpacing(SECTION_SPACING_PX)

    # ── Plot controls (shared) ─────────────────────────────────────────────
    _sec("SCATTER / BAR DISPLAY")
    _add("Scatter color:",     _ColorBtn("#3a7bd5"),                           "scatter_color")
    _add("Scatter alpha:",     _dspin(0.1, 1.0, 0.65, 0.05),                  "scatter_alpha")
    _add("Marker size:",       _spin(2, 20, 6),                                "marker_size")
    _add("Bar color:",         _ColorBtn("#3a7bd5"),                           "bar_color")
    _add("Bar alpha:",         _dspin(0.3, 1.0, 0.85, 0.05),                  "bar_alpha")
    _add("Identity line color:",_ColorBtn("#e74c3c"),                          "ideal_color")
    _add("Top N importances:", _spin(1, 100, 20),                              "top_n")
    _add("Confusion cmap:",    _combo(_COLORMAPS, "Blues"),                    "cm_cmap")
    _add("Secondary panel:",   _combo(_SECONDARY_MODES),                      "secondary_mode")

    # ── Method-specific controls ───────────────────────────────────────────
    if key == "logistic_regression":
        _sec("LOGISTIC REGRESSION")
        _add("C (inv. reg.):",  _dspin(1e-4, 1e4, 1.0, 0.5),                  "C")
        _add("Max iter:",       _spin(100, 10000, 1000),                       "max_iter")
        _add("Solver:",         _combo(_LR_SOLVERS, "lbfgs"),                  "solver")

    elif key == "ridge":
        _sec("RIDGE")
        _add("Alpha:",          _dspin(1e-4, 1e6, 1.0, 0.5),                  "alpha")

    elif key == "lasso":
        _sec("LASSO")
        _add("Alpha:",          _dspin(1e-6, 100.0, 0.01, 0.005),             "alpha")
        _add("Max iter:",       _spin(100, 10000, 1000),                       "max_iter")

    elif key == "decision_tree":
        _sec("DECISION TREE")
        _add("Max depth (0=∞):",_spin(0, 50, 0),                              "max_depth")
        _add("Min samples split:",_spin(2, 50, 2),                            "min_samples_split")
        _add("Min samples leaf:", _spin(1, 50, 1),                            "min_samples_leaf")

    elif key in ("random_forest", "extra_trees"):
        _sec("ENSEMBLE SETTINGS")
        _add("N estimators:",   _spin(10, 1000, 100),                          "n_estimators")
        _add("Max depth (0=∞):",_spin(0, 50, 0),                              "max_depth")
        _add("Min samples split:",_spin(2, 50, 2),                            "min_samples_split")

    elif key == "gradient_boosting":
        _sec("GRADIENT BOOSTING")
        _add("N estimators:",   _spin(10, 1000, 100),                          "n_estimators")
        _add("Learning rate:",  _dspin(1e-4, 1.0, 0.1, 0.01),                 "learning_rate")
        _add("Max depth:",      _spin(1, 15, 3),                               "max_depth")
        _add("Subsample:",      _dspin(0.1, 1.0, 1.0, 0.05),                  "subsample")

    elif key == "adaboost":
        _sec("ADABOOST")
        _add("N estimators:",   _spin(10, 500, 50),                            "n_estimators")
        _add("Learning rate:",  _dspin(0.01, 2.0, 1.0, 0.05),                 "learning_rate")

    elif key == "svm":
        _sec("SVM")
        _add("C:",              _dspin(1e-3, 1e4, 1.0, 0.5),                   "C")
        _add("Kernel:",         _combo(_SVM_KERNELS, "rbf"),                   "kernel")
        gam = QLineEdit("scale"); gam.setPlaceholderText("scale / auto / float")
        _add("Gamma:",          gam,                                            "gamma")
        _add("Epsilon (reg):",  _dspin(0.001, 1.0, 0.1, 0.01),                "epsilon")

    elif key == "knn":
        _sec("K-NEAREST NEIGHBOURS")
        _add("N neighbours:",   _spin(1, 100, 5),                              "n_neighbors")
        _add("Weights:",        _combo(_KNN_WEIGHTS, "uniform"),               "weights")
        _add("Metric:",         _combo(_KNN_METRICS, "minkowski"),             "metric")

    elif key == "mlp":
        _sec("MLP — sklearn")
        hl = QLineEdit("100,50"); hl.setPlaceholderText("e.g. 100,50,25")
        _add("Hidden layers:",  hl,                                             "hidden_sizes")
        _add("Activation:",     _combo(_MLP_ACTIVATIONS, "relu"),              "activation")
        _add("Solver:",         _combo(_MLP_SOLVERS, "adam"),                  "solver")
        _add("Alpha (L2):",     _dspin(1e-6, 0.1, 1e-4, 1e-5),               "alpha")
        _add("Max iter:",       _spin(50, 2000, 200),                          "max_iter")

    elif key == "pytorch_mlp":
        _sec("PYTORCH MLP")
        hl = QLineEdit("100,50"); hl.setPlaceholderText("e.g. 100,50,25")
        _add("Hidden layers:",  hl,                                             "hidden_sizes")
        _add("Activation:",     _combo(["relu", "tanh", "sigmoid"], "relu"),   "activation")
        _add("Epochs:",         _spin(10, 2000, 200),                          "n_epochs")
        _add("Learning rate:",  _dspin(1e-5, 0.1, 1e-3, 1e-4),               "lr")

    elif key == "naive_bayes":
        _sec("GAUSSIAN NAIVE BAYES")
        vs = QLineEdit("1e-9"); vs.setPlaceholderText("e.g. 1e-9")
        _add("Var smoothing:",  vs,                                             "var_smoothing")

    elif key == "bayesian_ridge":
        _sec("BAYESIAN RIDGE")
        _add("Max iter:",       _spin(100, 2000, 300),                         "max_iter")
        _add("Tolerance:",      _dspin(1e-8, 0.1, 1e-3, 1e-4),               "tol")

    elif key == "xgboost":
        _sec("XGBOOST")
        _add("N estimators:",   _spin(10, 1000, 100),                          "n_estimators")
        _add("Learning rate:",  _dspin(0.001, 0.5, 0.1, 0.01),               "learning_rate")
        _add("Max depth:",      _spin(1, 20, 3),                               "max_depth")
        _add("Subsample:",      _dspin(0.1, 1.0, 1.0, 0.05),                  "subsample")

    elif key == "lightgbm":
        _sec("LIGHTGBM")
        _add("N estimators:",   _spin(10, 1000, 100),                          "n_estimators")
        _add("Learning rate:",  _dspin(0.001, 0.5, 0.1, 0.01),               "learning_rate")
        _add("Max depth (0=∞):",_spin(0, 50, 0),                              "max_depth")
        _add("Num leaves:",     _spin(4, 512, 31),                             "num_leaves")

    lay.addStretch()

    def _getter() -> dict:
        raw = {k: _read(v) for k, v in controls.items()}
        # Parse scientific-notation fields submitted as QLineEdit
        for fld in ("var_smoothing", "tol", "alpha", "lr"):
            if fld in raw and isinstance(raw[fld], str):
                try:
                    raw[fld] = float(raw[fld])
                except (ValueError, TypeError):
                    pass
        return raw

    return w, _getter


# ---------------------------------------------------------------------------
# Sub-panel draw helpers
# ---------------------------------------------------------------------------

def _draw_parity(ax: Any, result: "_eng.MLResult", op: dict, sp: dict) -> None:
    """Parity plot: Actual vs Predicted with identity line and metric annotation."""
    sc = op.get("scatter_color", "#3a7bd5")
    sa = float(op.get("scatter_alpha", 0.65))
    ms = int(op.get("marker_size", 6))
    ic = op.get("ideal_color", "#e74c3c")

    y_t, y_p = result.y_test, result.y_pred
    ax.scatter(y_t, y_p, color=sc, alpha=sa, s=ms**2, linewidths=0, zorder=3)

    lo  = min(float(y_t.min()), float(y_p.min()))
    hi  = max(float(y_t.max()), float(y_p.max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=ic, linewidth=1.2, linestyle="--", label="Ideal", zorder=2)

    r2   = result.metrics.get("r2", float("nan"))
    rmse = result.metrics.get("rmse", float("nan"))
    mae  = result.metrics.get("mae", float("nan"))
    ax.text(0.05, 0.95,
            f"R\u00b2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=sp.get("tick_fontsize", 9),
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", alpha=0.85))

    ax.set_xlabel(f"Actual {result.target_col}", fontsize=sp.get("label_fontsize", 10))
    ax.set_ylabel(f"Predicted {result.target_col}", fontsize=sp.get("label_fontsize", 10))
    ax.set_title(f"{result.method_title} — Parity Plot",
                 fontsize=sp.get("title_fontsize", 12), fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def _draw_residuals(ax: Any, result: "_eng.MLResult", op: dict, sp: dict) -> None:
    """Residuals vs Predicted scatter with zero reference line."""
    sc = op.get("scatter_color", "#3a7bd5")
    sa = float(op.get("scatter_alpha", 0.65))
    ms = int(op.get("marker_size", 6))
    residuals = result.y_test - result.y_pred
    ax.scatter(result.y_pred, residuals,
               color=sc, alpha=sa, s=ms**2, linewidths=0, zorder=3)
    ax.axhline(0, color=op.get("ideal_color", "#e74c3c"),
               linewidth=1.2, linestyle="--", alpha=0.8, label="Zero residual")
    ax.set_xlabel(f"Predicted {result.target_col}", fontsize=sp.get("label_fontsize", 10))
    ax.set_ylabel("Residual (Actual − Predicted)", fontsize=sp.get("label_fontsize", 10))
    ax.set_title("Residuals vs Predicted",
                 fontsize=sp.get("title_fontsize", 12), fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def _draw_loss_curve(ax: Any, result: "_eng.MLResult", sp: dict) -> None:
    """Training loss / score curve if available in extra."""
    lc = result.extra.get("loss_curve")
    if lc and len(lc) > 0:
        ax.plot(range(1, len(lc) + 1), lc, color="#3a7bd5", linewidth=1.5)
        ax.set_xlabel("Iteration / Epoch", fontsize=sp.get("label_fontsize", 10))
        ax.set_ylabel("Loss / Score", fontsize=sp.get("label_fontsize", 10))
        ax.set_title("Training Curve",
                     fontsize=sp.get("title_fontsize", 12), fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No loss / score curve\navailable for this method.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")


def _draw_confusion(
    ax: Any, result: "_eng.MLResult", op: dict, sp: dict, fig: Figure,
) -> None:
    """Confusion matrix heatmap with cell annotations."""
    cm         = result.extra.get("confusion_matrix")
    class_names = result.extra.get("class_names") or []
    if cm is None:
        ax.text(0.5, 0.5, "No confusion matrix data.",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    cmap = op.get("cm_cmap", "Blues")
    im   = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = cm.shape[0]
    labels = class_names if len(class_names) == n else [str(i) for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right",
                                                 fontsize=sp.get("tick_fontsize", 9))
    ax.set_yticks(range(n)); ax.set_yticklabels(labels,
                                                 fontsize=sp.get("tick_fontsize", 9))

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=sp.get("tick_fontsize", 9),
                color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted", fontsize=sp.get("label_fontsize", 10))
    ax.set_ylabel("True",      fontsize=sp.get("label_fontsize", 10))
    ax.set_title(f"{result.method_title} — Confusion Matrix",
                 fontsize=sp.get("title_fontsize", 12), fontweight="bold")


def _draw_clf_secondary(
    ax: Any, result: "_eng.MLResult", op: dict, sp: dict,
) -> None:
    """ROC curve (binary) or metrics bar chart (multi-class)."""
    roc_fpr = result.extra.get("roc_fpr")
    roc_tpr = result.extra.get("roc_tpr")
    roc_auc = result.extra.get("roc_auc")

    sec_mode = op.get("secondary_mode", "residuals / metrics")
    if sec_mode == "loss curve":
        _draw_loss_curve(ax, result, sp)
        return

    if roc_fpr is not None and roc_tpr is not None:
        ax.plot(roc_fpr, roc_tpr, color=op.get("bar_color", "#3a7bd5"),
                linewidth=2.0, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=sp.get("label_fontsize", 10))
        ax.set_ylabel("True Positive Rate",  fontsize=sp.get("label_fontsize", 10))
        ax.set_title("ROC Curve", fontsize=sp.get("title_fontsize", 12), fontweight="bold")
        ax.legend(fontsize=sp.get("legend_fontsize", 9))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    else:
        m    = result.metrics
        keys = ["accuracy", "f1", "precision", "recall"]
        vals = [m.get(k, 0.0) for k in keys]
        bc   = op.get("bar_color", "#3a7bd5")
        ba   = float(op.get("bar_alpha", 0.85))
        bars = ax.bar(keys, vals, color=bc, alpha=ba, edgecolor="#1a3a6a", linewidth=0.5)
        ax.set_ylim(0, 1.08)
        for bar_, v in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=sp.get("tick_fontsize", 9))
        ax.set_ylabel("Score", fontsize=sp.get("label_fontsize", 10))
        ax.set_title("Classification Metrics",
                     fontsize=sp.get("title_fontsize", 12), fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def _draw_importances(ax: Any, result: "_eng.MLResult", op: dict, sp: dict) -> None:
    """Horizontal bar chart of feature importances or absolute coefficients."""
    imp = result.extra.get("importances")
    if imp is None or len(imp) == 0:
        ax.text(0.5, 0.5,
                "Feature importances\nnot available for\nthis method.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        return

    top_n = int(op.get("top_n", 20))
    top   = imp.nlargest(min(top_n, len(imp)))
    bc    = op.get("bar_color", "#3a7bd5")
    ba    = float(op.get("bar_alpha", 0.85))
    labels = [str(n)[:IMPORTANCE_LABEL_MAX_CHARS] for n in top.index]

    ax.barh(range(len(top)), top.values, color=bc, alpha=ba,
            edgecolor="#1a3a6a", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=sp.get("tick_fontsize", 9))
    ax.invert_yaxis()
    ax.axvline(0, color="#444", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Importance / |Coefficient|", fontsize=sp.get("label_fontsize", 10))
    ax.set_title(f"Top {len(top)} Feature Importances",
                 fontsize=sp.get("title_fontsize", 12), fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Main render functions
# ---------------------------------------------------------------------------

def _render_regression(
    result: "_eng.MLResult",
    op: dict,
    sp: dict,
) -> Figure:
    """
    Three-panel regression figure:
    Panel 1 — parity plot (Actual vs Predicted)
    Panel 2 — residuals vs predicted OR loss curve
    Panel 3 — feature importances
    """
    w_in = float(sp.get("width",  DEFAULT_FIG_WIDTH_IN))
    h_in = float(sp.get("height", DEFAULT_FIG_HEIGHT_IN))

    with plt.rc_context(rc_ctx(sp)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(w_in, h_in))

    _draw_parity(ax1, result, op, sp)

    sec_mode = op.get("secondary_mode", "residuals / metrics")
    if sec_mode == "loss curve":
        _draw_loss_curve(ax2, result, sp)
    else:
        _draw_residuals(ax2, result, op, sp)

    _draw_importances(ax3, result, op, sp)

    return finalise_figure(fig, sp)


def _render_classification(
    result: "_eng.MLResult",
    op: dict,
    sp: dict,
) -> Figure:
    """
    Three-panel classification figure:
    Panel 1 — confusion matrix
    Panel 2 — ROC curve (binary) / metrics bar / loss curve
    Panel 3 — feature importances
    """
    w_in = float(sp.get("width",  DEFAULT_FIG_WIDTH_IN))
    h_in = float(sp.get("height", DEFAULT_FIG_HEIGHT_IN))

    with plt.rc_context(rc_ctx(sp)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(w_in, h_in))

    _draw_confusion(ax1, result, op, sp, fig)
    _draw_clf_secondary(ax2, result, op, sp)
    _draw_importances(ax3, result, op, sp)

    return finalise_figure(fig, sp)


# ---------------------------------------------------------------------------
# Prediction stability diagnostic renderer
# ---------------------------------------------------------------------------

def _render_pred_stability(
    ps: _stab.PredictionStabilityResult,
    sp: Dict[str, Any],
) -> Figure:
    """Render a 2-panel prediction stability diagnostic figure.

    Left : per-variant metric values (grouped bars).
    Right: metric std bars colour-coded by stability label.
    """
    fig, (ax_var, ax_std) = plt.subplots(
        1, 2, figsize=(STAB_FIG_WIDTH_IN, STAB_FIG_HEIGHT_IN),
        dpi=DEFAULT_SCREEN_DPI,
    )

    metric_names = sorted(ps.mean_metrics.keys())
    K = ps.n_variants_used

    # ── Left: per-variant metric values ──────────────────────────────────
    x = np.arange(len(metric_names))
    width = 0.8 / max(K, 1)
    for k in range(K):
        vals = [ps.per_variant_metrics[k].get(m, 0.0) for m in metric_names]
        ax_var.bar(x + k * width, vals, width, label=f"v{k}", alpha=0.8)
    ax_var.set_xticks(x + width * (K - 1) / 2)
    ax_var.set_xticklabels(metric_names, fontsize=9)
    ax_var.set_ylabel("Metric value")
    ax_var.set_title(
        f"Prediction Stability — {K} variants\n"
        f"Δ_pred = {ps.delta_pred:.4f}",
        fontsize=10,
    )
    ax_var.legend(fontsize=8)
    ax_var.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    # ── Right: metric std bars ───────────────────────────────────────────
    stds = [ps.metric_std.get(m, 0.0) for m in metric_names]
    colors = [
        STAB_BAR_STABLE_COLOR
        if ps.stability_labels.get(m) == _stab.LABEL_FAMILY_STABLE
        else STAB_BAR_CONTINGENT_COLOR
        for m in metric_names
    ]
    ax_std.barh(metric_names, stds, color=colors, edgecolor="white", linewidth=0.5)
    ax_std.set_xlabel("Metric std across variants")
    ax_std.set_title("Metric Stability Labels", fontsize=10)
    from matplotlib.patches import Patch
    ax_std.legend(
        handles=[
            Patch(facecolor=STAB_BAR_STABLE_COLOR, label="family_stable"),
            Patch(facecolor=STAB_BAR_CONTINGENT_COLOR, label="contingent"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax_std.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Surrogate audit renderer
# ---------------------------------------------------------------------------

def _render_surrogate_audit(
    ar: _audit.SurrogateAuditResult,
    sp: Dict[str, Any],
) -> Figure:
    """Render a 2-panel surrogate audit diagnostic figure.

    Left : parity plot (actual vs predicted) for the surrogate model.
    Right: proxy feature importances bar chart.
    """
    has_imp = ar.feature_importances is not None and not ar.feature_importances.empty
    ncols = 2 if has_imp else 1
    ratios = [3, 2] if has_imp else [1]
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(AUDIT_FIG_WIDTH_IN, AUDIT_FIG_HEIGHT_IN),
        dpi=DEFAULT_SCREEN_DPI,
        gridspec_kw={"width_ratios": ratios},
    )
    if ncols == 1:
        axes = [axes]

    color = AUDIT_RECOVERABLE_COLOR if ar.is_proxy_recoverable else AUDIT_SAFE_COLOR
    flag = "PROXY-RECOVERABLE" if ar.is_proxy_recoverable else "NOT recoverable"

    # ── Left: parity plot ────────────────────────────────────────────────
    ax_par = axes[0]
    ax_par.scatter(ar.y_actual, ar.y_predicted, alpha=0.6, s=18,
                   edgecolors="white", linewidths=0.3, color=color)
    lo = min(ar.y_actual.min(), ar.y_predicted.min())
    hi = max(ar.y_actual.max(), ar.y_predicted.max())
    margin = (hi - lo) * 0.05
    ax_par.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", linewidth=0.8, alpha=0.5, label="Ideal")
    ax_par.set_xlabel(f"Actual {ar.target_col}")
    ax_par.set_ylabel(f"Predicted {ar.target_col}")
    ax_par.set_title(
        f"Surrogate Audit — {ar.surrogate_model_title}\n"
        f"R² = {ar.r2:.4f}   RMSE = {ar.rmse:.4f}   [{flag}]",
        fontsize=10,
    )
    ax_par.legend(fontsize=8)
    ax_par.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # ── Right: proxy feature importances ─────────────────────────────────
    if has_imp:
        ax_imp = axes[1]
        imp = ar.feature_importances.sort_values(ascending=True)
        ax_imp.barh(range(len(imp)), imp.values, color=color,
                    edgecolor="white", linewidth=0.5)
        ax_imp.set_yticks(range(len(imp)))
        ax_imp.set_yticklabels(imp.index, fontsize=8)
        ax_imp.set_xlabel("Importance / |coefficient|")
        ax_imp.set_title("Proxy Feature Contributions", fontsize=10)
        ax_imp.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Surrogate Audit configuration dialog
# ---------------------------------------------------------------------------

class _SurrogateAuditDialog(QDialog):
    """Small dialog to configure surrogate audit parameters."""

    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surrogate Audit Configuration")
        self.setMinimumSize(AUDIT_DIALOG_MIN_WIDTH_PX, AUDIT_DIALOG_MIN_HEIGHT_PX)

        layout = QVBoxLayout(self)

        # Target column
        layout.addWidget(QLabel("Structural parameter to predict:"))
        self._target_combo = QComboBox()
        self._target_combo.addItems(columns)
        layout.addWidget(self._target_combo)

        # Proxy features
        layout.addWidget(QLabel("Proxy-only features (select ≥ 1):"))
        self._proxy_list = QListWidget()
        for col in columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.Unchecked)
            self._proxy_list.addItem(item)
        layout.addWidget(self._proxy_list)

        # Surrogate model
        layout.addWidget(QLabel("Surrogate model:"))
        self._model_combo = QComboBox()
        models = _audit.available_surrogate_models()
        for key, title in models.items():
            self._model_combo.addItem(title, key)
        # Default to ridge
        idx = list(models.keys()).index(_audit.DEFAULT_SURROGATE_MODEL)
        self._model_combo.setCurrentIndex(idx)
        layout.addWidget(self._model_combo)

        # R² threshold
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("R² threshold:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(_audit.PROXY_R2_THRESHOLD_DEFAULT)
        self._threshold_spin.setDecimals(2)
        thresh_row.addWidget(self._threshold_spin)
        layout.addLayout(thresh_row)

        # OK / Cancel
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Run Audit")
        ok_btn.setObjectName("PrimaryButton")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def target_col(self) -> str:
        return self._target_combo.currentText()

    def proxy_features(self) -> List[str]:
        features = []
        for i in range(self._proxy_list.count()):
            item = self._proxy_list.item(i)
            if item.checkState() == Qt.Checked:
                features.append(item.text())
        return features

    def surrogate_model_key(self) -> str:
        return self._model_combo.currentData()

    def r2_threshold(self) -> float:
        return self._threshold_spin.value()


# ---------------------------------------------------------------------------
# MLDialog
# ---------------------------------------------------------------------------

class MLDialog(QDialog):
    """
    Unified machine learning analysis dialog.

    Left panel  : QTabWidget — Data / Options / Style / Export (4 tabs)
    Right panel : FigureCanvasQTAgg

    Computation occurs only on Generate click; no auto-refresh.
    """

    def __init__(self, key: str, store: SessionStore, parent=None):
        super().__init__(parent)
        self._key    = key
        self._store  = store
        self._fig:    Optional[Figure]      = None
        self._canvas: Optional[FigureCanvas] = None
        self._last_citable: Optional[_stab.CitableRecord] = None

        entry        = _eng._DISPATCH.get(key, (key,))
        method_title = entry[0]
        self.setWindowTitle(f"Machine Learning — {method_title}")
        self.setMinimumSize(
            CANVAS_MIN_WIDTH_PX + LEFT_PANEL_WIDTH_PX + 40,
            CANVAS_MIN_HEIGHT_PX + 80,
        )

        self._df = store.asset.dataframe if store.asset else pd.DataFrame()

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel ───────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setMinimumWidth(LEFT_PANEL_WIDTH_PX - 40)
        self._tabs.setMaximumWidth(LEFT_PANEL_WIDTH_PX + 60)
        root.addWidget(self._tabs)

        self._data_page = _MLDataPage(self._df)
        self._tabs.addTab(_scrolled(self._data_page), "Data")

        opt_inner, self._opt_getter = _make_options(key)
        self._tabs.addTab(_scrolled(opt_inner), "Options")

        sty_inner, self._sty_getter = make_style_panel()
        self._tabs.addTab(_scrolled(sty_inner), "Style")

        exp_inner, self._exp_getter = make_export_panel(
            self._save_figure, defaults={"prefix": "ml_result"},
        )
        self._tabs.addTab(_scrolled(exp_inner), "Export")

        # ── Right panel ──────────────────────────────────────────────────
        right = QVBoxLayout()
        root.addLayout(right, 1)

        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setObjectName("PrimaryButton")
        self._gen_btn.clicked.connect(self._generate)
        btn_row.addWidget(self._gen_btn)

        self._stab_btn = QPushButton("Stability")
        self._stab_btn.setToolTip(
            "Run this model across all SEAL variants and compute prediction "
            "stability diagnostics (requires SEAL conditioning)"
        )
        self._stab_btn.setEnabled(store.seal_result is not None)
        self._stab_btn.clicked.connect(self._run_stability)
        btn_row.addWidget(self._stab_btn)

        self._audit_btn = QPushButton("Surrogate Audit")
        self._audit_btn.setToolTip(
            "Test whether a structural parameter can be predicted from "
            "proxy features alone (DNV-2.0 Slide 10)"
        )
        self._audit_btn.clicked.connect(self._run_surrogate_audit)
        btn_row.addWidget(self._audit_btn)

        self._cite_btn = QPushButton("Export Record")
        self._cite_btn.setToolTip(
            "Export citable analysis record as JSON (available after Generate)"
        )
        self._cite_btn.setEnabled(False)
        self._cite_btn.clicked.connect(self._export_citable)
        btn_row.addWidget(self._cite_btn)

        btn_row.addStretch()
        right.addLayout(btn_row)

        self._placeholder = QLabel(
            "Select target and features in the Data tab, "
            "configure method options, then click  Generate."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setObjectName("SectionHint")
        self._placeholder.setMinimumSize(CANVAS_MIN_WIDTH_PX, CANVAS_MIN_HEIGHT_PX)
        right.addWidget(self._placeholder, 1)

        self._stab_report = QLabel("")
        self._stab_report.setWordWrap(True)
        self._stab_report.setObjectName("SectionHint")
        self._stab_report.setMaximumHeight(STAB_REPORT_MAX_HEIGHT_PX)
        self._stab_report.hide()
        right.addWidget(self._stab_report)

        self._right = right

    # ── Generate ─────────────────────────────────────────────────────────

    def _generate(self) -> None:
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("Training…")
        try:
            self._do_generate()
        except ImportError as ie:
            QMessageBox.warning(self, "Missing Dependency", str(ie))
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as exc:
            log.exception("MLDialog._generate")
            QMessageBox.critical(self, "Computation Error", str(exc))
        finally:
            self._gen_btn.setEnabled(True)
            self._gen_btn.setText("Generate")

    def _do_generate(self) -> None:
        key = self._key
        df  = self._df
        if df.empty:
            raise ValueError("No dataset loaded.")

        op  = self._opt_getter()
        sp  = self._sty_getter()
        dp  = self._data_page

        target_col  = dp.target()
        features    = dp.features()
        test_size   = dp.test_size()
        rs          = dp.random_state()
        standardize = dp.standardize()
        task_type   = dp.task_type()

        if not features:
            raise ValueError("Select at least one feature column in the Data tab.")

        from gui_ml import validate_task_type as _validate_task_type
        _validate_task_type(key, task_type, y_series=df[target_col])

        entry = _eng._DISPATCH.get(key)
        if entry is None:
            raise ValueError(f"Unknown method key: {key!r}")

        fn = entry[2]

        # Build kwargs: common + method-specific (only pass recognised ones)
        kwargs: dict = {
            "standardize": standardize,
            "test_size":   test_size,
            "rs":          rs,
            "task_type":   task_type,
        }
        # Forward all options that the engine function may accept
        for k, v in op.items():
            if k not in ("scatter_color", "scatter_alpha", "marker_size",
                         "bar_color", "bar_alpha", "ideal_color", "top_n",
                         "cm_cmap", "secondary_mode"):
                kwargs[k] = v

        result: _eng.MLResult = fn(df, features, target_col, **kwargs)

        # Wrap as citable record with model state hash (DNV-2.0 Slide 11)
        _contracts = {}
        if self._store.asset is not None:
            _contracts = self._store.asset.metadata.get(
                "observable_contracts", {}
            )
        self._last_citable = _stab.make_citable(
            result,
            variant_id=None,
            df=df[features + [target_col]].dropna(),
            contract_context=_contracts,
            method_config={
                "method": key,
                "target_col": target_col,
                "features": features,
                **{k: v for k, v in kwargs.items()
                   if not callable(v)},
            },
            model_state_hash=result.extra.get("model_state_hash"),
        )
        self._cite_btn.setEnabled(True)

        if result.task_type == "classification":
            fig = _render_classification(result, op, sp)
        else:
            fig = _render_regression(result, op, sp)

        self._set_canvas(fig)

    # ── Canvas management ────────────────────────────────────────────────

    def _set_canvas(self, fig: Figure) -> None:
        if self._canvas is not None:
            self._right.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        if self._placeholder.isVisible():
            self._placeholder.hide()
        self._fig    = fig
        self._canvas = FigureCanvas(fig)
        self._canvas.setMinimumSize(CANVAS_MIN_WIDTH_PX, CANVAS_MIN_HEIGHT_PX)
        self._right.addWidget(self._canvas, 1)
        self._canvas.draw()
        sync = getattr(self._sty_getter, "sync", None)
        if sync is not None:
            sync(fig)

    # ── Stability analysis ──────────────────────────────────────────────

    def _run_stability(self) -> None:
        self._stab_btn.setEnabled(False)
        self._stab_btn.setText("Computing…")
        try:
            self._do_stability()
        except Exception as exc:
            log.exception("MLDialog._run_stability")
            QMessageBox.critical(self, "Stability Error", str(exc))
        finally:
            self._stab_btn.setEnabled(self._store.seal_result is not None)
            self._stab_btn.setText("Stability")

    def _do_stability(self) -> None:
        seal_result = self._store.seal_result
        if seal_result is None:
            raise ValueError("No SEAL result available. Run SEAL conditioning first.")

        entry = _eng._DISPATCH.get(self._key)
        if entry is None:
            raise ValueError(f"Unknown method key: {self._key!r}")
        ml_fn = entry[2]

        dp = self._data_page
        target_col = dp.target()
        features = dp.features()
        if not target_col:
            raise ValueError("Select a target column first.")
        if not features:
            raise ValueError("Select at least one feature column.")

        op = self._opt_getter()
        kwargs: dict = {
            "standardize": dp.standardize(),
            "test_size":   dp.test_size(),
            "rs":          dp.random_state(),
            "task_type":   dp.task_type(),
        }
        for k, v in op.items():
            if k not in ("scatter_color", "scatter_alpha", "marker_size",
                         "bar_color", "bar_alpha", "ideal_color", "top_n",
                         "cm_cmap", "secondary_mode"):
                kwargs[k] = v

        ps = _stab.compute_prediction_stability(
            seal_result, ml_fn, features, target_col, **kwargs
        )
        sp = self._sty_getter()
        fig = _render_pred_stability(ps, sp)
        self._set_canvas(fig)

        n_stable = sum(
            1 for v in ps.stability_labels.values()
            if v == _stab.LABEL_FAMILY_STABLE
        )
        n_total = len(ps.stability_labels)
        self._stab_report.setText(
            f"Prediction stability — {ps.n_variants_used} variants  |  "
            f"Δ_pred = {ps.delta_pred:.4f}  |  "
            f"{n_stable}/{n_total} metrics family-stable"
        )
        self._stab_report.show()

    # ── Surrogate audit ─────────────────────────────────────────────────

    def _run_surrogate_audit(self) -> None:
        columns = [
            c for c in self._df.columns
            if pd.api.types.is_numeric_dtype(self._df[c])
        ]
        if len(columns) < 2:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Need at least 2 numeric columns for surrogate audit."
            )
            return

        dlg = _SurrogateAuditDialog(columns, self)
        if dlg.exec() != QDialog.Accepted:
            return

        target = dlg.target_col()
        proxies = dlg.proxy_features()
        model_key = dlg.surrogate_model_key()
        threshold = dlg.r2_threshold()

        if not proxies:
            QMessageBox.warning(
                self, "No Proxy Features",
                "Select at least one proxy feature."
            )
            return

        self._audit_btn.setEnabled(False)
        self._audit_btn.setText("Auditing…")
        try:
            result = _audit.compute_surrogate_audit(
                self._df, target, proxies,
                surrogate_model=model_key,
                r2_threshold=threshold,
            )
            sp = self._sty_getter()
            fig = _render_surrogate_audit(result, sp)
            self._set_canvas(fig)

            self._stab_report.setText(result.verdict)
            self._stab_report.show()
        except Exception as exc:
            log.exception("MLDialog._run_surrogate_audit")
            QMessageBox.critical(self, "Surrogate Audit Error", str(exc))
        finally:
            self._audit_btn.setEnabled(True)
            self._audit_btn.setText("Surrogate Audit")

    # ── Export citable record ───────────────────────────────────────────

    def _export_citable(self) -> None:
        """Serialize the last CitableRecord to a JSON file via save dialog."""
        import json

        cr = self._last_citable
        if cr is None:
            QMessageBox.information(
                self, "No Record",
                "Generate an ML analysis first to create a citable record.",
            )
            return

        # Build a JSON-safe dict manually (dataclasses.asdict would fail
        # on pandas/numpy objects inside analysis_result).
        ar = cr.analysis_result
        ar_dict = {
            "method_key": getattr(ar, "method_key", ""),
            "method_title": getattr(ar, "method_title", ""),
            "task_type": getattr(ar, "task_type", ""),
            "target_col": getattr(ar, "target_col", ""),
            "is_valid": getattr(ar, "is_valid", True),
        }
        metrics = getattr(ar, "metrics", None)
        if metrics is not None:
            ar_dict["metrics"] = {
                str(k): float(v) for k, v in metrics.items()
            }
        feat_names = getattr(ar, "feature_names", None)
        if feat_names is not None:
            ar_dict["feature_names"] = list(feat_names)
        record = {
            "analysis_result": ar_dict,
            "variant_id": cr.variant_id,
            "dataset_sha256": cr.dataset_sha256,
            "contract_context": cr.contract_context,
            "method_config": cr.method_config,
            "model_state_hash": cr.model_state_hash,
            "timestamp_utc": cr.timestamp_utc,
            "stability_label": cr.stability_label,
        }

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Citable Record",
            "ml_citable_record.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, default=str)
            QMessageBox.information(
                self, "Exported",
                f"Citable record saved to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Save ─────────────────────────────────────────────────────────────

    def _save_figure(self) -> None:
        if self._fig is None:
            QMessageBox.information(self, "No Figure", "Generate a plot first.")
            return
        ep     = self._exp_getter()
        fmt    = ep.get("fmt", "PNG").lower()
        dpi    = int(ep.get("dpi", DEFAULT_EXPORT_DPI))
        w_in   = float(ep.get("width",  DEFAULT_FIG_WIDTH_IN))
        h_in   = float(ep.get("height", DEFAULT_FIG_HEIGHT_IN))
        transp = bool(ep.get("transparent", False))
        prefix = ep.get("prefix", "ml_result") or "ml_result"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure",
            f"{prefix}.{fmt}",
            f"{fmt.upper()} Files (*.{fmt});;All Files (*)",
        )
        if not path:
            return
        try:
            orig_size = self._fig.get_size_inches()
            self._fig.set_size_inches(w_in, h_in)
            self._fig.savefig(
                path, dpi=dpi, bbox_inches="tight",
                transparent=transp, facecolor=self._fig.get_facecolor(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
        finally:
            self._fig.set_size_inches(*orig_size)
