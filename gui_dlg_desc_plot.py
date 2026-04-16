"""
DNV Scientific Module
---------------------
Role:
    Provides the DescriptorDialog — a fully configurable, canvas-embedded
    dialog for descriptor-modelling visualisations, with exhaustive per-method
    options, a shared style panel, and a complete export/save panel.

Scientific Context:
    Accepts a method key and a SessionStore; collects operator selections and
    method hyperparameters from the operator; dispatches to desc_engine for
    symbolic-feature generation and model fitting; renders the result as a
    two-panel figure (coefficient/R² bar chart + parity scatter) with full
    style and export control.

Invariants:
    - No computation occurs at dialog construction time; only on Generate.
    - Every numeric constant is a named Final with a unit suffix.
    - _apply_style() is the single site that writes per-axes matplotlib state.
    - The single FigureCanvas is replaced atomically on each Generate call.

Assumptions:
    - store.asset is a valid DataAsset with a DataFrame containing ≥ 2
      numeric columns.
    - desc_engine functions are pure and raise ImportError for missing deps.
    - matplotlib backend is "Agg".

Failure Modes:
    - Fewer than MIN_VALID_ROWS rows after feature generation: ValueError caught.
    - All-NaN formula: dropped silently by generate_features; is_valid=False shown.

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

from PySide6.QtCore    import Qt, QThread, Signal
from PySide6.QtGui     import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from gui_session import SessionStore
from gui_style import (
    make_style_panel, make_export_panel,
    apply_style, finalise_figure, rc_ctx,
)
import desc_engine as _eng

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout and default constants
# ---------------------------------------------------------------------------

#: Fixed width of the left configuration panel [px].
LEFT_PANEL_WIDTH_PX: Final[int] = 340

#: Minimum height of the canvas area [px].
CANVAS_MIN_HEIGHT_PX: Final[int] = 520

#: Minimum width of the canvas area [px].
CANVAS_MIN_WIDTH_PX: Final[int] = 700

#: Default figure width [inches].
DEFAULT_FIG_WIDTH_IN: Final[float] = 14.0

#: Default figure height [inches].
DEFAULT_FIG_HEIGHT_IN: Final[float] = 6.0

#: Default export DPI.
DEFAULT_EXPORT_DPI: Final[int] = 150

#: Default screen rendering DPI.
DEFAULT_SCREEN_DPI: Final[int] = 100

#: Spacing after section separators in left panel [px].
SECTION_SPACING_PX: Final[int] = 4

#: Keys for combinatorial search methods that share _render_search_chart.
_SEARCH_KEYS: Final[set] = {
    "greedy_forward", "greedy_backward", "armhc", "sa_metropolis", "sa_glauber",
    "rts", "cem", "pt", "qa", "pso", "gp",
}

_COLORMAPS: Final[List[str]] = [
    "coolwarm", "RdBu", "viridis", "plasma", "inferno",
    "Blues", "Greens", "Oranges", "Reds",
]
_FONT_FAMILIES: Final[List[str]] = [
    "sans-serif", "serif", "monospace",
    "Segoe UI", "Arial", "Helvetica", "Times New Roman", "DejaVu Sans",
]
_LEGEND_LOCS: Final[List[str]] = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center", "center",
]
_SCALES: Final[List[str]] = ["linear", "log", "symlog", "logit"]
_LINESTYLES: Final[List[str]] = ["--", "-", ":", "-."]


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
    w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step); w.setDecimals(4)
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

def _read(w) -> Any:
    if isinstance(w, QComboBox):      return w.currentText()
    if isinstance(w, QCheckBox):      return w.isChecked()
    if isinstance(w, QSpinBox):       return w.value()
    if isinstance(w, QDoubleSpinBox): return w.value()
    if isinstance(w, _ColorBtn):      return w.color()
    if isinstance(w, QLineEdit):      return w.text().strip()
    return None


# ---------------------------------------------------------------------------
# Data tab — target, features, unary/binary operators
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


class _DescDataPage(QWidget):
    """
    Data tab for DescriptorDialog.

    Provides: target column selector, feature multi-selector, unary/binary
    operation checkboxes, and feature generation cap spinboxes.
    """

    # Unary op labels and keys
    _UNARY_DEFS: Final[List[Tuple[str, str]]] = [
        ("x²",      "sq"),
        ("x³",      "cb"),
        ("1/x",     "inv"),
        ("|x|",     "abs"),
        ("√|x|",    "sqrt"),
        ("∛x",      "cbrt"),
        ("log|x|",  "log"),
        ("exp(x)",  "exp"),
    ]

    # Binary op labels and keys
    _BINARY_DEFS: Final[List[Tuple[str, str]]] = [
        ("a + b",         "add"),
        ("a − b",         "sub"),
        ("a × b",         "mul"),
        ("a / b",         "div"),
        ("|a − b|",       "abs_diff"),
        ("2/(1/a+1/b)",   "harmonic_mean"),
    ]

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(4)

        num_cols = list(df.select_dtypes(include="number").columns)
        all_cols = list(df.columns)

        # Target
        lay.addWidget(_eyebrow("TARGET COLUMN"))
        self._target = _combo(all_cols)
        lay.addWidget(_hrow("Target:", self._target))

        # Features
        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("BASE FEATURE COLUMNS"))
        self._sel = _MultiColSelector(num_cols)
        lay.addWidget(self._sel)
        br = QHBoxLayout()
        ba = QPushButton("All"); bn = QPushButton("None")
        ba.clicked.connect(self._sel.select_all)
        bn.clicked.connect(self._sel.deselect_all)
        br.addWidget(ba); br.addWidget(bn); br.addStretch()
        lay.addLayout(br)

        # Unary ops
        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("UNARY OPERATIONS"))
        ubox = QGroupBox(); ulay = QHBoxLayout(ubox)
        ulay.setContentsMargins(4, 2, 4, 2); ulay.setSpacing(6)
        self._unary_checks: Dict[str, QCheckBox] = {}
        for label, key in self._UNARY_DEFS:
            cb = QCheckBox(label)
            cb.setChecked(key in ("sq", "inv", "log"))
            self._unary_checks[key] = cb
            ulay.addWidget(cb)
        ulay.addStretch()
        lay.addWidget(ubox)

        # Binary ops
        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("BINARY OPERATIONS"))
        bbox = QGroupBox(); blay = QHBoxLayout(bbox)
        blay.setContentsMargins(4, 2, 4, 2); blay.setSpacing(6)
        self._binary_checks: Dict[str, QCheckBox] = {}
        for label, key in self._BINARY_DEFS:
            cb = QCheckBox(label)
            cb.setChecked(key in ("add", "sub", "mul"))
            self._binary_checks[key] = cb
            blay.addWidget(cb)
        blay.addStretch()
        lay.addWidget(bbox)

        # Generation caps
        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("GENERATION LIMITS"))
        self._combo_cap = _spin(5, 200,  _eng.FEATURE_COMBO_CAP_DEFAULT)
        self._total_cap = _spin(100, 20000, _eng.FEATURE_TOTAL_CAP_DEFAULT)
        lay.addWidget(_hrow("Combo cap:",      self._combo_cap))
        lay.addWidget(_hrow("Total feat cap:", self._total_cap))

        lay.addStretch()

    # ── Accessors ─────────────────────────────────────────────────────────

    def target(self) -> str:
        return self._target.currentText()

    def features(self) -> List[str]:
        sel = [c for c in self._sel.selected() if c != self.target()]
        return sel if sel else []

    def selected_ops(self) -> Dict[str, List[str]]:
        return {
            "unary":  [k for k, cb in self._unary_checks.items()  if cb.isChecked()],
            "binary": [k for k, cb in self._binary_checks.items() if cb.isChecked()],
        }

    def combo_cap(self) -> int:  return self._combo_cap.value()
    def total_cap(self) -> int:  return self._total_cap.value()


# ---------------------------------------------------------------------------
# Options panel — per-method controls
# ---------------------------------------------------------------------------

def _make_options(key: str) -> Tuple[QWidget, Callable[[], dict]]:
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
    controls: Dict[str, Any] = {}

    def _add(label: str, widget, key_: str) -> None:
        controls[key_] = widget; lay.addWidget(_hrow(label, widget))

    def _sec(text: str) -> None:
        lay.addWidget(_section_sep()); lay.addWidget(_eyebrow(text))
        lay.addSpacing(SECTION_SPACING_PX)

    # ── Shared display controls ───────────────────────────────────────────
    _sec("BAR DISPLAY")
    _add("Bar color:",         _ColorBtn("#3a7bd5"),                                "bar_color")
    _add("Negative color:",    _ColorBtn("#e74c3c"),                                "neg_color")
    _add("Bar alpha:",         _dspin(0.3, 1.0, 0.85, 0.05),                       "bar_alpha")
    _add("Edge color:",        _ColorBtn("#1a3a6a"),                                "edge_color")
    _add("Edge linewidth:",    _dspin(0.0, 3.0, 0.5, 0.1),                         "edge_lw")
    _add("Top N features:",    _spin(1, 200, 20),                                   "top_n")
    _add("Show value labels:", _check(True),                                        "show_labels")
    _add("Label font size:",   _spin(5, 16, 8),                                     "label_fontsize")

    _sec("PARITY PLOT")
    _add("Scatter color:",     _ColorBtn("#3a7bd5"),                                "scatter_color")
    _add("Scatter alpha:",     _dspin(0.1, 1.0, 0.65, 0.05),                       "scatter_alpha")
    _add("Marker size:",       _spin(2, 20, 6),                                     "marker_size")
    _add("Identity line color:",_ColorBtn("#e74c3c"),                              "ideal_color")
    _add("Show R² annotation:",_check(True),                                        "show_r2_annot")

    # ── Method-specific controls ──────────────────────────────────────────
    if key == "linear_screening":
        _sec("SCREENING FILTER")
        _add("R² threshold:",  _dspin(0.0, 1.0, 0.0, 0.05),                        "r2_threshold")

    elif key == "lasso":
        _sec("LASSO CV")
        _add("CV folds:",      _spin(2, 20, 5),                                     "cv")
        _add("Max iterations:",_spin(100, 50000, 5000),                             "max_iter")
        _add("Random state:",  _spin(0, 9999, 42),                                  "rs")

    elif key == "elasticnet":
        _sec("ELASTIC NET CV")
        _add("CV folds:",      _spin(2, 20, 5),                                     "cv")
        _add("Max iterations:",_spin(100, 50000, 5000),                             "max_iter")
        _add("Random state:",  _spin(0, 9999, 42),                                  "rs")

    elif key == "greedy_forward":
        _sec("GREEDY FORWARD")
        _add("Max features:",    _spin(1, 200, _eng.GREEDY_MAX_FEATURES_DEFAULT),   "max_features")
        _add("Preselect top K:", _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),     "preselect_k")

    elif key == "greedy_backward":
        _sec("GREEDY BACKWARD")
        _add("Min features:",    _spin(1, 200, _eng.GREEDY_MIN_FEATURES_DEFAULT),   "min_features")
        _add("Preselect top K:", _spin(1, 200, _eng.GREEDY_BACKWARD_PRESELECT_DEFAULT), "preselect_k")

    elif key == "armhc":
        _sec("ARMHC")
        _add("Max iterations:",          _spin(100, 20000, _eng.SEARCH_MAX_ITER_DEFAULT),                                   "max_iter")
        _add("Initial subset K:",        _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                                         "k_init")
        _add("Init mutation (%):",       _dspin(1.0, 50.0, _eng.ARMHC_INIT_MUTATION_INTENSITY_RATIO * 100.0, 1.0),        "init_mutation_pct")
        _add("Accept rate lo (%):",      _dspin(1.0, 49.0, _eng.ARMHC_ADAPT_LO_THRESHOLD * 100.0, 1.0),                  "adapt_lo_pct")
        _add("Accept rate hi (%):",      _dspin(2.0, 50.0, _eng.ARMHC_ADAPT_HI_THRESHOLD * 100.0, 1.0),                  "adapt_hi_pct")
        _add("Adapt window:",            _spin(10, 1000, _eng.ARMHC_ADAPT_WINDOW),                                         "adapt_window")
        _add("Intensity scale-up ×:",    _dspin(1.01, 5.0, _eng.ARMHC_INTENSITY_SCALE_UP_RATIO, 0.05),                    "intensity_scale_up")
        _add("Scale-down max frac:",     _dspin(0.01, 0.99, _eng.ARMHC_INTENSITY_SCALE_DOWN_MAX_RATIO, 0.05),              "intensity_scale_down_max")
        _add("Preselect top K:",         _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),                                    "preselect_k")
        _add("Random state:",            _spin(0, 9999, 42),                                                                "rs")

    elif key == "rts":
        _sec("REACTIVE TABU SEARCH")
        _add("Max iterations:",       _spin(100, 20000, _eng.SEARCH_MAX_ITER_DEFAULT),                              "max_iter")
        _add("Initial subset K:",     _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                                    "k_init")
        _add("Init tabu tenure:",     _spin(1, 100, _eng.RTS_INIT_TABU_TENURE_DEFAULT),                            "init_tabu_tenure")
        _add("Tenure adapt step:",    _spin(1, 20, _eng.RTS_TABU_ADAPT_STEP_DEFAULT),                              "tabu_adapt_step")
        _add("Candidates per iter:",  _spin(1, 200, _eng.RTS_CANDIDATES_PER_ITER_DEFAULT),                         "n_candidates")
        _add("Preselect top K:",      _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),                               "preselect_k")
        _add("Random state:",         _spin(0, 9999, 42),                                                           "rs")

    elif key == "cem":
        _sec("CROSS-ENTROPY METHOD")
        _add("Max iterations:",       _spin(10, 2000, _eng.CEM_MAX_ITER_DEFAULT),                                   "max_iter")
        _add("Initial subset K:",     _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                                    "k_init")
        _add("Samples per iter:",     _spin(5, 500, _eng.CEM_N_SAMPLES_DEFAULT),                                   "n_samples")
        _add("Elite fraction:",       _dspin(0.01, 0.99, _eng.CEM_ELITE_FRAC_DEFAULT, 0.05),                       "elite_frac")
        _add("Smoothing α:",          _dspin(0.01, 0.99, _eng.CEM_SMOOTHING_DEFAULT, 0.05),                        "smoothing")
        _add("Preselect top K:",      _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),                               "preselect_k")
        _add("Random state:",         _spin(0, 9999, 42),                                                           "rs")

    elif key == "pt":
        _sec("PARALLEL TEMPERING")
        _add("Max iterations:",       _spin(100, 20000, _eng.PT_MAX_ITER_DEFAULT),                                  "max_iter")
        _add("Initial subset K:",     _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                                    "k_init")
        _add("N replicas:",           _spin(2, 16, _eng.PT_N_REPLICAS_DEFAULT),                                    "n_replicas")
        _add("T min:",                _dspin(0.001, 1.0, _eng.PT_T_MIN_DEFAULT, 0.005),                            "t_min")
        _add("T max:",                _dspin(0.1, 20.0, _eng.PT_T_MAX_DEFAULT, 0.1),                               "t_max")
        _add("Swap interval:",        _spin(1, 500, _eng.PT_SWAP_INTERVAL_DEFAULT),                                 "swap_interval")
        _add("Preselect top K:",      _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),                               "preselect_k")
        _add("Random state:",         _spin(0, 9999, 42),                                                           "rs")

    elif key == "qa":
        _sec("QUANTUM ANNEALING (PIMC)")
        _add("Max iterations:",       _spin(100, 20000, _eng.SEARCH_MAX_ITER_DEFAULT),                              "max_iter")
        _add("Initial subset K:",     _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                                    "k_init")
        _add("Trotter slices P:",     _spin(1, 64, _eng.QA_N_TROTTER_DEFAULT),                                     "n_trotter")
        _add("γ initial:",            _dspin(0.01, 20.0, _eng.QA_GAMMA_INIT_DEFAULT, 0.1),                         "gamma_init")
        _add("γ final:",              _dspin(1e-6, 1.0, _eng.QA_GAMMA_FINAL_DEFAULT, 0.001),                       "gamma_final")
        _add("Classical T:",          _dspin(0.001, 5.0, _eng.QA_T_FIXED_DEFAULT, 0.05),                           "t_fixed")
        _add("Preselect top K:",      _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),                               "preselect_k")
        _add("Random state:",         _spin(0, 9999, 42),                                                           "rs")

    elif key == "pso":
        _sec("PARTICLE SWARM OPTIMIZATION")
        _add("Max iterations:",       _spin(100, 20000, _eng.SEARCH_MAX_ITER_DEFAULT),              "max_iter")
        _add("Initial subset K:",     _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                    "k_init")
        _add("N particles:",          _spin(2, 200, _eng.PSO_N_PARTICLES_DEFAULT),                 "n_particles")
        _add("Inertia w:",            _dspin(0.01, 2.0, _eng.PSO_W_DEFAULT, 0.05),                "w")
        _add("Cognitive c₁:",         _dspin(0.01, 4.0, _eng.PSO_C1_DEFAULT, 0.1),                "c1")
        _add("Social c₂:",            _dspin(0.01, 4.0, _eng.PSO_C2_DEFAULT, 0.1),                "c2")
        _add("V max:",                _dspin(0.1, 20.0, _eng.PSO_V_MAX_DEFAULT, 0.5),             "v_max")
        _add("Preselect top K:",      _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),               "preselect_k")
        _add("Random state:",         _spin(0, 9999, 42),                                          "rs")

    elif key == "gp":
        _sec("GENETIC PROGRAMMING")
        _add("Max generations:",      _spin(5, 5000, _eng.GP_MAX_GENS_DEFAULT),                    "max_iter")
        _add("Initial subset K:",     _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),                    "k_init")
        _add("Population size:",      _spin(4, 500, _eng.GP_POP_SIZE_DEFAULT),                     "pop_size")
        _add("Crossover prob:",       _dspin(0.0, 1.0, _eng.GP_CROSSOVER_PROB_DEFAULT, 0.05),     "crossover_prob")
        _add("Mutation prob:",        _dspin(0.0, 1.0, _eng.GP_MUTATION_PROB_DEFAULT, 0.01),       "mutation_prob")
        _add("Tournament size:",      _spin(2, 20, _eng.GP_TOURNAMENT_SIZE_DEFAULT),               "tournament_size")
        _add("Elitism count:",        _spin(0, 20, _eng.GP_ELITISM_COUNT_DEFAULT),                 "elitism")
        _add("Parsimony λ:",          _dspin(0.0, 0.1, _eng.GP_PARSIMONY_COEFF_DEFAULT, 0.001),   "parsimony_coeff")
        _add("Preselect top K:",      _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),               "preselect_k")
        _add("Random state:",         _spin(0, 9999, 42),                                          "rs")

    elif key in ("sa_metropolis", "sa_glauber"):
        _sec("SIMULATED ANNEALING")
        _add("Max iterations:",  _spin(100, 20000, _eng.SEARCH_MAX_ITER_DEFAULT),   "max_iter")
        _add("Initial subset K:", _spin(1, 50, _eng.SEARCH_K_INIT_DEFAULT),        "k_init")
        _add("Initial temp T₀:", _dspin(0.001, 10.0, _eng.SA_T0_DEFAULT, 0.05),   "t0")
        _add("Cooling ratio:",   _dspin(0.900, 0.9999, _eng.SA_COOLING_RATIO_DEFAULT, 0.001), "cooling")
        _add("Preselect top K:", _spin(1, 2000, _eng.SEARCH_PRESELECT_DEFAULT),     "preselect_k")
        _add("Random state:",    _spin(0, 9999, 42),                                "rs")

    lay.addStretch()

    def _getter() -> dict:
        return {k: _read(v) for k, v in controls.items()}

    return w, _getter


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_parity(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: str,
    title: str,
    op: dict,
    sp: dict,
) -> None:
    """
    Draw a parity (Predicted vs Actual) scatter with identity line into ax.
    """
    scatter_color = op.get("scatter_color", "#3a7bd5")
    scatter_alpha = float(op.get("scatter_alpha", 0.65))
    marker_size   = int(op.get("marker_size", 6))
    ideal_color   = op.get("ideal_color",   "#e74c3c")
    show_r2_annot = op.get("show_r2_annot", True)

    ax.scatter(y_true, y_pred, color=scatter_color,
               alpha=scatter_alpha, s=marker_size**2,
               linewidths=0, zorder=3)

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=ideal_color, linewidth=1.2, linestyle="--",
            label="Ideal fit", zorder=2)

    if show_r2_annot and len(y_true) > 1:
        from sklearn.metrics import r2_score, mean_squared_error as mse
        r2   = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mse(y_true, y_pred)))
        ax.text(0.05, 0.94,
                f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
                transform=ax.transAxes,
                fontsize=sp.get("tick_fontsize", 9),
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", alpha=0.85))

    ax.set_xlabel(f"Actual {target_col}", fontsize=sp.get("label_fontsize", 10))
    ax.set_ylabel(f"Predicted {target_col}", fontsize=sp.get("label_fontsize", 10))
    ax.set_title(sp.get("title") or title,
                 fontsize=sp.get("title_fontsize", 11), fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def _render_descriptor_chart(
    result: "_eng.DescriptorResult",
    op: dict,
    sp: dict,
) -> Figure:
    """
    Two-panel figure:
    - Left  panel: bar chart of scores (R² for screening, |coeff| for regularised)
    - Right panel: parity scatter (Actual vs Predicted)
    """
    w_in = sp.get("width",  DEFAULT_FIG_WIDTH_IN)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN)

    with plt.rc_context(rc_ctx(sp)):
        fig, (ax_bar, ax_par) = plt.subplots(1, 2, figsize=(w_in, h_in),
                                              gridspec_kw={"width_ratios": [1.2, 1]})

        rdf = result.results_df
        top_n = int(op.get("top_n", 20))

        # ── Left: bar chart ───────────────────────────────────────────────
        bar_color  = op.get("bar_color",  "#3a7bd5")
        neg_color  = op.get("neg_color",  "#e74c3c")
        bar_alpha  = float(op.get("bar_alpha", 0.85))
        edge_color = op.get("edge_color", "#1a3a6a")
        edge_lw    = float(op.get("edge_lw", 0.5))
        show_lbl   = op.get("show_labels", True)
        lbl_fs     = int(op.get("label_fontsize", 8))

        if result.method_key == "linear_screening":
            # Bar chart of cross-validated R² scores (no in-sample leakage)
            data = rdf.nlargest(top_n, "cv_r2")
            values     = data["cv_r2"].values
            labels     = [f[:40] for f in data["formula"].values]
            bar_colors = [bar_color] * len(values)
            xlabel     = "Cross-Validated R²"
            bar_title  = f"Top {len(values)} Formulas by CV R²  (n_gen={result.n_generated})"

        else:
            # Bar chart of coefficients (sorted by |coeff|)
            data = rdf.assign(_abs=rdf["coefficient"].abs()).nlargest(top_n, "_abs")
            values     = data["coefficient"].values
            labels     = [f[:40] for f in data["formula"].values]
            bar_colors = [neg_color if v < 0 else bar_color for v in values]
            xlabel     = "Standardised Coefficient"
            bar_title  = (
                f"{result.method_title} — {len(values)} selected "
                f"/ {result.n_generated} generated"
            )

        ax_bar.barh(range(len(values)), values,
                    color=bar_colors, alpha=bar_alpha,
                    edgecolor=edge_color, linewidth=edge_lw)
        ax_bar.set_yticks(range(len(values)))
        ax_bar.set_yticklabels(labels, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.axvline(0, color="#444", linewidth=0.7, linestyle="--")
        ax_bar.set_xlabel(xlabel, fontsize=sp.get("label_fontsize", 10))
        ax_bar.set_title(bar_title,
                         fontsize=sp.get("title_fontsize", 11), fontweight="bold")
        ax_bar.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

        if show_lbl:
            for y_pos, val in enumerate(values):
                x_off = (ax_bar.get_xlim()[1] - ax_bar.get_xlim()[0]) * 0.01
                ax_bar.text(val + x_off, y_pos, f"{val:.3f}",
                            va="center", ha="left", fontsize=lbl_fs)

        # ── Right: parity plot ────────────────────────────────────────────
        y_true = result.extra.get("y_true",      np.array([]))
        y_pred = result.extra.get(
            "y_pred_best" if result.method_key == "linear_screening" else "y_pred",
            np.array([]),
        )

        if len(y_true) > 1 and len(y_pred) == len(y_true):
            best_lbl = result.extra.get("best_formula", result.method_title)
            if len(best_lbl) > 50:
                best_lbl = best_lbl[:47] + "…"
            parity_title = (
                f"Best Formula\n{best_lbl}"
                if result.method_key == "linear_screening"
                else f"Overall Model Parity\n{result.method_title}"
            )
            _render_parity(ax_par, y_true, y_pred, result.target_col,
                           parity_title, op, sp)

            # Show training R² warning when n_features >= n_samples
            # (_render_parity already annotates with CV R²/RMSE at (0.05, 0.94))
            train_r2 = result.extra.get("train_r2")
            overfit  = result.extra.get("overfit_warning", False)
            if train_r2 is not None and overfit and op.get("show_r2_annot", True):
                ax_par.text(
                    0.05, 0.70,
                    f"[!] Train R\u00b2 = {train_r2:.4f}\n"
                    f"(n_feat \u2265 n_samples)",
                    transform=ax_par.transAxes,
                    fontsize=max(7, int(sp.get("tick_fontsize", 9)) - 1),
                    va="top", ha="left", color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#fff5f5", ec="#e74c3c", alpha=0.85),
                )
        else:
            ax_par.text(0.5, 0.5, "No parity data available.\nRun Generate first.",
                        ha="center", va="center", transform=ax_par.transAxes,
                        fontsize=10)
            ax_par.axis("off")

    return finalise_figure(fig, sp)


# ---------------------------------------------------------------------------
# Search renderer — 3-panel: coefficients | convergence | parity
# ---------------------------------------------------------------------------

def _render_search_chart(
    result: "_eng.DescriptorResult",
    op: dict,
    sp: dict,
) -> Figure:
    """
    Three-panel figure for combinatorial search methods:
    - Left   : horizontal bar chart of standardised coefficients (top-N)
    - Centre : convergence curve (best R² per iteration); SA methods add a
               right twin-axis showing temperature
    - Right  : parity scatter (CV Predicted vs Actual)
    """
    w_in = sp.get("width",  DEFAULT_FIG_WIDTH_IN)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN)

    has_temp    = "temp_history"    in result.extra
    has_intensity = "intensity_history" in result.extra
    has_tenure  = "tenure_history"  in result.extra

    with plt.rc_context(rc_ctx(sp)):
        fig, axes = plt.subplots(1, 3, figsize=(w_in, h_in),
                                 gridspec_kw={"width_ratios": [1.2, 1.1, 1]})
    ax_bar, ax_conv, ax_par = axes

    rdf    = result.results_df
    top_n  = int(op.get("top_n", 20))

    # ── Left: coefficient bar chart ───────────────────────────────────────
    bar_color  = op.get("bar_color",  "#3a7bd5")
    neg_color  = op.get("neg_color",  "#e74c3c")
    bar_alpha  = float(op.get("bar_alpha", 0.85))
    edge_color = op.get("edge_color", "#1a3a6a")
    edge_lw    = float(op.get("edge_lw", 0.5))
    show_lbl   = op.get("show_labels", True)
    lbl_fs     = int(op.get("label_fontsize", 8))

    if not rdf.empty and "coefficient" in rdf.columns:
        data       = rdf.assign(_abs=rdf["coefficient"].abs()).nlargest(top_n, "_abs")
        values     = data["coefficient"].values
        labels     = [f[:40] for f in data["formula"].values]
        bar_colors = [neg_color if v < 0 else bar_color for v in values]
        ax_bar.barh(range(len(values)), values,
                    color=bar_colors, alpha=bar_alpha,
                    edgecolor=edge_color, linewidth=edge_lw)
        ax_bar.set_yticks(range(len(values)))
        ax_bar.set_yticklabels(labels, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.axvline(0, color="#444", linewidth=0.7, linestyle="--")
        if show_lbl:
            for y_pos, val in enumerate(values):
                x_off = (ax_bar.get_xlim()[1] - ax_bar.get_xlim()[0]) * 0.01
                ax_bar.text(val + x_off, y_pos, f"{val:.3f}",
                            va="center", ha="left", fontsize=lbl_fs)
    else:
        ax_bar.text(0.5, 0.5, "No selected features.", ha="center", va="center",
                    transform=ax_bar.transAxes); ax_bar.axis("off")

    n_sel = result.extra.get("n_selected", 0)
    ax_bar.set_xlabel("Standardised Coefficient", fontsize=sp.get("label_fontsize", 10))
    ax_bar.set_title(
        f"{result.method_title}\n"
        f"{n_sel} selected / {result.n_generated} generated",
        fontsize=sp.get("title_fontsize", 11), fontweight="bold",
    )
    ax_bar.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    # ── Centre: convergence curve ─────────────────────────────────────────
    score_hist = result.extra.get("score_history", [])
    if score_hist:
        iters = range(len(score_hist))
        ax_conv.plot(iters, score_hist, color=bar_color, linewidth=1.5,
                     label="Best R²")
        ax_conv.set_xlabel("Iteration", fontsize=sp.get("label_fontsize", 10))
        ax_conv.set_ylabel("Best R²",   fontsize=sp.get("label_fontsize", 10))
        ax_conv.set_ylim(bottom=0.0)
        ax_conv.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

        if has_temp:
            temp_hist = result.extra.get("temp_history", [])
            if temp_hist:
                ax_twin = ax_conv.twinx()
                ax_twin.plot(range(len(temp_hist)), temp_hist,
                             color="#f39c12", linewidth=0.9, linestyle=":",
                             alpha=0.75, label="Temperature")
                ax_twin.set_ylabel("Temperature", fontsize=sp.get("label_fontsize", 9),
                                   color="#f39c12")
                ax_twin.tick_params(axis="y", labelcolor="#f39c12",
                                    labelsize=sp.get("tick_fontsize", 8))
        elif has_intensity:
            int_hist = result.extra.get("intensity_history", [])
            if int_hist:
                ax_twin = ax_conv.twinx()
                ax_twin.plot(range(len(int_hist)), [x * 100.0 for x in int_hist],
                             color="#9b59b6", linewidth=0.9, linestyle=":",
                             alpha=0.75, label="Mutation (%)")
                ax_twin.set_ylabel("Mutation Intensity (%)",
                                   fontsize=sp.get("label_fontsize", 9), color="#9b59b6")
                ax_twin.tick_params(axis="y", labelcolor="#9b59b6",
                                    labelsize=sp.get("tick_fontsize", 8))
        elif has_tenure:
            ten_hist = result.extra.get("tenure_history", [])
            if ten_hist:
                ax_twin = ax_conv.twinx()
                ax_twin.plot(range(len(ten_hist)), ten_hist,
                             color="#27ae60", linewidth=0.9, linestyle=":",
                             alpha=0.75, label="Tabu Tenure")
                ax_twin.set_ylabel("Tabu Tenure",
                                   fontsize=sp.get("label_fontsize", 9), color="#27ae60")
                ax_twin.tick_params(axis="y", labelcolor="#27ae60",
                                    labelsize=sp.get("tick_fontsize", 8))
    else:
        ax_conv.text(0.5, 0.5, "No convergence data.", ha="center", va="center",
                     transform=ax_conv.transAxes); ax_conv.axis("off")

    ax_conv.set_title("Search Convergence",
                      fontsize=sp.get("title_fontsize", 11), fontweight="bold")

    # ── Right: parity plot ────────────────────────────────────────────────
    y_true = result.extra.get("y_true", np.array([]))
    y_pred = result.extra.get("y_pred", np.array([]))

    if len(y_true) > 1 and len(y_pred) == len(y_true):
        _render_parity(ax_par, y_true, y_pred, result.target_col,
                       f"CV Parity\n{result.method_title}", op, sp)
        train_r2 = result.extra.get("train_r2")
        overfit  = result.extra.get("overfit_warning", False)
        if train_r2 is not None and overfit and op.get("show_r2_annot", True):
            ax_par.text(
                0.05, 0.70,
                f"[!] Train R\u00b2 = {train_r2:.4f}\n(n_feat \u2265 n_samples)",
                transform=ax_par.transAxes,
                fontsize=max(7, int(sp.get("tick_fontsize", 9)) - 1),
                va="top", ha="left", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#fff5f5", ec="#e74c3c", alpha=0.85),
            )
    else:
        ax_par.text(0.5, 0.5, "No parity data available.", ha="center", va="center",
                    transform=ax_par.transAxes); ax_par.axis("off")

    return finalise_figure(fig, sp)


# ---------------------------------------------------------------------------
# _SearchWorker — background thread for combinatorial search methods
# ---------------------------------------------------------------------------

class _SearchWorker(QThread):
    """
    Runs a single combinatorial search compute_* function in a background thread.

    Signals
    -------
    progress(current, total, best_score) : emitted every ~50 iterations
    finished_ok(DescriptorResult)        : emitted on successful completion
    error_occurred(str)                  : emitted on any exception
    """
    progress      = Signal(int, int, float)
    finished_ok   = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, fn, df, features, target_col, selected_ops, kwargs, parent=None):
        super().__init__(parent)
        self._fn           = fn
        self._df           = df
        self._features     = features
        self._target_col   = target_col
        self._selected_ops = selected_ops
        self._kwargs       = dict(kwargs)

    def _cb(self, current: int, total: int, best_score: float) -> None:
        self.progress.emit(current, total, best_score)

    def run(self) -> None:
        try:
            kw = dict(self._kwargs)
            kw["progress_cb"] = self._cb
            result = self._fn(
                self._df, self._features, self._target_col, self._selected_ops, **kw,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# DescriptorDialog
# ---------------------------------------------------------------------------

class DescriptorDialog(QDialog):
    """
    Unified descriptor-modelling analysis dialog.

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
        self._worker: Optional[_SearchWorker] = None
        self._op_cache: dict = {}
        self._sp_cache: dict = {}

        entry = _eng._DISPATCH.get(key, (key,))
        method_title = entry[0]
        self.setWindowTitle(f"Descriptors — {method_title}")
        self.setMinimumSize(
            CANVAS_MIN_WIDTH_PX + LEFT_PANEL_WIDTH_PX + 40,
            CANVAS_MIN_HEIGHT_PX + 80,
        )

        self._df = store.asset.dataframe if store.asset else pd.DataFrame()

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel (4 tabs) ──────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setMinimumWidth(LEFT_PANEL_WIDTH_PX - 40)
        self._tabs.setMaximumWidth(LEFT_PANEL_WIDTH_PX + 60)
        root.addWidget(self._tabs)

        self._data_page = _DescDataPage(self._df)
        self._tabs.addTab(_scrolled(self._data_page), "Data")

        opt_inner, self._opt_getter = _make_options(key)
        self._tabs.addTab(_scrolled(opt_inner), "Options")

        sty_inner, self._sty_getter = make_style_panel()
        self._tabs.addTab(_scrolled(sty_inner), "Style")

        exp_inner, self._exp_getter = make_export_panel(
            self._save_figure, defaults={"prefix": "descriptors"},
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
        btn_row.addStretch()
        right.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setVisible(False)
        right.addWidget(self._progress_bar)

        self._status_lbl = QLabel()
        self._status_lbl.setObjectName("SectionHint")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setVisible(False)
        right.addWidget(self._status_lbl)

        self._placeholder = QLabel(
            "Configure features and operators in the Data tab, then click  Generate."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setObjectName("SectionHint")
        self._placeholder.setMinimumSize(CANVAS_MIN_WIDTH_PX, CANVAS_MIN_HEIGHT_PX)
        right.addWidget(self._placeholder, 1)
        self._right = right

    # ── Generate ─────────────────────────────────────────────────────────

    def _generate(self) -> None:
        """Dispatch to async worker (search methods) or sync path (others)."""
        if self._worker is not None and self._worker.isRunning():
            return
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("Computing…")
        async_launched = False
        try:
            if self._key in _SEARCH_KEYS:
                async_launched = self._launch_search_worker()
            else:
                self._do_generate_sync()
        except ImportError as ie:
            QMessageBox.warning(self, "Missing Dependency", str(ie))
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as exc:
            log.exception("DescriptorDialog._generate")
            QMessageBox.critical(self, "Computation Error", str(exc))
        finally:
            if not async_launched:
                self._gen_btn.setEnabled(True)
                self._gen_btn.setText("Generate")

    def _do_generate_sync(self) -> None:
        """Synchronous path for non-search methods (fast; runs on main thread)."""
        key = self._key
        df  = self._df
        if df.empty:
            raise ValueError("No dataset loaded.")

        op  = self._opt_getter()
        sp  = self._sty_getter()
        dp  = self._data_page

        target_col   = dp.target()
        features     = dp.features()
        selected_ops = dp.selected_ops()
        combo_cap    = dp.combo_cap()
        total_cap    = dp.total_cap()

        if not features:
            raise ValueError("Select at least one feature column in the Data tab.")
        if not (selected_ops.get("unary") or selected_ops.get("binary")):
            raise ValueError(
                "Select at least one unary or binary operation in the Data tab."
            )

        entry = _eng._DISPATCH.get(key)
        if entry is None:
            raise ValueError(f"Unknown method key: {key!r}")

        fn = entry[2]
        kwargs: dict = {"combo_cap": combo_cap, "total_cap": total_cap}

        if key == "linear_screening":
            kwargs["r2_threshold"] = float(op.get("r2_threshold", 0.0))
        elif key == "lasso":
            kwargs["cv"]       = int(op.get("cv", 5))
            kwargs["max_iter"] = int(op.get("max_iter", 5000))
            kwargs["rs"]       = int(op.get("rs", 42))
        elif key == "elasticnet":
            kwargs["cv"]       = int(op.get("cv", 5))
            kwargs["max_iter"] = int(op.get("max_iter", 5000))
            kwargs["rs"]       = int(op.get("rs", 42))
        result: _eng.DescriptorResult = fn(
            df, features, target_col, selected_ops, **kwargs,
        )

        if not result.is_valid:
            QMessageBox.warning(
                self, "Degraded Result",
                f"{result.method_title}: no descriptors passed the selection "
                "criteria.  Try lowering the R² threshold or adding more operators.",
            )

        fig = _render_descriptor_chart(result, op, sp)
        self._set_canvas(fig)

    def _launch_search_worker(self) -> bool:
        """
        Validate inputs, build kwargs, start _SearchWorker.

        Returns True so _generate skips the finally re-enable.
        Raises ValueError on invalid inputs (caught before any thread starts).
        """
        key = self._key
        df  = self._df
        if df.empty:
            raise ValueError("No dataset loaded.")

        op  = self._opt_getter()
        sp  = self._sty_getter()
        dp  = self._data_page

        target_col   = dp.target()
        features     = dp.features()
        selected_ops = dp.selected_ops()
        combo_cap    = dp.combo_cap()
        total_cap    = dp.total_cap()

        if not features:
            raise ValueError("Select at least one feature column in the Data tab.")
        if not (selected_ops.get("unary") or selected_ops.get("binary")):
            raise ValueError(
                "Select at least one unary or binary operation in the Data tab."
            )

        entry = _eng._DISPATCH.get(key)
        if entry is None:
            raise ValueError(f"Unknown method key: {key!r}")

        fn = entry[2]
        kwargs: dict = {"combo_cap": combo_cap, "total_cap": total_cap}

        if key == "greedy_forward":
            kwargs["max_features"] = int(op.get("max_features", _eng.GREEDY_MAX_FEATURES_DEFAULT))
            kwargs["preselect_k"]  = int(op.get("preselect_k",  _eng.SEARCH_PRESELECT_DEFAULT))
            total_steps = kwargs["max_features"]

        elif key == "greedy_backward":
            kwargs["min_features"] = int(op.get("min_features", _eng.GREEDY_MIN_FEATURES_DEFAULT))
            kwargs["preselect_k"]  = int(op.get("preselect_k",  _eng.GREEDY_BACKWARD_PRESELECT_DEFAULT))
            total_steps = kwargs["preselect_k"]

        elif key == "armhc":
            kwargs["max_iter"]   = int(op.get("max_iter",   _eng.SEARCH_MAX_ITER_DEFAULT))
            kwargs["k_init"]     = int(op.get("k_init",     _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["init_mutation_intensity"] = (
                float(op.get("init_mutation_pct", _eng.ARMHC_INIT_MUTATION_INTENSITY_RATIO * 100.0)) / 100.0
            )
            kwargs["adapt_lo"]    = float(op.get("adapt_lo_pct", _eng.ARMHC_ADAPT_LO_THRESHOLD * 100.0)) / 100.0
            kwargs["adapt_hi"]    = float(op.get("adapt_hi_pct", _eng.ARMHC_ADAPT_HI_THRESHOLD * 100.0)) / 100.0
            kwargs["adapt_window"]            = int(op.get("adapt_window",          _eng.ARMHC_ADAPT_WINDOW))
            kwargs["intensity_scale_up"]      = float(op.get("intensity_scale_up",  _eng.ARMHC_INTENSITY_SCALE_UP_RATIO))
            kwargs["intensity_scale_down_max"] = float(op.get("intensity_scale_down_max", _eng.ARMHC_INTENSITY_SCALE_DOWN_MAX_RATIO))
            kwargs["preselect_k"] = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]          = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key in ("sa_metropolis", "sa_glauber"):
            kwargs["max_iter"]    = int(op.get("max_iter",  _eng.SEARCH_MAX_ITER_DEFAULT))
            kwargs["k_init"]      = int(op.get("k_init",    _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["t0"]          = float(op.get("t0",      _eng.SA_T0_DEFAULT))
            kwargs["cooling"]     = float(op.get("cooling", _eng.SA_COOLING_RATIO_DEFAULT))
            kwargs["preselect_k"] = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]          = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key == "rts":
            kwargs["max_iter"]         = int(op.get("max_iter",       _eng.SEARCH_MAX_ITER_DEFAULT))
            kwargs["k_init"]           = int(op.get("k_init",         _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["init_tabu_tenure"] = int(op.get("init_tabu_tenure", _eng.RTS_INIT_TABU_TENURE_DEFAULT))
            kwargs["tabu_adapt_step"]  = int(op.get("tabu_adapt_step",  _eng.RTS_TABU_ADAPT_STEP_DEFAULT))
            kwargs["n_candidates"]     = int(op.get("n_candidates",     _eng.RTS_CANDIDATES_PER_ITER_DEFAULT))
            kwargs["preselect_k"]      = int(op.get("preselect_k",      _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]               = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key == "cem":
            kwargs["max_iter"]    = int(op.get("max_iter",    _eng.CEM_MAX_ITER_DEFAULT))
            kwargs["k_init"]      = int(op.get("k_init",      _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["n_samples"]   = int(op.get("n_samples",   _eng.CEM_N_SAMPLES_DEFAULT))
            kwargs["elite_frac"]  = float(op.get("elite_frac", _eng.CEM_ELITE_FRAC_DEFAULT))
            kwargs["smoothing"]   = float(op.get("smoothing",  _eng.CEM_SMOOTHING_DEFAULT))
            kwargs["preselect_k"] = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]          = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key == "pt":
            kwargs["max_iter"]     = int(op.get("max_iter",    _eng.PT_MAX_ITER_DEFAULT))
            kwargs["k_init"]       = int(op.get("k_init",      _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["n_replicas"]   = int(op.get("n_replicas",  _eng.PT_N_REPLICAS_DEFAULT))
            kwargs["t_min"]        = float(op.get("t_min",     _eng.PT_T_MIN_DEFAULT))
            kwargs["t_max"]        = float(op.get("t_max",     _eng.PT_T_MAX_DEFAULT))
            kwargs["swap_interval"] = int(op.get("swap_interval", _eng.PT_SWAP_INTERVAL_DEFAULT))
            kwargs["preselect_k"]  = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]           = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key == "qa":
            kwargs["max_iter"]    = int(op.get("max_iter",   _eng.SEARCH_MAX_ITER_DEFAULT))
            kwargs["k_init"]      = int(op.get("k_init",     _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["n_trotter"]   = int(op.get("n_trotter",  _eng.QA_N_TROTTER_DEFAULT))
            kwargs["gamma_init"]  = float(op.get("gamma_init", _eng.QA_GAMMA_INIT_DEFAULT))
            kwargs["gamma_final"] = float(op.get("gamma_final", _eng.QA_GAMMA_FINAL_DEFAULT))
            kwargs["t_fixed"]     = float(op.get("t_fixed",  _eng.QA_T_FIXED_DEFAULT))
            kwargs["preselect_k"] = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]          = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key == "pso":
            kwargs["max_iter"]    = int(op.get("max_iter",    _eng.SEARCH_MAX_ITER_DEFAULT))
            kwargs["k_init"]      = int(op.get("k_init",      _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["n_particles"] = int(op.get("n_particles", _eng.PSO_N_PARTICLES_DEFAULT))
            kwargs["w"]           = float(op.get("w",         _eng.PSO_W_DEFAULT))
            kwargs["c1"]          = float(op.get("c1",        _eng.PSO_C1_DEFAULT))
            kwargs["c2"]          = float(op.get("c2",        _eng.PSO_C2_DEFAULT))
            kwargs["v_max"]       = float(op.get("v_max",     _eng.PSO_V_MAX_DEFAULT))
            kwargs["preselect_k"] = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]          = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        elif key == "gp":
            kwargs["max_iter"]       = int(op.get("max_iter",       _eng.GP_MAX_GENS_DEFAULT))
            kwargs["k_init"]         = int(op.get("k_init",         _eng.SEARCH_K_INIT_DEFAULT))
            kwargs["pop_size"]       = int(op.get("pop_size",       _eng.GP_POP_SIZE_DEFAULT))
            kwargs["crossover_prob"] = float(op.get("crossover_prob", _eng.GP_CROSSOVER_PROB_DEFAULT))
            kwargs["mutation_prob"]  = float(op.get("mutation_prob", _eng.GP_MUTATION_PROB_DEFAULT))
            kwargs["tournament_size"] = int(op.get("tournament_size", _eng.GP_TOURNAMENT_SIZE_DEFAULT))
            kwargs["elitism"]        = int(op.get("elitism",        _eng.GP_ELITISM_COUNT_DEFAULT))
            kwargs["parsimony_coeff"] = float(op.get("parsimony_coeff", _eng.GP_PARSIMONY_COEFF_DEFAULT))
            kwargs["preselect_k"]    = int(op.get("preselect_k", _eng.SEARCH_PRESELECT_DEFAULT))
            kwargs["rs"]             = int(op.get("rs", 42))
            total_steps = kwargs["max_iter"]

        else:
            total_steps = 100

        self._op_cache = op
        self._sp_cache = sp

        worker = _SearchWorker(fn, df, features, target_col, selected_ops, kwargs, self)
        worker.progress.connect(self._on_search_progress)
        worker.finished_ok.connect(self._on_search_finished)
        worker.error_occurred.connect(self._on_search_error)
        self._worker = worker

        self._progress_bar.setRange(0, total_steps)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%v / " + str(total_steps) + "  (%p%)")
        self._progress_bar.setVisible(True)
        self._status_lbl.setText("Initializing…")
        self._status_lbl.setVisible(True)

        worker.start()
        return True

    # ── Search worker callbacks ───────────────────────────────────────────

    def _on_search_progress(self, current: int, total: int, best_score: float) -> None:
        self._progress_bar.setValue(current)
        self._status_lbl.setText(
            f"Iteration {current:,} / {total:,}  ·  Best R\u00b2 = {best_score:.4f}"
        )

    def _on_search_finished(self, result) -> None:
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText("Generate")
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        if not result.is_valid:
            QMessageBox.warning(
                self, "Degraded Result",
                f"{result.method_title}: no features selected.  "
                "Try increasing iterations or adjusting the operator set.",
            )

        try:
            fig = _render_search_chart(result, self._op_cache, self._sp_cache)
            self._set_canvas(fig)
        except Exception as exc:
            log.exception("DescriptorDialog._on_search_finished render")
            QMessageBox.critical(self, "Render Error", str(exc))

    def _on_search_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText("Generate")
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        QMessageBox.critical(self, "Computation Error", msg)

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
        prefix = ep.get("prefix", "descriptors") or "descriptors"

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
