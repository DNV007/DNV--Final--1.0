"""
DNV Scientific Module
---------------------
Role:
    Provides the FeatureRankingDialog — a fully configurable, canvas-embedded
    dialog for 32 feature-importance and ranking visualisations, with exhaustive
    per-method options, a shared style panel, and a complete export/save panel.

Scientific Context:
    Accepts a method key and a SessionStore; dispatches to fi_engine for
    feature-score computation; renders the result as one of five figure types:
    importance bar chart, SHAP beeswarm proxy, RFE step curve, feature-map
    heatmap, or interaction heatmap.  All style parameters are applied through
    a single _finalise() / _apply_style() call path.

Invariants:
    - No computation occurs at dialog construction time; only on Generate.
    - Every numeric constant is a named Final with a unit suffix.
    - _apply_style() is the single site that writes per-axes Qt/matplotlib
      state; no display attribute is set elsewhere in the renderers.
    - The single FigureCanvas is replaced atomically on each Generate call.

Assumptions:
    - store.asset is a valid DataAsset with a DataFrame containing ≥ 2
      numeric columns.
    - fi_engine functions are pure and raise ImportError for missing deps.
    - matplotlib backend is "Agg" (set at application startup in main.py).

Failure Modes:
    - Fewer than 2 numeric columns: Generate shows an error dialog, no crash.
    - Missing optional dependency (torch): ImportError from fi_engine is caught
      and shown as an install-hint message.
    - Singular / ill-conditioned matrices: is_valid=False shown as annotation.

Provenance:
    - This module emits no transformation metadata.  Provenance originates
      in the calling FeatureRankingWindow/FeatureRankingDialog session.
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
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from gui_session import SessionStore
from gui_style import (
    ColorBtn, hrow, eyebrow, section_sep, scrolled, combo, spin, dspin, check,
    make_style_panel, make_export_panel,
    FONT_FAMILIES, LEGEND_LOCS, SCALES,
    is_categorical_axis, rc_ctx, apply_tick_format, apply_style, finalise_figure,
)
import fi_engine as _eng
import stability_engine as _stab

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout and default constants — all values carry explicit units in names
# ---------------------------------------------------------------------------

#: Fixed width of the left configuration panel [px].
LEFT_PANEL_WIDTH_PX: Final[int] = 320

#: Minimum height of the canvas area [px].
CANVAS_MIN_HEIGHT_PX: Final[int] = 520

#: Minimum width of the canvas area [px].
CANVAS_MIN_WIDTH_PX: Final[int] = 600

#: Default figure width [inches].
DEFAULT_FIG_WIDTH_IN: Final[float] = 12.0

#: Default figure height [inches].
DEFAULT_FIG_HEIGHT_IN: Final[float] = 7.0

#: Default figure DPI for screen rendering.
DEFAULT_SCREEN_DPI: Final[int] = 100

#: Default export DPI.
DEFAULT_EXPORT_DPI: Final[int] = 150

#: Spacing added after each section separator in the left panel [px].
SECTION_SPACING_PX: Final[int] = 4

#: Set of method keys that produce heatmap visualisations (not bar charts).
_HEATMAP_KEYS: Final[frozenset] = frozenset({"feature_maps", "feature_interactions"})

#: Set of method keys with non-standard renderers.
_SPECIAL_KEYS: Final[frozenset] = frozenset({"rfe", "shap", "feature_maps", "feature_interactions"})

_COLORMAPS: Final[List[str]] = [
    "coolwarm", "RdBu", "seismic", "bwr", "vlag", "icefire",
    "viridis", "plasma", "inferno", "magma",
    "Blues", "Greens", "Oranges", "Reds", "Purples",
    "YlOrRd", "BuGn", "PuBu",
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

#: Stability bar color for family-stable features.
STAB_BAR_STABLE_COLOR: Final[str] = "#27ae60"

#: Stability bar color for contingent features.
STAB_BAR_CONTINGENT_COLOR: Final[str] = "#e74c3c"

#: Maximum height of the stability text report panel [px].
STAB_REPORT_MAX_HEIGHT_PX: Final[int] = 120

#: Default figure width for stability diagnostic plots [inches].
STAB_FIG_WIDTH_IN: Final[float] = 14.0

#: Default figure height for stability diagnostic plots [inches].
STAB_FIG_HEIGHT_IN: Final[float] = 6.0


# ---------------------------------------------------------------------------
# _ColorBtn — inline colour picker
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
        self._color = c
        self._refresh()


# ---------------------------------------------------------------------------
# Widget micro-helpers
# ---------------------------------------------------------------------------

def _combo(items: List[str], default: str = "") -> QComboBox:
    w = QComboBox()
    w.addItems(items)
    idx = items.index(default) if default in items else 0
    w.setCurrentIndex(idx)
    return w


def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w


def _dspin(lo: float, hi: float, val: float, step: float = 0.05) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step); w.setDecimals(3)
    return w


def _check(state: bool, label: str = "") -> QCheckBox:
    w = QCheckBox(label); w.setChecked(state); return w


def _hrow(label: str, widget: QWidget) -> QWidget:
    row = QWidget()
    lay = QHBoxLayout(row); lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label); lbl.setMinimumWidth(130)
    lay.addWidget(lbl); lay.addWidget(widget, 1)
    return row


def _eyebrow(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("ReportPanelEyebrow"); return lbl


def _section_sep() -> QFrame:
    f = QFrame(); f.setObjectName("Divider")
    f.setFixedHeight(1); f.setFrameShape(QFrame.HLine); return f


def _scrolled(inner: QWidget) -> QScrollArea:
    sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(inner)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return sa


def _read(w) -> Any:
    """Read any supported control type."""
    if isinstance(w, QComboBox):      return w.currentText()
    if isinstance(w, QCheckBox):      return w.isChecked()
    if isinstance(w, QSpinBox):       return w.value()
    if isinstance(w, QDoubleSpinBox): return w.value()
    if isinstance(w, _ColorBtn):      return w.color()
    if isinstance(w, QLineEdit):      return w.text().strip()
    return None


# ---------------------------------------------------------------------------
# Data tab — target + feature selector (all 32 methods use a target)
# ---------------------------------------------------------------------------

class _MultiColSelector(QListWidget):
    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.addItem(item)
        self.setMaximumHeight(200)

    def selected(self) -> List[str]:
        return [self.item(i).text() for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked]

    def select_all(self) -> None:
        for i in range(self.count()): self.item(i).setCheckState(Qt.Checked)

    def deselect_all(self) -> None:
        for i in range(self.count()): self.item(i).setCheckState(Qt.Unchecked)


class _DataPage(QWidget):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)

        num_cols = list(df.select_dtypes(include="number").columns)
        all_cols = list(df.columns)

        lay.addWidget(_eyebrow("TARGET COLUMN"))
        self._target = _combo(all_cols)
        lay.addWidget(_hrow("Target:", self._target))

        lay.addSpacing(4)
        lay.addWidget(_eyebrow("FEATURE COLUMNS"))

        hint = QLabel("Check features to include  (numeric only used).")
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._sel = _MultiColSelector(num_cols)
        lay.addWidget(self._sel)

        btn_row = QHBoxLayout()
        ba = QPushButton("All"); bn = QPushButton("None")
        ba.clicked.connect(self._sel.select_all)
        bn.clicked.connect(self._sel.deselect_all)
        btn_row.addWidget(ba); btn_row.addWidget(bn); btn_row.addStretch()
        lay.addLayout(btn_row)

        lay.addSpacing(4)
        lay.addWidget(_eyebrow("PREPROCESSING"))
        self._standardize = _check(True, " Standardise features (zero mean, unit variance)")
        lay.addWidget(self._standardize)

        lay.addStretch()

    def target(self) -> str:
        return self._target.currentText()

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        sel = [c for c in self._sel.selected() if c != self.target()]
        if not sel:
            sel = [c for c in df.select_dtypes("number").columns if c != self.target()]
        return df[sel + [self.target()]]

    def standardize(self) -> bool:
        return self._standardize.isChecked()


# ---------------------------------------------------------------------------
# Options panel — exhaustive per-method controls
# ---------------------------------------------------------------------------

def _make_options(key: str) -> Tuple[QWidget, Callable[[], dict]]:
    """
    Build the Options tab for the given feature-ranking method key.

    Returns (widget, getter) where getter() → dict of option values.
    """
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
    controls: Dict[str, Any] = {}

    def _add(label: str, widget, key_: str) -> None:
        controls[key_] = widget; lay.addWidget(_hrow(label, widget))

    def _sec(text: str) -> None:
        lay.addWidget(_section_sep()); lay.addWidget(_eyebrow(text))
        lay.addSpacing(SECTION_SPACING_PX)

    # ── Shared bar-chart controls (most methods) ──────────────────────────
    if key not in _HEATMAP_KEYS:
        _sec("BAR DISPLAY")
        _add("Bar color:",         _ColorBtn("#3a7bd5"),                                  "bar_color")
        _add("Negative color:",    _ColorBtn("#e74c3c"),                                  "neg_color")
        _add("Bar alpha:",         _dspin(0.3, 1.0, 0.85, 0.05),                         "bar_alpha")
        _add("Edge color:",        _ColorBtn("#1a3a6a"),                                  "edge_color")
        _add("Edge linewidth:",    _dspin(0.0, 3.0, 0.5, 0.1),                           "edge_lw")
        _add("Top N features:",    _spin(1, 200, 20),                                     "top_n")
        _add("Orientation:",       _combo(["horizontal", "vertical"], "horizontal"),       "orientation")
        _add("Show value labels:", _check(True),                                          "show_labels")
        _add("Label font size:",   _spin(5, 16, 8),                                       "label_fontsize")
        _add("Sort descending:",   _check(True),                                          "sort_desc")
        _add("Abs values:",        _check(False),                                         "abs_values")
        _add("Normalize 0-1:",     _check(False),                                         "normalize")

    # ── Heatmap controls (feature_maps, feature_interactions) ─────────────
    if key in _HEATMAP_KEYS:
        _sec("HEATMAP")
        _add("Colormap:",          _combo(_COLORMAPS, "coolwarm"),                        "cmap")
        _add("Annotate cells:",    _check(True),                                          "annotate")
        _add("Annot font size:",   _spin(5, 20, 8),                                       "annot_fontsize")
        _add("vmin:",              _dspin(-1.0, 0.0, -1.0, 0.05),                        "vmin")
        _add("vmax:",              _dspin(0.0, 1.0, 1.0, 0.05),                          "vmax")
        _add("Mask upper tri:",    _check(True),                                          "mask_upper")
        _add("Cell linewidths:",   _dspin(0.0, 3.0, 0.4, 0.1),                          "linewidths")
        _add("Cell line color:",   _ColorBtn("#aaaaaa"),                                  "linecolor")
        _add("Colorbar shrink:",   _dspin(0.3, 1.0, 0.8, 0.05),                         "cbar_shrink")
        _add("Colorbar label:",    QLineEdit(),                                           "cbar_label")

    # ── SHAP beeswarm ─────────────────────────────────────────────────────
    if key == "shap":
        _sec("SHAP OPTIONS")
        _add("N background:",      _spin(10, 500, 50),                                    "n_background")
        _add("N coalitions:",      _spin(100, 5000, 500),                                 "n_coalitions")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")
        _sec("BEESWARM DISPLAY")
        _add("Colormap:",          _combo(_COLORMAPS, "coolwarm"),                        "cmap")
        _add("Point size:",        _spin(5, 200, 20),                                     "point_size")
        _add("Point alpha:",       _dspin(0.1, 1.0, 0.6, 0.05),                         "point_alpha")

    # ── RFE ───────────────────────────────────────────────────────────────
    elif key == "rfe":
        _sec("RFE OPTIONS")
        _add("N estimators (RF):", _spin(10, 500, 100),                                   "n_estimators")
        _add("CV folds:",          _spin(2, 10, 5),                                       "cv")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")
        _sec("CURVE DISPLAY")
        _add("Line color:",        _ColorBtn("#3a7bd5"),                                  "line_color")
        _add("Fill color:",        _ColorBtn("#a8c4e8"),                                  "fill_color")
        _add("Fill alpha:",        _dspin(0.1, 0.8, 0.25, 0.05),                        "fill_alpha")
        _add("Line width:",        _dspin(0.5, 5.0, 1.5, 0.5),                          "line_width")
        _add("Marker:",            _combo(["o", "s", "^", "D", "none"], "o"),            "marker")
        _add("Marker size:",       _spin(2, 15, 5),                                       "marker_size")

    # ── Tree-based methods ────────────────────────────────────────────────
    elif key in ("decision_tree", "random_forest", "gradient_boosting", "boruta"):
        _sec(f"{key.upper().replace('_', ' ')} OPTIONS")
        if key != "decision_tree":
            _add("N estimators:",  _spin(10, 1000, 100),                                  "n_estimators")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")
        if key == "boruta":
            _add("N trials:",      _spin(10, 200, 50),                                    "n_trials")

    # ── Regularisation methods ────────────────────────────────────────────
    elif key in ("lasso", "elasticnet"):
        _sec(f"{key.upper().replace('_', ' ')} OPTIONS")
        _add("Alpha (λ):",         _dspin(1e-4, 50.0, 0.01, 0.01),                      "alpha")
        _add("Max iterations:",    _spin(100, 50000, 5000),                               "max_iter")
        if key == "elasticnet":
            _add("L1 ratio:",      _dspin(0.0, 1.0, 0.5, 0.05),                         "l1_ratio")

    elif key == "ridge":
        _sec("RIDGE OPTIONS")
        _add("Alpha (λ):",         _dspin(1e-4, 1000.0, 1.0, 0.1),                      "alpha")

    elif key == "sparse_pca":
        _sec("SPARSE PCA OPTIONS")
        _add("N components:",      _spin(1, 50, 5),                                       "n_components")
        _add("Alpha (sparsity):",  _dspin(0.01, 10.0, 1.0, 0.1),                        "alpha")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")

    # ── LIME ─────────────────────────────────────────────────────────────
    elif key == "lime":
        _sec("LIME OPTIONS")
        _add("N perturbations:",   _spin(100, 5000, 1000),                                "n_perturbations")
        _add("N query points:",    _spin(1, 50, 5),                                       "n_display")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")

    # ── Permutation importance ────────────────────────────────────────────
    elif key == "permutation_importance":
        _sec("PERMUTATION OPTIONS")
        _add("N estimators (RF):", _spin(10, 500, 100),                                   "n_estimators")
        _add("N repeats:",         _spin(2, 50, 10),                                      "n_repeats")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")
        _sec("ERROR BAR DISPLAY")
        _add("Show error bars:",   _check(True),                                          "show_errorbars")
        _add("Error bar color:",   _ColorBtn("#555555"),                                  "err_color")
        _add("Capsize:",           _spin(0, 10, 3),                                       "capsize")

    # ── Feature interactions ──────────────────────────────────────────────
    elif key == "feature_interactions":
        _sec("INTERACTION OPTIONS")
        _add("N estimators (RF):", _spin(10, 500, 100),                                   "n_estimators")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")
        _add("Threshold:",         _dspin(0.0, 1.0, 0.0, 0.01),                         "threshold")

    # ── GA / PSO ─────────────────────────────────────────────────────────
    elif key == "genetic_algorithm":
        _sec("GENETIC ALGORITHM OPTIONS")
        _add("Population size:",   _spin(10, 500, 50),                                    "pop_size")
        _add("N generations:",     _spin(10, 1000, 100),                                  "n_generations")
        _add("Mutation rate:",     _dspin(0.001, 0.5, 0.01, 0.005),                     "mutation_rate")
        _add("N estimators (RF):", _spin(10, 500, 50),                                    "n_estimators")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")

    elif key == "pso":
        _sec("PSO OPTIONS")
        _add("N particles:",       _spin(5, 200, 30),                                     "n_particles")
        _add("N iterations:",      _spin(10, 1000, 100),                                  "n_iterations")
        _add("N estimators (RF):", _spin(10, 500, 50),                                    "n_estimators")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")

    # ── ReliefF ───────────────────────────────────────────────────────────
    elif key == "relieff":
        _sec("RELIEFF OPTIONS")
        _add("N neighbors:",       _spin(1, 50, 10),                                      "n_neighbors")

    # ── MRMR ─────────────────────────────────────────────────────────────
    elif key == "mrmr":
        _sec("MRMR OPTIONS")
        _add("N features select:", _spin(1, 100, 10),                                     "n_features_to_select")

    # ── Recursive Feature Clustering ──────────────────────────────────────
    elif key == "recursive_feature_clustering":
        _sec("RFC OPTIONS")
        _add("N clusters:",        _spin(2, 50, 5),                                       "n_clusters")

    # ── Autoencoder importance ────────────────────────────────────────────
    elif key == "autoencoder_importance":
        _sec("AUTOENCODER MLP OPTIONS")
        _add("Hidden layer 1:",    _spin(4, 256, 32),                                     "hidden1")
        _add("Hidden layer 2:",    _spin(0, 256, 16),                                     "hidden2")
        _add("Max iterations:",    _spin(50, 2000, 300),                                  "max_iter")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")

    # ── Deep Learning Attribution ─────────────────────────────────────────
    elif key == "deep_learning_attribution":
        _sec("DEEP LEARNING OPTIONS")
        _add("Hidden size:",       _spin(8, 512, 64),                                     "hidden_size")
        _add("N epochs:",          _spin(10, 2000, 100),                                  "n_epochs")
        _add("Learning rate:",     _dspin(1e-5, 0.1, 1e-3, 1e-4),                       "lr")
        _add("Random state:",      _spin(0, 9999, 42),                                    "random_state")

    # ── Markov Blanket ────────────────────────────────────────────────────
    elif key == "markov_blanket":
        _sec("MARKOV BLANKET OPTIONS")
        note = QLabel("Uses IAMB algorithm.  Target column is set in the Data tab.")
        note.setObjectName("SectionHint"); note.setWordWrap(True)
        lay.addWidget(note)

    lay.addStretch()

    def _getter() -> dict:
        return {k: _read(v) for k, v in controls.items()}

    return w, _getter


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_bar(
    scores: pd.Series,
    title: str,
    xlabel: str,
    op: dict,
    sp: dict,
    errors: Optional[np.ndarray] = None,
) -> Figure:
    """
    Horizontal or vertical bar chart of feature importance scores.

    Supports positive/negative colour splitting, value labels, error bars,
    top-N filtering, absolute values, and 0-1 normalisation.
    """
    data = scores.copy()
    if op.get("abs_values", False):
        data = data.abs()
    if op.get("normalize", False):
        dmax = data.abs().max()
        if dmax > 0:
            data = data / dmax
    if op.get("sort_desc", True):
        data = data.sort_values(ascending=False)
    top_n = int(op.get("top_n", 20))
    if len(data) > top_n:
        data = data.iloc[:top_n]
        if errors is not None:
            # align errors to the same index
            err_s = pd.Series(errors, index=scores.index)
            errors = err_s.loc[data.index].values

    bar_color = op.get("bar_color", "#3a7bd5")
    neg_color = op.get("neg_color", "#e74c3c")
    bar_alpha  = float(op.get("bar_alpha", 0.85))
    edge_color = op.get("edge_color", "#1a3a6a")
    edge_lw    = float(op.get("edge_lw", 0.5))
    show_labels= op.get("show_labels", True)
    lbl_fs     = int(op.get("label_fontsize", 8))
    orientation= op.get("orientation", "horizontal")
    bar_colors = [neg_color if v < 0 else bar_color for v in data.values]

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN)

    with plt.rc_context(rc_ctx(sp)):
        fig, ax = plt.subplots(figsize=(w_in, h_in))

        err_kw: dict = {"elinewidth": 0.8, "capthick": 0.8}
        show_errorbars = bool(op.get("show_errorbars", False)) and errors is not None

        if orientation == "horizontal":
            bars = ax.barh(
                range(len(data)), data.values,
                color=bar_colors, alpha=bar_alpha,
                edgecolor=edge_color, linewidth=edge_lw,
                xerr=(errors if show_errorbars else None),
                error_kw={**err_kw, "ecolor": op.get("err_color", "#555555"),
                           "capsize": int(op.get("capsize", 3))},
            )
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(0, color="#444", linewidth=0.7, linestyle="--")
            ax.set_xlabel(sp.get("xlabel") or xlabel, fontsize=sp.get("label_fontsize", 10))
            if show_labels:
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                for bar_, val in zip(bars, data.values):
                    ax.text(
                        val + 0.002 * x_range,
                        bar_.get_y() + bar_.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=lbl_fs,
                    )
        else:
            bars = ax.bar(
                range(len(data)), data.values,
                color=bar_colors, alpha=bar_alpha,
                edgecolor=edge_color, linewidth=edge_lw,
                yerr=(errors if show_errorbars else None),
                error_kw={**err_kw, "ecolor": op.get("err_color", "#555555"),
                           "capsize": int(op.get("capsize", 3))},
            )
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha="right", fontsize=9)
            ax.axhline(0, color="#444", linewidth=0.7, linestyle="--")
            ax.set_ylabel(sp.get("ylabel") or xlabel, fontsize=sp.get("label_fontsize", 10))
            if show_labels:
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                for bar_, val in zip(bars, data.values):
                    ax.text(
                        bar_.get_x() + bar_.get_width() / 2,
                        val + 0.005 * y_range,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=lbl_fs,
                    )

        ax.set_title(
            sp.get("title") or title,
            fontsize=sp.get("title_fontsize", 12), fontweight="bold",
        )
        ax.grid(
            True, axis="x" if orientation == "horizontal" else "y",
            linestyle="--", linewidth=0.4, alpha=0.5,
        )

    return finalise_figure(fig, sp)


def _render_heatmap(
    matrix: pd.DataFrame,
    title: str,
    op: dict,
    sp: dict,
) -> Figure:
    """Seaborn heatmap for feature_maps and feature_interactions."""
    import seaborn as sns

    mask = (
        np.triu(np.ones_like(matrix.values, dtype=bool))
        if op.get("mask_upper", True) else None
    )
    cbar_kws: dict = {"shrink": float(op.get("cbar_shrink", 0.8))}
    cbar_label = op.get("cbar_label", "")
    if cbar_label: cbar_kws["label"] = cbar_label

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN)

    with plt.rc_context(rc_ctx(sp)):
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        sns.heatmap(
            matrix,
            mask=mask,
            annot=op.get("annotate", True),
            fmt=".2f",
            annot_kws={"size": int(op.get("annot_fontsize", 8))},
            cmap=op.get("cmap", "coolwarm"),
            linewidths=float(op.get("linewidths", 0.4)),
            linecolor=op.get("linecolor", "#aaaaaa"),
            vmin=float(op.get("vmin", -1.0)),
            vmax=float(op.get("vmax",  1.0)),
            cbar_kws=cbar_kws,
            ax=ax,
        )
        ax.set_title(
            sp.get("title") or title,
            fontsize=sp.get("title_fontsize", 12), fontweight="bold",
        )

    return finalise_figure(fig, sp)


def _render_rfe_curve(
    result: "_eng.FeatureRankingResult",
    op: dict,
    sp: dict,
) -> Figure:
    """RFE cross-validation accuracy vs number of features curve."""
    grid_scores = result.extra.get("grid_scores")
    support     = result.extra.get("support")
    scores_s    = result.scores

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN * 0.85)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN * 0.90)

    line_color  = op.get("line_color",  "#3a7bd5")
    fill_color  = op.get("fill_color",  "#a8c4e8")
    fill_alpha  = float(op.get("fill_alpha", 0.25))
    line_width  = float(op.get("line_width",  1.5))
    marker      = op.get("marker", "o")
    marker_size = int(op.get("marker_size", 5))
    if marker == "none":
        marker = None

    with plt.rc_context(rc_ctx(sp)):
        fig, axes = plt.subplots(1, 2, figsize=(w_in, h_in))
        ax_cv, ax_imp = axes

        # Left: CV score curve
        if grid_scores is not None and len(grid_scores) > 0:
            n_range = np.arange(1, len(grid_scores) + 1)
            means   = np.array([np.mean(s) for s in grid_scores])
            stds    = np.array([np.std(s)  for s in grid_scores])
            ax_cv.plot(n_range, means, color=line_color, lw=line_width,
                       marker=marker, markersize=marker_size)
            ax_cv.fill_between(n_range, means - stds, means + stds,
                               alpha=fill_alpha, color=fill_color)
            best_n = int(n_range[np.argmax(means)])
            ax_cv.axvline(best_n, color="#e74c3c", linestyle="--", linewidth=0.9,
                          label=f"Best n={best_n}")
            ax_cv.legend(fontsize=8)
        else:
            ax_cv.text(0.5, 0.5, "CV scores not available",
                       ha="center", va="center", transform=ax_cv.transAxes)

        ax_cv.set_xlabel("Number of Features", fontsize=10)
        ax_cv.set_ylabel("CV Score", fontsize=10)
        ax_cv.set_title(sp.get("title") or "RFE — Cross-Validation Score",
                        fontsize=sp.get("title_fontsize", 11), fontweight="bold")
        ax_cv.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

        # Right: selected feature importances
        data = scores_s.sort_values(ascending=False).head(int(op.get("top_n", 20)))
        bar_color = op.get("bar_color", "#3a7bd5")
        ax_imp.barh(range(len(data)), data.values,
                    color=bar_color, alpha=float(op.get("bar_alpha", 0.85)),
                    edgecolor=op.get("edge_color", "#1a3a6a"),
                    linewidth=float(op.get("edge_lw", 0.5)))
        ax_imp.set_yticks(range(len(data)))
        ax_imp.set_yticklabels(data.index, fontsize=9)
        ax_imp.invert_yaxis()
        ax_imp.set_xlabel("1 / Ranking", fontsize=10)
        ax_imp.set_title("Feature Rankings", fontsize=sp.get("title_fontsize", 11),
                         fontweight="bold")
        ax_imp.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    return finalise_figure(fig, sp)


def _render_shap_beeswarm(
    result: "_eng.FeatureRankingResult",
    op: dict,
    sp: dict,
) -> Figure:
    """
    SHAP beeswarm proxy: scatter of SHAP values per feature.

    Uses shap_matrix (n_samples × n_features) from result.extra.
    Points are coloured by normalised feature value magnitude.
    """
    shap_matrix = result.extra.get("shap_matrix")
    scores      = result.scores

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN)

    with plt.rc_context(rc_ctx(sp)):
        if shap_matrix is None or not isinstance(shap_matrix, np.ndarray):
            # Fall back to plain bar chart
            return _render_bar(scores,
                               "SHAP Mean |Values|", "|SHAP|",
                               op, sp)

        top_n    = int(op.get("top_n", 20))
        cmap     = op.get("cmap", "coolwarm")
        pt_size  = int(op.get("point_size", 20))
        pt_alpha = float(op.get("point_alpha", 0.6))

        # Sort features by mean |SHAP|
        order  = scores.sort_values(ascending=False).head(top_n).index.tolist()
        n_feat = len(order)

        feat_idx = {f: i for i, f in enumerate(result.scores.index)}
        fig, ax  = plt.subplots(figsize=(w_in, h_in))

        cmap_obj = plt.get_cmap(cmap)

        for row_pos, feat in enumerate(reversed(order)):
            fi  = feat_idx.get(feat)
            if fi is None:
                continue
            col = shap_matrix[:, fi]
            # colour by normalised column magnitude
            norm = col / (np.abs(col).max() + 1e-12)
            colors = cmap_obj((norm + 1) / 2)
            jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(col))
            ax.scatter(col, row_pos + jitter,
                       c=colors, s=pt_size, alpha=pt_alpha, linewidths=0)

        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(list(reversed(order)), fontsize=9)
        ax.axvline(0, color="#444", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value", fontsize=sp.get("label_fontsize", 10))
        ax.set_title(
            sp.get("title") or f"SHAP Beeswarm — top {top_n} features",
            fontsize=sp.get("title_fontsize", 12), fontweight="bold",
        )
        ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                                   norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        try:
            cb = fig.colorbar(sm, ax=ax, shrink=0.6)
            cb.set_label("Feature value (normalised)", fontsize=8)
        except Exception:
            pass

    return finalise_figure(fig, sp)


# ---------------------------------------------------------------------------
# Stability diagnostic renderer
# ---------------------------------------------------------------------------

def _render_feature_stability(
    fs: _stab.FeatureStabilityResult,
    sp: Dict[str, Any],
) -> Figure:
    """Render a 2-panel stability diagnostic figure.

    Left : per-feature σ²_f bars colour-coded by stability label.
    Right: K × K Kendall τ heatmap.
    """
    fig, (ax_var, ax_tau) = plt.subplots(
        1, 2, figsize=(STAB_FIG_WIDTH_IN, STAB_FIG_HEIGHT_IN),
        dpi=DEFAULT_SCREEN_DPI,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── Left: variance bar chart ─────────────────────────────────────────
    features = fs.sigma2_f.sort_values(ascending=True).index.tolist()
    variances = [fs.sigma2_f[f] for f in features]
    colors = [
        STAB_BAR_STABLE_COLOR
        if fs.stability_labels.get(f) == _stab.LABEL_FAMILY_STABLE
        else STAB_BAR_CONTINGENT_COLOR
        for f in features
    ]
    y_pos = range(len(features))
    ax_var.barh(y_pos, variances, color=colors, edgecolor="white", linewidth=0.5)
    ax_var.set_yticks(list(y_pos))
    ax_var.set_yticklabels(features, fontsize=8)
    ax_var.set_xlabel("σ²_f  (importance variance across variants)")
    ax_var.set_title(
        f"Feature Stability — {fs.n_variants_used} variants\n"
        f"mean τ = {fs.mean_kendall_tau:.3f}   "
        f"top-{_stab.TOP_K_OVERLAP_DEFAULT} overlap = {fs.top_k_overlap:.1%}",
        fontsize=10,
    )
    # Legend
    from matplotlib.patches import Patch
    ax_var.legend(
        handles=[
            Patch(facecolor=STAB_BAR_STABLE_COLOR, label="family_stable"),
            Patch(facecolor=STAB_BAR_CONTINGENT_COLOR, label="contingent"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax_var.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    # ── Right: Kendall τ heatmap ─────────────────────────────────────────
    K = fs.kendall_tau_matrix.shape[0]
    im = ax_tau.imshow(
        fs.kendall_tau_matrix,
        cmap="RdYlGn", vmin=-1, vmax=1,
        aspect="equal",
    )
    ax_tau.set_xticks(range(K))
    ax_tau.set_yticks(range(K))
    ax_tau.set_xticklabels([f"v{i}" for i in range(K)], fontsize=8)
    ax_tau.set_yticklabels([f"v{i}" for i in range(K)], fontsize=8)
    ax_tau.set_title("Pairwise Kendall τ", fontsize=10)
    # Annotate cells
    for i in range(K):
        for j in range(K):
            ax_tau.text(
                j, i, f"{fs.kendall_tau_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(fs.kendall_tau_matrix[i, j]) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax_tau, shrink=0.7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Proxy stability renderer
# ---------------------------------------------------------------------------

def _render_proxy_stability(
    results: Dict[str, _stab.ProxyStabilityResult],
    sp: Dict[str, Any],
) -> Figure:
    """Render proxy substitution stability: one panel per group."""
    n_groups = len(results)
    fig, axes = plt.subplots(
        1, max(n_groups, 1),
        figsize=(STAB_FIG_WIDTH_IN, STAB_FIG_HEIGHT_IN),
        dpi=DEFAULT_SCREEN_DPI,
        squeeze=False,
    )
    for idx, (gname, pr) in enumerate(results.items()):
        ax = axes[0, idx]
        features = pr.sigma2_f.sort_values(ascending=True).index.tolist()
        variances = [pr.sigma2_f[f] for f in features]
        colors = [
            STAB_BAR_STABLE_COLOR
            if pr.stability_labels.get(f) == _stab.LABEL_FAMILY_STABLE
            else STAB_BAR_CONTINGENT_COLOR
            for f in features
        ]
        ax.barh(range(len(features)), variances, color=colors,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel("σ²_f")
        ax.set_title(
            f"Proxy group: {gname}\n"
            f"members: {', '.join(pr.members)}\n"
            f"τ = {pr.mean_kendall_tau:.3f}",
            fontsize=9,
        )
        ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FeatureRankingDialog
# ---------------------------------------------------------------------------

class FeatureRankingDialog(QDialog):
    """
    Unified feature-ranking analysis dialog.

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

        entry = _eng._DISPATCH.get(key, (key,))
        method_title = entry[0]
        self.setWindowTitle(f"Feature Ranking — {method_title}")
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

        # Tab 1 — Data
        self._data_page = _DataPage(self._df)
        self._tabs.addTab(_scrolled(self._data_page), "Data")

        # Tab 2 — Options
        opt_inner, self._opt_getter = _make_options(key)
        self._tabs.addTab(_scrolled(opt_inner), "Options")

        # Tab 3 — Style
        sty_inner, self._sty_getter = make_style_panel()
        self._tabs.addTab(_scrolled(sty_inner), "Style")

        # Tab 4 — Export
        exp_inner, self._exp_getter = make_export_panel(
            self._save_figure, defaults={"prefix": "feature_ranking"},
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
            "Run this method across all SEAL variants and compute stability "
            "diagnostics (requires SEAL conditioning)"
        )
        self._stab_btn.setEnabled(store.seal_result is not None)
        self._stab_btn.clicked.connect(self._run_stability)
        btn_row.addWidget(self._stab_btn)

        self._proxy_btn = QPushButton("Proxy Stability")
        self._proxy_btn.setToolTip(
            "Test interpretation stability under proxy substitution "
            "(requires proxy_group in observable contracts)"
        )
        self._proxy_btn.clicked.connect(self._run_proxy_stability)
        btn_row.addWidget(self._proxy_btn)

        self._cite_btn = QPushButton("Export Record")
        self._cite_btn.setToolTip("Export citable analysis record as JSON (available after SHAP)")
        self._cite_btn.setEnabled(False)
        self._cite_btn.clicked.connect(self._export_citable)
        btn_row.addWidget(self._cite_btn)

        btn_row.addStretch()
        right.addLayout(btn_row)

        self._placeholder = QLabel("Configure options and click  Generate.")
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
        self._gen_btn.setText("Computing…")
        try:
            self._do_generate()
        except ImportError as ie:
            QMessageBox.warning(self, "Missing Dependency", str(ie))
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as exc:
            log.exception("FeatureRankingDialog._generate")
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
        sub_df      = dp.features(df)
        standardize = dp.standardize()

        entry = _eng._DISPATCH.get(key)
        if entry is None:
            raise ValueError(f"Unknown method key: {key!r}")

        fn = entry[2]

        # ── Build keyword arguments per method ────────────────────────────
        kwargs: dict = {}
        if key in ("decision_tree", "gradient_boosting", "boruta"):
            kwargs["random_state"] = int(op.get("random_state", 42))
        if key in ("random_forest", "rfe", "gradient_boosting",
                   "permutation_importance", "feature_interactions",
                   "genetic_algorithm", "pso", "boruta"):
            kwargs["n_estimators"] = int(op.get("n_estimators", 100))
        if key == "boruta":
            kwargs["n_trials"] = int(op.get("n_trials", 50))
        if key in ("lasso", "elasticnet", "ridge"):
            kwargs["alpha"] = float(op.get("alpha", 0.01))
        if key in ("lasso", "elasticnet"):
            kwargs["max_iter"] = int(op.get("max_iter", 5000))
        if key == "elasticnet":
            kwargs["l1_ratio"] = float(op.get("l1_ratio", 0.5))
        if key == "sparse_pca":
            kwargs["n_components"] = int(op.get("n_components", 5))
            kwargs["alpha"]        = float(op.get("alpha", 1.0))
            kwargs["rs"]           = int(op.get("random_state", 42))
        if key == "rfe":
            kwargs["n_estimators"] = int(op.get("n_estimators", 100))
            kwargs["cv"]           = int(op.get("cv", 5))
            kwargs["rs"]           = int(op.get("random_state", 42))
        if key == "shap":
            kwargs["n_background"] = int(op.get("n_background", 50))
            kwargs["n_coalitions"] = int(op.get("n_coalitions", 500))
            kwargs["rs"]           = int(op.get("random_state", 42))
        if key == "lime":
            kwargs["n_perturbations"] = int(op.get("n_perturbations", 1000))
            kwargs["n_display"]       = int(op.get("n_display", 5))
            kwargs["rs"]              = int(op.get("random_state", 42))
        if key == "permutation_importance":
            kwargs["n_estimators"] = int(op.get("n_estimators", 100))
            kwargs["n_repeats"]    = int(op.get("n_repeats", 10))
            kwargs["rs"]           = int(op.get("random_state", 42))
        if key == "feature_interactions":
            kwargs["n_estimators"] = int(op.get("n_estimators", 100))
            kwargs["rs"]           = int(op.get("random_state", 42))
        if key == "genetic_algorithm":
            kwargs["pop_size"]      = int(op.get("pop_size", 50))
            kwargs["n_generations"] = int(op.get("n_generations", 100))
            kwargs["mutation_rate"] = float(op.get("mutation_rate", 0.01))
            kwargs["n_estimators"]  = int(op.get("n_estimators", 50))
            kwargs["rs"]            = int(op.get("random_state", 42))
        if key == "pso":
            kwargs["n_particles"]  = int(op.get("n_particles", 30))
            kwargs["n_iterations"] = int(op.get("n_iterations", 100))
            kwargs["n_estimators"] = int(op.get("n_estimators", 50))
            kwargs["rs"]           = int(op.get("random_state", 42))
        if key == "relieff":
            kwargs["n_neighbors"] = int(op.get("n_neighbors", 10))
        if key == "mrmr":
            kwargs["n_features_to_select"] = int(op.get("n_features_to_select", 10))
        if key == "recursive_feature_clustering":
            kwargs["n_clusters"] = int(op.get("n_clusters", 5))
        if key == "autoencoder_importance":
            h1 = int(op.get("hidden1", 32)); h2 = int(op.get("hidden2", 16))
            kwargs["hidden_layers"] = [h1] + ([h2] if h2 > 0 else [])
            kwargs["max_iter"]      = int(op.get("max_iter", 300))
            kwargs["rs"]            = int(op.get("random_state", 42))
        if key == "deep_learning_attribution":
            kwargs["hidden_size"] = int(op.get("hidden_size", 64))
            kwargs["n_epochs"]    = int(op.get("n_epochs", 100))
            kwargs["lr"]          = float(op.get("lr", 1e-3))
            kwargs["rs"]          = int(op.get("random_state", 42))

        result: _eng.FeatureRankingResult = fn(sub_df, target_col, **kwargs)
        if not result.is_valid:
            QMessageBox.warning(self, "Degraded Result",
                                f"{result.method_title}: result may be inaccurate "
                                "(ill-conditioned matrix or degenerate data).")

        # ── Dispatch to renderer ─────────────────────────────────────────
        method_title = result.method_title

        if key == "feature_maps":
            corr_matrix = result.extra.get("corr_matrix")
            if corr_matrix is None:
                corr_matrix = sub_df.select_dtypes("number").corr()
            fig = _render_heatmap(corr_matrix,
                                  f"Feature Map — {target_col}", op, sp)

        elif key == "feature_interactions":
            imat = result.extra.get("interaction_matrix")
            if imat is None:
                raise ValueError("Interaction matrix not available in result.extra.")
            fig = _render_heatmap(imat, "Feature Interaction H-Statistic", op, sp)

        elif key == "rfe":
            fig = _render_rfe_curve(result, op, sp)

        elif key == "shap":
            fig = _render_shap_beeswarm(result, op, sp)
            # Wrap SHAP result as a citable record (DNV-2.0 Slide 11)
            _contracts = {}
            if self._store.asset is not None:
                _contracts = self._store.asset.metadata.get(
                    "observable_contracts", {}
                )
            self._last_citable = _stab.make_citable(
                result,
                variant_id=None,
                df=sub_df,
                contract_context=_contracts,
                method_config={
                    "method": "shap",
                    "n_background": op.get("n_background", 50),
                    "n_coalitions": op.get("n_coalitions", 500),
                    "random_state": op.get("random_state", 42),
                    "target_col": target_col,
                },
            )
            self._cite_btn.setEnabled(True)

        else:
            errors: Optional[np.ndarray] = None
            if key == "permutation_importance":
                stds = result.extra.get("importances_std")
                if stds is not None:
                    errors = np.array([stds.get(f, 0.0) for f in result.scores.index])
            fig = _render_bar(
                result.scores,
                f"{method_title} — target: {target_col}",
                "Importance Score",
                op, sp,
                errors=errors,
            )

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
            log.exception("FeatureRankingDialog._run_stability")
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
        fi_fn = entry[2]

        target_col = self._data_page.target()
        if not target_col:
            raise ValueError("Select a target column first.")

        fs = _stab.compute_feature_stability(seal_result, fi_fn, target_col)
        sp = self._sty_getter()
        fig = _render_feature_stability(fs, sp)
        self._set_canvas(fig)

        n_stable = sum(
            1 for v in fs.stability_labels.values()
            if v == _stab.LABEL_FAMILY_STABLE
        )
        n_total = len(fs.stability_labels)
        self._stab_report.setText(
            f"Stability analysis — {fs.n_variants_used} variants  |  "
            f"mean Kendall τ = {fs.mean_kendall_tau:.3f}  |  "
            f"top-{_stab.TOP_K_OVERLAP_DEFAULT} overlap = {fs.top_k_overlap:.1%}  |  "
            f"{n_stable}/{n_total} features family-stable"
        )
        self._stab_report.show()

    # ── Proxy substitution stability ─────────────────────────────────────

    def _run_proxy_stability(self) -> None:
        self._proxy_btn.setEnabled(False)
        self._proxy_btn.setText("Computing…")
        try:
            self._do_proxy_stability()
        except Exception as exc:
            log.exception("FeatureRankingDialog._run_proxy_stability")
            QMessageBox.critical(self, "Proxy Stability Error", str(exc))
        finally:
            self._proxy_btn.setEnabled(True)
            self._proxy_btn.setText("Proxy Stability")

    def _do_proxy_stability(self) -> None:
        contracts = []
        if self._store.asset:
            contracts = self._store.asset.metadata.get("observable_contracts", [])
        groups = _stab.find_proxy_groups(contracts)
        if not groups:
            raise ValueError(
                "No proxy groups found. Declare proxy_group in observable "
                "contracts (e.g. proxy_group='ionic_radius') on at least "
                "2 columns to enable proxy substitution testing."
            )

        entry = _eng._DISPATCH.get(self._key)
        if entry is None:
            raise ValueError(f"Unknown method key: {self._key!r}")
        fi_fn = entry[2]
        target_col = self._data_page.target()
        if not target_col:
            raise ValueError("Select a target column first.")

        results = _stab.compute_proxy_stability(
            self._df, groups, fi_fn, target_col,
        )
        if not results:
            raise ValueError("No proxy stability results (all substitutions failed).")

        sp = self._sty_getter()
        fig = _render_proxy_stability(results, sp)
        self._set_canvas(fig)

        summary_parts = []
        for gname, pr in results.items():
            n_s = sum(1 for v in pr.stability_labels.values() if v == _stab.LABEL_FAMILY_STABLE)
            summary_parts.append(
                f"{gname}: {pr.n_substitutions} subs, "
                f"τ={pr.mean_kendall_tau:.3f}, "
                f"{n_s}/{len(pr.stability_labels)} stable"
            )
        self._stab_report.setText("Proxy stability — " + "  |  ".join(summary_parts))
        self._stab_report.show()

    # ── Export citable record ───────────────────────────────────────────

    def _export_citable(self) -> None:
        """Serialize the last CitableRecord to a JSON file via save dialog."""
        import json

        cr = self._last_citable
        if cr is None:
            QMessageBox.information(
                self, "No Record",
                "Generate a SHAP analysis first to create a citable record.",
            )
            return

        # Build a JSON-safe dict manually (dataclasses.asdict would fail
        # on pandas/numpy objects inside analysis_result).
        ar = cr.analysis_result
        ar_dict = {
            "method_key": getattr(ar, "method_key", ""),
            "method_title": getattr(ar, "method_title", ""),
            "target_col": getattr(ar, "target_col", ""),
            "is_valid": getattr(ar, "is_valid", True),
        }
        scores = getattr(ar, "scores", None)
        if scores is not None:
            ar_dict["scores"] = {
                str(k): float(v) for k, v in scores.items()
            }
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
            "fi_citable_record.json",
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
        prefix = ep.get("prefix", "feature_ranking") or "feature_ranking"

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
                path, dpi=dpi,
                bbox_inches="tight",
                transparent=transp,
                facecolor=self._fig.get_facecolor(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
        finally:
            self._fig.set_size_inches(*orig_size)
