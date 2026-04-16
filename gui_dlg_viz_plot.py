"""
DNV Scientific Module
---------------------
Role:
    Provides the PlotDialog — a unified, canvas-embedded dialog for 18
    visualization types with exhaustive per-method options, a fully
    configurable style panel, and a complete export/save panel.

Scientific Context:
    Accepts a plot_type key and a SessionStore; collects column selections
    and method parameters from the operator; dispatches to a typed renderer;
    renders the result into a FigureCanvasQTAgg with full style control over
    typography, axes, grid, spines, legend (all matplotlib legend kwargs),
    figure background, and physical output dimensions.

Invariants:
    - _apply_style() is the single mutation site for all per-axes style state.
    - _finalise() applies _apply_style() to every axes and migrates
      seaborn figure-level legends to their owning axes.
    - No renderer sets legend, tick, or spine attributes — _finalise owns them.
    - The single FigureCanvas is replaced atomically on each Generate call.

Assumptions:
    - store.asset is a valid DataAsset with a non-empty DataFrame.
    - matplotlib backend is "Agg".
    - seaborn >= 0.13 for density_norm= and inner="quart" API.

Failure Modes:
    - All-NaN column: dropped silently by dropna(); renderer shows empty plot.
    - seaborn not installed: affected plot types fall back to pure matplotlib.
    - sklearn not installed: tsne / pca renderers show a diagnostic label.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

try:
    from sklearn.decomposition import PCA as _PCA
    from sklearn.manifold import TSNE as _TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from scipy.interpolate import CubicSpline as _CubicSpline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from PySide6.QtCore    import Qt, Signal, QTimer
from PySide6.QtGui     import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from gui_session import SessionStore
from gui_style import (
    make_style_panel, make_export_panel,
    apply_style, finalise_figure, rc_ctx,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPENSIVE = {"tsne", "pca", "pairplot", "scatter3d", "candlestick"}

_PLOT_CATALOGUE: List[Tuple[str, str, str]] = [
    ("Histogram",            "histogram",   "UNIVARIATE"),
    ("Box Plot",             "boxplot",     "UNIVARIATE"),
    ("Violin Plot",          "violin",      "UNIVARIATE"),
    ("KDE / Density",        "kde",         "UNIVARIATE"),
    ("Scatter Plot",         "scatter",     "BIVARIATE"),
    ("Joint Plot",           "joint",       "BIVARIATE"),
    ("Pair Plot",            "pairplot",    "BIVARIATE"),
    ("Correlation Heatmap",  "heatmap",     "BIVARIATE"),
    ("Parallel Coordinates", "parallel",    "MULTIVARIATE"),
    ("3-D Scatter",          "scatter3d",   "MULTIVARIATE"),
    ("Line Plot",            "line",        "TIME SERIES"),
    ("Area Plot",            "area",        "TIME SERIES"),
    ("Candlestick / OHLC",   "candlestick", "TIME SERIES"),
    ("Bar / Column Chart",   "bar",         "CATEGORICAL"),
    ("Swarm Plot",           "swarm",       "CATEGORICAL"),
    ("Strip Plot",           "strip",       "CATEGORICAL"),
    ("t-SNE Embedding",      "tsne",        "DIMENSIONALITY"),
    ("PCA Biplot",           "pca",         "DIMENSIONALITY"),
]

_COLORMAPS = [
    "viridis","plasma","inferno","magma","cividis",
    "Blues","Greens","Oranges","Reds","Purples",
    "coolwarm","RdBu","seismic","bwr",
    "tab10","tab20","Set1","Set2","Paired",
]

_PALETTES = ["tab10","Set1","Set2","Set3","husl","hls","muted","pastel","deep","bright"]

_MARKERS  = ["o","s","^","D","x","+","*","v","<",">","p","h","8"]
_LINESTYLES = ["solid","dashed","dotted","dashdot"]

_FONT_FAMILIES = [
    "sans-serif","serif","monospace",
    "Segoe UI","Arial","Helvetica","Times New Roman","DejaVu Sans","Courier New",
]

_LEGEND_LOCS = [
    "best","upper right","upper left","lower right","lower left",
    "center","right","upper center","lower center","center left","center right",
]

# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

class _ColorBtn(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, initial: str = "#1f77b4", parent=None):
        super().__init__(parent)
        self._hex = initial
        self._refresh()
        self.setFixedSize(32, 24)
        self.clicked.connect(self._pick)

    def _refresh(self):
        self.setStyleSheet(
            f"background:{self._hex}; border:1px solid #666; border-radius:3px;"
        )

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._hex), self, "Pick colour")
        if c.isValid():
            self._hex = c.name()
            self._refresh()
            self.colorChanged.emit(self._hex)

    def color(self) -> str:
        return self._hex


def _hrow(label: str, widget: QWidget) -> QWidget:
    w = QWidget(); lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
    lbl = QLabel(label); lbl.setMinimumWidth(170)
    lay.addWidget(lbl); lay.addWidget(widget); lay.addStretch(1)
    return w


def _eyebrow(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("ReportPanelEyebrow")
    return lbl


def _section_sep() -> QFrame:
    sep = QFrame(); sep.setObjectName("Divider")
    sep.setFixedHeight(1); sep.setFrameShape(QFrame.HLine)
    return sep


def _scrolled(inner: QWidget) -> QScrollArea:
    sa = QScrollArea(); sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame); sa.setWidget(inner)
    return sa


def _combo(items, current=None) -> QComboBox:
    cb = QComboBox(); cb.addItems(items)
    if current and current in items:
        cb.setCurrentText(current)
    return cb


def _spin(lo, hi, val, step=1) -> QSpinBox:
    s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    return s


def _dspin(lo, hi, val, step=0.05) -> QDoubleSpinBox:
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    return s


def _check(default: bool) -> QCheckBox:
    c = QCheckBox(); c.setChecked(default); return c


# ---------------------------------------------------------------------------
# Data pages
# ---------------------------------------------------------------------------

class _MultiColPage(QWidget):
    changed = Signal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(4)
        lay.addWidget(_eyebrow("SELECT COLUMNS"))
        self._list = QListWidget(); self._list.setAlternatingRowColors(True)
        for col in df.columns:
            it = QListWidgetItem(str(col))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self._list.addItem(it)
        lay.addWidget(self._list, 1)
        btn = QPushButton("Select all numeric")
        btn.clicked.connect(lambda: self._select_dtype(df, "number"))
        lay.addWidget(btn, 0, Qt.AlignLeft)
        self._list.itemChanged.connect(lambda _: self.changed.emit())

    def _select_dtype(self, df, include):
        cols = set(df.select_dtypes(include=include).columns)
        for i in range(self._list.count()):
            it = self._list.item(i)
            it.setCheckState(Qt.Checked if it.text() in cols else Qt.Unchecked)

    def selected(self) -> List[str]:
        return [self._list.item(i).text()
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]


class _XYPage(QWidget):
    changed = Signal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        cols = [str(c) for c in df.columns]
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(6)
        lay.addWidget(_eyebrow("AXIS MAPPING"))
        self._x = _combo(cols); self._y = _combo(cols)
        if len(cols) > 1: self._y.setCurrentIndex(1)
        self._hue = QComboBox(); self._hue.addItem("— none —"); self._hue.addItems(cols)
        lay.addWidget(_hrow("X column:", self._x))
        lay.addWidget(_hrow("Y column:", self._y))
        lay.addWidget(_hrow("Hue / colour by:", self._hue))
        lay.addStretch(1)
        for w in (self._x, self._y, self._hue):
            w.currentIndexChanged.connect(lambda _: self.changed.emit())

    def x(self) -> str: return self._x.currentText()
    def y(self) -> str: return self._y.currentText()
    def hue(self) -> Optional[str]:
        t = self._hue.currentText(); return None if t.startswith("—") else t


class _XYMultiYPage(QWidget):
    changed = Signal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        cols = [str(c) for c in df.columns]
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(6)
        lay.addWidget(_eyebrow("AXIS MAPPING"))
        self._x = _combo(cols)
        lay.addWidget(_hrow("X column (index):", self._x))
        lay.addWidget(_eyebrow("Y COLUMNS"))
        self._list = QListWidget(); self._list.setAlternatingRowColors(True)
        for col in df.columns:
            it = QListWidgetItem(str(col))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self._list.addItem(it)
        lay.addWidget(self._list, 1)
        self._x.currentIndexChanged.connect(lambda _: self.changed.emit())
        self._list.itemChanged.connect(lambda _: self.changed.emit())

    def x(self) -> str: return self._x.currentText()
    def ys(self) -> List[str]:
        return [self._list.item(i).text()
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]


class _XYCatPage(QWidget):
    changed = Signal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        cols = [str(c) for c in df.columns]
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(6)
        lay.addWidget(_eyebrow("AXIS MAPPING"))
        self._x   = _combo(cols)
        self._y   = QComboBox(); self._y.addItem("— none —"); self._y.addItems(cols)
        self._cat = QComboBox(); self._cat.addItem("— none —"); self._cat.addItems(cols)
        lay.addWidget(_hrow("Value column:", self._x))
        lay.addWidget(_hrow("Category (Y axis):", self._y))
        lay.addWidget(_hrow("Hue:", self._cat))
        lay.addStretch(1)
        for w in (self._x, self._y, self._cat):
            w.currentIndexChanged.connect(lambda _: self.changed.emit())

    def x(self) -> str: return self._x.currentText()
    def y(self) -> Optional[str]:
        t = self._y.currentText(); return None if t.startswith("—") else t
    def hue(self) -> Optional[str]:
        t = self._cat.currentText(); return None if t.startswith("—") else t


class _OHLCPage(QWidget):
    changed = Signal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        cols = [str(c) for c in df.columns]
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(6)
        lay.addWidget(_eyebrow("OHLC COLUMN MAPPING"))
        self._date  = QComboBox(); self._date.addItem("— row index —"); self._date.addItems(cols)
        self._open  = _combo(cols); self._high = _combo(cols)
        self._low   = _combo(cols); self._close = _combo(cols)
        for cb, names in [(self._open,["open","Open","OPEN"]),
                          (self._high,["high","High","HIGH"]),
                          (self._low, ["low","Low","LOW"]),
                          (self._close,["close","Close","CLOSE"])]:
            for n in names:
                if n in cols: cb.setCurrentText(n); break
        lay.addWidget(_hrow("Date / index:", self._date))
        lay.addWidget(_hrow("Open:",  self._open))
        lay.addWidget(_hrow("High:",  self._high))
        lay.addWidget(_hrow("Low:",   self._low))
        lay.addWidget(_hrow("Close:", self._close))
        lay.addStretch(1)
        for w in (self._date, self._open, self._high, self._low, self._close):
            w.currentIndexChanged.connect(lambda _: self.changed.emit())

    def date_col(self) -> Optional[str]:
        t = self._date.currentText(); return None if t.startswith("—") else t
    def open(self)  -> str: return self._open.currentText()
    def high(self)  -> str: return self._high.currentText()
    def low(self)   -> str: return self._low.currentText()
    def close(self) -> str: return self._close.currentText()


class _DimRedPage(QWidget):
    changed = Signal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(6)
        lay.addWidget(_eyebrow("FEATURE COLUMNS"))
        self._list = QListWidget(); self._list.setAlternatingRowColors(True)
        for col in df.select_dtypes(include="number").columns:
            it = QListWidgetItem(str(col))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self._list.addItem(it)
        lay.addWidget(self._list, 1)
        self._hue = QComboBox(); self._hue.addItem("— none —")
        self._hue.addItems([str(c) for c in df.columns])
        lay.addWidget(_hrow("Colour by:", self._hue))
        self._list.itemChanged.connect(lambda _: self.changed.emit())
        self._hue.currentIndexChanged.connect(lambda _: self.changed.emit())

    def selected(self) -> List[str]:
        return [self._list.item(i).text()
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]

    def hue(self) -> Optional[str]:
        t = self._hue.currentText(); return None if t.startswith("—") else t


def _make_data_page(plot_type: str, df: pd.DataFrame) -> QWidget:
    if plot_type in ("histogram","boxplot","violin","kde","pairplot","heatmap","parallel"):
        return _MultiColPage(df)
    if plot_type in ("scatter","joint"):
        return _XYPage(df)
    if plot_type in ("line","area"):
        return _XYMultiYPage(df)
    if plot_type in ("bar","swarm","strip"):
        return _XYCatPage(df)
    if plot_type == "candlestick":
        return _OHLCPage(df)
    if plot_type in ("tsne","pca","scatter3d"):
        return _DimRedPage(df)
    return _MultiColPage(df)


# ---------------------------------------------------------------------------
# Options pages
# ---------------------------------------------------------------------------

def _make_options(plot_type: str, df: pd.DataFrame = None) -> Tuple[QWidget, Callable[[], dict]]:
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(5)
    controls: Dict[str, Any] = {}
    cols = [str(c) for c in df.columns] if df is not None else []

    def _add(label, widget, key):
        controls[key] = widget; lay.addWidget(_hrow(label, widget))

    def _sec(title):
        lay.addWidget(_section_sep()); lay.addWidget(_eyebrow(title))

    # ── histogram ────────────────────────────────────────────────────────────
    if plot_type == "histogram":
        lay.addWidget(_eyebrow("BINNING"))
        _add("Bins:", _spin(2, 500, 30), "bins")
        _add("Fill type:", _combo(["bar","step","stepfilled"], "bar"), "fill_type")
        _add("Stat:", _combo(["count","density","probability","percent"]), "stat")
        _add("Cumulative:", _check(False), "cumulative")
        _sec("APPEARANCE")
        _add("Bar colour:", _ColorBtn("#1f77b4"), "color")
        _add("Bar alpha:", _dspin(0.1, 1.0, 0.75), "bar_alpha")
        _add("Edge colour:", _ColorBtn("#ffffff"), "edge_color")
        _add("Edge width:", _dspin(0.0, 3.0, 0.5), "edge_width")
        _add("KDE overlay:", _check(True), "kde")
        _add("KDE colour:", _ColorBtn("#e74c3c"), "kde_color")
        _add("KDE linewidth:", _dspin(0.5, 4.0, 1.5), "kde_lw")
        _add("Log y-axis:", _check(False), "log_count")

    # ── boxplot ───────────────────────────────────────────────────────────────
    elif plot_type == "boxplot":
        lay.addWidget(_eyebrow("GEOMETRY"))
        _add("Orientation:", _combo(["vertical","horizontal"]), "orient")
        _add("Notched:", _check(False), "notch")
        _add("Whisker range (IQR ×):", _dspin(0.5, 3.0, 1.5, 0.1), "whis")
        _add("Cap width:", _dspin(0.1, 1.0, 0.5, 0.05), "capwidth")
        _sec("APPEARANCE")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Line width:", _dspin(0.5, 3.0, 1.2, 0.1), "linewidth")
        _add("Flier size:", _spin(1, 20, 5), "fliersize")
        _add("Flier colour:", _ColorBtn("#555555"), "flier_color")
        _sec("OVERLAYS")
        _add("Show mean line:", _check(False), "showmeans")
        _add("Mean colour:", _ColorBtn("#e74c3c"), "mean_color")
        _add("Overlay points:", _check(False), "show_points")
        _add("Point alpha:", _dspin(0.1, 1.0, 0.3), "point_alpha")

    # ── violin ────────────────────────────────────────────────────────────────
    elif plot_type == "violin":
        lay.addWidget(_eyebrow("GEOMETRY"))
        _add("Inner:", _combo(["quart","box","point","stick","None"]), "inner")
        _add("Density norm:", _combo(["width","area","count"]), "density_norm")
        _add("BW adjust:", _dspin(0.1, 3.0, 1.0, 0.1), "bw_adjust")
        _add("Cut:", _dspin(0.0, 3.0, 2.0, 0.1), "cut")
        _add("Split (needs hue):", _check(False), "split")
        _sec("APPEARANCE")
        _add("Palette:", _combo(_PALETTES, "muted"), "palette")
        _add("Line width:", _dspin(0.3, 3.0, 1.2, 0.1), "linewidth")
        _add("Saturation:", _dspin(0.1, 1.0, 0.75, 0.05), "saturation")
        _add("Fill alpha:", _dspin(0.1, 1.0, 0.85, 0.05), "fill_alpha")

    # ── kde ───────────────────────────────────────────────────────────────────
    elif plot_type == "kde":
        lay.addWidget(_eyebrow("ESTIMATION"))
        _add("Bandwidth:", _combo(["scott","silverman","0.1","0.3","0.5","1.0","2.0"]), "bw")
        _add("Multiple:", _combo(["layer","fill","stack","dodge"]), "multiple")
        _add("Common norm:", _check(False), "common_norm")
        _add("Cumulative:", _check(False), "cumulative")
        _add("Cut:", _dspin(0.0, 3.0, 3.0, 0.1), "cut")
        _add("Clip low:", QLineEdit(), "clip_lo")
        _add("Clip high:", QLineEdit(), "clip_hi")
        _sec("APPEARANCE")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Fill:", _check(True), "fill")
        _add("Fill alpha:", _dspin(0.05, 1.0, 0.4, 0.05), "alpha")
        _add("Line width:", _dspin(0.5, 4.0, 1.5, 0.1), "linewidth")
        controls["clip_lo"].setPlaceholderText("auto")
        controls["clip_hi"].setPlaceholderText("auto")

    # ── scatter ───────────────────────────────────────────────────────────────
    elif plot_type == "scatter":
        lay.addWidget(_eyebrow("MAPPING"))
        _add("Colour:", _ColorBtn("#1f77b4"), "color")
        _add("Palette (hue):", _combo(_PALETTES), "palette")
        size_cb = QComboBox(); size_cb.addItem("— fixed —"); size_cb.addItems(cols)
        _add("Size column (bubble):", size_cb, "size_col")
        _sec("MARKERS")
        _add("Marker size:", _spin(1, 300, 30), "s")
        _add("Marker style:", _combo(_MARKERS), "marker")
        _add("Alpha:", _dspin(0.01, 1.0, 0.7), "alpha")
        _add("Edge colour:", _ColorBtn("#333333"), "edge_color")
        _add("Edge width:", _dspin(0.0, 2.0, 0.0, 0.1), "edge_width")
        _add("Jitter X:", _dspin(0.0, 0.5, 0.0, 0.02), "jitter_x")
        _add("Jitter Y:", _dspin(0.0, 0.5, 0.0, 0.02), "jitter_y")
        _sec("REGRESSION")
        _add("Regression line:", _check(False), "regline")
        _add("Poly degree:", _spin(1, 5, 1), "poly_degree")
        _add("Confidence band:", _check(False), "ci_band")
        _add("CI level:", _dspin(0.5, 0.99, 0.95, 0.01), "ci_level")
        _add("Reg colour:", _ColorBtn("#e74c3c"), "reg_color")

    # ── joint ─────────────────────────────────────────────────────────────────
    elif plot_type == "joint":
        lay.addWidget(_eyebrow("JOINT"))
        _add("Joint kind:", _combo(["scatter","hex","kde","reg"]), "kind")
        _add("Colour:", _ColorBtn("#1f77b4"), "color")
        _add("Alpha:", _dspin(0.05, 1.0, 0.7, 0.05), "alpha")
        _sec("MARGINAL")
        _add("Marginal kind:", _combo(["hist","kde","box","rug"]), "marginal_kind")
        _add("Marginal colour:", _ColorBtn("#888888"), "marginal_color")
        _add("Fill marginal:", _check(True), "fill_marginal")
        _add("Marginal bins:", _spin(5, 100, 20), "marginal_bins")
        _sec("LAYOUT")
        _add("Space:", _dspin(0.0, 0.5, 0.1, 0.02), "space")
        _add("Joint : marginal ratio:", _spin(2, 8, 5), "ratio")

    # ── pairplot ──────────────────────────────────────────────────────────────
    elif plot_type == "pairplot":
        lay.addWidget(_eyebrow("GEOMETRY"))
        _add("Diagonal kind:", _combo(["auto","hist","kde"]), "diag_kind")
        _add("Off-diag kind:", _combo(["scatter","kde","reg"]), "off_diag_kind")
        _add("Corner only:", _check(False), "corner")
        _add("Markers:", _combo(_MARKERS), "markers")
        _sec("APPEARANCE")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Alpha:", _dspin(0.05, 1.0, 0.7, 0.05), "alpha")
        _add("Diag hist bins:", _spin(5, 100, 20), "diag_bins")
        if cols:
            hue_cb = QComboBox(); hue_cb.addItem("— none —"); hue_cb.addItems(cols)
            _add("Hue column:", hue_cb, "hue_col")

    # ── heatmap ───────────────────────────────────────────────────────────────
    elif plot_type == "heatmap":
        lay.addWidget(_eyebrow("CORRELATION"))
        _add("Method:", _combo(["pearson","spearman","kendall"]), "corr_method")
        _add("Mask upper triangle:", _check(False), "mask_upper")
        _add("Robust colour range:", _check(False), "robust")
        _add("vmin:", _dspin(-1.0, 0.0, -1.0, 0.1), "vmin")
        _add("vmax:", _dspin(0.0, 1.0, 1.0, 0.1), "vmax")
        _sec("APPEARANCE")
        _add("Colormap:", _combo(_COLORMAPS, "coolwarm"), "cmap")
        _add("Annotate:", _check(True), "annot")
        _add("Annotation format:", _combo([".2f",".1f",".3f","d",".4f"]), "fmt")
        _add("Annotation font size:", _spin(5, 16, 9), "annot_fontsize")
        _add("Cell line width:", _dspin(0.0, 2.0, 0.5, 0.1), "linewidths")
        _add("Cell line colour:", _ColorBtn("#ffffff"), "linecolor")
        _sec("COLORBAR")
        _add("Colorbar label:", QLineEdit(), "cbar_label")
        _add("Colorbar shrink:", _dspin(0.3, 1.0, 0.8, 0.05), "cbar_shrink")
        controls["cbar_label"].setPlaceholderText("correlation")

    # ── parallel ──────────────────────────────────────────────────────────────
    elif plot_type == "parallel":
        lay.addWidget(_eyebrow("DATA"))
        _add("Normalize:", _combo(["minmax","zscore","none"]), "normalize_method")
        if cols:
            cc = QComboBox(); cc.addItem("— row index —"); cc.addItems(cols)
            _add("Colour column:", cc, "color_col")
        _sec("APPEARANCE")
        _add("Colormap:", _combo(_COLORMAPS, "viridis"), "cmap")
        _add("Line alpha:", _dspin(0.05, 1.0, 0.5, 0.05), "alpha")
        _add("Line width:", _dspin(0.1, 5.0, 0.8, 0.1), "lw")
        _add("Curved lines:", _check(False), "curved")
        _add("Show axis labels:", _check(True), "show_axes_labels")
        _add("Axis colour:", _ColorBtn("#888888"), "axes_color")
        _add("Axis line width:", _dspin(0.3, 3.0, 0.8, 0.1), "axes_lw")

    # ── scatter3d ─────────────────────────────────────────────────────────────
    elif plot_type == "scatter3d":
        lay.addWidget(_eyebrow("MARKERS"))
        _add("Marker size:", _spin(1, 200, 20), "s")
        _add("Alpha:", _dspin(0.05, 1.0, 0.8, 0.05), "alpha")
        _add("Colormap:", _combo(_COLORMAPS, "viridis"), "cmap")
        _add("Depth shade:", _check(True), "depthshade")
        _add("Show colorbar:", _check(True), "show_colorbar")
        _sec("VIEW ANGLE")
        _add("Elevation °:", _spin(-90, 90, 20), "elev")
        _add("Azimuth °:", _spin(-180, 180, -60), "azim")

    # ── line ──────────────────────────────────────────────────────────────────
    elif plot_type == "line":
        lay.addWidget(_eyebrow("LINE"))
        _add("Line width:", _dspin(0.3, 8.0, 1.5, 0.2), "lw")
        _add("Line style:", _combo(_LINESTYLES), "linestyle")
        _add("Alpha:", _dspin(0.05, 1.0, 1.0, 0.05), "alpha")
        _add("Palette:", _combo(_PALETTES), "palette")
        _sec("MARKERS")
        _add("Show markers:", _check(False), "markers")
        _add("Marker style:", _combo(_MARKERS + ["none"], "none"), "marker_style")
        _add("Marker size:", _spin(1, 20, 4), "marker_size")
        _sec("TRANSFORMS")
        _add("Smoothing window:", _spin(0, 100, 0), "smooth_window")
        _add("Step mode:", _combo(["none","pre","post","mid"]), "step_mode")
        _sec("FILL")
        _add("Fill under:", _check(False), "fill_under")
        _add("Fill alpha:", _dspin(0.0, 0.8, 0.15, 0.05), "fill_alpha")

    # ── area ──────────────────────────────────────────────────────────────────
    elif plot_type == "area":
        lay.addWidget(_eyebrow("GEOMETRY"))
        _add("Fill alpha:", _dspin(0.05, 1.0, 0.5, 0.05), "alpha")
        _add("Stacked:", _check(False), "stacked")
        _add("Baseline:", _combo(["zero","sym","wiggle","weighted_wiggle"]), "baseline")
        _add("Step mode:", _combo(["none","pre","post","mid"]), "step_mode")
        _sec("APPEARANCE")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Show edge:", _check(True), "edge_visible")
        _add("Edge line width:", _dspin(0.3, 2.0, 0.8, 0.1), "edge_lw")

    # ── candlestick ───────────────────────────────────────────────────────────
    elif plot_type == "candlestick":
        lay.addWidget(_eyebrow("CANDLE"))
        _add("Up colour:", _ColorBtn("#26a69a"), "up_color")
        _add("Down colour:", _ColorBtn("#ef5350"), "down_color")
        _add("Bar width:", _dspin(0.1, 1.0, 0.6, 0.05), "width")
        _add("Wick line width:", _dspin(0.3, 2.0, 0.8, 0.1), "wick_lw")
        _add("OHLC bar style:", _check(False), "show_ohlc_bars")
        _sec("MOVING AVERAGES")
        _add("MA-5:", _check(False), "ma_5")
        _add("MA-5 colour:", _ColorBtn("#f39c12"), "ma_color_5")
        _add("MA-20:", _check(False), "ma_20")
        _add("MA-20 colour:", _ColorBtn("#3498db"), "ma_color_20")
        _add("MA-50:", _check(False), "ma_50")
        _add("MA-50 colour:", _ColorBtn("#e74c3c"), "ma_color_50")
        _sec("VOLUME")
        _add("Show volume:", _check(False), "show_volume")
        _add("Volume alpha:", _dspin(0.1, 1.0, 0.4, 0.05), "volume_alpha")

    # ── bar ───────────────────────────────────────────────────────────────────
    elif plot_type == "bar":
        lay.addWidget(_eyebrow("AGGREGATION"))
        _add("Aggregation:", _combo(["mean","median","sum","count"]), "agg")
        _add("CI type:", _combo(["ci","sd","se","pi"]), "ci_type")
        _add("Error bar cap size:", _spin(0, 20, 5), "capsize")
        _add("Error bar colour:", _ColorBtn("#333333"), "errcolor")
        _sec("APPEARANCE")
        _add("Orientation:", _combo(["vertical","horizontal"]), "orient")
        _add("Stacked:", _check(False), "stacked")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Bar alpha:", _dspin(0.1, 1.0, 0.85, 0.05), "bar_alpha")
        _add("Saturation:", _dspin(0.1, 1.0, 0.75, 0.05), "saturation")
        _add("Edge colour:", _ColorBtn("#333333"), "edgecolor")
        _add("Edge width:", _dspin(0.0, 2.0, 0.5, 0.1), "edgewidth")
        _add("Sort bars:", _check(False), "sort_bars")

    # ── swarm ─────────────────────────────────────────────────────────────────
    elif plot_type == "swarm":
        lay.addWidget(_eyebrow("POINTS"))
        _add("Point size:", _dspin(0.5, 10.0, 3.0, 0.5), "s")
        _add("Alpha:", _dspin(0.05, 1.0, 0.7, 0.05), "alpha")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Orientation:", _combo(["vertical","horizontal"]), "orient")
        _add("Dodge:", _check(False), "dodge")
        _add("Line width:", _dspin(0.0, 2.0, 0.0, 0.1), "linewidth")
        _add("Edge colour:", _ColorBtn("#333333"), "edgecolor")
        _add("Max points (warn):", _spin(100, 5000, 500), "warn_threshold")

    # ── strip ─────────────────────────────────────────────────────────────────
    elif plot_type == "strip":
        lay.addWidget(_eyebrow("POINTS"))
        _add("Point size:", _dspin(0.5, 10.0, 3.0, 0.5), "s")
        _add("Alpha:", _dspin(0.05, 1.0, 0.7, 0.05), "alpha")
        _add("Jitter:", _dspin(0.0, 0.5, 0.1, 0.02), "jitter")
        _add("Palette:", _combo(_PALETTES), "palette")
        _add("Orientation:", _combo(["vertical","horizontal"]), "orient")
        _add("Dodge:", _check(False), "dodge")
        _add("Line width:", _dspin(0.0, 2.0, 0.0, 0.1), "linewidth")
        _add("Edge colour:", _ColorBtn("#333333"), "edgecolor")
        _add("Z-order:", _spin(1, 10, 1), "zorder")

    # ── tsne ──────────────────────────────────────────────────────────────────
    elif plot_type == "tsne":
        lay.addWidget(_eyebrow("ALGORITHM"))
        _add("Perplexity:", _spin(2, 100, 30), "perplexity")
        _add("Iterations:", _spin(250, 5000, 1000), "n_iter")
        _add("Learning rate:", _combo(["auto","100","200","500","1000"]), "learning_rate")
        _add("Metric:", _combo(["euclidean","cosine","manhattan","correlation"]), "metric")
        _add("Init:", _combo(["pca","random"]), "init")
        _add("Early exaggeration:", _dspin(1.0, 50.0, 12.0, 1.0), "early_exaggeration")
        _add("Random state:", _spin(0, 9999, 42), "random_state")
        _sec("APPEARANCE")
        _add("Point size:", _spin(1, 200, 15), "s")
        _add("Point alpha:", _dspin(0.1, 1.0, 0.8, 0.05), "point_alpha")
        _add("Colormap:", _combo(_COLORMAPS, "tab10"), "cmap")
        _add("Show point labels:", _check(False), "show_labels")

    # ── pca ───────────────────────────────────────────────────────────────────
    elif plot_type == "pca":
        lay.addWidget(_eyebrow("ALGORITHM"))
        _add("Components:", _spin(2, 10, 2), "n_components")
        _add("PC on X axis:", _spin(1, 10, 1), "pc_x")
        _add("PC on Y axis:", _spin(1, 10, 2), "pc_y")
        _add("Standardise features:", _check(True), "standardize")
        _add("Random state:", _spin(0, 9999, 42), "random_state")
        _sec("BIPLOT")
        _add("Loading vectors:", _check(True), "biplot")
        _add("Loading scale:", _dspin(0.1, 2.0, 1.0, 0.1), "loading_scale")
        _add("Loading colour:", _ColorBtn("#e74c3c"), "loading_color")
        _add("Max loadings (0=all):", _spin(0, 50, 0), "n_top_loadings")
        _sec("APPEARANCE")
        _add("Point size:", _spin(1, 200, 25), "s")
        _add("Point alpha:", _dspin(0.1, 1.0, 0.8, 0.05), "point_alpha")
        _add("Colormap:", _combo(_COLORMAPS, "tab10"), "cmap")
        _add("Scree plot inset:", _check(True), "show_scree")

    lay.addStretch(1)

    def _getter() -> dict:
        out: dict = {}
        for key, widget in controls.items():
            if isinstance(widget, QDoubleSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QComboBox):
                out[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, _ColorBtn):
                out[key] = widget.color()
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text().strip()
            else:
                out[key] = None
        return out

    return w, _getter


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _r_histogram(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns[:1])
    n = max(len(cols), 1); ncols_grid = min(n, 3); nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(4.5 * ncols_grid, 3.5 * nrows_grid), squeeze=False)
    flat = axes.flatten()
    histtype = op.get("fill_type", "bar")
    for i, col in enumerate(cols):
        ax = flat[i]; data = df[col].dropna()
        cumul = op.get("cumulative", False)
        ax.hist(data.values, bins=int(op.get("bins", 30)),
                histtype=histtype,
                density=(op.get("stat", "count") == "density"),
                cumulative=cumul,
                color=op.get("color", "#1f77b4"),
                alpha=op.get("bar_alpha", 0.75),
                edgecolor=op.get("edge_color", "#ffffff"),
                linewidth=op.get("edge_width", 0.5),
                log=op.get("log_count", False))
        if op.get("kde") and len(data) > 1 and not cumul:
            try:
                from scipy.stats import gaussian_kde
                kde_fn = gaussian_kde(data.values)
                xs = np.linspace(data.min(), data.max(), 300)
                ax2 = ax.twinx()
                ax2.plot(xs, kde_fn(xs), color=op.get("kde_color", "#e74c3c"),
                         lw=op.get("kde_lw", 1.5))
                ax2.set_ylabel("density", fontsize=7)
                ax2.tick_params(labelsize=7)
            except Exception: pass
        ax.set_title(str(col), fontsize=10)
        ax.set_xlabel("value"); ax.set_ylabel(op.get("stat", "count"))
    for j in range(len(cols), len(flat)): flat[j].set_visible(False)
    return finalise_figure(fig, sp)


def _r_boxplot(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns)
    data = df[cols].dropna(how="all")
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(cols)), 5))
    vert = op.get("orient", "vertical") == "vertical"
    flier_props = dict(marker="o", markersize=op.get("fliersize", 5),
                       markerfacecolor=op.get("flier_color", "#555555"),
                       markeredgewidth=0.5)
    mean_props  = dict(color=op.get("mean_color", "#e74c3c"), linewidth=1.5)
    if _HAS_SNS:
        if sp.get("sns_style"): sns.set_style(sp["sns_style"])
        _pal = sns.color_palette(op.get("palette", "tab10"), n_colors=len(cols))
        sns.boxplot(data=data[cols], ax=ax, notch=op.get("notch", False),
                    palette=op.get("palette", "tab10"), orient="v" if vert else "h",
                    linewidth=op.get("linewidth", 1.2),
                    flierprops=flier_props,
                    showmeans=op.get("showmeans", False),
                    meanprops=mean_props,
                    whis=op.get("whis", 1.5),
                    width=op.get("capwidth", 0.5))
        if op.get("show_points"):
            sns.stripplot(data=data[cols], ax=ax, color="black",
                          alpha=op.get("point_alpha", 0.3), size=2, orient="v" if vert else "h")
        # Inject proxy handles so _apply_style can build a legend from column names.
        for col, clr in zip(cols, _pal):
            ax.plot([], [], lw=6, color=clr, label=str(col))
    else:
        _cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"] * len(cols))
        ax.boxplot([data[c].dropna().values for c in cols], labels=cols,
                   vert=vert, notch=op.get("notch", False),
                   patch_artist=True, flierprops=flier_props,
                   showmeans=op.get("showmeans", False), meanprops=mean_props,
                   whis=op.get("whis", 1.5))
        for i, col in enumerate(cols):
            ax.plot([], [], lw=6, color=_cycle[i % len(_cycle)], label=str(col))
    return finalise_figure(fig, sp)


def _r_violin(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns)
    data = df[cols].dropna(how="all")
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(cols)), 5))
    inner_v = op.get("inner", "quart")
    if inner_v == "None": inner_v = None
    if _HAS_SNS:
        if sp.get("sns_style"): sns.set_style(sp["sns_style"])
        _pal = sns.color_palette(op.get("palette", "muted"), n_colors=len(cols))
        sns.violinplot(data=data[cols], ax=ax, inner=inner_v,
                       density_norm=op.get("density_norm", "width"),
                       bw_adjust=op.get("bw_adjust", 1.0),
                       cut=op.get("cut", 2.0),
                       linewidth=op.get("linewidth", 1.2),
                       saturation=op.get("saturation", 0.75),
                       palette=op.get("palette", "muted"))
        # Inject proxy handles so _apply_style can build a legend from column names.
        for col, clr in zip(cols, _pal):
            ax.fill_between([], [], color=clr, alpha=0.8, label=str(col))
    else:
        _cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"] * len(cols))
        ax.violinplot([data[c].dropna().values for c in cols], showmedians=True)
        ax.set_xticks(range(1, len(cols)+1)); ax.set_xticklabels(cols, rotation=45)
        for i, col in enumerate(cols):
            ax.plot([], [], lw=6, color=_cycle[i % len(_cycle)], label=str(col))
    return finalise_figure(fig, sp)


def _r_kde(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns)
    fig, ax = plt.subplots(figsize=(7, 4))
    if _HAS_SNS:
        if sp.get("sns_style"): sns.set_style(sp["sns_style"])
        bw_raw = op.get("bw", "scott")
        try: bw_val: Any = float(bw_raw)
        except: bw_val = bw_raw
        cl_lo = op.get("clip_lo", ""); cl_hi = op.get("clip_hi", "")
        try:    clip_lo = float(cl_lo) if cl_lo else None
        except: clip_lo = None
        try:    clip_hi = float(cl_hi) if cl_hi else None
        except: clip_hi = None
        clip = None
        if clip_lo is not None and clip_hi is not None: clip = (clip_lo, clip_hi)
        palette = sns.color_palette(op.get("palette", "tab10"), n_colors=len(cols))
        for col, clr in zip(cols, palette):
            data = df[col].dropna()
            if len(data) < 2: continue
            kw: dict = dict(x=data.values, ax=ax,
                            fill=op.get("fill", True),
                            alpha=op.get("alpha", 0.4),
                            bw_method=bw_val,
                            multiple=op.get("multiple", "layer"),
                            common_norm=op.get("common_norm", False),
                            cumulative=op.get("cumulative", False),
                            cut=op.get("cut", 3.0),
                            linewidth=op.get("linewidth", 1.5),
                            label=str(col), color=clr)
            if clip: kw["clip"] = clip
            sns.kdeplot(**kw)
    else:
        for col in cols:
            df[col].dropna().plot.kde(ax=ax, label=str(col))
    return finalise_figure(fig, sp)


def _r_scatter(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x = dp.x(); y = dp.y(); hue = dp.hue()
    xv = df[x].values; yv = df[y].values
    if op.get("jitter_x", 0) > 0:
        xv = xv + np.random.uniform(-op["jitter_x"], op["jitter_x"], len(xv))
    if op.get("jitter_y", 0) > 0:
        yv = yv + np.random.uniform(-op["jitter_y"], op["jitter_y"], len(yv))
    fig, ax = plt.subplots(figsize=(6, 5))
    size_col = op.get("size_col", "— fixed —")
    sizes = df[size_col].values if size_col and size_col != "— fixed —" and size_col in df.columns else None
    if _HAS_SNS:
        if sp.get("sns_style"): sns.set_style(sp["sns_style"])
        kw: dict = dict(ax=ax, alpha=op.get("alpha", 0.7),
                        marker=op.get("marker", "o"),
                        edgecolor=op.get("edge_color", "#333333") if op.get("edge_width", 0) > 0 else "none",
                        linewidth=op.get("edge_width", 0))
        tmp = df.copy(); tmp["__x"] = xv; tmp["__y"] = yv
        kw["x"] = "__x"; kw["y"] = "__y"; kw["data"] = tmp
        if hue: kw["hue"] = hue; kw["palette"] = op.get("palette", "tab10")
        else:   kw["color"] = op.get("color", "#1f77b4")
        if sizes is not None: kw["size"] = size_col; kw["sizes"] = (20, 400)
        else: kw["s"] = op.get("s", 30)
        sns.scatterplot(**kw)
    else:
        c = op.get("color", "#1f77b4")
        ax.scatter(xv, yv, alpha=op.get("alpha", 0.7),
                   s=op.get("s", 30), c=c,
                   edgecolors=op.get("edge_color", "none"),
                   linewidths=op.get("edge_width", 0))
    if op.get("regline"):
        try:
            mask = ~(np.isnan(xv) | np.isnan(yv))
            xm, ym = xv[mask].astype(float), yv[mask].astype(float)
            deg = int(op.get("poly_degree", 1))
            coeffs = np.polyfit(xm, ym, deg)
            xs = np.linspace(xm.min(), xm.max(), 300)
            ys = np.polyval(coeffs, xs)
            ax.plot(xs, ys, color=op.get("reg_color", "#e74c3c"), lw=1.8, label=f"poly-{deg} fit")
            if op.get("ci_band") and deg == 1:
                ci = float(op.get("ci_level", 0.95))
                n = len(xm); se = np.sqrt(np.sum((ym - np.polyval(coeffs, xm))**2) / (n-2))
                from scipy import stats as _stats
                t_val = _stats.t.ppf((1 + ci) / 2, df=n - 2)
                xbar = xm.mean(); sxx = np.sum((xm - xbar)**2)
                ci_band = t_val * se * np.sqrt(1/n + (xs - xbar)**2 / sxx)
                ax.fill_between(xs, ys - ci_band, ys + ci_band,
                                alpha=0.2, color=op.get("reg_color", "#e74c3c"))
        except Exception: pass
    if not sp.get("xlabel"): ax.set_xlabel(str(x))
    if not sp.get("ylabel"): ax.set_ylabel(str(y))
    return finalise_figure(fig, sp)


def _r_joint(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x = dp.x(); y = dp.y()
    kind = op.get("kind", "scatter"); alpha = op.get("alpha", 0.7)
    color = op.get("color", "#1f77b4")
    marg  = op.get("marginal_kind", "hist")
    marg_c = op.get("marginal_color", "#888888")
    ratio = int(op.get("ratio", 5)); space = op.get("space", 0.1)
    data = df[[x, y]].dropna()
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(ratio + 1, ratio + 1, hspace=space, wspace=space)
    ax_j = fig.add_subplot(gs[1:, :-1])
    ax_t = fig.add_subplot(gs[0, :-1], sharex=ax_j)
    ax_r = fig.add_subplot(gs[1:, -1], sharey=ax_j)
    xv = data[x].values; yv = data[y].values
    if kind == "scatter":
        ax_j.scatter(xv, yv, alpha=alpha, s=20, color=color)
    elif kind == "hex":
        ax_j.hexbin(xv, yv, gridsize=25, cmap="viridis", alpha=alpha)
    elif kind in ("kde", "reg") and _HAS_SNS:
        sns.kdeplot(x=xv, y=yv, ax=ax_j, fill=True, alpha=0.5, color=color)
        if kind == "reg":
            try:
                m, b = np.polyfit(xv, yv, 1)
                xs = np.linspace(xv.min(), xv.max(), 200)
                ax_j.plot(xs, m*xs+b, color="#e74c3c", lw=1.5)
            except Exception: pass
    else:
        ax_j.scatter(xv, yv, alpha=alpha, color=color)
    fill_m = op.get("fill_marginal", True)
    bins_m = int(op.get("marginal_bins", 20))
    if marg == "hist":
        ax_t.hist(xv, bins=bins_m, color=marg_c, alpha=0.6 if fill_m else 1.0)
        ax_r.hist(yv, bins=bins_m, color=marg_c, alpha=0.6 if fill_m else 1.0,
                  orientation="horizontal")
    elif marg == "kde" and _HAS_SNS:
        sns.kdeplot(x=xv, ax=ax_t, fill=fill_m, color=marg_c, linewidth=1.2)
        sns.kdeplot(y=yv, ax=ax_r, fill=fill_m, color=marg_c, linewidth=1.2)
    elif marg == "rug":
        ax_t.plot(xv, np.zeros_like(xv) + 0.5, "|", color=marg_c, ms=10)
        ax_r.plot(np.zeros_like(yv) + 0.5, yv, "_", color=marg_c, ms=10)
    else:
        ax_t.hist(xv, bins=bins_m, color=marg_c, alpha=0.5)
        ax_r.hist(yv, bins=bins_m, color=marg_c, alpha=0.5, orientation="horizontal")
    plt.setp(ax_t.get_xticklabels(), visible=False)
    plt.setp(ax_r.get_yticklabels(), visible=False)
    ax_j.set_xlabel(str(x)); ax_j.set_ylabel(str(y))
    if sp.get("title"): fig.suptitle(sp["title"], y=1.01)
    return finalise_figure(fig, sp)


def _r_pairplot(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns[:5])
    data = df[cols].dropna()
    hue_col = op.get("hue_col", "")
    if hue_col and hue_col.startswith("—"): hue_col = ""
    if hue_col and hue_col in df.columns:
        data = df[cols + [hue_col]].dropna()
    if not _HAS_SNS:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "seaborn required", ha="center"); return fig
    if sp.get("sns_style"): sns.set_style(sp["sns_style"])
    diag = op.get("diag_kind", "auto")
    if diag == "auto": diag = None
    off  = op.get("off_diag_kind", "scatter")
    dkws = {"bins": int(op.get("diag_bins", 20))} if diag == "hist" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.pairplot(data,
                         hue=hue_col if hue_col else None,
                         diag_kind=diag,
                         kind=off,
                         corner=op.get("corner", False),
                         markers=op.get("markers", "o"),
                         palette=op.get("palette", "tab10"),
                         plot_kws={"alpha": op.get("alpha", 0.7), "s": 15},
                         diag_kws=dkws)
    if sp.get("title"): g.fig.suptitle(sp["title"], y=1.02)
    fig = g.fig
    plt.close("all")
    return fig


def _r_heatmap(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns)
    data = df[cols].select_dtypes(include="number").dropna(how="all")
    method = op.get("corr_method", "pearson")
    corr = data.corr(method=method)
    n = len(corr)
    mask = np.triu(np.ones_like(corr, dtype=bool)) if op.get("mask_upper") else None
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(4, n * 0.65)))
    annot_kws = {"size": int(op.get("annot_fontsize", 9))}
    cbar_kws  = {"shrink": op.get("cbar_shrink", 0.8)}
    cbar_lbl  = op.get("cbar_label", "")
    if cbar_lbl: cbar_kws["label"] = cbar_lbl
    if _HAS_SNS:
        sns.set_style("white")
        sns.heatmap(corr, ax=ax, cmap=op.get("cmap", "coolwarm"),
                    annot=op.get("annot", True), fmt=op.get("fmt", ".2f"),
                    vmin=op.get("vmin", -1.0), vmax=op.get("vmax", 1.0),
                    mask=mask, linewidths=op.get("linewidths", 0.5),
                    linecolor=op.get("linecolor", "#ffffff"),
                    annot_kws=annot_kws, cbar_kws=cbar_kws,
                    robust=op.get("robust", False), square=True)
    else:
        im = ax.imshow(corr.values, cmap=op.get("cmap", "coolwarm"),
                       vmin=op.get("vmin", -1.0), vmax=op.get("vmax", 1.0))
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, shrink=op.get("cbar_shrink", 0.8))
    return finalise_figure(fig, sp)


def _r_parallel(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    cols = dp.selected() if hasattr(dp, "selected") else []
    if not cols: cols = list(df.select_dtypes(include="number").columns[:8])
    data = df[cols].dropna(how="all").select_dtypes(include="number")
    nm = op.get("normalize_method", "minmax")
    if nm == "minmax":
        norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
    elif nm == "zscore":
        norm = (data - data.mean()) / (data.std() + 1e-12)
    else:
        norm = data.copy()
    color_col = op.get("color_col", "")
    if color_col and color_col.startswith("—"): color_col = ""
    if color_col and color_col in df.columns:
        raw = df.loc[norm.index, color_col]
        try:   c_vals = raw.astype(float).values
        except: c_vals = raw.astype("category").cat.codes.values
        c_vals = (c_vals - c_vals.min()) / (np.ptp(c_vals) + 1e-12)
    else:
        c_vals = None
    cmap_fn = plt.get_cmap(op.get("cmap", "viridis"))
    fig, ax  = plt.subplots(figsize=(max(6, len(cols) * 1.2), 5))
    curved = op.get("curved", False) and _HAS_SCIPY
    for idx, (_, row) in enumerate(norm.iterrows()):
        vals = row.values; xs = np.arange(len(cols))
        c = cmap_fn(c_vals[idx]) if c_vals is not None else cmap_fn(row.mean())
        if curved and len(cols) >= 3:
            xs_fine = np.linspace(0, len(cols)-1, (len(cols)-1)*10)
            try:
                cs = _CubicSpline(xs, vals)
                ax.plot(xs_fine, cs(xs_fine), color=c,
                        alpha=op.get("alpha", 0.5), lw=op.get("lw", 0.8))
            except Exception:
                ax.plot(xs, vals, color=c, alpha=op.get("alpha", 0.5), lw=op.get("lw", 0.8))
        else:
            ax.plot(xs, vals, color=c, alpha=op.get("alpha", 0.5), lw=op.get("lw", 0.8))
    if op.get("show_axes_labels", True):
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right",
                            color=op.get("axes_color", "#888888"))
    for xi in range(len(cols)):
        ax.axvline(xi, color=op.get("axes_color", "#888888"),
                   lw=op.get("axes_lw", 0.8), zorder=0)
    ax.set_ylabel(f"normalised ({nm})")
    return finalise_figure(fig, sp)


def _r_scatter3d(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    cols = dp.selected() if hasattr(dp, "selected") else []
    hue  = dp.hue() if hasattr(dp, "hue") else None
    if len(cols) < 3: cols = list(df.select_dtypes(include="number").columns[:3])
    data = df[cols[:3]].dropna()
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=op.get("elev", 20), azim=op.get("azim", -60))
    xv, yv, zv = data.iloc[:,0].values, data.iloc[:,1].values, data.iloc[:,2].values
    c_vals = None
    if hue and hue in df.columns:
        raw = df.loc[data.index, hue]
        try:   c_vals = raw.astype(float).values
        except: c_vals = raw.astype("category").cat.codes.values
    sc = ax.scatter(xv, yv, zv,
                    c=c_vals, cmap=op.get("cmap", "viridis"),
                    s=op.get("s", 20), alpha=op.get("alpha", 0.8),
                    depthshade=op.get("depthshade", True))
    if c_vals is not None and op.get("show_colorbar", True):
        cbl = sp.get("colorbar_label", "") or str(hue)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=cbl)
    ax.set_xlabel(str(cols[0])); ax.set_ylabel(str(cols[1])); ax.set_zlabel(str(cols[2]))
    if sp.get("title"): ax.set_title(sp["title"])
    fig.set_facecolor(sp.get("fig_facecolor", "#ffffff"))
    if sp.get("tight_layout", True):
        try: fig.tight_layout()
        except Exception: pass
    return fig


def _r_line(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x_col = dp.x() if hasattr(dp, "x") else None
    y_cols = dp.ys() if hasattr(dp, "ys") else []
    if not y_cols: y_cols = list(df.select_dtypes(include="number").columns[:3])
    fig, ax = plt.subplots(figsize=(9, 4))
    pal = plt.get_cmap(op.get("palette", "tab10") if op.get("palette","tab10") in plt.colormaps else "tab10")
    ls  = op.get("linestyle", "solid")
    step = op.get("step_mode", "none")
    for i, col in enumerate(y_cols):
        xv = df[x_col].values if x_col else np.arange(len(df))
        yv = df[col].values
        sm = int(op.get("smooth_window", 0))
        if sm > 1:
            yv = pd.Series(yv).rolling(sm, center=True, min_periods=1).mean().values
        mask = ~(np.isnan(xv.astype(float)) | np.isnan(yv)) if x_col else ~np.isnan(yv)
        xs = xv[mask] if x_col else np.where(mask)[0]; ys = yv[mask]
        clr = pal(i % 10)
        mkr = op.get("marker_style", "none")
        if mkr == "none": mkr = None
        draw_kw = dict(lw=op.get("lw", 1.5), alpha=op.get("alpha", 1.0),
                       linestyle=ls, marker=mkr,
                       markersize=op.get("marker_size", 4) if mkr else 0,
                       label=str(col), color=clr)
        if step != "none":
            ax.step(xs, ys, where=step, **draw_kw)
        else:
            ax.plot(xs, ys, **draw_kw)
        if op.get("fill_under"):
            ax.fill_between(xs, ys, alpha=op.get("fill_alpha", 0.15), color=clr)
    if x_col: ax.set_xlabel(str(x_col))
    return finalise_figure(fig, sp)


def _r_area(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x_col = dp.x() if hasattr(dp, "x") else None
    y_cols = dp.ys() if hasattr(dp, "ys") else []
    if not y_cols: y_cols = list(df.select_dtypes(include="number").columns[:3])
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = df[x_col].values if x_col else np.arange(len(df))
    pal_name = op.get("palette", "tab10")
    try: pal_colors = plt.get_cmap(pal_name).colors[:len(y_cols)]
    except Exception: pal_colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(y_cols))]
    edge_kw = dict(lw=op.get("edge_lw", 0.8)) if op.get("edge_visible", True) else dict(lw=0)
    step = op.get("step_mode", "none")
    if op.get("stacked"):
        baseline = op.get("baseline", "zero")
        ys_stack = [df[c].fillna(0).values for c in y_cols]
        ax.stackplot(xs, *ys_stack,
                     labels=y_cols,
                     alpha=op.get("alpha", 0.5),
                     baseline=baseline,
                     colors=pal_colors)
    else:
        for i, col in enumerate(y_cols):
            yv = df[col].fillna(0).values
            clr = pal_colors[i] if i < len(pal_colors) else plt.get_cmap("tab10")(i % 10)
            if step != "none":
                ax.step(xs, yv, where=step, color=clr, label=str(col), **edge_kw)
                ax.fill_between(xs, yv, step=step, alpha=op.get("alpha", 0.5), color=clr)
            else:
                ax.fill_between(xs, yv, alpha=op.get("alpha", 0.5),
                                label=str(col), color=clr, **edge_kw)
    if x_col: ax.set_xlabel(str(x_col))
    return finalise_figure(fig, sp)


def _r_candlestick(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    date_col = dp.date_col() if hasattr(dp, "date_col") else None
    o = dp.open(); h = dp.high(); l = dp.low(); c = dp.close()
    data = df[[o, h, l, c]].dropna()
    n = min(len(data), 300); data = data.iloc[-n:]
    xs = np.arange(len(data))
    up_c   = op.get("up_color",   "#26a69a")
    dn_c   = op.get("down_color", "#ef5350")
    width  = op.get("width", 0.6)
    wick_lw= op.get("wick_lw", 0.8)
    show_vol = op.get("show_volume", False)
    show_ohlc = op.get("show_ohlc_bars", False)

    if show_vol:
        fig, (ax, ax_vol) = plt.subplots(2, 1, figsize=(max(8, n*0.05+3), 7),
                                          gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(max(8, n*0.05+3), 5))
        ax_vol = None

    close_vals = data[c].values
    open_vals  = data[o].values
    high_vals  = data[h].values
    low_vals   = data[l].values

    for i in range(len(data)):
        ov, hv, lv, cv = float(open_vals[i]), float(high_vals[i]), float(low_vals[i]), float(close_vals[i])
        color = up_c if cv >= ov else dn_c
        if show_ohlc:
            ax.plot([i-0.25, i], [ov, ov], color=color, lw=wick_lw+0.5)
            ax.plot([i, i+0.25], [cv, cv], color=color, lw=wick_lw+0.5)
            ax.plot([i, i], [lv, hv], color=color, lw=wick_lw)
        else:
            ax.bar(i, abs(cv-ov), bottom=min(ov, cv), width=width,
                   color=color, edgecolor=color, linewidth=0.3)
            ax.plot([i, i], [lv, hv], color=color, lw=wick_lw)

    # Moving averages
    for period, flag, clr_key in [(5,"ma_5","ma_color_5"),(20,"ma_20","ma_color_20"),(50,"ma_50","ma_color_50")]:
        if op.get(flag):
            ma = pd.Series(close_vals).rolling(period).mean().values
            valid = ~np.isnan(ma)
            ax.plot(xs[valid], ma[valid], color=op.get(clr_key,"#888"),
                    lw=1.2, label=f"MA-{period}")
    # MA labels are picked up by _finalise → _apply_style legend block.

    if ax_vol is not None and "Volume" in df.columns or (date_col and show_vol):
        pass  # volume bars would need a volume column

    tick_every = max(1, n // 10)
    ticks = xs[::tick_every]
    if date_col and date_col in df.columns:
        labels = [str(data.iloc[i][date_col])[:10] for i in ticks]
    else:
        labels = [str(data.index[i]) for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Price")
    return finalise_figure(fig, sp)


def _r_bar(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x_col = dp.x(); y_col = dp.y(); hue = dp.hue()
    fig, ax = plt.subplots(figsize=(8, 5))
    vert = op.get("orient", "vertical") == "vertical"
    agg_map = {"mean": np.mean, "median": np.median, "sum": np.sum, "count": len}
    estimator = agg_map[op.get("agg", "mean")]
    ci_type = op.get("ci_type", "ci")
    errbar: Any = ("ci", 95) if ci_type == "ci" else ("sd", 1) if ci_type == "sd" else ("se", 1) if ci_type == "se" else ("pi", 95)
    if _HAS_SNS and y_col:
        if sp.get("sns_style"): sns.set_style(sp["sns_style"])
        kw: dict = dict(data=df, ax=ax, estimator=estimator, errorbar=errbar,
                        palette=op.get("palette", "tab10"),
                        saturation=op.get("saturation", 0.75),
                        capsize=op.get("capsize", 5) / 100.0,
                        errcolor=op.get("errcolor", "#333333"))
        if vert: kw["x"] = y_col; kw["y"] = x_col
        else:    kw["x"] = x_col; kw["y"] = y_col
        if hue: kw["hue"] = hue
        sns.barplot(**kw)
        for patch in ax.patches:
            patch.set_alpha(op.get("bar_alpha", 0.85))
            if op.get("edgewidth", 0.5) > 0:
                patch.set_edgecolor(op.get("edgecolor", "#333333"))
                patch.set_linewidth(op.get("edgewidth", 0.5))
    elif x_col and y_col:
        agg_fn = op.get("agg", "mean")
        grouped = df.groupby(y_col)[x_col].agg(agg_fn)
        if op.get("sort_bars"): grouped = grouped.sort_values(ascending=False)
        ax.bar(range(len(grouped)), grouped.values,
               color="#1f77b4", alpha=op.get("bar_alpha", 0.85),
               edgecolor=op.get("edgecolor", "#333"), linewidth=op.get("edgewidth", 0.5))
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right")
    return finalise_figure(fig, sp)


def _r_swarm(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x_col = dp.x(); y_col = dp.y(); hue = dp.hue()
    fig, ax = plt.subplots(figsize=(8, 5))
    thresh = int(op.get("warn_threshold", 500))
    if not _HAS_SNS:
        ax.text(0.5, 0.5, "seaborn required", ha="center", transform=ax.transAxes); return fig
    n_pts = len(df) if y_col is None else len(df[y_col].dropna())
    if n_pts > thresh:
        ax.text(0.5, 0.5, f"Dataset has {n_pts} points — swarm may be slow.\nReduce data or use strip plot.",
                ha="center", va="center", transform=ax.transAxes,
                wrap=True, color="#c0392b"); return fig
    if sp.get("sns_style"): sns.set_style(sp["sns_style"])
    orient_v = op.get("orient", "vertical") == "vertical"
    kw: dict = dict(data=df, ax=ax, size=op.get("s", 3.0),
                    alpha=op.get("alpha", 0.7),
                    palette=op.get("palette", "tab10"),
                    dodge=op.get("dodge", False),
                    linewidth=op.get("linewidth", 0.0),
                    edgecolor=op.get("edgecolor", "#333333"),
                    orient="v" if orient_v else "h")
    if y_col: kw["x"] = y_col; kw["y"] = x_col
    else:     kw["x"] = x_col
    if hue: kw["hue"] = hue
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); sns.swarmplot(**kw)
    return finalise_figure(fig, sp)


def _r_strip(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    x_col = dp.x(); y_col = dp.y(); hue = dp.hue()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not _HAS_SNS:
        ax.text(0.5, 0.5, "seaborn required", ha="center", transform=ax.transAxes); return fig
    if sp.get("sns_style"): sns.set_style(sp["sns_style"])
    orient_v = op.get("orient", "vertical") == "vertical"
    kw: dict = dict(data=df, ax=ax, size=op.get("s", 3.0),
                    alpha=op.get("alpha", 0.7),
                    jitter=op.get("jitter", 0.1),
                    palette=op.get("palette", "tab10"),
                    dodge=op.get("dodge", False),
                    linewidth=op.get("linewidth", 0.0),
                    edgecolor=op.get("edgecolor", "#333333"),
                    zorder=int(op.get("zorder", 1)),
                    orient="v" if orient_v else "h")
    if y_col: kw["x"] = y_col; kw["y"] = x_col
    else:     kw["x"] = x_col
    if hue: kw["hue"] = hue
    sns.stripplot(**kw)
    return finalise_figure(fig, sp)


def _r_tsne(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    if not _HAS_SKLEARN:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "scikit-learn required", ha="center"); return fig
    cols = dp.selected() if hasattr(dp, "selected") else []
    hue  = dp.hue() if hasattr(dp, "hue") else None
    if not cols: cols = list(df.select_dtypes(include="number").columns)
    data = df[cols].dropna()
    if len(data) < 10:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Need ≥10 rows", ha="center"); return fig
    X = data.values; X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    lr: Any = op.get("learning_rate", "auto")
    try: lr = float(lr)
    except: lr = "auto"
    tsne = _TSNE(n_components=2,
                 perplexity=min(int(op.get("perplexity", 30)), len(data)-1),
                 n_iter=int(op.get("n_iter", 1000)),
                 learning_rate=lr,
                 metric=op.get("metric", "euclidean"),
                 init=op.get("init", "pca"),
                 early_exaggeration=float(op.get("early_exaggeration", 12.0)),
                 random_state=int(op.get("random_state", 42)))
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); emb = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    c_vals = None
    if hue and hue in df.columns:
        raw = df.loc[data.index, hue]
        try:   c_vals = raw.astype(float).values
        except: c_vals = raw.astype("category").cat.codes.values
    sc = ax.scatter(emb[:,0], emb[:,1], c=c_vals, cmap=op.get("cmap","tab10"),
                    s=int(op.get("s", 15)), alpha=float(op.get("point_alpha", 0.8)))
    if c_vals is not None:
        cbl = sp.get("colorbar_label", "") or str(hue)
        fig.colorbar(sc, ax=ax, label=cbl)
    if op.get("show_labels") and hue and hue in df.columns:
        labels = df.loc[data.index, hue].astype(str).values
        for xi, yi, lbl in zip(emb[:,0], emb[:,1], labels):
            ax.annotate(lbl, (xi, yi), fontsize=6, alpha=0.6)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    return finalise_figure(fig, sp)


def _r_pca(df: pd.DataFrame, dp, op: dict, sp: dict) -> Figure:
    if not _HAS_SKLEARN:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "scikit-learn required", ha="center"); return fig
    cols = dp.selected() if hasattr(dp, "selected") else []
    hue  = dp.hue() if hasattr(dp, "hue") else None
    if not cols: cols = list(df.select_dtypes(include="number").columns)
    data = df[cols].dropna()
    X = data.values
    if op.get("standardize", True): X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    n_comp = min(int(op.get("n_components", 2)), X.shape[1], X.shape[0])
    pca = _PCA(n_components=n_comp, random_state=int(op.get("random_state", 42)))
    scores = pca.fit_transform(X); evr = pca.explained_variance_ratio_
    pc_x = min(int(op.get("pc_x", 1)), n_comp) - 1
    pc_y = min(int(op.get("pc_y", 2)), n_comp) - 1
    if pc_x == pc_y: pc_y = (pc_x + 1) % n_comp
    fig, ax = plt.subplots(figsize=(7, 6))
    c_vals = None
    if hue and hue in df.columns:
        raw = df.loc[data.index, hue]
        try:   c_vals = raw.astype(float).values
        except: c_vals = raw.astype("category").cat.codes.values
    sc = ax.scatter(scores[:, pc_x], scores[:, pc_y],
                    c=c_vals, cmap=op.get("cmap", "tab10"),
                    s=int(op.get("s", 25)), alpha=float(op.get("point_alpha", 0.8)))
    if c_vals is not None:
        cbl = sp.get("colorbar_label", "") or str(hue)
        fig.colorbar(sc, ax=ax, label=cbl)
    ax.set_xlabel(f"PC{pc_x+1} ({evr[pc_x]*100:.1f}%)")
    ax.set_ylabel(f"PC{pc_y+1} ({evr[pc_y]*100:.1f}%)")
    if op.get("biplot"):
        loadings = pca.components_.T
        scale = np.abs(scores[:, [pc_x, pc_y]]).max() * 0.4 * float(op.get("loading_scale", 1.0))
        top_n = int(op.get("n_top_loadings", 0))
        feats = cols
        if top_n > 0:
            magnitudes = np.sqrt(loadings[:, pc_x]**2 + loadings[:, pc_y]**2)
            idx_order = np.argsort(magnitudes)[::-1][:top_n]
        else:
            idx_order = range(len(cols))
        lclr = op.get("loading_color", "#e74c3c")
        for i in idx_order:
            lx = loadings[i, pc_x] * scale; ly = loadings[i, pc_y] * scale
            ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=lclr, lw=1.2))
            ax.text(lx * 1.08, ly * 1.08, str(feats[i]), fontsize=7, color=lclr)
    if op.get("show_scree", True) and n_comp > 1:
        ins = fig.add_axes([0.65, 0.65, 0.25, 0.22])
        ins.bar(range(1, n_comp+1), evr * 100, color="#3498db", alpha=0.7)
        ins.set_xlabel("PC", fontsize=7); ins.set_ylabel("%var", fontsize=7)
        ins.tick_params(labelsize=6)
    return finalise_figure(fig, sp)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_RENDERERS: Dict[str, Callable] = {
    "histogram":   _r_histogram,
    "boxplot":     _r_boxplot,
    "violin":      _r_violin,
    "kde":         _r_kde,
    "scatter":     _r_scatter,
    "joint":       _r_joint,
    "pairplot":    _r_pairplot,
    "heatmap":     _r_heatmap,
    "parallel":    _r_parallel,
    "scatter3d":   _r_scatter3d,
    "line":        _r_line,
    "area":        _r_area,
    "candlestick": _r_candlestick,
    "bar":         _r_bar,
    "swarm":       _r_swarm,
    "strip":       _r_strip,
    "tsne":        _r_tsne,
    "pca":         _r_pca,
}


def _render(plot_type: str, df: pd.DataFrame, data_page, opt_p: dict, style_p: dict) -> Figure:
    renderer = _RENDERERS.get(plot_type)
    if renderer is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No renderer for '{plot_type}'", ha="center"); return fig
    rc = rc_ctx(style_p)
    with matplotlib.rc_context(rc):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return renderer(df, data_page, opt_p, style_p)


# ---------------------------------------------------------------------------
# PlotDialog
# ---------------------------------------------------------------------------

class PlotDialog(QDialog):
    def __init__(self, plot_type: str, store: SessionStore, parent=None):
        super().__init__(parent)
        self._plot_type = plot_type
        self._store     = store
        self._fig: Optional[Figure] = None
        self._timer = QTimer(self); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._generate)

        display = next((d for d, k, _ in _PLOT_CATALOGUE if k == plot_type), plot_type)
        self.setWindowTitle(f"Plot — {display}"); self.setModal(True); self.resize(1320, 780)

        root = QHBoxLayout(self); root.setContentsMargins(10, 10, 10, 10); root.setSpacing(8)

        # ── Left panel ─────────────────────────────────────────────────────
        left = QTabWidget(); left.setMinimumWidth(330); left.setMaximumWidth(460)
        df = store.asset.dataframe if store.asset is not None else pd.DataFrame()

        self._data_page = _make_data_page(plot_type, df)
        left.addTab(_scrolled(self._data_page), "Data")

        opt_w, self._opt_getter = _make_options(plot_type, df)
        left.addTab(_scrolled(opt_w), "Options")

        style_w, self._style_getter = make_style_panel()
        left.addTab(_scrolled(style_w), "Style")

        export_w, self._export_getter = make_export_panel(
            self._save_figure, defaults={"prefix": "plot"},
        )
        left.addTab(_scrolled(export_w), "Export")

        root.addWidget(left, 0)

        # ── Right panel ────────────────────────────────────────────────────
        right = QWidget(); rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0); rlay.setSpacing(6)

        expensive = plot_type in _EXPENSIVE
        tbar = QHBoxLayout()
        if expensive:
            note = QLabel(f"⚠  '{display}' is compute-intensive — click Generate.")
            note.setObjectName("SectionHint"); tbar.addWidget(note)
        else:
            tbar.addStretch(1)
        gen_btn = QPushButton("Generate")
        gen_btn.clicked.connect(self._generate)
        tbar.addWidget(gen_btn)
        rlay.addLayout(tbar)

        self._canvas_container = QWidget()
        self._canvas_lay = QVBoxLayout(self._canvas_container)
        self._canvas_lay.setContentsMargins(0, 0, 0, 0)
        ph = QLabel("Configure parameters then click Generate (or change a setting).")
        ph.setAlignment(Qt.AlignCenter); ph.setObjectName("SectionHint")
        self._canvas_lay.addWidget(ph)
        rlay.addWidget(self._canvas_container, 1)

        root.addWidget(right, 1)

        if not expensive:
            self._connect_auto_refresh()
        if not expensive and store.asset is not None:
            QTimer.singleShot(250, self._generate)

    # ── auto-refresh ────────────────────────────────────────────────────────

    def _connect_auto_refresh(self):
        def _kick():
            self._timer.start(500)
        if hasattr(self._data_page, "changed"):
            self._data_page.changed.connect(_kick)
        seen: set = set()
        for wtype in (QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox):
            for w in self.findChildren(wtype):
                wid = id(w)
                if wid in seen: continue
                seen.add(wid)
                if isinstance(w, QComboBox):
                    w.currentIndexChanged.connect(lambda _, _w=w: _kick())
                elif isinstance(w, QCheckBox):
                    w.stateChanged.connect(lambda _, _w=w: _kick())
                else:
                    w.valueChanged.connect(lambda _, _w=w: _kick())

    def _df(self) -> pd.DataFrame:
        return self._store.asset.dataframe if self._store.asset is not None else pd.DataFrame()

    # ── generate ─────────────────────────────────────────────────────────────

    def _generate(self):
        df = self._df()
        if df.empty:
            QMessageBox.warning(self, "No data", "Load a dataset first."); return
        try:
            plt.close("all")
            fig = _render(self._plot_type, df, self._data_page,
                          self._opt_getter(), self._style_getter())
            self._set_canvas(fig)
        except Exception as exc:
            log.exception("PlotDialog render error plot_type=%s", self._plot_type)
            self._show_error(str(exc))

    def _set_canvas(self, fig: Figure):
        for i in reversed(range(self._canvas_lay.count())):
            w = self._canvas_lay.itemAt(i).widget()
            if w: w.setParent(None)
        canvas = FigureCanvas(fig); canvas.setMinimumSize(400, 300)
        self._canvas_lay.addWidget(canvas); canvas.draw()
        sync = getattr(self._style_getter, "sync", None)
        if sync is not None:
            sync(fig)
        if self._fig and self._fig is not fig:
            plt.close(self._fig)
        self._fig = fig

    def _show_error(self, msg: str):
        for i in reversed(range(self._canvas_lay.count())):
            w = self._canvas_lay.itemAt(i).widget()
            if w: w.setParent(None)
        lbl = QLabel(f"Render error:\n{msg}")
        lbl.setWordWrap(True); lbl.setObjectName("SectionHint"); lbl.setAlignment(Qt.AlignTop)
        self._canvas_lay.addWidget(lbl)

    # ── save ─────────────────────────────────────────────────────────────────

    def _save_figure(self):
        if self._fig is None:
            QMessageBox.warning(self, "Nothing to save", "Generate a plot first."); return
        ep = self._export_getter()
        fmt = ep["fmt"].lower(); dpi = int(ep["dpi"])
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure",
            os.path.expanduser(f"~/{ep['prefix']}.{fmt}"),
            f"{ep['fmt']} files (*.{fmt});;All files (*)",
        )
        if not path: return
        try:
            plt.close("all")
            fig = _render(self._plot_type, self._df(), self._data_page,
                          self._opt_getter(), self._style_getter())
            fig.set_size_inches(ep["width"], ep["height"])
            fig.savefig(path, dpi=dpi, bbox_inches="tight",
                        transparent=ep["transparent"], format=fmt)
            plt.close(fig)
            QMessageBox.information(self, "Saved", f"Figure saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            log.exception("PlotDialog._save_figure")
