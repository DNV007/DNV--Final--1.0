"""
DNV Scientific Module
---------------------
Role:
    Provides the CorrelationDialog — a fully configurable, canvas-embedded
    dialog for 22 correlation and dependence measure visualisations, with
    exhaustive per-method options, a shared style panel, and a complete
    export/save panel.

Scientific Context:
    Accepts a method key and a SessionStore; dispatches to cor_engine for
    pairwise matrix computation or ranked series; renders the result as one
    of five figure types: heatmap + RadViz, MI/LASSO bar chart, CCF stem
    plot, DAG network, or Bayesian Network graph.  All style parameters are
    applied through a single finalise_figure() / apply_style() call path.

Invariants:
    - No computation occurs at dialog construction time; only on Generate.
    - Every numeric constant is a named Final with a unit suffix.
    - apply_style() is the single site that writes per-axes Qt/matplotlib
      state; no display attribute is set elsewhere in the renderers.
    - The single FigureCanvas is replaced atomically on each Generate call.

Assumptions:
    - store.asset is a valid DataAsset with a DataFrame containing ≥ 2
      numeric columns.
    - cor_engine functions are pure and raise ImportError for missing deps.
    - matplotlib backend is "Agg" (set at application startup in main.py).

Failure Modes:
    - Fewer than 2 numeric columns: Generate shows an error dialog, no crash.
    - Missing optional dependency (statsmodels, pgmpy, pyinform): ImportError
      from cor_engine is caught and shown as an install-hint message.
    - Singular covariance in partial_correlation: is_valid=False is shown as
      a warning annotation on the heatmap.

Provenance:
    - This module emits no transformation metadata.  Provenance originates
      in the calling CorrelationWindow/CorrelationDialog session.
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
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.decomposition import PCA as _PCA
from sklearn.cluster import DBSCAN as _DBSCAN

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from gui_session import SessionStore
from gui_style import (
    ColorBtn, hrow, eyebrow, section_sep, scrolled, combo, spin, dspin, check,
    make_style_panel, make_export_panel,
    is_categorical_axis, rc_ctx, apply_tick_format, apply_style, finalise_figure,
)
import cor_engine as _eng

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
DEFAULT_FIG_WIDTH_IN: Final[float] = 14.0

#: Default figure height [inches].
DEFAULT_FIG_HEIGHT_IN: Final[float] = 7.0

#: Default figure DPI for screen rendering.
DEFAULT_SCREEN_DPI: Final[int] = 100

#: Default export DPI.
DEFAULT_EXPORT_DPI: Final[int] = 150

#: Default DBSCAN epsilon for RadViz cluster detection [dimensionless].
DEFAULT_DBSCAN_EPS: Final[float] = 0.20

#: Default DBSCAN minimum samples.
DEFAULT_DBSCAN_MIN_SAMPLES: Final[int] = 2

#: Default correlation threshold for high-pair circle annotation [dimensionless].
DEFAULT_CORR_THRESHOLD: Final[float] = 0.80

#: Default CCF lag count.
DEFAULT_CCF_NLAGS: Final[int] = 40

#: Default LASSO regularisation strength [dimensionless].
DEFAULT_LASSO_ALPHA: Final[float] = 0.10

#: Spacing added after each section separator in the left panel [px].
SECTION_SPACING_PX: Final[int] = 4

#: Marker glyph cycle used by the RadViz scatter to keep overlapping feature
#: points distinguishable when inline text annotation is off.  Length 16 so
#: tab10/Set2 color palettes cycle at a different period, giving ≥ 40 unique
#: (marker, color) pairs before any combination repeats.
MARKER_CYCLE: Final[List[str]] = [
    "o", "s", "^", "D", "v", "P", "*", "X",
    "h", "<", ">", "p", "d", "H", "8", "1",
]

_COLORMAPS: Final[List[str]] = [
    "coolwarm","RdBu","seismic","bwr","vlag","icefire",
    "viridis","plasma","inferno","magma",
    "Blues","Greens","Oranges","Reds","Purples",
    "YlOrRd","BuGn","PuBu",
]
_PALETTES: Final[List[str]] = ["tab10","Set1","Set2","Set3","husl","hls","muted","deep"]
_LINESTYLES: Final[List[str]] = ["--", "-", ":", "-."]
_SCALES: Final[List[str]] = ["linear","log","symlog","logit"]
_METRICS: Final[List[str]] = ["euclidean","cityblock","chebyshev","cosine","correlation"]
_KERNELS: Final[List[str]] = ["rbf","linear"]
_CORR_METHODS: Final[List[str]] = ["pearson","spearman","kendall"]


def _read(w) -> Any:
    """Read any supported control type."""
    if isinstance(w, QComboBox):      return w.currentText()
    if isinstance(w, QCheckBox):      return w.isChecked()
    if isinstance(w, QSpinBox):       return w.value()
    if isinstance(w, QDoubleSpinBox): return w.value()
    if isinstance(w, ColorBtn):       return w.color()
    if isinstance(w, QLineEdit):      return w.text().strip()
    return None


# ---------------------------------------------------------------------------
# Column selector
# ---------------------------------------------------------------------------

class _MultiColSelector(QListWidget):
    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        for col in columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.addItem(item)
        self.setMaximumHeight(170)

    def selected(self) -> List[str]:
        return [self.item(i).text() for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked]

    def select_all(self) -> None:
        for i in range(self.count()): self.item(i).setCheckState(Qt.Checked)

    def deselect_all(self) -> None:
        for i in range(self.count()): self.item(i).setCheckState(Qt.Unchecked)


# ---------------------------------------------------------------------------
# Data pages
# ---------------------------------------------------------------------------

class _MatrixDataPage(QWidget):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(eyebrow("NUMERIC COLUMNS"))
        num_cols = list(df.select_dtypes(include="number").columns)
        self._sel = _MultiColSelector(num_cols)
        lay.addWidget(self._sel)
        btn_row = QHBoxLayout()
        ba = QPushButton("All"); bn = QPushButton("None")
        ba.clicked.connect(self._sel.select_all)
        bn.clicked.connect(self._sel.deselect_all)
        btn_row.addWidget(ba); btn_row.addWidget(bn); btn_row.addStretch()
        lay.addLayout(btn_row); lay.addStretch()

    def selected_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._sel.selected()
        return df[cols] if cols else df.select_dtypes(include="number")


class _TargetDataPage(QWidget):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        num_cols = list(df.select_dtypes(include="number").columns)
        lay.addWidget(eyebrow("TARGET COLUMN"))
        self._target = combo(num_cols)
        lay.addWidget(hrow("Target:", self._target))
        lay.addWidget(eyebrow("FEATURE COLUMNS"))
        self._sel = _MultiColSelector(num_cols)
        lay.addWidget(self._sel)
        ba = QPushButton("All"); bn = QPushButton("None")
        ba.clicked.connect(self._sel.select_all)
        bn.clicked.connect(self._sel.deselect_all)
        btn_row = QHBoxLayout()
        btn_row.addWidget(ba); btn_row.addWidget(bn); btn_row.addStretch()
        lay.addLayout(btn_row); lay.addStretch()

    def target(self) -> str:    return self._target.currentText()
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._sel.selected() if c != self.target()]
        return df[cols + [self.target()]] if cols else df.select_dtypes("number")


class _TwoColDataPage(QWidget):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        num_cols = list(df.select_dtypes(include="number").columns)
        lay.addWidget(eyebrow("SERIES X (lead)"))
        self._x = combo(num_cols)
        lay.addWidget(hrow("X column:", self._x))
        lay.addWidget(eyebrow("SERIES Y (lag)"))
        self._y = combo(num_cols, default=(num_cols[1] if len(num_cols) > 1 else num_cols[0]))
        lay.addWidget(hrow("Y column:", self._y))
        lay.addStretch()

    def x_col(self) -> str: return self._x.currentText()
    def y_col(self) -> str: return self._y.currentText()


# ---------------------------------------------------------------------------
# Options panel — exhaustive per-method controls
# ---------------------------------------------------------------------------

def _make_options(key: str) -> Tuple[QWidget, Callable[[], dict]]:
    """
    Build the Options tab for the given method key.

    Returns (widget, getter) where getter() → dict of option values.
    """
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
    controls: Dict[str, Any] = {}

    def _add(label: str, widget, key_: str) -> None:
        controls[key_] = widget; lay.addWidget(hrow(label, widget))

    def _sec(text: str) -> None:
        lay.addWidget(section_sep()); lay.addWidget(eyebrow(text))
        lay.addSpacing(SECTION_SPACING_PX)

    # ── Shared heatmap controls (all matrix methods) ──────────────────────
    if key not in ("mi", "lasso", "ccf", "dag", "bayesian"):
        _sec("HEATMAP")
        _add("Colormap:",         combo(_COLORMAPS, "coolwarm"),             "cmap")
        _add("Annotate cells:",   check(True),                               "annotate")
        _add("Annot font size:",  spin(5, 20, 8),                            "annot_fontsize")
        _add("Annot format:",     combo(["0.1f","0.2f","0.3f","0.1e","0.2e"], "0.2f"), "annot_fmt")
        _add("vmin:",             dspin(-1.0, 0.0, -1.0, 0.05),             "vmin")
        _add("vmax:",             dspin(0.0, 1.0,  1.0, 0.05),              "vmax")
        _add("Mask upper tri:",   check(True),                               "mask_upper")
        _add("Cell linewidths:",  dspin(0.0, 3.0, 0.4, 0.1),               "linewidths")
        _add("Cell line color:",  ColorBtn("#aaaaaa"),                       "linecolor")
        _add("Colorbar shrink:",  dspin(0.3, 1.0, 0.8, 0.05),              "cbar_shrink")
        _add("Colorbar label:",   QLineEdit(),                                "cbar_label")
        controls["cbar_label"].setPlaceholderText("e.g. Correlation")
        _add("Robust colorscale:",check(False),                              "robust")

        _sec("THRESHOLD ANNOTATION")
        _add("Threshold:",        dspin(0.0, 1.0, DEFAULT_CORR_THRESHOLD, 0.05), "threshold")
        _add("Circle color:",     ColorBtn("#111111"),                       "circle_color")
        _add("Circle linewidth:", dspin(0.5, 5.0, 2.0, 0.5),               "circle_lw")

        _sec("RADVIZ (PCA PROJECTION)")
        _add("Show RadViz:",      check(True),                               "show_radviz")
        _add("DBSCAN eps:",       dspin(0.05, 2.0, DEFAULT_DBSCAN_EPS, 0.05), "dbscan_eps")
        _add("DBSCAN min pts:",   spin(1, 20, DEFAULT_DBSCAN_MIN_SAMPLES),  "dbscan_min")
        _add("Point size:",       spin(20, 500, 120),                       "point_size")
        _add("Point alpha:",      dspin(0.1, 1.0, 0.85, 0.05),             "point_alpha")
        _add("Label mode:",       combo(["legend","inline","off"], "legend"), "label_mode")
        _add("Label font size:",  spin(5, 18, 8),                           "label_fontsize")

    # ── Method-specific controls ──────────────────────────────────────────
    if key == "distance":
        _sec("DISTANCE METRIC")
        _add("Metric:", combo(_METRICS, "euclidean"), "metric")

    elif key == "hsic":
        _sec("HSIC KERNEL")
        _add("Kernel:", combo(_KERNELS, "rbf"), "kernel")

    elif key == "polychoric":
        _sec("ORDINAL SPEARMAN (BINNED)")
        _add("Discretize:",   check(False), "discretize")
        _add("Bins:",         spin(2, 20, 5), "bins")

    elif key in ("mic", "mice"):
        _sec("k-NN MUTUAL INFORMATION")
        _add("Random state:", spin(0, 9999, 0), "random_state")

    elif key == "sparse_partial":
        _sec("GRAPHICAL LASSO")
        _add("CV folds:", spin(2, 10, 5), "cv")

    elif key == "granger":
        _sec("GRANGER CAUSALITY")
        _add("Max lag:", spin(1, 30, 5), "max_lag")

    elif key == "transfer_entropy":
        _sec("TRANSFER ENTROPY")
        _add("History k:", spin(1, 10, 1), "k")
        _add("Discretisation bins:", spin(2, 20, _eng.TE_DEFAULT_N_BINS), "n_bins")

    elif key == "autoencoder":
        _sec("MLP RECONSTRUCTION")
        _add("Hidden layer 1:", spin(4, 256, 16), "hidden1")
        _add("Hidden layer 2:", spin(0, 256, 8),  "hidden2")
        _add("Max iterations:", spin(50, 1000, 300), "max_iter")
        _add("Random state:",   spin(0, 9999, 42),   "random_state")

    elif key == "mi":
        _sec("MUTUAL INFORMATION")
        _add("Bar color:",        ColorBtn("#3a7bd5"), "bar_color")
        _add("Bar alpha:",        dspin(0.3, 1.0, 0.85, 0.05), "bar_alpha")
        _add("Edge color:",       ColorBtn("#1a3a6a"), "edge_color")
        _add("Edge linewidth:",   dspin(0.0, 3.0, 0.5, 0.1), "edge_lw")
        _add("Top N features:",   spin(1, 200, 20), "top_n")
        _add("Orientation:",      combo(["horizontal","vertical"], "horizontal"), "orientation")
        _add("Show value labels:",check(True), "show_labels")
        _add("Label font size:",  spin(5, 16, 8), "label_fontsize")
        _add("Sort descending:",  check(True), "sort_desc")

    elif key == "lasso":
        _sec("LASSO")
        # α = 0 selects LassoCV (CV-optimal penalty).
        _add("Alpha (λ, 0=CV):",  dspin(0.0, 50.0, DEFAULT_LASSO_ALPHA, 0.01), "alpha")
        _add("CV folds (if α=0):", spin(2, 10, 5), "cv_folds")
        _add("Max iterations:",   spin(100, 50000, 5000), "max_iter")
        _sec("BAR DISPLAY")
        _add("Positive color:",   ColorBtn("#3a7bd5"), "pos_color")
        _add("Negative color:",   ColorBtn("#e74c3c"), "neg_color")
        _add("Bar alpha:",        dspin(0.3, 1.0, 0.85, 0.05), "bar_alpha")
        _add("Edge linewidth:",   dspin(0.0, 3.0, 0.5, 0.1), "edge_lw")
        _add("Top N features:",   spin(1, 200, 20), "top_n")
        _add("Only non-zero:",    check(False), "nonzero_only")
        _add("Show value labels:",check(True), "show_labels")
        _add("Orientation:",      combo(["horizontal","vertical"], "horizontal"), "orientation")

    elif key == "ccf":
        _sec("CCF")
        _add("Max lags:",         spin(5, 300, DEFAULT_CCF_NLAGS), "nlags")
        _add("Show CI lines:",    check(True), "show_ci")
        _add("CI level %:",       combo(["90","95","99"], "95"), "ci_level")
        _add("CI color:",         ColorBtn("#e74c3c"), "ci_color")
        _add("Stem color:",       ColorBtn("#3a7bd5"), "stem_color")
        _add("Marker size:",      spin(1, 15, 3), "marker_size")
        _add("Baseline color:",   ColorBtn("#2c3e50"), "baseline_color")

    elif key == "dag":
        _sec("DAG")
        _add("Threshold:",        dspin(0.0, 1.0, 0.30, 0.05), "threshold")
        _add("Corr method:",      combo(_CORR_METHODS, "pearson"), "method")
        _add("Positive color:",   ColorBtn("#3a7bd5"), "pos_color")
        _add("Negative color:",   ColorBtn("#e74c3c"), "neg_color")
        _add("Arrow width scale:",dspin(0.5, 5.0, 1.5, 0.5), "arrow_width_scale")
        _add("Node size:",        spin(100, 5000, 800), "node_size")
        _add("Node fill color:",  ColorBtn("#ecf0f1"), "node_color")
        _add("Node edge color:",  ColorBtn("#2c3e50"), "node_edge_color")
        _add("Label font size:",  spin(5, 20, 8), "label_fontsize")

    elif key == "bayesian":
        _sec("BAYESIAN NETWORK")
        _add("Discretize bins:", spin(2, 20, 5), "n_bins")
        _add("Node size:",       spin(100, 5000, 900), "node_size")
        _add("Node fill color:", ColorBtn("#d5e8f7"), "node_color")
        _add("Edge color:",      ColorBtn("#e67e22"), "edge_color")
        _add("Label font size:", spin(5, 20, 8), "label_fontsize")

    lay.addStretch()

    def _getter() -> dict:
        return {k: _read(v) for k, v in controls.items()}

    return w, _getter


# ---------------------------------------------------------------------------
# Style panel — full shared panel (identical pattern to gui_dlg_viz_plot.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_heatmap_radviz(
    matrix: pd.DataFrame,
    op: dict,
    sp: dict,
    title: str,
    is_valid: bool = True,
) -> Figure:
    """
    Lower-triangle correlation heatmap with threshold-circle overlays,
    and optional RadViz (PCA + DBSCAN projection) side panel.
    """
    import seaborn as sns
    show_radviz = op.get("show_radviz", True)
    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN) if "width" in sp else DEFAULT_FIG_WIDTH_IN
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN) if "height" in sp else DEFAULT_FIG_HEIGHT_IN

    ncols_fig = 2 if show_radviz else 1
    width_ratios = [1.4, 1] if show_radviz else None

    with plt.rc_context(rc_ctx(sp)):
        fig, axes = plt.subplots(
            1, ncols_fig,
            figsize=(w_in, h_in),
            gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
        )
        ax_heat = axes[0] if show_radviz else axes

        # ── Heatmap ──────────────────────────────────────────────────────
        mask = np.triu(np.ones_like(matrix.values, dtype=bool)) if op.get("mask_upper", True) else None
        cbar_kws: dict = {"shrink": float(op.get("cbar_shrink", 0.8))}
        cbar_label = op.get("cbar_label", "")
        if cbar_label: cbar_kws["label"] = cbar_label

        sns.heatmap(
            matrix,
            mask=mask,
            annot=op.get("annotate", True),
            fmt=str(op.get("annot_fmt", "0.2f")),
            annot_kws={"size": int(op.get("annot_fontsize", 8))},
            cmap=op.get("cmap", "coolwarm"),
            linewidths=float(op.get("linewidths", 0.4)),
            linecolor=op.get("linecolor", "#aaaaaa"),
            vmin=float(op.get("vmin", -1.0)),
            vmax=float(op.get("vmax",  1.0)),
            robust=bool(op.get("robust", False)),
            cbar_kws=cbar_kws,
            ax=ax_heat,
        )
        threshold = float(op.get("threshold", DEFAULT_CORR_THRESHOLD))
        circ_color = op.get("circle_color", "#111111")
        circ_lw    = float(op.get("circle_lw", 2.0))
        n = len(matrix.columns)
        count = 0
        for i in range(n):
            for j in range(i):
                val = matrix.iloc[i, j]
                if pd.notna(val) and abs(float(val)) > threshold:
                    count += 1
                    ax_heat.add_patch(Circle(
                        (j + 0.5, i + 0.5), 0.44,
                        color=circ_color, fill=False, linewidth=circ_lw,
                    ))

        htitle = sp.get("title") or f"{title}\n{count} pairs |r| > {threshold:.2f}"
        ax_heat.set_title(htitle, fontsize=sp.get("title_fontsize", 11), fontweight="bold")
        if not is_valid:
            ax_heat.set_xlabel(
                "⚠ Singular covariance — identity fallback used",
                color="#c0392b", fontsize=9,
            )

        # ── RadViz ───────────────────────────────────────────────────────
        if show_radviz:
            ax_rv = axes[1]
            try:
                data_pca = matrix.fillna(0).values
                pca = _PCA(n_components=2)
                pcs = pca.fit_transform(data_pca)
                pcs_norm = pcs / (np.max(np.abs(pcs)) + 1e-8)

                eps_val  = float(op.get("dbscan_eps", DEFAULT_DBSCAN_EPS))
                min_pts  = int(op.get("dbscan_min", DEFAULT_DBSCAN_MIN_SAMPLES))
                db       = _DBSCAN(eps=eps_val, min_samples=min_pts).fit(pcs_norm)
                labels   = db.labels_

                pt_size   = int(op.get("point_size", 120))
                pt_alpha  = float(op.get("point_alpha", 0.85))
                lbl_fs    = int(op.get("label_fontsize", 8))
                label_mode = str(op.get("label_mode", "legend"))

                # Distinct (marker, color) pair per feature so overlapping
                # points stay distinguishable even without inline text; the
                # legend maps the combined glyph back to the feature name.
                marker_cycle = MARKER_CYCLE
                palette = (
                    list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
                    + list(plt.cm.Set3.colors)
                )
                feats = list(matrix.columns)
                handles = []
                for i, feat in enumerate(feats):
                    marker = marker_cycle[i % len(marker_cycle)]
                    color  = palette[i % len(palette)]
                    sc = ax_rv.scatter(
                        pcs_norm[i, 0], pcs_norm[i, 1],
                        s=pt_size, color=color, marker=marker,
                        alpha=pt_alpha, edgecolors="#333", linewidths=0.6,
                        zorder=3, label=feat,
                    )
                    handles.append(sc)
                    if label_mode == "inline":
                        ax_rv.annotate(
                            feat, (pcs_norm[i, 0], pcs_norm[i, 1]),
                            fontsize=lbl_fs, ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points",
                        )

                # DBSCAN cluster hulls kept as before (crimson dashed).
                for k in set(labels):
                    if k == -1: continue
                    mask_c = (labels == k)
                    xy = pcs_norm[mask_c]
                    center = xy.mean(axis=0)
                    radius = np.max(np.linalg.norm(xy - center, axis=1)) + 0.06
                    ax_rv.add_artist(plt.Circle(
                        center, radius, color="crimson",
                        fill=False, linewidth=1.8, linestyle="--",
                    ))
                    ax_rv.annotate(
                        f"C{k+1}", (center[0], center[1] + radius + 0.03),
                        ha="center", fontsize=9, fontweight="bold",
                    )

                ev = pca.explained_variance_ratio_
                ax_rv.set_xlabel(f"PC 1 ({ev[0]:.1%})", fontsize=9)
                ax_rv.set_ylabel(f"PC 2 ({ev[1]:.1%})", fontsize=9)
                ax_rv.set_title("RadViz — PCA projection", fontsize=10, fontweight="bold")
                ax_rv.set_xlim(-1.4, 1.4); ax_rv.set_ylim(-1.4, 1.4)
                ax_rv.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

                if label_mode == "legend" and handles:
                    ncol = 1 if len(feats) <= 12 else 2
                    ax_rv.legend(
                        handles=handles,
                        loc="center left", bbox_to_anchor=(1.02, 0.5),
                        fontsize=lbl_fs, frameon=True, framealpha=0.85,
                        ncol=ncol, handletextpad=0.4, labelspacing=0.35,
                        borderpad=0.3,
                    )

            except Exception as exc:
                log.warning("RadViz render failed: %s", exc)
                ax_rv.text(0.5, 0.5, f"RadViz error:\n{exc}",
                           ha="center", va="center", transform=ax_rv.transAxes,
                           fontsize=9, color="#c0392b")

    return finalise_figure(fig, sp)


def _render_bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    op: dict,
    sp: dict,
) -> Figure:
    """Horizontal (or vertical) bar chart for MI / LASSO outputs."""
    top_n = int(op.get("top_n", 20))
    sort_desc = op.get("sort_desc", True)
    orientation = op.get("orientation", "horizontal")

    data = series.head(top_n) if sort_desc else series.tail(top_n).iloc[::-1]

    pos_color = op.get("pos_color", op.get("bar_color", "#3a7bd5"))
    neg_color = op.get("neg_color", "#e74c3c")
    bar_alpha  = float(op.get("bar_alpha", 0.85))
    edge_lw    = float(op.get("edge_lw", 0.5))
    show_labels= op.get("show_labels", True)
    lbl_fs     = int(op.get("label_fontsize", 8))

    bar_colors = [neg_color if v < 0 else pos_color for v in data.values]

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN * 0.75)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN)

    with plt.rc_context(rc_ctx(sp)):
        fig, ax = plt.subplots(figsize=(w_in, h_in))

        if orientation == "horizontal":
            bars = ax.barh(
                range(len(data)), data.values,
                color=bar_colors, alpha=bar_alpha,
                edgecolor="#333333", linewidth=edge_lw,
            )
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
            ax.set_xlabel(xlabel, fontsize=10)
            if show_labels:
                for bar_, val in zip(bars, data.values):
                    ax.text(
                        val + 0.002 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        bar_.get_y() + bar_.get_height() / 2,
                        f"{val:.3f}", va="center", ha="left", fontsize=lbl_fs,
                    )
        else:
            bars = ax.bar(
                range(len(data)), data.values,
                color=bar_colors, alpha=bar_alpha,
                edgecolor="#333333", linewidth=edge_lw,
            )
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha="right", fontsize=9)
            ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
            ax.set_ylabel(xlabel, fontsize=10)
            if show_labels:
                for bar_, val in zip(bars, data.values):
                    ax.text(
                        bar_.get_x() + bar_.get_width() / 2,
                        val + 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=lbl_fs,
                    )

        ax.set_title(sp.get("title") or title, fontsize=sp.get("title_fontsize", 12), fontweight="bold")
        ax.grid(True, axis="x" if orientation == "horizontal" else "y",
                linestyle="--", linewidth=0.4, alpha=0.5)

    return finalise_figure(fig, sp)


def _render_ccf(
    lags: np.ndarray,
    vals: np.ndarray,
    x_name: str,
    y_name: str,
    op: dict,
    sp: dict,
) -> Figure:
    """CCF stem plot with configurable CI bands."""
    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN * 0.75)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN * 0.85)

    with plt.rc_context(rc_ctx(sp)):
        fig, ax = plt.subplots(figsize=(w_in, h_in))

        stem_color    = op.get("stem_color",     "#3a7bd5")
        baseline_color= op.get("baseline_color", "#2c3e50")
        marker_size   = int(op.get("marker_size", 3))
        ci_color      = op.get("ci_color",       "#e74c3c")

        markerline, stemlines, baseline = ax.stem(lags, vals)
        plt.setp(stemlines, linewidth=0.8, color=stem_color)
        plt.setp(markerline, markersize=marker_size, color=stem_color)
        plt.setp(baseline, linewidth=0.8, color=baseline_color)

        if op.get("show_ci", True):
            ci_pct = int(op.get("ci_level", "95").replace("%",""))
            z_map  = {90: 1.645, 95: 1.960, 99: 2.576}
            z      = z_map.get(ci_pct, 1.960)
            ci_val = z / np.sqrt(max(len(vals), 1))
            ax.axhline( ci_val, color=ci_color, linewidth=0.9, linestyle="--",
                        label=f"{ci_pct}% CI (±{ci_val:.3f})")
            ax.axhline(-ci_val, color=ci_color, linewidth=0.9, linestyle="--")
            ax.legend(fontsize=8)

        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("Cross-Correlation", fontsize=10)
        ax.set_title(
            sp.get("title") or f"CCF\n{x_name} (lead) vs {y_name} (lag)",
            fontsize=sp.get("title_fontsize", 11), fontweight="bold",
        )
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    return finalise_figure(fig, sp)


def _render_dag(
    df: pd.DataFrame,
    op: dict,
    sp: dict,
) -> Figure:
    """Directed dependency graph rendered with pure matplotlib (no networkx)."""
    numeric = df.select_dtypes(include="number").dropna()
    method    = op.get("method", "pearson")
    threshold = float(op.get("threshold", 0.30))
    corr      = numeric.corr(method=method)
    nodes     = list(corr.columns)
    n         = len(nodes)

    edges: List[tuple] = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i >= j: continue
            val = float(corr.iloc[i, j])
            if abs(val) > threshold:
                src, dst = (a, b) if val > 0 else (b, a)
                edges.append((src, dst, val))

    pos_color  = op.get("pos_color",      "#3a7bd5")
    neg_color  = op.get("neg_color",      "#e74c3c")
    aw_scale   = float(op.get("arrow_width_scale", 1.5))
    node_size  = int(op.get("node_size",  800))
    node_color = op.get("node_color",     "#ecf0f1")
    node_ec    = op.get("node_edge_color","#2c3e50")
    lbl_fs     = int(op.get("label_fontsize", 8))

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN * 0.8)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN * 1.1)

    with plt.rc_context(rc_ctx(sp)):
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {nd: (np.cos(a), np.sin(a)) for nd, a in zip(nodes, angles)}

        for (src, dst, val) in edges:
            xs, ys = pos[src]; xd, yd = pos[dst]
            lw = min(4.0, abs(val) * aw_scale * 3)
            color = pos_color if val > 0 else neg_color
            ax.annotate(
                "", xy=(xd, yd), xytext=(xs, ys),
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=lw, connectionstyle="arc3,rad=0.12",
                ),
            )

        ax.scatter(
            [pos[nd][0] for nd in nodes],
            [pos[nd][1] for nd in nodes],
            s=node_size, color=node_color,
            edgecolors=node_ec, linewidths=1.5, zorder=5,
        )
        for nd in nodes:
            ax.text(pos[nd][0], pos[nd][1], nd,
                    ha="center", va="center",
                    fontsize=lbl_fs, fontweight="bold", zorder=6)

        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(
            sp.get("title") or (
                f"DAG ({method} |r| > {threshold:.2f})\n{len(edges)} edges"
            ),
            fontsize=sp.get("title_fontsize", 11), fontweight="bold",
        )

    return finalise_figure(fig, sp)


def _render_bayesian(edges: list, nodes: list, op: dict, sp: dict) -> Figure:
    """Learned Bayesian Network structure rendered with matplotlib."""
    n          = len(nodes)
    node_size  = int(op.get("node_size", 900))
    node_color = op.get("node_color", "#d5e8f7")
    edge_color = op.get("edge_color", "#e67e22")
    lbl_fs     = int(op.get("label_fontsize", 8))

    w_in = sp.get("width", DEFAULT_FIG_WIDTH_IN * 0.8)
    h_in = sp.get("height", DEFAULT_FIG_HEIGHT_IN * 1.1)

    with plt.rc_context(rc_ctx(sp)):
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        if n == 0:
            ax.text(0.5, 0.5, "No nodes.", ha="center", va="center",
                    transform=ax.transAxes)
            return finalise_figure(fig, sp)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {nd: (np.cos(a), np.sin(a)) for nd, a in zip(nodes, angles)}

        for (src, dst) in edges:
            if src not in pos or dst not in pos: continue
            ax.annotate(
                "", xy=pos[dst], xytext=pos[src],
                arrowprops=dict(
                    arrowstyle="-|>", color=edge_color,
                    lw=1.5, connectionstyle="arc3,rad=0.15",
                ),
            )

        ax.scatter(
            [pos[nd][0] for nd in nodes],
            [pos[nd][1] for nd in nodes],
            s=node_size, color=node_color,
            edgecolors="#2c3e50", linewidths=1.5, zorder=5,
        )
        for nd in nodes:
            ax.text(pos[nd][0], pos[nd][1], nd,
                    ha="center", va="center",
                    fontsize=lbl_fs, fontweight="bold", zorder=6)

        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(
            sp.get("title") or "Bayesian Network Structure",
            fontsize=sp.get("title_fontsize", 11), fontweight="bold",
        )

    return finalise_figure(fig, sp)


# ---------------------------------------------------------------------------
# CorrelationDialog
# ---------------------------------------------------------------------------

class CorrelationDialog(QDialog):
    """
    Unified correlation analysis dialog.

    Left panel  : QTabWidget — Data / Options / Style / Export (4 tabs)
    Right panel : FigureCanvasQTAgg

    Computation occurs only on Generate click; no auto-refresh.
    """

    def __init__(self, key: str, store: SessionStore, parent=None):
        super().__init__(parent)
        self._key   = key
        self._store = store
        self._fig:    Optional[Figure]      = None
        self._canvas: Optional[FigureCanvas] = None

        method_title = _eng._DISPATCH.get(key, (key,))[0]
        self.setWindowTitle(f"Correlation — {method_title}")
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
        data_tab = self._build_data_tab(key, self._df)
        self._data_page = data_tab
        self._tabs.addTab(scrolled(data_tab), "Data")

        # Tab 2 — Options
        opt_inner, self._opt_getter = _make_options(key)
        self._tabs.addTab(scrolled(opt_inner), "Options")

        # Tab 3 — Style
        sty_inner, self._sty_getter = make_style_panel(
            defaults={"tick_rotation_x": 45, "grid": False})
        self._tabs.addTab(scrolled(sty_inner), "Style")

        # Tab 4 — Export
        exp_inner, self._exp_getter = make_export_panel(
            self._save_figure,
            defaults={"prefix": "correlation", "dpi": DEFAULT_EXPORT_DPI,
                      "width": DEFAULT_FIG_WIDTH_IN, "height": DEFAULT_FIG_HEIGHT_IN})
        self._tabs.addTab(scrolled(exp_inner), "Export")

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

        self._placeholder = QLabel("Configure options and click  Generate.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setObjectName("SectionHint")
        self._placeholder.setMinimumSize(CANVAS_MIN_WIDTH_PX, CANVAS_MIN_HEIGHT_PX)
        right.addWidget(self._placeholder, 1)
        self._right = right

    # ── Data tab factory ─────────────────────────────────────────────────

    def _build_data_tab(self, key: str, df: pd.DataFrame) -> QWidget:
        if key in ("mi", "lasso"):
            return _TargetDataPage(df)
        if key == "ccf":
            return _TwoColDataPage(df)
        return _MatrixDataPage(df)

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
            log.exception("CorrelationDialog._generate")
            QMessageBox.critical(self, "Computation Error", str(exc))
        finally:
            self._gen_btn.setEnabled(True)
            self._gen_btn.setText("Generate")

    def _do_generate(self) -> None:
        key = self._key
        df  = self._df
        if df.empty:
            raise ValueError("No dataset loaded.")

        op = self._opt_getter()
        sp = self._sty_getter()

        # ── Special series methods ────────────────────────────────────────
        if key == "mi":
            page: _TargetDataPage = self._data_page   # type: ignore[assignment]
            target = page.target()
            sub_df = page.features(df)
            series = _eng.mutual_information_scores(sub_df, target)
            if op.get("sort_desc", True):
                series = series.sort_values(ascending=False)
            fig = _render_bar(
                series,
                f"Mutual Information vs '{target}'",
                "MI Score",
                op, sp,
            )
            self._set_canvas(fig)
            return

        if key == "lasso":
            page2: _TargetDataPage = self._data_page  # type: ignore[assignment]
            target = page2.target()
            sub_df = page2.features(df)
            alpha    = float(op.get("alpha", DEFAULT_LASSO_ALPHA))
            max_iter = int(op.get("max_iter", 5000))
            cv_folds = int(op.get("cv_folds", 5))
            series = _eng.lasso_coefficients(
                sub_df, target,
                alpha=alpha, max_iter=max_iter, cv_folds=cv_folds,
            )
            if op.get("nonzero_only", False):
                series = series[series != 0]
            # The engine stores the used α (CV-selected if user passed 0) in
            # series.name; surface it in the title so the reader always knows
            # which penalty produced the shown coefficients.
            used_alpha = series.name or f"α={alpha}"
            fig = _render_bar(
                series,
                f"LASSO Coefficients — target '{target}'  [{used_alpha}]",
                "Standardised Coefficient",
                op, sp,
            )
            self._set_canvas(fig)
            return

        if key == "ccf":
            page3: _TwoColDataPage = self._data_page  # type: ignore[assignment]
            x_col, y_col = page3.x_col(), page3.y_col()
            nlags = int(op.get("nlags", DEFAULT_CCF_NLAGS))
            lags, vals = _eng.ccf_series(df[x_col], df[y_col], nlags=nlags)
            fig = _render_ccf(lags, vals, x_col, y_col, op, sp)
            self._set_canvas(fig)
            return

        if key == "dag":
            num_df = df.select_dtypes(include="number")
            if len(num_df.columns) < 2:
                raise ValueError("At least 2 numeric columns required.")
            fig = _render_dag(num_df, op, sp)
            self._set_canvas(fig)
            return

        if key == "bayesian":
            sub_df = (
                self._data_page.selected_df(df)
                if isinstance(self._data_page, _MatrixDataPage)
                else df.select_dtypes("number")
            )
            n_bins = int(op.get("n_bins", 5))
            edges  = _eng.bayesian_network_edges(sub_df, n_bins=n_bins)
            nodes  = list(sub_df.select_dtypes("number").columns)
            fig = _render_bayesian(edges, nodes, op, sp)
            self._set_canvas(fig)
            return

        # ── Matrix methods ────────────────────────────────────────────────
        if isinstance(self._data_page, _MatrixDataPage):
            sub_df = self._data_page.selected_df(df)
        else:
            sub_df = df.select_dtypes(include="number")

        sub_num = sub_df.select_dtypes(include="number")
        if len(sub_num.columns) < 2:
            raise ValueError("Select at least 2 numeric columns.")

        is_valid = True

        if key == "partial":
            matrix, is_valid = _eng.partial_correlation_matrix(sub_num)
        elif key == "distance":
            matrix = _eng.distance_correlation_matrix(
                sub_num, metric=str(op.get("metric", "euclidean"))
            )
        elif key == "hsic":
            matrix = _eng.hsic_matrix(sub_num, kernel=str(op.get("kernel", "rbf")))
        elif key == "polychoric":
            matrix = _eng.polychoric_matrix(
                sub_num,
                bins=int(op.get("bins", 5)),
                discretize=bool(op.get("discretize", False)),
            )
        elif key in ("mic", "mice"):
            matrix = _eng.mic_matrix(sub_num)
        elif key == "granger":
            matrix = _eng.granger_matrix(sub_num, max_lag=int(op.get("max_lag", 5)))
        elif key == "transfer_entropy":
            matrix = _eng.transfer_entropy_matrix(
                sub_num,
                k=int(op.get("k", 1)),
                n_bins=int(op.get("n_bins", _eng.TE_DEFAULT_N_BINS)),
            )
        elif key == "sparse_partial":
            matrix, is_valid = _eng.sparse_partial_correlation(
                sub_num, cv=int(op.get("cv", 5))
            )
        elif key == "cca":
            matrix = _eng.cca_matrix(sub_num)
        elif key == "autoencoder":
            matrix = _eng.autoencoder_dependency(sub_num)
        else:
            fn = _eng._DISPATCH.get(key, (None, None, None))[2]
            if fn is None:
                raise ValueError(f"Unknown method: {key!r}")
            matrix = fn(sub_num)

        method_title = _eng._DISPATCH.get(key, (key,))[0]
        fig = _render_heatmap_radviz(matrix, op, sp, method_title, is_valid=is_valid)
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

    # ── Save ─────────────────────────────────────────────────────────────

    def _save_figure(self) -> None:
        if self._fig is None:
            QMessageBox.information(self, "No Figure", "Generate a plot first.")
            return
        ep = self._exp_getter()
        fmt    = ep.get("fmt", "PNG").lower()
        dpi    = int(ep.get("dpi", DEFAULT_EXPORT_DPI))
        w_in   = float(ep.get("width",  DEFAULT_FIG_WIDTH_IN))
        h_in   = float(ep.get("height", DEFAULT_FIG_HEIGHT_IN))
        transp = bool(ep.get("transparent", False))
        prefix = ep.get("prefix", "correlation") or "correlation"

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
            )
            self._fig.set_size_inches(*orig_size)
            QMessageBox.information(self, "Saved", f"Figure saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
