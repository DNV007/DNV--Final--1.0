"""
DNV Scientific Module
---------------------
Role:
    Provides SEALDialog — a fully configurable analysis dialog for
    Distribution-Constrained Data Conditioning (SEAL / DNV-1.0 §1).
    Exposes physical bounds, quantile tolerances, distortion budget,
    and transform family as operator-tunable parameters; dispatches to
    seal_engine; renders a four-panel diagnostic figure; and applies a
    chosen admissible variant to the session via DataCleaningWindow.

Scientific Context:
    Implements the SEAL operator UI: operator selects columns, declares
    physical bounds, sets tolerance thresholds, and triggers the search
    for contract-feasible transformations.  The four-panel figure shows
    EVI (Envelope Violation Influence) per sample, constraint residuals
    before/after, distribution comparison across the admissible family,
    and the Pareto trade-off between distortion and admissibility.

Invariants:
    - No computation occurs at dialog construction time; only on Generate.
    - All constants are named Final with unit suffixes.
    - _set_canvas is the single site that replaces the matplotlib canvas.
    - The background _SEALWorker thread owns the SEALResult; the main
      thread only receives it via Signal and then renders.

Assumptions:
    - store.asset is a valid DataAsset with ≥ 2 numeric columns.
    - cleaning_win.apply_cleaning_operation accepts (operation, df, summary, params).
    - matplotlib backend is "Agg".

Failure Modes:
    - No numeric columns selected: Generate raises ValueError shown in dialog.
    - Fewer than MIN_SEAL_ROWS rows: engine returns is_valid=False; dialog shows
      a warning QMessageBox.
    - Worker exception: shown in QMessageBox.critical.

Provenance:
    - This module emits no transformation metadata; provenance originates
      in DataCleaningWindow.apply_cleaning_operation on Apply.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore    import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)
from PySide6.QtGui import QColor

from gui_session import SessionStore
import seal_engine as _eng

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Fixed width of the left configuration panel [px].
LEFT_PANEL_WIDTH_PX: Final[int] = 380

#: Minimum canvas height [px].
CANVAS_MIN_HEIGHT_PX: Final[int] = 520

#: Minimum canvas width [px].
CANVAS_MIN_WIDTH_PX: Final[int] = 760

#: Default figure width [inches].
DEFAULT_FIG_WIDTH_IN: Final[float] = 14.0

#: Default figure height [inches].
DEFAULT_FIG_HEIGHT_IN: Final[float] = 8.0

#: Default export DPI.
DEFAULT_EXPORT_DPI: Final[int] = 150

#: Default screen render DPI.
DEFAULT_SCREEN_DPI: Final[int] = 100

#: Minimum column label width in HRow widgets [px].
HROW_LABEL_MIN_WIDTH_PX: Final[int] = 160

#: Spacing after section separators in left panel [px].
SECTION_SPACING_PX: Final[int] = 4

#: Maximum EVI samples shown in the top-EVI bar chart [dimensionless].
EVI_TOP_N_DEFAULT: Final[int] = 30

#: KDE bandwidth multiplier for distribution overlay [dimensionless].
KDE_BW_FACTOR: Final[float] = 1.0

#: Number of points for KDE evaluation grid [dimensionless].
KDE_EVAL_POINTS: Final[int] = 256

#: Alpha for original distribution fill in comparison panel [dimensionless].
DIST_ORIG_ALPHA: Final[float] = 0.35

#: Alpha for variant KDE lines in comparison panel [dimensionless].
DIST_VAR_ALPHA: Final[float] = 0.70

#: Color for positive EVI bars (high influence) [hex string].
EVI_POS_COLOR: Final[str] = "#e74c3c"

#: Color for negative EVI bars [hex string].
EVI_NEG_COLOR: Final[str] = "#3498db"

#: Color for "before" bars in residual panel [hex string].
RESID_BEFORE_COLOR: Final[str] = "#95a5a6"

#: Color for satisfied "after" bars in residual panel [hex string].
RESID_OK_COLOR: Final[str] = "#27ae60"

#: Color for violated "after" bars in residual panel [hex string].
RESID_VIOL_COLOR: Final[str] = "#e67e22"

#: Color for the Pareto scatter primary variants [hex string].
PARETO_COLOR: Final[str] = "#3a7bd5"

#: Placeholder text for bound fields [string].
BOUND_PLACEHOLDER_TEXT: Final[str] = "none"


# ---------------------------------------------------------------------------
# Micro-helpers
# ---------------------------------------------------------------------------

def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w

def _dspin(lo: float, hi: float, val: float, step: float = 0.01,
           decimals: int = 3) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setValue(val)
    w.setSingleStep(step); w.setDecimals(decimals); return w

def _combo(items: List[str], default: str = "") -> QComboBox:
    w = QComboBox(); w.addItems(items)
    idx = items.index(default) if default in items else 0
    w.setCurrentIndex(idx); return w

def _check(state: bool, label: str = "") -> QCheckBox:
    w = QCheckBox(label); w.setChecked(state); return w

def _hrow(label: str, widget: QWidget,
          label_width: int = HROW_LABEL_MIN_WIDTH_PX) -> QWidget:
    row = QWidget(); lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label); lbl.setMinimumWidth(label_width)
    lay.addWidget(lbl); lay.addWidget(widget, 1); return row

def _eyebrow(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("ReportPanelEyebrow"); return lbl

def _section_sep() -> QFrame:
    f = QFrame(); f.setObjectName("Divider")
    f.setFixedHeight(1); f.setFrameShape(QFrame.HLine); return f

def _scrolled(inner: QWidget) -> QScrollArea:
    sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(inner)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); return sa


# ---------------------------------------------------------------------------
# _SEALWorker — background QThread
# ---------------------------------------------------------------------------

class _SEALWorker(QThread):
    """
    Runs seal_engine.compute_seal in a background thread.

    Signals
    -------
    progress(current, total)  : emitted per grid candidate evaluated
    finished_ok(SEALResult)   : emitted on success
    error_occurred(str)       : emitted on any exception
    """
    progress       = Signal(int, int)
    finished_ok    = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, df, selected_cols, bounds_list, kwargs, parent=None):
        super().__init__(parent)
        self._df           = df
        self._selected_cols = selected_cols
        self._bounds_list  = bounds_list
        self._kwargs       = dict(kwargs)

    def _cb(self, current: int, total: int) -> None:
        self.progress.emit(current, total)

    def run(self) -> None:
        try:
            kw = dict(self._kwargs)
            kw["progress_cb"] = self._cb
            result = _eng.compute_seal(
                self._df, self._selected_cols, self._bounds_list, **kw
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ---------------------------------------------------------------------------
# _BoundsTable — synced to column selection
# ---------------------------------------------------------------------------

class _BoundsTable(QTableWidget):
    """
    Two-column table (Lo bound / Hi bound) that tracks the column selection.

    Each cell is a QLineEdit with placeholder "none"; non-empty text is
    parsed to float on readback.
    """

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Column", "Lo bound", "Hi bound"])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def sync(self, cols: List[str]) -> None:
        """Rebuild rows to match cols; preserve existing bound entries."""
        existing: Dict[str, Tuple[str, str]] = {}
        for r in range(self.rowCount()):
            col_item = self.item(r, 0)
            lo_w = self.cellWidget(r, 1)
            hi_w = self.cellWidget(r, 2)
            if col_item and lo_w and hi_w:
                existing[col_item.text()] = (lo_w.text(), hi_w.text())

        self.setRowCount(len(cols))
        for r, col in enumerate(cols):
            name_item = QTableWidgetItem(col)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.setItem(r, 0, name_item)

            lo_edit = QLineEdit()
            lo_edit.setPlaceholderText(BOUND_PLACEHOLDER_TEXT)
            hi_edit = QLineEdit()
            hi_edit.setPlaceholderText(BOUND_PLACEHOLDER_TEXT)

            if col in existing:
                lo_edit.setText(existing[col][0])
                hi_edit.setText(existing[col][1])

            self.setCellWidget(r, 1, lo_edit)
            self.setCellWidget(r, 2, hi_edit)

        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def get_bounds_list(self, cols: List[str]) -> List[_eng.ColumnBoundSpec]:
        """Read current table values → list of ColumnBoundSpec."""
        result: List[_eng.ColumnBoundSpec] = []
        for r in range(self.rowCount()):
            col_item = self.item(r, 0)
            lo_w = self.cellWidget(r, 1)
            hi_w = self.cellWidget(r, 2)
            if col_item is None or lo_w is None or hi_w is None:
                continue
            col = col_item.text()
            lo: Optional[float] = None
            hi: Optional[float] = None
            try:
                t = lo_w.text().strip()
                if t and t.lower() != BOUND_PLACEHOLDER_TEXT:
                    lo = float(t)
            except ValueError:
                pass
            try:
                t = hi_w.text().strip()
                if t and t.lower() != BOUND_PLACEHOLDER_TEXT:
                    hi = float(t)
            except ValueError:
                pass
            result.append(_eng.ColumnBoundSpec(col=col, lo=lo, hi=hi))
        return result


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _rc_ctx(sp: dict) -> dict:
    ctx: dict = {"font.family": sp.get("font_family", "sans-serif")}
    if sp.get("latex", False):
        ctx["text.usetex"] = True
    return ctx


def _render_seal_figure(result: "_eng.SEALResult", op: dict, sp: dict) -> Figure:
    """
    Four-panel diagnostic figure for SEAL:

    [0,0] Top-EVI samples (horizontal bar)
    [0,1] Constraint residuals before / after best (grouped horizontal bar)
    [1,0] Distribution comparison: original vs K variants (first selected col)
    [1,1] Admissible family Pareto scatter (Ω vs d_after)
    """
    w_in = float(sp.get("width",  DEFAULT_FIG_WIDTH_IN))
    h_in = float(sp.get("height", DEFAULT_FIG_HEIGHT_IN))
    bar_alpha = float(op.get("bar_alpha", 0.80))
    tick_fs   = int(sp.get("tick_fontsize", 8))
    lbl_fs    = int(sp.get("label_fontsize", 10))
    title_fs  = int(sp.get("title_fontsize", 11))

    with plt.rc_context(_rc_ctx(sp)):
        fig, axes = plt.subplots(2, 2, figsize=(w_in, h_in))
    ax_evi, ax_res, ax_dist, ax_par = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── [0,0] Top-EVI bar chart ───────────────────────────────────────────
    top_n = int(op.get("top_n_evi", EVI_TOP_N_DEFAULT))
    evi = result.evi_scores
    if len(evi) > 0:
        abs_evi = evi.abs().nlargest(top_n)
        idx_top = abs_evi.index
        vals    = evi.loc[idx_top].values
        colors  = [EVI_POS_COLOR if v >= 0 else EVI_NEG_COLOR for v in vals]
        y_pos   = range(len(vals))
        ax_evi.barh(list(y_pos), vals, color=colors, alpha=bar_alpha, edgecolor="none")
        ax_evi.set_yticks(list(y_pos))
        ax_evi.set_yticklabels([f"#{i}" for i in idx_top], fontsize=tick_fs)
        ax_evi.invert_yaxis()
        ax_evi.axvline(0, color="#444", linewidth=0.6, linestyle="--")
        ax_evi.set_xlabel("EVI Score (δᵢ)", fontsize=lbl_fs)
    else:
        ax_evi.text(0.5, 0.5, "No EVI data.", ha="center", va="center",
                    transform=ax_evi.transAxes)
    ax_evi.set_title("Envelope Violation Influence", fontsize=title_fs, fontweight="bold")
    ax_evi.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    # ── [0,1] Constraint residuals ────────────────────────────────────────
    crs = result.constraint_residuals
    if crs:
        labels  = [f"{cr.col[:12]}\n{cr.constraint}" for cr in crs]
        before  = [cr.excess_before for cr in crs]
        after   = [cr.excess_after_best for cr in crs]
        a_colors = [RESID_OK_COLOR if a == 0.0 else RESID_VIOL_COLOR for a in after]
        y_pos   = np.arange(len(crs))
        h_bar   = 0.35
        ax_res.barh(y_pos + h_bar / 2, before, h_bar,
                    color=RESID_BEFORE_COLOR, alpha=bar_alpha, label="Before")
        ax_res.barh(y_pos - h_bar / 2, after, h_bar,
                    color=a_colors, alpha=bar_alpha, label="After best")
        ax_res.set_yticks(list(y_pos))
        ax_res.set_yticklabels(labels, fontsize=tick_fs)
        ax_res.invert_yaxis()
        ax_res.axvline(0, color="#444", linewidth=0.6, linestyle="--")
        ax_res.set_xlabel("Excess violation (r_j − θ_j)₊", fontsize=lbl_fs)
        ax_res.legend(fontsize=tick_fs, loc="lower right")
    else:
        ax_res.text(0.5, 0.5, "No constraint data.", ha="center", va="center",
                    transform=ax_res.transAxes)
    ax_res.set_title("Constraint Residuals", fontsize=title_fs, fontweight="bold")
    ax_res.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    # ── [1,0] Distribution comparison for first column ────────────────────
    col0 = result.selected_cols[0] if result.selected_cols else None
    orig_df = op.get("_orig_df")  # passed from _generate via op_cache
    if col0 is not None and result.transformed_dfs:
        from scipy.stats import gaussian_kde
        colors_var = plt.cm.plasma(np.linspace(0.2, 0.8, len(result.transformed_dfs)))

        # Gather all values for axis range
        all_series = [df_k[col0].dropna().values for df_k in result.transformed_dfs]
        if orig_df is not None and col0 in orig_df.columns:
            all_series.append(orig_df[col0].dropna().values)
        x_all = np.concatenate(all_series) if all_series else np.array([0.0, 1.0])
        x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
        margin = 0.05 * max(x_max - x_min, 1e-6)
        xs = np.linspace(x_min - margin, x_max + margin, KDE_EVAL_POINTS)

        # Original distribution
        if orig_df is not None and col0 in orig_df.columns:
            orig_vals = orig_df[col0].dropna().values
            if len(orig_vals) >= 3:
                try:
                    kde_orig = gaussian_kde(orig_vals)
                    dens_orig = kde_orig(xs)
                    ax_dist.fill_between(xs, dens_orig, alpha=DIST_ORIG_ALPHA,
                                         color="#3498db", label="Original")
                    ax_dist.plot(xs, dens_orig, color="#2980b9", linewidth=1.2)
                except Exception:
                    pass

        # Admissible family
        for k, (df_k, c_k) in enumerate(zip(result.transformed_dfs, colors_var)):
            vals_k = df_k[col0].dropna().values
            if len(vals_k) < 3:
                continue
            try:
                kde_k  = gaussian_kde(vals_k)
                dens_k = kde_k(xs)
                lw  = 1.8 if k == result.best_variant_idx else 0.9
                lbl = f"v{k}" + (" ★" if k == result.best_variant_idx else "")
                ax_dist.plot(xs, dens_k, color=tuple(c_k), linewidth=lw,
                             alpha=DIST_VAR_ALPHA, label=lbl)
            except Exception:
                pass

        ax_dist.set_xlabel(col0, fontsize=lbl_fs)
        ax_dist.set_ylabel("Density", fontsize=lbl_fs)
        if len(result.transformed_dfs) + 1 <= 8:
            ax_dist.legend(fontsize=tick_fs, loc="best")

    ax_dist.set_title(
        f"Distribution Family: {col0 or '—'}",
        fontsize=title_fs, fontweight="bold",
    )
    ax_dist.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # ── [1,1] Pareto scatter Ω vs d_after ────────────────────────────────
    if result.variants:
        omegas  = [v.distortion for v in result.variants]
        d_afts  = [v.d_after    for v in result.variants]
        var_ids = [v.variant_id for v in result.variants]
        cmap_pts = plt.cm.viridis(np.linspace(0.1, 0.9, len(result.variants)))
        for k, (om, da, vid, c_k) in enumerate(
                zip(omegas, d_afts, var_ids, cmap_pts)):
            ax_par.scatter(om, da, color=tuple(c_k), s=80,
                           zorder=5, edgecolors="#333", linewidths=0.5)
            ax_par.annotate(f"v{vid}", (om, da),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=tick_fs)
        tau_val = float(op.get("distortion_budget", _eng.SEAL_DISTORTION_BUDGET_DEFAULT))
        ax_par.axhline(tau_val, color="#e74c3c", linewidth=0.8,
                       linestyle="--", label=f"τ = {tau_val:.2f}")
        # Show original admissibility distance as baseline reference
        if result.d_before > 0.0:
            ax_par.axhline(result.d_before, color="#95a5a6", linewidth=0.8,
                           linestyle=":", label=f"d_before = {result.d_before:.3f}")
        ax_par.set_xlabel("Distortion Ω(T;X)", fontsize=lbl_fs)
        ax_par.set_ylabel("d(P_T(X), Q(θ))", fontsize=lbl_fs)
        ax_par.legend(fontsize=tick_fs)
    ax_par.set_title("Admissible Family: Ω vs Constraint Violation",
                     fontsize=title_fs, fontweight="bold")
    ax_par.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    suptitle = sp.get("suptitle", "")
    if suptitle:
        fig.suptitle(suptitle, fontsize=int(sp.get("suptitle_fontsize", 14)))
    if sp.get("tight_layout", True):
        try:
            fig.tight_layout()
        except Exception:
            pass

    return fig


# ---------------------------------------------------------------------------
# SEALDialog
# ---------------------------------------------------------------------------

class SEALDialog(QDialog):
    """
    Four-tab SEAL analysis dialog.

    Left panel : QTabWidget — Data / Constraints / Style / Export
    Right panel : Generate button, progress bar, FigureCanvas, Apply row.
    """

    def __init__(self, store: SessionStore, cleaning_win, parent=None):
        super().__init__(parent)
        self._store       = store
        self._cleaning_win = cleaning_win
        self._result: Optional[_eng.SEALResult] = None
        self._worker: Optional[_SEALWorker]      = None
        self._fig:    Optional[Figure]            = None
        self._canvas: Optional[FigureCanvas]      = None
        self._op_cache: dict = {}
        self._sp_cache: dict = {}

        self.setWindowTitle("SEAL — Distribution-Constrained Data Conditioning")
        self.setMinimumSize(
            LEFT_PANEL_WIDTH_PX + CANVAS_MIN_WIDTH_PX + 24,
            CANVAS_MIN_HEIGHT_PX + 80,
        )

        if store.asset is None:
            lay = QVBoxLayout(self)
            lay.addWidget(QLabel("No active dataset."))
            return

        self._df = store.asset.dataframe
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Left panel ────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setMinimumWidth(LEFT_PANEL_WIDTH_PX - 40)
        self._tabs.setMaximumWidth(LEFT_PANEL_WIDTH_PX + 60)

        self._data_w, self._col_list, self._bounds_tbl = self._make_data_tab()
        self._con_w,  self._con_controls               = self._make_constraints_tab()
        self._sty_w,  self._sty_getter                  = self._make_style_tab()
        self._exp_w                                     = self._make_export_tab()

        self._tabs.addTab(_scrolled(self._data_w),  "Data")
        self._tabs.addTab(_scrolled(self._con_w),   "Constraints")
        self._tabs.addTab(_scrolled(self._sty_w),   "Style")
        self._tabs.addTab(self._exp_w,              "Export")

        root.addWidget(self._tabs)

        # ── Right panel ───────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        self._gen_btn = QPushButton("Generate")
        self._gen_btn.clicked.connect(self._generate)
        right.addWidget(self._gen_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 49)
        self._progress_bar.setVisible(False)
        right.addWidget(self._progress_bar)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("SectionHint")
        self._status_lbl.setVisible(False)
        right.addWidget(self._status_lbl)

        self._canvas_holder = QWidget()
        self._canvas_holder.setMinimumSize(CANVAS_MIN_WIDTH_PX, CANVAS_MIN_HEIGHT_PX)
        self._canvas_lay = QVBoxLayout(self._canvas_holder)
        self._canvas_lay.setContentsMargins(0, 0, 0, 0)
        ph = QLabel("Configure parameters and click Generate.")
        ph.setObjectName("SectionHint")
        ph.setAlignment(Qt.AlignCenter)
        self._canvas_lay.addWidget(ph)
        right.addWidget(self._canvas_holder, 1)

        # Apply row
        apply_row = QHBoxLayout()
        apply_row.addWidget(QLabel("Apply variant:"))
        self._variant_combo = QComboBox()
        self._variant_combo.setMinimumWidth(160)
        self._variant_combo.setEnabled(False)
        apply_row.addWidget(self._variant_combo)
        self._apply_btn = QPushButton("Apply to Dataset")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_variant)
        apply_row.addWidget(self._apply_btn)
        apply_row.addStretch()
        right.addLayout(apply_row)

        root.addLayout(right, 1)

    # ── Tab builders ──────────────────────────────────────────────────────

    def _make_data_tab(self) -> Tuple[QWidget, QListWidget, _BoundsTable]:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        lay.addWidget(_eyebrow("COLUMN SELECTION"))
        hint = QLabel("Select numeric columns to condition.")
        hint.setObjectName("SectionHint"); hint.setWordWrap(True)
        lay.addWidget(hint)

        col_list = QListWidget()
        col_list.setSelectionMode(QListWidget.MultiSelection)
        numeric_cols = [c for c in self._df.columns
                        if pd.api.types.is_numeric_dtype(self._df[c])]
        for c in numeric_cols:
            item = QListWidgetItem(c)
            col_list.addItem(item)
        col_list.setMinimumHeight(120)
        lay.addWidget(col_list)

        lay.addWidget(_section_sep())
        lay.addWidget(_eyebrow("PHYSICAL BOUNDS"))
        hint2 = QLabel(
            "Declare physical admissibility bounds per column. "
            "Leave blank for unconstrained. Values in the same units as the data."
        )
        hint2.setObjectName("SectionHint"); hint2.setWordWrap(True)
        lay.addWidget(hint2)

        bounds_tbl = _BoundsTable()
        bounds_tbl.setMinimumHeight(160)
        lay.addWidget(bounds_tbl)

        def _sync_bounds() -> None:
            selected = [col_list.item(i).text()
                        for i in range(col_list.count())
                        if col_list.item(i).isSelected()]
            bounds_tbl.sync(selected)

        col_list.itemSelectionChanged.connect(_sync_bounds)

        # Select all by default
        col_list.selectAll()
        _sync_bounds()

        lay.addStretch()
        return w, col_list, bounds_tbl

    def _make_constraints_tab(self) -> Tuple[QWidget, dict]:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
        controls: Dict[str, Any] = {}

        def _add(label: str, widget, key: str) -> None:
            controls[key] = widget; lay.addWidget(_hrow(label, widget))

        def _sec(text: str) -> None:
            lay.addWidget(_section_sep())
            lay.addWidget(_eyebrow(text))
            lay.addSpacing(SECTION_SPACING_PX)

        lay.addWidget(_eyebrow("ADMISSIBILITY TOLERANCES"))

        _add("Tail mass tolerance θ_tail:",
             _dspin(0.00, 0.50, _eng.SEAL_TAIL_TOL_DEFAULT, 0.01, 3),
             "tail_tol")
        _add("Quantile dev tol θ_q (IQR):",
             _dspin(0.00, 2.00, _eng.SEAL_Q_TOL_IQR_DEFAULT, 0.05, 3),
             "q_tol_iqr")
        _add("Distortion budget τ:",
             _dspin(0.01, 5.00, _eng.SEAL_DISTORTION_BUDGET_DEFAULT, 0.05, 3),
             "distortion_budget")

        _sec("TRANSFORM FAMILY")

        tf_combo = _combo(["winsorize", "soft_clamp"], "winsorize")
        controls["transform_type"] = tf_combo
        lay.addWidget(_hrow("Transform type:", tf_combo))

        sm_spin = _dspin(0.01, 0.50, _eng.SEAL_SOFT_SMOOTHNESS_DEFAULT, 0.01, 3)
        sm_spin.setEnabled(False)
        controls["soft_smoothness"] = sm_spin
        lay.addWidget(_hrow("Soft smoothness:", sm_spin))

        def _toggle_smooth(idx: int) -> None:
            sm_spin.setEnabled(tf_combo.currentText() == "soft_clamp")
        tf_combo.currentIndexChanged.connect(_toggle_smooth)

        _sec("SEARCH PARAMETERS")

        _add("N variants K:",    _spin(1, 20, _eng.SEAL_N_VARIANTS_DEFAULT), "n_variants")
        _add("Random seed:",     _spin(0, 9999, 42),                          "rs")

        _sec("FIGURE OPTIONS")
        _add("Top EVI samples:", _spin(5, 200, EVI_TOP_N_DEFAULT), "top_n_evi")

        lay.addStretch()
        return w, controls

    def _make_style_tab(self) -> Tuple[QWidget, Callable[[], dict]]:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
        controls: Dict[str, Any] = {}

        def _add(label: str, widget, key: str) -> None:
            controls[key] = widget; lay.addWidget(_hrow(label, widget))

        def _sec(text: str) -> None:
            lay.addWidget(_section_sep()); lay.addWidget(_eyebrow(text))

        lay.addWidget(_eyebrow("LABELS"))
        suptitle_edit = QLineEdit(); suptitle_edit.setPlaceholderText("none")
        _add("Figure suptitle:", suptitle_edit, "suptitle")

        _sec("TYPOGRAPHY")
        _add("Font family:", _combo(["sans-serif","serif","monospace",
                                      "Segoe UI","Arial","DejaVu Sans"], "sans-serif"),
             "font_family")
        _add("Title fontsize:",  _spin(6, 24, 11), "title_fontsize")
        _add("Suptitle size:",   _spin(6, 28, 14), "suptitle_fontsize")
        _add("Axis label size:", _spin(6, 22, 10), "label_fontsize")
        _add("Tick label size:", _spin(5, 18, 8),  "tick_fontsize")

        _sec("BARS & ALPHA")
        _add("Bar alpha:", _dspin(0.2, 1.0, 0.80, 0.05, 2), "bar_alpha")

        _sec("FIGURE SIZE")
        _add("Width [in]:",  _dspin(6.0, 30.0, DEFAULT_FIG_WIDTH_IN,  0.5, 1), "width")
        _add("Height [in]:", _dspin(4.0, 20.0, DEFAULT_FIG_HEIGHT_IN, 0.5, 1), "height")

        _sec("LAYOUT")
        _add("Tight layout:", _check(True), "tight_layout")

        _sec("EXPORT")
        _add("Export DPI:", _spin(72, 600, DEFAULT_EXPORT_DPI), "export_dpi")

        lay.addStretch()

        def _getter() -> dict:
            out: dict = {}
            for k, widget in controls.items():
                if isinstance(widget, QComboBox):
                    out[k] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    out[k] = widget.isChecked()
                elif isinstance(widget, QSpinBox):
                    out[k] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    out[k] = widget.value()
                elif isinstance(widget, QLineEdit):
                    out[k] = widget.text().strip()
            return out

        return w, _getter

    def _make_export_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(8)

        lay.addWidget(_eyebrow("EXPORT FIGURE"))

        out_row = QHBoxLayout()
        self._out_dir = QLineEdit(); self._out_dir.setPlaceholderText("Output folder")
        out_row.addWidget(self._out_dir, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        out_row.addWidget(browse_btn)
        lay.addLayout(out_row)

        self._out_stem = QLineEdit(); self._out_stem.setPlaceholderText("seal_result")
        lay.addWidget(_hrow("Filename stem:", self._out_stem))

        exp_btn = QPushButton("Export PNG")
        exp_btn.clicked.connect(self._export_figure)
        lay.addWidget(exp_btn, 0, Qt.AlignLeft)
        lay.addStretch()
        return w

    # ── Canvas management ─────────────────────────────────────────────────

    def _set_canvas(self, fig: Figure) -> None:
        if self._canvas is not None:
            self._canvas_lay.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        if self._fig is not None:
            plt.close(self._fig)

        self._fig    = fig
        self._canvas = FigureCanvas(fig)
        self._canvas.setMinimumSize(CANVAS_MIN_WIDTH_PX, CANVAS_MIN_HEIGHT_PX)
        fig.set_dpi(DEFAULT_SCREEN_DPI)
        self._canvas_lay.addWidget(self._canvas)
        self._canvas.draw()

    # ── Column / constraint readers ───────────────────────────────────────

    def _selected_cols(self) -> List[str]:
        return [self._col_list.item(i).text()
                for i in range(self._col_list.count())
                if self._col_list.item(i).isSelected()]

    def _read_con(self) -> dict:
        out: dict = {}
        for k, w in self._con_controls.items():
            if isinstance(w, QComboBox):
                out[k] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[k] = w.isChecked()
            elif isinstance(w, QSpinBox):
                out[k] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                out[k] = w.value()
        return out

    # ── Generate ──────────────────────────────────────────────────────────

    def _generate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("Computing…")
        async_launched = False
        try:
            cols = self._selected_cols()
            if not cols:
                raise ValueError("Select at least one column in the Data tab.")

            bounds_list = self._bounds_tbl.get_bounds_list(cols)
            con = self._read_con()
            sp  = self._sty_getter()

            kwargs = {
                "tail_tol":          float(con.get("tail_tol",          _eng.SEAL_TAIL_TOL_DEFAULT)),
                "q_tol_iqr":         float(con.get("q_tol_iqr",         _eng.SEAL_Q_TOL_IQR_DEFAULT)),
                "distortion_budget": float(con.get("distortion_budget",  _eng.SEAL_DISTORTION_BUDGET_DEFAULT)),
                "transform_type":    str(con.get("transform_type",       "winsorize")),
                "soft_smoothness":   float(con.get("soft_smoothness",    _eng.SEAL_SOFT_SMOOTHNESS_DEFAULT)),
                "n_variants":        int(con.get("n_variants",           _eng.SEAL_N_VARIANTS_DEFAULT)),
                "rs":                int(con.get("rs",                   42)),
            }

            self._op_cache = dict(con)
            self._op_cache["distortion_budget"] = kwargs["distortion_budget"]
            self._op_cache["_orig_df"]          = self._df
            self._sp_cache = sp

            worker = _SEALWorker(self._df, cols, bounds_list, kwargs, self)
            worker.progress.connect(self._on_progress)
            worker.finished_ok.connect(self._on_finished)
            worker.error_occurred.connect(self._on_error)
            self._worker = worker

            self._progress_bar.setRange(0, 49)
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(True)
            self._status_lbl.setText("Evaluating grid candidates…")
            self._status_lbl.setVisible(True)

            worker.start()
            async_launched = True

        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
        except Exception as exc:
            log.exception("SEALDialog._generate")
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            if not async_launched:
                self._gen_btn.setEnabled(True)
                self._gen_btn.setText("Generate")

    # ── Worker callbacks ──────────────────────────────────────────────────

    def _on_progress(self, current: int, total: int) -> None:
        self._progress_bar.setValue(current)
        self._status_lbl.setText(f"Candidate {current} / {total}")

    def _on_finished(self, result: "_eng.SEALResult") -> None:
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText("Generate")
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        self._result = result

        if not result.is_valid:
            QMessageBox.warning(self, "SEAL Warning", result.message)
            return

        # Populate variant combo
        self._variant_combo.clear()
        for v in result.variants:
            self._variant_combo.addItem(
                f"v{v.variant_id}  Ω={v.distortion:.4f}  d={v.d_after:.4f}"
            )
        best_idx = result.best_variant_idx
        self._variant_combo.setCurrentIndex(best_idx)
        self._variant_combo.setEnabled(True)
        self._apply_btn.setEnabled(True)

        try:
            fig = _render_seal_figure(result, self._op_cache, self._sp_cache)
            self._set_canvas(fig)
        except Exception as exc:
            log.exception("SEALDialog._on_finished render")
            QMessageBox.critical(self, "Render Error", str(exc))

    def _on_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_lbl.setVisible(False)
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText("Generate")
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        QMessageBox.critical(self, "SEAL Error", msg)

    # ── Apply variant ─────────────────────────────────────────────────────

    def _apply_variant(self) -> None:
        if self._result is None:
            return
        k = self._variant_combo.currentIndex()
        if k < 0 or k >= len(self._result.transformed_dfs):
            QMessageBox.warning(self, "Apply Error", "No variant selected."); return

        variant  = self._result.variants[k]
        df_new   = self._result.transformed_dfs[k]

        params = {
            "variant_id":        variant.variant_id,
            "distortion":        round(variant.distortion, 6),
            "d_after":           round(variant.d_after, 6),
            "n_cols":            len(self._result.selected_cols),
            "selected_cols":     list(self._result.selected_cols),
            "transform_type":    self._op_cache.get("transform_type", "winsorize"),
            "transform_params":  variant.transform_params,
            # Full SEAL hyperparameters for epistemic transparency (DNV-1.0 §5)
            "tail_tol":          self._op_cache.get("tail_tol"),
            "q_tol_iqr":         self._op_cache.get("q_tol_iqr"),
            "distortion_budget": self._op_cache.get("distortion_budget"),
            "soft_smoothness":   self._op_cache.get("soft_smoothness"),
            "n_variants":        self._op_cache.get("n_variants"),
            "rs":                self._op_cache.get("rs"),
        }
        summary = (
            f"SEAL conditioning: variant v{variant.variant_id}  "
            f"Ω={variant.distortion:.4f}  d_after={variant.d_after:.4f}  "
            f"cols={list(self._result.selected_cols)}"
        )

        self._cleaning_win.apply_cleaning_operation(
            "seal_conditioning", df_new, summary, params
        )
        self._store.set_seal_result(self._result)

    # ── Export ────────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._out_dir.setText(folder)

    def _export_figure(self) -> None:
        if self._fig is None:
            QMessageBox.information(self, "No figure", "Generate first."); return
        folder = self._out_dir.text().strip() or "."
        stem   = self._out_stem.text().strip() or "seal_result"
        sp     = self._sty_getter()
        dpi    = int(sp.get("export_dpi", DEFAULT_EXPORT_DPI))
        path   = f"{folder}/{stem}.png"
        try:
            self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
            QMessageBox.information(self, "Exported", f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
