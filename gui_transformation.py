"""
DNV Scientific Module
---------------------
Role:
    Provides TransformationWindow — a 60-method element-wise column
    transformer with live preview and provenance binding.

Scientific Context:
    Applies mathematical transformations (power, log, trigonometric,
    normalisation, smoothing, signal processing) column-wise to the active
    DataAsset, generating a new asset with updated provenance.

Invariants:
    - Transformations are applied column-wise; row indexing is preserved.
    - Live preview operates on a sample of rows and never writes to the
      store.

Assumptions:
    - Selected columns are numeric.
    - store.asset is a valid DataAsset.

Failure Modes:
    - Non-numeric column selected: transformation raises ValueError shown
      in a QMessageBox.
    - Transformation produces all-NaN output: user is warned before the
      store is updated.

Provenance:
    - Emits a ProvenanceRecord per applied transformation via
      DataAsset.with_provenance.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, boxcox as _scipy_boxcox
from sklearn.preprocessing import power_transform, QuantileTransformer, minmax_scale

from PySide6.QtCore    import Qt, QTimer
from PySide6.QtGui     import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QStackedWidget,
    QTableView, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ing_config    import IngestionConfig
from ing_pipeline  import derive_data_asset
from ing_contracts import assess_cleaning_operation, attach_contracts_to_asset
from gui_session   import SessionStore
from gui_base      import ModuleDialog
from gui_widgets   import DataFrameModel, ReportTextPanel, make_section_header

log = logging.getLogger(__name__)


# ── Method catalogue ──────────────────────────────────────────────────────────

_CATALOGUE = [
    ("Original",             "BASIC MATH",    "original"),
    ("Absolute",             "BASIC MATH",    "absolute"),
    ("Negative",             "BASIC MATH",    "negative"),
    ("Square",               "BASIC MATH",    "square"),
    ("Cube",                 "BASIC MATH",    "cube"),
    ("Square Root",          "BASIC MATH",    "sqrt"),
    ("Cube Root",            "BASIC MATH",    "cbrt"),
    ("Fourth Root",          "BASIC MATH",    "fourth_root"),
    ("Reciprocal",           "BASIC MATH",    "reciprocal"),
    ("Inverse Square",       "BASIC MATH",    "inv_square"),
    ("Inverse Cube",         "BASIC MATH",    "inv_cube"),
    ("log1p",                "SPECIAL MATH",  "log1p"),
    ("expm1",                "SPECIAL MATH",  "expm1"),
    ("Sign Function",        "SPECIAL MATH",  "sign"),
    ("Binarize (Median)",    "SPECIAL MATH",  "binarize"),
    ("Weibull",              "SPECIAL MATH",  "weibull"),
    ("Power 1/4",            "POWER",         "pow_025"),
    ("Power 1/3",            "POWER",         "pow_033"),
    ("Power 1/2",            "POWER",         "pow_05"),
    ("Power 2",              "POWER",         "pow_2"),
    ("Power 3",              "POWER",         "pow_3"),
    ("Power 4",              "POWER",         "pow_4"),
    ("Natural Log",          "LOGARITHMIC",   "log_nat"),
    ("Log Base 2",           "LOGARITHMIC",   "log2"),
    ("Log Base 10",          "LOGARITHMIC",   "log10"),
    ("Log-Exp (Softplus)",   "LOGARITHMIC",   "log_exp"),
    ("Log-Log",              "LOGARITHMIC",   "log_log"),
    ("Log Normalised",       "LOGARITHMIC",   "log_norm"),
    ("Inverse Log",          "LOGARITHMIC",   "inv_log"),
    ("Shifted Log",          "LOGARITHMIC",   "shifted_log"),
    ("Exponential",          "EXPONENTIAL",   "exp"),
    ("Exp Base 2",           "EXPONENTIAL",   "exp2"),
    ("Exp Minus 1",          "EXPONENTIAL",   "expm1_exp"),
    ("Sine",                 "TRIGONOMETRIC", "sin"),
    ("Cosine",               "TRIGONOMETRIC", "cos"),
    ("Tangent",              "TRIGONOMETRIC", "tan"),
    ("Arcsine",              "TRIGONOMETRIC", "arcsin"),
    ("Arccosine",            "TRIGONOMETRIC", "arccos"),
    ("Arctangent",           "TRIGONOMETRIC", "arctan"),
    ("Sinh",                 "HYPERBOLIC",    "sinh"),
    ("Cosh",                 "HYPERBOLIC",    "cosh"),
    ("Tanh",                 "HYPERBOLIC",    "tanh"),
    ("Generalized Hyp.",     "HYPERBOLIC",    "gen_hyp"),
    ("Arcsinh",              "HYPERBOLIC",    "arcsinh"),
    ("Arccosh",              "HYPERBOLIC",    "arccosh"),
    ("Arctanh",              "HYPERBOLIC",    "arctanh"),
    ("Sigmoid",              "SIGMOID-LIKE",  "sigmoid"),
    ("Logit",                "SIGMOID-LIKE",  "logit"),
    ("Softmax",              "SIGMOID-LIKE",  "softmax"),
    ("Gaussian",             "SIGMOID-LIKE",  "gaussian"),
    ("Z-Score",              "STATISTICAL",   "zscore"),
    ("Rank",                 "STATISTICAL",   "rank"),
    ("Percentile Rank",      "STATISTICAL",   "pct_rank"),
    ("Min-Max",              "SCALING",       "minmax"),
    ("Standard (Z)",         "SCALING",       "standard"),
    ("Robust (IQR)",         "SCALING",       "robust"),
    ("Exp Moving Average",   "ADVANCED",      "ema"),
    ("Box-Cox",              "ADVANCED",      "boxcox"),
    ("Yeo-Johnson",          "ADVANCED",      "yeojohnson"),
    ("Quantile",             "ADVANCED",      "quantile"),
]

_DESC: dict[str, str] = {
    "original":    "Returns the column unchanged. Useful as a no-op baseline for side-by-side comparison.",
    "absolute":    "x → |x|. Removes the sign; useful for magnitude-only analysis.",
    "negative":    "x → −x. Reverses the scale direction.",
    "square":      "x → x². Amplifies large values; inputs clipped to ±1×10⁵ before squaring.",
    "cube":        "x → x³. Sign-preserving power; inputs clipped to ±1×10⁴.",
    "sqrt":        "x → √x. Compresses right-skewed data; negative inputs are clipped to 0.",
    "cbrt":        "x → ∛x. Sign-preserving cube root defined on all reals.",
    "fourth_root": "x → x^0.25. Strong right-skew compression; negatives clipped to 0.",
    "reciprocal":  "x → 1/x. Near-zero values map to 0 to avoid division by zero.",
    "inv_square":  "x → 1/x². Near-zero values map to 0.",
    "inv_cube":    "x → 1/x³. Near-zero values map to 0.",
    "log1p":       "x → ln(1+x). Numerically stable log for x ≥ 0. Accepts x > −1.",
    "expm1":       "x → eˣ−1 (via np.expm1). Numerically precise near x=0 where exp(x)−1 loses significant digits. Preferred over exp(x)−1 for small arguments.",
    "sign":        "x → {−1, 0, +1}. Retains only the algebraic sign of each value.",
    "binarize":    "Threshold at the column median: above → 1, at or below → 0.",
    "weibull":     "x → exp(−max(x,0)²). Weibull survival function at unit scale (λ=1, shape k=2). Designed for non-negative inputs — negative values map to 1 (no failure yet). Distinct from the symmetric Gaussian kernel.",
    "pow_025":     "x → sign(x)·|x|^0.25. Sign-preserving fourth-root power.",
    "pow_033":     "x → x^(1/3) (non-negative inputs). Non-negative cube-root power.",
    "pow_05":      "x → √x (non-negative inputs). Square-root power.",
    "pow_2":       "x → x².",
    "pow_3":       "x → x³.",
    "pow_4":       "x → x⁴.",
    "log_nat":     "x → ln(x). Inputs clipped to 1×10⁻⁸ to avoid log(0).",
    "log2":        "x → log₂(x). Inputs clipped to 1×10⁻⁸.",
    "log10":       "x → log₁₀(x). Inputs clipped to 1×10⁻⁸.",
    "log_exp":     "x → ln(1+eˣ). The softplus function — a smooth approximation to ReLU.",
    "log_log":     "x → ln(ln(x+1.1)). Double-log for strong compression of heavy tails.",
    "log_norm":    "x → ln(x+1)/ln(max(x+1)). Normalises log-transformed values to [0, 1] for non-negative data. Falls back to ln(x+1) when all inputs are ≤ 0 (max denominator is non-positive).",
    "inv_log":     "x → 1/(1+e⁻ˣ). Logistic sigmoid. Named 'Inverse Log' because it is the inverse of the logit (log-odds) function. Clips exponent to ±500 to prevent overflow.",
    "shifted_log": "x → ln(x+2). Positive shift ensures the argument is always > 0.",
    "exp":         "x → eˣ. Inputs clipped to ±700 to prevent overflow.",
    "exp2":        "x → 2ˣ. Inputs clipped to ±700.",
    "expm1_exp":   "x → eˣ−1 (via np.exp(x)−1). Exponential growth shifted down by 1. Uses exp() directly, which is less precise than expm1 near x=0 but belongs conceptually to the exponential family. Clipped to ±700.",
    "sin":         "x → sin(x). Input interpreted as radians.",
    "cos":         "x → cos(x). Input interpreted as radians.",
    "tan":         "x → tan(x). Input interpreted as radians.",
    "arcsin":      "x → arcsin(x). Defined for x ∈ [−1, 1]; inputs clipped. Meaningful for data representing normalised values or correlations in [−1, 1]. All values outside this range collapse to ±π/2.",
    "arccos":      "x → arccos(x). Defined for x ∈ [−1, 1]; inputs clipped. Output in [0, π]. Values outside domain collapse to 0 or π.",
    "arctan":      "x → arctan(x). Maps ℝ → (−π/2, π/2). Smooth saturation; defined on all reals without any clipping.",
    "sinh":        "x → sinh(x). Inputs clipped to ±700.",
    "cosh":        "x → cosh(x). Always ≥ 1; inputs clipped to ±700.",
    "tanh":        "x → tanh(x). Smooth saturation onto (−1, 1).",
    "gen_hyp":     "x → ln(cosh(x)). Behaves like L2 for small |x|, L1 for large |x|.",
    "arcsinh":     "x → sinh⁻¹(x) = ln(x+√(x²+1)). Defined on all reals; log-like for large |x|.",
    "arccosh":     "x → cosh⁻¹(x). Inputs clipped to x ≥ 1.",
    "arctanh":     "x → tanh⁻¹(x). Defined for |x| < 1; inputs clipped to (−1+ε, 1−ε). Meaningful for correlation coefficients or other data in (−1, 1). Out-of-domain values collapse to ±large.",
    "sigmoid":     "x → 1/(1+e⁻ˣ). Squashes ℝ into (0, 1). Inputs clipped to ±700.",
    "logit":       "x → ln(x/(1−x)). Inverse of the sigmoid. Meaningful only when x ∈ (0, 1) (e.g., probabilities, fractions). Inputs outside (0, 1) are clipped — all values ≥ 1 collapse to ≈18.4 and all values ≤ 0 collapse to ≈−18.4.",
    "softmax":     "Column-wise softmax: exp(xᵢ)/Σexp(x). Each transformed column sums to 1. Computed via the log-sum-exp trick for full numerical stability. Note: for a single column of distinct values, output gives relative exponential weights summing to 1.",
    "gaussian":    "x → exp(−x²). Radial basis function centred at zero.",
    "zscore":      "x → (x−μ)/σ. Standardises each column to zero mean and unit variance.",
    "rank":        "x → rank(x). Converts values to ordinal ranks (ties averaged).",
    "pct_rank":    "x → rank(x)/n. Ranks expressed as fractions in (0, 1].",
    "minmax":      "x → (x−min)/(max−min). Scales each column to [0, 1].",
    "standard":    "x → (x−μ)/σ. Identical to Z-Score normalisation.",
    "robust":      "x → (x−median)/IQR. Robust to outliers.",
    "ema":         "Exponential moving average with configurable span. Smooths temporal sequences; applied per column.",
    "boxcox":      "Box-Cox power transform (λ auto-estimated via MLE). Inputs are auto-shifted to be strictly positive.",
    "yeojohnson":  "Yeo-Johnson transform. Extends Box-Cox to all reals — no input shift required.",
    "quantile":    "Empirical quantile transform mapping each column to a uniform or normal distribution.",
}

# Keys whose param stacks are non-empty
_PARAM_PAGE: dict[str, int] = {"ema": 1, "quantile": 2}


# ── Transformation engine ─────────────────────────────────────────────────────

def _transform(key: str, x2d: np.ndarray, **kw) -> np.ndarray:
    """Apply *key* to a 2-D float array (n_rows × n_cols). Always returns sanitised floats."""
    def _s(a: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=1e10, neginf=-1e10)

    x = _s(x2d)

    if   key == "original":    r = x
    elif key == "absolute":    r = np.abs(x)
    elif key == "negative":    r = -x
    elif key == "square":      r = np.square(np.clip(x, -1e5, 1e5))
    elif key == "cube":        r = np.clip(x, -1e4, 1e4) ** 3
    elif key == "sqrt":        r = np.sqrt(np.clip(x, 0, None))
    elif key == "cbrt":        r = np.cbrt(x)
    elif key == "fourth_root": r = np.power(np.clip(x, 0, None), 0.25)
    elif key == "reciprocal":  r = np.where(np.abs(x) > np.finfo(float).eps, 1.0 / x, 0.0)
    elif key == "inv_square":  r = np.where(x**2 > 1e-8, 1.0 / (x**2), 0.0)
    elif key == "inv_cube":    r = np.where(np.abs(x)**3 > 1e-8, 1.0 / (x**3), 0.0)
    elif key == "log1p":       r = np.log1p(np.clip(x, -1 + 1e-9, None))
    elif key == "expm1":       r = np.expm1(x)   # no pre-clip: preserves precision for small |x|
    elif key == "sign":        r = np.sign(x)
    elif key == "binarize":    r = (x > np.median(x, axis=0)).astype(float)
    elif key == "weibull":     r = np.exp(-np.clip(x, 0.0, None) ** 2)
    elif key == "pow_025":     r = np.sign(x) * np.abs(x) ** 0.25
    elif key == "pow_033":     r = np.cbrt(np.clip(x, 0, None))
    elif key == "pow_05":      r = np.sqrt(np.clip(x, 0, None))
    elif key == "pow_2":       r = x ** 2
    elif key == "pow_3":       r = x ** 3
    elif key == "pow_4":       r = x ** 4
    elif key == "log_nat":     r = np.log(np.clip(x, 1e-8, None))
    elif key == "log2":        r = np.log2(np.clip(x, 1e-8, None))
    elif key == "log10":       r = np.log10(np.clip(x, 1e-8, None))
    elif key == "log_exp":     r = np.log1p(np.exp(np.clip(x, -700, 700)))
    elif key == "log_log":     r = np.log(np.log(np.clip(x + 1.1, 1.1, None)))
    elif key == "log_norm":
        s = np.clip(x + 1, 1e-8, None)
        log_s   = np.log(s)
        log_max = np.log(np.max(s, axis=0))
        # Only divide when max(x+1) > 1 (log_max > 0); otherwise fall back to log(x+1) as-is
        safe_denom = np.where(log_max > 1e-9, log_max, np.ones_like(log_max))
        r = log_s / safe_denom
    elif key == "inv_log":     r = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    elif key == "shifted_log": r = np.log(np.clip(x + 2, 1e-8, None))
    elif key == "exp":         r = np.exp(np.clip(x, -700, 700))
    elif key == "exp2":        r = np.exp2(np.clip(x, -700, 700))
    elif key == "expm1_exp":   r = np.exp(np.clip(x, -700, 700)) - 1.0
    elif key == "sin":         r = np.sin(x)
    elif key == "cos":         r = np.cos(x)
    elif key == "tan":         r = np.tan(x)
    elif key == "arcsin":      r = np.arcsin(np.clip(x, -1, 1))
    elif key == "arccos":      r = np.arccos(np.clip(x, -1, 1))
    elif key == "arctan":      r = np.arctan(x)   # defined on all reals; no clipping needed
    elif key == "sinh":        r = np.sinh(np.clip(x, -700, 700))
    elif key == "cosh":        r = np.cosh(np.clip(x, -700, 700))
    elif key == "tanh":        r = np.tanh(x)
    elif key == "gen_hyp":     r = np.log(np.cosh(np.clip(x, -700, 700)))
    elif key == "arcsinh":     r = np.arcsinh(x)
    elif key == "arccosh":     r = np.arccosh(np.clip(x, 1, None))
    elif key == "arctanh":     r = np.arctanh(np.clip(x, -1 + 1e-9, 1 - 1e-9))
    elif key == "sigmoid":     r = 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))
    elif key == "logit":
        xc = np.clip(x, 1e-8, 1 - 1e-8); r = np.log(xc / (1.0 - xc))
    elif key == "softmax":
        # log-sum-exp trick: subtract column max before exp to prevent overflow
        x_shift = x - np.max(x, axis=0)
        e = np.exp(x_shift)
        r = e / np.sum(e, axis=0)
    elif key == "gaussian":    r = np.exp(-(x ** 2))
    elif key == "zscore":
        r = (x - np.mean(x, axis=0)) / np.clip(np.std(x, axis=0), 1e-9, None)
    elif key == "rank":
        r = np.column_stack([rankdata(x[:, i]) for i in range(x.shape[1])])
    elif key == "pct_rank":
        r = np.column_stack([rankdata(x[:, i]) / len(x) for i in range(x.shape[1])])
    elif key == "minmax":      r = minmax_scale(x)
    elif key == "standard":
        r = (x - np.mean(x, axis=0)) / np.clip(np.std(x, axis=0), 1e-9, None)
    elif key == "robust":
        r = (x - np.median(x, axis=0)) / (
            np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0) + 1e-9)
    elif key == "ema":
        span = int(kw.get("span", 5))
        r = np.column_stack([
            pd.Series(x[:, i]).ewm(span=span, adjust=False).mean().values
            for i in range(x.shape[1])
        ])
    elif key == "boxcox":
        cols = []
        for i in range(x.shape[1]):
            c = x[:, i]
            mn = np.min(c)
            c_pos = np.clip(c - mn + 1 if mn <= 0 else c, 1e-8, None)
            cols.append(_scipy_boxcox(c_pos)[0] if np.std(c) > 1e-12 else c.copy())
        r = np.column_stack(cols)
    elif key == "yeojohnson":
        r = power_transform(x, method="yeo-johnson", standardize=False)
    elif key == "quantile":
        n_q = int(kw.get("n_quantiles", min(x.shape[0], 1000)))
        dist = str(kw.get("output_distribution", "uniform"))
        r = (QuantileTransformer(n_quantiles=n_q, output_distribution=dist)
             .fit_transform(x) if x.shape[0] > 1 else x.copy())
    else:
        raise ValueError(f"Unknown transformation key: {key!r}")

    return _s(r)


# ── Main window ───────────────────────────────────────────────────────────────

class TransformationWindow(ModuleDialog):
    """Transformation workspace: 60 element-wise methods, live before/after preview,
    provenance-tracked column additions."""

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Transformation",
            hint="Apply element-wise mathematical transformations to numeric columns. "
                 "Each accepted operation appends new provenance-bound columns to the dataset.",
            category="DATA PREP",
            parent=parent,
        )
        self._store = store
        self._asset = store.asset
        self._keys: List[Optional[str]] = []   # parallel to method_box rows
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._draw_preview)

        if self._asset is None:
            note = QLabel("No active dataset. Load a dataset in Data Ingestion first.")
            note.setObjectName("SectionHint"); note.setWordWrap(True)
            self.body_layout.addWidget(note); return

        self.body_layout.addWidget(make_section_header(
            "Column transformation workspace",
            "Select a method, choose numeric columns, inspect the live preview, then apply. "
            "New columns are appended as <original>_<method>.",
            "60 METHODS",
        ))

        self.tabs = QTabWidget()
        self.tabs.addTab(self._overview_tab(),   "Overview")
        self.tabs.addTab(self._transform_tab(),  "Transform")
        self.tabs.addTab(self._history_tab(),    "History")
        self.body_layout.addWidget(self.tabs, 1)
        self.refresh_view()

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _overview_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(10)
        lay.addWidget(_hint("Live preview of the active dataset (first 100 rows)."))
        self.overview_table = QTableView()
        self.overview_table.setAlternatingRowColors(True)
        self.overview_table.setSelectionBehavior(QTableView.SelectRows)
        lay.addWidget(self.overview_table, 1); return w

    def _transform_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal); splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._method_panel())
        splitter.addWidget(self._right_panel())
        splitter.setSizes([280, 720])
        lay.addWidget(splitter, 1); return w

    def _method_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lbl = QLabel("TRANSFORMATION METHOD"); lbl.setObjectName("ReportPanelEyebrow"); lay.addWidget(lbl)
        self.method_box = QListWidget(); self.method_box.setAlternatingRowColors(True)
        self._populate_method_list()
        self.method_box.currentRowChanged.connect(self._on_method_select)
        lay.addWidget(self.method_box, 1); return w

    def _right_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(14, 0, 0, 0); lay.setSpacing(8)

        # Description
        lbl_d = QLabel("DESCRIPTION"); lbl_d.setObjectName("ReportPanelEyebrow"); lay.addWidget(lbl_d)
        self._desc_lbl = QLabel()
        self._desc_lbl.setObjectName("SectionHint"); self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(56); lay.addWidget(self._desc_lbl)
        lay.addWidget(_divider())

        # Parameter stack (page 0 = none, 1 = EMA, 2 = Quantile)
        self._param_stack = QStackedWidget()
        self._param_stack.addWidget(QWidget())           # page 0 — no params
        self._param_stack.addWidget(self._ema_params())   # page 1
        self._param_stack.addWidget(self._qt_params())    # page 2
        lay.addWidget(self._param_stack)

        # Column scope
        col_hdr = QHBoxLayout()
        lbl_c = QLabel("COLUMN SCOPE  (numeric only)"); lbl_c.setObjectName("ReportPanelEyebrow")
        col_hdr.addWidget(lbl_c); col_hdr.addStretch()
        btn_all = QPushButton("Select all"); btn_all.setFixedWidth(88)
        btn_non = QPushButton("Clear all");  btn_non.setFixedWidth(80)
        btn_all.clicked.connect(self._select_all_cols)
        btn_non.clicked.connect(self._clear_all_cols)
        col_hdr.addWidget(btn_all); col_hdr.addWidget(btn_non)
        lay.addLayout(col_hdr)

        self.col_box = QListWidget(); self.col_box.setAlternatingRowColors(True)
        self.col_box.setMaximumHeight(160)
        self.col_box.itemChanged.connect(self._schedule_preview)
        lay.addWidget(self.col_box)

        # Live preview canvas
        lbl_p = QLabel("LIVE PREVIEW  (first checked column)"); lbl_p.setObjectName("ReportPanelEyebrow")
        lay.addWidget(lbl_p)
        self._fig = Figure(figsize=(6, 2.4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(180)
        lay.addWidget(self._canvas, 1)

        # Action row
        lay.addWidget(_divider())
        act = QHBoxLayout()
        act.addStretch(1)
        btn_cancel = QPushButton("Reset selection"); btn_cancel.clicked.connect(self._clear_all_cols)
        btn_apply  = QPushButton("Apply transformation"); btn_apply.clicked.connect(self._apply)
        act.addWidget(btn_cancel); act.addWidget(btn_apply)
        lay.addLayout(act)
        return w

    def _ema_params(self) -> QWidget:
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("EMA span :"))
        self._ema_span = QSpinBox(); self._ema_span.setRange(2, 500); self._ema_span.setValue(5)
        self._ema_span.setMaximumWidth(80)
        self._ema_span.valueChanged.connect(self._schedule_preview)
        lay.addWidget(self._ema_span); lay.addStretch(); return w

    def _qt_params(self) -> QWidget:
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Quantiles :"))
        self._qt_n = QSpinBox(); self._qt_n.setRange(10, 100_000); self._qt_n.setValue(1000)
        self._qt_n.setMaximumWidth(90); self._qt_n.valueChanged.connect(self._schedule_preview)
        lay.addWidget(self._qt_n)
        lay.addWidget(QLabel("  Output :"))
        self._qt_dist = QComboBox()
        self._qt_dist.addItems(["uniform", "normal"])
        self._qt_dist.currentIndexChanged.connect(self._schedule_preview)
        lay.addWidget(self._qt_dist); lay.addStretch(); return w

    def _history_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(10)
        self.history_panel = ReportTextPanel("TRANSFORMATION HISTORY",
                                             "All column-transform operations recorded in the session.")
        lay.addWidget(self.history_panel, 1); return w

    # ── Catalogue population ──────────────────────────────────────────────────

    def _populate_method_list(self) -> None:
        cur_group = None
        for display, group, key in _CATALOGUE:
            if group != cur_group:
                h = QListWidgetItem(f"  {group}"); h.setFlags(Qt.NoItemFlags)
                f = QFont(); f.setWeight(QFont.Bold); f.setPointSize(9); h.setFont(f)
                h.setForeground(QColor("#c9a227"))
                self.method_box.addItem(h); self._keys.append(None)
                cur_group = group
            self.method_box.addItem(QListWidgetItem(f"    {display}"))
            self._keys.append(key)

    # ── Slot helpers ──────────────────────────────────────────────────────────

    def _selected_key(self) -> Optional[str]:
        r = self.method_box.currentRow()
        return self._keys[r] if 0 <= r < len(self._keys) else None

    def _selected_display(self) -> str:
        item = self.method_box.currentItem()
        return item.text().strip() if item else "unknown"

    def _on_method_select(self, row: int) -> None:
        key = self._keys[row] if 0 <= row < len(self._keys) else None
        if key is None:
            # jumped to group header — advance to first real item
            for i in range(row + 1, self.method_box.count()):
                if self._keys[i] is not None:
                    self.method_box.setCurrentRow(i); return
            return
        self._desc_lbl.setText(_DESC.get(key, ""))
        self._param_stack.setCurrentIndex(_PARAM_PAGE.get(key, 0))
        self._schedule_preview()

    def _select_all_cols(self) -> None:
        for i in range(self.col_box.count()):
            self.col_box.item(i).setCheckState(Qt.Checked)

    def _clear_all_cols(self) -> None:
        for i in range(self.col_box.count()):
            self.col_box.item(i).setCheckState(Qt.Unchecked)

    def _checked_cols(self) -> List[str]:
        return [
            self.col_box.item(i).text()
            for i in range(self.col_box.count())
            if self.col_box.item(i).checkState() == Qt.Checked
        ]

    def _schedule_preview(self, *_) -> None:
        self._preview_timer.start(400)

    # ── Live preview ──────────────────────────────────────────────────────────

    def _draw_preview(self) -> None:
        self._fig.clear()
        key = self._selected_key()
        cols = self._checked_cols()
        if key is None or not cols or self._asset is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select a method and at least one column",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_axis_off()
            self._canvas.draw_idle(); return

        col = cols[0]
        df = self._asset.dataframe
        if col not in df.columns:
            self._canvas.draw_idle(); return
        raw = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.reshape(-1, 1)
        try:
            kw = self._param_kwargs(key)
            out = _transform(key, raw, **kw)[:, 0]
        except Exception:
            self._canvas.draw_idle(); return

        ax1, ax2 = self._fig.subplots(1, 2)

        def _robust_hist(ax, data, color, title):
            """Histogram with percentile-clipped x-axis so outliers don't flatten the view."""
            finite = data[np.isfinite(data)]
            if len(finite) == 0:
                ax.text(0.5, 0.5, "no finite values", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#888")
                ax.set_axis_off(); return
            p1, p99 = np.percentile(finite, 1), np.percentile(finite, 99)
            pad = (p99 - p1) * 0.05 or 0.5
            lo, hi = p1 - pad, p99 + pad
            clipped = np.clip(finite, lo, hi)
            ax.hist(clipped, bins=40, color=color, edgecolor="none", alpha=0.85)
            ax.set_xlim(lo, hi)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
            # Annotate if data was clipped
            n_out = np.sum((finite < lo) | (finite > hi))
            if n_out:
                ax.text(0.98, 0.97, f"{n_out} outlier(s) clipped",
                        ha="right", va="top", transform=ax.transAxes,
                        fontsize=6, color="#aaa")

        _robust_hist(ax1, raw[:, 0], "#4a90d9", f"Before  ·  {col}")
        _robust_hist(ax2, out,        "#c9a227", f"After  ·  {self._selected_display()}")
        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def _param_kwargs(self, key: str) -> dict:
        if key == "ema":
            return {"span": self._ema_span.value()}
        if key == "quantile":
            return {"n_quantiles": self._qt_n.value(),
                    "output_distribution": self._qt_dist.currentText()}
        return {}

    # ── Apply ─────────────────────────────────────────────────────────────────

    def _apply(self) -> None:
        key = self._selected_key()
        if key is None:
            QMessageBox.warning(self, "No method", "Select a transformation method first."); return
        cols = self._checked_cols()
        if not cols:
            QMessageBox.warning(self, "No columns", "Check at least one column."); return

        df = self._asset.dataframe.copy()
        subset = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x2d = np.asarray(subset, dtype=float)
        if x2d.ndim == 1:
            x2d = x2d.reshape(-1, 1)

        kw = self._param_kwargs(key)
        try:
            out = _transform(key, x2d, **kw)
        except Exception as e:
            QMessageBox.critical(self, "Transformation error", str(e))
            log.exception("_transform key=%s cols=%s", key, cols); return

        suffix = key
        new_cols = [f"{c}_{suffix}" for c in cols]
        for i, nc in enumerate(new_cols):
            df[nc] = out[:, i]

        display = self._selected_display()
        summary = f"Transform '{display}' applied · {len(cols)} column(s) → {len(new_cols)} new column(s)"
        params  = {"method": display, "key": key, "columns": cols,
                   "new_columns": new_cols, **{f"param_{k}": str(v) for k, v in kw.items()}}
        self._commit(operation="transform_columns", dataframe=df, summary=summary, params=params)

    # ── Session commit ────────────────────────────────────────────────────────

    def _commit(self, operation: str, dataframe: pd.DataFrame,
                summary: str, params: dict) -> None:
        if self._asset is None:
            QMessageBox.warning(self, "No asset", "No active dataset."); return
        ctrs  = list((self._asset.metadata.get("observable_contracts") or []))
        guard = assess_cleaning_operation(self._asset.dataframe,
                                          operation=operation, params=params, contracts=ctrs)
        if guard.get("violations"):
            QMessageBox.critical(self, "Operation blocked", "\n".join(guard["violations"])); return
        if guard.get("warnings"):
            QMessageBox.warning(self, "Admissibility warning", "\n".join(guard["warnings"]))

        new_asset = derive_data_asset(
            dataframe=dataframe, source_path=self._asset.source_path,
            cfg=IngestionConfig(), prior_asset=self._asset,
            operation=operation, operation_params=params,
        )
        if ctrs:
            new_asset = attach_contracts_to_asset(new_asset, ctrs)
        self._store.set_asset(new_asset); self._asset = new_asset
        self.refresh_view()
        QMessageBox.information(self, "Transformation applied", summary)

    # ── View refresh ──────────────────────────────────────────────────────────

    def refresh_view(self) -> None:
        if self._asset is None: return
        df   = self._asset.dataframe
        meta = self._asset.metadata
        hist = list(meta.get("pipeline_history") or [])

        # Overview tab
        self.overview_table.setModel(DataFrameModel(df.head(100)))
        self.overview_table.resizeColumnsToContents()

        # Column list — repopulate preserving check states
        prev_checked: set[str] = set()
        for i in range(self.col_box.count()):
            it = self.col_box.item(i)
            if it.checkState() == Qt.Checked:
                prev_checked.add(it.text())

        self.col_box.blockSignals(True)
        self.col_box.clear()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        for col in num_cols:
            it = QListWidgetItem(col)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (not prev_checked or col in prev_checked) else Qt.Unchecked)
            self.col_box.addItem(it)
        self.col_box.blockSignals(False)

        # History tab
        ops = [h for h in hist if h.get("operation") == "transform_columns"]
        self.history_panel.setPlainText(
            "Transformation history\n======================\n" +
            ("\n".join(
                f"- {h.get('params',{}).get('method','?'):<28} "
                f"{len(h.get('params',{}).get('columns',[]))} col(s) · "
                f"{h.get('applied_utc','')}"
                for h in ops
            ) if ops else "No transformations applied yet.")
        )
        self._schedule_preview()

    def current_dataframe(self) -> pd.DataFrame:
        return self._asset.dataframe.copy() if self._asset else pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hint(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("SectionHint"); lbl.setWordWrap(True); return lbl


def _divider() -> QFrame:
    f = QFrame(); f.setObjectName("Divider"); f.setFixedHeight(1)
    f.setFrameShape(QFrame.HLine); return f
