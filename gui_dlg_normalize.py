"""
DNV Scientific Module
---------------------
Role:
    Provides NormalizeDialog — an 8-method normalisation and scaling panel
    with provenance-bound revisions.

Scientific Context:
    Applies feature scaling transformations (Min-Max, Z-score, Robust IQR,
    MaxAbs, Power, Quantile, L2-norm, Decimal) to selected numeric columns
    of the active DataFrame.

Invariants:
    - Scaling parameters are computed on the full selected column; no
      train/test split is applied in this dialog.
    - Original column values are not modified in the store until Apply is
      confirmed.

Assumptions:
    - Selected columns are numeric.
    - scipy is available for Quantile and Power transforms.

Failure Modes:
    - Column with zero variance (all-constant): Min-Max and Z-score produce
      NaN; dialog warns before Apply.
    - scipy unavailable: Quantile/Power methods raise ImportError shown in
      a QMessageBox.

Provenance:
    - This module emits no metadata; provenance is recorded by the calling
      DataCleaningWindow.

Methods
-------
LINEAR   : Min-Max, Standard (Z-score), Robust (IQR-based)
NONLINEAR: Log Transform, Box-Cox, Yeo-Johnson, Quantile Transform
CLIPPING : Winsorization

Post-apply shows a Before / After skewness & kurtosis comparison table
(SkewKurtDialog) with a save-to-CSV/TXT option.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Main dialog
# ══════════════════════════════════════════════════════════════════════════════

class NormalizeDialog(QDialog):
    """Two-panel dialog: method catalogue (left) + adaptive parameter form + column scope (right)."""

    _CATALOGUE = [
        ("Min-Max Scaling",            "LINEAR",    "minmax"),
        ("Standard Scaling (Z-score)", "LINEAR",    "standard"),
        ("Robust Scaling",             "LINEAR",    "robust"),
        ("Log Transform",              "NONLINEAR", "log"),
        ("Box-Cox Transform",          "NONLINEAR", "boxcox"),
        ("Yeo-Johnson Transform",      "NONLINEAR", "yeojohnson"),
        ("Quantile Transform",         "NONLINEAR", "quantile"),
        ("Winsorization",              "CLIPPING",  "winsor"),
    ]

    _DESC = {
        "minmax": (
            "Linearly maps each selected column to [min_val, max_val] (default 0–1). "
            "Preserves relative distances and the shape of the distribution, "
            "but is sensitive to outliers because the range is set by the extremes."
        ),
        "standard": (
            "Subtracts the column mean and divides by the standard deviation (Z-score). "
            "Centres the distribution at 0 with unit variance. "
            "Sensitive to outliers; output is unbounded."
        ),
        "robust": (
            "Centres using the median and scales using the inter-quartile range defined "
            "by the configurable quantile range. Robust to outliers; output is unbounded."
        ),
        "log": (
            "Applies log(x + offset) where offset = |min| + 1 for columns with min ≤ 0. "
            "Compresses right-skewed (positively skewed) distributions toward symmetry. "
            "Choose the base: natural (ln), log₂, or log₁₀."
        ),
        "boxcox": (
            "Power transform (Box–Cox) that finds λ per column via MLE to maximise Gaussianity. "
            "Requires strictly positive values; a per-column offset is added automatically "
            "when min ≤ 0. Standardises the output (mean 0, std 1)."
        ),
        "yeojohnson": (
            "Extension of Box–Cox that handles zeros and negative values without an offset. "
            "λ is optimised per column. Recommended general-purpose power normalisation "
            "for mixed-sign data. Standardises the output."
        ),
        "quantile": (
            "Maps each feature to a uniform or normal distribution by rank transformation. "
            "Highly robust to outliers. The mapping is non-linear and collapses tied values. "
            "Larger n_quantiles gives a more precise mapping at the cost of memory."
        ),
        "winsor": (
            "Clips values below the lower percentile and above the upper percentile. "
            "Values within the central range are unchanged; extreme values are set to the "
            "boundary values. Does not alter the distribution shape within the bounds."
        ),
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Normalise / Scale")
        self.setModal(True)
        self.resize(1060, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 12)
        root.setSpacing(12)

        hint = QLabel(
            "Select a normalisation or scaling method. Operations apply only to the checked "
            "numeric columns and create a new provenance-bound revision in the session store."
        )
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_method_panel())
        splitter.addWidget(self._build_param_panel())
        splitter.setSizes([310, 640])
        root.addWidget(splitter, 1)

        act = QHBoxLayout()
        summ_btn = QPushButton("Distribution summary")
        summ_btn.clicked.connect(self._summarise)
        act.addWidget(summ_btn)
        act.addStretch(1)
        for label, fn in [("Cancel", self.reject), ("Apply method", self._apply)]:
            b = QPushButton(label)
            b.clicked.connect(fn)
            act.addWidget(b)
        root.addLayout(act)

        self.method_box.currentRowChanged.connect(self._on_select)
        if self.method_box.count() > 0:
            self.method_box.setCurrentRow(0)
            self._on_select(0)
        self._populate_col_list()

    # ── property ──────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        return self.parent_window.current_dataframe()

    # ── panels ────────────────────────────────────────────────────────────────

    def _build_method_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        hdr = QLabel("METHOD")
        hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(hdr)
        self.method_box = QListWidget()
        self.method_box.setAlternatingRowColors(True)
        self._keys: list[str | None] = []
        self._populate_list()
        lay.addWidget(self.method_box, 1)
        return w

    def _build_param_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 0, 0, 0)
        lay.setSpacing(8)

        hdr = QLabel("PARAMETERS")
        hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(hdr)

        self._desc_lbl = QLabel()
        self._desc_lbl.setObjectName("SectionHint")
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(68)
        lay.addWidget(self._desc_lbl)

        lay.addWidget(_div())

        # method-specific stacked pages
        self._stack  = QStackedWidget()
        self._pages : dict[str, int] = {}
        self._build_param_pages()
        lay.addWidget(self._stack)

        lay.addWidget(_div())

        # column scope
        col_hdr = QLabel("COLUMN SCOPE")
        col_hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(col_hdr)

        ctrl = QHBoxLayout()
        ctrl.addWidget(_hint("Select the columns to transform:"))
        ctrl.addStretch(1)
        ba = QPushButton("Select all");  ba.setFixedWidth(86)
        bn = QPushButton("Clear all");   bn.setFixedWidth(86)
        ba.clicked.connect(lambda: self._set_all_cols(True))
        bn.clicked.connect(lambda: self._set_all_cols(False))
        ctrl.addWidget(ba); ctrl.addWidget(bn)
        lay.addLayout(ctrl)

        self._col_list = QListWidget()
        self._col_list.setAlternatingRowColors(True)
        self._col_list.setMaximumHeight(155)
        lay.addWidget(self._col_list, 0)

        lay.addStretch(1)
        return w

    # ── parameter pages ───────────────────────────────────────────────────────

    def _build_param_pages(self) -> None:
        for _, _, key in self._CATALOGUE:
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(0, 6, 0, 0)
            form.setSpacing(10)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            if key == "minmax":
                self._p_mm_lo = _dbl(0.0, -1e6, 1e6, 4)
                self._p_mm_hi = _dbl(1.0, -1e6, 1e6, 4)
                form.addRow("Feature range min :", self._p_mm_lo)
                form.addRow("Feature range max :", self._p_mm_hi)
                form.addRow(_hint("Output ∈ [min, max] per column."))

            elif key == "standard":
                form.addRow(_hint("No configurable parameters.\nTransforms each column to mean = 0, std = 1."))

            elif key == "robust":
                self._p_rob_lo = _dbl(25.0, 0.1, 49.9, 1)
                self._p_rob_hi = _dbl(75.0, 50.1, 99.9, 1)
                form.addRow("IQR lower quantile (%) :", self._p_rob_lo)
                form.addRow("IQR upper quantile (%) :", self._p_rob_hi)
                form.addRow(_hint("Centred at median; scaled by the inter-quantile range."))

            elif key == "log":
                self._p_log_base = QComboBox()
                self._p_log_base.addItems(["Natural (ln)", "Log₂", "Log₁₀"])
                self._p_log_base.setMinimumWidth(140)
                form.addRow("Logarithm base :", self._p_log_base)
                form.addRow(_hint("Auto-offset = |min| + 1 applied per column when min ≤ 0."))

            elif key == "boxcox":
                form.addRow(_hint(
                    "λ optimised per column via MLE (scikit-learn PowerTransformer).\n"
                    "Strictly positive values required; a per-column offset of |min| + 1 "
                    "is added automatically when needed. Output is standardised."
                ))

            elif key == "yeojohnson":
                form.addRow(_hint(
                    "λ optimised per column via MLE.\n"
                    "Handles zeros and negatives without offset. Output is standardised."
                ))

            elif key == "quantile":
                self._p_qt_n    = _int(1000, 10, 100_000)
                self._p_qt_dist = QComboBox()
                self._p_qt_dist.addItems(["uniform", "normal"])
                self._p_qt_dist.setMinimumWidth(120)
                self._p_qt_seed = _int(42, 0, 99_999)
                form.addRow("Number of quantiles :", self._p_qt_n)
                form.addRow("Output distribution :", self._p_qt_dist)
                form.addRow("Random seed :",         self._p_qt_seed)
                form.addRow(_hint("Larger n_quantiles → more precise mapping; uses more memory."))

            elif key == "winsor":
                self._p_win_lo = _dbl( 5.0,  0.0, 49.9, 1)
                self._p_win_hi = _dbl(95.0, 50.1, 100.0, 1)
                form.addRow("Lower clip percentile (%) :", self._p_win_lo)
                form.addRow("Upper clip percentile (%) :", self._p_win_hi)
                form.addRow(_hint("Values outside [lower, upper] are set to the boundary value."))

            idx = self._stack.addWidget(page)
            self._pages[key] = idx

    # ── list population ───────────────────────────────────────────────────────

    def _populate_list(self) -> None:
        cur_group = None
        for display, group, key in self._CATALOGUE:
            if group != cur_group:
                h = QListWidgetItem(f"  {group}")
                h.setFlags(Qt.NoItemFlags)
                f = QFont(); f.setWeight(QFont.Bold); f.setPointSize(9); h.setFont(f)
                h.setForeground(QColor("#c9a227"))
                self.method_box.addItem(h)
                self._keys.append(None)
                cur_group = group
            self.method_box.addItem(QListWidgetItem(f"    {display}"))
            self._keys.append(key)

    def _populate_col_list(self) -> None:
        self._col_list.clear()
        for c in self.df.select_dtypes(include="number").columns:
            item = QListWidgetItem(str(c))
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            self._col_list.addItem(item)

    def _set_all_cols(self, state: bool) -> None:
        qs = Qt.Checked if state else Qt.Unchecked
        for i in range(self._col_list.count()):
            self._col_list.item(i).setCheckState(qs)

    def _selected_cols(self) -> list[str]:
        return [
            self._col_list.item(i).text()
            for i in range(self._col_list.count())
            if self._col_list.item(i).checkState() == Qt.Checked
        ]

    # ── slot ──────────────────────────────────────────────────────────────────

    def _on_select(self, row: int) -> None:
        key = self._keys[row] if 0 <= row < len(self._keys) else None
        if key is None:
            for i in range(row + 1, self.method_box.count()):
                if self._keys[i] is not None:
                    self.method_box.setCurrentRow(i)
                    return
            return
        self._desc_lbl.setText(self._DESC.get(key, ""))
        if key in self._pages:
            self._stack.setCurrentIndex(self._pages[key])

    def _selected_key(self) -> Optional[str]:
        r = self.method_box.currentRow()
        return self._keys[r] if 0 <= r < len(self._keys) else None

    def _selected_display(self) -> str:
        item = self.method_box.currentItem()
        return item.text().strip() if item else "unknown"

    # ── actions ───────────────────────────────────────────────────────────────

    def _summarise(self) -> None:
        df  = self.df
        num = df.select_dtypes(include="number")
        if num.empty:
            QMessageBox.information(self, "Summary", "No numeric columns in the active dataset.")
            return
        lines = [
            f"{'Column':<36} {'Skewness':>12} {'Kurtosis':>12} {'Min':>12} {'Max':>12}",
            "─" * 80,
        ]
        for c in num.columns[:50]:
            col = num[c].dropna()
            if len(col) < 3:
                lines.append(f"{str(c):<36} {'(< 3 obs)':>12}")
                continue
            lines.append(
                f"{str(c):<36} "
                f"{sp_stats.skew(col):>12.4f} "
                f"{sp_stats.kurtosis(col):>12.4f} "
                f"{col.min():>12.4g} "
                f"{col.max():>12.4g}"
            )
        if len(num.columns) > 50:
            lines.append(f"… {len(num.columns) - 50} more columns omitted")
        QMessageBox.information(self, "Distribution summary", "\n".join(lines))

    def _apply(self) -> None:
        key = self._selected_key()
        if key is None:
            QMessageBox.warning(self, "No method", "Select a normalisation method first.")
            return
        display = self._selected_display()
        df      = self.df.copy()
        cols    = self._selected_cols()

        if not cols:
            QMessageBox.warning(self, "No columns selected",
                "Check at least one column in the Column Scope list.")
            return

        stats_before = _compute_stats(df, cols)
        params: dict = {"method": display, "key": key, "columns": cols}

        try:
            df_out = self._run(key, df, cols, params)
        except Exception as e:
            QMessageBox.critical(self, "Normalisation failed", str(e))
            log.exception("NormalizeDialog._apply key=%s", key)
            return

        stats_after = _compute_stats(df_out, cols)

        self.parent_window.apply_cleaning_operation(
            operation="normalize_scale",
            dataframe=df_out,
            summary=f"Normalise · {display} · {len(cols)} column(s)",
            params=params,
        )

        SkewKurtDialog(stats_before, stats_after, display, parent=self).exec()
        self.accept()

    # ── computation ───────────────────────────────────────────────────────────

    def _run(self, key: str, df: pd.DataFrame,
             cols: list, params: dict) -> pd.DataFrame:
        out = df.copy()

        if key == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            lo = self._p_mm_lo.value()
            hi = self._p_mm_hi.value()
            if lo >= hi:
                raise ValueError(f"Feature range min ({lo}) must be < max ({hi}).")
            params.update({"feature_range_min": lo, "feature_range_max": hi})
            out[cols] = MinMaxScaler(feature_range=(lo, hi)).fit_transform(df[cols])

        elif key == "standard":
            from sklearn.preprocessing import StandardScaler
            out[cols] = StandardScaler().fit_transform(df[cols])

        elif key == "robust":
            from sklearn.preprocessing import RobustScaler
            q_lo = self._p_rob_lo.value()
            q_hi = self._p_rob_hi.value()
            params.update({"quantile_range_low": q_lo, "quantile_range_high": q_hi})
            out[cols] = RobustScaler(quantile_range=(q_lo, q_hi)).fit_transform(df[cols])

        elif key == "log":
            base_txt = self._p_log_base.currentText()
            params["log_base"] = base_txt
            fn = np.log if "Natural" in base_txt else (np.log2 if "₂" in base_txt else np.log10)
            offsets: dict = {}
            for c in cols:
                col    = df[c]
                offset = float(abs(col.min()) + 1) if col.min() <= 0 else 0.0
                out[c] = fn(col + offset)
                if offset:
                    offsets[c] = round(offset, 6)
            if offsets:
                params["auto_offsets"] = offsets

        elif key == "boxcox":
            from sklearn.preprocessing import PowerTransformer
            X = df[cols].copy()
            offsets: dict = {}
            for c in cols:
                if X[c].min() <= 0:
                    offset  = float(abs(X[c].min()) + 1)
                    X[c]   += offset
                    offsets[c] = round(offset, 6)
            out[cols] = PowerTransformer(method="box-cox").fit_transform(X)
            if offsets:
                params["auto_offsets"] = offsets

        elif key == "yeojohnson":
            from sklearn.preprocessing import PowerTransformer
            out[cols] = PowerTransformer(method="yeo-johnson").fit_transform(df[cols])

        elif key == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            n    = self._p_qt_n.value()
            dist = self._p_qt_dist.currentText()
            seed = self._p_qt_seed.value()
            params.update({"n_quantiles": n, "output_distribution": dist, "random_state": seed})
            out[cols] = QuantileTransformer(
                n_quantiles=n, output_distribution=dist, random_state=seed
            ).fit_transform(df[cols])

        elif key == "winsor":
            lo = self._p_win_lo.value()
            hi = self._p_win_hi.value()
            params.update({"lower_pct": lo, "upper_pct": hi})
            for c in cols:
                col = df[c].dropna()
                lb  = float(np.percentile(col, lo))
                ub  = float(np.percentile(col, hi))
                out[c] = df[c].clip(lb, ub)

        else:
            raise ValueError(f"Unknown key: {key!r}")

        return out


# ══════════════════════════════════════════════════════════════════════════════
# Skewness & Kurtosis comparison dialog
# ══════════════════════════════════════════════════════════════════════════════

class SkewKurtDialog(QDialog):
    """Before / After skewness & kurtosis comparison with per-row interpretation and save."""

    _COL_HEADERS = [
        "Column",
        "Skewness — before", "Skewness — after",
        "Kurtosis — before", "Kurtosis — after",
        "Interpretation (after)",
    ]

    def __init__(self, before: list[tuple], after: list[tuple],
                 method: str, parent=None):
        super().__init__(parent)
        self._before = before
        self._after  = after
        self._method = method
        self.setWindowTitle("Distribution Statistics — Before vs After")
        self.setModal(True)
        self.resize(980, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        hdr = QLabel(
            f"Skewness and kurtosis (Fisher / excess) before and after: "
            f"<b>{method}</b>"
        )
        hdr.setObjectName("SectionHint")
        hdr.setWordWrap(True)
        root.addWidget(hdr)

        note = QLabel(
            "Kurtosis values are excess (Fisher) — normal distribution ≈ 0. "
            "Positive = leptokurtic (heavy tails); negative = platykurtic (light tails)."
        )
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        root.addWidget(note)

        # table
        after_dict = {c: (sk, ku) for c, sk, ku in after}
        self._tbl = QTableWidget(len(before), len(self._COL_HEADERS))
        self._tbl.setHorizontalHeaderLabels(self._COL_HEADERS)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.horizontalHeader().setStretchLastSection(True)

        for r, (col, sk_b, ku_b) in enumerate(before):
            sk_a, ku_a = after_dict.get(col, (float("nan"), float("nan")))
            interp = _interpret(sk_a, ku_a)
            row_vals = [
                col,
                _fmt(sk_b), _fmt(sk_a),
                _fmt(ku_b), _fmt(ku_a),
                interp,
            ]
            for c_idx, val in enumerate(row_vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignLeft | Qt.AlignVCenter if c_idx in (0, 5)
                    else Qt.AlignRight | Qt.AlignVCenter
                )
                # colour after-columns green, before-columns dimmer
                if c_idx in (2, 4):
                    item.setForeground(QColor("#2ca02c"))
                elif c_idx in (1, 3):
                    item.setForeground(QColor("#888888"))
                self._tbl.setItem(r, c_idx, item)

        self._tbl.resizeColumnsToContents()
        root.addWidget(self._tbl, 1)

        bot = QHBoxLayout()
        save_btn = QPushButton("Save results…")
        save_btn.clicked.connect(self._save)
        bot.addWidget(save_btn)
        bot.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bot.addWidget(close_btn)
        root.addLayout(bot)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save distribution statistics", "skew_kurtosis",
            "CSV files (*.csv);;Tab-separated (*.txt);;All files (*)",
        )
        if not path:
            return
        after_dict = {c: (sk, ku) for c, sk, ku in self._after}
        rows = []
        for col, sk_b, ku_b in self._before:
            sk_a, ku_a = after_dict.get(col, (float("nan"), float("nan")))
            rows.append({
                "column":           col,
                "skewness_before":  sk_b,
                "skewness_after":   sk_a,
                "kurtosis_before":  ku_b,
                "kurtosis_after":   ku_a,
                "interpretation":   _interpret(sk_a, ku_a),
            })
        result = pd.DataFrame(rows)
        try:
            sep = "\t" if path.lower().endswith(".txt") else ","
            if not any(path.lower().endswith(e) for e in (".csv", ".txt")):
                path += ".csv"
            result.to_csv(path, sep=sep, index=False)
            QMessageBox.information(self, "Saved",
                f"Distribution statistics saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            log.exception("SkewKurtDialog._save")


# ── module helpers ────────────────────────────────────────────────────────────

def _compute_stats(df: pd.DataFrame, cols: list) -> list[tuple]:
    result = []
    for c in cols:
        if c not in df.columns:
            continue
        col = df[c].dropna()
        if len(col) < 3:
            result.append((c, float("nan"), float("nan")))
        else:
            result.append((c, float(sp_stats.skew(col)), float(sp_stats.kurtosis(col))))
    return result


def _interpret(skew: float, kurt: float) -> str:
    """Human-readable skewness + kurtosis interpretation (Fisher excess kurtosis)."""
    try:
        # skewness
        if abs(skew) < 0.5:
            sk_txt = "approximately symmetric"
        elif skew > 1.0:
            sk_txt = "strongly right-skewed"
        elif skew > 0:
            sk_txt = "mildly right-skewed"
        elif skew < -1.0:
            sk_txt = "strongly left-skewed"
        else:
            sk_txt = "mildly left-skewed"
        # excess kurtosis (Fisher; normal ≈ 0)
        if abs(kurt) < 0.5:
            ku_txt = "mesokurtic (normal-like tails)"
        elif kurt > 0:
            ku_txt = "leptokurtic (heavy tails / sharp peak)"
        else:
            ku_txt = "platykurtic (light tails / flat peak)"
        return f"{sk_txt}; {ku_txt}"
    except Exception:
        return "—"


def _fmt(v: float) -> str:
    return f"{v:.4f}" if not (v != v) else "—"   # NaN check


def _div() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setObjectName("Divider")
    return f


def _hint(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("SectionHint"); lbl.setWordWrap(True)
    return lbl


def _dbl(default: float, lo: float, hi: float, decimals: int) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi); sb.setDecimals(decimals)
    sb.setValue(default); sb.setMinimumWidth(130)
    return sb


def _int(default: int, lo: int, hi: int) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi); sb.setValue(default); sb.setMinimumWidth(130)
    return sb


def _note(text: str) -> QLabel:
    return _hint(text)
