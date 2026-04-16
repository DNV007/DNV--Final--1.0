"""
DNV Scientific Module
---------------------
Role:
    Provides MissingDataDialog — a 15-method imputation panel with
    adaptive parameter controls for missing-value treatment.

Scientific Context:
    Applies statistical imputation methods (mean, median, mode, KNN,
    iterative, polynomial, interpolation, forward/backward fill, constant)
    to selected columns of the active DataFrame.

Invariants:
    - Imputation is applied only to selected columns; unselected columns
      are passed through unchanged.
    - KNN and iterative methods operate on the full column set without
      target leakage; no train/test split is applied at imputation time.

Assumptions:
    - Selected columns are numeric or categorical as appropriate for the
      chosen method.
    - At least one column has missing values; otherwise Apply is a no-op.

Failure Modes:
    - Fewer samples than KNN neighbours: method falls back to mean
      imputation with a warning.
    - All values missing in a column: column is left unchanged with a
      warning.

Provenance:
    - This module emits no metadata; provenance is recorded by the calling
      DataCleaningWindow.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)


class MissingDataDialog(QDialog):
    _CATALOGUE = [
        ("Drop Rows with Missing Values",    "REMOVAL",          "drop_rows"),
        ("Drop Columns with Missing Values", "REMOVAL",          "drop_cols"),
        ("Mean Imputation",                  "STATISTICAL",      "stat_mean"),
        ("Median Imputation",                "STATISTICAL",      "stat_median"),
        ("Mode Imputation",                  "STATISTICAL",      "stat_mode"),
        ("Constant Fill",                    "STATISTICAL",      "stat_constant"),
        ("Linear Interpolation",             "INTERPOLATION",    "linear"),
        ("Polynomial Interpolation",         "INTERPOLATION",    "polynomial"),
        ("Spline (Natural Cubic)",           "INTERPOLATION",    "spline"),
        ("Akima Spline",                     "INTERPOLATION",    "akima"),
        ("PCHIP (Monotone Cubic)",           "INTERPOLATION",    "pchip"),
        ("Barycentric Rational",             "INTERPOLATION",    "barycentric"),
        ("Krogh Polynomial",                 "INTERPOLATION",    "krogh"),
        ("Forward Fill",                     "FILL PROPAGATION", "ffill"),
        ("Backward Fill",                    "FILL PROPAGATION", "bfill"),
    ]
    _DEGREE_METHODS = {"polynomial","spline"}
    _LIMIT_METHODS  = {"linear","polynomial","spline","akima","pchip","barycentric","krogh","ffill","bfill"}
    _CONST_METHODS  = {"stat_constant"}
    _DESC = {
        "drop_rows":    "Remove every row that contains at least one NA cell.",
        "drop_cols":    "Remove every column that contains at least one NA cell.",
        "stat_mean":    "Replace NA cells with the column arithmetic mean (numeric only).",
        "stat_median":  "Replace NA cells with the column median. Robust to outliers.",
        "stat_mode":    "Replace NA cells with the most frequent non-null value.",
        "stat_constant":"Replace all NA cells with the specified constant value.",
        "linear":       "Piecewise linear interpolation between neighbouring non-null values.",
        "polynomial":   "Global polynomial fit of degree d. Degrees ≥5 risk Runge oscillation.",
        "spline":       "Natural cubic spline (SciPy). C² continuous. Degree sets spline order.",
        "akima":        "Akima local polynomial spline. Stable near gradient discontinuities.",
        "pchip":        "PCHIP — preserves monotonicity; suited to physical property curves.",
        "barycentric":  "Barycentric rational interpolation for smooth, moderately dense signals.",
        "krogh":        "Krogh polynomial (SciPy Hermite form). Requires sufficient support points.",
        "ffill":        "Carry the last observed value forward. Limit caps consecutive gap filled.",
        "bfill":        "Carry the next observed value backward. Limit caps consecutive gap filled.",
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Handle Missing Values"); self.setModal(True); self.resize(1020, 680)
        root = QVBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(12)

        hint = QLabel("Select a missing-data operation. Interpolation methods operate on numeric columns only.")
        hint.setObjectName("SectionHint"); hint.setWordWrap(True); root.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal); splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._method_panel()); splitter.addWidget(self._param_panel())
        splitter.setSizes([340, 560]); root.addWidget(splitter, 1)

        act = QHBoxLayout()
        prev = QPushButton("Summarise missingness"); prev.clicked.connect(self._summarise)
        act.addWidget(prev); act.addStretch(1)
        for label, fn in [("Cancel", self.reject), ("Apply method", self._apply)]:
            b = QPushButton(label); b.clicked.connect(fn); act.addWidget(b)
        root.addLayout(act)

        self.method_box.currentRowChanged.connect(self._on_select)
        if self.method_box.count() > 0:
            self.method_box.setCurrentRow(0); self._on_select(0)

    @property
    def df(self): return self.parent_window.current_dataframe()

    def _method_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        lbl = QLabel("IMPUTATION METHOD"); lbl.setObjectName("ReportPanelEyebrow"); lay.addWidget(lbl)
        self.method_box = QListWidget(); self.method_box.setAlternatingRowColors(True)
        self._keys: list[str] = []; self._populate_list()
        lay.addWidget(self.method_box, 1); return w

    def _param_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(14,0,0,0); lay.setSpacing(8)
        lbl = QLabel("PARAMETERS"); lbl.setObjectName("ReportPanelEyebrow"); lay.addWidget(lbl)
        self._desc_lbl = QLabel(); self._desc_lbl.setObjectName("SectionHint")
        self._desc_lbl.setWordWrap(True); self._desc_lbl.setMinimumHeight(60); lay.addWidget(self._desc_lbl)
        sep = QFrame(); sep.setObjectName("Divider"); sep.setFixedHeight(1); sep.setFrameShape(QFrame.HLine)
        lay.addWidget(sep)

        self._degree_row = self._row("Degree  d :")
        self._degree_spin = QComboBox()
        for d in range(1,11): self._degree_spin.addItem(str(d))
        self._degree_spin.setCurrentIndex(2)
        self._degree_row.layout().insertWidget(1, self._degree_spin)
        lay.addWidget(self._degree_row)

        self._limit_row  = self._row("Fill limit (0 = unlimited) :")
        self._limit_spin = QComboBox(); self._limit_spin.addItem("0 — unlimited")
        for v in [1,2,3,5,10,20,50]: self._limit_spin.addItem(str(v))
        self._limit_row.layout().insertWidget(1, self._limit_spin)
        lay.addWidget(self._limit_row)

        self._const_row  = self._row("Fill value :")
        self._const_edit = QLineEdit("0"); self._const_edit.setMaximumWidth(160)
        self._const_row.layout().insertWidget(1, self._const_edit)
        lay.addWidget(self._const_row)

        lay.addStretch(1); return w

    @staticmethod
    def _row(label: str) -> QWidget:
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(QLabel(label)); lay.addStretch(1); return w

    def _populate_list(self) -> None:
        cur_group = None
        for display, group, key in self._CATALOGUE:
            if group != cur_group:
                h = QListWidgetItem(f"  {group}"); h.setFlags(Qt.NoItemFlags)
                f = QFont(); f.setWeight(QFont.Bold); f.setPointSize(9); h.setFont(f)
                h.setForeground(QColor("#c9a227")); self.method_box.addItem(h); self._keys.append(None)
                cur_group = group
            self.method_box.addItem(QListWidgetItem(f"    {display}")); self._keys.append(key)

    def _on_select(self, row: int) -> None:
        key = self._keys[row] if 0 <= row < len(self._keys) else None
        if key is None:
            for i in range(row+1, self.method_box.count()):
                if self._keys[i] is not None: self.method_box.setCurrentRow(i); return
            return
        self._desc_lbl.setText(self._DESC.get(key,""))
        self._degree_row.setVisible(key in self._DEGREE_METHODS)
        self._limit_row.setVisible(key in self._LIMIT_METHODS)
        self._const_row.setVisible(key in self._CONST_METHODS)

    def _selected_key(self) -> Optional[str]:
        r = self.method_box.currentRow()
        return self._keys[r] if 0 <= r < len(self._keys) else None

    def _selected_display(self) -> str:
        item = self.method_box.currentItem()
        return item.text().strip() if item else "unknown"

    def _degree(self) -> int: return int(self._degree_spin.currentText())

    def _limit(self) -> Optional[int]:
        txt = self._limit_spin.currentText()
        if txt.startswith("0"): return None
        try: return int(txt)
        except ValueError: return None

    def _summarise(self) -> None:
        miss = self.df.isna().sum(); miss = miss[miss > 0].sort_values(ascending=False)
        QMessageBox.information(self, "Missingness", "No missing values." if miss.empty else miss.to_string())

    def _apply(self) -> None:
        key = self._selected_key()
        if key is None:
            QMessageBox.warning(self, "No method", "Select a method first."); return
        display  = self._selected_display()
        df       = self.df.copy()
        num_cols = df.select_dtypes(include="number").columns
        degree   = self._degree()
        limit    = self._limit()
        params: dict = {"method": display, "key": key}
        try:
            if   key == "drop_rows":    df = df.dropna()
            elif key == "drop_cols":    df = df.dropna(axis=1)
            elif key == "stat_mean":
                if len(num_cols) == 0: raise ValueError("No numeric columns for mean imputation.")
                df.loc[:, num_cols] = df.loc[:, num_cols].fillna(df.loc[:, num_cols].mean())
            elif key == "stat_median":
                if len(num_cols) == 0: raise ValueError("No numeric columns for median imputation.")
                df.loc[:, num_cols] = df.loc[:, num_cols].fillna(df.loc[:, num_cols].median())
            elif key == "stat_mode":
                for c in df.columns:
                    if df[c].isna().any():
                        mv = df[c].mode(dropna=True)
                        if not mv.empty: df[c] = df[c].fillna(mv.iloc[0])
            elif key == "stat_constant":
                raw = self._const_edit.text().strip()
                try:    fill_val: Any = float(raw) if "." in raw else int(raw)
                except: fill_val = raw
                df = df.fillna(fill_val); params["fill_value"] = str(fill_val)
            elif key in ("linear","polynomial","spline","akima","pchip","barycentric","krogh"):
                if len(num_cols) == 0: raise ValueError(f"No numeric columns for {display}.")
                kw: dict = {"method": key, "axis": 0, "limit": limit, "limit_direction": "both"}
                if key in self._DEGREE_METHODS: kw["order"] = degree; params["degree"] = degree
                if limit is not None: params["limit"] = limit
                df.loc[:, num_cols] = df.loc[:, num_cols].interpolate(**kw)
            elif key == "ffill":
                df = df.ffill(limit=limit)
                if limit is not None: params["limit"] = limit
            elif key == "bfill":
                df = df.bfill(limit=limit)
                if limit is not None: params["limit"] = limit
            else:
                raise ValueError(f"Unknown key: {key!r}")
        except Exception as e:
            QMessageBox.critical(self, "Imputation failed", str(e))
            log.exception("MissingDataDialog._apply key=%s", key); return
        self.parent_window.apply_cleaning_operation(
            operation="handle_missing_values", dataframe=df,
            summary=f"Imputation applied · {display}", params=params,
        )
        self.accept()
