"""
DNV Scientific Module
---------------------
Role:
    Provides shared GUI widgets — DataFrameModel, ReportTextPanel, and
    layout factory helpers used across all module windows.

Scientific Context:
    Defines the Qt model/view bridge for pandas DataFrames, the read-only
    report text panel, and reusable divider and signature widgets; no
    scientific computation.

Invariants:
    - DataFrameModel wraps an immutable DataFrame view; mutation of the
      source DataFrame is not reflected until a new model is set.
    - make_signature_widget always produces a widget carrying APP_TITLE,
      APP_VERSION, and APP_ANALYST.

Assumptions:
    - Input DataFrame has string-convertible values in all cells.
    - QFont "Consolas" or a monospace fallback is available on the host.

Failure Modes:
    - Cell value cannot be converted to str: displayed as empty string.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui  import QCursor, QFont, QTextOption
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

APP_TITLE   = "DNV-1.0"
APP_VERSION = "KR"
APP_ANALYST = "EDA"


# ── DataFrameModel ───────────────────────────────────────────────────────────

class DataFrameModel(QAbstractTableModel):
    """Read-only table model backed by a pandas DataFrame.
    Uses itertuples() for fast construction on large preview frames.
    """
    _MAX_CELL = 400

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._cols = list(df.columns)
        self._data: List[List[str]] = []
        for row in df.itertuples(index=False, name=None):
            cells: List[str] = []
            for v in row:
                try:    is_na = pd.isna(v)
                except (TypeError, ValueError): is_na = False
                if is_na:
                    cells.append("")
                else:
                    s = str(v)
                    cells.append(s if len(s) <= self._MAX_CELL else s[:self._MAX_CELL] + " …")
            self._data.append(cells)

    def rowCount(self, parent=QModelIndex()) -> int:    return len(self._data)
    def columnCount(self, parent=QModelIndex()) -> int: return len(self._cols)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole: return None
        return self._data[index.row()][index.column()]

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._cols[section]) if orientation == Qt.Horizontal else str(section)

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable


# ── ReportTextPanel ──────────────────────────────────────────────────────────

class ReportTextPanel(QWidget):
    """Copy-oriented audit surface with eyebrow label and one-click copy."""

    def __init__(self, eyebrow: str, hint: str, parent: QWidget | None = None):
        super().__init__(parent)
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        card = QFrame(); card.setObjectName("ReportPanelCard")
        cl   = QVBoxLayout(card); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)

        hdr = QFrame(); hdr.setObjectName("ReportPanelHeader")
        hl  = QHBoxLayout(hdr); hl.setContentsMargins(14,10,14,10); hl.setSpacing(10)

        tc = QVBoxLayout(); tc.setContentsMargins(0,0,0,0); tc.setSpacing(2)
        el = QLabel(eyebrow); el.setObjectName("ReportPanelEyebrow")
        hl2 = QLabel(hint);   hl2.setObjectName("ReportPanelHint"); hl2.setWordWrap(True)
        tc.addWidget(el); tc.addWidget(hl2)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.copy_btn.clicked.connect(self._copy)
        hl.addLayout(tc, 1); hl.addWidget(self.copy_btn, 0, Qt.AlignTop)

        self.text_view = QTextEdit()
        self.text_view.setObjectName("ReportTextView")
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Consolas", 10))
        self.text_view.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.text_view.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)

        cl.addWidget(hdr); cl.addWidget(self.text_view, 1)
        root.addWidget(card)

    def setPlainText(self, text: str) -> None: self.text_view.setPlainText(text)
    def toPlainText(self) -> str:              return self.text_view.toPlainText()
    def _copy(self) -> None:                   QApplication.clipboard().setText(self.text_view.toPlainText())


# ── Factory helpers ──────────────────────────────────────────────────────────

def make_divider() -> QFrame:
    d = QFrame(); d.setObjectName("Divider"); d.setFixedHeight(1); d.setFrameShape(QFrame.HLine)
    return d


def make_section_header(title: str, hint: str = "", category: str = "") -> QWidget:
    w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)
    if category:
        c = QLabel(category); c.setObjectName("CategoryBadge"); lay.addWidget(c)
    t = QLabel(title); t.setObjectName("SectionTitle"); lay.addWidget(t)
    if hint:
        h = QLabel(hint); h.setObjectName("SectionHint"); h.setWordWrap(True); lay.addWidget(h)
    return w


def make_signature_widget(analysis_id: str = "–") -> QWidget:
    try:
        from ing_provenance import utc_now_iso
        ts = utc_now_iso()
    except Exception:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
    text = (f"Analyst: {APP_ANALYST}   |   Generated by: {APP_TITLE} v{APP_VERSION}"
            f"   |   {ts}   |   Record: {analysis_id}")
    lbl = QLabel(text); lbl.setObjectName("SignatureLine"); lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lay.addWidget(lbl)
    return w
