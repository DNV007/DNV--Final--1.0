"""
DNV Scientific Module
---------------------
Role:
    Provides ModuleDialog base class and IngestWorker background thread
    for all DNV module windows.

Scientific Context:
    Defines the structural contract for all eight pipeline module windows —
    header panel, card body, signature footer, and background ingestion
    thread lifecycle.

Invariants:
    - All module windows inherit ModuleDialog and receive a card_layout body region.
    - IngestWorker cancellation is idempotent and always joined before close.

Assumptions:
    - QApplication and a primary screen are available at construction time.
    - gui_widgets.make_signature_widget and gui_widgets.make_divider are importable.

Failure Modes:
    - Worker thread not stopped before close: wait(3000) enforces a 3-second
      hard timeout after which the thread is abandoned.

Provenance:
    - This module emits no transformation metadata; provenance originates
      in the calling module windows.
"""
from __future__ import annotations

import logging, traceback
from typing import List, Optional

from PySide6.QtCore    import QThread, Signal
from PySide6.QtGui     import QCloseEvent, QColor
from PySide6.QtWidgets import (
    QApplication, QDialog, QFrame, QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from ing_pipeline import build_data_asset
from gui_widgets  import make_divider, make_signature_widget

log = logging.getLogger(__name__)
APP_TITLE = "DNV-1.0"


def _clamp(v: float, lo: float, hi: float) -> int:
    return max(int(lo), min(int(v), int(hi)))


# ── Background ingestion worker ───────────────────────────────────────────────

class IngestWorker(QThread):
    """Run build_data_asset in a background thread."""
    ok  = Signal(object)
    err = Signal(str)

    def __init__(self, path: str, cfg):
        super().__init__()
        self.path = path; self.cfg = cfg; self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled: return
            asset = build_data_asset(self.path, cfg=self.cfg)
            if not self._cancelled: self.ok.emit(asset)
        except Exception:
            if not self._cancelled: self.err.emit(traceback.format_exc())


# ── ModuleDialog base ─────────────────────────────────────────────────────────

class ModuleDialog(QDialog):
    """Base class for all module windows: header, card body, signature footer."""

    def __init__(self, title: str, hint: str, category: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_TITLE}  —  {title}")
        screen = QApplication.primaryScreen().geometry()
        self.setMinimumSize(int(screen.width() * 0.62), int(screen.height() * 0.62))

        root = QVBoxLayout(self)
        mg   = _clamp(screen.width() * 0.020, 14, 30)
        root.setContentsMargins(mg, mg, mg, mg)
        root.setSpacing(_clamp(screen.height() * 0.018, 14, 24))

        # Header
        hdr = QFrame(); hdr.setObjectName("HeaderPanel")
        hl  = QVBoxLayout(hdr)
        hp  = _clamp(screen.width() * 0.018, 14, 28)
        vp  = _clamp(screen.height() * 0.018, 12, 24)
        hl.setContentsMargins(hp, vp, hp, vp); hl.setSpacing(6)
        cat = QLabel(category); cat.setObjectName("CategoryBadge"); hl.addWidget(cat)
        tr  = QHBoxLayout(); tr.setSpacing(14)
        tl  = QLabel(title);  tl.setObjectName("SectionTitle")
        sb  = QLabel("ACTIVE"); sb.setObjectName("StatusBadge")
        tr.addWidget(tl); tr.addWidget(sb); tr.addStretch(); hl.addLayout(tr)
        hn  = QLabel(hint); hn.setObjectName("SectionHint"); hn.setWordWrap(True); hl.addWidget(hn)
        sh  = QGraphicsDropShadowEffect(); sh.setBlurRadius(24); sh.setXOffset(0)
        sh.setYOffset(8); sh.setColor(QColor(0,0,0,60)); hdr.setGraphicsEffect(sh)
        root.addWidget(hdr)
        root.addWidget(make_divider())

        # Content card — scrollable body inside a styled card frame
        card = QFrame(); card.setObjectName("PanelCard")
        cs   = QGraphicsDropShadowEffect(); cs.setBlurRadius(40); cs.setXOffset(0)
        cs.setYOffset(14); cs.setColor(QColor(0,0,0,100)); card.setGraphicsEffect(cs)
        card_outer = QVBoxLayout(card)
        card_outer.setContentsMargins(0, 0, 0, 0)
        card_outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_body = QWidget()
        cp = _clamp(screen.width() * 0.020, 14, 32)
        self.body_layout = QVBoxLayout(scroll_body)
        self.body_layout.setContentsMargins(cp, cp, cp, cp)
        self.body_layout.setSpacing(_clamp(screen.height() * 0.014, 12, 20))
        scroll.setWidget(scroll_body)
        card_outer.addWidget(scroll)

        self.card_layout = self.body_layout
        root.addWidget(card, 1)
        from gui_tile import ModuleTileButton
        self.tile_buttons: List[ModuleTileButton] = []

        root.addWidget(make_signature_widget())

    def closeEvent(self, event: QCloseEvent) -> None:
        worker: Optional[IngestWorker] = getattr(self, "_worker", None)
        if worker is not None and worker.isRunning():
            worker.cancel(); worker.quit(); worker.wait(3000)
        super().closeEvent(event)
