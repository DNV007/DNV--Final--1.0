"""
DNV Scientific Module
---------------------
Role:
    Provides PlaceholderWindow — a minimal scaffold displayed for any
    pipeline module not yet dispatched.

Scientific Context:
    No scientific computation; renders a titled panel indicating the
    module is pending implementation.

Invariants:
    - PlaceholderWindow always displays the supplied title and category
      text without modification.

Assumptions:
    - ModuleDialog (gui_base) is importable and functional.

Failure Modes:
    - Title or category is empty string: displayed as empty label without
      error.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from gui_base import ModuleDialog


class PlaceholderWindow(ModuleDialog):
    def __init__(self, title: str, category: str, parent=None):
        super().__init__(
            title=title,
            hint="Module scaffold — contract-aware analysis pending implementation.",
            category=category, parent=parent,
        )
        inner = QFrame(); inner.setObjectName("PanelCard")
        lay   = QVBoxLayout(inner); lay.setContentsMargins(24,24,24,24); lay.setSpacing(12)

        badge = QLabel("[MODULE]"); badge.setObjectName("CategoryBadge")
        badge.setAlignment(Qt.AlignCenter); lay.addWidget(badge)

        msg = QLabel(
            f"{title}\n\nThis module is scaffolded for DNV-native implementation.\n"
            "It will provide: algorithm selection, parameter validation,\n"
            "progress tracking, reproducible audit logging, and export."
        )
        msg.setObjectName("SectionHint"); msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True); lay.addWidget(msg); lay.addStretch(1)
        self.body_layout.addWidget(inner)
