#!/usr/bin/env python3
"""
DNV Scientific Module
---------------------
Role:
    Application entry point.  Constructs the Qt application singleton,
    applies the global stylesheet and typography baseline, and launches
    the MainWindow event loop.

Scientific Context:
    No domain transformation occurs here.  This module is a pure bootstrap
    sequence with no observable state of its own.

Invariants:
    - Exactly one QApplication instance is created per process lifetime.
    - The global font is applied before any widget is constructed.
    - sys.exit() receives the return code from app.exec() without modification.

Assumptions:
    - A display server is available (X11, Wayland, or Win32).
    - Python ≥ 3.10 (match/case syntax is not used; dataclasses with Final are).

Failure Modes:
    - No display available: QApplication raises at construction; process exits.
    - Missing PySide6: ImportError before main() is reached.

Provenance:
    - Module: main.py
    - Conforms to: DNV Scientific Module Standard v1.0
"""
import logging
import sys
from typing import Final

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from PySide6.QtGui     import QFont
from PySide6.QtWidgets import QApplication

from gui_theme import get_stylesheet
from gui_main  import MainWindow

#: Application-wide baseline font size [pt].
APP_FONT_SIZE_PT: Final[int] = 12

#: Application-wide baseline font family.
APP_FONT_FAMILY: Final[str] = "Segoe UI"


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet())

    font = QFont(APP_FONT_FAMILY, APP_FONT_SIZE_PT)
    font.setHintingPreference(QFont.PreferFullHinting)
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setKerning(True)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
