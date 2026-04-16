"""
DNV Scientific Module
---------------------
Role:
    Provides ThemeManager singleton, ThemeMode enum, and the full Qt
    stylesheet for light and dark modes.

Scientific Context:
    Defines all visual tokens (colours, fonts, spacing) for the DNV
    application shell; no scientific computation.

Invariants:
    - ThemeManager is a singleton; get_stylesheet() always returns a
      complete QSS string for the active mode.
    - Colour palettes are declared as named dicts; no palette value is
      anonymous.

Assumptions:
    - Qt stylesheet engine is available via QApplication.
    - ThemeMode.LIGHT and ThemeMode.DARK are the only valid modes.

Failure Modes:
    - Unknown mode requested: ThemeManager raises ValueError.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional


class ThemeMode(Enum):
    DARK  = "dark"
    LIGHT = "light"


class ThemeManager:
    """Singleton theme source of truth. Use ThemeManager.instance()."""
    _instance: Optional["ThemeManager"] = None
    _mode: ThemeMode = ThemeMode.LIGHT

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    def toggle(self) -> ThemeMode:
        self._mode = ThemeMode.DARK if self._mode == ThemeMode.LIGHT else ThemeMode.LIGHT
        return self._mode

    def is_dark(self) -> bool:
        return self._mode == ThemeMode.DARK


# module-level singleton
theme = ThemeManager.instance()


def get_stylesheet() -> str:
    if theme.is_dark():
        return """
QMainWindow, QDialog {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #0e1218, stop:0.4 #0b0f15, stop:1 #060809); }
QLabel#AppTitle { color:#c9a227; font-size:28px; font-weight:700;
    letter-spacing:-0.5px; font-family:'Segoe UI','SF Pro Display',system-ui; }
QLabel#AppSubtitle { color:#7a8a96; font-size:12px; letter-spacing:1.5px; font-family:'Segoe UI',monospace; }
QLabel#SignatureLine { color:#5a6a76; font-size:11px; font-family:'Courier New',monospace; letter-spacing:0.2px; }
QLabel#SectionTitle { color:#e8edf2; font-size:13px; font-weight:700; font-family:'Segoe UI Semibold',system-ui; }
QLabel#SectionHint  { color:#6a7a86; font-size:11px; font-family:'Segoe UI',system-ui; }
QLabel#CategoryBadge { color:#c9a227; font-size:10px; font-weight:700; letter-spacing:1.8px;
    padding:3px 8px; background:rgba(201,162,39,0.10);
    border:1px solid rgba(201,162,39,0.25); border-radius:4px; font-family:'Segoe UI',monospace; }
QLabel#StatusBadge { color:#00b86a; font-size:10px; font-weight:700; letter-spacing:1.5px;
    padding:3px 8px; background:rgba(0,184,106,0.10); border:1px solid rgba(0,184,106,0.25); border-radius:4px; }
QFrame#PanelCard { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1a1f2e,stop:1 #12161f);
    border:1px solid rgba(255,255,255,0.06); border-radius:12px; }
QFrame#HeaderPanel { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #181d28,stop:1 #101418);
    border:1px solid rgba(201,162,39,0.12); border-radius:10px; }
QFrame#Divider { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 transparent, stop:0.3 rgba(201,162,39,0.18), stop:0.5 rgba(201,162,39,0.28),
    stop:0.7 rgba(201,162,39,0.18), stop:1 transparent); }
QToolButton#ModuleTile { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 rgba(30,36,50,0.7),stop:1 rgba(20,25,36,0.85));
    border:1px solid rgba(255,255,255,0.07); border-radius:10px;
    padding:16px; color:#c8d4dc; font-family:'Segoe UI',system-ui; }
QToolButton#ModuleTile:hover { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 rgba(44,52,72,0.8),stop:1 rgba(32,38,56,0.92));
    border:1px solid rgba(201,162,39,0.30); color:#e8edf2; }
QToolButton#ModuleTile:pressed { background:rgba(20,25,36,0.95); border:1px solid rgba(201,162,39,0.48); }
QPushButton { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
    border-radius:6px; color:#c8d4dc; padding:6px 14px; font-family:'Segoe UI',system-ui; font-size:12px; }
QPushButton:hover { background:rgba(255,255,255,0.10); border:1px solid rgba(201,162,39,0.35); color:#e8edf2; }
QPushButton:pressed { background:rgba(0,0,0,0.2); }
QPushButton:disabled { color:#3a4a56; border-color:rgba(255,255,255,0.04); }
QPushButton#ThemeToggle { background:rgba(201,162,39,0.08); border:1px solid rgba(201,162,39,0.25);
    color:#c9a227; padding:5px 14px; border-radius:6px; font-size:11px; font-weight:700; }
QPushButton#ThemeToggle:hover { background:rgba(201,162,39,0.15); border-color:rgba(201,162,39,0.45); }
QLineEdit { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);
    border-radius:6px; color:#c8d4dc; padding:5px 10px; font-size:12px;
    selection-background-color:rgba(201,162,39,0.30); }
QLineEdit:focus { border-color:rgba(201,162,39,0.45); }
QTextEdit { background:#0c1016; border:1px solid rgba(255,255,255,0.07); border-radius:6px;
    color:#b8c4cc; font-family:'Courier New',monospace; font-size:11px;
    selection-background-color:rgba(201,162,39,0.28); }
QFrame#ReportPanelCard { background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #181e2c,stop:1 #111520);
    border:1px solid rgba(255,255,255,0.06); border-radius:10px; }
QFrame#ReportPanelHeader { background:rgba(255,255,255,0.03); border-bottom:1px solid rgba(255,255,255,0.07);
    border-top-left-radius:10px; border-top-right-radius:10px; }
QLabel#ReportPanelEyebrow { color:#c9a227; font-size:10px; font-weight:700; letter-spacing:1.2px; }
QLabel#ReportPanelHint    { color:#5a6a76; font-size:11px; }
QTextEdit#ReportTextView  { background:transparent; border:none; color:#a8b8c4;
    padding:10px 12px 12px 12px; selection-background-color:rgba(201,162,39,0.22); }
QTabWidget::pane { border:1px solid rgba(255,255,255,0.08); border-radius:6px; background:rgba(14,18,24,0.80); }
QTabBar { qproperty-elideMode:ElideRight; }
QTabBar::tab { background:rgba(255,255,255,0.04); color:#7a8a96; border:1px solid rgba(255,255,255,0.06);
    border-bottom:none; border-radius:5px 5px 0 0; padding:6px 12px; font-size:11px;
    margin-right:2px; min-width:60px; max-width:140px; }
QTabBar::tab:selected { background:rgba(201,162,39,0.10); color:#c9a227; border-color:rgba(201,162,39,0.25); }
QTabBar::tab:hover:!selected { background:rgba(255,255,255,0.07); color:#c8d4dc; }
QTableView, QTableWidget { background:#0c1016; alternate-background-color:#101418;
    border:1px solid rgba(255,255,255,0.07); border-radius:6px;
    gridline-color:rgba(255,255,255,0.05); color:#b8c4cc; font-size:11px;
    selection-background-color:rgba(201,162,39,0.25); selection-color:#ffffff; }
QHeaderView::section { background:#141820; color:#7a8a96; border:none;
    border-right:1px solid rgba(255,255,255,0.06); border-bottom:1px solid rgba(255,255,255,0.08);
    padding:5px 8px; font-size:10px; font-weight:700; letter-spacing:0.5px; }
QCheckBox { color:#b8c4cc; font-size:12px; spacing:6px; }
QCheckBox::indicator { width:14px; height:14px; background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.18); border-radius:3px; }
QCheckBox::indicator:checked { background:rgba(201,162,39,0.70); border-color:rgba(201,162,39,0.90); }
QScrollArea { border:none; background:transparent; }
QScrollBar:vertical, QScrollBar:horizontal { background:transparent; border:none; }
QScrollBar:vertical   { width:10px;  margin:6px 2px; border-radius:5px; }
QScrollBar:horizontal { height:10px; margin:2px 6px; border-radius:5px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background:rgba(201,162,39,0.20); border-radius:5px; min-height:40px; min-width:40px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background:rgba(201,162,39,0.38); }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:transparent; }
QGroupBox { border:1px solid rgba(255,255,255,0.08); border-radius:8px; margin-top:16px;
    font-size:11px; font-weight:700; color:#7a8a96; letter-spacing:0.8px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
QSplitter::handle { background:rgba(255,255,255,0.06); width:1px; }
QComboBox { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
    border-radius:5px; color:#c8d4dc; padding:4px 8px; font-size:12px; min-height:22px; }
QComboBox:hover { border-color:rgba(201,162,39,0.35); }
QComboBox::drop-down { border:none; width:20px; }
QComboBox::down-arrow { image:none; border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:5px solid #7a8a96; margin-right:6px; }
QComboBox QAbstractItemView { background:#1a1f2e; border:1px solid rgba(255,255,255,0.10);
    color:#c8d4dc; selection-background-color:rgba(201,162,39,0.25); selection-color:#ffffff;
    outline:none; padding:2px; }
QSpinBox, QDoubleSpinBox { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);
    border-radius:5px; color:#c8d4dc; padding:3px 6px; font-size:12px; min-height:22px; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color:rgba(201,162,39,0.45); }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width:16px; border:none; background:rgba(255,255,255,0.04); }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background:rgba(201,162,39,0.15); }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image:none;
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-bottom:4px solid #7a8a96; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image:none;
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:4px solid #7a8a96; }
QProgressBar { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10);
    border-radius:5px; color:#c8d4dc; font-size:11px; text-align:center; min-height:18px; }
QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 rgba(201,162,39,0.60), stop:1 rgba(201,162,39,0.35)); border-radius:4px; }
QToolTip { background:#1a1f2e; border:1px solid rgba(201,162,39,0.25);
    color:#c8d4dc; padding:4px 8px; font-size:11px; border-radius:4px; }
QListWidget { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:6px; color:#b8c4cc; font-size:11px; outline:none;
    selection-background-color:rgba(201,162,39,0.25); }
QListWidget::item { padding:3px 6px; }
QListWidget::item:selected { background:rgba(201,162,39,0.20); color:#e8edf2; }
"""
    else:
        return """
QMainWindow, QDialog { background:#f0f3f6; }
QLabel#AppTitle { color:#1a3a6a; font-size:28px; font-weight:700;
    letter-spacing:-0.5px; font-family:'Segoe UI','SF Pro Display',system-ui; }
QLabel#AppSubtitle { color:#8090a0; font-size:12px; letter-spacing:1.5px; font-family:'Segoe UI',monospace; }
QLabel#SignatureLine { color:#8090a0; font-size:11px; font-family:'Courier New',monospace; letter-spacing:0.2px; }
QLabel#SectionTitle { color:#1a2028; font-size:13px; font-weight:700; font-family:'Segoe UI Semibold',system-ui; }
QLabel#SectionHint  { color:#6a7a88; font-size:11px; font-family:'Segoe UI',system-ui; }
QLabel#CategoryBadge { color:#1a5fa8; font-size:10px; font-weight:700; letter-spacing:1.8px;
    padding:3px 8px; background:rgba(26,95,168,0.08); border:1px solid rgba(26,95,168,0.22); border-radius:4px; }
QLabel#StatusBadge { color:#1a7a4a; font-size:10px; font-weight:700; letter-spacing:1.5px;
    padding:3px 8px; background:rgba(26,122,74,0.08); border:1px solid rgba(26,122,74,0.22); border-radius:4px; }
QFrame#PanelCard   { background:#ffffff; border:1px solid #d6dce3; border-radius:12px; }
QFrame#HeaderPanel { background:#ffffff; border:1px solid #c8d4e0; border-radius:10px; }
QFrame#Divider { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 transparent, stop:0.3 rgba(26,95,168,0.15), stop:0.5 rgba(26,95,168,0.28),
    stop:0.7 rgba(26,95,168,0.15), stop:1 transparent); }
QToolButton#ModuleTile { background:#ffffff; border:1px solid #d6dce3; border-radius:10px;
    padding:16px; color:#3a4a58; font-family:'Segoe UI',system-ui; }
QToolButton#ModuleTile:hover  { background:#f4f8fc; border:1px solid #1a5fa8; color:#1a2028; }
QToolButton#ModuleTile:pressed{ background:#e8eef4; border:1px solid #1a4a88; }
QPushButton { background:#f0f3f6; border:1px solid #c8d4e0; border-radius:6px;
    color:#3a4a58; padding:6px 14px; font-size:12px; }
QPushButton:hover    { background:#e4ecf4; border-color:#1a5fa8; color:#1a2028; }
QPushButton:pressed  { background:#d8e4f0; }
QPushButton:disabled { color:#aab0b8; border-color:#d8dde4; }
QPushButton#ThemeToggle { background:rgba(26,95,168,0.07); border:1px solid rgba(26,95,168,0.25);
    color:#1a5fa8; padding:5px 14px; border-radius:6px; font-size:11px; font-weight:700; }
QPushButton#ThemeToggle:hover { background:rgba(26,95,168,0.14); border-color:rgba(26,95,168,0.45); }
QLineEdit { background:#ffffff; border:1px solid #c8d4e0; border-radius:6px;
    color:#1a2028; padding:5px 10px; font-size:12px; selection-background-color:rgba(26,95,168,0.25); }
QLineEdit:focus { border-color:#1a5fa8; }
QTextEdit { background:#fafbfc; border:1px solid #d6dce3; border-radius:6px; color:#2a3440;
    font-family:'Courier New',monospace; font-size:11px; selection-background-color:rgba(26,95,168,0.20); }
QFrame#ReportPanelCard   { background:#ffffff; border:1px solid #d6dce3; border-radius:10px; }
QFrame#ReportPanelHeader { background:#f6f8fb; border-bottom:1px solid #e1e7ee;
    border-top-left-radius:10px; border-top-right-radius:10px; }
QLabel#ReportPanelEyebrow { color:#1a5fa8; font-size:10px; font-weight:700; letter-spacing:1.2px; }
QLabel#ReportPanelHint    { color:#50606f; font-size:11px; }
QTextEdit#ReportTextView  { background:transparent; border:none; color:#26323e;
    padding:10px 12px 12px 12px; selection-background-color:rgba(26,95,168,0.22); }
QTabWidget::pane { border:1px solid #d6dce3; border-radius:6px; background:#fafbfc; }
QTabBar { qproperty-elideMode:ElideRight; }
QTabBar::tab { background:#edf0f4; color:#6a7a88; border:1px solid #d6dce3;
    border-bottom:none; border-radius:5px 5px 0 0; padding:6px 12px; font-size:11px;
    margin-right:2px; min-width:60px; max-width:140px; }
QTabBar::tab:selected   { background:#ffffff; color:#1a5fa8; border-color:#c0ccd8; }
QTabBar::tab:hover:!selected { background:#f0f4f8; color:#1a2028; }
QTableView, QTableWidget { background:#ffffff; alternate-background-color:#f8fafc;
    border:1px solid #d6dce3; border-radius:6px; gridline-color:#e8edf2; color:#2a3440; font-size:11px;
    selection-background-color:rgba(26,95,168,0.25); selection-color:#1a2028; }
QHeaderView::section { background:#f0f3f6; color:#5a6a78; border:none;
    border-right:1px solid #d6dce3; border-bottom:1px solid #d6dce3;
    padding:5px 8px; font-size:10px; font-weight:700; letter-spacing:0.5px; }
QCheckBox { color:#3a4a58; font-size:12px; spacing:6px; }
QCheckBox::indicator { width:14px; height:14px; background:#ffffff; border:1px solid #b0bcc8; border-radius:3px; }
QCheckBox::indicator:checked { background:#1a5fa8; border-color:#1a5fa8; }
QScrollArea { border:none; background:transparent; }
QScrollBar:vertical, QScrollBar:horizontal { background:transparent; border:none; }
QScrollBar:vertical   { width:10px;  margin:6px 2px; }
QScrollBar:horizontal { height:10px; margin:2px 6px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background:#c0ccd8; border-radius:5px; min-height:40px; min-width:40px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background:#9aa8b8; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:transparent; }
QGroupBox { border:1px solid #d6dce3; border-radius:8px; margin-top:16px;
    font-size:11px; font-weight:700; color:#6a7a88; letter-spacing:0.8px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
QSplitter::handle { background:#d6dce3; width:1px; }
QComboBox { background:#ffffff; border:1px solid #c8d4e0; border-radius:5px;
    color:#1a2028; padding:4px 8px; font-size:12px; min-height:22px; }
QComboBox:hover { border-color:#1a5fa8; }
QComboBox::drop-down { border:none; width:20px; }
QComboBox::down-arrow { image:none; border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:5px solid #6a7a88; margin-right:6px; }
QComboBox QAbstractItemView { background:#ffffff; border:1px solid #c8d4e0;
    color:#1a2028; selection-background-color:rgba(26,95,168,0.20); selection-color:#1a2028;
    outline:none; padding:2px; }
QSpinBox, QDoubleSpinBox { background:#ffffff; border:1px solid #c8d4e0;
    border-radius:5px; color:#1a2028; padding:3px 6px; font-size:12px; min-height:22px; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color:#1a5fa8; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width:16px; border:none; background:#f0f3f6; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background:#dce4ee; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image:none;
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-bottom:4px solid #6a7a88; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image:none;
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:4px solid #6a7a88; }
QProgressBar { background:#e8edf2; border:1px solid #c8d4e0;
    border-radius:5px; color:#3a4a58; font-size:11px; text-align:center; min-height:18px; }
QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #1a5fa8, stop:1 #4a8fd8); border-radius:4px; }
QToolTip { background:#ffffff; border:1px solid #c8d4e0;
    color:#1a2028; padding:4px 8px; font-size:11px; border-radius:4px; }
QListWidget { background:#ffffff; border:1px solid #d6dce3;
    border-radius:6px; color:#2a3440; font-size:11px; outline:none;
    selection-background-color:rgba(26,95,168,0.20); }
QListWidget::item { padding:3px 6px; }
QListWidget::item:selected { background:rgba(26,95,168,0.15); color:#1a2028; }
"""
