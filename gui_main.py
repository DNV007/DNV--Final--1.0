"""
DNV Scientific Module
---------------------
Role:
    Constructs and manages the application's primary viewport: an 8-module
    workflow grid, a provenance-aware header, and a theme-toggle control.
    Delegates all child-window lifecycle management and responsive layout
    recomputation to explicit, named transformations.

Scientific Context:
    Transforms two independent observables — MainWindow width (px) and
    MainWindow height (px) — into a LayoutParams record that governs all
    spacing, margin, font-size, and grid-gap values.  The 8 workflow modules
    are ordered to reflect a canonical materials-informatics pipeline:
    ingestion → cleaning → transformation → visualization → correlation →
    feature ranking → descriptors → machine learning.

Invariants:
    - All margin, spacing, and font-size values are clamped to their declared
      [min, max] ranges for all positive viewport dimensions.
    - Grid column stretches are uniform (stretch = 1) across all 4 columns.
    - The open-window registry contains only currently visible QDialog instances.
    - Theme state is mutated exclusively through `theme.toggle()`; no other
      site writes to the theme singleton.

Assumptions:
    - `QApplication.primaryScreen()` returns a valid screen at startup.
    - `gui_tile.compute_display_params` and `gui_tile.TileDisplayParams` are
      available (refactored gui_tile contract).
    - All child-window constructors accept (SessionStore, QWidget) or
      (title, category, QWidget) as documented below.
    - `new_analysis_id()` is stable for the lifetime of a single MainWindow
      instance (one ID per session).

Failure Modes:
    - Missing child-window module (ImportError): fatal at import time, by design.
    - `primaryScreen()` returns None (headless environment): AttributeError at
      __init__; caller must ensure a display is available.
    - Child window constructor raises: propagates to the click callback;
      the open-window registry is not corrupted (append occurs after show).

Provenance:
    - Module: gui_main.py
    - Conforms to: DNV Scientific Module Standard v1.0
    - LayoutParams is the typed, immutable witness of each responsive update.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Final, List

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QCloseEvent, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QDialog, QFrame, QGraphicsDropShadowEffect,
    QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ing_provenance    import new_analysis_id
from gui_theme         import ThemeMode, theme, get_stylesheet
from gui_session       import SessionStore
from gui_tile          import ModuleTileButton
from gui_widgets       import make_divider, make_section_header, make_signature_widget
from gui_ingestion      import DataIngestionWindow
from gui_cleaning       import DataCleaningWindow
from gui_transformation import TransformationWindow
from gui_visualization  import VisualizationWindow
from gui_correlation       import CorrelationWindow
from gui_feature_ranking   import FeatureRankingWindow
from gui_descriptors       import DescriptorsWindow
from gui_ml                import MLWindow
from gui_placeholder    import PlaceholderWindow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application identity constants
# ---------------------------------------------------------------------------

APP_TITLE:    Final[str] = "DNV-1.0"
APP_SUBTITLE: Final[str] = "Materials Informatics Suite"

#: Number of tile columns in the module grid.
GRID_COLS: Final[int] = 4

#: Fraction of screen width used as the minimum window dimension [dimensionless].
SCREEN_MIN_FRACTION: Final[float] = 0.68

#: Minimum content body width as a fraction of screen width [dimensionless].
BODY_MIN_WIDTH_FRACTION: Final[float] = 0.72


# ---------------------------------------------------------------------------
# Layout scaling constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Outer margin [px], clamped range.
OUTER_MARGIN_MIN_PX: Final[int] = 12
OUTER_MARGIN_MAX_PX: Final[int] = 28

#: Outer vertical spacing [px], clamped range.
OUTER_SPACING_MIN_PX: Final[int] = 12
OUTER_SPACING_MAX_PX: Final[int] = 22

#: Header horizontal padding [px], clamped range.
HEADER_HPAD_MIN_PX: Final[int] = 14
HEADER_HPAD_MAX_PX: Final[int] = 30

#: Header vertical padding [px], clamped range.
HEADER_VPAD_MIN_PX: Final[int] = 12
HEADER_VPAD_MAX_PX: Final[int] = 24

#: Title font size [pt], clamped range.
TITLE_FONT_MIN_PT: Final[int] = 22
TITLE_FONT_MAX_PT: Final[int] = 38

#: Grid gap (horizontal and vertical) [px], clamped range.
GRID_GAP_MIN_PX: Final[int] = 12
GRID_GAP_MAX_PX: Final[int] = 24

#: Card inner padding [px], clamped range.
CARD_PAD_MIN_PX: Final[int] = 12
CARD_PAD_MAX_PX: Final[int] = 26

#: Scaling ratios [dimensionless or pt/px].
OUTER_MARGIN_WIDTH_RATIO:  Final[float] = 0.020
OUTER_SPACING_HEIGHT_RATIO: Final[float] = 0.016
HEADER_HPAD_WIDTH_RATIO:   Final[float] = 0.020
HEADER_VPAD_HEIGHT_RATIO:  Final[float] = 0.018
TITLE_FONT_WIDTH_RATIO:    Final[float] = 1.0 / 32.0
GRID_GAP_WIDTH_RATIO:      Final[float] = 0.018
CARD_PAD_WIDTH_RATIO:      Final[float] = 0.018

#: Minimum effective viewport dimensions used when the window is not yet shown.
DEFAULT_VIEWPORT_WIDTH_PX:  Final[int] = 1400
DEFAULT_VIEWPORT_HEIGHT_PX: Final[int] = 900

#: Header shadow geometry [px / alpha].
HEADER_SHADOW_BLUR_PX:    Final[int] = 28
HEADER_SHADOW_YOFFSET_PX: Final[int] = 10
HEADER_SHADOW_ALPHA:      Final[int] = 70

#: Card shadow geometry [px / alpha].
CARD_SHADOW_BLUR_PX:    Final[int] = 40
CARD_SHADOW_YOFFSET_PX: Final[int] = 16
CARD_SHADOW_ALPHA:      Final[int] = 90

#: Font family used for the application title.
TITLE_FONT_FAMILY: Final[str] = "Segoe UI"

#: Fixed spacing between header row widgets [px].
HEADER_ROW_SPACING_PX: Final[int] = 16

#: Fixed spacing within the header layout [px].
HEADER_INNER_SPACING_PX: Final[int] = 8

#: Fixed spacing within the body layout [px].
BODY_SPACING_PX: Final[int] = 20

#: Fixed card inner layout spacing [px].
CARD_INNER_SPACING_PX: Final[int] = 18


# ---------------------------------------------------------------------------
# Module registry — defines the 8 canonical workflow stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModuleDescriptor:
    """
    Immutable descriptor for one workflow module tile.

    Fields
    ------
    title:      Primary tile label.
    subtitle:   Secondary tile label (semantic summary).
    category:   Semantic category tag used for routing and provenance.
    scheme:     Colour scheme key passed to ModuleTileButton.
    icon_file:  Asset filename relative to ASSETS_DIR.
    """
    title:     str
    subtitle:  str
    category:  str
    scheme:    str
    icon_file: str


#: Ordered pipeline stages; index encodes canonical processing order.
WORKFLOW_MODULES: Final[tuple[ModuleDescriptor, ...]] = (
    ModuleDescriptor("Data Ingestion",   "load · declare · audit",   "DATA INPUT",   "steel",   "loadfile.png"),
    ModuleDescriptor("Data Cleaning",    "condition · constrain",    "DATA QUALITY", "rose",    "missingvalues.png"),
    ModuleDescriptor("Transformation",   "derive · map · document",  "DATA PREP",    "violet",  "Transform.png"),
    ModuleDescriptor("Visualization",    "plots · diagnostics",      "ANALYTICS",    "amber",   "visualization.png"),
    ModuleDescriptor("Correlation",      "marginal structure",       "STATISTICS",   "cyan",    "correlation.png"),
    ModuleDescriptor("Feature Ranking",  "attribution · persistence","ML PREP",      "emerald", "featureimportance.png"),
    ModuleDescriptor("Descriptors",      "contracts · provenance",   "FEATURE ENG",  "amber",   "descriptors.png"),
    ModuleDescriptor("Machine Learning", "train · validate · audit", "MODELLING",    "steel",   "machinelearning.png"),
)


# ---------------------------------------------------------------------------
# Layout parameters — pure data, no Qt coupling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutParams:
    """
    All layout parameters derived from a single (viewport_width, viewport_height)
    observation.  Every field carries an explicit unit in its name.

    Units
    -----
    *_px  : pixels  (integer)
    *_pt  : points  (integer)
    """
    outer_margin_px:   int   # [px]  applied to all four sides of the root layout
    outer_spacing_px:  int   # [px]  vertical spacing between root layout children
    header_hpad_px:    int   # [px]  header left/right padding
    header_vpad_px:    int   # [px]  header top/bottom padding
    title_font_pt:     int   # [pt]  application title label font size
    grid_gap_px:       int   # [px]  horizontal and vertical grid spacing
    card_pad_px:       int   # [px]  card inner padding (all sides)


def compute_layout_params(
    viewport_width_px: int,
    viewport_height_px: int,
) -> LayoutParams:
    """
    Pure function: (viewport_width, viewport_height) → LayoutParams.

    Each derived quantity is a linear function of viewport dimensions clamped
    to its declared invariant range.  No Qt calls are made.

    Parameters
    ----------
    viewport_width_px:
        Current window width [px].  Must be > 0.
    viewport_height_px:
        Current window height [px].  Must be > 0.
    """
    def _scale_clamp(value: float, lo: int, hi: int) -> int:
        return max(lo, min(int(value), hi))

    return LayoutParams(
        outer_margin_px  = _scale_clamp(viewport_width_px  * OUTER_MARGIN_WIDTH_RATIO,  OUTER_MARGIN_MIN_PX,  OUTER_MARGIN_MAX_PX),
        outer_spacing_px = _scale_clamp(viewport_height_px * OUTER_SPACING_HEIGHT_RATIO, OUTER_SPACING_MIN_PX, OUTER_SPACING_MAX_PX),
        header_hpad_px   = _scale_clamp(viewport_width_px  * HEADER_HPAD_WIDTH_RATIO,   HEADER_HPAD_MIN_PX,   HEADER_HPAD_MAX_PX),
        header_vpad_px   = _scale_clamp(viewport_height_px * HEADER_VPAD_HEIGHT_RATIO,  HEADER_VPAD_MIN_PX,   HEADER_VPAD_MAX_PX),
        title_font_pt    = _scale_clamp(viewport_width_px  * TITLE_FONT_WIDTH_RATIO,    TITLE_FONT_MIN_PT,    TITLE_FONT_MAX_PT),
        grid_gap_px      = _scale_clamp(viewport_width_px  * GRID_GAP_WIDTH_RATIO,      GRID_GAP_MIN_PX,      GRID_GAP_MAX_PX),
        card_pad_px      = _scale_clamp(viewport_width_px  * CARD_PAD_WIDTH_RATIO,      CARD_PAD_MIN_PX,      CARD_PAD_MAX_PX),
    )


# ---------------------------------------------------------------------------
# Child-window factory — isolated, named, side-effect-free routing
# ---------------------------------------------------------------------------

def _build_child_window(
    descriptor: ModuleDescriptor,
    store: SessionStore,
    parent: QMainWindow,
) -> QDialog:
    """
    Route a ModuleDescriptor to its concrete child-window constructor.

    This is the single dispatch point for module → window mapping.
    Adding a new module requires only a new branch here and a new entry
    in WORKFLOW_MODULES — no other site changes.

    Parameters
    ----------
    descriptor:
        The module being opened.
    store:
        Shared session store passed to data-aware windows.
    parent:
        Qt parent widget (MainWindow).
    """
    title = descriptor.title
    if title == "Data Ingestion":
        return DataIngestionWindow(store, parent)
    if title == "Data Cleaning":
        return DataCleaningWindow(store, parent)
    if title == "Transformation":
        return TransformationWindow(store, parent)
    if title == "Visualization":
        return VisualizationWindow(store, parent)
    if title == "Correlation":
        return CorrelationWindow(store, parent)
    if title == "Feature Ranking":
        return FeatureRankingWindow(store, parent)
    if title == "Descriptors":
        return DescriptorsWindow(store, parent)
    if title == "Machine Learning":
        return MLWindow(store, parent)
    return PlaceholderWindow(title, descriptor.category, parent)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    DNV-1.0 primary application window.

    Composes a provenance-aware header, a scrollable 4-column module grid,
    and a theme-toggle control.  All spacing and sizing is derived from
    `compute_layout_params`; no arithmetic appears in the widget body.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE}  —  {APP_SUBTITLE}")

        screen = QApplication.primaryScreen().geometry()
        min_w = int(screen.width()  * SCREEN_MIN_FRACTION)
        min_h = int(screen.height() * SCREEN_MIN_FRACTION)
        self.setMinimumSize(min_w, min_h)

        self._store:        SessionStore          = SessionStore()
        self._open_windows: List[QDialog]         = []
        self._tile_buttons: List[ModuleTileButton] = []

        root = QWidget()
        self._outer = QVBoxLayout(root)

        self._build_header()
        self._outer.addWidget(make_divider())
        self._build_body(screen_width_px=screen.width())

        self.setCentralWidget(root)
        self._apply_layout_params(
            compute_layout_params(DEFAULT_VIEWPORT_WIDTH_PX, DEFAULT_VIEWPORT_HEIGHT_PX)
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        hdr = QFrame()
        hdr.setObjectName("HeaderPanel")
        self._hdr_lay = QVBoxLayout(hdr)
        self._hdr_lay.setSpacing(HEADER_INNER_SPACING_PX)

        row = QHBoxLayout()
        row.setSpacing(HEADER_ROW_SPACING_PX)

        self._title_lbl = QLabel(APP_TITLE)
        self._title_lbl.setObjectName("AppTitle")
        row.addWidget(self._title_lbl)

        sub = QLabel(APP_SUBTITLE.upper())
        sub.setObjectName("AppSubtitle")
        row.addWidget(sub)
        row.addStretch()

        self._theme_btn = QPushButton(
            "Light mode" if theme.is_dark() else "Dark mode"
        )
        self._theme_btn.setObjectName("ThemeToggle")
        self._theme_btn.setCursor(Qt.PointingHandCursor)
        self._theme_btn.clicked.connect(self._toggle_theme)
        row.addWidget(self._theme_btn)

        self._hdr_lay.addLayout(row)
        self._hdr_lay.addWidget(make_signature_widget(new_analysis_id()))

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(HEADER_SHADOW_BLUR_PX)
        shadow.setXOffset(0)
        shadow.setYOffset(HEADER_SHADOW_YOFFSET_PX)
        shadow.setColor(QColor(0, 0, 0, HEADER_SHADOW_ALPHA))
        hdr.setGraphicsEffect(shadow)

        self._outer.addWidget(hdr)

    # ------------------------------------------------------------------
    # Scrollable module grid
    # ------------------------------------------------------------------

    def _build_body(self, screen_width_px: int) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(BODY_SPACING_PX)
        body_layout.addWidget(make_section_header(
            "Workflow modules",
            "Eight stages for contract-aware, reproducible materials informatics: "
            "data ingestion → cleaning → transformation → visualization → "
            "correlation → feature ranking → descriptors → ML.",
            "8 MODULES",
        ))

        card = QFrame()
        card.setObjectName("PanelCard")
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(CARD_SHADOW_BLUR_PX)
        card_shadow.setXOffset(0)
        card_shadow.setYOffset(CARD_SHADOW_YOFFSET_PX)
        card_shadow.setColor(QColor(0, 0, 0, CARD_SHADOW_ALPHA))
        card.setGraphicsEffect(card_shadow)

        self._card_lay = QVBoxLayout(card)
        self._card_lay.setSpacing(CARD_INNER_SPACING_PX)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(GRID_GAP_MIN_PX)
        self._grid.setVerticalSpacing(GRID_GAP_MIN_PX)

        for i, descriptor in enumerate(WORKFLOW_MODULES):
            row, col = divmod(i, GRID_COLS)
            btn = ModuleTileButton(
                title       = descriptor.title,
                subtitle    = descriptor.subtitle,
                category    = descriptor.category,
                icon_scheme = descriptor.scheme,
                icon_file   = descriptor.icon_file,
                callback    = self._make_callback(descriptor),
                parent      = self,
            )
            self._grid.addWidget(btn, row, col)
            self._tile_buttons.append(btn)

        for col in range(GRID_COLS):
            self._grid.setColumnStretch(col, 1)

        self._card_lay.addLayout(self._grid)
        body_layout.addWidget(card)
        body.setMinimumWidth(int(screen_width_px * BODY_MIN_WIDTH_FRACTION))
        scroll.setWidget(body)
        self._outer.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Callback factory
    # ------------------------------------------------------------------

    def _make_callback(self, descriptor: ModuleDescriptor) -> Callable[[], None]:
        """
        Return a zero-argument callable that opens the child window for
        `descriptor`.  Captures descriptor and self; no mutable closure state.
        """
        def _open() -> None:
            # Prune stale references before appending
            self._open_windows = [d for d in self._open_windows if d.isVisible()]
            window = _build_child_window(descriptor, self._store, self)
            window.setModal(False)
            window.show()
            self._open_windows.append(window)
        return _open

    # ------------------------------------------------------------------
    # Layout parameter application
    # ------------------------------------------------------------------

    def _apply_layout_params(self, params: LayoutParams) -> None:
        """
        Apply a `LayoutParams` observation to all layout managers.

        This is the single point of contact between the pure scaling
        computation and Qt layout state — no arithmetic appears elsewhere
        in the widget body.
        """
        self._outer.setContentsMargins(
            params.outer_margin_px, params.outer_margin_px,
            params.outer_margin_px, params.outer_margin_px,
        )
        self._outer.setSpacing(params.outer_spacing_px)

        self._hdr_lay.setContentsMargins(
            params.header_hpad_px, params.header_vpad_px,
            params.header_hpad_px, params.header_vpad_px,
        )

        title_font = QFont(TITLE_FONT_FAMILY, params.title_font_pt)
        title_font.setWeight(QFont.Bold)
        self._title_lbl.setFont(title_font)

        self._grid.setHorizontalSpacing(params.grid_gap_px)
        self._grid.setVerticalSpacing(params.grid_gap_px)

        self._card_lay.setContentsMargins(
            params.card_pad_px, params.card_pad_px,
            params.card_pad_px, params.card_pad_px,
        )

        for btn in self._tile_buttons:
            btn.adapt_to_size(self.width() or DEFAULT_VIEWPORT_WIDTH_PX,
                              self.height() or DEFAULT_VIEWPORT_HEIGHT_PX)

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_layout_params(
            compute_layout_params(
                max(self.width(),  1),
                max(self.height(), 1),
            )
        )

    # ------------------------------------------------------------------
    # Theme toggle
    # ------------------------------------------------------------------

    def _toggle_theme(self) -> None:
        mode = theme.toggle()
        self._theme_btn.setText(
            "Light mode" if mode == ThemeMode.DARK else "Dark mode"
        )
        QApplication.instance().setStyleSheet(get_stylesheet())
        self._apply_layout_params(
            compute_layout_params(
                max(self.width(),  1),
                max(self.height(), 1),
            )
        )
        self.update()
        QApplication.processEvents()
        log.info("Theme toggled: %s", mode.value.upper())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        for window in list(self._open_windows):
            if window.isVisible():
                window.close()
        super().closeEvent(event)
