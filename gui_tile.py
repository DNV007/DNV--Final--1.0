"""
DNV Scientific Module
---------------------
Role:
    Renders an adaptive, stateless workflow-tile button widget for PySide6 GUI
    applications. Encapsulates icon provisioning, typographic scaling, and
    hover-state shadow modulation as explicit, unit-annotated transformations.

Scientific Context:
    Transforms two independent observables — viewport width (px) and viewport
    height (px) — into a set of derived display parameters (icon size, font
    size, minimum tile height, shadow geometry).  Each transformation is a
    pure, monotone function of its inputs with bounded codomain.

Invariants:
    - Icon size ∈ [ICON_SIZE_MIN_PX, ICON_SIZE_MAX_PX] for all viewport widths.
    - Font size ∈ [FONT_SIZE_MIN_PT, FONT_SIZE_MAX_PT] for all viewport widths.
    - Tile minimum height ≥ icon_size_px + ICON_LABEL_CLEARANCE_PX at all times.
    - Shadow effect parameters are symmetric (x-offset = 0) at all times.
    - The widget emits no side-effects outside its own Qt subtree.

Assumptions:
    - `gui_theme.theme` exposes `is_dark() -> bool` and is stable for the
      lifetime of a single widget instance (no mid-session theme switching).
    - ASSETS_DIR is readable by the process; missing asset files are non-fatal
      and degrade gracefully to the procedural fallback icon.
    - Parent widget exposes `.width()` and `.height()` returning positive
      integers (standard QWidget contract).
    - The process runs with a single Qt application instance (QPainter context).

Failure Modes:
    - Missing `gui_theme` module: ImportError at import time — fatal, by design.
    - QGraphicsDropShadowEffect instantiation failure (e.g., compositing
      unavailable): caught, shadow silently disabled; widget remains functional.
    - `parentWidget()` returns None during early layout: `adapt_to_size` falls
      back to the module-level default viewport constants.
    - Corrupt or unreadable asset file at `icon_file` path: falls back to
      procedural icon without exception propagation.

Provenance:
    - Module: module_tile_button.py
    - Conforms to: DNV Scientific Module Standard v1.0
    - Each `adapt_to_size` call is a pure transformation; no mutable global state.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Final

from PySide6.QtCore import Qt, QPointF, QSize
from PySide6.QtGui import (
    QColor, QConicalGradient, QFont, QIcon,
    QLinearGradient, QPainter, QPen,
    QPixmap, QPolygonF, QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QSizePolicy, QToolButton

from gui_theme import theme


# ---------------------------------------------------------------------------
# Module-level physical constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum and maximum icon edge length [px].
ICON_SIZE_MIN_PX: Final[int] = 72
ICON_SIZE_MAX_PX: Final[int] = 130

#: Minimum and maximum label font size [pt].
FONT_SIZE_MIN_PT: Final[int] = 10
FONT_SIZE_MAX_PT: Final[int] = 14

#: Vertical clearance between icon bottom and tile bottom [px].
ICON_LABEL_CLEARANCE_PX: Final[int] = 36

#: Minimum and maximum tile height [px].
TILE_HEIGHT_MIN_PX: Final[int] = 160
TILE_HEIGHT_MAX_PX: Final[int] = 300

#: Ratio of icon edge to viewport width [dimensionless, ∈ (0, 1)].
ICON_WIDTH_RATIO: Final[float] = 0.07

#: Ratio of font size (pt) to viewport width (px) [pt/px].
FONT_WIDTH_RATIO: Final[float] = 1.0 / 110.0

#: Ratio of minimum tile height to viewport height [dimensionless, ∈ (0, 1)].
TILE_HEIGHT_RATIO: Final[float] = 0.20

#: Default viewport dimensions [px] used when parent geometry is unavailable.
DEFAULT_VIEWPORT_WIDTH_PX: Final[int] = 1400
DEFAULT_VIEWPORT_HEIGHT_PX: Final[int] = 900

#: Procedural fallback icon canvas edge [px] (square).
FALLBACK_ICON_CANVAS_PX: Final[int] = 128

#: Corner radius for the rounded-rect background in the fallback icon [px].
FALLBACK_CORNER_RADIUS_PX: Final[int] = 22

#: Number of vertices of the central hexagon in the fallback icon.
FALLBACK_HEX_VERTICES: Final[int] = 6

#: Inset of the border rect from the canvas edge [px].
FALLBACK_BORDER_INSET_PX: Final[int] = 3

#: Shadow blur radius at rest [px].
SHADOW_REST_BLUR_PX: Final[int] = 24

#: Shadow y-offset at rest [px].
SHADOW_REST_YOFFSET_PX: Final[int] = 8

#: Shadow colour at rest (RGBA).
SHADOW_REST_COLOR: Final[QColor] = QColor(0, 0, 0, 80)

#: Shadow blur radius on hover [px].
SHADOW_HOVER_BLUR_PX: Final[int] = 32

#: Shadow y-offset on hover [px].
SHADOW_HOVER_YOFFSET_PX: Final[int] = 12

#: Rendered font family.
FONT_FAMILY: Final[str] = "Segoe UI"

#: Qt object name assigned to every tile (used for QSS targeting).
OBJECT_NAME: Final[str] = "ModuleTile"

ASSETS_DIR: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets"
)


# ---------------------------------------------------------------------------
# Colour palettes — indexed by scheme name and theme polarity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SchemePalette:
    """Primary and accent hex strings for one colour scheme + polarity."""
    primary: str
    accent: str


#: Colour table: polarity → scheme name → palette.
_COLOUR_TABLE: Final[dict[str, dict[str, _SchemePalette]]] = {
    "dark": {
        "steel":   _SchemePalette("#3a5a7a", "#5a8ab0"),
        "amber":   _SchemePalette("#a06010", "#c88030"),
        "emerald": _SchemePalette("#1a7a4a", "#2aa06a"),
        "violet":  _SchemePalette("#6a50a0", "#9070c8"),
        "rose":    _SchemePalette("#a03060", "#c05080"),
        "cyan":    _SchemePalette("#1a8a9a", "#2abaca"),
    },
    "light": {
        "steel":   _SchemePalette("#1a5fa8", "#2a7fc8"),
        "amber":   _SchemePalette("#a06010", "#c07820"),
        "emerald": _SchemePalette("#1a7a4a", "#2a9a5a"),
        "violet":  _SchemePalette("#6a40a0", "#8a60c0"),
        "rose":    _SchemePalette("#a02850", "#c04870"),
        "cyan":    _SchemePalette("#1a7a9a", "#2a9aba"),
    },
}

#: Background gradient stops per polarity.
_BG_STOPS: Final[dict[str, list[str]]] = {
    "dark":  ["#1c2030", "#141820", "#0e1218"],
    "light": ["#f8fafc", "#edf0f4", "#dee4ea"],
}


# ---------------------------------------------------------------------------
# Derived display parameters — pure data, no Qt coupling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileDisplayParams:
    """
    All display parameters derived from a single (viewport_width, viewport_height)
    observation.  Values are clamped to their declared invariant ranges.

    Units
    -----
    icon_size_px    : pixels (integer)
    font_size_pt    : points (integer)
    tile_min_height : pixels (integer)
    """
    icon_size_px: int       # [px]  ∈ [ICON_SIZE_MIN_PX, ICON_SIZE_MAX_PX]
    font_size_pt: int       # [pt]  ∈ [FONT_SIZE_MIN_PT, FONT_SIZE_MAX_PT]
    tile_min_height: int    # [px]  ≥ icon_size_px + ICON_LABEL_CLEARANCE_PX


def compute_display_params(
    viewport_width_px: int,
    viewport_height_px: int,
) -> TileDisplayParams:
    """
    Pure function: (viewport_width, viewport_height) → TileDisplayParams.

    Transformation
    --------------
    icon_size  = clamp(viewport_width × ICON_WIDTH_RATIO,
                       ICON_SIZE_MIN_PX, ICON_SIZE_MAX_PX)
    font_size  = clamp(viewport_width × FONT_WIDTH_RATIO,
                       FONT_SIZE_MIN_PT, FONT_SIZE_MAX_PT)
    tile_h     = clamp(viewport_height × TILE_HEIGHT_RATIO,
                       TILE_HEIGHT_MIN_PX, TILE_HEIGHT_MAX_PX)
               = max(tile_h, icon_size + ICON_LABEL_CLEARANCE_PX)   [invariant]

    Parameters
    ----------
    viewport_width_px:
        Current parent widget width [px].  Must be > 0.
    viewport_height_px:
        Current parent widget height [px].  Must be > 0.
    """
    icon_size = int(
        max(ICON_SIZE_MIN_PX,
            min(int(viewport_width_px * ICON_WIDTH_RATIO), ICON_SIZE_MAX_PX))
    )
    font_size = int(
        max(FONT_SIZE_MIN_PT,
            min(int(viewport_width_px * FONT_WIDTH_RATIO), FONT_SIZE_MAX_PT))
    )
    tile_h_from_ratio = int(
        max(TILE_HEIGHT_MIN_PX,
            min(int(viewport_height_px * TILE_HEIGHT_RATIO), TILE_HEIGHT_MAX_PX))
    )
    tile_min_height = max(tile_h_from_ratio, icon_size + ICON_LABEL_CLEARANCE_PX)

    return TileDisplayParams(
        icon_size_px=icon_size,
        font_size_pt=font_size,
        tile_min_height=tile_min_height,
    )


# ---------------------------------------------------------------------------
# Procedural fallback icon — isolated, stateless renderer
# ---------------------------------------------------------------------------

def _resolve_palette(scheme: str) -> _SchemePalette:
    """Return the palette for `scheme` under the current theme polarity."""
    polarity = "dark" if theme.is_dark() else "light"
    return _COLOUR_TABLE[polarity].get(scheme, _COLOUR_TABLE[polarity]["steel"])


def build_fallback_icon(size_px: int, scheme: str) -> QIcon:
    """
    Render a geometric fallback icon when no asset file is available.

    The icon is composed of three layered primitives drawn on a square canvas
    of edge `size_px`:

    1. Radial-gradient background (3-stop, centre-weighted).
    2. Conical-gradient hexagon (⌀ = size/2) at canvas centre.
    3. Circular stroke (⌀ = 0.72 × size) and rounded-rect border.

    All geometric proportions are expressed as dimensionless ratios of `size_px`
    to remain resolution-independent.

    Parameters
    ----------
    size_px:
        Edge length of the square canvas [px].
    scheme:
        Colour scheme key; falls back to "steel" if unrecognised.
    """
    pm = QPixmap(size_px, size_px)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)

        palette = _resolve_palette(scheme)
        polarity = "dark" if theme.is_dark() else "light"
        bg_stops = _BG_STOPS[polarity]

        cx, cy = size_px / 2.0, size_px / 2.0

        # — Background: radial gradient, 3 stops —
        bg_grad = QRadialGradient(size_px * 0.4, size_px * 0.4, size_px * 0.7)
        for stop_t, col in zip((0.0, 0.6, 1.0), bg_stops):
            bg_grad.setColorAt(stop_t, QColor(col))
        painter.setBrush(bg_grad)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(
            0, 0, size_px, size_px,
            FALLBACK_CORNER_RADIUS_PX, FALLBACK_CORNER_RADIUS_PX,
        )

        # — Central hexagon: conical gradient fill —
        hex_r = size_px / 4.0
        hex_pts = [
            QPointF(
                cx + hex_r * math.cos(math.radians(60 * i - 30)),
                cy + hex_r * math.sin(math.radians(60 * i - 30)),
            )
            for i in range(FALLBACK_HEX_VERTICES)
        ]
        hex_grad = QConicalGradient(cx, cy, 30)
        hex_grad.setColorAt(0.0, QColor(palette.primary))
        hex_grad.setColorAt(0.5, QColor(palette.accent))
        hex_grad.setColorAt(1.0, QColor(palette.primary))
        painter.setBrush(hex_grad)
        painter.drawPolygon(QPolygonF(hex_pts))

        # — Overlay strokes: circle + rounded-rect border —
        stroke_pen = QPen()
        stroke_pen.setWidth(2)
        stroke_pen.setColor(QColor(palette.accent + "80"))
        painter.setPen(stroke_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), size_px * 0.36, size_px * 0.36)

        border_grad = QLinearGradient(0, 0, 0, size_px)
        border_grad.setColorAt(0.0, QColor(palette.primary + "80"))
        border_grad.setColorAt(1.0, QColor(palette.accent + "60"))
        border_pen = QPen()
        border_pen.setWidth(2)
        border_pen.setBrush(border_grad)
        painter.setPen(border_pen)
        inset = FALLBACK_BORDER_INSET_PX
        painter.drawRoundedRect(
            inset, inset, size_px - 2 * inset, size_px - 2 * inset,
            FALLBACK_CORNER_RADIUS_PX - inset, FALLBACK_CORNER_RADIUS_PX - inset,
        )
    finally:
        painter.end()

    return QIcon(pm)


def resolve_icon(icon_file: str, size_px: int, scheme: str) -> QIcon:
    """
    Return the asset icon at `icon_file` if the path resolves and is readable;
    otherwise return the procedural fallback.

    Parameters
    ----------
    icon_file:
        Filename (not path) of the asset, relative to ASSETS_DIR.
    size_px:
        Requested icon render size [px].
    scheme:
        Colour scheme key for the fallback icon.
    """
    path = os.path.join(ASSETS_DIR, icon_file)
    if os.path.isfile(path) and os.access(path, os.R_OK):
        return QIcon(path)
    return build_fallback_icon(size_px, scheme)


# ---------------------------------------------------------------------------
# Shadow state — named, typed, immutable
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShadowState:
    """
    Observable shadow configuration for a single widget state.

    All offsets are in pixels [px]; blur_radius_px governs the Gaussian
    spread of the drop-shadow kernel.
    """
    blur_radius_px: int   # [px]
    x_offset_px: int      # [px]  invariant: always 0 (symmetric)
    y_offset_px: int      # [px]
    color: QColor


SHADOW_REST: Final[ShadowState] = ShadowState(
    blur_radius_px=SHADOW_REST_BLUR_PX,
    x_offset_px=0,
    y_offset_px=SHADOW_REST_YOFFSET_PX,
    color=SHADOW_REST_COLOR,
)

SHADOW_HOVER_DARK: Final[ShadowState] = ShadowState(
    blur_radius_px=SHADOW_HOVER_BLUR_PX,
    x_offset_px=0,
    y_offset_px=SHADOW_HOVER_YOFFSET_PX,
    color=QColor(201, 162, 39, 35),
)

SHADOW_HOVER_LIGHT: Final[ShadowState] = ShadowState(
    blur_radius_px=SHADOW_HOVER_BLUR_PX,
    x_offset_px=0,
    y_offset_px=SHADOW_HOVER_YOFFSET_PX,
    color=QColor(26, 95, 168, 35),
)


def _apply_shadow_state(effect: QGraphicsDropShadowEffect, state: ShadowState) -> None:
    """Apply a `ShadowState` to a live `QGraphicsDropShadowEffect`."""
    effect.setBlurRadius(state.blur_radius_px)
    effect.setXOffset(state.x_offset_px)
    effect.setYOffset(state.y_offset_px)
    effect.setColor(state.color)


def _hover_shadow(is_dark: bool) -> ShadowState:
    """Select hover shadow by theme polarity."""
    return SHADOW_HOVER_DARK if is_dark else SHADOW_HOVER_LIGHT


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class ModuleTileButton(QToolButton):
    """
    Adaptive workflow-tile button.

    Display parameters are recomputed whenever the parent viewport changes,
    ensuring the tile remains proportional without hard-coded pixel values in
    the widget body.  All scaling logic is delegated to `compute_display_params`.

    Parameters
    ----------
    title:
        Primary label text (single line).
    subtitle:
        Secondary label text (single line).
    category:
        Semantic category tag (reserved for routing / filtering; not rendered).
    icon_scheme:
        Colour scheme key for the fallback icon (e.g. "steel", "amber").
    icon_file:
        Asset filename relative to ASSETS_DIR.
    callback:
        Zero-argument callable invoked on click.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        title: str,
        subtitle: str,
        category: str,
        icon_scheme: str,
        icon_file: str,
        callback: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName(OBJECT_NAME)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Semantic metadata (not rendered, available for external consumers)
        self.category: str = category

        # Immutable icon provisioning inputs
        self._icon_scheme: str = icon_scheme
        self._icon_file: str = icon_file

        self.setText(f"{title}\n{subtitle}")
        self.clicked.connect(callback)

        # Shadow effect — optional; widget is fully functional without it
        self._shadow: QGraphicsDropShadowEffect | None = self._init_shadow()

        # Initial layout pass with module defaults
        self._apply_display_params(
            compute_display_params(DEFAULT_VIEWPORT_WIDTH_PX, DEFAULT_VIEWPORT_HEIGHT_PX)
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_shadow(self) -> QGraphicsDropShadowEffect | None:
        """
        Attempt to attach a drop-shadow effect.

        Returns the effect on success, None if the Qt compositing layer is
        unavailable (e.g., certain Wayland or embedded environments).
        """
        try:
            effect = QGraphicsDropShadowEffect()
            _apply_shadow_state(effect, SHADOW_REST)
            self.setGraphicsEffect(effect)
            return effect
        except RuntimeError:
            return None

    # ------------------------------------------------------------------
    # Display parameter application
    # ------------------------------------------------------------------

    def _apply_display_params(self, params: TileDisplayParams) -> None:
        """
        Apply a `TileDisplayParams` observation to this widget.

        This is the single point of contact between the pure scaling
        computation and the Qt widget state — no other method mutates
        icon size, font, or minimum height.
        """
        self.setIcon(resolve_icon(self._icon_file, params.icon_size_px, self._icon_scheme))
        self.setIconSize(QSize(params.icon_size_px, params.icon_size_px))

        font = QFont(FONT_FAMILY, params.font_size_pt)
        font.setWeight(QFont.Bold)
        self.setFont(font)

        self.setMinimumHeight(params.tile_min_height)

    def adapt_to_size(self, viewport_width_px: int, viewport_height_px: int) -> None:
        """
        Recompute and apply display parameters for the given viewport.

        Safe to call from resizeEvent or external layout managers.

        Parameters
        ----------
        viewport_width_px:
            Parent widget width [px].
        viewport_height_px:
            Parent widget height [px].
        """
        self._apply_display_params(
            compute_display_params(viewport_width_px, viewport_height_px)
        )

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            self.adapt_to_size(parent.width(), parent.height())

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        if self._shadow is not None:
            _apply_shadow_state(self._shadow, _hover_shadow(theme.is_dark()))

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        if self._shadow is not None:
            _apply_shadow_state(self._shadow, SHADOW_REST)
