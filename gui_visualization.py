"""
DNV Scientific Module
---------------------
Role:
    Renders the Visualization workspace — a categorised registry of 18
    plot-type launchers, each opening a fully configured PlotDialog on demand.

Scientific Context:
    Presents a typed, categorised catalogue of PlotDescriptor records to the
    operator.  Each descriptor identifies a visualisation method and its domain
    context.  No data transformation is performed here; this module is a pure
    dispatch layer between the session store and the plot-configuration dialog.

Invariants:
    - Every PlotDescriptor in _PLOT_REGISTRY has a non-empty key, title,
      subtitle, and category.
    - Each category present in _CATEGORY_ORDER appears at least once in
      _PLOT_REGISTRY.
    - PLOT_BUTTON_MIN_HEIGHT_PX ≥ PLOT_BUTTON_TITLE_FONT_PT
                                   + PLOT_BUTTON_SUBTITLE_FONT_PT
                                   + 2 × PLOT_BUTTON_V_MARGIN_PX.

Assumptions:
    - store.asset is either None or a valid DataAsset with a non-empty
      dataframe attribute.
    - PlotDialog accepts (key: str, store: SessionStore, parent) as its
      constructor signature.
    - The Qt application instance is initialised before VisualizationWindow
      is constructed.

Failure Modes:
    - store.asset is None: workspace renders a diagnostic QLabel; no
      launchers are shown.  All button-construction code is skipped.
    - A PlotDescriptor key not recognised by PlotDialog: PlotDialog renders
      an internal error panel; no exception propagates to this module.

Provenance:
    - This module emits no transformation metadata; it is a pure dispatch
      layer.  Provenance originates in PlotDialog on plot generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout

from gui_base         import ModuleDialog
from gui_session      import SessionStore
from gui_widgets      import make_section_header
from gui_dlg_viz_plot import PlotDialog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum height of each plot-type button [px].
PLOT_BUTTON_MIN_HEIGHT_PX: Final[int] = 52

#: Font size for the button primary (title) label [pt].
PLOT_BUTTON_TITLE_FONT_PT: Final[int] = 10

#: Font size for the button secondary (subtitle) label [pt].
PLOT_BUTTON_SUBTITLE_FONT_PT: Final[int] = 8

#: Horizontal inner margin of the button label container [px].
PLOT_BUTTON_H_MARGIN_PX: Final[int] = 10

#: Vertical inner margin of the button label container [px].
PLOT_BUTTON_V_MARGIN_PX: Final[int] = 6

#: Pixel gap between title and subtitle labels inside the button [px].
PLOT_BUTTON_LABEL_SPACING_PX: Final[int] = 1

#: Horizontal gap between adjacent buttons within one category row [px].
CATEGORY_ROW_H_SPACING_PX: Final[int] = 8

#: Vertical spacing inserted after each category divider line [px].
SECTION_POST_DIVIDER_SPACING_PX: Final[int] = 4

#: Vertical spacing inserted after the workspace section header [px].
WORKSPACE_HEADER_SPACING_PX: Final[int] = 8


# ---------------------------------------------------------------------------
# PlotDescriptor — typed, immutable record for one visualisation method
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotDescriptor:
    """
    Typed, immutable descriptor for a single visualisation method.

    Fields
    ------
    key      : machine identifier forwarded verbatim to PlotDialog
    title    : operator-facing primary label (rendered on the button)
    subtitle : domain-context secondary label (rendered below title)
    category : semantic grouping label (rendered as section header;
               not rendered on the button itself)
    """
    key:      str
    title:    str
    subtitle: str
    category: str


# ---------------------------------------------------------------------------
# Plot registry — single source of truth for all 18 visualisation methods
# ---------------------------------------------------------------------------

#: Ordered sequence of category names; controls section render order.
_CATEGORY_ORDER: Final[Tuple[str, ...]] = (
    "UNIVARIATE",
    "BIVARIATE",
    "MULTIVARIATE",
    "TIME SERIES",
    "CATEGORICAL",
    "DIMENSIONALITY REDUCTION",
)

#: Complete typed registry of all supported plot methods.
_PLOT_REGISTRY: Final[Tuple[PlotDescriptor, ...]] = (
    # UNIVARIATE
    PlotDescriptor("histogram",   "Histogram",            "distribution · KDE overlay",      "UNIVARIATE"),
    PlotDescriptor("boxplot",     "Box Plot",             "quartiles · outlier markers",      "UNIVARIATE"),
    PlotDescriptor("violin",      "Violin Plot",          "density × quartile fusion",        "UNIVARIATE"),
    PlotDescriptor("kde",         "KDE / Density",        "kernel density estimation",        "UNIVARIATE"),
    # BIVARIATE
    PlotDescriptor("scatter",     "Scatter Plot",         "bivariate · hue · OLS fit",        "BIVARIATE"),
    PlotDescriptor("joint",       "Joint Plot",           "marginal distributions",           "BIVARIATE"),
    PlotDescriptor("pairplot",    "Pair Plot",            "all-pairs correlation matrix",     "BIVARIATE"),
    PlotDescriptor("heatmap",     "Correlation Heatmap",  "coefficient map · annotated",      "BIVARIATE"),
    # MULTIVARIATE
    PlotDescriptor("parallel",    "Parallel Coordinates", "multivariate profiles",            "MULTIVARIATE"),
    PlotDescriptor("scatter3d",   "3-D Scatter",          "3-axis spatial scatter",           "MULTIVARIATE"),
    # TIME SERIES
    PlotDescriptor("line",        "Line Plot",            "temporal · index series",          "TIME SERIES"),
    PlotDescriptor("area",        "Area Plot",            "cumulative · stacked area",        "TIME SERIES"),
    PlotDescriptor("candlestick", "Candlestick / OHLC",   "price action · OHLC bars",         "TIME SERIES"),
    # CATEGORICAL
    PlotDescriptor("bar",         "Bar / Column Chart",   "aggregated · CI error bars",       "CATEGORICAL"),
    PlotDescriptor("swarm",       "Swarm Plot",           "non-overlapping point cloud",      "CATEGORICAL"),
    PlotDescriptor("strip",       "Strip Plot",           "jittered 1-D scatter",             "CATEGORICAL"),
    # DIMENSIONALITY REDUCTION
    PlotDescriptor("tsne",        "t-SNE Embedding",      "non-linear 2-D embedding",         "DIMENSIONALITY REDUCTION"),
    PlotDescriptor("pca",         "PCA Biplot",           "linear · loading vectors",         "DIMENSIONALITY REDUCTION"),
)


# ---------------------------------------------------------------------------
# PlotButton — text-only launcher for one PlotDescriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PlotButtonDisplayParams:
    """
    All display parameters for _PlotButton derived from module-level constants.

    Units
    -----
    min_height_px      : pixels [px]
    title_font_pt      : typographic points [pt]
    subtitle_font_pt   : typographic points [pt]
    h_margin_px        : pixels [px]
    v_margin_px        : pixels [px]
    label_spacing_px   : pixels [px]
    """
    min_height_px:    int
    title_font_pt:    int
    subtitle_font_pt: int
    h_margin_px:      int
    v_margin_px:      int
    label_spacing_px: int


def _compute_plot_button_params() -> _PlotButtonDisplayParams:
    """
    Pure function: module constants → _PlotButtonDisplayParams.

    Makes no Qt calls, writes no state, has no I/O.
    All values are sourced from named Final constants declared above.
    """
    return _PlotButtonDisplayParams(
        min_height_px=PLOT_BUTTON_MIN_HEIGHT_PX,
        title_font_pt=PLOT_BUTTON_TITLE_FONT_PT,
        subtitle_font_pt=PLOT_BUTTON_SUBTITLE_FONT_PT,
        h_margin_px=PLOT_BUTTON_H_MARGIN_PX,
        v_margin_px=PLOT_BUTTON_V_MARGIN_PX,
        label_spacing_px=PLOT_BUTTON_LABEL_SPACING_PX,
    )


class _PlotButton(QPushButton):
    """
    Text-only launcher button for a single PlotDescriptor.

    Renders a two-line label (title above, subtitle below).  All display
    parameters are applied through a single _apply_display_params call;
    no Qt state is written elsewhere in this class.

    Parameters
    ----------
    descriptor : PlotDescriptor
        Typed record identifying the plot method this button launches.
    callback   : zero-argument callable invoked on click
    parent     : optional Qt parent widget
    """

    def __init__(self, descriptor: PlotDescriptor, callback, parent=None):
        super().__init__(parent)
        self.descriptor = descriptor
        self.setCursor(Qt.PointingHandCursor)

        self._title_lbl = QLabel(descriptor.title, self)
        self._title_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._sub_lbl = QLabel(descriptor.subtitle, self)
        self._sub_lbl.setObjectName("SectionHint")
        self._sub_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

        inner_lay = QVBoxLayout(self)
        inner_lay.addWidget(self._title_lbl)
        inner_lay.addWidget(self._sub_lbl)

        self._apply_display_params(_compute_plot_button_params())
        self.clicked.connect(callback)

    def _apply_display_params(self, params: _PlotButtonDisplayParams) -> None:
        """
        Single site that writes all Qt display state from a params record.

        No display attribute is set outside this method.
        """
        self.setMinimumHeight(params.min_height_px)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(params.title_font_pt)
        self._title_lbl.setFont(title_font)

        sub_font = QFont()
        sub_font.setPointSize(params.subtitle_font_pt)
        self._sub_lbl.setFont(sub_font)

        lay = self.layout()
        lay.setContentsMargins(
            params.h_margin_px,
            params.v_margin_px,
            params.h_margin_px,
            params.v_margin_px,
        )
        lay.setSpacing(params.label_spacing_px)


# ---------------------------------------------------------------------------
# Callback factory — module-level, not defined inline in _build_body
# ---------------------------------------------------------------------------

def _make_plot_callback(descriptor: PlotDescriptor, store: SessionStore, parent):
    """
    Factory: PlotDescriptor × SessionStore × parent → zero-argument callable.

    Returns a zero-argument lambda so that PySide6's signal-slot mechanism
    can trim the bool emitted by QPushButton.clicked without binding it to
    the descriptor key.

    Parameters
    ----------
    descriptor : PlotDescriptor
        The plot method to open.
    store      : SessionStore
        Active session; forwarded to PlotDialog unchanged.
    parent     : QWidget
        Dialog parent widget.
    """
    return lambda: PlotDialog(descriptor.key, store, parent).exec()


# ---------------------------------------------------------------------------
# VisualizationWindow
# ---------------------------------------------------------------------------

class VisualizationWindow(ModuleDialog):
    """
    Visualization workspace window.

    Renders one _PlotButton per registered PlotDescriptor, grouped under
    their declared category headers.  The window is a pure dispatch layer;
    it performs no data transformation.
    """

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Visualization",
            hint=(
                "Diagnostic and publication-oriented plots with explicit "
                "export context and replay records."
            ),
            category="DATA VISUALIZATION · 18 TYPES",
            parent=parent,
        )
        self._store = store

        if store.asset is None:
            note = QLabel(
                "No active dataset. Load a dataset in Data Ingestion first."
            )
            note.setObjectName("SectionHint")
            note.setWordWrap(True)
            self.body_layout.addWidget(note)
            return

        self._build_body()

    def _build_body(self) -> None:
        """Construct the categorised launcher grid."""
        self.body_layout.addWidget(make_section_header(
            "Chart library",
            "Select a visualisation type to configure parameters and "
            "generate a reproducible plot.",
            "18 TYPES",
        ))
        self.body_layout.addSpacing(WORKSPACE_HEADER_SPACING_PX)

        for category in _CATEGORY_ORDER:
            descriptors = [d for d in _PLOT_REGISTRY if d.category == category]
            if not descriptors:
                continue

            # Category section header
            hdr = QLabel(category)
            hdr.setObjectName("ReportPanelEyebrow")
            self.body_layout.addWidget(hdr)

            # One-row grid of buttons for this category
            row = QGridLayout()
            row.setHorizontalSpacing(CATEGORY_ROW_H_SPACING_PX)
            row.setVerticalSpacing(0)

            for col, descriptor in enumerate(descriptors):
                btn = _PlotButton(
                    descriptor,
                    _make_plot_callback(descriptor, self._store, self),
                    self,
                )
                row.addWidget(btn, 0, col)
                row.setColumnStretch(col, 1)

            self.body_layout.addLayout(row)

            sep = QFrame()
            sep.setObjectName("Divider")
            sep.setFixedHeight(1)
            sep.setFrameShape(QFrame.HLine)
            self.body_layout.addWidget(sep)
            self.body_layout.addSpacing(SECTION_POST_DIVIDER_SPACING_PX)

        self.body_layout.addStretch(1)
