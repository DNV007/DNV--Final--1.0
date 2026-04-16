"""
DNV Scientific Module
---------------------
Role:
    Renders the Correlation workspace — a categorised registry of 22
    correlation and dependence method launchers, each opening a fully
    configured CorrelationDialog on demand.

Scientific Context:
    Presents a typed, categorised catalogue of CorrelationDescriptor records
    to the operator.  Each descriptor identifies a dependence measure and its
    scientific domain.  No data transformation is performed here; this module
    is a pure dispatch layer between the session store and the correlation
    analysis dialog.

Invariants:
    - Every CorrelationDescriptor in _CORRELATION_REGISTRY has a non-empty
      key, title, subtitle, and category.
    - Each category in _CATEGORY_ORDER appears at least once in
      _CORRELATION_REGISTRY.
    - CORR_BUTTON_MIN_HEIGHT_PX ≥ CORR_BUTTON_TITLE_FONT_PT
                                  + CORR_BUTTON_SUBTITLE_FONT_PT
                                  + 2 × CORR_BUTTON_V_MARGIN_PX.

Assumptions:
    - store.asset is either None or a valid DataAsset with a non-empty
      dataframe attribute containing at least 2 numeric columns.
    - CorrelationDialog accepts (key: str, store: SessionStore, parent)
      as its constructor signature.
    - The Qt application instance is initialised before CorrelationWindow
      is constructed.

Failure Modes:
    - store.asset is None: workspace renders a diagnostic QLabel; no
      launchers are shown.
    - A method key not handled by CorrelationDialog: CorrelationDialog
      raises ValueError on Generate; this module is not affected.
    - Missing optional dependency (statsmodels, pgmpy, pyinform): the
      CorrelationDialog shows a user-facing install hint at Generate time;
      no exception propagates to this module.

Provenance:
    - This module emits no transformation metadata; it is a pure dispatch
      layer.  Provenance originates in CorrelationDialog on computation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout

from gui_base           import ModuleDialog
from gui_session        import SessionStore
from gui_widgets        import make_section_header
from gui_dlg_cor_plot   import CorrelationDialog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum height of each correlation method button [px].
CORR_BUTTON_MIN_HEIGHT_PX: Final[int] = 52

#: Font size for the button primary (title) label [pt].
CORR_BUTTON_TITLE_FONT_PT: Final[int] = 10

#: Font size for the button secondary (subtitle) label [pt].
CORR_BUTTON_SUBTITLE_FONT_PT: Final[int] = 8

#: Horizontal inner margin of the button label container [px].
CORR_BUTTON_H_MARGIN_PX: Final[int] = 10

#: Vertical inner margin of the button label container [px].
CORR_BUTTON_V_MARGIN_PX: Final[int] = 6

#: Pixel gap between title and subtitle labels inside the button [px].
CORR_BUTTON_LABEL_SPACING_PX: Final[int] = 1

#: Horizontal gap between adjacent buttons within one category row [px].
CORR_ROW_H_SPACING_PX: Final[int] = 8

#: Vertical spacing inserted after each category divider line [px].
CORR_POST_DIVIDER_SPACING_PX: Final[int] = 4

#: Vertical spacing inserted after the workspace section header [px].
CORR_WORKSPACE_HEADER_SPACING_PX: Final[int] = 8


# ---------------------------------------------------------------------------
# CorrelationDescriptor — typed, immutable record for one measure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelationDescriptor:
    """
    Typed, immutable descriptor for a single correlation or dependence measure.

    Fields
    ------
    key      : machine identifier forwarded verbatim to CorrelationDialog
    title    : operator-facing primary label (rendered on the button)
    subtitle : domain-context secondary label (rendered below title)
    category : semantic grouping label (rendered as section header)
    """
    key:      str
    title:    str
    subtitle: str
    category: str


# ---------------------------------------------------------------------------
# Registry — single source of truth for all 22 correlation measures
# ---------------------------------------------------------------------------

#: Ordered sequence of category names; controls section render order.
_CATEGORY_ORDER: Final[Tuple[str, ...]] = (
    "BASIC CORRELATIONS",
    "NONLINEAR CORRELATION",
    "ADVANCED & MULTIVARIATE",
    "TIME SERIES",
    "SPECIAL OUTPUTS",
)

#: Complete typed registry of all supported correlation methods.
_CORRELATION_REGISTRY: Final[Tuple[CorrelationDescriptor, ...]] = (
    # BASIC CORRELATIONS
    CorrelationDescriptor("pearson",     "Pearson",           "linear · parametric",           "BASIC CORRELATIONS"),
    CorrelationDescriptor("spearman",    "Spearman",          "rank · monotone · robust",      "BASIC CORRELATIONS"),
    CorrelationDescriptor("kendall",     "Kendall Tau",       "concordance · small samples",   "BASIC CORRELATIONS"),
    # NONLINEAR CORRELATION
    CorrelationDescriptor("mic",         "MIC",               "Reshef maximal info coeff",     "NONLINEAR CORRELATION"),
    CorrelationDescriptor("distance",    "Distance Corr",     "nonlinear · metric space",      "NONLINEAR CORRELATION"),
    CorrelationDescriptor("hoeffding",   "Hoeffding's D",     "omnibus independence test",     "NONLINEAR CORRELATION"),
    CorrelationDescriptor("chatterjee",  "Chatterjee's ξ",    "rank · asymmetric · 2021",      "NONLINEAR CORRELATION"),
    CorrelationDescriptor("polychoric",  "Polychoric",        "Olsson ML · latent normal",     "NONLINEAR CORRELATION"),
    CorrelationDescriptor("hsic",        "HSIC",              "kernel · independence test",    "NONLINEAR CORRELATION"),
    CorrelationDescriptor("mice",        "k-NN MI (k=5)",     "Kraskov · higher-k variant",    "NONLINEAR CORRELATION"),
    # ADVANCED & MULTIVARIATE
    CorrelationDescriptor("r_statistic", "Blomqvist β",       "medial · robust rank stat",     "ADVANCED & MULTIVARIATE"),
    CorrelationDescriptor("sparse_partial","Sparse Partial",  "GraphLASSO · precision net",    "ADVANCED & MULTIVARIATE"),
    CorrelationDescriptor("cca",         "Pairwise CCA",      "1-D canonical ≡ |Pearson|",     "ADVANCED & MULTIVARIATE"),
    CorrelationDescriptor("partial",     "Partial Corr",      "controlling covariates",        "ADVANCED & MULTIVARIATE"),
    CorrelationDescriptor("autoencoder", "Autoencoder",       "bottleneck reconstruction",     "ADVANCED & MULTIVARIATE"),
    CorrelationDescriptor("bayesian",    "Bayesian Network",  "DAG structure · K2 score",      "ADVANCED & MULTIVARIATE"),
    # TIME SERIES
    CorrelationDescriptor("ccf",         "CCF",               "cross-correlation · lags",      "TIME SERIES"),
    CorrelationDescriptor("granger",     "Granger Causality", "temporal precedence · F-test",  "TIME SERIES"),
    CorrelationDescriptor("transfer_entropy","Transfer Entropy","information flow · TE",       "TIME SERIES"),
    # SPECIAL OUTPUTS
    CorrelationDescriptor("mi",          "Mutual Info",       "feature → target MI scores",   "SPECIAL OUTPUTS"),
    CorrelationDescriptor("lasso",       "LASSO Regression",  "L1 feature selection",         "SPECIAL OUTPUTS"),
    CorrelationDescriptor("dag",         "DAG Visualisation", "directed graph · threshold",   "SPECIAL OUTPUTS"),
)


# ---------------------------------------------------------------------------
# _CorrButtonDisplayParams — pure-function / data-record pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CorrButtonDisplayParams:
    """
    All display parameters for _CorrButton derived from module-level constants.

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


def _compute_corr_button_params() -> _CorrButtonDisplayParams:
    """
    Pure function: module constants → _CorrButtonDisplayParams.

    Makes no Qt calls, writes no state, has no I/O.
    """
    return _CorrButtonDisplayParams(
        min_height_px=CORR_BUTTON_MIN_HEIGHT_PX,
        title_font_pt=CORR_BUTTON_TITLE_FONT_PT,
        subtitle_font_pt=CORR_BUTTON_SUBTITLE_FONT_PT,
        h_margin_px=CORR_BUTTON_H_MARGIN_PX,
        v_margin_px=CORR_BUTTON_V_MARGIN_PX,
        label_spacing_px=CORR_BUTTON_LABEL_SPACING_PX,
    )


# ---------------------------------------------------------------------------
# _CorrButton — text-only launcher for one CorrelationDescriptor
# ---------------------------------------------------------------------------

class _CorrButton(QPushButton):
    """
    Text-only launcher button for a single CorrelationDescriptor.

    Renders a two-line label (title above, subtitle below).  All display
    parameters are applied through a single _apply_display_params call.

    Parameters
    ----------
    descriptor : CorrelationDescriptor
    callback   : zero-argument callable invoked on click
    parent     : optional Qt parent widget
    """

    def __init__(self, descriptor: CorrelationDescriptor, callback, parent=None):
        super().__init__(parent)
        self.descriptor = descriptor
        self.setCursor(Qt.PointingHandCursor)

        self._title_lbl = QLabel(descriptor.title, self)
        self._title_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._title_lbl.setWordWrap(True)

        self._sub_lbl = QLabel(descriptor.subtitle, self)
        self._sub_lbl.setObjectName("SectionHint")
        self._sub_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._sub_lbl.setWordWrap(True)

        inner_lay = QVBoxLayout(self)
        inner_lay.addWidget(self._title_lbl)
        inner_lay.addWidget(self._sub_lbl)

        self._apply_display_params(_compute_corr_button_params())
        self.clicked.connect(callback)

    def _apply_display_params(self, params: _CorrButtonDisplayParams) -> None:
        """Single site that writes all Qt display state from a params record."""
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
            params.h_margin_px, params.v_margin_px,
            params.h_margin_px, params.v_margin_px,
        )
        lay.setSpacing(params.label_spacing_px)


# ---------------------------------------------------------------------------
# Callback factory — module-level, not defined inline in _build_body
# ---------------------------------------------------------------------------

def _make_corr_callback(descriptor: CorrelationDescriptor, store: SessionStore, parent):
    """
    Factory: CorrelationDescriptor × SessionStore × parent → zero-arg callable.

    Returns a zero-argument lambda so that PySide6's signal-slot mechanism
    can trim the bool emitted by QPushButton.clicked without binding it to
    the descriptor key.
    """
    return lambda: CorrelationDialog(descriptor.key, store, parent).exec()


# ---------------------------------------------------------------------------
# CorrelationWindow
# ---------------------------------------------------------------------------

class CorrelationWindow(ModuleDialog):
    """
    Correlation workspace window.

    Renders one _CorrButton per registered CorrelationDescriptor, grouped
    under their declared category headers.  The window is a pure dispatch
    layer; it performs no data transformation.
    """

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Correlation",
            hint=(
                "Statistical and information-theoretic dependence measures "
                "across 5 method families — from Pearson to Transfer Entropy."
            ),
            category="DEPENDENCE ANALYSIS · 22 METHODS",
            parent=parent,
        )
        self._store = store

        if store.asset is None:
            note = QLabel("No active dataset. Load a dataset in Data Ingestion first.")
            note.setObjectName("SectionHint")
            note.setWordWrap(True)
            self.body_layout.addWidget(note)
            return

        self._build_body()

    def _build_body(self) -> None:
        """Construct the categorised launcher grid."""
        self.body_layout.addWidget(make_section_header(
            "Correlation library",
            "Select a dependence measure to configure and generate a reproducible analysis.",
            "22 METHODS",
        ))
        self.body_layout.addSpacing(CORR_WORKSPACE_HEADER_SPACING_PX)

        for category in _CATEGORY_ORDER:
            descriptors = [d for d in _CORRELATION_REGISTRY if d.category == category]
            if not descriptors:
                continue

            hdr = QLabel(category)
            hdr.setObjectName("ReportPanelEyebrow")
            self.body_layout.addWidget(hdr)

            row = QGridLayout()
            row.setHorizontalSpacing(CORR_ROW_H_SPACING_PX)
            row.setVerticalSpacing(0)

            for col, descriptor in enumerate(descriptors):
                btn = _CorrButton(
                    descriptor,
                    _make_corr_callback(descriptor, self._store, self),
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
            self.body_layout.addSpacing(CORR_POST_DIVIDER_SPACING_PX)

        self.body_layout.addStretch(1)
