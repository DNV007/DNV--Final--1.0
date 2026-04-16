"""
DNV Scientific Module
---------------------
Role:
    Renders the Descriptors workspace — a categorised registry of 15
    descriptor-modelling method launchers, each opening a fully configured
    DescriptorDialog on demand.

Scientific Context:
    Presents a typed, categorised catalogue of DescriptorMethodDescriptor
    records to the operator.  Each record identifies a symbolic-regression or
    regularised-regression method for discovering compact, physically
    interpretable descriptors of a target property.  No data transformation
    is performed here; this module is a pure dispatch layer.

Invariants:
    - Every DescriptorMethodDescriptor in _DESC_REGISTRY has a non-empty
      key, title, subtitle, and category.
    - Each category in _CATEGORY_ORDER appears at least once in _DESC_REGISTRY.
    - DESC_BUTTON_MIN_HEIGHT_PX ≥ DESC_BUTTON_TITLE_FONT_PT
                                  + DESC_BUTTON_SUBTITLE_FONT_PT
                                  + 2 × DESC_BUTTON_V_MARGIN_PX.

Assumptions:
    - store.asset is either None or a valid DataAsset with a non-empty
      dataframe attribute containing at least 2 numeric columns.
    - DescriptorDialog accepts (key: str, store: SessionStore, parent)
      as its constructor signature.
    - The Qt application instance is initialised before DescriptorsWindow
      is constructed.

Failure Modes:
    - store.asset is None: workspace renders a diagnostic QLabel; no
      launchers are shown.
Provenance:
    - This module emits no transformation metadata; it is a pure dispatch
      layer.  Provenance originates in DescriptorDialog on computation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout

from gui_base              import ModuleDialog
from gui_session           import SessionStore
from gui_widgets           import make_section_header
from gui_dlg_desc_plot     import DescriptorDialog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum height of each method button [px].
DESC_BUTTON_MIN_HEIGHT_PX: Final[int] = 56

#: Font size for the button primary (title) label [pt].
DESC_BUTTON_TITLE_FONT_PT: Final[int] = 10

#: Font size for the button secondary (subtitle) label [pt].
DESC_BUTTON_SUBTITLE_FONT_PT: Final[int] = 8

#: Horizontal inner margin of the button label container [px].
DESC_BUTTON_H_MARGIN_PX: Final[int] = 12

#: Vertical inner margin of the button label container [px].
DESC_BUTTON_V_MARGIN_PX: Final[int] = 8

#: Pixel gap between title and subtitle labels inside the button [px].
DESC_BUTTON_LABEL_SPACING_PX: Final[int] = 2

#: Horizontal gap between adjacent buttons within one category row [px].
DESC_ROW_H_SPACING_PX: Final[int] = 10

#: Vertical spacing inserted after each category divider line [px].
DESC_POST_DIVIDER_SPACING_PX: Final[int] = 4

#: Vertical spacing inserted after the workspace section header [px].
DESC_WORKSPACE_HEADER_SPACING_PX: Final[int] = 8


# ---------------------------------------------------------------------------
# DescriptorMethodDescriptor — typed, immutable record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DescriptorMethodDescriptor:
    """
    Typed, immutable descriptor for a single descriptor-modelling method.

    Fields
    ------
    key      : machine identifier forwarded verbatim to DescriptorDialog
    title    : operator-facing primary label (rendered on the button)
    subtitle : domain-context secondary label (rendered below title)
    category : semantic grouping label (rendered as section header)
    """
    key:      str
    title:    str
    subtitle: str
    category: str


# ---------------------------------------------------------------------------
# Registry — single source of truth for all descriptor methods
# ---------------------------------------------------------------------------

#: Ordered sequence of category names; controls section render order.
_CATEGORY_ORDER: Final[Tuple[str, ...]] = (
    "BASIC SCREENING",
    "REGULARISATION-BASED",
    "COMBINATORIAL SEARCH",
)

#: Complete typed registry of all supported descriptor-modelling methods.
_DESC_REGISTRY: Final[Tuple[DescriptorMethodDescriptor, ...]] = (
    # BASIC SCREENING
    DescriptorMethodDescriptor(
        "linear_screening",
        "Linear Screening",
        "R² per formula · rank all descriptors",
        "BASIC SCREENING",
    ),
    # REGULARISATION-BASED
    DescriptorMethodDescriptor(
        "lasso",
        "Lasso CV",
        "L1 sparsity · cross-validated α",
        "REGULARISATION-BASED",
    ),
    DescriptorMethodDescriptor(
        "ridge",
        "Ridge CV",
        "L2 shrinkage · log-spaced α grid",
        "REGULARISATION-BASED",
    ),
    DescriptorMethodDescriptor(
        "elasticnet",
        "Elastic Net CV",
        "L1 + L2 · group sparsity",
        "REGULARISATION-BASED",
    ),
    # COMBINATORIAL SEARCH
    DescriptorMethodDescriptor(
        "greedy_forward",
        "Greedy Forward",
        "Sequential add · R² gain",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "greedy_backward",
        "Greedy Backward",
        "Sequential remove · min harm",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "armhc",
        "ARMHC",
        "Adaptive hill climbing · adaptive mutation",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "sa_metropolis",
        "SA — Metropolis",
        "Annealing · exp acceptance",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "sa_glauber",
        "SA — Glauber",
        "Annealing · sigmoid acceptance",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "rts",
        "Reactive Tabu",
        "Tabu search · reactive tenure",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "cem",
        "Cross-Entropy",
        "Probabilistic model · elite update",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "pt",
        "Parallel Tempering",
        "Multi-replica SA · swap acceptance",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "qa",
        "Quantum Annealing",
        "PIMC · Trotter-slice tunnelling",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "pso",
        "Particle Swarm",
        "Binary PSO · sigmoid transfer",
        "COMBINATORIAL SEARCH",
    ),
    DescriptorMethodDescriptor(
        "gp",
        "Genetic Programming",
        "Variable-length · parsimony pressure",
        "COMBINATORIAL SEARCH",
    ),
)


# ---------------------------------------------------------------------------
# _DescButtonDisplayParams — pure-function / data-record pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DescButtonDisplayParams:
    """
    All display parameters for _DescButton derived from module-level constants.

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


def _compute_desc_button_params() -> _DescButtonDisplayParams:
    """
    Pure function: module constants → _DescButtonDisplayParams.

    Makes no Qt calls, writes no state, has no I/O.
    """
    return _DescButtonDisplayParams(
        min_height_px=DESC_BUTTON_MIN_HEIGHT_PX,
        title_font_pt=DESC_BUTTON_TITLE_FONT_PT,
        subtitle_font_pt=DESC_BUTTON_SUBTITLE_FONT_PT,
        h_margin_px=DESC_BUTTON_H_MARGIN_PX,
        v_margin_px=DESC_BUTTON_V_MARGIN_PX,
        label_spacing_px=DESC_BUTTON_LABEL_SPACING_PX,
    )


# ---------------------------------------------------------------------------
# _DescButton — text-only launcher
# ---------------------------------------------------------------------------

class _DescButton(QPushButton):
    """
    Text-only launcher button for a single DescriptorMethodDescriptor.

    Renders a two-line label (title above, subtitle below).  All display
    parameters are applied through a single _apply_display_params call.

    Parameters
    ----------
    descriptor : DescriptorMethodDescriptor
    callback   : zero-argument callable invoked on click
    parent     : optional Qt parent widget
    """

    def __init__(self, descriptor: DescriptorMethodDescriptor, callback, parent=None):
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

        self._apply_display_params(_compute_desc_button_params())
        self.clicked.connect(callback)

    def _apply_display_params(self, params: _DescButtonDisplayParams) -> None:
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

def _make_desc_callback(
    descriptor: DescriptorMethodDescriptor,
    store: SessionStore,
    parent,
):
    """
    Factory: DescriptorMethodDescriptor × SessionStore × parent → zero-arg callable.

    Returns a zero-argument lambda so that PySide6's signal-slot mechanism
    can trim the bool emitted by QPushButton.clicked without binding it to
    the descriptor key.
    """
    return lambda: DescriptorDialog(descriptor.key, store, parent).exec()


# ---------------------------------------------------------------------------
# DescriptorsWindow
# ---------------------------------------------------------------------------

class DescriptorsWindow(ModuleDialog):
    """
    Descriptor Modelling workspace window.

    Renders one _DescButton per registered DescriptorMethodDescriptor, grouped
    under their declared category headers.  The window is a pure dispatch
    layer; it performs no data transformation.
    """

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Descriptors",
            hint=(
                "Symbolic feature generation via unary/binary operators followed "
                "by linear screening or regularised regression to identify "
                "compact, physically interpretable property descriptors."
            ),
            category="DESCRIPTOR MODELLING · 14 METHODS",
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
            "Descriptor modelling library",
            "Select a method to generate symbolic features and identify "
            "the most predictive property descriptors.",
            "14 METHODS",
        ))
        self.body_layout.addSpacing(DESC_WORKSPACE_HEADER_SPACING_PX)

        for category in _CATEGORY_ORDER:
            descriptors = [d for d in _DESC_REGISTRY if d.category == category]
            if not descriptors:
                continue

            hdr = QLabel(category)
            hdr.setObjectName("ReportPanelEyebrow")
            self.body_layout.addWidget(hdr)

            row = QGridLayout()
            row.setHorizontalSpacing(DESC_ROW_H_SPACING_PX)
            row.setVerticalSpacing(0)

            for col, descriptor in enumerate(descriptors):
                btn = _DescButton(
                    descriptor,
                    _make_desc_callback(descriptor, self._store, self),
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
            self.body_layout.addSpacing(DESC_POST_DIVIDER_SPACING_PX)

        self.body_layout.addStretch(1)
