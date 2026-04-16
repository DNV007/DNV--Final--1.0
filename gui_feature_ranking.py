"""
DNV Scientific Module
---------------------
Role:
    Renders the Feature Ranking workspace — a categorised registry of 32
    feature-importance and ranking method launchers, each opening a fully
    configured FeatureRankingDialog on demand.

Scientific Context:
    Presents a typed, categorised catalogue of FeatureDescriptor records
    to the operator.  Each descriptor identifies an importance method and
    its scientific domain.  No data transformation is performed here; this
    module is a pure dispatch layer between the session store and the
    feature-ranking analysis dialog.

Invariants:
    - Every FeatureDescriptor in _FI_REGISTRY has a non-empty key, title,
      subtitle, and category.
    - Each category in _CATEGORY_ORDER appears at least once in _FI_REGISTRY.
    - FI_BUTTON_MIN_HEIGHT_PX ≥ FI_BUTTON_TITLE_FONT_PT
                                + FI_BUTTON_SUBTITLE_FONT_PT
                                + 2 × FI_BUTTON_V_MARGIN_PX.

Assumptions:
    - store.asset is either None or a valid DataAsset with a non-empty
      dataframe attribute containing at least 2 numeric columns.
    - FeatureRankingDialog accepts (key: str, store: SessionStore, parent)
      as its constructor signature.
    - The Qt application instance is initialised before FeatureRankingWindow
      is constructed.

Failure Modes:
    - store.asset is None: workspace renders a diagnostic QLabel; no
      launchers are shown.
    - A method key not handled by FeatureRankingDialog: FeatureRankingDialog
      raises ValueError on Generate; this module is not affected.
    - Missing optional dependency (torch): the FeatureRankingDialog shows a
      user-facing install hint at Generate time; no exception propagates here.

Provenance:
    - This module emits no transformation metadata; it is a pure dispatch
      layer.  Provenance originates in FeatureRankingDialog on computation.
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
from gui_dlg_fi_plot       import FeatureRankingDialog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum height of each method button [px].
FI_BUTTON_MIN_HEIGHT_PX: Final[int] = 52

#: Font size for the button primary (title) label [pt].
FI_BUTTON_TITLE_FONT_PT: Final[int] = 10

#: Font size for the button secondary (subtitle) label [pt].
FI_BUTTON_SUBTITLE_FONT_PT: Final[int] = 8

#: Horizontal inner margin of the button label container [px].
FI_BUTTON_H_MARGIN_PX: Final[int] = 10

#: Vertical inner margin of the button label container [px].
FI_BUTTON_V_MARGIN_PX: Final[int] = 6

#: Pixel gap between title and subtitle labels inside the button [px].
FI_BUTTON_LABEL_SPACING_PX: Final[int] = 1

#: Horizontal gap between adjacent buttons within one category row [px].
FI_ROW_H_SPACING_PX: Final[int] = 8

#: Vertical spacing inserted after each category divider line [px].
FI_POST_DIVIDER_SPACING_PX: Final[int] = 4

#: Vertical spacing inserted after the workspace section header [px].
FI_WORKSPACE_HEADER_SPACING_PX: Final[int] = 8

#: Maximum number of buttons per grid row within a category.
FI_BUTTONS_PER_ROW: Final[int] = 4


# ---------------------------------------------------------------------------
# FeatureDescriptor — typed, immutable record for one method
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureDescriptor:
    """
    Typed, immutable descriptor for a single feature-importance method.

    Fields
    ------
    key      : machine identifier forwarded verbatim to FeatureRankingDialog
    title    : operator-facing primary label (rendered on the button)
    subtitle : domain-context secondary label (rendered below title)
    category : semantic grouping label (rendered as section header)
    """
    key:      str
    title:    str
    subtitle: str
    category: str


# ---------------------------------------------------------------------------
# Registry — single source of truth for all 32 feature-ranking methods
# ---------------------------------------------------------------------------

#: Ordered sequence of category names; controls section render order.
_CATEGORY_ORDER: Final[Tuple[str, ...]] = (
    "BASIC MODEL-BASED",
    "REGULARISATION-BASED",
    "ENSEMBLE & TREE",
    "STATISTICAL & INFORMATION",
    "MODEL-AGNOSTIC",
    "OPTIMISATION & SEARCH",
    "PROBABILISTIC & DEEP LEARNING",
)

#: Complete typed registry of all 32 supported feature-ranking methods.
_FI_REGISTRY: Final[Tuple[FeatureDescriptor, ...]] = (
    # BASIC MODEL-BASED
    FeatureDescriptor("decision_tree",            "Decision Tree",       "Gini / entropy split",         "BASIC MODEL-BASED"),
    FeatureDescriptor("random_forest",            "Random Forest",       "MDI · bagged trees",           "BASIC MODEL-BASED"),
    FeatureDescriptor("linear_regression",        "Linear Regression",   "OLS / Logistic coeff",         "BASIC MODEL-BASED"),
    FeatureDescriptor("svm",                      "SVM Coefficients",    "Linear SVM · margin weight",   "BASIC MODEL-BASED"),
    FeatureDescriptor("feature_maps",             "Feature Maps",        "Correlation heatmap",          "BASIC MODEL-BASED"),
    # REGULARISATION-BASED
    FeatureDescriptor("lasso",                    "Lasso",               "L1 sparsity · shrinkage",      "REGULARISATION-BASED"),
    FeatureDescriptor("ridge",                    "Ridge",               "L2 shrinkage · stable",        "REGULARISATION-BASED"),
    FeatureDescriptor("elasticnet",               "Elastic Net",         "L1+L2 · group sparsity",       "REGULARISATION-BASED"),
    FeatureDescriptor("sparse_pca",               "Sparse PCA",          "L1 loading · dimensionality",  "REGULARISATION-BASED"),
    # ENSEMBLE & TREE
    FeatureDescriptor("gradient_boosting",        "Gradient Boosting",   "GBM · gain split",             "ENSEMBLE & TREE"),
    FeatureDescriptor("rfe",                      "RFE",                 "Recursive elimination · CV",   "ENSEMBLE & TREE"),
    FeatureDescriptor("boruta",                   "Boruta",              "Shadow features · RF",         "ENSEMBLE & TREE"),
    # STATISTICAL & INFORMATION
    FeatureDescriptor("anova_f",                  "ANOVA F",             "F-value · linear assoc",       "STATISTICAL & INFORMATION"),
    FeatureDescriptor("chi_square",               "Chi-Square",          "Categorical · independence",   "STATISTICAL & INFORMATION"),
    FeatureDescriptor("kendall_tau",              "Kendall Tau",         "Rank concordance",             "STATISTICAL & INFORMATION"),
    FeatureDescriptor("mann_whitney",             "Mann-Whitney",        "Rank-biserial · effect size",  "STATISTICAL & INFORMATION"),
    FeatureDescriptor("kruskal_wallis",           "Kruskal-Wallis",      "Non-parametric ANOVA",         "STATISTICAL & INFORMATION"),
    FeatureDescriptor("mutual_information",       "Mutual Info",         "MI(Xi;Y) · non-linear",        "STATISTICAL & INFORMATION"),
    FeatureDescriptor("information_bottleneck",   "Info Bottleneck",     "MI(Xi;Y) / H(Xi) ratio",       "STATISTICAL & INFORMATION"),
    FeatureDescriptor("mrmr",                     "MRMR",                "Min-redundancy max-relevance", "STATISTICAL & INFORMATION"),
    # MODEL-AGNOSTIC
    FeatureDescriptor("permutation_importance",   "Permutation",         "Score drop · model-free",      "MODEL-AGNOSTIC"),
    FeatureDescriptor("shap",                     "SHAP",                "KernelSHAP · Shapley values",  "MODEL-AGNOSTIC"),
    FeatureDescriptor("lime",                     "LIME",                "Local linear · perturbation",  "MODEL-AGNOSTIC"),
    FeatureDescriptor("feature_interactions",     "H-Statistic",         "Interaction network · PD",     "MODEL-AGNOSTIC"),
    # OPTIMISATION & SEARCH
    FeatureDescriptor("genetic_algorithm",        "Genetic Algorithm",   "Binary GA · selection freq",   "OPTIMISATION & SEARCH"),
    FeatureDescriptor("pso",                      "PSO",                 "Particle swarm · gbest",       "OPTIMISATION & SEARCH"),
    FeatureDescriptor("relieff",                  "ReliefF",             "kNN rank · class priors",      "OPTIMISATION & SEARCH"),
    FeatureDescriptor("recursive_feature_clustering","RFC",              "Cluster + MI repr",            "OPTIMISATION & SEARCH"),
    # PROBABILISTIC & DEEP LEARNING
    FeatureDescriptor("bayesian_variable_importance","Bayesian Var",     "BayesianRidge |coef|/σ",       "PROBABILISTIC & DEEP LEARNING"),
    FeatureDescriptor("markov_blanket",           "Markov Blanket",      "IAMB · CMI pruning",           "PROBABILISTIC & DEEP LEARNING"),
    FeatureDescriptor("autoencoder_importance",   "Autoencoder",         "MLP Δ-MSE masked feat",        "PROBABILISTIC & DEEP LEARNING"),
    FeatureDescriptor("deep_learning_attribution","Deep Learning",       "Grad × Input · PyTorch",       "PROBABILISTIC & DEEP LEARNING"),
)


# ---------------------------------------------------------------------------
# _FIButtonDisplayParams — pure-function / data-record pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FIButtonDisplayParams:
    """
    All display parameters for _FIButton derived from module-level constants.

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


def _compute_fi_button_params() -> _FIButtonDisplayParams:
    """
    Pure function: module constants → _FIButtonDisplayParams.

    Makes no Qt calls, writes no state, has no I/O.
    """
    return _FIButtonDisplayParams(
        min_height_px=FI_BUTTON_MIN_HEIGHT_PX,
        title_font_pt=FI_BUTTON_TITLE_FONT_PT,
        subtitle_font_pt=FI_BUTTON_SUBTITLE_FONT_PT,
        h_margin_px=FI_BUTTON_H_MARGIN_PX,
        v_margin_px=FI_BUTTON_V_MARGIN_PX,
        label_spacing_px=FI_BUTTON_LABEL_SPACING_PX,
    )


# ---------------------------------------------------------------------------
# _FIButton — text-only launcher for one FeatureDescriptor
# ---------------------------------------------------------------------------

class _FIButton(QPushButton):
    """
    Text-only launcher button for a single FeatureDescriptor.

    Renders a two-line label (title above, subtitle below).  All display
    parameters are applied through a single _apply_display_params call.

    Parameters
    ----------
    descriptor : FeatureDescriptor
    callback   : zero-argument callable invoked on click
    parent     : optional Qt parent widget
    """

    def __init__(self, descriptor: FeatureDescriptor, callback, parent=None):
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

        self._apply_display_params(_compute_fi_button_params())
        self.clicked.connect(callback)

    def _apply_display_params(self, params: _FIButtonDisplayParams) -> None:
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

def _make_fi_callback(descriptor: FeatureDescriptor, store: SessionStore, parent):
    """
    Factory: FeatureDescriptor × SessionStore × parent → zero-arg callable.

    Returns a zero-argument lambda so that PySide6's signal-slot mechanism
    can trim the bool emitted by QPushButton.clicked without binding it to
    the descriptor key.
    """
    return lambda: FeatureRankingDialog(descriptor.key, store, parent).exec()


# ---------------------------------------------------------------------------
# FeatureRankingWindow
# ---------------------------------------------------------------------------

class FeatureRankingWindow(ModuleDialog):
    """
    Feature Ranking workspace window.

    Renders one _FIButton per registered FeatureDescriptor, grouped under
    their declared category headers.  The window is a pure dispatch layer;
    it performs no data transformation.
    """

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Feature Ranking",
            hint=(
                "Model-based, statistical, model-agnostic, and deep learning "
                "feature-importance methods across 7 families — 32 techniques."
            ),
            category="ATTRIBUTION ANALYSIS · 32 METHODS",
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
            "Feature importance library",
            "Select a method to configure and generate a reproducible feature-ranking analysis.",
            "32 METHODS",
        ))
        self.body_layout.addSpacing(FI_WORKSPACE_HEADER_SPACING_PX)

        for category in _CATEGORY_ORDER:
            descriptors = [d for d in _FI_REGISTRY if d.category == category]
            if not descriptors:
                continue

            hdr = QLabel(category)
            hdr.setObjectName("ReportPanelEyebrow")
            self.body_layout.addWidget(hdr)

            # Wrap into rows of FI_BUTTONS_PER_ROW
            for row_start in range(0, len(descriptors), FI_BUTTONS_PER_ROW):
                row_descs = descriptors[row_start : row_start + FI_BUTTONS_PER_ROW]
                row = QGridLayout()
                row.setHorizontalSpacing(FI_ROW_H_SPACING_PX)
                row.setVerticalSpacing(0)

                for col, descriptor in enumerate(row_descs):
                    btn = _FIButton(
                        descriptor,
                        _make_fi_callback(descriptor, self._store, self),
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
            self.body_layout.addSpacing(FI_POST_DIVIDER_SPACING_PX)

        self.body_layout.addStretch(1)
