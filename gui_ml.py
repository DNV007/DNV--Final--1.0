"""
DNV Scientific Module
---------------------
Role:
    Renders the Machine Learning workspace — a categorised registry of 17
    model-training method launchers, each opening a fully configured
    MLDialog on demand.

Scientific Context:
    Presents a typed, categorised catalogue of MLMethodDescriptor records to
    the operator.  Each record identifies a supervised ML method (linear,
    tree-based, ensemble, kernel, neural network, or probabilistic) along with
    its task domain.  No model fitting is performed here; this module is a pure
    dispatch layer.

Invariants:
    - Every MLMethodDescriptor in _ML_REGISTRY has a non-empty key, title,
      subtitle, and category.
    - Each category in _CATEGORY_ORDER appears at least once in _ML_REGISTRY.
    - ML_BUTTON_MIN_HEIGHT_PX ≥ ML_BUTTON_TITLE_FONT_PT
                                + ML_BUTTON_SUBTITLE_FONT_PT
                                + 2 × ML_BUTTON_V_MARGIN_PX.

Assumptions:
    - store.asset is either None or a valid DataAsset with a non-empty
      dataframe attribute containing at least 2 numeric columns.
    - MLDialog accepts (key: str, store: SessionStore, parent) as its
      constructor signature.
    - The Qt application instance is initialised before MLWindow is constructed.

Failure Modes:
    - store.asset is None: workspace renders a diagnostic QLabel; no launchers.
    - Task mismatch for method-specific keys (e.g. linear_regression on a
      classification target): ValueError shown in MLDialog on Generate.
    - PyTorch unavailable: ImportError shown in MLDialog on Generate for pytorch_mlp.
    - XGBoost / LightGBM not installed: ImportError shown in MLDialog on Generate.

Provenance:
    - This module emits no transformation metadata; it is a pure dispatch layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple

from PySide6.QtCore    import Qt
from PySide6.QtGui     import QFont
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout

from gui_base          import ModuleDialog
from gui_session       import SessionStore
from gui_widgets       import make_section_header
from gui_dlg_ml_plot   import MLDialog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants — all values carry explicit units in name
# ---------------------------------------------------------------------------

#: Minimum height of each method button [px].
ML_BUTTON_MIN_HEIGHT_PX: Final[int] = 52

#: Font size for the button primary (title) label [pt].
ML_BUTTON_TITLE_FONT_PT: Final[int] = 10

#: Font size for the button secondary (subtitle) label [pt].
ML_BUTTON_SUBTITLE_FONT_PT: Final[int] = 8

#: Horizontal inner margin of the button label container [px].
ML_BUTTON_H_MARGIN_PX: Final[int] = 10

#: Vertical inner margin of the button label container [px].
ML_BUTTON_V_MARGIN_PX: Final[int] = 6

#: Pixel gap between title and subtitle labels inside the button [px].
ML_BUTTON_LABEL_SPACING_PX: Final[int] = 1

#: Horizontal gap between adjacent buttons within one category row [px].
ML_ROW_H_SPACING_PX: Final[int] = 8

#: Vertical spacing inserted after each category divider line [px].
ML_POST_DIVIDER_SPACING_PX: Final[int] = 4

#: Vertical spacing inserted after the workspace section header [px].
ML_WORKSPACE_HEADER_SPACING_PX: Final[int] = 8

#: Maximum number of buttons per grid row within a category.
ML_BUTTONS_PER_ROW: Final[int] = 4


# ---------------------------------------------------------------------------
# MLMethodDescriptor — typed, immutable record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLMethodDescriptor:
    """
    Typed, immutable descriptor for a single ML method.

    Fields
    ------
    key      : machine identifier forwarded verbatim to MLDialog
    title    : operator-facing primary label (rendered on the button)
    subtitle : domain-context secondary label (rendered below title)
    category : semantic grouping label (rendered as section header)
    """
    key:       str
    title:     str
    subtitle:  str
    category:  str
    task_type: str  # "regression", "classification", or "both"


# ---------------------------------------------------------------------------
# Registry — single source of truth for all 17 ML methods
# ---------------------------------------------------------------------------

#: Ordered sequence of category names; controls section render order.
_CATEGORY_ORDER: Final[Tuple[str, ...]] = (
    "LINEAR & REGULARISED",
    "TREE-BASED",
    "ENSEMBLE",
    "KERNEL & NEIGHBOURS",
    "NEURAL NETWORK",
    "PROBABILISTIC",
    "GRADIENT BOOSTING",
)

#: Complete typed registry of all 17 supported ML methods.
_ML_REGISTRY: Final[Tuple[MLMethodDescriptor, ...]] = (
    # LINEAR & REGULARISED
    MLMethodDescriptor("linear_regression",   "Linear Regression",   "OLS · regression only",            "LINEAR & REGULARISED", "regression"),
    MLMethodDescriptor("logistic_regression", "Logistic Regression", "L2 · OvR · classification only",   "LINEAR & REGULARISED", "classification"),
    MLMethodDescriptor("ridge",               "Ridge",               "L2 · regression & classification", "LINEAR & REGULARISED", "both"),
    MLMethodDescriptor("lasso",               "Lasso",               "L1 · sparse · regression only",    "LINEAR & REGULARISED", "regression"),
    # TREE-BASED
    MLMethodDescriptor("decision_tree",       "Decision Tree",       "CART · Gini / MSE split",          "TREE-BASED",           "both"),
    MLMethodDescriptor("random_forest",       "Random Forest",       "Bagged trees · MDI scores",        "TREE-BASED",           "both"),
    MLMethodDescriptor("extra_trees",         "Extra Trees",         "Randomised splits · fast",         "TREE-BASED",           "both"),
    # ENSEMBLE
    MLMethodDescriptor("gradient_boosting",   "Gradient Boosting",   "sklearn GBM · staged loss",        "ENSEMBLE",             "both"),
    MLMethodDescriptor("adaboost",            "AdaBoost",            "Adaptive reweighting · SAMME",     "ENSEMBLE",             "both"),
    # KERNEL & NEIGHBOURS
    MLMethodDescriptor("svm",                 "SVM",                 "RBF kernel · margin maximisation", "KERNEL & NEIGHBOURS",  "both"),
    MLMethodDescriptor("knn",                 "K-Nearest Neighbours","Distance voting · k-NN",           "KERNEL & NEIGHBOURS",  "both"),
    # NEURAL NETWORK
    MLMethodDescriptor("mlp",                 "MLP (sklearn)",       "Adam · hidden layers",             "NEURAL NETWORK",       "both"),
    MLMethodDescriptor("pytorch_mlp",         "PyTorch MLP",         "GPU-capable · custom depth",       "NEURAL NETWORK",       "both"),
    # PROBABILISTIC
    MLMethodDescriptor("naive_bayes",         "Naive Bayes",         "Gaussian · classification only",   "PROBABILISTIC",        "classification"),
    MLMethodDescriptor("bayesian_ridge",      "Bayesian Ridge",      "ARD · probabilistic weights",      "PROBABILISTIC",        "regression"),
    # GRADIENT BOOSTING
    MLMethodDescriptor("xgboost",             "XGBoost",             "Extreme gradient boosting",        "GRADIENT BOOSTING",    "both"),
    MLMethodDescriptor("lightgbm",            "LightGBM",            "Leaf-wise histogram boosting",     "GRADIENT BOOSTING",    "both"),
)


# ---------------------------------------------------------------------------
# Task-type compatibility lookup (consumed by gui_dlg_ml_plot._do_generate)
# ---------------------------------------------------------------------------

#: Map method key → supported task_type ("regression" | "classification" | "both").
METHOD_TASK_TYPE: Final[dict[str, str]] = {d.key: d.task_type for d in _ML_REGISTRY}


def validate_task_type(
    method_key: str,
    requested: str,
    y_series=None,
) -> None:
    """
    Raise ValueError if the requested task_type is incompatible with the method.

    ``requested`` is the user's Data-tab selection ("auto"/"regression"/
    "classification").  When ``requested == "auto"`` and ``y_series`` is
    provided, task type is resolved from the target cardinality (matching
    ml_engine._prepare's rule) before the compatibility check.
    """
    supported = METHOD_TASK_TYPE.get(method_key, "both")
    if supported == "both":
        return

    resolved = requested
    if resolved == "auto" and y_series is not None:
        import numpy as _np
        import pandas as _pd
        y_clean = _pd.Series(y_series).dropna()
        if len(y_clean) == 0:
            return
        is_numeric = _pd.api.types.is_numeric_dtype(y_clean)
        if not is_numeric:
            resolved = "classification"
        else:
            vals = y_clean.values
            is_int_like = bool(_np.all(vals == vals.astype(_np.int64)))
            resolved = (
                "classification"
                if (y_clean.nunique() <= 20 and is_int_like)
                else "regression"
            )

    if resolved == "auto" or resolved == supported:
        return
    raise ValueError(
        f"Method {method_key!r} only supports task_type={supported!r}, "
        f"but the target resolves to task_type={resolved!r}. "
        f"Change the Task type on the Data tab or pick a compatible method."
    )


# ---------------------------------------------------------------------------
# _MLButtonDisplayParams — pure-function / data-record pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MLButtonDisplayParams:
    """
    All display parameters for _MLButton derived from module-level constants.

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


def _compute_ml_button_params() -> _MLButtonDisplayParams:
    """
    Pure function: module constants → _MLButtonDisplayParams.

    Makes no Qt calls, writes no state, has no I/O.
    """
    return _MLButtonDisplayParams(
        min_height_px=ML_BUTTON_MIN_HEIGHT_PX,
        title_font_pt=ML_BUTTON_TITLE_FONT_PT,
        subtitle_font_pt=ML_BUTTON_SUBTITLE_FONT_PT,
        h_margin_px=ML_BUTTON_H_MARGIN_PX,
        v_margin_px=ML_BUTTON_V_MARGIN_PX,
        label_spacing_px=ML_BUTTON_LABEL_SPACING_PX,
    )


# ---------------------------------------------------------------------------
# _MLButton — text-only launcher
# ---------------------------------------------------------------------------

class _MLButton(QPushButton):
    """
    Text-only launcher button for a single MLMethodDescriptor.

    Renders a two-line label (title above, subtitle below).  All display
    parameters are applied through a single _apply_display_params call.

    Parameters
    ----------
    descriptor : MLMethodDescriptor
    callback   : zero-argument callable invoked on click
    parent     : optional Qt parent widget
    """

    def __init__(self, descriptor: MLMethodDescriptor, callback, parent=None):
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

        self._apply_display_params(_compute_ml_button_params())
        self.clicked.connect(callback)

    def _apply_display_params(self, params: _MLButtonDisplayParams) -> None:
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

def _make_ml_callback(
    descriptor: MLMethodDescriptor,
    store: SessionStore,
    parent,
):
    """
    Factory: MLMethodDescriptor × SessionStore × parent → zero-arg callable.

    Returns a zero-argument lambda so that PySide6's signal-slot mechanism
    can trim the bool emitted by QPushButton.clicked without binding it to
    the descriptor key.
    """
    return lambda: MLDialog(descriptor.key, store, parent).exec()


# ---------------------------------------------------------------------------
# MLWindow
# ---------------------------------------------------------------------------

class MLWindow(ModuleDialog):
    """
    Machine Learning workspace window.

    Renders one _MLButton per registered MLMethodDescriptor, grouped under
    their declared category headers.  The window is a pure dispatch layer;
    it performs no model fitting.
    """

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Machine Learning",
            hint=(
                "Supervised model training and evaluation across linear, tree-based, "
                "ensemble, kernel, neural network, probabilistic, and gradient-boosting "
                "families — 17 methods with full hyperparameter control and diagnostic plots."
            ),
            category="MODELLING · 17 METHODS",
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
            "Machine learning library",
            "Select a method to configure, train, and evaluate a supervised model "
            "with full diagnostic plots.",
            "17 METHODS",
        ))
        self.body_layout.addSpacing(ML_WORKSPACE_HEADER_SPACING_PX)

        for category in _CATEGORY_ORDER:
            descriptors = [d for d in _ML_REGISTRY if d.category == category]
            if not descriptors:
                continue

            hdr = QLabel(category)
            hdr.setObjectName("ReportPanelEyebrow")
            self.body_layout.addWidget(hdr)

            # Wrap into rows of ML_BUTTONS_PER_ROW
            for row_start in range(0, len(descriptors), ML_BUTTONS_PER_ROW):
                row_descs = descriptors[row_start : row_start + ML_BUTTONS_PER_ROW]
                row = QGridLayout()
                row.setHorizontalSpacing(ML_ROW_H_SPACING_PX)
                row.setVerticalSpacing(0)

                for col, descriptor in enumerate(row_descs):
                    btn = _MLButton(
                        descriptor,
                        _make_ml_callback(descriptor, self._store, self),
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
            self.body_layout.addSpacing(ML_POST_DIVIDER_SPACING_PX)

        self.body_layout.addStretch(1)
