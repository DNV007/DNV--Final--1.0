"""
DNV Scientific Module
---------------------
Role:
    Provides shared, fully customisable plot-style infrastructure —
    make_style_panel, make_export_panel, apply_style, finalise_figure —
    used by every analysis dialog (Visualization, Correlation, Feature
    Ranking, Descriptors, Machine Learning).

Scientific Context:
    Centralises all user-facing matplotlib/seaborn style controls into a
    single module so that every plot in the application is fully and
    uniformly customisable.  No scientific computation.

Invariants:
    - make_style_panel always returns (widget, getter) where getter is a
      zero-arg callable returning a complete style dict.
    - apply_style never raises; all mutations are guarded by try/except.
    - finalise_figure calls apply_style on every axes in the figure.

Assumptions:
    - matplotlib and seaborn are available.
    - Qt application instance is initialised before any widget is created.

Failure Modes:
    - Invalid tick format string: silently ignored.
    - Colorbar on an axes without a mappable: no-op.

Provenance:
    - This module emits no transformation metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

from PySide6.QtCore    import Signal, Qt
from PySide6.QtGui     import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

import matplotlib.ticker as mticker
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

#: Default title font size [pt].
DEFAULT_TITLE_FONTSIZE_PT: Final[int] = 13

#: Default suptitle font size [pt].
DEFAULT_SUPTITLE_FONTSIZE_PT: Final[int] = 14

#: Default axis-label font size [pt].
DEFAULT_LABEL_FONTSIZE_PT: Final[int] = 11

#: Default tick-label font size [pt].
DEFAULT_TICK_FONTSIZE_PT: Final[int] = 9

#: Default legend font size [pt].
DEFAULT_LEGEND_FONTSIZE_PT: Final[int] = 9

#: Default grid alpha.
DEFAULT_GRID_ALPHA: Final[float] = 0.3

#: Default minor-grid alpha.
DEFAULT_MINOR_GRID_ALPHA: Final[float] = 0.15

#: Default spine colour.
DEFAULT_SPINE_COLOR: Final[str] = "#cccccc"

#: Default axes background colour.
DEFAULT_BG_COLOR: Final[str] = "#ffffff"

#: Minimum width for the label column in style rows [px].
STYLE_LABEL_MIN_WIDTH_PX: Final[int] = 170

#: Height of the per-tick rename text areas [px].
TICK_MAP_HEIGHT_PX: Final[int] = 90

#: Maximum number of tick labels shown in the sync-generated placeholder [count].
TICK_MAP_PLACEHOLDER_LIMIT: Final[int] = 12


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

FONT_FAMILIES: Final[List[str]] = [
    "sans-serif", "serif", "monospace",
    "Segoe UI", "Arial", "Helvetica", "Times New Roman",
    "DejaVu Sans", "Courier New",
]

LEGEND_LOCS: Final[List[str]] = [
    "best", "upper right", "upper left", "lower right", "lower left",
    "center", "right", "upper center", "lower center",
    "center left", "center right",
]

SCALES: Final[List[str]] = ["linear", "log", "symlog", "logit"]

GRID_LINESTYLES: Final[List[str]] = ["--", "-", ":", "-."]

SNS_STYLES: Final[List[str]] = [
    "whitegrid", "darkgrid", "white", "dark", "ticks",
]


# ---------------------------------------------------------------------------
# Widget helpers (public — usable by all dialogs)
# ---------------------------------------------------------------------------

class ColorBtn(QPushButton):
    """Colour picker button that displays and stores a hex colour."""
    colorChanged = Signal(str)

    def __init__(self, initial: str = "#1f77b4", parent=None):
        super().__init__(parent)
        self._hex = initial
        self._refresh()
        self.setFixedSize(32, 24)
        self.clicked.connect(self._pick)

    def _refresh(self):
        self.setStyleSheet(
            f"background:{self._hex}; border:1px solid #666; border-radius:3px;"
        )

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._hex), self, "Pick colour")
        if c.isValid():
            self._hex = c.name()
            self._refresh()
            self.colorChanged.emit(self._hex)

    def color(self) -> str:
        return self._hex


def hrow(label: str, widget: QWidget) -> QWidget:
    """Label + widget in an HBox row."""
    w = QWidget(); lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
    lbl = QLabel(label); lbl.setMinimumWidth(STYLE_LABEL_MIN_WIDTH_PX)
    lay.addWidget(lbl); lay.addWidget(widget); lay.addStretch(1)
    return w


def eyebrow(text: str) -> QLabel:
    """Section eyebrow label."""
    lbl = QLabel(text); lbl.setObjectName("ReportPanelEyebrow")
    return lbl


def section_sep() -> QFrame:
    """Thin horizontal divider."""
    sep = QFrame(); sep.setObjectName("Divider")
    sep.setFixedHeight(1); sep.setFrameShape(QFrame.HLine)
    return sep


def scrolled(inner: QWidget) -> QScrollArea:
    """Wrap *inner* in a frameless QScrollArea."""
    sa = QScrollArea(); sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame); sa.setWidget(inner)
    return sa


def combo(items, current=None) -> QComboBox:
    cb = QComboBox(); cb.addItems(items)
    if current and current in items:
        cb.setCurrentText(current)
    return cb


def spin(lo, hi, val, step=1) -> QSpinBox:
    s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    return s


def dspin(lo, hi, val, step=0.05) -> QDoubleSpinBox:
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    return s


def check(default: bool) -> QCheckBox:
    c = QCheckBox(); c.setChecked(default); return c


# ---------------------------------------------------------------------------
# Full style panel
# ---------------------------------------------------------------------------

def make_style_panel(
    defaults: Optional[Dict[str, Any]] = None,
) -> Tuple[QWidget, Callable[[], dict]]:
    """Build the universal style panel with every customisation control.

    Parameters
    ----------
    defaults : dict, optional
        Override any default value.  Keys match the getter output keys.

    Returns
    -------
    (widget, getter)  where getter() → dict of all current control values.
    """
    d = defaults or {}
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(5)
    controls: Dict[str, Any] = {}

    def _add(label, widget, key):
        controls[key] = widget; lay.addWidget(hrow(label, widget))

    def _sec(title):
        lay.addWidget(section_sep()); lay.addWidget(eyebrow(title))

    # ── LABELS ──────────────────────────────────────────────────────────
    lay.addWidget(eyebrow("LABELS"))
    _add("Title:", QLineEdit(), "title")
    _add("X label:", QLineEdit(), "xlabel")
    _add("Y label:", QLineEdit(), "ylabel")
    _add("Figure suptitle:", QLineEdit(), "suptitle")
    controls["suptitle"].setPlaceholderText("none")

    # ── TYPOGRAPHY ──────────────────────────────────────────────────────
    _sec("TYPOGRAPHY")
    _add("Font family:", combo(FONT_FAMILIES), "font_family")
    _add("Title size:", spin(6, 36, d.get("title_fontsize", DEFAULT_TITLE_FONTSIZE_PT)), "title_fontsize")
    _add("Suptitle size:", spin(6, 36, d.get("suptitle_fontsize", DEFAULT_SUPTITLE_FONTSIZE_PT)), "suptitle_fontsize")
    _add("Axis label size:", spin(6, 28, d.get("label_fontsize", DEFAULT_LABEL_FONTSIZE_PT)), "label_fontsize")
    _add("Tick label size:", spin(5, 22, d.get("tick_fontsize", DEFAULT_TICK_FONTSIZE_PT)), "tick_fontsize")
    _add("LaTeX rendering:", check(d.get("latex", False)), "latex")

    # ── TICK FORMATTING ────────────────────────────────────────────────
    _sec("TICK FORMATTING")
    _add("X tick rotation °:", spin(-90, 90, d.get("tick_rotation_x", 0)), "tick_rotation_x")
    _add("Y tick rotation °:", spin(-90, 90, d.get("tick_rotation_y", 0)), "tick_rotation_y")
    xfmt = QLineEdit(); xfmt.setPlaceholderText("auto  e.g. %.2f  %d  sci")
    _add("X tick format:", xfmt, "tick_format_x")
    yfmt = QLineEdit(); yfmt.setPlaceholderText("auto  e.g. %.2f  %d  sci")
    _add("Y tick format:", yfmt, "tick_format_y")
    _add("Max X ticks (0=auto):", spin(0, 50, d.get("max_xticks", 0)), "max_xticks")
    _add("Max Y ticks (0=auto):", spin(0, 50, d.get("max_yticks", 0)), "max_yticks")
    _add("X category stride (1=all):", spin(1, 50, d.get("xtick_stride", 1)), "xtick_stride")
    _add("Y category stride (1=all):", spin(1, 50, d.get("ytick_stride", 1)), "ytick_stride")
    _add("Tick length (pt):", dspin(0.0, 20.0, d.get("tick_length", 4.0), 0.5), "tick_length")
    _add("Tick width (pt):", dspin(0.0, 5.0, d.get("tick_width", 0.8), 0.1), "tick_width")

    # Per-tick label rewriting.  One mapping per line: "original -> display".
    # Lines without "->" are treated as ordered positional replacements.
    xmap = QPlainTextEdit(); xmap.setFixedHeight(TICK_MAP_HEIGHT_PX)
    xmap.setPlaceholderText("original -> display  (one per line)")
    _add("X tick rename:", xmap, "xtick_label_map")
    ymap = QPlainTextEdit(); ymap.setFixedHeight(TICK_MAP_HEIGHT_PX)
    ymap.setPlaceholderText("original -> display  (one per line)")
    _add("Y tick rename:", ymap, "ytick_label_map")

    # ── AXES & SCALE ──────────────────────────────────────────────────
    _sec("AXES & SCALE")
    _add("X scale:", combo(SCALES), "xscale")
    _add("Y scale:", combo(SCALES), "yscale")
    xlim_lo = QLineEdit(); xlim_lo.setPlaceholderText("auto")
    xlim_hi = QLineEdit(); xlim_hi.setPlaceholderText("auto")
    ylim_lo = QLineEdit(); ylim_lo.setPlaceholderText("auto")
    ylim_hi = QLineEdit(); ylim_hi.setPlaceholderText("auto")
    _add("X min:", xlim_lo, "xlim_lo"); _add("X max:", xlim_hi, "xlim_hi")
    _add("Y min:", ylim_lo, "ylim_lo"); _add("Y max:", ylim_hi, "ylim_hi")
    _add("Invert X axis:", check(d.get("invert_x", False)), "invert_x")
    _add("Invert Y axis:", check(d.get("invert_y", False)), "invert_y")
    _add("Equal aspect ratio:", check(d.get("aspect_equal", False)), "aspect_equal")

    # ── GRID ──────────────────────────────────────────────────────────
    _sec("GRID")
    _add("Major grid:", check(d.get("grid", True)), "grid")
    _add("Grid alpha:", dspin(0.05, 1.0, d.get("grid_alpha", DEFAULT_GRID_ALPHA), 0.05), "grid_alpha")
    _add("Grid line style:", combo(GRID_LINESTYLES), "grid_linestyle")
    _add("Minor grid:", check(d.get("minor_grid", False)), "minor_grid")
    _add("Minor grid alpha:", dspin(0.05, 0.5, d.get("minor_grid_alpha", DEFAULT_MINOR_GRID_ALPHA), 0.05), "minor_grid_alpha")
    _add("Minor grid line style:", combo(GRID_LINESTYLES, ":"), "minor_grid_linestyle")

    # ── SPINES ────────────────────────────────────────────────────────
    _sec("SPINES")
    _add("Top spine:", check(d.get("spine_top", True)), "spine_top")
    _add("Right spine:", check(d.get("spine_right", True)), "spine_right")
    _add("Bottom spine:", check(d.get("spine_bottom", True)), "spine_bottom")
    _add("Left spine:", check(d.get("spine_left", True)), "spine_left")
    _add("Spine colour:", ColorBtn(d.get("spine_color", DEFAULT_SPINE_COLOR)), "spine_color")
    _add("Spine width (pt):", dspin(0.0, 5.0, d.get("spine_width", 0.8), 0.1), "spine_width")

    # ── LEGEND ────────────────────────────────────────────────────────
    _sec("LEGEND")
    _add("Show legend:",       check(d.get("legend", True)),                "legend")
    _add("Location:",          combo(LEGEND_LOCS),                          "legend_loc")
    _add("Outside axes:",      check(d.get("legend_outside", False)),       "legend_outside")
    _add("Outside X:",         dspin(0.0, 2.0, d.get("legend_bbox_x", 1.01), 0.01), "legend_bbox_x")
    _add("Outside Y:",         dspin(0.0, 2.0, d.get("legend_bbox_y", 1.0),  0.01), "legend_bbox_y")
    _add("Font size:",         spin(6, 20, d.get("legend_fontsize", DEFAULT_LEGEND_FONTSIZE_PT)), "legend_fontsize")
    _add("Title:",             QLineEdit(),                                 "legend_title")
    controls["legend_title"].setPlaceholderText("none")
    _add("Title font size:",   spin(6, 20, d.get("legend_title_fontsize", DEFAULT_LEGEND_FONTSIZE_PT)), "legend_title_fontsize")
    _add("Columns (ncol):",    spin(1, 10, d.get("legend_ncol", 1)),        "legend_ncol")
    _add("Frame:",             check(d.get("legend_frameon", True)),         "legend_frameon")
    _add("Frame alpha:",       dspin(0.0, 1.0, d.get("legend_framealpha", 0.8), 0.05), "legend_framealpha")
    _add("Face color:",        ColorBtn(d.get("legend_facecolor", "#ffffff")), "legend_facecolor")
    _add("Edge color:",        ColorBtn(d.get("legend_edgecolor", "#cccccc")), "legend_edgecolor")
    _add("Fancy box:",         check(d.get("legend_fancybox", False)),       "legend_fancybox")
    _add("Shadow:",            check(d.get("legend_shadow", False)),         "legend_shadow")
    _add("Marker scale:",      dspin(0.5, 5.0, d.get("legend_markerscale", 1.0), 0.1), "legend_markerscale")
    _add("Handle length:",     dspin(0.5, 5.0, d.get("legend_handlelength", 2.0), 0.1), "legend_handlelength")
    _add("Label spacing:",     dspin(0.1, 3.0, d.get("legend_labelspacing", 0.5), 0.1), "legend_labelspacing")
    _add("Border pad:",        dspin(0.1, 2.0, d.get("legend_borderpad", 0.4), 0.1), "legend_borderpad")
    _add("Handle-text pad:",   dspin(0.1, 3.0, d.get("legend_handletextpad", 0.8), 0.1), "legend_handletextpad")
    _add("Column spacing:",    dspin(0.5, 5.0, d.get("legend_columnspacing", 2.0), 0.1), "legend_columnspacing")

    # ── BACKGROUND & LAYOUT ──────────────────────────────────────────
    _sec("BACKGROUND & LAYOUT")
    _add("Axes background:",   ColorBtn(d.get("bg", DEFAULT_BG_COLOR)),     "bg")
    _add("Figure background:", ColorBtn(d.get("fig_facecolor", DEFAULT_BG_COLOR)), "fig_facecolor")
    _add("Seaborn style:",     combo(SNS_STYLES),                           "sns_style")
    _add("Tight layout:",      check(d.get("tight_layout", True)),          "tight_layout")
    _add("Fig width (in, 0=auto):",  dspin(0.0, 40.0, d.get("fig_width", 0.0), 0.5), "fig_width")
    _add("Fig height (in, 0=auto):", dspin(0.0, 30.0, d.get("fig_height", 0.0), 0.5), "fig_height")
    _add("Colorbar label:",    QLineEdit(),                                 "colorbar_label")
    controls["colorbar_label"].setPlaceholderText("none")

    lay.addStretch(1)

    # ── Getter ────────────────────────────────────────────────────────
    def _getter() -> dict:
        def _f(le):
            try:
                return float(le.text())
            except (ValueError, TypeError):
                return None
        out: dict = {}
        for key, widget in controls.items():
            if isinstance(widget, QDoubleSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QSpinBox):
                out[key] = widget.value()
            elif isinstance(widget, QComboBox):
                out[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, ColorBtn):
                out[key] = widget.color()
            elif isinstance(widget, QPlainTextEdit):
                out[key] = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text().strip()
        out["xlim"] = (_f(xlim_lo), _f(xlim_hi))
        out["ylim"] = (_f(ylim_lo), _f(ylim_hi))
        return out

    # ── sync(fig): push current figure state into placeholders ───────
    # Called by dialogs after a Generate click.  Reads auto-generated
    # labels from the first axes and writes them as placeholder text so
    # the user sees what would be rewritten if left blank.
    def _sync(fig) -> None:
        axes = fig.get_axes()
        if not axes:
            return
        ax = axes[0]
        _set_placeholder(controls.get("title"), ax.get_title() or "")
        _set_placeholder(controls.get("xlabel"), ax.get_xlabel() or "")
        _set_placeholder(controls.get("ylabel"), ax.get_ylabel() or "")
        _set_tick_placeholder(
            controls.get("xtick_label_map"),
            [t.get_text() for t in ax.get_xticklabels() if t.get_text()],
        )
        _set_tick_placeholder(
            controls.get("ytick_label_map"),
            [t.get_text() for t in ax.get_yticklabels() if t.get_text()],
        )
    _getter.sync = _sync  # type: ignore[attr-defined]

    return w, _getter


def _set_placeholder(widget, text: str) -> None:
    if widget is None or not text:
        return
    try:
        widget.setPlaceholderText(text)
    except AttributeError:
        pass


def _set_tick_placeholder(widget, labels: list) -> None:
    if widget is None or not labels:
        return
    shown = labels[:TICK_MAP_PLACEHOLDER_LIMIT]
    more = "" if len(labels) <= TICK_MAP_PLACEHOLDER_LIMIT else f"\n…(+{len(labels) - TICK_MAP_PLACEHOLDER_LIMIT} more)"
    hint = "\n".join(f"{lab} -> {lab}" for lab in shown) + more
    try:
        widget.setPlaceholderText(hint)
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Export panel
# ---------------------------------------------------------------------------

def make_export_panel(
    save_callback: Callable,
    defaults: Optional[Dict[str, Any]] = None,
) -> Tuple[QWidget, Callable[[], dict]]:
    """Build the universal export panel.

    Parameters
    ----------
    save_callback : callable
        The dialog's _save_figure method (connected to the Save button).
    defaults : dict, optional
        Override default export values.
    """
    d = defaults or {}
    w = QWidget(); lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(5)
    controls: Dict[str, Any] = {}

    def _add(label, widget, key):
        controls[key] = widget; lay.addWidget(hrow(label, widget))

    lay.addWidget(eyebrow("OUTPUT FORMAT"))
    _add("Format:", combo(["PNG", "PDF", "SVG", "EPS", "TIFF"]), "fmt")
    _add("DPI:", spin(72, 1200, d.get("dpi", 150)), "dpi")
    _add("Width (in):", dspin(2.0, 24.0, d.get("width", 8.0), 0.5), "width")
    _add("Height (in):", dspin(2.0, 24.0, d.get("height", 6.0), 0.5), "height")
    _add("Transparent bg:", check(d.get("transparent", False)), "transparent")
    lay.addWidget(section_sep())
    lay.addWidget(eyebrow("FILE"))
    pf = QLineEdit(d.get("prefix", "plot"))
    _add("Filename prefix:", pf, "prefix")
    save_btn = QPushButton("Save figure…")
    save_btn.clicked.connect(save_callback)
    lay.addWidget(save_btn, 0, Qt.AlignLeft)
    lay.addStretch(1)

    def _getter() -> dict:
        return {
            "fmt":         controls["fmt"].currentText(),
            "dpi":         controls["dpi"].value(),
            "width":       controls["width"].value(),
            "height":      controls["height"].value(),
            "transparent": controls["transparent"].isChecked(),
            "prefix":      controls["prefix"].text().strip() or "plot",
        }
    return w, _getter


# ---------------------------------------------------------------------------
# Categorical axis detection
# ---------------------------------------------------------------------------

def is_categorical_axis(axis) -> bool:
    """Return True if *axis* uses a StrCategoryLocator or FixedLocator.

    Prevents set_xscale("linear") from destroying categorical tick labels
    (seaborn violin/box/bar/swarm/strip) or heatmap column ticks.
    """
    from matplotlib.ticker import FixedLocator
    loc = axis.get_major_locator()
    if isinstance(loc, FixedLocator):
        return True
    try:
        from matplotlib.category import StrCategoryLocator
        return isinstance(loc, StrCategoryLocator)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# RC context builder
# ---------------------------------------------------------------------------

def rc_ctx(sp: dict) -> dict:
    """Build an matplotlib rc-params dict from style params."""
    rc: dict = {}
    if sp.get("latex"):
        rc["text.usetex"] = True; rc["font.family"] = "serif"
    elif sp.get("font_family"):
        rc["font.family"] = sp["font_family"]
    return rc


# ---------------------------------------------------------------------------
# Tick format helper
# ---------------------------------------------------------------------------

def apply_tick_format(ax, axis: str, fmt_str: str) -> None:
    """Apply a user-specified tick format to an axis ('x' or 'y')."""
    if not fmt_str or fmt_str.lower() in ("", "auto"):
        return
    the_axis = ax.xaxis if axis == "x" else ax.yaxis
    if is_categorical_axis(the_axis):
        return
    if fmt_str.lower() == "sci":
        the_axis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        the_axis.get_major_formatter().set_scientific(True)
        the_axis.get_major_formatter().set_powerlimits((-2, 2))
    else:
        try:
            the_axis.set_major_formatter(mticker.FormatStrFormatter(fmt_str))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tick label rewriting
# ---------------------------------------------------------------------------

def parse_label_map(text: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse a user-entered tick-rename textarea.

    Two forms accepted, mixable:
      * ``original -> display``  — keyed rename (order-independent)
      * ``display``               — positional rename (ith line → ith tick)

    Returns
    -------
    (keyed, positional)
        keyed      : dict mapping original label → display label
        positional : list of display labels in file order, applied to any
                     ticks not covered by the keyed map
    """
    keyed: Dict[str, str] = {}
    positional: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "->" in line:
            lhs, _, rhs = line.partition("->")
            keyed[lhs.strip()] = rhs.strip()
        else:
            positional.append(line)
    return keyed, positional


def apply_label_map(ax, axis: str, text: str) -> None:
    """Rewrite tick labels on *axis* ('x' or 'y') using a user-entered map.

    Safe on any axis type: if the axis has no locatable labels, this is a
    no-op.  Does not re-locate ticks — only rewrites the display strings.
    """
    if not text or not text.strip():
        return
    keyed, positional = parse_label_map(text)
    if not keyed and not positional:
        return
    the_axis = ax.xaxis if axis == "x" else ax.yaxis
    ticks = list(the_axis.get_majorticklocs())
    if not ticks:
        return
    current = [t.get_text() for t in the_axis.get_majorticklabels()]
    # If labels haven't been populated yet (formatter not run), fall back
    # to stringified tick positions so positional rename still works.
    if not any(current):
        current = [str(t) for t in ticks]
    new_labels: List[str] = []
    for idx, lab in enumerate(current):
        if lab in keyed:
            new_labels.append(keyed[lab])
        elif idx < len(positional):
            new_labels.append(positional[idx])
        else:
            new_labels.append(lab)
    try:
        the_axis.set_major_locator(mticker.FixedLocator(ticks))
        the_axis.set_ticklabels(new_labels)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# thin_ticks — tick density control for numeric AND categorical axes
# ---------------------------------------------------------------------------

def thin_ticks(ax, sp: dict) -> None:
    """Reduce tick density to prevent overlapping-label 'black patch'.

    Two independent mechanisms:
      * Numeric axes (MaxNLocator cap) — honours max_xticks / max_yticks.
      * Categorical / FixedLocator axes — honours xtick_stride / ytick_stride
        by keeping every Nth tick and blanking the rest.  MaxNLocator would
        destroy the string labels (seaborn bar/violin, heatmap column names),
        so stride is the only safe thinning operation for these axes.
    """
    max_xt = sp.get("max_xticks", 0) or 0
    max_yt = sp.get("max_yticks", 0) or 0
    xstride = sp.get("xtick_stride", 1) or 1
    ystride = sp.get("ytick_stride", 1) or 1

    def _apply(axis_obj, max_n: int, stride: int) -> None:
        if is_categorical_axis(axis_obj):
            if stride and stride > 1:
                locs = list(axis_obj.get_majorticklocs())
                labels = [t.get_text() for t in axis_obj.get_majorticklabels()]
                kept_labels = [
                    lbl if (i % stride == 0) else ""
                    for i, lbl in enumerate(labels)
                ]
                try:
                    axis_obj.set_major_locator(mticker.FixedLocator(locs))
                    axis_obj.set_ticklabels(kept_labels)
                except Exception:
                    pass
        else:
            if max_n and max_n > 0:
                try:
                    axis_obj.set_major_locator(mticker.MaxNLocator(nbins=max_n))
                except Exception:
                    pass

    _apply(ax.xaxis, max_xt, xstride)
    _apply(ax.yaxis, max_yt, ystride)


# ---------------------------------------------------------------------------
# apply_style — single mutation site for all per-axes style state
# ---------------------------------------------------------------------------

def apply_style(ax, sp: dict, fig: Figure = None) -> None:
    """Apply the full style dict *sp* to a single matplotlib Axes.

    This is the **single site** that writes display state to an axes.
    Renderers must never set display attributes directly.
    """
    # ── Labels & Typography ───────────────────────────────────────────
    if sp.get("title"):
        ax.set_title(sp["title"], fontsize=sp.get("title_fontsize", DEFAULT_TITLE_FONTSIZE_PT))
    if sp.get("xlabel"):
        ax.set_xlabel(sp["xlabel"], fontsize=sp.get("label_fontsize", DEFAULT_LABEL_FONTSIZE_PT))
    if sp.get("ylabel"):
        ax.set_ylabel(sp["ylabel"], fontsize=sp.get("label_fontsize", DEFAULT_LABEL_FONTSIZE_PT))

    # ── Tick params ───────────────────────────────────────────────────
    tick_kw: dict = {"labelsize": sp.get("tick_fontsize", DEFAULT_TICK_FONTSIZE_PT)}
    tick_len = sp.get("tick_length")
    tick_wid = sp.get("tick_width")
    if tick_len is not None and tick_len > 0:
        tick_kw["length"] = tick_len
    if tick_wid is not None and tick_wid > 0:
        tick_kw["width"] = tick_wid
    ax.tick_params(**tick_kw)
    ax.tick_params(axis="x", rotation=sp.get("tick_rotation_x", 0))
    ax.tick_params(axis="y", rotation=sp.get("tick_rotation_y", 0))

    apply_tick_format(ax, "x", sp.get("tick_format_x", ""))
    apply_tick_format(ax, "y", sp.get("tick_format_y", ""))

    # ── Scales (must precede thin_ticks: set_*scale resets locators) ─
    if not is_categorical_axis(ax.xaxis):
        try:
            ax.set_xscale(sp.get("xscale", "linear"))
        except Exception:
            pass
    if not is_categorical_axis(ax.yaxis):
        try:
            ax.set_yscale(sp.get("yscale", "linear"))
        except Exception:
            pass

    # ── Tick density control (prevents black-patch overlap) ─────────
    thin_ticks(ax, sp)

    # ── Per-tick label rewriting ─────────────────────────────────────
    apply_label_map(ax, "x", sp.get("xtick_label_map", ""))
    apply_label_map(ax, "y", sp.get("ytick_label_map", ""))

    # ── Limits ────────────────────────────────────────────────────────
    xlim = sp.get("xlim", (None, None))
    ylim = sp.get("ylim", (None, None))
    if xlim and any(v is not None for v in xlim):
        cur = list(ax.get_xlim())
        if xlim[0] is not None: cur[0] = xlim[0]
        if xlim[1] is not None: cur[1] = xlim[1]
        try:
            ax.set_xlim(cur)
        except Exception:
            pass
    if ylim and any(v is not None for v in ylim):
        cur = list(ax.get_ylim())
        if ylim[0] is not None: cur[0] = ylim[0]
        if ylim[1] is not None: cur[1] = ylim[1]
        try:
            ax.set_ylim(cur)
        except Exception:
            pass

    # ── Inversion / aspect ────────────────────────────────────────────
    if sp.get("invert_x"):
        ax.invert_xaxis()
    if sp.get("invert_y"):
        ax.invert_yaxis()
    if sp.get("aspect_equal"):
        try:
            ax.set_aspect("equal")
        except Exception:
            pass

    # ── Grid ─────────────────────────────────────────────────────────
    if sp.get("grid"):
        ax.grid(True, alpha=sp.get("grid_alpha", DEFAULT_GRID_ALPHA),
                linestyle=sp.get("grid_linestyle", "--"))
    if sp.get("minor_grid"):
        ax.minorticks_on()
        ax.grid(True, which="minor",
                alpha=sp.get("minor_grid_alpha", DEFAULT_MINOR_GRID_ALPHA),
                linestyle=sp.get("minor_grid_linestyle", ":"))

    # ── Background ────────────────────────────────────────────────────
    ax.set_facecolor(sp.get("bg", DEFAULT_BG_COLOR))

    # ── Spines ────────────────────────────────────────────────────────
    spine_w = sp.get("spine_width", 0.8)
    for spine_name in ("top", "right", "bottom", "left"):
        vis = sp.get(f"spine_{spine_name}", True)
        ax.spines[spine_name].set_visible(vis)
        if vis:
            ax.spines[spine_name].set_color(sp.get("spine_color", DEFAULT_SPINE_COLOR))
            ax.spines[spine_name].set_linewidth(spine_w)

    # ── Legend ────────────────────────────────────────────────────────
    if not sp.get("legend", True):
        leg = ax.get_legend()
        if leg:
            leg.remove()
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            try:
                legend_kw: dict = dict(
                    handles=handles,
                    labels=labels,
                    fontsize=sp.get("legend_fontsize", DEFAULT_LEGEND_FONTSIZE_PT),
                    ncol=sp.get("legend_ncol", 1),
                    title=sp.get("legend_title", "") or None,
                    title_fontsize=sp.get("legend_title_fontsize", DEFAULT_LEGEND_FONTSIZE_PT),
                    frameon=sp.get("legend_frameon", True),
                    framealpha=sp.get("legend_framealpha", 0.8),
                    facecolor=sp.get("legend_facecolor", DEFAULT_BG_COLOR),
                    edgecolor=sp.get("legend_edgecolor", DEFAULT_SPINE_COLOR),
                    fancybox=sp.get("legend_fancybox", False),
                    shadow=sp.get("legend_shadow", False),
                    markerscale=sp.get("legend_markerscale", 1.0),
                    handlelength=sp.get("legend_handlelength", 2.0),
                    labelspacing=sp.get("legend_labelspacing", 0.5),
                    borderpad=sp.get("legend_borderpad", 0.4),
                    handletextpad=sp.get("legend_handletextpad", 0.8),
                    columnspacing=sp.get("legend_columnspacing", 2.0),
                )
                if sp.get("legend_outside", False):
                    legend_kw["bbox_to_anchor"] = (
                        sp.get("legend_bbox_x", 1.01),
                        sp.get("legend_bbox_y", 1.0),
                    )
                    legend_kw["loc"] = "upper left"
                else:
                    legend_kw["loc"] = sp.get("legend_loc", "best")
                ax.legend(**legend_kw)
            except Exception:
                pass

    # ── Figure-level (if provided) ────────────────────────────────────
    if fig is not None:
        fig.set_facecolor(sp.get("fig_facecolor", DEFAULT_BG_COLOR))
        suptitle = sp.get("suptitle", "")
        if suptitle:
            fig.suptitle(suptitle, fontsize=sp.get("suptitle_fontsize", DEFAULT_SUPTITLE_FONTSIZE_PT))


# ---------------------------------------------------------------------------
# _build_legend_kw — reusable legend kwargs builder
# ---------------------------------------------------------------------------

def _build_legend_kw(sp: dict, handles, labels) -> dict:
    """Build legend kwargs dict from style params, handles, and labels."""
    legend_kw: dict = dict(
        handles=handles,
        labels=labels,
        fontsize=sp.get("legend_fontsize", DEFAULT_LEGEND_FONTSIZE_PT),
        ncol=sp.get("legend_ncol", 1),
        title=sp.get("legend_title", "") or None,
        title_fontsize=sp.get("legend_title_fontsize", DEFAULT_LEGEND_FONTSIZE_PT),
        frameon=sp.get("legend_frameon", True),
        framealpha=sp.get("legend_framealpha", 0.8),
        facecolor=sp.get("legend_facecolor", DEFAULT_BG_COLOR),
        edgecolor=sp.get("legend_edgecolor", DEFAULT_SPINE_COLOR),
        fancybox=sp.get("legend_fancybox", False),
        shadow=sp.get("legend_shadow", False),
        markerscale=sp.get("legend_markerscale", 1.0),
        handlelength=sp.get("legend_handlelength", 2.0),
        labelspacing=sp.get("legend_labelspacing", 0.5),
        borderpad=sp.get("legend_borderpad", 0.4),
        handletextpad=sp.get("legend_handletextpad", 0.8),
        columnspacing=sp.get("legend_columnspacing", 2.0),
    )
    if sp.get("legend_outside", False):
        legend_kw["bbox_to_anchor"] = (
            sp.get("legend_bbox_x", 1.01),
            sp.get("legend_bbox_y", 1.0),
        )
        legend_kw["loc"] = "upper left"
    else:
        legend_kw["loc"] = sp.get("legend_loc", "best")
    return legend_kw


# ---------------------------------------------------------------------------
# finalise_figure — figure-level style + layout
# ---------------------------------------------------------------------------

def finalise_figure(fig: Figure, sp: dict) -> Figure:
    """Apply figure-level style and layout to every axes.

    Pipeline:
      1. Set figure background and suptitle.
      2. Apply user-specified figure dimensions.
      3. Migrate figure-level legends (seaborn hue=) to owning axes.
      4. Call apply_style on every axes.
      5. Apply tight_layout.
    """
    fig.set_facecolor(sp.get("fig_facecolor", DEFAULT_BG_COLOR))
    suptitle = sp.get("suptitle", "")
    if suptitle:
        fig.suptitle(suptitle, fontsize=sp.get("suptitle_fontsize", DEFAULT_SUPTITLE_FONTSIZE_PT))

    # User-specified figure dimensions
    fw = sp.get("fig_width", 0.0)
    fh = sp.get("fig_height", 0.0)
    if fw and fh and fw > 0 and fh > 0:
        fig.set_size_inches(fw, fh)

    # Migrate figure-level legends (seaborn hue=) to their owning axes
    for fig_leg in list(fig.legends):
        handles = fig_leg.legend_handles
        labels = [t.get_text() for t in fig_leg.get_texts()]
        fig_leg.remove()
        target = next(
            (a for a in fig.get_axes() if a.get_lines() or a.collections),
            None,
        )
        if target is not None and handles:
            try:
                target.legend(**_build_legend_kw(sp, handles, labels))
            except Exception:
                pass

    for ax in fig.get_axes():
        apply_style(ax, sp, fig)

    if sp.get("tight_layout", True):
        try:
            fig.tight_layout()
        except Exception:
            pass
    return fig
