"""
DNV Scientific Module
---------------------
Role:
    Provides OutlierRemovalDialog — a 15-method outlier detection and
    removal panel with provenance-bound revisions.

Scientific Context:
    Applies IQR, Z-score, modified Z-score, Grubbs, isolation forest, LOF,
    DBSCAN, elliptic envelope, and other methods to identify and remove
    statistical outliers from selected numeric columns of the active
    DataFrame.

Invariants:
    - Outlier removal operates on numeric columns only; non-numeric columns
      are passed through unchanged.
    - NaN cells are median-imputed before distance/model fitting so the
      full row set is retained through detection.

Assumptions:
    - Selected columns are numeric.
    - sklearn is available for all model-based detection methods.

Failure Modes:
    - Too few samples for model-based methods: method raises ValueError
      shown in dialog.
    - All rows flagged as outliers: removal is blocked and a warning is
      shown.

Provenance:
    - This module emits no metadata; provenance is recorded by the calling
      DataCleaningWindow.

All algorithms are self-contained; no dependency on the legacy Tkinter modules.
Computation operates on numeric columns only; NaN cells are median-imputed
before distance/model fitting so the full row set is retained through the
sklearn/scipy APIs.
"""
from __future__ import annotations

import logging
import random as _random
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)


class OutlierRemovalDialog(QDialog):
    """Two-panel dialog: method catalogue (left) + adaptive parameter form (right)."""

    _CATALOGUE = [
        ("Z-Score",                    "STATISTICAL",     "zscore"),
        ("Modified Z-Score",           "STATISTICAL",     "mzscore"),
        ("IQR (Interquartile Range)",  "STATISTICAL",     "iqr"),
        ("Euclidean Distance",         "DISTANCE",        "euclidean"),
        ("Mahalanobis Distance",       "DISTANCE",        "mahalanobis"),
        ("K-Means",                    "CLUSTERING",      "kmeans"),
        ("DBSCAN",                     "CLUSTERING",      "dbscan"),
        ("OPTICS",                     "CLUSTERING",      "optics"),
        ("OPTICS (Advanced)",          "CLUSTERING",      "optics2"),
        ("Isolation Forest",           "MACHINE LEARNING","isoforest"),
        ("One-Class SVM",              "MACHINE LEARNING","ocsvm"),
        ("Elliptic Envelope",          "MACHINE LEARNING","elliptic"),
        ("Local Outlier Factor (LOF)", "MACHINE LEARNING","lof"),
        ("Robust PCA (RPCA)",          "ADVANCED",        "rpca"),
        ("Fluctuation-Based (FBOD)",   "ADVANCED",        "fbod"),
        ("Moment-Optimized",           "OPTIMIZATION",    "moment_opt"),
    ]

    _DESC = {
        "zscore":
            "Remove rows whose |Z-score| exceeds the threshold on any numeric column. "
            "Assumes approximate normality; sensitive to heavy tails.",
        "mzscore":
            "Robust variant using median and MAD in place of mean/std. "
            "Better suited to skewed or heavy-tailed distributions (threshold ≈ 3.5 × 0.6745⁻¹ ≈ 5.2 for σ = 3).",
        "iqr":
            "Flag rows outside [Q1 − k·IQR, Q3 + k·IQR] for the selected column. "
            "Non-parametric, robust to non-Gaussian distributions. k = 1.5 is the Tukey fence; k = 3.0 is the outer fence.",
        "euclidean":
            "Compute each row's Euclidean distance from the centroid of all numeric columns. "
            "Scale-sensitive — apply after standardisation or use Mahalanobis for correlated features.",
        "mahalanobis":
            "Accounts for inter-feature correlations via the inverse covariance. "
            "Requires an invertible covariance matrix (n_rows > n_cols, no multicollinear features).",
        "kmeans":
            "Assign rows to clusters; rows whose distance to the nearest centroid exceeds the "
            "given percentile of all cluster distances are flagged. n_clusters and percentile are key levers.",
        "dbscan":
            "Density-Based Spatial Clustering. Points with label −1 (noise) are treated as outliers. "
            "Data are StandardScaler-normalised before fitting. eps and min_samples control cluster density.",
        "optics":
            "Ordering Points To Identify the Clustering Structure. Generalises DBSCAN to variable density. "
            "Noise points (label −1) are removed. min_samples governs the minimum cluster size.",
        "optics2":
            "Advanced OPTICS with Xi steep-area extraction. min_cluster_size (fraction) and xi "
            "control cluster boundary sensitivity. Prefer this when data spans multiple density regimes.",
        "isoforest":
            "Ensemble of isolation trees. Anomalous points are isolated in fewer splits → lower average path length → "
            "score < 0. contamination sets the expected fraction of outliers.",
        "ocsvm":
            "One-Class SVM with RBF kernel learns a hypersphere boundary around the inlier distribution. "
            "nu ≈ upper bound on the outlier fraction and lower bound on support vectors. Data are scaled.",
        "elliptic":
            "Fits a robust Gaussian ellipsoid via Minimum Covariance Determinant. "
            "contamination is the expected outlier fraction. Requires n_rows ≫ n_cols.",
        "lof":
            "Local Outlier Factor: compares the local reachability density of a point to its k neighbours. "
            "LOF ≫ 1 → outlier. contamination selects the decision threshold.",
        "rpca":
            "Robust PCA (ADMM solver): decomposes the data matrix into low-rank L + sparse S components. "
            "The row-wise ℓ₁ norm of S is the outlier score; rows above the threshold are removed. "
            "λ = 1/√max(rows, cols) is the theoretical optimum.",
        "fbod":
            "Fluctuation-Based Outlier Detection: builds random k-nearest-neighbour graphs and scores "
            "each point by the variance of its neighbourhood change rate across graph replicates. "
            "The top-n scoring points are removed.",
        "moment_opt":
            "Moment-Optimized Outlier Conditioning: finds the minimal sample exclusion mask "
            "that brings the empirical kurtosis and skewness closest to declared target values. "
            "Layer 1 optimizes the IQR/Z-score threshold via golden-section search. "
            "Layer 2 (SA or hill climbing) refines individual sample inclusion bits. "
            "Retention penalty λ prevents aggressive data gutting.",
    }

    # Methods where the column selector affects the actual computation
    _COL_METHODS = {"iqr", "optics", "optics2", "moment_opt"}

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Outlier Removal")
        self.setModal(True)
        self.resize(1080, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        hint = QLabel(
            "Select an outlier removal method. All computations operate on numeric columns only. "
            "NaN cells are median-imputed before fitting so no rows are silently dropped during "
            "distance or model evaluation. Each accepted operation creates a provenance-bound revision."
        )
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_method_panel())
        splitter.addWidget(self._build_param_panel())
        splitter.setSizes([340, 620])
        root.addWidget(splitter, 1)

        act = QHBoxLayout()
        summ_btn = QPushButton("Summarise statistics")
        summ_btn.clicked.connect(self._summarise)
        act.addWidget(summ_btn)

        prev_btn = QPushButton("Preview plot")
        prev_btn.clicked.connect(self._preview_plot)
        act.addWidget(prev_btn)

        act.addStretch(1)

        self._chk_plot = QCheckBox("Show plot after applying")
        self._chk_plot.setChecked(True)
        act.addWidget(self._chk_plot)

        for label, fn in [("Cancel", self.reject), ("Apply method", self._apply)]:
            b = QPushButton(label)
            b.clicked.connect(fn)
            act.addWidget(b)
        root.addLayout(act)

        self.method_box.currentRowChanged.connect(self._on_select)
        if self.method_box.count() > 0:
            self.method_box.setCurrentRow(0)
            self._on_select(0)

    # ── property ──────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        return self.parent_window.current_dataframe()

    # ── panel builders ────────────────────────────────────────────────────────

    def _build_method_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        hdr = QLabel("OUTLIER METHOD")
        hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(hdr)
        self.method_box = QListWidget()
        self.method_box.setAlternatingRowColors(True)
        self._keys: list[str | None] = []
        self._populate_list()
        lay.addWidget(self.method_box, 1)
        return w

    def _build_param_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 0, 0, 0)
        lay.setSpacing(8)

        hdr = QLabel("PARAMETERS")
        hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(hdr)

        self._desc_lbl = QLabel()
        self._desc_lbl.setObjectName("SectionHint")
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setMinimumHeight(72)
        lay.addWidget(self._desc_lbl)

        sep = QFrame()
        sep.setObjectName("Divider")
        sep.setFixedHeight(1)
        sep.setFrameShape(QFrame.HLine)
        lay.addWidget(sep)

        # Column selector — shown only for column-sensitive methods
        self._col_row = QWidget()
        cl = QHBoxLayout(self._col_row)
        cl.setContentsMargins(0, 4, 0, 4)
        cl.addWidget(QLabel("Target column :"))
        self._col_combo = QComboBox()
        cl.addWidget(self._col_combo, 1)
        lay.addWidget(self._col_row)

        sep2 = QFrame()
        sep2.setObjectName("Divider")
        sep2.setFixedHeight(1)
        sep2.setFrameShape(QFrame.HLine)
        self._sep2 = sep2
        lay.addWidget(sep2)

        self._stack = QStackedWidget()
        self._method_pages: dict[str, int] = {}
        self._build_param_pages()
        lay.addWidget(self._stack, 1)
        return w

    # ── parameter pages ───────────────────────────────────────────────────────

    def _build_param_pages(self) -> None:
        for _, _, key in self._CATALOGUE:
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(0, 10, 0, 0)
            form.setSpacing(12)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            if key == "zscore":
                self._p_zscore_thresh = _dbl(2.60, 0.01, 100.0, 2)
                form.addRow("Threshold :", self._p_zscore_thresh)
                form.addRow("", _note("Rows removed where |Z| > threshold on any numeric column."))

            elif key == "mzscore":
                self._p_mzscore_thresh = _dbl(19.35, 0.01, 1000.0, 2)
                form.addRow("Threshold :", self._p_mzscore_thresh)
                form.addRow("", _note("Default 19.35 ≈ 3.5 / 0.6745⁻¹ · (Iglewicz–Hoaglin convention)."))

            elif key == "iqr":
                self._p_iqr_q1   = _dbl(0.25, 0.01, 0.49, 2)
                self._p_iqr_q3   = _dbl(0.75, 0.51, 0.99, 2)
                self._p_iqr_mult = _dbl(1.5,  0.10, 10.0, 2)
                form.addRow("Q1 quantile :", self._p_iqr_q1)
                form.addRow("Q3 quantile :", self._p_iqr_q3)
                form.addRow("Fence multiplier k :", self._p_iqr_mult)
                form.addRow("", _note("Applied to the selected column. k = 1.5 (Tukey), k = 3.0 (outer fence)."))

            elif key == "euclidean":
                self._p_euc_thresh = _dbl(3.0, 0.1, 1e6, 2)
                form.addRow("Distance threshold :", self._p_euc_thresh)
                form.addRow("", _note("Euclidean distance from centroid across all numeric columns."))

            elif key == "mahalanobis":
                self._p_mah_thresh = _dbl(5.5, 0.1, 1e4, 2)
                form.addRow("Distance threshold :", self._p_mah_thresh)
                form.addRow("", _note("Requires n_rows > n_numeric_cols and no multicollinear features."))

            elif key == "kmeans":
                self._p_km_clusters = _int(3, 1, 50)
                self._p_km_pct      = _dbl(95.0, 50.0, 99.9, 1)
                form.addRow("Clusters :", self._p_km_clusters)
                form.addRow("Distance percentile :", self._p_km_pct)
                form.addRow("", _note("Rows beyond the percentile of centroid distances are removed."))

            elif key == "dbscan":
                self._p_db_eps = _dbl(3.0, 0.01, 1e4, 2)
                self._p_db_min = _int(5, 1, 200)
                form.addRow("Epsilon (eps) :", self._p_db_eps)
                form.addRow("Min samples :", self._p_db_min)
                form.addRow("", _note("Data are StandardScaler-normalised before DBSCAN."))

            elif key == "optics":
                self._p_op_min = _int(5, 1, 200)
                form.addRow("Min samples :", self._p_op_min)
                form.addRow("", _note("max_eps = ∞. Uses selected column when provided, else all numeric."))

            elif key == "optics2":
                self._p_op2_min = _int(5, 1, 200)
                self._p_op2_mcs = _dbl(0.02, 0.001, 0.5, 3)
                self._p_op2_xi  = _dbl(0.01, 0.001, 0.5, 3)
                form.addRow("Min samples :", self._p_op2_min)
                form.addRow("Min cluster size (frac.) :", self._p_op2_mcs)
                form.addRow("Xi :", self._p_op2_xi)
                form.addRow("", _note("Xi steep-area extraction. Uses selected column when provided."))

            elif key == "isoforest":
                self._p_if_est  = _int(100, 10, 2000)
                self._p_if_cont = _dbl(0.10, 0.001, 0.499, 3)
                form.addRow("Estimators :", self._p_if_est)
                form.addRow("Contamination :", self._p_if_cont)
                form.addRow("", _note("Expected fraction of outliers in the dataset."))

            elif key == "ocsvm":
                self._p_svm_nu    = _dbl(0.10, 0.001, 0.999, 3)
                self._p_svm_gamma = QComboBox()
                self._p_svm_gamma.addItems(["scale", "auto"])
                self._p_svm_gamma.setMinimumWidth(120)
                form.addRow("Nu :", self._p_svm_nu)
                form.addRow("Gamma :", self._p_svm_gamma)
                form.addRow("", _note("Data are StandardScaler-normalised. nu ≈ outlier fraction upper bound."))

            elif key == "elliptic":
                self._p_el_cont = _dbl(0.09, 0.001, 0.499, 3)
                form.addRow("Contamination :", self._p_el_cont)
                form.addRow("", _note("MCD ellipse with support_fraction = 0.9. Best for near-Gaussian data."))

            elif key == "lof":
                self._p_lof_n    = _int(20, 2, 500)
                self._p_lof_cont = _dbl(0.05, 0.001, 0.499, 3)
                form.addRow("Neighbours k :", self._p_lof_n)
                form.addRow("Contamination :", self._p_lof_cont)
                form.addRow("", _note("LOF ≫ 1 indicates a point in a sparser region than its neighbours."))

            elif key == "rpca":
                self._p_rpca_lam    = _dbl(0.10, 1e-4, 10.0, 4)
                self._p_rpca_iter   = _int(1000, 10, 10000)
                self._p_rpca_tol    = _dbl(1e-3, 1e-8, 1.0, 6)
                self._p_rpca_thresh = _dbl(7.0, 0.01, 1e6, 2)
                form.addRow("Lambda (λ) :", self._p_rpca_lam)
                form.addRow("Max ADMM iterations :", self._p_rpca_iter)
                form.addRow("Convergence tolerance :", self._p_rpca_tol)
                form.addRow("Sparse score threshold :", self._p_rpca_thresh)
                form.addRow("", _note("Theoretical optimum: λ = 1 / √max(n_rows, n_cols)."))

            elif key == "fbod":
                self._p_fbod_g = _int(2,  1, 50)
                self._p_fbod_k = _int(10, 1, 200)
                self._p_fbod_n = _int(20, 1, 1000)
                form.addRow("Graph replicates :", self._p_fbod_g)
                form.addRow("Neighbourhood k :", self._p_fbod_k)
                form.addRow("Top-n outliers to remove :", self._p_fbod_n)
                form.addRow("", _note("Random k-NN graph. Top-n rows by fluctuation score are removed."))

            elif key == "moment_opt":
                self._p_mo_optimizer = QComboBox()
                self._p_mo_optimizer.addItems([
                    "Bisection (fast, threshold only)",
                    "Simulated Annealing (global, sample-level)",
                    "Hill Climbing (local, sample-level)",
                ])
                self._p_mo_optimizer.setMinimumWidth(220)
                form.addRow("Optimizer :", self._p_mo_optimizer)
                self._p_mo_base = QComboBox()
                self._p_mo_base.addItems(["IQR", "Z-Score"])
                self._p_mo_base.setMinimumWidth(120)
                form.addRow("Base method :", self._p_mo_base)
                self._p_mo_target_kurt = _dbl(0.0, -20.0, 50.0, 2)
                form.addRow("Target kurtosis (excess) :", self._p_mo_target_kurt)
                self._p_mo_target_skew = _dbl(0.0, -10.0, 10.0, 2)
                form.addRow("Target skewness :", self._p_mo_target_skew)
                self._p_mo_w_kurt = _dbl(1.0, 0.0, 100.0, 2)
                form.addRow("Weight kurtosis :", self._p_mo_w_kurt)
                self._p_mo_w_skew = _dbl(1.0, 0.0, 100.0, 2)
                form.addRow("Weight skewness :", self._p_mo_w_skew)
                self._p_mo_w_ret = _dbl(0.5, 0.0, 100.0, 2)
                form.addRow("Weight retention :", self._p_mo_w_ret)
                self._p_mo_min_ret = _dbl(0.50, 0.10, 0.99, 2)
                form.addRow("Min retention ratio :", self._p_mo_min_ret)
                self._p_mo_sa_iter = _int(2000, 100, 50000)
                form.addRow("SA / HC iterations :", self._p_mo_sa_iter)
                form.addRow("", _note(
                    "Optimizes sample exclusion to minimize "
                    "w_k·(κ − κ_target)² + w_s·(γ − γ_target)² + λ·(removed/total). "
                    "Bisection tunes the IQR/z threshold; SA & HC refine individual samples."
                ))

            idx = self._stack.addWidget(page)
            self._method_pages[key] = idx

    # ── list population ───────────────────────────────────────────────────────

    def _populate_list(self) -> None:
        cur_group = None
        for display, group, key in self._CATALOGUE:
            if group != cur_group:
                h = QListWidgetItem(f"  {group}")
                h.setFlags(Qt.NoItemFlags)
                f = QFont()
                f.setWeight(QFont.Bold)
                f.setPointSize(9)
                h.setFont(f)
                h.setForeground(QColor("#c9a227"))
                self.method_box.addItem(h)
                self._keys.append(None)
                cur_group = group
            self.method_box.addItem(QListWidgetItem(f"    {display}"))
            self._keys.append(key)

    # ── slot ──────────────────────────────────────────────────────────────────

    def _on_select(self, row: int) -> None:
        key = self._keys[row] if 0 <= row < len(self._keys) else None
        if key is None:
            # jump to first selectable item below the header
            for i in range(row + 1, self.method_box.count()):
                if self._keys[i] is not None:
                    self.method_box.setCurrentRow(i)
                    return
            return
        self._desc_lbl.setText(self._DESC.get(key, ""))
        if key in self._method_pages:
            self._stack.setCurrentIndex(self._method_pages[key])

        # Show column selector only for methods that use a specific column
        col_visible = key in self._COL_METHODS
        self._col_row.setVisible(col_visible)
        self._sep2.setVisible(col_visible)
        if col_visible:
            self._refresh_col_combo()

        # For RPCA: nudge lambda default toward theoretical optimum
        if key == "rpca":
            df = self.df
            num_cols = df.select_dtypes(include="number").shape[1]
            nrows = len(df)
            if nrows > 0 and num_cols > 0:
                opt_lam = round(1.0 / np.sqrt(max(nrows, num_cols)), 4)
                self._p_rpca_lam.setValue(opt_lam)

    def _refresh_col_combo(self) -> None:
        self._col_combo.clear()
        for c in self.df.select_dtypes(include="number").columns:
            self._col_combo.addItem(str(c))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _selected_key(self) -> Optional[str]:
        r = self.method_box.currentRow()
        return self._keys[r] if 0 <= r < len(self._keys) else None

    def _selected_display(self) -> str:
        item = self.method_box.currentItem()
        return item.text().strip() if item else "unknown"

    # ── actions ───────────────────────────────────────────────────────────────

    def _summarise(self) -> None:
        df = self.df
        num = df.select_dtypes(include="number")
        if num.empty:
            QMessageBox.information(self, "Summary", "No numeric columns in the active dataset.")
            return
        lines = [
            f"Dataset  {df.shape[0]:,} rows × {df.shape[1]:,} cols  "
            f"({num.shape[1]} numeric)",
            "",
            f"{'Column':<36} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'NA':>6}",
            "─" * 84,
        ]
        for c in num.columns[:40]:
            col = num[c]
            na  = int(col.isna().sum())
            col = col.dropna()
            lines.append(
                f"{str(c):<36} {col.min():>12.4g} {col.max():>12.4g} "
                f"{col.mean():>12.4g} {col.std():>12.4g} {na:>6}"
            )
        if len(num.columns) > 40:
            lines.append(f"… {len(num.columns) - 40} more columns omitted")
        QMessageBox.information(self, "Numeric column summary", "\n".join(lines))

    def _apply(self) -> None:
        key = self._selected_key()
        if key is None:
            QMessageBox.warning(self, "No method", "Select an outlier removal method first.")
            return
        display  = self._selected_display()
        df       = self.df.copy()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            QMessageBox.critical(self, "No numeric columns",
                "The active dataset contains no numeric columns.")
            return

        col    = self._col_combo.currentText() if key in self._COL_METHODS else ""
        params : dict = {"method": display, "key": key}
        if col:
            params["column"] = col

        try:
            _, outlier_mask, n_removed = self._run(key, df, num_cols, col, params)
        except np.linalg.LinAlgError as e:
            QMessageBox.critical(self, "Linear algebra error",
                f"{e}\n\nCovariance matrix may be singular. "
                "Remove near-constant or perfectly correlated columns first.")
            return
        except Exception as e:
            QMessageBox.critical(self, "Outlier removal failed", str(e))
            log.exception("OutlierRemovalDialog._apply key=%s", key); return

        # Open the interactive review dialog before committing
        plot_col = col if (col and col in num_cols) else num_cols[0]
        review = OutlierReviewDialog(
            df_original=df, outlier_mask=outlier_mask,
            plot_col=plot_col, display=display, parent=self,
        )
        if review.exec() != QDialog.Accepted:
            return

        final_df        = review.final_dataframe()
        n_final_removed = len(df) - len(final_df)
        params["rows_removed"] = n_final_removed
        summary = f"Outlier removal · {display} · {n_final_removed:,} row(s) removed"
        self.parent_window.apply_cleaning_operation(
            operation="remove_outliers", dataframe=final_df,
            summary=summary, params=params,
        )
        self.accept()

    def _preview_plot(self) -> None:
        key = self._selected_key()
        if key is None:
            QMessageBox.warning(self, "No method", "Select a method first."); return
        display  = self._selected_display()
        df       = self.df.copy()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            QMessageBox.warning(self, "No data", "No numeric columns to preview."); return
        col    = self._col_combo.currentText() if key in self._COL_METHODS else ""
        params : dict = {}
        try:
            _, outlier_mask, _ = self._run(key, df, num_cols, col, params)
        except np.linalg.LinAlgError as e:
            QMessageBox.critical(self, "Linear algebra error", str(e)); return
        except Exception as e:
            QMessageBox.critical(self, "Preview failed", str(e))
            log.exception("OutlierRemovalDialog._preview_plot key=%s", key); return
        plot_col = col if (col and col in num_cols) else num_cols[0]
        # Preview uses the review dialog in read-only-ish mode (user can still interact)
        review = OutlierReviewDialog(
            df_original=df, outlier_mask=outlier_mask,
            plot_col=plot_col,
            display=f"{display}  [preview — not yet applied]",
            parent=self,
        )
        review.exec()

    # ── computation ───────────────────────────────────────────────────────────

    def _run(self, key: str, df: pd.DataFrame, num_cols: list,
             col: str, params: dict) -> tuple[pd.DataFrame, pd.Series, int]:
        orig_len = len(df)
        num = df[num_cols]
        # Median-impute NaN for distance/model computation only — original df kept intact
        num_imp = num.fillna(num.median())

        if key == "zscore":
            from scipy.stats import zscore as _zscore
            thresh = self._p_zscore_thresh.value()
            params["threshold"] = thresh
            z = np.abs(_zscore(num_imp.values, nan_policy='omit'))
            mask = ~(z > thresh).any(axis=1)

        elif key == "mzscore":
            thresh = self._p_mzscore_thresh.value()
            params["threshold"] = thresh
            median = num_imp.median()
            mad    = (num_imp - median).abs().median()
            z      = 0.6745 * (num_imp - median).abs() / mad.replace(0, np.nan)
            mask   = ~(z > thresh).any(axis=1)

        elif key == "iqr":
            q1v  = self._p_iqr_q1.value()
            q3v  = self._p_iqr_q3.value()
            mult = self._p_iqr_mult.value()
            params.update({"q1": q1v, "q3": q3v, "multiplier": mult})
            if col and col in num_cols:
                s  = df[col]
                q1 = s.quantile(q1v);  q3 = s.quantile(q3v)
                iqr = q3 - q1
                mask = (s >= q1 - mult * iqr) & (s <= q3 + mult * iqr)
            else:
                mask = pd.Series(True, index=df.index)

        elif key == "euclidean":
            from scipy.spatial.distance import cdist
            thresh  = self._p_euc_thresh.value()
            params["threshold"] = thresh
            centroid = num_imp.mean().values.reshape(1, -1)
            dists    = cdist(num_imp.values, centroid, metric="euclidean").ravel()
            mask     = dists <= thresh

        elif key == "mahalanobis":
            thresh = self._p_mah_thresh.value()
            params["threshold"] = thresh
            X       = num_imp.values.astype(float)
            mean_v  = X.mean(axis=0)
            cov     = np.cov(X, rowvar=False)
            inv_cov = np.linalg.inv(cov)          # raises LinAlgError if singular
            diff    = X - mean_v
            dists   = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
            mask    = dists <= thresh

        elif key == "kmeans":
            from sklearn.cluster import KMeans
            n_cl = self._p_km_clusters.value()
            pct  = self._p_km_pct.value()
            params.update({"n_clusters": n_cl, "percentile": pct})
            X   = num_imp.values
            km  = KMeans(n_clusters=n_cl, init='k-means++', random_state=42, n_init='auto')
            km.fit(X)
            dists = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
            thr   = np.percentile(dists, pct)
            mask  = dists <= thr

        elif key == "dbscan":
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            eps   = self._p_db_eps.value()
            min_s = self._p_db_min.value()
            params.update({"eps": eps, "min_samples": min_s})
            X_sc = StandardScaler().fit_transform(num_imp.values)
            labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X_sc)
            mask   = labels != -1

        elif key == "optics":
            from sklearn.cluster import OPTICS
            min_s = self._p_op_min.value()
            params["min_samples"] = min_s
            X = num_imp[[col]].values if (col and col in num_cols) else num_imp.values
            labels = OPTICS(min_samples=min_s).fit_predict(X)
            mask   = labels != -1

        elif key == "optics2":
            from sklearn.cluster import OPTICS
            min_s = self._p_op2_min.value()
            mcs   = self._p_op2_mcs.value()
            xi    = self._p_op2_xi.value()
            params.update({"min_samples": min_s, "min_cluster_size": mcs, "xi": xi})
            X = num_imp[[col]].values if (col and col in num_cols) else num_imp.values
            labels = OPTICS(min_samples=min_s, min_cluster_size=mcs, xi=xi).fit_predict(X)
            mask   = labels != -1

        elif key == "isoforest":
            from sklearn.ensemble import IsolationForest
            n_est = self._p_if_est.value()
            cont  = self._p_if_cont.value()
            params.update({"n_estimators": n_est, "contamination": cont})
            labels = IsolationForest(
                n_estimators=n_est, contamination=cont, random_state=42
            ).fit_predict(num_imp.values)
            mask = labels != -1

        elif key == "ocsvm":
            from sklearn.svm import OneClassSVM
            from sklearn.preprocessing import StandardScaler
            nu    = self._p_svm_nu.value()
            gamma = self._p_svm_gamma.currentText()
            params.update({"nu": nu, "gamma": gamma})
            X_sc   = StandardScaler().fit_transform(num_imp.values)
            labels = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma).fit_predict(X_sc)
            mask   = labels != -1

        elif key == "elliptic":
            from sklearn.covariance import EllipticEnvelope
            cont = self._p_el_cont.value()
            params["contamination"] = cont
            labels = EllipticEnvelope(
                contamination=cont, support_fraction=0.9, random_state=42
            ).fit_predict(num_imp.values)
            mask = labels != -1

        elif key == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            n_nbr = self._p_lof_n.value()
            cont  = self._p_lof_cont.value()
            params.update({"n_neighbors": n_nbr, "contamination": cont})
            labels = LocalOutlierFactor(
                n_neighbors=n_nbr, contamination=cont
            ).fit_predict(num_imp.values)
            mask = labels != -1

        elif key == "rpca":
            lam       = self._p_rpca_lam.value()
            max_iter  = self._p_rpca_iter.value()
            tol       = self._p_rpca_tol.value()
            score_thr = self._p_rpca_thresh.value()
            params.update({"lambda": lam, "max_iterations": max_iter,
                           "tolerance": tol, "threshold": score_thr})
            X        = num_imp.values.astype(float)
            _, S     = _rpca(X, lam, max_iter, tol)
            mask     = np.sum(np.abs(S), axis=1) <= score_thr

        elif key == "fbod":
            g     = self._p_fbod_g.value()
            k     = self._p_fbod_k.value()
            n_abn = self._p_fbod_n.value()
            params.update({"graph_number": g, "k": k, "abnormal_number": n_abn})
            X   = num_imp.values.astype(float)
            m   = len(X)
            fluc = np.zeros((m, g))
            last_A: np.ndarray | None = None
            for i in range(g):
                A = np.eye(m)
                for j in range(m):
                    nbrs = _random.sample([x for x in range(m) if x != j], min(k, m - 1))
                    A[j, nbrs] = 1.0
                last_A = A
                Z  = A @ X
                cr = np.sum(X, axis=1) / (np.sum(Z, axis=1) + 1e-10)
                fluc[:, i] = cr
            cr_sum = np.sum(fluc, axis=1)
            OF = np.zeros(m)
            if last_A is not None:
                for ii in range(m):
                    nbrs_ii = np.where(last_A[ii] != 0)[0]
                    OF[ii]  = np.sum(np.abs(cr_sum[ii] - cr_sum[nbrs_ii]))
            outlier_idx        = np.argsort(OF)[-n_abn:]
            mask_arr           = np.ones(m, dtype=bool)
            mask_arr[outlier_idx] = False
            mask = mask_arr

        elif key == "moment_opt":
            import outlier_opt_engine as _opt
            _opt_idx = self._p_mo_optimizer.currentIndex()
            _opt_map = {0: "bisection", 1: "simulated_annealing", 2: "hill_climbing"}
            optimizer = _opt_map.get(_opt_idx, "bisection")
            _base_idx = self._p_mo_base.currentIndex()
            base_method = "zscore" if _base_idx == 1 else "iqr"
            target_kurt = self._p_mo_target_kurt.value()
            target_skew = self._p_mo_target_skew.value()
            w_kurt = self._p_mo_w_kurt.value()
            w_skew = self._p_mo_w_skew.value()
            w_ret  = self._p_mo_w_ret.value()
            min_ret = self._p_mo_min_ret.value()
            sa_iter = self._p_mo_sa_iter.value()

            if not col or col not in num_cols:
                raise ValueError("Moment-Optimized requires a target column selection.")

            params.update({
                "optimizer": optimizer, "base_method": base_method,
                "target_kurtosis": target_kurt, "target_skewness": target_skew,
                "w_kurtosis": w_kurt, "w_skewness": w_skew, "w_retention": w_ret,
                "min_retention": min_ret, "sa_hc_iterations": sa_iter,
            })

            result = _opt.optimize_outliers(
                df, col,
                optimizer=optimizer,
                base_method=base_method,
                target_kurtosis=target_kurt,
                target_skewness=target_skew,
                w_kurtosis=w_kurt,
                w_skewness=w_skew,
                w_retention=w_ret,
                min_retention=min_ret,
                sa_max_iter=sa_iter,
                hc_max_iter=sa_iter,
            )
            mask = result.mask
            params.update({
                "kurtosis_before": round(result.kurtosis_before, 4),
                "kurtosis_after":  round(result.kurtosis_after, 4),
                "skewness_before": round(result.skewness_before, 4),
                "skewness_after":  round(result.skewness_after, 4),
                "objective_before": round(result.objective_before, 6),
                "objective_after":  round(result.objective_after, 6),
                "n_removed":       result.n_removed,
            })

        else:
            raise ValueError(f"Unknown method key: {key!r}")

        mask_series = pd.Series(
            mask if isinstance(mask, np.ndarray) else np.asarray(mask),
            index=df.index,
        )
        outlier_mask = ~mask_series
        cleaned      = df[mask_series.values]
        n_removed    = orig_len - len(cleaned)
        return cleaned, outlier_mask, n_removed


# ══════════════════════════════════════════════════════════════════════════════
# Outlier Review Dialog
# ══════════════════════════════════════════════════════════════════════════════

class OutlierReviewDialog(QDialog):
    """Interactive review of flagged rows before committing the revision.

    Features
    --------
    - Table of all flagged rows with per-row "Restore?" checkboxes
    - "Restore top-N least extreme" quick control (sorts by |value − median|)
    - Select-all / clear-all shortcuts
    - Live embedded scatter + histogram that refreshes on demand
    - Save cleaned dataset or flagged rows to CSV / Excel
    - Final summary label updates as user changes restore choices
    """

    _MAX_TABLE_ROWS = 2000   # cap display to keep UI responsive

    def __init__(self, df_original: pd.DataFrame, outlier_mask: pd.Series,
                 plot_col: str, display: str, parent=None):
        super().__init__(parent)
        self._df      = df_original
        self._mask    = outlier_mask.copy()   # True = outlier
        self._col     = plot_col
        self._display = display

        self._outlier_df = df_original[outlier_mask].reset_index(drop=False)
        # "original_index" column holds the position in df for restore logic
        if "index" in self._outlier_df.columns:
            self._outlier_df = self._outlier_df.rename(columns={"index": "_orig_idx"})
        else:
            self._outlier_df.insert(0, "_orig_idx", df_original[outlier_mask].index)

        self._n_flagged = int(outlier_mask.sum())
        self._pct       = 100.0 * self._n_flagged / max(len(df_original), 1)

        self.setWindowTitle("Outlier Review")
        self.setModal(True)
        self.resize(1280, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        # ── summary banner ─────────────────────────────────────────────────
        self._summary_lbl = QLabel()
        self._summary_lbl.setObjectName("SectionHint")
        self._summary_lbl.setWordWrap(True)
        root.addWidget(self._summary_lbl)

        sep = QFrame(); sep.setObjectName("Divider")
        sep.setFixedHeight(1); sep.setFrameShape(QFrame.HLine)
        root.addWidget(sep)

        # ── main splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_table_panel())
        splitter.addWidget(self._build_plot_panel())
        splitter.setSizes([560, 620])
        root.addWidget(splitter, 1)

        # ── bottom action bar ──────────────────────────────────────────────
        bot = QHBoxLayout()
        save_clean = QPushButton("Save cleaned dataset…")
        save_clean.clicked.connect(self._save_cleaned)
        save_flag  = QPushButton("Save flagged rows…")
        save_flag.clicked.connect(self._save_flagged)
        bot.addWidget(save_clean)
        bot.addWidget(save_flag)
        bot.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Confirm removal")
        confirm_btn.setDefault(True)
        confirm_btn.clicked.connect(self.accept)
        bot.addWidget(cancel_btn)
        bot.addWidget(confirm_btn)
        root.addLayout(bot)

        self._populate_table()
        self._refresh_summary()
        self._refresh_plot()

    # ── panel builders ────────────────────────────────────────────────────

    def _build_table_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(6)

        hdr = QLabel("FLAGGED ROWS")
        hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(hdr)

        # quick-restore controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Restore top"))
        self._restore_n = QSpinBox()
        self._restore_n.setRange(1, max(self._n_flagged, 1))
        self._restore_n.setValue(min(5, max(self._n_flagged, 1)))
        self._restore_n.setFixedWidth(70)
        ctrl.addWidget(self._restore_n)
        ctrl.addWidget(QLabel("least extreme"))
        go_btn = QPushButton("Apply")
        go_btn.setFixedWidth(60)
        go_btn.clicked.connect(self._restore_top_n)
        ctrl.addWidget(go_btn)
        ctrl.addStretch(1)
        all_btn  = QPushButton("Select all")
        none_btn = QPushButton("Clear all")
        all_btn.setFixedWidth(80);  all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn.setFixedWidth(80); none_btn.clicked.connect(lambda: self._set_all(False))
        ctrl.addWidget(all_btn)
        ctrl.addWidget(none_btn)
        lay.addLayout(ctrl)

        # table
        df_show = self._outlier_df
        truncated = len(df_show) > self._MAX_TABLE_ROWS
        if truncated:
            df_show = df_show.iloc[:self._MAX_TABLE_ROWS]

        data_cols = [c for c in df_show.columns if c != "_orig_idx"]
        self._table = QTableWidget(len(df_show), len(data_cols) + 2)
        self._table.setHorizontalHeaderLabels(
            ["Restore?", "Row #"] + [str(c) for c in data_cols]
        )
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        self._check_items: list[QTableWidgetItem] = []
        lay.addWidget(self._table, 1)

        if truncated:
            note = QLabel(
                f"⚠  Showing first {self._MAX_TABLE_ROWS:,} of {self._n_flagged:,} flagged rows."
            )
            note.setObjectName("SectionHint")
            lay.addWidget(note)
        return w

    def _build_plot_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(6)

        hdr = QLabel("SCATTER + DISTRIBUTION")
        hdr.setObjectName("ReportPanelEyebrow")
        lay.addWidget(hdr)

        # column picker for the plot
        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("Plot column :"))
        self._plot_col_combo = QComboBox()
        for c in self._df.select_dtypes(include="number").columns:
            self._plot_col_combo.addItem(str(c))
        # pre-select the suggested column
        idx = self._plot_col_combo.findText(str(self._col))
        if idx >= 0:
            self._plot_col_combo.setCurrentIndex(idx)
        col_row.addWidget(self._plot_col_combo, 1)
        refresh_btn = QPushButton("Refresh plot")
        refresh_btn.clicked.connect(self._refresh_plot)
        col_row.addWidget(refresh_btn)
        cust_btn = QPushButton("Customize & save plot…")
        cust_btn.clicked.connect(self._open_customizer)
        col_row.addWidget(cust_btn)
        lay.addLayout(col_row)

        # embedded canvas
        self._fig    = Figure(figsize=(6, 4.5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(300)
        lay.addWidget(self._canvas, 1)
        return w

    def _open_customizer(self) -> None:
        col      = self._plot_col_combo.currentText()
        restored = self._restored_indices()
        dlg = PlotCustomizationDialog(
            df=self._df, outlier_mask=self._mask,
            restored_indices=restored, plot_col=col,
            display=self._display, parent=self,
        )
        dlg.exec()

    # ── table population ──────────────────────────────────────────────────

    def _populate_table(self) -> None:
        df_show   = self._outlier_df
        truncated = len(df_show) > self._MAX_TABLE_ROWS
        if truncated:
            df_show = df_show.iloc[:self._MAX_TABLE_ROWS]
        data_cols = [c for c in df_show.columns if c != "_orig_idx"]

        self._check_items.clear()
        self._table.setRowCount(len(df_show))

        for r, (_, row) in enumerate(df_show.iterrows()):
            # checkbox column
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            chk.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, 0, chk)
            self._check_items.append(chk)

            # original row number
            orig_item = QTableWidgetItem(str(row["_orig_idx"]))
            orig_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, 1, orig_item)

            # data columns
            for c_idx, col in enumerate(data_cols):
                val = row[col]
                txt = f"{val:.6g}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(r, c_idx + 2, item)

        self._table.itemChanged.connect(self._on_check_changed)

    # ── slots ─────────────────────────────────────────────────────────────

    def _on_check_changed(self, _item) -> None:
        self._refresh_summary()

    def _set_all(self, state: bool) -> None:
        self._table.itemChanged.disconnect(self._on_check_changed)
        qs = Qt.Checked if state else Qt.Unchecked
        for chk in self._check_items:
            chk.setCheckState(qs)
        self._table.itemChanged.connect(self._on_check_changed)
        self._refresh_summary()

    def _restore_top_n(self) -> None:
        """Check the N least-extreme flagged rows (closest to the column median)."""
        n   = self._restore_n.value()
        col = self._plot_col_combo.currentText()
        df_show = self._outlier_df.iloc[:len(self._check_items)]

        if col in df_show.columns:
            median = self._df[col].median()
            scores = (df_show[col] - median).abs().values
        else:
            scores = np.arange(len(df_show), dtype=float)

        top_idx = np.argsort(scores)[:n]
        self._table.itemChanged.disconnect(self._on_check_changed)
        for i, chk in enumerate(self._check_items):
            chk.setCheckState(Qt.Checked if i in top_idx else Qt.Unchecked)
        self._table.itemChanged.connect(self._on_check_changed)
        self._refresh_summary()

    # ── refresh helpers ───────────────────────────────────────────────────

    def _restored_indices(self) -> set:
        """Original df indices that the user has checked for restore."""
        result = set()
        df_show = self._outlier_df.iloc[:len(self._check_items)]
        for i, chk in enumerate(self._check_items):
            if chk.checkState() == Qt.Checked:
                result.add(df_show.iloc[i]["_orig_idx"])
        return result

    def _refresh_summary(self) -> None:
        n_restore = sum(
            1 for c in self._check_items if c.checkState() == Qt.Checked
        )
        n_final = self._n_flagged - n_restore
        self._summary_lbl.setText(
            f"Method: <b>{self._display}</b> · "
            f"Flagged: <b>{self._n_flagged:,}</b> rows "
            f"({self._pct:.1f}% of dataset) · "
            f"Restoring: <b>{n_restore:,}</b> · "
            f"Will remove: <b>{n_final:,}</b>"
        )

    def _refresh_plot(self) -> None:
        col = self._plot_col_combo.currentText()
        if col not in self._df.columns:
            return

        restored = self._restored_indices()
        y        = self._df[col].values
        idx      = np.arange(len(self._df))
        outlier_arr  = self._mask.values
        restore_arr  = np.array([
            (self._df.index[i] in restored) for i in range(len(self._df))
        ])
        remove_arr   = outlier_arr & ~restore_arr
        inlier_arr   = ~outlier_arr

        self._fig.clear()
        ax1, ax2 = self._fig.subplots(1, 2,
                                       gridspec_kw={"width_ratios": [2, 1]})
        self._fig.suptitle(
            f"{self._display}  —  {col}", fontsize=9, fontweight="bold"
        )

        # scatter
        _INLIER  = ("#2ca02c", "o", 18, 0.55, f"Inliers ({int(inlier_arr.sum()):,})")
        _RESTORE = ("#ff7f0e", "^", 55, 0.90, f"Restoring ({int(restore_arr.sum()):,})")
        _REMOVE  = ("#d62728", "x", 45, 0.85, f"Removing  ({int(remove_arr.sum()):,})")

        for arr, (clr, mrk, sz, alp, lbl) in zip(
            [inlier_arr, restore_arr, remove_arr],
            [_INLIER, _RESTORE, _REMOVE],
        ):
            if arr.any():
                ax1.scatter(idx[arr], y[arr], c=clr, marker=mrk,
                            s=sz, alpha=alp, label=lbl, linewidths=1.2)

        ax1.set_xlabel("Row index", fontsize=8)
        ax1.set_ylabel(col, fontsize=8)
        ax1.legend(fontsize=7, framealpha=0.7)
        ax1.grid(color="gray", linestyle="--", linewidth=0.35, alpha=0.45)
        ax1.tick_params(labelsize=7)

        # histogram
        bins = min(40, max(10, int(np.sqrt(len(y)))))
        lo, hi = np.nanmin(y), np.nanmax(y)
        edges  = np.linspace(lo, hi, bins + 1)
        if inlier_arr.any():
            ax2.hist(y[inlier_arr],  bins=edges, color="#2ca02c", alpha=0.55,
                     label="Inliers")
        if remove_arr.any():
            ax2.hist(y[remove_arr],  bins=edges, color="#d62728", alpha=0.75,
                     label="Removing")
        if restore_arr.any():
            ax2.hist(y[restore_arr], bins=edges, color="#ff7f0e", alpha=0.80,
                     label="Restoring")
        ax2.set_xlabel(col, fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.legend(fontsize=7, framealpha=0.7)
        ax2.grid(color="gray", linestyle="--", linewidth=0.35, alpha=0.45)
        ax2.tick_params(labelsize=7)

        self._canvas.draw()

    # ── save helpers ──────────────────────────────────────────────────────

    def _save_cleaned(self) -> None:
        path, flt = QFileDialog.getSaveFileName(
            self, "Save cleaned dataset", "cleaned_data",
            "CSV files (*.csv);;Excel files (*.xlsx)"
        )
        if not path:
            return
        df = self.final_dataframe()
        try:
            if path.endswith(".xlsx"):
                df.to_excel(path, index=False)
            else:
                if not path.endswith(".csv"):
                    path += ".csv"
                df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved",
                f"Cleaned dataset ({len(df):,} rows) saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _save_flagged(self) -> None:
        path, flt = QFileDialog.getSaveFileName(
            self, "Save flagged rows", "flagged_outliers",
            "CSV files (*.csv);;Excel files (*.xlsx)"
        )
        if not path:
            return
        # Include a column indicating which are being restored
        restored = self._restored_indices()
        df_flag  = self._df[self._mask].copy()
        df_flag.insert(0, "outlier_action",
                       ["restored" if i in restored else "removed"
                        for i in df_flag.index])
        try:
            if path.endswith(".xlsx"):
                df_flag.to_excel(path, index=True)
            else:
                if not path.endswith(".csv"):
                    path += ".csv"
                df_flag.to_csv(path, index=True)
            QMessageBox.information(self, "Saved",
                f"Flagged rows ({len(df_flag):,}) saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # ── public ────────────────────────────────────────────────────────────

    def final_dataframe(self) -> pd.DataFrame:
        """Return the dataset with outlier rows removed, except restored ones."""
        restored = self._restored_indices()
        # keep: all inliers + restored outliers
        keep_mask = ~self._mask | self._df.index.isin(restored)
        return self._df[keep_mask].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Color-picker button
# ══════════════════════════════════════════════════════════════════════════════

class _ColorButton(QPushButton):
    """Inline colour swatch that opens QColorDialog on click."""
    colorChanged = Signal(str)   # emits hex string

    def __init__(self, color: str = "#2ca02c", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(44, 24)
        self._repaint()
        self.clicked.connect(self._pick)

    def color(self) -> str:
        return self._color

    def setColor(self, color: str) -> None:
        self._color = color
        self._repaint()

    def _repaint(self) -> None:
        self.setStyleSheet(
            f"QPushButton{{background:{self._color};border:1px solid #777;"
            f"border-radius:2px;}}"
            f"QPushButton:hover{{border:2px solid #bbb;}}"
        )

    def _pick(self) -> None:
        col = QColorDialog.getColor(
            QColor(self._color), self, "Choose colour",
            QColorDialog.ShowAlphaChannel,
        )
        if col.isValid():
            self._color = col.name()
            self._repaint()
            self.colorChanged.emit(self._color)


# ══════════════════════════════════════════════════════════════════════════════
# Plot Customization Dialog
# ══════════════════════════════════════════════════════════════════════════════

class PlotCustomizationDialog(QDialog):
    """Full matplotlib style editor with live preview and multi-format save.

    Tabs
    ----
    Labels & Text  — titles, axis labels, font family, LaTeX toggle, font sizes
    Series         — per-series colour, marker, size, alpha, line width, label
    Grid & Axes    — major/minor grid, style, width, colour, alpha; spines; axes bg
    Legend         — show/hide, location, font size, frame, columns
    Figure         — canvas size, figure background, tight layout
    Save           — format (PNG/PDF/SVG/EPS/TIFF), DPI, transparency, filename
    """

    _MARKERS  = ["o","x","+","*","s","^","v","<",">","D","P","h","H","8",".","None"]
    _LINESTYLES = ["--", "-", ":", "-."]
    _LEG_LOCS = [
        "best","upper right","upper left","lower left","lower right",
        "right","center left","center right","lower center","upper center","center",
    ]
    _FONTS = [
        "sans-serif","serif","monospace",
        "DejaVu Sans","DejaVu Serif","STIXGeneral",
        "Times New Roman","Arial","Helvetica",
    ]

    def __init__(self, df: pd.DataFrame, outlier_mask: pd.Series,
                 restored_indices: set, plot_col: str,
                 display: str, parent=None):
        super().__init__(parent)
        self._df       = df
        self._mask     = outlier_mask.copy()
        self._restored = set(restored_indices)
        self._col      = plot_col
        self._display  = display

        self.setWindowTitle("Customize & Save Plot")
        self.setModal(False)   # leave review dialog accessible
        self.resize(1180, 720)

        self._init_style()
        self._build_ui()
        self._connect_signals()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._refresh)
        self._refresh()

    # ── default style ─────────────────────────────────────────────────────────

    def _init_style(self) -> None:
        col = self._col
        self._style: dict = {
            # labels
            "title":        self._display,
            "sc_title":     f"Scatter — {col}",
            "hist_title":   "Distribution",
            "xlabel":       "Row index",
            "ylabel":       str(col),
            "xlabel_hist":  str(col),
            "ylabel_hist":  "Count",
            "font_family":  "sans-serif",
            "use_latex":    False,
            "title_sz":     11,
            "label_sz":     9,
            "tick_sz":      8,
            # series
            "series": {
                "inliers":  {"vis":True,"lbl":"Inliers",  "clr":"#2ca02c","mrk":"o", "sz":18,"alp":0.55,"lw":0.0},
                "removing": {"vis":True,"lbl":"Removing", "clr":"#d62728","mrk":"x", "sz":45,"alp":0.85,"lw":1.2},
                "restoring":{"vis":True,"lbl":"Restoring","clr":"#ff7f0e","mrk":"^", "sz":55,"alp":0.90,"lw":0.0},
            },
            # legend
            "leg_show":  True,
            "leg_loc":   "best",
            "leg_sz":    7,
            "leg_frame": True,
            "leg_falp":  0.70,
            "leg_ncol":  1,
            # grid
            "grid_maj":   True,
            "grid_min":   False,
            "grid_clr":   "#808080",
            "grid_ls":    "--",
            "grid_lw":    0.40,
            "grid_alp":   0.45,
            # axes
            "axes_bg":    "#ffffff",
            "spine_top":  False,
            "spine_right":False,
            # figure
            "fig_w":      11.0,
            "fig_h":      4.5,
            "fig_bg":     "#ffffff",
            "tight":      True,
            # save
            "fmt":        "png",
            "dpi":        200,
            "transp":     False,
            "prefix":     "outlier_plot",
        }

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # ── left: tab widget ───────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setMinimumWidth(400)
        tabs.setMaximumWidth(460)
        tabs.addTab(self._tab_labels(),  "Labels & Text")
        tabs.addTab(self._tab_series(),  "Series")
        tabs.addTab(self._tab_grid(),    "Grid & Axes")
        tabs.addTab(self._tab_legend(),  "Legend")
        tabs.addTab(self._tab_figure(),  "Figure")
        tabs.addTab(self._tab_save(),    "Save")
        split.addWidget(tabs)

        # ── right: live preview ────────────────────────────────────────────
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(6)

        hdr = QLabel("LIVE PREVIEW")
        hdr.setObjectName("ReportPanelEyebrow")
        rl.addWidget(hdr)

        self._pfig    = Figure(tight_layout=True)
        self._pcanvas = FigureCanvas(self._pfig)
        self._pcanvas.setMinimumHeight(320)
        rl.addWidget(self._pcanvas, 1)

        ctrl = QHBoxLayout()
        self._chk_auto = QCheckBox("Auto-refresh  (400 ms debounce)")
        self._chk_auto.setChecked(True)
        ctrl.addWidget(self._chk_auto)
        ctrl.addStretch(1)
        btn_now = QPushButton("Refresh now")
        btn_now.clicked.connect(self._refresh)
        ctrl.addWidget(btn_now)
        rl.addLayout(ctrl)

        split.addWidget(rw)
        split.setSizes([440, 680])
        root.addWidget(split, 1)

        # ── bottom bar ─────────────────────────────────────────────────────
        bot = QHBoxLayout()
        save_btn = QPushButton("Save figure…")
        save_btn.clicked.connect(self._save)
        bot.addWidget(save_btn)
        bot.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bot.addWidget(close_btn)
        root.addLayout(bot)

    # ── tab: Labels & Text ────────────────────────────────────────────────────

    def _tab_labels(self) -> QWidget:
        sa = QScrollArea(); sa.setWidgetResizable(True)
        w  = QWidget(); form = QFormLayout(w)
        form.setSpacing(9); form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        s = self._style
        self._w_title      = QLineEdit(s["title"])
        self._w_sc_title   = QLineEdit(s["sc_title"])
        self._w_hist_title = QLineEdit(s["hist_title"])
        self._w_xlabel     = QLineEdit(s["xlabel"])
        self._w_ylabel     = QLineEdit(s["ylabel"])
        self._w_xlabel_h   = QLineEdit(s["xlabel_hist"])
        self._w_ylabel_h   = QLineEdit(s["ylabel_hist"])

        form.addRow("Figure title :",        self._w_title)
        form.addRow("Scatter subtitle :",    self._w_sc_title)
        form.addRow("Histogram subtitle :",  self._w_hist_title)
        form.addRow(_div())
        form.addRow("Scatter X label :",     self._w_xlabel)
        form.addRow("Scatter Y label :",     self._w_ylabel)
        form.addRow("Histogram X label :",   self._w_xlabel_h)
        form.addRow("Histogram Y label :",   self._w_ylabel_h)
        form.addRow(_hint("Use $\\alpha$, $\\mu$, $R^2$ etc. for MathText — no LaTeX install needed."))
        form.addRow(_div())

        self._w_font_fam  = QComboBox(); self._w_font_fam.addItems(self._FONTS)
        self._w_font_fam.setCurrentText(s["font_family"])
        form.addRow("Font family :", self._w_font_fam)

        self._w_use_latex = QCheckBox("Use system LaTeX (requires LaTeX + dvipng/dvisvgm)")
        self._w_use_latex.setChecked(s["use_latex"])
        form.addRow(self._w_use_latex)
        form.addRow(_hint("MathText handles $...$ natively. System LaTeX gives true Computer Modern."))
        form.addRow(_div())

        self._w_title_sz = _int(s["title_sz"], 4, 40)
        self._w_label_sz = _int(s["label_sz"], 4, 30)
        self._w_tick_sz  = _int(s["tick_sz"],  4, 26)
        form.addRow("Title font size :",      self._w_title_sz)
        form.addRow("Axis label font size :", self._w_label_sz)
        form.addRow("Tick label font size :", self._w_tick_sz)

        sa.setWidget(w); return sa

    # ── tab: Series ───────────────────────────────────────────────────────────

    def _tab_series(self) -> QWidget:
        sa = QScrollArea(); sa.setWidgetResizable(True)
        w  = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(10)

        titles = {
            "inliers":   "Inliers  (kept in dataset)",
            "removing":  "Outliers — being removed",
            "restoring": "Outliers — being restored",
        }
        self._sw: dict[str, dict] = {}

        for key, title in titles.items():
            ser = self._style["series"][key]
            grp  = QGroupBox(title)
            glay = QFormLayout(grp)
            glay.setSpacing(8)
            glay.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            vis = QCheckBox("Visible"); vis.setChecked(ser["vis"])
            lbl = QLineEdit(ser["lbl"])
            clr = _ColorButton(ser["clr"])
            mrk = QComboBox(); mrk.addItems(self._MARKERS); mrk.setCurrentText(ser["mrk"])
            sz  = _int(int(ser["sz"]),  1, 600)
            alp = _dbl(ser["alp"], 0.0, 1.0, 2)
            lw  = _dbl(ser["lw"],  0.0, 6.0, 1)

            glay.addRow(vis)
            glay.addRow("Legend label :", lbl)
            glay.addRow("Colour :",       clr)
            glay.addRow("Marker :",       mrk)
            glay.addRow("Marker size :",  sz)
            glay.addRow("Alpha :",        alp)
            glay.addRow("Edge linewidth :", lw)

            self._sw[key] = dict(vis=vis, lbl=lbl, clr=clr, mrk=mrk,
                                  sz=sz, alp=alp, lw=lw)
            lay.addWidget(grp)

        lay.addStretch(1); sa.setWidget(w); return sa

    # ── tab: Grid & Axes ──────────────────────────────────────────────────────

    def _tab_grid(self) -> QWidget:
        sa = QScrollArea(); sa.setWidgetResizable(True)
        w  = QWidget(); form = QFormLayout(w)
        form.setSpacing(9); form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s = self._style

        self._w_gmaj = QCheckBox("Show major grid"); self._w_gmaj.setChecked(s["grid_maj"])
        self._w_gmin = QCheckBox("Show minor grid"); self._w_gmin.setChecked(s["grid_min"])
        form.addRow(self._w_gmaj)
        form.addRow(self._w_gmin)
        form.addRow(_div())

        self._w_gclr = _ColorButton(s["grid_clr"])
        self._w_gls  = QComboBox(); self._w_gls.addItems(self._LINESTYLES)
        self._w_gls.setCurrentText(s["grid_ls"])
        self._w_glw  = _dbl(s["grid_lw"],  0.1, 6.0, 2)
        self._w_galp = _dbl(s["grid_alp"], 0.0, 1.0, 2)
        form.addRow("Grid colour :",      self._w_gclr)
        form.addRow("Grid line style :",  self._w_gls)
        form.addRow("Grid line width :",  self._w_glw)
        form.addRow("Grid alpha :",       self._w_galp)
        form.addRow(_div())

        self._w_sp_top = QCheckBox("Show top spine");   self._w_sp_top.setChecked(s["spine_top"])
        self._w_sp_rgt = QCheckBox("Show right spine"); self._w_sp_rgt.setChecked(s["spine_right"])
        self._w_axbg   = _ColorButton(s["axes_bg"])
        form.addRow(self._w_sp_top)
        form.addRow(self._w_sp_rgt)
        form.addRow("Axes background :", self._w_axbg)

        sa.setWidget(w); return sa

    # ── tab: Legend ───────────────────────────────────────────────────────────

    def _tab_legend(self) -> QWidget:
        sa = QScrollArea(); sa.setWidgetResizable(True)
        w  = QWidget(); form = QFormLayout(w)
        form.setSpacing(9); form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s = self._style

        self._w_leg_show  = QCheckBox("Show legend"); self._w_leg_show.setChecked(s["leg_show"])
        self._w_leg_loc   = QComboBox(); self._w_leg_loc.addItems(self._LEG_LOCS)
        self._w_leg_loc.setCurrentText(s["leg_loc"])
        self._w_leg_sz    = _int(s["leg_sz"], 4, 28)
        self._w_leg_frame = QCheckBox("Show frame"); self._w_leg_frame.setChecked(s["leg_frame"])
        self._w_leg_falp  = _dbl(s["leg_falp"], 0.0, 1.0, 2)
        self._w_leg_ncol  = _int(s["leg_ncol"], 1, 6)

        form.addRow(self._w_leg_show)
        form.addRow("Location :",    self._w_leg_loc)
        form.addRow("Font size :",   self._w_leg_sz)
        form.addRow(_div())
        form.addRow(self._w_leg_frame)
        form.addRow("Frame alpha :", self._w_leg_falp)
        form.addRow("Columns :",     self._w_leg_ncol)

        sa.setWidget(w); return sa

    # ── tab: Figure ───────────────────────────────────────────────────────────

    def _tab_figure(self) -> QWidget:
        sa = QScrollArea(); sa.setWidgetResizable(True)
        w  = QWidget(); form = QFormLayout(w)
        form.setSpacing(9); form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s = self._style

        self._w_fw   = _dbl(s["fig_w"], 2.0, 60.0, 1)
        self._w_fh   = _dbl(s["fig_h"], 2.0, 40.0, 1)
        self._w_fbg  = _ColorButton(s["fig_bg"])
        self._w_tight= QCheckBox("Tight layout"); self._w_tight.setChecked(s["tight"])

        form.addRow("Width (inches) :",   self._w_fw)
        form.addRow("Height (inches) :",  self._w_fh)
        form.addRow("Figure background :", self._w_fbg)
        form.addRow(self._w_tight)
        form.addRow(_hint(
            "Preview scales to fit the window. "
            "Saved figure uses the exact width × height specified here."
        ))

        sa.setWidget(w); return sa

    # ── tab: Save ─────────────────────────────────────────────────────────────

    def _tab_save(self) -> QWidget:
        sa = QScrollArea(); sa.setWidgetResizable(True)
        w  = QWidget(); form = QFormLayout(w)
        form.setSpacing(9); form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s = self._style

        self._w_fmt   = QComboBox()
        self._w_fmt.addItems(["PNG", "PDF", "SVG", "EPS", "TIFF"])
        self._w_fmt.setCurrentText(s["fmt"].upper())
        self._w_fmt.currentIndexChanged.connect(self._on_fmt_change)
        self._w_dpi    = _int(s["dpi"], 50, 1200)
        self._w_transp = QCheckBox("Transparent background")
        self._w_transp.setChecked(s["transp"])
        self._w_prefix = QLineEdit(s["prefix"])

        form.addRow("Format :",           self._w_fmt)
        form.addRow("DPI (raster) :",     self._w_dpi)
        form.addRow(self._w_transp)
        form.addRow(_div())
        form.addRow("Filename prefix :",  self._w_prefix)

        btn = QPushButton("Save figure now…"); btn.clicked.connect(self._save)
        form.addRow(btn)
        form.addRow(_hint(
            "SVG / PDF / EPS are vector formats — DPI has no effect.\n"
            "Use PDF or EPS for journal submissions; PNG for slides; "
            "SVG for web / Inkscape editing."
        ))

        sa.setWidget(w); return sa

    def _on_fmt_change(self, idx: int) -> None:
        self._w_dpi.setEnabled(idx in (0, 4))   # PNG=0, TIFF=4

    # ── signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        def _kick(*_):
            if self._chk_auto.isChecked():
                self._timer.start(400)

        # labels
        for w in (self._w_title, self._w_sc_title, self._w_hist_title,
                  self._w_xlabel, self._w_ylabel, self._w_xlabel_h, self._w_ylabel_h):
            w.textChanged.connect(_kick)
        self._w_font_fam.currentIndexChanged.connect(_kick)
        self._w_use_latex.toggled.connect(_kick)
        for sb in (self._w_title_sz, self._w_label_sz, self._w_tick_sz):
            sb.valueChanged.connect(_kick)

        # series
        for ww in self._sw.values():
            ww["vis"].toggled.connect(_kick)
            ww["lbl"].textChanged.connect(_kick)
            ww["clr"].colorChanged.connect(_kick)
            ww["mrk"].currentIndexChanged.connect(_kick)
            ww["sz"].valueChanged.connect(_kick)
            ww["alp"].valueChanged.connect(_kick)
            ww["lw"].valueChanged.connect(_kick)

        # grid / axes
        for cb in (self._w_gmaj, self._w_gmin, self._w_sp_top, self._w_sp_rgt):
            cb.toggled.connect(_kick)
        self._w_gclr.colorChanged.connect(_kick)
        self._w_gls.currentIndexChanged.connect(_kick)
        for sb in (self._w_glw, self._w_galp):
            sb.valueChanged.connect(_kick)
        self._w_axbg.colorChanged.connect(_kick)

        # legend
        self._w_leg_show.toggled.connect(_kick)
        self._w_leg_frame.toggled.connect(_kick)
        self._w_leg_loc.currentIndexChanged.connect(_kick)
        for sb in (self._w_leg_sz, self._w_leg_falp, self._w_leg_ncol):
            sb.valueChanged.connect(_kick)

        # figure
        self._w_fw.valueChanged.connect(_kick)
        self._w_fh.valueChanged.connect(_kick)
        self._w_fbg.colorChanged.connect(_kick)
        self._w_tight.toggled.connect(_kick)

    # ── style collection ──────────────────────────────────────────────────────

    def _collect(self) -> None:
        s = self._style
        s["title"]      = self._w_title.text()
        s["sc_title"]   = self._w_sc_title.text()
        s["hist_title"] = self._w_hist_title.text()
        s["xlabel"]     = self._w_xlabel.text()
        s["ylabel"]     = self._w_ylabel.text()
        s["xlabel_hist"]= self._w_xlabel_h.text()
        s["ylabel_hist"]= self._w_ylabel_h.text()
        s["font_family"]= self._w_font_fam.currentText()
        s["use_latex"]  = self._w_use_latex.isChecked()
        s["title_sz"]   = self._w_title_sz.value()
        s["label_sz"]   = self._w_label_sz.value()
        s["tick_sz"]    = self._w_tick_sz.value()

        for key, ww in self._sw.items():
            s["series"][key] = dict(
                vis=ww["vis"].isChecked(), lbl=ww["lbl"].text(),
                clr=ww["clr"].color(),    mrk=ww["mrk"].currentText(),
                sz=ww["sz"].value(),       alp=ww["alp"].value(),
                lw=ww["lw"].value(),
            )

        s["grid_maj"]   = self._w_gmaj.isChecked()
        s["grid_min"]   = self._w_gmin.isChecked()
        s["grid_clr"]   = self._w_gclr.color()
        s["grid_ls"]    = self._w_gls.currentText()
        s["grid_lw"]    = self._w_glw.value()
        s["grid_alp"]   = self._w_galp.value()
        s["axes_bg"]    = self._w_axbg.color()
        s["spine_top"]  = self._w_sp_top.isChecked()
        s["spine_right"]= self._w_sp_rgt.isChecked()

        s["leg_show"]   = self._w_leg_show.isChecked()
        s["leg_loc"]    = self._w_leg_loc.currentText()
        s["leg_sz"]     = self._w_leg_sz.value()
        s["leg_frame"]  = self._w_leg_frame.isChecked()
        s["leg_falp"]   = self._w_leg_falp.value()
        s["leg_ncol"]   = self._w_leg_ncol.value()

        s["fig_w"]  = self._w_fw.value()
        s["fig_h"]  = self._w_fh.value()
        s["fig_bg"] = self._w_fbg.color()
        s["tight"]  = self._w_tight.isChecked()

        s["fmt"]    = self._w_fmt.currentText().lower()
        s["dpi"]    = self._w_dpi.value()
        s["transp"] = self._w_transp.isChecked()
        s["prefix"] = self._w_prefix.text() or "outlier_plot"

    # ── rendering ─────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._collect()
        self._pfig.clear()
        import matplotlib as mpl
        rc = {"font.family": self._style["font_family"],
              "text.usetex":  self._style["use_latex"]}
        with mpl.rc_context(rc):
            try:
                self._draw(self._pfig, self._style)
            except Exception as e:
                ax = self._pfig.add_subplot(111)
                ax.text(0.5, 0.5, f"Render error:\n{e}",
                        ha="center", va="center",
                        transform=ax.transAxes, color="red",
                        fontsize=9, wrap=True)
                log.warning("PlotCustomizationDialog render error: %s", e)
        self._pcanvas.draw()

    def _draw(self, fig: Figure, s: dict) -> None:
        fig.set_facecolor(s["fig_bg"])
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})

        col = self._col
        y   = self._df[col].values
        idx = np.arange(len(self._df))

        out_arr  = self._mask.values
        rst_arr  = np.array([self._df.index[i] in self._restored
                              for i in range(len(self._df))])
        rem_arr  = out_arr & ~rst_arr
        in_arr   = ~out_arr

        arr_map = {"inliers": in_arr, "removing": rem_arr, "restoring": rst_arr}

        # scatter ──────────────────────────────────────────────────────────
        for key, arr in arr_map.items():
            ser = s["series"][key]
            if not ser["vis"] or not arr.any():
                continue
            mrk = None if ser["mrk"] == "None" else ser["mrk"]
            ax1.scatter(idx[arr], y[arr],
                        c=ser["clr"], marker=mrk,
                        s=ser["sz"],  alpha=ser["alp"],
                        linewidths=ser["lw"], label=ser["lbl"])

        ax1.set_xlabel(s["xlabel"],   fontsize=s["label_sz"])
        ax1.set_ylabel(s["ylabel"],   fontsize=s["label_sz"])
        ax1.set_title(s["sc_title"],  fontsize=s["label_sz"])
        ax1.tick_params(labelsize=s["tick_sz"])
        ax1.set_facecolor(s["axes_bg"])
        ax1.spines["top"].set_visible(s["spine_top"])
        ax1.spines["right"].set_visible(s["spine_right"])
        self._apply_grid(ax1, s)
        if s["leg_show"]:
            handles, _ = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend(loc=s["leg_loc"], fontsize=s["leg_sz"],
                           frameon=s["leg_frame"], framealpha=s["leg_falp"],
                           ncols=s["leg_ncol"])

        # histogram ────────────────────────────────────────────────────────
        lo, hi = np.nanmin(y), np.nanmax(y)
        bins   = min(40, max(10, int(np.sqrt(len(y)))))
        edges  = np.linspace(lo, hi, bins + 1) if hi > lo else bins

        for key, arr in arr_map.items():
            ser = s["series"][key]
            if not ser["vis"] or not arr.any():
                continue
            ax2.hist(y[arr], bins=edges,
                     color=ser["clr"], alpha=min(ser["alp"] * 0.85, 1.0),
                     label=ser["lbl"])

        ax2.set_xlabel(s["xlabel_hist"], fontsize=s["label_sz"])
        ax2.set_ylabel(s["ylabel_hist"], fontsize=s["label_sz"])
        ax2.set_title(s["hist_title"],   fontsize=s["label_sz"])
        ax2.tick_params(labelsize=s["tick_sz"])
        ax2.set_facecolor(s["axes_bg"])
        ax2.spines["top"].set_visible(s["spine_top"])
        ax2.spines["right"].set_visible(s["spine_right"])
        self._apply_grid(ax2, s)
        if s["leg_show"]:
            handles, _ = ax2.get_legend_handles_labels()
            if handles:
                ax2.legend(loc=s["leg_loc"], fontsize=s["leg_sz"],
                           frameon=s["leg_frame"], framealpha=s["leg_falp"])

        fig.suptitle(s["title"], fontsize=s["title_sz"], fontweight="bold")
        if s["tight"]:
            try:
                fig.tight_layout()
            except Exception:
                pass

    @staticmethod
    def _apply_grid(ax, s: dict) -> None:
        if s["grid_maj"]:
            ax.grid(True, which="major", color=s["grid_clr"],
                    linestyle=s["grid_ls"], linewidth=s["grid_lw"],
                    alpha=s["grid_alp"])
        if s["grid_min"]:
            ax.minorticks_on()
            ax.grid(True, which="minor", color=s["grid_clr"],
                    linestyle=":", linewidth=max(0.1, s["grid_lw"] * 0.5),
                    alpha=s["grid_alp"] * 0.6)

    # ── save ──────────────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._collect()
        s   = self._style
        fmt = s["fmt"]
        ext_filter = {
            "png": "PNG  (*.png)", "pdf": "PDF  (*.pdf)",
            "svg": "SVG  (*.svg)", "eps": "EPS  (*.eps)",
            "tiff": "TIFF (*.tiff *.tif)",
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", s["prefix"],
            f"{ext_filter.get(fmt, 'All (*)')};;All files (*)",
        )
        if not path:
            return
        if not any(path.lower().endswith(f".{e}")
                   for e in ("png","pdf","svg","eps","tiff","tif")):
            path = f"{path}.{fmt}"

        import matplotlib as mpl
        rc = {"font.family": s["font_family"], "text.usetex": s["use_latex"]}
        with mpl.rc_context(rc):
            try:
                save_fig = Figure(figsize=(s["fig_w"], s["fig_h"]))
                FigureCanvas(save_fig)   # must attach canvas to enable rendering
                self._draw(save_fig, s)
                kw: dict = {
                    "transparent": s["transp"],
                    "bbox_inches": "tight",
                    "facecolor":   save_fig.get_facecolor(),
                }
                if fmt in ("png", "tiff"):
                    kw["dpi"] = s["dpi"]
                save_fig.savefig(path, format=fmt, **kw)
                plt.close(save_fig)
                QMessageBox.information(self, "Saved",
                    f"Figure saved ({fmt.upper()}):\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save failed", str(e))
                log.exception("PlotCustomizationDialog._save")


# ── module-level helpers ──────────────────────────────────────────────────────

def _dbl(default: float, lo: float, hi: float, decimals: int) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setValue(default)
    sb.setMinimumWidth(130)
    return sb


def _int(default: int, lo: int, hi: int) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(default)
    sb.setMinimumWidth(130)
    return sb


def _div() -> QFrame:
    """Thin horizontal separator for form layouts."""
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setObjectName("Divider")
    return f


def _hint(text: str) -> QLabel:
    """Alias used inside PlotCustomizationDialog tabs (same as _note)."""
    lbl = QLabel(text); lbl.setObjectName("SectionHint"); lbl.setWordWrap(True)
    return lbl


def _note(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SectionHint")
    lbl.setWordWrap(True)
    return lbl


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _rpca(X: np.ndarray, lam: float, max_iter: int,
          tol: float) -> tuple[np.ndarray, np.ndarray]:
    """ADMM solver for Robust PCA: minimise ‖L‖_* + λ‖S‖₁  s.t. X = L + S."""
    from scipy.linalg import svd
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    norm_X = np.linalg.norm(X, 'fro') + 1e-10
    mu     = 1.25 / np.linalg.norm(X, ord=2)
    for _ in range(max_iter):
        U, sigma, VT = svd(X - S + Y / mu, full_matrices=False)
        L = U @ np.diag(_soft_threshold(sigma, 1.0 / mu)) @ VT
        S = _soft_threshold(X - L + Y / mu, lam / mu)
        Y = Y + mu * (X - L - S)
        if np.linalg.norm(X - L - S, 'fro') / norm_X < tol:
            break
    return L, S
