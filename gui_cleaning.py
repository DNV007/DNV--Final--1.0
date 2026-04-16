"""
DNV Scientific Module
---------------------
Role:
    Provides DataCleaningWindow — a 7-tab revision workspace for
    contract-aware data cleaning operations on the active DataAsset.

Scientific Context:
    Exposes column management, derived-column creation, missing-value
    imputation, normalisation, outlier removal, SEAL conditioning, and
    transformation as sequential or independent operations on the active
    DataAsset.

Invariants:
    - apply_cleaning_operation is the single site that writes a new
      DataAsset to the SessionStore.
    - Every cleaning operation records a ProvenanceRecord before updating
      the store.

Assumptions:
    - store.asset is a valid DataAsset with at least one numeric column
      before any cleaning tab is activated.
    - All sub-dialogs return a cleaned DataFrame and a summary string.

Failure Modes:
    - No asset loaded: cleaning tabs display a warning label rather than
      controls.
    - Sub-dialog cancelled: store is not updated.

Provenance:
    - Emits a ProvenanceRecord per apply_cleaning_operation call via
      DataAsset.with_provenance.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd
from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
    QFrame, QLabel, QMessageBox, QPushButton,
    QTableView, QTabWidget, QVBoxLayout,
)

from ing_config    import IngestionConfig
from ing_pipeline  import derive_data_asset
from ing_contracts import assess_cleaning_operation, attach_contracts_to_asset
from gui_session   import SessionStore
from gui_base      import ModuleDialog
from gui_widgets   import DataFrameModel, ReportTextPanel, make_section_header
from gui_dlg_columns  import ManageColumnsDialog
from gui_dlg_derived  import DerivedColumnDialog
from gui_dlg_missing  import MissingDataDialog
from gui_dlg_outliers   import OutlierRemovalDialog
from gui_dlg_normalize  import NormalizeDialog
from gui_dlg_seal       import SEALDialog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback factories — module-level to avoid inline closures (DNV style)
# ---------------------------------------------------------------------------

def _make_manage_columns_cb(parent: "DataCleaningWindow"):
    """Factory: returns a zero-arg callable that opens ManageColumnsDialog."""
    return lambda: ManageColumnsDialog(parent).exec()


def _make_derived_column_cb(parent: "DataCleaningWindow"):
    """Factory: returns a zero-arg callable that opens DerivedColumnDialog."""
    return lambda: DerivedColumnDialog(parent).exec()


def _make_missing_data_cb(parent: "DataCleaningWindow"):
    """Factory: returns a zero-arg callable that opens MissingDataDialog."""
    return lambda: MissingDataDialog(parent).exec()


def _make_outlier_removal_cb(parent: "DataCleaningWindow"):
    """Factory: returns a zero-arg callable that opens OutlierRemovalDialog."""
    return lambda: OutlierRemovalDialog(parent).exec()


def _make_normalize_cb(parent: "DataCleaningWindow"):
    """Factory: returns a zero-arg callable that opens NormalizeDialog."""
    return lambda: NormalizeDialog(parent).exec()


def _make_seal_cb(store: "SessionStore", parent: "DataCleaningWindow"):
    """Factory: returns a zero-arg callable that opens SEALDialog."""
    return lambda: SEALDialog(store, parent, parent).exec()


class DataCleaningWindow(ModuleDialog):
    """Revision workspace: 8 tabs — Overview, Columns, Derived, Missing Data, Outlier Removal, Normalize/Scale, SEAL Conditioning, Ledger."""

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Data Cleaning",
            hint="Condition, constrain, and revise the active dataset. Each accepted operation creates a new provenance-bound revision.",
            category="DATA QUALITY · 8 WORKSPACES", parent=parent,
        )
        self._store = store
        self._asset = store.asset

        if self._asset is None:
            note = QLabel("No active dataset. Load a dataset in Data Ingestion first.")
            note.setObjectName("SectionHint"); note.setWordWrap(True)
            self.body_layout.addWidget(note); return

        self.body_layout.addWidget(make_section_header(
            "Asset revision workspace",
            "Every accepted operation creates a fresh DataAsset revision in the session store.",
            "8 WORKSPACES",
        ))

        summary_card = QFrame(); summary_card.setObjectName("PanelCard")
        sl = QVBoxLayout(summary_card); sl.setContentsMargins(18,18,18,18); sl.setSpacing(10)
        self.summary_lbl = QLabel(); self.summary_lbl.setObjectName("SectionHint")
        self.summary_lbl.setWordWrap(True); sl.addWidget(self.summary_lbl)
        self.body_layout.addWidget(summary_card, 0)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.addTab(self._overview_tab(),  "Overview")
        self.tabs.addTab(self._columns_tab(),   "Columns")
        self.tabs.addTab(self._derived_tab(),   "Derived")
        self.tabs.addTab(self._missing_tab(),   "Missing")
        self.tabs.addTab(self._outlier_tab(),   "Outliers")
        self.tabs.addTab(self._normalize_tab(), "Normalize")
        self.tabs.addTab(self._seal_tab(),      "SEAL")
        self.tabs.addTab(self._ledger_tab(),    "Ledger")
        self.body_layout.addWidget(self.tabs, 1)
        self.refresh_view()

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _overview_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint("Preview of the active session asset. The module never edits in place."))
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QTableView.SelectRows)
        lay.addWidget(self.preview_table, 1); return w

    def _columns_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint("Retain, rename, and reorder observables."))
        self.btn_manage = QPushButton("Open column manager")
        self.btn_manage.clicked.connect(_make_manage_columns_cb(self))
        lay.addWidget(self.btn_manage, 0, Qt.AlignLeft)
        self.columns_info = ReportTextPanel("COLUMN MANAGEMENT", "Identifier removals flagged before commit.")
        lay.addWidget(self.columns_info, 1); return w

    def _derived_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint("Create provenance-bound derived observables from constrained expressions."))
        self.btn_derive = QPushButton("Create derived column")
        self.btn_derive.clicked.connect(_make_derived_column_cb(self))
        lay.addWidget(self.btn_derive, 0, Qt.AlignLeft)
        self.derived_info = ReportTextPanel("DERIVED OBSERVABLES", "Expressions validated syntactically and recorded.")
        lay.addWidget(self.derived_info, 1); return w

    def _missing_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint("Apply deterministic missing-data operations with dtype-aware guards."))
        self.btn_missing = QPushButton("Open missing-data operator")
        self.btn_missing.clicked.connect(_make_missing_data_cb(self))
        lay.addWidget(self.btn_missing, 0, Qt.AlignLeft)
        self.missing_info = ReportTextPanel("MISSING-DATA OPERATIONS", "Interpolation restricted to numeric observables.")
        lay.addWidget(self.missing_info, 1); return w

    def _outlier_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint(
            "Detect and remove anomalous observations using 15 algorithms across 5 families: "
            "statistical, distance-based, clustering, machine-learning, and advanced methods. "
            "Each accepted operation creates a provenance-bound revision."
        ))
        self.btn_outlier = QPushButton("Open outlier removal operator")
        self.btn_outlier.clicked.connect(_make_outlier_removal_cb(self))
        lay.addWidget(self.btn_outlier, 0, Qt.AlignLeft)
        self.outlier_info = ReportTextPanel(
            "OUTLIER REMOVAL OPERATIONS",
            "All methods operate on numeric columns only. NaN cells are median-imputed before fitting."
        )
        lay.addWidget(self.outlier_info, 1); return w

    def _normalize_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint(
            "Normalize or scale numeric columns using 8 methods across 3 families: "
            "linear (Min-Max, Standard, Robust), non-linear (Log, Box-Cox, Yeo-Johnson, Quantile), "
            "and clipping (Winsorization). Before/after skewness and kurtosis diagnostics are shown."
        ))
        self.btn_normalize = QPushButton("Open normalize/scale operator")
        self.btn_normalize.clicked.connect(_make_normalize_cb(self))
        lay.addWidget(self.btn_normalize, 0, Qt.AlignLeft)
        self.normalize_info = ReportTextPanel(
            "NORMALIZE/SCALE OPERATIONS",
            "All methods operate on selected numeric columns only."
        )
        lay.addWidget(self.normalize_info, 1); return w

    def _seal_tab(self):
        from PySide6.QtWidgets import QWidget
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)
        lay.addWidget(self._hint(
            "Distribution-Constrained Data Conditioning (SEAL — DNV-1.0 §1). "
            "Searches for minimal, contract-feasible transformations that move the empirical "
            "distribution into a user-declared admissible envelope Q(θ), returning K distinct "
            "admissible variants. EVI scores identify which samples drive constraint violations."
        ))
        btn = QPushButton("Open SEAL operator")
        btn.clicked.connect(_make_seal_cb(self._store, self))
        lay.addWidget(btn, 0, Qt.AlignLeft)
        self.seal_info = ReportTextPanel("SEAL CONDITIONING", "No SEAL conditioning operations recorded.")
        lay.addWidget(self.seal_info, 1)
        return w

    def _ledger_tab(self):
        from PySide6.QtWidgets import QWidget, QHBoxLayout
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)

        btn_row = QHBoxLayout()
        self.btn_replay = QPushButton("Replay Pipeline")
        self.btn_replay.setToolTip(
            "Re-apply all recorded operations from a source dataset "
            "(DNV-2.0: replayable provenance logs)"
        )
        self.btn_replay.clicked.connect(self._replay_pipeline)
        btn_row.addWidget(self.btn_replay)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.status_panel = ReportTextPanel("REVISION LEDGER", "Audit trail for the current cleaning session.")
        lay.addWidget(self.status_panel, 1); return w

    def _replay_pipeline(self) -> None:
        """Replay the full provenance pipeline on the original source data."""
        if self._asset is None:
            QMessageBox.warning(self, "No asset", "No active dataset."); return
        history = self._asset.metadata.get("pipeline_history", [])
        if not history:
            QMessageBox.information(self, "Nothing to replay", "No pipeline operations recorded."); return
        source = self._asset.source_path
        if not source:
            QMessageBox.warning(self, "No source", "Cannot replay: source file path not recorded."); return

        from ing_provenance import replay_pipeline
        import replay_handlers  # noqa: F401 — registers handlers on import
        from ing_loader import load_file

        try:
            source_df = load_file(source)
            result_df = replay_pipeline(source_df, history)
            summary = f"Replayed {len(history)} operations → {result_df.shape[0]} rows × {result_df.shape[1]} cols"
            self.apply_cleaning_operation(
                "replay_pipeline", result_df, summary,
                {"n_operations": len(history), "source_path": str(source)},
            )
        except Exception as exc:
            log.exception("DataCleaningWindow._replay_pipeline")
            QMessageBox.critical(self, "Replay Error", str(exc))

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text); lbl.setObjectName("SectionHint"); lbl.setWordWrap(True); return lbl

    # ── Public interface used by dialogs ──────────────────────────────────────

    def current_dataframe(self) -> pd.DataFrame:
        return self._asset.dataframe.copy() if self._asset else pd.DataFrame()

    def _contracts(self) -> List[dict]:
        return list((self._asset.metadata.get("observable_contracts") or []) if self._asset else [])

    def refresh_view(self) -> None:
        if self._asset is None: return
        df   = self._asset.dataframe; meta = self._asset.metadata
        hist = list(meta.get("pipeline_history") or [])
        ctrs = self._contracts()

        self.summary_lbl.setText(
            f"Rows {df.shape[0]:,} · Columns {df.shape[1]:,} · "
            f"Hash {str(meta.get('dataframe_sha256','NA'))[:12]}… · "
            f"Revisions {len(hist)} · Bound contracts {len(ctrs)}"
        )
        self.preview_table.setModel(DataFrameModel(df.head(100)))
        self.preview_table.resizeColumnsToContents()

        self.columns_info.setPlainText(
            "Current columns\n===============\n" + "\n".join(str(c) for c in df.columns[:60])
        )
        drv = [h.get("params",{}).get("new_column_name","") for h in hist if h.get("operation") == "create_derived_column"]
        self.derived_info.setPlainText(
            "Derived observables\n===================\n" +
            ("\n".join(f"- {n}" for n in drv if n) if drv else "No derived observables recorded.")
        )
        miss = df.isna().sum(); miss = miss[miss > 0].sort_values(ascending=False)
        self.missing_info.setPlainText(
            "Missingness overview\n====================\n" +
            ("\n".join(f"{str(c):<36} {int(v):>8,}" for c, v in list(miss.items())[:40])
             if len(miss) else "No missing values in the active revision.")
        )
        outl_ops = [h for h in hist if h.get("operation") == "remove_outliers"]
        self.outlier_info.setPlainText(
            "Outlier removal history\n=======================\n" +
            ("\n".join(
                f"- {h.get('params',{}).get('method','?'):<36} "
                f"removed {h.get('params',{}).get('rows_removed','?')} rows · "
                f"{h.get('applied_utc','')}"
                for h in outl_ops
            ) if outl_ops else "No outlier removal operations recorded.")
        )
        norm_ops = [h for h in hist if h.get("operation") == "normalize_scale"]
        self.normalize_info.setPlainText(
            "Normalize/Scale history\n=======================\n" +
            ("\n".join(
                f"- {h.get('params',{}).get('method','?'):<36} "
                f"cols {h.get('params',{}).get('n_cols','?')} · "
                f"{h.get('applied_utc','')}"
                for h in norm_ops
            ) if norm_ops else "No normalization operations recorded.")
        )
        seal_ops = [h for h in hist if h.get("operation") == "seal_conditioning"]
        self.seal_info.setPlainText(
            "SEAL conditioning history\n=========================\n" +
            ("\n".join(
                f"- v{h.get('params',{}).get('variant_id','?')}  "
                f"Ω={h.get('params',{}).get('distortion','?'):.4f}  "
                f"d_after={h.get('params',{}).get('d_after','?'):.4f}  "
                f"{h.get('applied_utc','')}"
                for h in seal_ops
            ) if seal_ops else "No SEAL conditioning operations recorded.")
        )
        lines = [
            "[OK]   Session asset available",
            f"[OK]   Hash: {str(meta.get('dataframe_sha256','NA'))}",
            f"[{'OK' if ctrs else 'WARN'}]   {'Bound contracts: '+str(len(ctrs)) if ctrs else 'No bound contracts; guards fall back to dtype families.'}",
        ]
        for entry in hist[-8:]:
            lines.append(f"[OK]   {entry.get('operation','op')} · {entry.get('applied_utc','')}"[:120])
        if not hist: lines.append("[OK]   No cleaning revisions applied yet.")
        self.status_panel.setPlainText("\n".join(lines))

    def apply_cleaning_operation(self, operation: str, dataframe: pd.DataFrame,
                                  summary: str, params: dict) -> None:
        if self._asset is None:
            QMessageBox.warning(self, "No asset", "No active dataset."); return
        ctrs  = self._contracts()
        guard = assess_cleaning_operation(self._asset.dataframe,
                                          operation=operation, params=params, contracts=ctrs)
        if guard.get("violations"):
            QMessageBox.critical(self, "Operation blocked", "\n".join(guard["violations"])); return
        if guard.get("warnings"):
            QMessageBox.warning(self, "Admissibility warning", "\n".join(guard["warnings"]))

        new_asset = derive_data_asset(
            dataframe=dataframe, source_path=self._asset.source_path,
            cfg=IngestionConfig(), prior_asset=self._asset,
            operation=operation, operation_params=params,
        )
        if ctrs: new_asset = attach_contracts_to_asset(new_asset, ctrs)
        self._store.set_asset(new_asset); self._asset = new_asset
        self.refresh_view()
        cur = self.status_panel.toPlainText().strip()
        extra = [f"[OK]   {summary}"] + [f"[WARN] {w}" for w in guard.get("warnings",[])]
        self.status_panel.setPlainText((cur + "\n" + "\n".join(extra)).strip())
        QMessageBox.information(self, "Cleaning applied", summary)
