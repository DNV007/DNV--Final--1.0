"""
DNV Scientific Module
---------------------
Role:
    Provides DataIngestionWindow — the file-load, contract-declaration,
    and provenance-binding entry point for the DNV pipeline.

Scientific Context:
    Operates on raw file paths; invokes the ingestion pipeline
    (ing_pipeline.build_data_asset) in a background IngestWorker thread
    and stores the resulting DataAsset in the SessionStore for all
    downstream modules.

Invariants:
    - Only one IngestWorker runs at a time; a running worker is cancelled
      before a new load starts.
    - The SessionStore is updated only after a successful build_data_asset
      return.

Assumptions:
    - The selected file is readable and in a supported format
      (CSV/TSV/Excel/Parquet/JSON).
    - store is a valid SessionStore instance.

Failure Modes:
    - File unreadable or unsupported format: IngestWorker emits err signal;
      dialog shows QMessageBox.critical.
    - User cancels mid-load: worker is cancelled and store is not updated.

Provenance:
    - Provenance is embedded in the DataAsset produced by build_data_asset;
      this module binds it to the SessionStore.
"""
from __future__ import annotations

import dataclasses, logging, os
from typing import List, Optional

import pandas as pd
from PySide6.QtCore    import Qt
from PySide6.QtGui     import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton,
    QTableView, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from ing_config    import IngestionConfig
from ing_provenance import save_json, utc_now_iso
from ing_analysis  import infer_schema
from ing_contracts import attach_contracts_to_asset, validate_contracts
from ing_rendering import DataIngestorV2
from gui_session   import SessionStore
from gui_theme     import theme
from gui_base      import ModuleDialog, IngestWorker
from gui_widgets   import DataFrameModel, ReportTextPanel, APP_ANALYST, APP_TITLE, APP_VERSION

log = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataIngestionWindow(ModuleDialog):
    """Load datasets, declare observable contracts, validate, export reports."""

    CONTRACT_COLUMNS = [
        "observable","denotes","kind","units",
        "lower_bound","upper_bound","nullable",
        "admissible_operations","derivation","reference_convention","role",
        "protocol","proxy_group",
    ]

    def __init__(self, store: SessionStore, parent=None):
        super().__init__(
            title="Data Ingestion",
            hint="Load a dataset, inspect schema, declare observable contracts, and validate each variable against its stated constraints.",
            category="DATA INPUT", parent=parent,
        )
        self._store  = store
        self._asset  = store.asset
        self._df:     Optional[pd.DataFrame] = None if self._asset is None else self._asset.dataframe
        self._stats:  Optional[dict]         = None if self._asset is None else self._asset.statistics
        self._report: Optional[dict]         = None if self._asset is None else self._asset.integrity_report
        self._path:   Optional[str]          = None if self._asset is None else self._asset.source_path
        self._worker: Optional[IngestWorker] = None

        self._build_toolbar()
        self._build_tabs()
        self._wire()
        if self._asset is not None:
            self._apply_asset(self._asset)
            self._ok("Restored dataset from session store")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = QFrame(); bar.setObjectName("PanelCard")
        bl  = QVBoxLayout(bar); bl.setContentsMargins(14,12,14,12); bl.setSpacing(10)

        r1 = QHBoxLayout(); r1.setSpacing(8)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Open a dataset  —  CSV / TSV / TXT / Parquet / Excel / JSON")
        self.btn_open = QPushButton("Open dataset…")
        self.btn_save = QPushButton("Save report"); self.btn_save.setEnabled(self._asset is not None)
        r1.addWidget(QLabel("Dataset:")); r1.addWidget(self.path_edit, 1)
        r1.addWidget(self.btn_open); r1.addWidget(self.btn_save)
        bl.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(16)
        self.chk_norm  = QCheckBox("Normalise columns"); self.chk_norm.setChecked(True)
        self.chk_strip = QCheckBox("Strip whitespace");  self.chk_strip.setChecked(True)
        self.chk_num   = QCheckBox("Coerce numeric")
        self.chk_dt    = QCheckBox("Parse dates")
        r2.addWidget(QLabel("Ingestion options:"))
        for c in (self.chk_norm, self.chk_strip, self.chk_num, self.chk_dt): r2.addWidget(c)
        r2.addStretch(1); bl.addLayout(r2)

        lbl = QLabel("Status"); lbl.setObjectName("SectionTitle"); bl.addWidget(lbl)
        self.status_text = QTextEdit(); self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120); self.status_text.setFont(QFont("Courier New", 10))
        bl.addWidget(self.status_text)
        self.body_layout.addWidget(bar)

    def _build_tabs(self) -> None:
        self.tabs = QTabWidget()
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(False)
        self.tabs.addTab(self.preview_table, "Preview")
        self.tabs.addTab(self._contracts_panel(), "Observable Contracts")
        self.tabs.addTab(self._validation_panel(), "Validation")
        self.stats_text     = ReportTextPanel("STATISTICAL SUMMARY",   "Copy-ready descriptive output.")
        self.integrity_text = ReportTextPanel("INTEGRITY REPORT",      "Structured ingestion diagnostics.")
        self.missing_text   = ReportTextPanel("MISSING-DATA SCAN",     "Column-wise missingness.")
        self.tabs.addTab(self.stats_text,     "Statistics")
        self.tabs.addTab(self.integrity_text, "Integrity Report")
        self.tabs.addTab(self.missing_text,   "Missing Data")
        self.body_layout.addWidget(self.tabs, 1)

    def _contracts_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)
        h = QLabel("Each row declares what a column denotes and constrains its admissible values.")
        h.setObjectName("SectionHint"); h.setWordWrap(True); lay.addWidget(h)
        br = QHBoxLayout(); br.setSpacing(8)
        self.btn_rebuild = QPushButton("Rebuild from schema")
        self.btn_export  = QPushButton("Export JSON"); self.btn_export.setEnabled(False)
        br.addWidget(self.btn_rebuild); br.addWidget(self.btn_export); br.addStretch(1); lay.addLayout(br)
        self.contract_table = QTableWidget(0, len(self.CONTRACT_COLUMNS))
        self.contract_table.setHorizontalHeaderLabels(self.CONTRACT_COLUMNS)
        self.contract_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.contract_table.horizontalHeader().setStretchLastSection(True)
        self.contract_table.verticalHeader().setVisible(False)
        self.contract_table.setAlternatingRowColors(True)
        lay.addWidget(self.contract_table, 1)
        return w

    def _validation_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)
        h = QLabel("Validates each observable against declared kind, units, bounds, and nullability.")
        h.setObjectName("SectionHint"); h.setWordWrap(True); lay.addWidget(h)
        br = QHBoxLayout()
        self.btn_validate = QPushButton("Run validation"); self.btn_validate.setEnabled(False)
        br.addWidget(self.btn_validate); br.addStretch(1); lay.addLayout(br)
        self.validation_table = QTableWidget(0, 5)
        self.validation_table.setHorizontalHeaderLabels(["Observable","Status","Checks run","Violations","Warnings"])
        self.validation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.validation_table.verticalHeader().setVisible(False)
        self.validation_table.setAlternatingRowColors(True)
        self.validation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.validation_table, 1)
        return w

    def _wire(self) -> None:
        self.btn_open.clicked.connect(self._open_dialog)
        self.btn_save.clicked.connect(self._save_report)
        self.btn_rebuild.clicked.connect(self._rebuild_contracts)
        self.btn_export.clicked.connect(self._export_contracts)
        self.btn_validate.clicked.connect(self._run_validation)

    # ── Config ────────────────────────────────────────────────────────────────

    def _cfg(self) -> IngestionConfig:
        return IngestionConfig(
            normalize_columns=self.chk_norm.isChecked(),
            strip_whitespace=self.chk_strip.isChecked(),
            coerce_numeric=self.chk_num.isChecked(),
            parse_dates=self.chk_dt.isChecked(),
        )

    # ── Status helpers ────────────────────────────────────────────────────────

    def _log(self, line: str) -> None:
        cur = self.status_text.toPlainText().strip()
        self.status_text.setPlainText((cur + "\n" + line).strip())
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    def _ok(self, m: str)   -> None: self._log(f"[OK]   {m}")
    def _warn(self, m: str) -> None: self._log(f"[WARN] {m}")
    def _fail(self, m: str) -> None: self._log(f"[FAIL] {m}")

    # ── File loading ──────────────────────────────────────────────────────────

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset", BASE_DIR,
            "Datasets (*.csv *.tsv *.txt *.parquet *.xlsx *.xls *.json *.jsonl *.ndjson);;All files (*.*)")
        if path: self._ingest(path)

    def _ingest(self, path: str) -> None:
        path = path.strip()
        if not path: return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Not found", f"File not found:\n{path}"); return

        if self._store.has_asset():
            existing = self._store.asset.source_path or ""
            if os.path.abspath(existing) != os.path.abspath(path):
                ans = QMessageBox.question(self, "Overwrite dataset",
                    f"Current: {existing}\nNew: {path}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if ans != QMessageBox.Yes: return

        if self._worker and self._worker.isRunning():
            self._worker.cancel(); self._worker.quit(); self._worker.wait(3000)

        self._path = path; self.path_edit.setText(path)
        for b in (self.btn_open, self.btn_save, self.btn_export, self.btn_validate):
            b.setEnabled(False)
        self._log(f"[…]   Ingesting {os.path.basename(path)}")
        for t in (self.stats_text, self.integrity_text, self.missing_text):
            t.setPlainText("Ingestion in progress…")

        self._worker = IngestWorker(path, self._cfg())
        self._worker.ok.connect(self._on_ok)
        self._worker.err.connect(self._on_err)
        self._worker.start()

    def _on_ok(self, asset) -> None:
        self._worker = None
        self._store.set_asset(asset)
        self._apply_asset(asset)
        self.btn_open.setEnabled(True); self.btn_save.setEnabled(True)
        self._ok(f"Loaded {asset.dataframe.shape[0]:,} rows × {asset.dataframe.shape[1]:,} cols"
                 f"  [{asset.metadata.get('analysis_id','')}]")

    def _on_err(self, msg: str) -> None:
        self._worker = None; self.btn_open.setEnabled(True)
        self._fail("Ingestion failed — see log"); QMessageBox.critical(self, "Ingestion failed", msg[:800])

    def _apply_asset(self, asset) -> None:
        self._asset  = asset; self._df = asset.dataframe
        self._stats  = asset.statistics; self._report = asset.integrity_report
        self._path   = asset.source_path; self.path_edit.setText(self._path or "")
        self.preview_table.setModel(DataFrameModel(asset.dataframe.head(200)))
        self.preview_table.resizeColumnsToContents()
        r = DataIngestorV2(cfg=self._cfg())
        self.stats_text.setPlainText(r.basic_stats_text(asset.dataframe))
        self.integrity_text.setPlainText(r.integrity_text(asset.integrity_report))
        mbc = asset.integrity_report.get("missingness", {}).get("missing_by_column", {})
        if mbc:
            lines = [f"{'Column':<40}  {'Missing':>8}", "-"*52]
            for col, cnt in sorted(mbc.items(), key=lambda x: -x[1]):
                lines.append(f"{col:<40}  {cnt:>8,}")
            self.missing_text.setPlainText("\n".join(lines))
        else:
            self.missing_text.setPlainText("No missing values detected.")
        self.btn_validate.setEnabled(True)

    # ── Contracts ─────────────────────────────────────────────────────────────

    @staticmethod
    def _kind(dtype: str) -> str:
        d = dtype.lower()
        if any(t in d for t in ("int","float","complex")): return "numeric"
        if any(t in d for t in ("datetime","timestamp")):  return "datetime"
        if "bool" in d:                                    return "boolean"
        return "categorical"

    @staticmethod
    def _units(col: str) -> str:
        c = col.lower()
        for tok, u in [("temperature","K"),("temp","K"),("pressure","Pa"),
                       ("energy","eV"),("bandgap","eV"),("voltage","V"),
                       ("length","m"),("mass","g"),("density","g/cm³"),
                       ("time","s"),("frequency","Hz")]:
            if tok in c: return u
        return ""

    @staticmethod
    def _role(col: str) -> str:
        c = col.lower()
        if any(t in c for t in ("id","uuid","key","index","identifier")): return "identifier"
        if any(t in c for t in ("target","label","y","output","response")): return "target"
        return "feature"

    def _defaults(self, col: str, meta: dict) -> dict:
        return {"observable": col, "denotes": "",
                "kind": self._kind(meta.get("dtype","object")), "units": self._units(col),
                "lower_bound": "", "upper_bound": "", "nullable": "true",
                "admissible_operations": "view, summarise", "derivation": "declared observable",
                "reference_convention": "", "role": self._role(col),
                "protocol": "", "proxy_group": ""}

    def _rebuild_contracts(self) -> None:
        if self._df is None: return
        schema    = infer_schema(self._df, cfg=self._cfg())
        contracts = [self._defaults(col, meta) for col, meta in schema["columns"].items()]
        self._populate_table(contracts)
        self._bind(contracts)
        self.btn_export.setEnabled(True)

    def _populate_table(self, contracts: List[dict]) -> None:
        self.contract_table.setRowCount(0)
        for ri, row in enumerate(contracts):
            self.contract_table.insertRow(ri)
            for ci, key in enumerate(self.CONTRACT_COLUMNS):
                self.contract_table.setItem(ri, ci, QTableWidgetItem(str(row.get(key,""))))
        self.contract_table.resizeColumnsToContents()

    def _collect(self) -> List[dict]:
        rows = []
        for r in range(self.contract_table.rowCount()):
            row: dict = {}
            for c, key in enumerate(self.CONTRACT_COLUMNS):
                item = self.contract_table.item(r, c)
                row[key] = item.text().strip() if item else ""
            rows.append(row)
        return rows

    def _bind(self, contracts: List[dict]) -> None:
        if self._asset is None: return
        try:
            b = attach_contracts_to_asset(self._asset, contracts)
            self._store.set_asset(b); self._asset = b; self._report = b.integrity_report
        except Exception: log.exception("Contract binding failed")

    def _save_report(self) -> None:
        if not self._report: return
        out, _ = QFileDialog.getSaveFileName(self, "Save report", BASE_DIR, "JSON (*.json)")
        if not out: return
        try:   save_json(self._report, out); QMessageBox.information(self, "Saved", f"Report saved:\n{out}")
        except OSError as e: QMessageBox.critical(self, "Save failed", str(e))

    def _export_contracts(self) -> None:
        out, _ = QFileDialog.getSaveFileName(self, "Export contracts", BASE_DIR, "JSON (*.json)")
        if not out: return
        payload = {
            "schema_version": {"ingestion":"3.0","pipeline":"1.1"},
            "dataset_file": self._path, "exported_utc": utc_now_iso(),
            "analysis_id":  (self._report.get("meta",{}).get("analysis_id","") if self._report else ""),
            "analyst": APP_ANALYST, "generator": f"{APP_TITLE} v{APP_VERSION}",
            "loader_config": dataclasses.asdict(self._cfg()),
            "dataframe_sha256": (self._report.get("meta",{}).get("dataframe_sha256","") if self._report else ""),
            "contracts": self._collect(),
        }
        try:   save_json(payload, out); QMessageBox.information(self, "Exported", f"Contracts exported:\n{out}")
        except OSError as e: QMessageBox.critical(self, "Export failed", str(e))

    def _run_validation(self) -> None:
        if self._df is None:
            QMessageBox.warning(self, "No data", "Load a dataset first."); return
        contracts = self._collect()
        if not contracts:
            QMessageBox.warning(self, "No contracts", "Define at least one contract first."); return
        self._bind(contracts)
        results = validate_contracts(self._df, contracts)
        self.status_text.clear(); self.validation_table.setRowCount(0)
        COLORS = {"PASS":("#00b86a","#1a7a4a"),"FAIL":("#e05050","#b03030"),
                  "WARN":("#c88030","#a06010"),"SKIP":("#7a8a96","#5a6a78")}
        for ri, res in enumerate(results):
            status = res.get("status","SKIP")
            if status == "FAIL": self._fail(f"Contract violation: {res.get('observable','?')}")
            elif status == "WARN": self._warn(f"Contract warning: {res.get('observable','?')}")
            self.validation_table.insertRow(ri)
            cd, cl = COLORS.get(status, ("#7a8a96","#5a6a78"))
            stat_item = QTableWidgetItem(status)
            stat_item.setForeground(QColor(cd if theme.is_dark() else cl))
            items = [QTableWidgetItem(res["observable"]), stat_item,
                     QTableWidgetItem(str(res["checks_run"])),
                     QTableWidgetItem("; ".join(res.get("violations",[])) or "—"),
                     QTableWidgetItem("; ".join(res.get("warnings",[])) or "—")]
            for ci, item in enumerate(items):
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.validation_table.setItem(ri, ci, item)
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Validation":
                self.tabs.setCurrentIndex(i); break
