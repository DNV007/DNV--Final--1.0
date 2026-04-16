"""
DNV Scientific Module
---------------------
Role:
    Provides ManageColumnsDialog — a retain/rename dialog for column
    management within a provenance-bound edit.

Scientific Context:
    No scientific computation; maps existing column names to user-declared
    retain/rename decisions and returns an updated DataFrame.

Invariants:
    - At least one column must be retained; the dialog rejects a
      zero-column selection.
    - Rename targets must be unique; duplicate names are rejected before
      Apply.

Assumptions:
    - The input DataFrame has at least one column.
    - Column names are string-convertible.

Failure Modes:
    - All columns deselected: Apply raises ValueError shown in dialog.
    - Duplicate rename target: Apply shows a QMessageBox warning.

Provenance:
    - This module emits no metadata; provenance is recorded by the calling
      DataCleaningWindow.
"""
from __future__ import annotations

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class ManageColumnsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Manage Columns"); self.setModal(True); self.resize(920, 680)
        self._widgets: dict[str, dict] = {}

        root = QVBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(12)
        hdr = QLabel("Select the observables to retain and rename them where required.")
        hdr.setObjectName("SectionHint"); hdr.setWordWrap(True); root.addWidget(hdr)

        ctrl = QHBoxLayout()
        for label, fn in [("Select all", lambda: self._toggle(True)),
                          ("Deselect all", lambda: self._toggle(False)),
                          ("Reset names", self._reset)]:
            b = QPushButton(label); b.clicked.connect(fn); ctrl.addWidget(b)
        ctrl.addStretch(1); root.addLayout(ctrl)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Keep","Original name","New name"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        root.addWidget(self.table, 1)
        self._populate()

        act = QHBoxLayout(); act.addStretch(1)
        for label, fn in [("Cancel", self.reject), ("Apply changes", self._apply)]:
            b = QPushButton(label); b.clicked.connect(fn); act.addWidget(b)
        root.addLayout(act)

    @property
    def df(self): return self.parent_window.current_dataframe()

    def _populate(self) -> None:
        self.table.setRowCount(0); self._widgets.clear()
        for row, col in enumerate(self.df.columns):
            self.table.insertRow(row)
            keep = QCheckBox(); keep.setChecked(True)
            name = QTableWidgetItem(str(col))
            name.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            new  = QLineEdit(str(col))
            self.table.setCellWidget(row, 0, keep)
            self.table.setItem(row, 1, name)
            self.table.setCellWidget(row, 2, new)
            self._widgets[str(col)] = {"keep": keep, "new_name": new}

    def _toggle(self, state: bool) -> None:
        for e in self._widgets.values(): e["keep"].setChecked(state)

    def _reset(self) -> None:
        for orig, e in self._widgets.items(): e["new_name"].setText(orig)

    def _apply(self) -> None:
        rename_map: dict[str, str] = {}; keep_final: list[str] = []
        for orig, e in self._widgets.items():
            if not e["keep"].isChecked(): continue
            new = e["new_name"].text().strip()
            if not new:
                QMessageBox.warning(self, "Invalid name", f"Column '{orig}' cannot be empty."); return
            if new != orig: rename_map[orig] = new
            keep_final.append(new)
        if not keep_final:
            QMessageBox.warning(self, "No selection", "Select at least one column."); return
        if len(keep_final) != len(set(keep_final)):
            QMessageBox.warning(self, "Duplicate names", "Final column names must be unique."); return
        df = self.df.copy()
        if rename_map: df = df.rename(columns=rename_map)
        df = df.loc[:, keep_final]
        self.parent_window.apply_cleaning_operation(
            operation="manage_columns", dataframe=df,
            summary=f"Columns updated · kept {len(keep_final)} · renamed {len(rename_map)}",
            params={"kept_count": len(keep_final), "renamed_count": len(rename_map),
                    "rename_map": rename_map, "columns_final": keep_final},
        )
        self.accept()
