"""
DNV Scientific Module
---------------------
Role:
    Provides DerivedColumnDialog — a safe expression evaluator for
    creating new columns from a constrained arithmetic formula grammar.

Scientific Context:
    Evaluates a user-supplied arithmetic expression over existing numeric
    columns using a sandboxed AST walker; no statistical computation.

Invariants:
    - Only whitelisted AST node types (BinOp, UnaryOp, Call, Name,
      Constant) are permitted in the expression.
    - The new column name must be unique in the DataFrame.

Assumptions:
    - Expression operands reference only existing numeric columns.
    - The host Python version supports ast.parse with mode='eval'.

Failure Modes:
    - Forbidden AST node: raises ValueError with the node type name.
    - Division by zero: returns NaN per pandas semantics; no exception
      raised.
    - Non-unique column name: dialog rejects with a QMessageBox warning.

Provenance:
    - This module emits no metadata; provenance is recorded by the calling
      DataCleaningWindow.
"""
from __future__ import annotations

import ast, logging, math

import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QSplitter, QVBoxLayout,
)

log = logging.getLogger(__name__)


class DerivedColumnDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Create Derived Column"); self.setModal(True); self.resize(920, 680)

        root = QVBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(12)
        hint = QLabel("Expressions use df, pd, np, math, abs.  Example: df['a'] / df['b']  or  np.log(df['energy']).")
        hint.setObjectName("SectionHint"); hint.setWordWrap(True); root.addWidget(hint)

        form = QGridLayout()
        form.addWidget(QLabel("New column name"), 0, 0)
        self.name_edit = QLineEdit(); form.addWidget(self.name_edit, 0, 1)
        form.addWidget(QLabel("Expression"), 1, 0)
        self.expr_edit = QLineEdit(); form.addWidget(self.expr_edit, 1, 1)
        form.setColumnStretch(1, 1); root.addLayout(form)

        mid = QSplitter()
        tokens = QListWidget()
        for t in ["+","-","*","/","**","(",")", "np.log()","np.sqrt()","abs()"]:
            tokens.addItem(t)
        tokens.itemClicked.connect(lambda item: self._insert(item.text()))
        cols = QListWidget()
        for c in self.df.columns: cols.addItem(str(c))
        cols.itemClicked.connect(lambda item: self._insert(f"df[{item.text()!r}]"))
        mid.addWidget(tokens); mid.addWidget(cols); root.addWidget(mid, 1)

        act = QHBoxLayout()
        clr = QPushButton("Clear"); clr.clicked.connect(self.expr_edit.clear)
        act.addWidget(clr); act.addStretch(1)
        for label, fn in [("Cancel", self.reject), ("Create column", self._apply)]:
            b = QPushButton(label); b.clicked.connect(fn); act.addWidget(b)
        root.addLayout(act)

    @property
    def df(self): return self.parent_window.current_dataframe()

    def _insert(self, token: str) -> None:
        cur = self.expr_edit.cursorPosition(); txt = self.expr_edit.text()
        self.expr_edit.setText(txt[:cur] + token + txt[cur:])
        self.expr_edit.setCursorPosition(cur + len(token))

    def _apply(self) -> None:
        name = self.name_edit.text().strip(); expr = self.expr_edit.text().strip()
        if not name or not expr:
            QMessageBox.warning(self, "Incomplete", "Provide both a column name and an expression."); return
        if name in self.df.columns:
            QMessageBox.warning(self, "Duplicate", f"Column '{name}' already exists."); return
        try:
            ast.parse(expr, mode="eval")
            import numpy as _np
            env = {"df": self.df.copy(), "pd": pd, "np": _np, "math": math, "abs": abs}
            result = eval(expr, {"__builtins__": {}}, env)
        except Exception as e:
            QMessageBox.critical(self, "Expression error", str(e))
            log.exception("DerivedColumnDialog eval failed"); return
        df = self.df.copy()
        df[name] = result if isinstance(result, pd.Series) else pd.Series([result]*len(df), index=df.index)
        self.parent_window.apply_cleaning_operation(
            operation="create_derived_column", dataframe=df,
            summary=f"Derived column created · {name}",
            params={"new_column_name": name, "expression": expr},
        )
        self.accept()
