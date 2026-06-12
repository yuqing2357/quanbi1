"""Algorithm browser + parameter form + runner controls.

Layout (top-down):

  ┌ Category tree (Horizon / Fault / Reservoir / Trap / Measure / AI) ┐
  ├ Description ─────────────────────────────────────────────────────┤
  ├ SchemaForm (params + layer pickers) ─────────────────────────────┤
  ├ [Run] [Cancel] ──────────────────────────────────────────────────┤
  ├ Progress bar + message ──────────────────────────────────────────┤
  └ Last result summary ─────────────────────────────────────────────┘

The dock is driven by ``algorithms.registry.registry`` — every algorithm that
decorated itself with ``@register_algorithm`` shows up automatically. Run
results are auto-added to the LayerStore by ``AlgorithmTask`` (see runner.py).
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtGui import QUndoStack

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.registry import AlgorithmRegistry
from yj_studio.algorithms.runner import AlgorithmRunner, AlgorithmTask
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.undo_commands import AddLayerCommand
from yj_studio.ui.widgets.schema_form import SchemaForm
from yj_studio.ui.text import algorithm_category_label, parameter_role_label

logger = logging.getLogger(__name__)


class AlgorithmDock(QDockWidget):
    def __init__(
        self,
        layer_store: LayerStore,
        registry: AlgorithmRegistry,
        runner: AlgorithmRunner,
        parent: QWidget | None = None,
        *,
        undo_stack: QUndoStack | None = None,
    ) -> None:
        super().__init__("算法", parent)
        self._layer_store = layer_store
        self._registry = registry
        self._runner = runner
        self._undo_stack = undo_stack
        self._current_algorithm: type[Algorithm] | None = None
        self._current_task: AlgorithmTask | None = None

        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Vertical, body)

        self._tree = QTreeWidget(splitter)
        self._tree.setHeaderLabels(["算法"])
        self._tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        splitter.addWidget(self._tree)

        details = QWidget(splitter)
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self._description_label = QLabel("", details)
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet("color: #666;")
        details_layout.addWidget(self._description_label)

        self._form = SchemaForm(layer_store, details)
        details_layout.addWidget(self._form, 1)

        controls = QHBoxLayout()
        self._run_button = QPushButton("运行", details)
        self._run_button.clicked.connect(self._on_run_clicked)
        self._cancel_button = QPushButton("取消", details)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        controls.addWidget(self._run_button)
        controls.addWidget(self._cancel_button)
        details_layout.addLayout(controls)

        self._progress = QProgressBar(details)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        details_layout.addWidget(self._progress)

        self._summary_label = QLabel("", details)
        self._summary_label.setWordWrap(True)
        details_layout.addWidget(self._summary_label)

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter)

        self.setWidget(body)

        # Populate the algorithm tree from the registry the first time.
        self._rebuild_tree()
        layer_store.layer_added.connect(lambda _id: self._form.refresh_layer_choices())
        layer_store.layer_removed.connect(lambda _id: self._form.refresh_layer_choices())
        layer_store.layer_changed.connect(lambda _id, _f: self._form.refresh_layer_choices())

    # ---------------------------------------------------------- tree

    def _rebuild_tree(self) -> None:
        self._tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for cls in sorted(self._registry.iter_algorithms(), key=lambda c: (c.category, c.label)):
            parent = groups.get(cls.category)
            if parent is None:
                parent = QTreeWidgetItem([algorithm_category_label(cls.category)])
                self._tree.addTopLevelItem(parent)
                groups[cls.category] = parent
            item = QTreeWidgetItem([cls.label])
            item.setData(0, Qt.ItemDataRole.UserRole, cls.id)
            item.setToolTip(0, cls.description)
            parent.addChild(item)
        self._tree.expandAll()

    def _on_tree_selection_changed(self) -> None:
        items = self._tree.selectedItems()
        algorithm_cls: type[Algorithm] | None = None
        if items:
            algorithm_id = items[0].data(0, Qt.ItemDataRole.UserRole)
            if algorithm_id:
                try:
                    algorithm_cls = self._registry.get(str(algorithm_id))
                except KeyError:
                    algorithm_cls = None
        self._current_algorithm = algorithm_cls
        if algorithm_cls is None:
            self._description_label.setText("")
            self._form.set_algorithm(None, None)
            self._run_button.setEnabled(False)
            return
        self._description_label.setText(algorithm_cls.description)
        self._form.set_algorithm(algorithm_cls.input_schema, dict(algorithm_cls.layer_inputs))
        self._run_button.setEnabled(True)
        self._summary_label.setText("")
        self._progress.setValue(0)

    # ---------------------------------------------------------- run / cancel

    def _on_run_clicked(self) -> None:
        if self._current_algorithm is None:
            return
        error = self._form.validate()
        if error:
            QMessageBox.warning(self, "算法参数", error)
            return
        collected = self._form.collect()
        layers = {}
        for role, layer_id in collected["layers"].items():
            try:
                layers[role] = self._layer_store.get(layer_id)
            except KeyError:
                QMessageBox.warning(
                    self,
                    "算法参数",
                    f"缺少角色“{parameter_role_label(role)}”对应的图层",
                )
                return

        self._summary_label.setText("")
        self._progress.setValue(0)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(self._current_algorithm.supports_cancel)

        task = self._runner.submit(self._current_algorithm, collected["params"], layers)
        self._current_task = task
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_finished)
        task.errored.connect(self._on_errored)
        task.cancelled.connect(self._on_cancelled)

    def _on_cancel_clicked(self) -> None:
        if self._current_task is not None:
            self._current_task.cancel()

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        if message:
            self._summary_label.setText(message)

    def _on_finished(self, output_layers, summary: str) -> None:
        self._reset_buttons()
        self._progress.setValue(100)
        if output_layers:
            if self._undo_stack is not None:
                self._undo_stack.beginMacro(f"运行 {self._algorithm_label_for_undo()}")
                try:
                    for layer in output_layers:
                        self._undo_stack.push(AddLayerCommand(self._layer_store, layer))
                finally:
                    self._undo_stack.endMacro()
            else:
                for layer in output_layers:
                    self._layer_store.add(layer)
        self._summary_label.setText(summary or f"已生成 {len(output_layers)} 个图层。")

    def _algorithm_label_for_undo(self) -> str:
        return self._current_algorithm.label if self._current_algorithm else "算法"

    def _on_errored(self, message: str, traceback_text: str) -> None:
        self._reset_buttons()
        self._summary_label.setText(f"错误：{message}")
        logger.error("Algorithm error: %s\n%s", message, traceback_text)

    def _on_cancelled(self) -> None:
        self._reset_buttons()
        self._summary_label.setText("已取消")

    def _reset_buttons(self) -> None:
        self._run_button.setEnabled(self._current_algorithm is not None)
        self._cancel_button.setEnabled(False)
        self._current_task = None
