from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import WellLayer
from yj_studio.ui.text import well_display_mode_label


class WellSectionDock(QDockWidget):
    """Select wells and request an in-application connected-well section."""

    build_requested = pyqtSignal(list, str)

    def __init__(self, layer_store: LayerStore, parent: QWidget | None = None) -> None:
        super().__init__("井剖面", parent)
        self._layer_store = layer_store
        self._items: dict[str, QListWidgetItem] = {}

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel("选择井", content))

        self.list_widget = QListWidget(content)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list_widget)

        self.mode_box = QComboBox(content)
        self.mode_box.addItem(well_display_mode_label("lith"), "lith")
        self.mode_box.addItem(well_display_mode_label("por"), "por")
        layout.addWidget(self.mode_box)

        build_button = QPushButton("打开剖面", content)
        build_button.clicked.connect(self._emit_build_requested)
        layout.addWidget(build_button)
        self.setWidget(content)

        layer_store.layer_added.connect(self._add_layer)
        layer_store.layer_removed.connect(self._remove_layer)
        layer_store.layer_changed.connect(self._refresh_layer)
        for layer in layer_store.iter_layers():
            if isinstance(layer, WellLayer):
                self._add_layer(layer.id)

    def selected_wells(self) -> list[str]:
        names: list[str] = []
        for item in self.list_widget.selectedItems():
            layer_id = str(item.data(Qt.ItemDataRole.UserRole))
            layer = self._layer_store.get(layer_id)
            if isinstance(layer, WellLayer):
                names.append(layer.well_name or layer.name)
        return names

    def _add_layer(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, WellLayer):
            return
        item = QListWidgetItem(layer.well_name or layer.name)
        item.setData(Qt.ItemDataRole.UserRole, layer_id)
        self._items[layer_id] = item
        self.list_widget.addItem(item)

    def _remove_layer(self, layer_id: str) -> None:
        item = self._items.pop(layer_id, None)
        if item is None:
            return
        row = self.list_widget.row(item)
        if row >= 0:
            self.list_widget.takeItem(row)

    def _refresh_layer(self, layer_id: str, _field: str) -> None:
        item = self._items.get(layer_id)
        layer = self._layer_store.get(layer_id)
        if item is None:
            if isinstance(layer, WellLayer):
                self._add_layer(layer_id)
            return
        if not isinstance(layer, WellLayer):
            self._remove_layer(layer_id)
            return
        item.setText(layer.well_name or layer.name)

    def _emit_build_requested(self) -> None:
        self.build_requested.emit(self.selected_wells(), str(self.mode_box.currentData()))
