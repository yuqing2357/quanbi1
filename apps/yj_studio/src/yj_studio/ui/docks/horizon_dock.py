from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from yj_studio.scene.layer import Layer
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import HorizonLayer


class HorizonDock(QDockWidget):
    """List horizon layers and horizon-specific view actions."""

    layer_activated = pyqtSignal(str)
    structure_map_requested = pyqtSignal(str)
    high_point_requested = pyqtSignal(str)
    along_horizon_requested = pyqtSignal(str)

    def __init__(self, layer_store: LayerStore, parent: QWidget | None = None) -> None:
        super().__init__("Horizons", parent)
        self._layer_store = layer_store
        self._items: dict[str, QTreeWidgetItem] = {}
        self._updating = False

        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.tree = QTreeWidget(panel)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Name", "Type", "Details"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_activated)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree, 1)

        structure_button = QPushButton("Structure Map", panel)
        structure_button.clicked.connect(self._emit_structure_map_requested)
        layout.addWidget(structure_button)

        high_button = QPushButton("Jump High Point", panel)
        high_button.clicked.connect(self._emit_high_point_requested)
        layout.addWidget(high_button)

        along_button = QPushButton("Along Horizon", panel)
        along_button.clicked.connect(self._emit_along_horizon_requested)
        layout.addWidget(along_button)
        self.setWidget(panel)

        layer_store.layer_added.connect(self._add_item)
        layer_store.layer_removed.connect(self._remove_item)
        layer_store.layer_changed.connect(self._refresh_item)
        layer_store.selection_changed.connect(self._sync_selection)

        for layer in layer_store.iter_layers():
            if isinstance(layer, HorizonLayer):
                self._add_item(layer.id)

    def _add_item(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, HorizonLayer):
            return
        item = QTreeWidgetItem()
        item.setData(0, Qt.ItemDataRole.UserRole, layer_id)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
        self._items[layer_id] = item
        self._set_item_values(item, layer)
        self.tree.addTopLevelItem(item)
        self.tree.resizeColumnToContents(0)

    def _remove_item(self, layer_id: str) -> None:
        item = self._items.pop(layer_id, None)
        if item is None:
            return
        index = self.tree.indexOfTopLevelItem(item)
        if index >= 0:
            self.tree.takeTopLevelItem(index)

    def _refresh_item(self, layer_id: str, _field: str) -> None:
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, HorizonLayer):
            self._remove_item(layer_id)
            return
        item = self._items.get(layer_id)
        if item is None:
            self._add_item(layer_id)
            return
        self._set_item_values(item, layer)

    def _set_item_values(self, item: QTreeWidgetItem, layer: HorizonLayer) -> None:
        self._updating = True
        try:
            item.setText(0, layer.name)
            item.setText(1, layer.kind)
            item.setText(2, _layer_details(layer))
            item.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
        finally:
            self._updating = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        layer_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        layer = self._layer_store.get(layer_id)
        updates: dict[str, object] = {}
        visible = item.checkState(0) == Qt.CheckState.Checked
        if visible != layer.visible:
            updates["visible"] = visible
        name = item.text(0)
        if name != layer.name:
            updates["name"] = name
        if updates:
            self._layer_store.update(layer_id, **updates)

    def _on_item_activated(self, item: QTreeWidgetItem) -> None:
        self.layer_activated.emit(str(item.data(0, Qt.ItemDataRole.UserRole)))

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        layer_ids = [
            str(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole)
        ]
        self._layer_store.select(layer_ids)

    def _sync_selection(self, layer_ids: list[str]) -> None:
        self._updating = True
        try:
            selection = set(layer_ids)
            for layer_id, item in self._items.items():
                item.setSelected(layer_id in selection)
        finally:
            self._updating = False

    def _current_horizon_id(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            selected = self.tree.selectedItems()
            item = selected[0] if selected else None
        if item is None:
            return None
        return str(item.data(0, Qt.ItemDataRole.UserRole))

    def _emit_structure_map_requested(self) -> None:
        layer_id = self._current_horizon_id()
        if layer_id:
            self.structure_map_requested.emit(layer_id)

    def _emit_high_point_requested(self) -> None:
        layer_id = self._current_horizon_id()
        if layer_id:
            self.high_point_requested.emit(layer_id)

    def _emit_along_horizon_requested(self) -> None:
        layer_id = self._current_horizon_id()
        if layer_id:
            self.along_horizon_requested.emit(layer_id)


def _layer_details(layer: Layer) -> str:
    path = layer.metadata.get("path")
    if path:
        return str(path)
    return ""
