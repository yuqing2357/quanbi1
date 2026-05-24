from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import MeasurementLayer


class MeasurementDock(QDockWidget):
    """List measurement layers produced by the measurement tool."""

    def __init__(self, layer_store: LayerStore, parent: QWidget | None = None) -> None:
        super().__init__("Measurements", parent)
        self._layer_store = layer_store
        self._items: dict[str, QTreeWidgetItem] = {}
        self._updating = False

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget(content)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Name", "Values", "Units"])
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)
        self.setWidget(content)

        layer_store.layer_added.connect(self._on_layer_added)
        layer_store.layer_removed.connect(self._on_layer_removed)
        layer_store.layer_changed.connect(self._on_layer_changed)
        layer_store.selection_changed.connect(self._sync_selection)
        for layer in layer_store.iter_by_type(MeasurementLayer):
            self._on_layer_added(layer.id)

    def _on_layer_added(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, MeasurementLayer):
            return
        item = QTreeWidgetItem()
        item.setData(0, Qt.ItemDataRole.UserRole, layer_id)
        self._items[layer_id] = item
        self._set_item_values(item, layer)
        self.tree.addTopLevelItem(item)
        self.tree.resizeColumnToContents(0)

    def _on_layer_removed(self, layer_id: str) -> None:
        item = self._items.pop(layer_id, None)
        if item is None:
            return
        row = self.tree.indexOfTopLevelItem(item)
        if row >= 0:
            self.tree.takeTopLevelItem(row)

    def _on_layer_changed(self, layer_id: str, _field: str) -> None:
        item = self._items.get(layer_id)
        if item is None:
            self._on_layer_added(layer_id)
            return
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, MeasurementLayer):
            self._on_layer_removed(layer_id)
            return
        self._set_item_values(item, layer)

    def _set_item_values(self, item: QTreeWidgetItem, layer: MeasurementLayer) -> None:
        self._updating = True
        try:
            item.setText(0, layer.name)
            item.setText(1, _format_values(layer.values))
            item.setText(2, _format_units(layer.units))
            item.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
        finally:
            self._updating = False

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        current = self.tree.currentItem()
        if current is None:
            return
        layer_id = str(current.data(0, Qt.ItemDataRole.UserRole))
        if layer_id:
            self._layer_store.select([layer_id])

    def _sync_selection(self, layer_ids: list[str]) -> None:
        self._updating = True
        try:
            selection = set(layer_ids)
            for layer_id, item in self._items.items():
                item.setSelected(layer_id in selection)
        finally:
            self._updating = False


def _format_values(values: dict[str, float]) -> str:
    if not values:
        return ""
    return ", ".join(f"{key}={value:.3f}" for key, value in values.items())


def _format_units(units: dict[str, str]) -> str:
    if not units:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in units.items())
