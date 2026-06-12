from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QDockWidget, QTreeWidget, QTreeWidgetItem, QWidget

from yj_studio.scene.layer import Layer
from yj_studio.scene.layer_store import LayerStore
from yj_studio.ui.text import layer_kind_label


class LayerFilterDock(QDockWidget):
    """Small filtered layer list used by type-specific manager docks."""

    layer_activated = pyqtSignal(str)

    def __init__(
        self,
        title: str,
        layer_store: LayerStore,
        accepts: Callable[[Layer], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self._layer_store = layer_store
        self._accepts = accepts
        self._items: dict[str, QTreeWidgetItem] = {}
        self._updating = False

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["名称", "类型", "详情"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_activated)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.setWidget(self.tree)

        layer_store.layer_added.connect(self._add_item)
        layer_store.layer_removed.connect(self._remove_item)
        layer_store.layer_changed.connect(self._refresh_item)
        layer_store.selection_changed.connect(self._sync_selection)

        for layer in layer_store.iter_layers():
            if self._accepts(layer):
                self._add_item(layer.id)

    def _add_item(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if not self._accepts(layer):
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
        if not self._accepts(layer):
            self._remove_item(layer_id)
            return
        item = self._items.get(layer_id)
        if item is None:
            self._add_item(layer_id)
            return
        self._set_item_values(item, layer)

    def _set_item_values(self, item: QTreeWidgetItem, layer: Layer) -> None:
        self._updating = True
        try:
            item.setText(0, layer.name)
            item.setText(1, layer_kind_label(layer.kind))
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
        layer_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        self.layer_activated.emit(layer_id)

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


def _layer_details(layer: Layer) -> str:
    path = layer.metadata.get("path")
    if path:
        return f"路径：{path}"
    if layer.kind == "well":
        return f"深度：{layer.metadata.get('z_top', '?')}..{layer.metadata.get('z_bottom', '?')}"
    if layer.kind == "well_log":
        return f"来源：{layer.metadata.get('source_path', '')}"
    return ""
