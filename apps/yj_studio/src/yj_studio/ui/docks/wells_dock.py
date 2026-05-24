from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import WellLayer, WellLogLayer


class WellsDock(QDockWidget):
    """Single entry point for well trajectories and their display mode."""

    layer_activated = pyqtSignal(str)
    display_mode_changed = pyqtSignal(str)

    def __init__(self, layer_store: LayerStore, parent: QWidget | None = None) -> None:
        super().__init__("Wells", parent)
        self._layer_store = layer_store
        self._items: dict[str, QTreeWidgetItem] = {}
        self._updating = False

        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Display", panel))
        self.display_mode_box = QComboBox(panel)
        self.display_mode_box.addItem("Well only", "none")
        self.display_mode_box.addItem("Lithology", "lith")
        self.display_mode_box.addItem("Porosity", "por")
        self.display_mode_box.addItem("Permeability", "perm")
        self.display_mode_box.currentIndexChanged.connect(self._on_display_mode_changed)
        mode_row.addWidget(self.display_mode_box, 1)
        layout.addLayout(mode_row)

        self.tree = QTreeWidget(panel)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Name", "Type", "Details"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_activated)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree, 1)
        self.setWidget(panel)

        layer_store.layer_added.connect(self._add_item)
        layer_store.layer_removed.connect(self._remove_item)
        layer_store.layer_changed.connect(self._refresh_item)
        layer_store.selection_changed.connect(self._sync_selection)

        for layer in layer_store.iter_layers():
            if isinstance(layer, WellLayer):
                self._add_item(layer.id)

    @property
    def display_mode(self) -> str:
        return str(self.display_mode_box.currentData() or "none")

    def set_display_mode(self, mode: str) -> None:
        index = self.display_mode_box.findData(mode)
        if index < 0:
            return
        was_blocked = self.display_mode_box.blockSignals(True)
        try:
            self.display_mode_box.setCurrentIndex(index)
        finally:
            self.display_mode_box.blockSignals(was_blocked)

    def _add_item(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, WellLayer):
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
        if not isinstance(layer, WellLayer):
            self._remove_item(layer_id)
            return
        item = self._items.get(layer_id)
        if item is None:
            self._add_item(layer_id)
            return
        self._set_item_values(item, layer)

    def _set_item_values(self, item: QTreeWidgetItem, layer: WellLayer) -> None:
        self._updating = True
        try:
            item.setText(0, layer.name)
            item.setText(1, layer.kind)
            item.setText(2, _well_details(layer))
            item.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
        finally:
            self._updating = False

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        layer_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        layer = self._layer_store.get(layer_id)
        if not isinstance(layer, WellLayer):
            return
        visible = item.checkState(0) == Qt.CheckState.Checked
        updates: dict[str, object] = {}
        if visible != layer.visible:
            updates["visible"] = visible
        name = item.text(0)
        if name != layer.name:
            updates["name"] = name
        if updates:
            self._layer_store.update(layer_id, **updates)
        if not visible:
            well_name = layer.well_name or layer.name
            for log_layer in list(self._layer_store.iter_by_type(WellLogLayer)):
                if log_layer.well_name == well_name and log_layer.visible:
                    self._layer_store.update(log_layer.id, visible=False)

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

    def _on_display_mode_changed(self) -> None:
        self.display_mode_changed.emit(self.display_mode)


def _well_details(layer: WellLayer) -> str:
    if layer.head_position is not None:
        x, y, z = layer.head_position
        return f"head=({x:.1f}, {y:.1f}, {z:.1f})"
    return f"depth={layer.metadata.get('z_top', '?')}..{layer.metadata.get('z_bottom', '?')}"
