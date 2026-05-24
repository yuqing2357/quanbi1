from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Iterable

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QUndoStack
from PyQt6.QtWidgets import (
    QColorDialog,
    QDockWidget,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from yj_studio.scene.layer import Layer
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import WellLogLayer
from yj_studio.scene.undo_commands import (
    MergeLayersCommand,
    RemoveLayerCommand,
    RenameLayerCommand,
    SetColorCommand,
    SetOpacityCommand,
    SetVisibleCommand,
    SplitLayerCommand,
)


class LayerTreeDock(QDockWidget):
    """LayerTree with visibility, rename, right-click menu, all through QUndoStack."""

    def __init__(
        self,
        layer_store: LayerStore,
        parent: QWidget | None = None,
        *,
        undo_stack: QUndoStack | None = None,
    ) -> None:
        super().__init__("Layers", parent)
        self._layer_store = layer_store
        self._undo_stack = undo_stack
        self._items: dict[str, QTreeWidgetItem] = {}
        self._updating = False
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Name", "Type", "Details"])
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.setWidget(self.tree)

        layer_store.layer_added.connect(self._add_item)
        layer_store.layer_removed.connect(self._remove_item)
        layer_store.layer_changed.connect(self._refresh_item)
        layer_store.selection_changed.connect(self._sync_selection)

    # ------------------------------------------------------------------ items

    def _add_item(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if _is_hidden_layer(layer):
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
        if _is_hidden_layer(layer):
            self._remove_item(layer_id)
            return
        item = self._items.get(layer_id)
        if item is None:
            return
        self._set_item_values(item, layer)

    def _set_item_values(self, item: QTreeWidgetItem, layer: Layer) -> None:
        self._updating = True
        try:
            item.setText(0, layer.name)
            item.setText(1, layer.kind)
            item.setText(2, _layer_details(layer))
            item.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
        finally:
            self._updating = False

    # ------------------------------------------------------------------ user edits

    def _push(self, command) -> None:
        if self._undo_stack is None:
            command.redo()
        else:
            self._undo_stack.push(command)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating:
            return
        layer_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        try:
            layer = self._layer_store.get(layer_id)
        except KeyError:
            return
        if column != 0:
            return
        new_visible = item.checkState(0) == Qt.CheckState.Checked
        new_name = item.text(0)
        changes_visible = new_visible != layer.visible
        changes_name = bool(new_name) and new_name != layer.name
        if not (changes_visible or changes_name):
            return
        macro = self._undo_stack is not None and changes_visible and changes_name
        if macro:
            self._undo_stack.beginMacro(self.tr("Edit layer"))
        try:
            if changes_visible:
                self._push(SetVisibleCommand(self._layer_store, layer_id, new_visible))
            if changes_name:
                self._push(RenameLayerCommand(self._layer_store, layer_id, new_name))
        finally:
            if macro:
                self._undo_stack.endMacro()

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        layer_ids = self._selected_layer_ids()
        self._layer_store.select(layer_ids)

    def _sync_selection(self, layer_ids: list[str]) -> None:
        self._updating = True
        try:
            selection = set(layer_ids)
            for layer_id, item in self._items.items():
                item.setSelected(layer_id in selection)
        finally:
            self._updating = False

    def _selected_layer_ids(self) -> list[str]:
        return [
            str(item.data(0, Qt.ItemDataRole.UserRole))
            for item in self.tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole)
        ]

    # ------------------------------------------------------------------ context menu

    def _show_context_menu(self, point: QPoint) -> None:
        item = self.tree.itemAt(point)
        if item is None:
            return
        selected = self._selected_layer_ids()
        if not selected:
            return
        primary_id = str(item.data(0, Qt.ItemDataRole.UserRole))
        if primary_id not in selected:
            selected = [primary_id]
        try:
            primary = self._layer_store.get(primary_id)
        except KeyError:
            return

        menu = QMenu(self)
        rename_action = menu.addAction(self.tr("Rename..."))
        color_action = menu.addAction(self.tr("Color..."))
        opacity_action = menu.addAction(self.tr("Opacity..."))
        menu.addSeparator()
        merge_action = menu.addAction(self.tr("Merge selected"))
        merge_action.setEnabled(self._can_merge(selected))
        split_action = menu.addAction(self.tr("Split"))
        split_action.setEnabled(self._can_split(primary))
        menu.addSeparator()
        delete_action = menu.addAction(self.tr("Delete"))

        chosen = menu.exec(self.tree.viewport().mapToGlobal(point))
        if chosen is None:
            return
        if chosen is rename_action:
            self._action_rename(primary)
        elif chosen is color_action:
            self._action_color(primary)
        elif chosen is opacity_action:
            self._action_opacity(primary)
        elif chosen is merge_action:
            self._action_merge(selected)
        elif chosen is split_action:
            self._action_split(primary)
        elif chosen is delete_action:
            self._action_delete(selected)

    def _action_rename(self, layer: Layer) -> None:
        new_name, ok = QInputDialog.getText(
            self, self.tr("Rename layer"), self.tr("New name:"), text=layer.name
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == layer.name:
            return
        self._push(RenameLayerCommand(self._layer_store, layer.id, new_name))

    def _action_color(self, layer: Layer) -> None:
        r, g, b, a = (float(c) for c in layer.color)
        initial = QColor.fromRgbF(r, g, b, a)
        result = QColorDialog.getColor(
            initial,
            self,
            self.tr("Pick Color"),
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not result.isValid():
            return
        new_color = (
            float(result.redF()),
            float(result.greenF()),
            float(result.blueF()),
            float(result.alphaF()),
        )
        if new_color == tuple(float(c) for c in layer.color):
            return
        self._push(SetColorCommand(self._layer_store, layer.id, new_color))

    def _action_opacity(self, layer: Layer) -> None:
        pct, ok = QInputDialog.getInt(
            self,
            self.tr("Opacity"),
            self.tr("Opacity (0-100):"),
            value=int(round(float(layer.opacity) * 100)),
            min=0,
            max=100,
        )
        if not ok:
            return
        new_value = float(pct) / 100.0
        if abs(new_value - float(layer.opacity)) < 1e-6:
            return
        self._push(SetOpacityCommand(self._layer_store, layer.id, new_value))

    def _action_delete(self, layer_ids: Iterable[str]) -> None:
        ids = list(layer_ids)
        if not ids:
            return
        confirm = QMessageBox.question(
            self,
            self.tr("Delete layers"),
            self.tr("Remove {count} layer(s)? This can be undone.").format(count=len(ids)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if self._undo_stack is not None:
            self._undo_stack.beginMacro(self.tr("Delete layers"))
        try:
            for layer_id in ids:
                self._push(RemoveLayerCommand(self._layer_store, layer_id))
        finally:
            if self._undo_stack is not None:
                self._undo_stack.endMacro()

    def _can_merge(self, layer_ids: Iterable[str]) -> bool:
        ids = list(layer_ids)
        if len(ids) < 2:
            return False
        kinds: set[str] = set()
        for layer_id in ids:
            try:
                kinds.add(self._layer_store.get(layer_id).kind)
            except KeyError:
                return False
        if len(kinds) != 1:
            return False
        return next(iter(kinds)) in {"horizon_stick", "fault_stick", "polygon", "annotation"}

    def _action_merge(self, layer_ids: Iterable[str]) -> None:
        ids = list(layer_ids)
        if not self._can_merge(ids):
            return
        merger = _merge_layers_factory(self._layer_store)
        try:
            self._push(MergeLayersCommand(self._layer_store, ids, merger))
        except ValueError as exc:
            QMessageBox.warning(self, self.tr("Merge"), str(exc))

    def _can_split(self, layer: Layer) -> bool:
        return layer.kind in {"horizon_stick", "fault_stick", "polygon"}

    def _action_split(self, layer: Layer) -> None:
        if not self._can_split(layer):
            return
        try:
            self._push(SplitLayerCommand(self._layer_store, layer.id, _split_layer))
        except ValueError as exc:
            QMessageBox.information(self, self.tr("Split"), str(exc))


# ---------------------------------------------------------------------- helpers


def _layer_details(layer: Layer) -> str:
    if layer.kind == "volume":
        shape = getattr(layer, "shape", None)
        cmap = getattr(layer, "cmap", "")
        return f"shape={shape}, cmap={cmap}"
    if layer.kind in {"horizon", "fault_surface", "lith_body"}:
        return str(layer.metadata.get("path", ""))
    if layer.kind == "well":
        head = getattr(layer, "head_position", None)
        return f"head={head}"
    if layer.kind == "well_log":
        count = layer.metadata.get("sample_count", "")
        return f"{getattr(layer, 'well_name', '')}, samples={count}"
    if layer.kind == "arbitrary_section":
        traces = layer.metadata.get("trace_count", "")
        z_range = layer.metadata.get("z_range", "")
        return f"traces={traces}, z={z_range}"
    return ""


def _is_hidden_layer(layer: Layer) -> bool:
    return isinstance(layer, WellLogLayer)


# Per-kind configuration: which attribute carries the (N, ...) array of points
# that can be concatenated for merge / sliced one-cluster-per-layer for split.
_POINT_ATTR_BY_KIND: dict[str, str] = {
    "horizon_stick": "points",
    "fault_stick": "sticks",
    "polygon": "vertices",
}


def _merge_layers_factory(store: LayerStore):
    """Return a merger that concatenates point arrays of same-kind layers."""

    import numpy as np

    def merge(sources: list[Layer]) -> Layer:
        from uuid import uuid4

        first = sources[0]
        merged = replace(first)
        merged.id = str(uuid4())
        merged.name = f"{first.name} (merged)"

        if first.kind == "annotation":
            merged_items: list = []
            for src in sources:
                merged_items.extend(deepcopy(list(getattr(src, "items", []))))
            merged.items = merged_items
            return merged

        attr = _POINT_ATTR_BY_KIND.get(first.kind)
        if attr is not None:
            arrays = [getattr(src, attr) for src in sources if getattr(src, attr) is not None]
            if arrays:
                setattr(merged, attr, np.concatenate(arrays, axis=0))
        return merged

    return merge


def _split_layer(layer: Layer) -> list[Layer]:
    """Split a point-collection layer; one new layer per row of the point array."""

    from uuid import uuid4

    if layer.kind == "annotation":
        items = list(getattr(layer, "items", []))
        if len(items) < 2:
            raise ValueError("Layer has fewer than 2 annotations to split")
        parts: list[Layer] = []
        for index, item in enumerate(items, start=1):
            clone = replace(layer)
            clone.id = str(uuid4())
            clone.name = f"{layer.name} #{index}"
            clone.items = [deepcopy(item)]
            parts.append(clone)
        return parts

    attr = _POINT_ATTR_BY_KIND.get(layer.kind)
    if attr is None:
        raise ValueError(f"Layer kind '{layer.kind}' cannot be split")
    array = getattr(layer, attr)
    if array is None or len(array) < 2:
        raise ValueError("Layer has fewer than 2 points to split")
    parts = []
    for index, row in enumerate(array, start=1):
        clone = replace(layer)
        clone.id = str(uuid4())
        clone.name = f"{layer.name} #{index}"
        setattr(clone, attr, row[None, ...].copy())
        parts.append(clone)
    return parts
