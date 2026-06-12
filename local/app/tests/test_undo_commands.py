from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QUndoStack

from yj_studio.scene import LayerStore
from yj_studio.scene.layers import HorizonStickLayer, VolumeLayer
from yj_studio.scene.undo_commands import (
    AddLayerCommand,
    MergeLayersCommand,
    RemoveLayerCommand,
    RenameLayerCommand,
    SetColorCommand,
    SetLayerFieldCommand,
    SetOpacityCommand,
    SetVisibleCommand,
    SplitLayerCommand,
)
from yj_studio.ui.docks.layer_tree_dock import _merge_layers_factory, _split_layer


def _make_store_with_volume() -> tuple[LayerStore, str]:
    store = LayerStore()
    layer = VolumeLayer(name="seismic", volume_id="seismic", shape=(4, 5, 6))
    layer_id = store.add(layer)
    return store, layer_id


def test_set_visible_undo_redo(qapp) -> None:
    store, layer_id = _make_store_with_volume()
    stack = QUndoStack()
    stack.push(SetVisibleCommand(store, layer_id, False))
    assert store.get(layer_id).visible is False
    stack.undo()
    assert store.get(layer_id).visible is True
    stack.redo()
    assert store.get(layer_id).visible is False


def test_set_color_and_opacity_undo(qapp) -> None:
    store, layer_id = _make_store_with_volume()
    stack = QUndoStack()
    stack.push(SetColorCommand(store, layer_id, (0.1, 0.2, 0.3, 0.5)))
    stack.push(SetOpacityCommand(store, layer_id, 0.4))
    assert store.get(layer_id).color == (0.1, 0.2, 0.3, 0.5)
    assert store.get(layer_id).opacity == 0.4
    stack.undo()
    stack.undo()
    assert store.get(layer_id).opacity == 1.0
    assert store.get(layer_id).color == (1.0, 1.0, 1.0, 1.0)


def test_rename_command(qapp) -> None:
    store, layer_id = _make_store_with_volume()
    stack = QUndoStack()
    stack.push(RenameLayerCommand(store, layer_id, "renamed"))
    assert store.get(layer_id).name == "renamed"
    stack.undo()
    assert store.get(layer_id).name == "seismic"


def test_set_layer_field_merges_consecutive_edits(qapp) -> None:
    store, layer_id = _make_store_with_volume()
    stack = QUndoStack()
    stack.push(SetOpacityCommand(store, layer_id, 0.9))
    stack.push(SetOpacityCommand(store, layer_id, 0.5))
    stack.push(SetOpacityCommand(store, layer_id, 0.2))
    # Three pushes with the same field should merge into one undo step.
    assert stack.count() == 1
    assert store.get(layer_id).opacity == 0.2
    stack.undo()
    assert store.get(layer_id).opacity == 1.0


def test_add_remove_round_trip(qapp) -> None:
    store = LayerStore()
    stack = QUndoStack()
    layer = VolumeLayer(name="seismic", volume_id="seismic", shape=(2, 2, 2))
    add_cmd = AddLayerCommand(store, layer)
    stack.push(add_cmd)
    layer_id = add_cmd.layer_id
    assert layer_id is not None
    assert len(store) == 1
    stack.push(RemoveLayerCommand(store, layer_id))
    assert len(store) == 0
    stack.undo()
    assert len(store) == 1
    stack.undo()
    assert len(store) == 0


def test_volume_layer_roi_roundtrip_dict() -> None:
    layer = VolumeLayer(
        name="vol",
        volume_id="seismic",
        shape=(10, 20, 30),
        roi=(2, 5, 4, 8, 0, 9),
    )
    payload = layer.to_dict()
    restored = VolumeLayer.from_dict(payload)
    assert restored.roi == (2, 5, 4, 8, 0, 9)
    assert restored.effective_roi() == (2, 5, 4, 8, 0, 9)


def test_volume_layer_effective_roi_clamps_and_drops_full_box() -> None:
    layer = VolumeLayer(name="vol", volume_id="seismic", shape=(10, 20, 30))
    # No ROI -> None
    assert layer.effective_roi() is None
    # ROI that covers the whole volume collapses to None.
    layer.roi = (0, 9, 0, 19, 0, 29)
    assert layer.effective_roi() is None
    # Out-of-range ROI is clamped.
    layer.roi = (-5, 50, 3, 100, 0, 200)
    assert layer.effective_roi() == (0, 9, 3, 19, 0, 29)


def test_merge_horizon_sticks(qapp) -> None:
    store = LayerStore()
    a = HorizonStickLayer(name="A", points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
    b = HorizonStickLayer(name="B", points=np.array([[2, 2, 2]], dtype=np.float32))
    aid = store.add(a)
    bid = store.add(b)
    stack = QUndoStack()
    merger = _merge_layers_factory(store)
    stack.push(MergeLayersCommand(store, [aid, bid], merger))
    assert len(store) == 1
    merged = next(store.iter_layers())
    assert isinstance(merged, HorizonStickLayer)
    assert merged.points.shape == (3, 3)
    stack.undo()
    assert len(store) == 2


def test_split_horizon_stick(qapp) -> None:
    store = LayerStore()
    layer = HorizonStickLayer(
        name="A",
        points=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
    )
    layer_id = store.add(layer)
    stack = QUndoStack()
    stack.push(SplitLayerCommand(store, layer_id, _split_layer))
    assert len(store) == 3
    for part in store.iter_layers():
        assert isinstance(part, HorizonStickLayer)
        assert part.points.shape == (1, 3)
    stack.undo()
    assert len(store) == 1


def test_set_roi_field_command(qapp) -> None:
    store, layer_id = _make_store_with_volume()
    stack = QUndoStack()
    stack.push(SetLayerFieldCommand(store, layer_id, "roi", (1, 2, 0, 4, 0, 5)))
    assert store.get(layer_id).roi == (1, 2, 0, 4, 0, 5)
    stack.undo()
    assert store.get(layer_id).roi is None
