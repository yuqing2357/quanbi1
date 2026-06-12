from __future__ import annotations

from PyQt6.QtGui import QUndoStack

from yj_studio.data.volume_store import VolumeStore
from yj_studio.scene import LayerStore
from yj_studio.scene.layers import VolumeLayer
from yj_studio.ui.docks.property_dock import PropertyDock


def test_property_dock_reflects_current_layer(qapp) -> None:
    store = LayerStore()
    volume_store = VolumeStore()
    stack = QUndoStack()
    dock = PropertyDock(store, volume_store, stack)

    layer = VolumeLayer(
        name="seismic",
        volume_id="seismic",
        shape=(4, 5, 6),
        clim=(-0.5, 0.5),
        cmap="Petrel",
        opacity=0.8,
        visible=True,
    )
    layer_id = store.add(layer)
    store.select([layer_id])

    assert dock._name_edit.text() == "seismic"
    assert dock._visible_check.isChecked() is True
    assert dock._opacity_slider.value() == 80
    assert dock._cmap_combo.currentText() == "Petrel"
    assert dock._clim_min.value() == -0.5
    assert dock._clim_max.value() == 0.5


def test_property_dock_pushes_undo_commands(qapp) -> None:
    store = LayerStore()
    volume_store = VolumeStore()
    stack = QUndoStack()
    dock = PropertyDock(store, volume_store, stack)

    layer = VolumeLayer(name="seismic", volume_id="seismic", shape=(4, 5, 6))
    layer_id = store.add(layer)
    store.select([layer_id])

    # Simulate user typing a new name + pressing Enter.
    dock._name_edit.setText("renamed")
    dock._commit_name()
    assert stack.count() == 1
    assert store.get(layer_id).name == "renamed"

    # Toggle visibility through the property dock.
    dock._visible_check.setChecked(False)
    assert stack.count() == 2
    assert store.get(layer_id).visible is False
    stack.undo()
    assert store.get(layer_id).visible is True
