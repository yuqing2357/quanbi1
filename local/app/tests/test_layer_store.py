from __future__ import annotations

from yj_studio.scene import LayerStore
from yj_studio.scene.layers import VolumeLayer


def test_layer_store_add_update_select_remove() -> None:
    store = LayerStore()
    events: list[tuple[str, object]] = []
    store.layer_added.connect(lambda layer_id: events.append(("added", layer_id)))
    store.layer_changed.connect(lambda layer_id, field: events.append(("changed", field)))
    store.selection_changed.connect(lambda layer_ids: events.append(("selection", list(layer_ids))))
    store.layer_removed.connect(lambda layer_id: events.append(("removed", layer_id)))

    layer = VolumeLayer(name="seismic", volume_id="seismic", shape=(2, 3, 4))
    layer_id = store.add(layer)
    store.update(layer_id, opacity=0.5, visible=False)
    store.select([layer_id])
    removed = store.remove(layer_id)

    assert removed is layer
    assert len(store) == 0
    assert events[0] == ("added", layer_id)
    assert ("changed", "opacity") in events
    assert ("changed", "visible") in events
    assert ("selection", [layer_id]) in events
    assert events[-1] == ("removed", layer_id)


def test_volume_layer_serialization_summary() -> None:
    layer = VolumeLayer(
        name="seismic",
        volume_id="seismic",
        shape=(1684, 976, 654),
        clim=(-1.0, 1.0),
        cmap="Petrel",
    )

    payload = layer.to_dict()
    restored = VolumeLayer.from_dict(payload)

    assert payload["kind"] == "volume"
    assert restored.name == "seismic"
    assert restored.shape == (1684, 976, 654)
    assert restored.clim == (-1.0, 1.0)

