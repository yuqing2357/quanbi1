from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import VolumeLayer
from yj_studio.ui.docks.ai_dock import (
    _preferred_visible_volume_layer,
    _select_volume_layer_for_ai,
)


def test_ai_volume_selection_prefers_visible_lithology_for_reservoir_index():
    store = LayerStore()
    seismic = VolumeLayer(
        name="地震体数据",
        volume_id="seismic",
        shape=(1684, 1451, 1201),
        slice_indices={"inline": 842},
        visible=False,
    )
    lithology = VolumeLayer(
        name="岩性模型",
        volume_id="model_lithology",
        shape=(4452, 2796, 1443),
        slice_indices={"inline": 2226},
        visible=True,
    )
    store.add(seismic)
    store.add(lithology)

    selected = _select_volume_layer_for_ai(store, "inline", 2226)

    assert selected is lithology


def test_ai_volume_selection_rejects_out_of_bounds_seismic_index():
    store = LayerStore()
    seismic = VolumeLayer(
        name="地震体数据",
        volume_id="seismic",
        shape=(1684, 1451, 1201),
        slice_indices={"inline": 842},
        visible=True,
    )
    store.add(seismic)

    assert _select_volume_layer_for_ai(store, "inline", 2226) is None


def test_ai_volume_selection_prefers_visible_layer_even_when_index_fits_both():
    store = LayerStore()
    seismic = VolumeLayer(
        name="地震体数据",
        volume_id="seismic",
        shape=(1684, 1451, 1201),
        slice_indices={"inline": 100},
        visible=False,
    )
    lithology = VolumeLayer(
        name="岩性模型",
        volume_id="model_lithology",
        shape=(4452, 2796, 1443),
        slice_indices={"inline": 100},
        visible=True,
    )
    store.add(seismic)
    store.add(lithology)

    assert _select_volume_layer_for_ai(store, "inline", 100) is lithology


def test_ai_sync_prefers_selected_visible_volume():
    store = LayerStore()
    seismic = VolumeLayer(
        name="地震体数据",
        volume_id="seismic",
        shape=(1684, 1451, 1201),
        visible=True,
    )
    lithology = VolumeLayer(
        name="岩性模型",
        volume_id="model_lithology",
        shape=(4452, 2796, 1443),
        visible=True,
    )
    store.add(seismic)
    store.add(lithology)
    store.select([lithology.id])

    assert _preferred_visible_volume_layer(store) is lithology
