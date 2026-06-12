from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from yj_studio.data.volume_store import VolumeStore
from yj_studio.io.readers.volume_npy import VolumeSpec
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import VolumeLayer, WellLayer, WellLogLayer
from yj_studio.services.well_section_service import build_well_section_data


def test_build_well_section_data_samples_seismic_and_logs() -> None:
    scratch = Path(__file__).resolve().parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{uuid4().hex}_well_section_cube.npy"
    volume = np.arange(6 * 5 * 4, dtype=np.float32).reshape((6, 5, 4))
    np.save(path, volume)

    store = VolumeStore()
    store.register(VolumeSpec(key="cube", path=path, label="cube", cmap="gray", filename=path.name))
    volume_layer = VolumeLayer(name="cube", volume_id="cube", shape=volume.shape)
    layer_store = LayerStore()
    layer_store.add(volume_layer)
    layer_store.add(
        WellLayer(name="W1", well_name="W1", head_position=(1.0, 1.0, 0.0), trajectory=np.zeros((2, 3)))
    )
    layer_store.add(
        WellLayer(name="W2", well_name="W2", head_position=(4.0, 1.0, 0.0), trajectory=np.zeros((2, 3)))
    )
    layer_store.add(
        WellLogLayer(
            name="W1 POR",
            well_name="W1",
            mode="por",
            samples=np.asarray([[1.0, 1.0, 1.0, 0.2]], dtype=np.float32),
        )
    )

    data = build_well_section_data(layer_store, store, volume_layer, ["W1", "W2"], mode="por")

    assert data.names == ("W1", "W2")
    assert data.seismic.shape[0] == 4
    assert data.seismic.shape[1] >= 2
    assert data.depths_m.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert len(data.wells[0].logs) == 1
