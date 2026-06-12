from __future__ import annotations

import numpy as np

from yj_studio.algorithms.serialization import (
    layer_to_payload,
    layers_to_payloads,
    payload_to_layer,
    payloads_to_layers,
)
from yj_studio.scene.layers import (
    HorizonLayer,
    HorizonStickLayer,
    MeasurementLayer,
    VolumeLayer,
)


def test_round_trip_volume_layer() -> None:
    layer = VolumeLayer(
        name="seismic",
        volume_id="seismic",
        shape=(4, 5, 6),
        clim=(-1.0, 1.0),
        cmap="Petrel",
        roi=(0, 3, 0, 4, 0, 5),
    )
    payload = layer_to_payload(layer)
    restored = payload_to_layer(payload)
    assert isinstance(restored, VolumeLayer)
    assert restored.shape == (4, 5, 6)
    assert restored.cmap == "Petrel"
    assert restored.roi == (0, 3, 0, 4, 0, 5)


def test_round_trip_horizon_layer_preserves_arrays() -> None:
    sample = np.linspace(50.0, 80.0, num=20, dtype=np.float32).reshape(4, 5)
    layer = HorizonLayer(name="T", sample=sample, mask=None)
    payload = layer_to_payload(layer)
    restored = payload_to_layer(payload)
    assert isinstance(restored, HorizonLayer)
    np.testing.assert_array_equal(restored.sample, sample)


def test_round_trip_measurement_geometry() -> None:
    geom = np.array(
        [[0, 0, 50.0, 60.0, 10.0], [1, 1, 55.0, 60.0, 5.0]], dtype=np.float32
    )
    layer = MeasurementLayer(
        name="thk",
        geometry=geom,
        values={"mean_m": 7.5},
        units={"mean_m": "m"},
    )
    payload = layer_to_payload(layer)
    restored = payload_to_layer(payload)
    assert isinstance(restored, MeasurementLayer)
    np.testing.assert_array_equal(restored.geometry, geom)
    assert restored.values["mean_m"] == 7.5
    assert restored.units["mean_m"] == "m"


def test_layers_to_payloads_round_trip() -> None:
    stick = HorizonStickLayer(
        name="stick", points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    )
    payloads = layers_to_payloads({"seed": stick})
    layers = payloads_to_layers(payloads)
    assert isinstance(layers["seed"], HorizonStickLayer)
    np.testing.assert_array_equal(layers["seed"].points, stick.points)
