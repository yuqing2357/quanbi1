from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from yj_studio.data.volume_store import VolumeStore
from yj_studio.io.readers.volume_npy import VolumeSpec
from yj_studio.scene.layers import HorizonLayer, VolumeLayer, WellLayer
from yj_studio.services.section_service import (
    extract_orthogonal_section,
    horizon_intersection,
    well_intersection,
)
from yj_studio.services.horizon_service import find_horizon_high_point, sample_volume_along_horizon
from yj_studio.view.view_2d_section import section_navigation_state


def test_extract_inline_section_uses_z_by_xline_orientation() -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
    scratch = Path(__file__).resolve().parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{uuid4().hex}_cube.npy"
    np.save(path, volume)
    store = VolumeStore()
    store.register(VolumeSpec(key="cube", path=path, label="cube", cmap="gray", filename=path.name))
    layer = VolumeLayer(name="cube", volume_id="cube", shape=volume.shape)

    section = extract_orthogonal_section(store, layer, "inline", 1)

    assert section.values.shape == (5, 4)
    np.testing.assert_array_equal(section.values, volume[1, :, :].T)
    assert section.x_label == "Xline"
    assert section.y_label == "Sample"


def test_section_navigation_state_shows_neighbor_inline_numbers() -> None:
    state = section_navigation_state("inline", 1479, (2959, 2201, 2826))

    assert state["previous_text"] == "← 上一个 Inline（1478）"
    assert state["current_text"] == "当前 Inline：1479 / 2958"
    assert state["next_text"] == "下一个 Inline（1480）→"
    assert state["previous_enabled"] is True
    assert state["next_enabled"] is True


def test_section_navigation_state_disables_xline_at_bounds() -> None:
    first = section_navigation_state("xline", 0, (10, 20, 30))
    last = section_navigation_state("xline", 19, (10, 20, 30))

    assert first["previous_enabled"] is False
    assert first["next_text"] == "下一个 Xline（1）→"
    assert last["previous_text"] == "← 上一个 Xline（18）"
    assert last["next_enabled"] is False


def test_horizon_intersection_inline_returns_one_row() -> None:
    sample = np.asarray([[10.0, 11.0, np.nan], [20.0, 21.0, 22.0]], dtype=np.float32)
    mask = np.asarray([[True, True, True], [True, False, True]])
    layer = HorizonLayer(name="T3", sample=sample, mask=mask, color=(1.0, 0.0, 0.0, 1.0))

    line = horizon_intersection(layer, "inline", 1)

    assert line is not None
    np.testing.assert_array_equal(line.x, np.asarray([0.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(line.y, np.asarray([20.0, 22.0], dtype=np.float32))


def test_find_horizon_high_point_uses_shallowest_valid_sample() -> None:
    sample = np.asarray([[8.0, 4.0], [2.0, 5.0]], dtype=np.float32)
    mask = np.asarray([[True, True], [False, True]])
    layer = HorizonLayer(name="T3", sample=sample, mask=mask)

    point = find_horizon_high_point(layer)

    assert point.inline == 0
    assert point.xline == 1
    assert point.sample == 4.0


def test_sample_volume_along_horizon_interpolates_z() -> None:
    volume = np.zeros((2, 2, 4), dtype=np.float32)
    for z in range(volume.shape[2]):
        volume[:, :, z] = float(z)
    scratch = Path(__file__).resolve().parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{uuid4().hex}_horizon_cube.npy"
    np.save(path, volume)
    store = VolumeStore()
    store.register(VolumeSpec(key="cube", path=path, label="cube", cmap="gray", filename=path.name))
    volume_layer = VolumeLayer(name="cube", volume_id="cube", shape=volume.shape)
    horizon = HorizonLayer(
        name="T3",
        sample=np.asarray([[0.0, 1.5], [2.0, 3.0]], dtype=np.float32),
        mask=np.asarray([[True, True], [True, False]]),
    )

    data = sample_volume_along_horizon(store, volume_layer, horizon)

    np.testing.assert_allclose(data.values[0, 0], 0.0)
    np.testing.assert_allclose(data.values[0, 1], 1.5)
    np.testing.assert_allclose(data.values[1, 0], 2.0)
    assert np.isnan(data.values[1, 1])


def test_well_intersection_xline_projects_to_inline_sample() -> None:
    trajectory = np.asarray([[3.0, 8.0, 100.0], [3.0, 8.0, 120.0]], dtype=np.float32)
    layer = WellLayer(
        name="W1",
        well_name="W1",
        trajectory=trajectory,
        head_position=(3.0, 8.0, 100.0),
    )

    line = well_intersection(layer, "xline", 8)

    assert line is not None
    np.testing.assert_array_equal(line.x, np.asarray([3.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(line.y, np.asarray([100.0, 120.0], dtype=np.float32))
