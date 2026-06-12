from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from yj_studio.data.arbitrary_section import resample_polyline_xy, sample_arbitrary_section
from yj_studio.ui.dialogs.arbitrary_section_dialog import WellMapPoint, parse_polyline_text, snap_point_to_well
from yj_studio.view.view_arbitrary_section import _matplotlib_cmap_name


def test_sample_arbitrary_section_samples_volume_along_polyline() -> None:
    x = np.arange(4, dtype=np.float32)[:, None, None]
    y = np.arange(3, dtype=np.float32)[None, :, None]
    z = np.arange(2, dtype=np.float32)[None, None, :]
    volume = x * 100.0 + y * 10.0 + z

    section = sample_arbitrary_section(
        volume,
        np.asarray([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32),
        z_start=0,
        z_end=1,
        horizontal_step=1.0,
        max_trace_count=10,
        order=1,
    )

    assert section.values.shape == (2, 3)
    np.testing.assert_allclose(section.distances, np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(section.values[0], np.asarray([10.0, 110.0, 210.0], dtype=np.float32))
    np.testing.assert_allclose(section.values[1], np.asarray([11.0, 111.0, 211.0], dtype=np.float32))


def test_resample_polyline_keeps_last_vertex() -> None:
    sampled, distances = resample_polyline_xy(
        np.asarray([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        horizontal_step=1.0,
        max_trace_count=10,
    )

    assert sampled.shape == (5, 2)
    np.testing.assert_allclose(distances[-1], 4.0)
    np.testing.assert_allclose(sampled[-1], np.asarray([2.0, 2.0], dtype=np.float32))


def test_parse_polyline_text_accepts_commas_tabs_and_semicolons() -> None:
    points = parse_polyline_text("1, 2\n3\t4\n5;6")

    np.testing.assert_allclose(
        points,
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_snap_point_to_well_uses_nearest_well_inside_radius() -> None:
    wells = (
        WellMapPoint(name="W1", inline=10.0, xline=20.0),
        WellMapPoint(name="W2", inline=100.0, xline=200.0),
    )

    point, well = snap_point_to_well((12.0, 21.0), wells, 5.0)

    assert point == (10.0, 20.0)
    assert well is wells[0]
    unsnapped, missed = snap_point_to_well((40.0, 40.0), wells, 5.0)
    assert unsnapped == (40.0, 40.0)
    assert missed is None


def test_arbitrary_section_view_maps_petrel_to_matplotlib_cmap() -> None:
    assert _matplotlib_cmap_name("Petrel") == "seismic"
    assert _matplotlib_cmap_name("not-a-real-cmap") == "seismic"


def test_main_window_open_arbitrary_section_adds_layer_and_view() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.io.readers.volume_npy import VolumeSpec
    from yj_studio.scene.layers import ArbitrarySectionLayer, VolumeLayer, WellLayer
    from yj_studio.ui.main_window import MainWindow

    scratch = Path(__file__).resolve().parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{uuid4().hex}_arbitrary_section_cube.npy"
    volume = np.arange(4 * 3 * 2, dtype=np.float32).reshape((4, 3, 2))
    np.save(path, volume)

    app = create_application([])
    window = MainWindow(auto_load=False, enable_3d=False)
    window.volume_store.register(VolumeSpec(key="cube", path=path, label="cube", cmap="Petrel", filename=path.name))
    volume_layer = VolumeLayer(
        name="cube",
        volume_id="cube",
        shape=volume.shape,
        cmap="Petrel",
        slice_indices={"z": 1},
    )
    window._active_volume_layer_id = window.layer_store.add(volume_layer)
    window.layer_store.add(WellLayer(name="W1", well_name="W1", head_position=(2.0, 1.0, 0.0)))

    window._open_arbitrary_section(
        np.asarray([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32),
        z_start=0,
        z_end=1,
        max_trace_count=10,
    )

    layers = list(window.layer_store.iter_by_type(ArbitrarySectionLayer))
    assert len(layers) == 1
    assert layers[0].image is not None
    assert layers[0].image.shape == (2, 3)
    assert window._views_area is not None
    assert window._views_area.count() == 2
    well_points = window._well_map_points()
    assert well_points == (WellMapPoint(name="W1", inline=2.0, xline=1.0),)
    assert window._topdown_slice(volume_layer) is not None
    window.close()
    app.quit()
