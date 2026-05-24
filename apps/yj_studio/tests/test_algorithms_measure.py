from __future__ import annotations

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.measure import MeasureAreaAlgorithm, MeasureDistanceAlgorithm
from yj_studio.scene.layers import HorizonStickLayer, MeasurementLayer, PolygonLayer


def test_measure_distance_on_polygon_path() -> None:
    runner = AlgorithmRunner()
    layer = PolygonLayer(
        name="path",
        vertices=np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], dtype=np.float32),
    )
    result = runner.run_sync(
        MeasureDistanceAlgorithm,
        params={"depth_step_m": 10.0, "name": "len"},
        input_layers={"path": layer},
    )
    assert result.ok, result.error
    out = result.output_layers[0]
    assert isinstance(out, MeasurementLayer)
    # 3 + 4 = 7 cells of planar distance; 0 vertical delta
    assert out.values["total_xy"] == 7.0
    assert out.values["total_3d_m"] == 7.0


def test_measure_distance_with_vertical_segment() -> None:
    runner = AlgorithmRunner()
    layer = HorizonStickLayer(
        name="stick",
        points=np.array([[0, 0, 0], [0, 0, 3]], dtype=np.float32),  # 3 sample drop
    )
    result = runner.run_sync(
        MeasureDistanceAlgorithm,
        params={"depth_step_m": 10.0, "name": "len"},
        input_layers={"path": layer},
    )
    assert result.ok
    out = result.output_layers[0]
    assert out.values["total_xy"] == 0.0
    # |dz|=3 samples * 10 m/sample = 30 m, no horizontal component
    assert out.values["total_3d_m"] == 30.0


def test_measure_area_shoelace() -> None:
    runner = AlgorithmRunner()
    layer = PolygonLayer(
        name="poly",
        vertices=np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], dtype=np.float32),
    )
    result = runner.run_sync(
        MeasureAreaAlgorithm,
        params={"depth_step_m": 10.0, "name": "area"},
        input_layers={"polygon": layer},
    )
    assert result.ok
    out = result.output_layers[0]
    assert out.values["area_xy"] == 12.0
    assert out.values["perimeter_xy"] == 14.0
