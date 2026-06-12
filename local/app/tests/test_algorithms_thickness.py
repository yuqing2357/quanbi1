from __future__ import annotations

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.thickness import ThicknessAlgorithm
from yj_studio.scene.layers import HorizonLayer, MeasurementLayer


def _make_horizon_pair(constant_delta_samples: float = 5.0):
    nx, ny = 10, 8
    top_z = np.full((nx, ny), 100.0, dtype=np.float32)
    bot_z = top_z + constant_delta_samples
    top_z[0, 0] = np.nan  # invalid cell
    top = HorizonLayer(name="T_top", sample=top_z, mask=None)
    bottom = HorizonLayer(name="T_bottom", sample=bot_z, mask=None)
    return top, bottom


def test_thickness_in_process_run() -> None:
    runner = AlgorithmRunner()
    top, bottom = _make_horizon_pair(constant_delta_samples=5.0)
    result = runner.run_sync(
        ThicknessAlgorithm,
        params={"depth_step_m": 10.0, "name": "thk"},
        input_layers={"top": top, "bottom": bottom},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    measurement = result.output_layers[0]
    assert isinstance(measurement, MeasurementLayer)
    # delta=5 samples * 10 m/sample = 50 m thickness
    assert measurement.values["mean_m"] == 50.0
    assert measurement.values["min_m"] == 50.0
    assert measurement.values["max_m"] == 50.0
    # one cell is masked out -> coverage < 1
    assert measurement.values["coverage"] < 1.0
    assert measurement.geometry is not None and measurement.geometry.shape[1] == 5


def test_thickness_rejects_shape_mismatch() -> None:
    runner = AlgorithmRunner()
    top = HorizonLayer(name="T", sample=np.zeros((4, 4), dtype=np.float32))
    bot = HorizonLayer(name="B", sample=np.zeros((4, 5), dtype=np.float32))
    result = runner.run_sync(
        ThicknessAlgorithm,
        params={"depth_step_m": 10.0, "name": "thk"},
        input_layers={"top": top, "bottom": bot},
    )
    assert not result.ok
    # Error wording is localized; assert on the language-independent shapes.
    assert "(4, 4)" in (result.error or "")
    assert "(4, 5)" in (result.error or "")
