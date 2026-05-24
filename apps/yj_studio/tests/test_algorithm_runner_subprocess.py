"""End-to-end test: ``AlgorithmRunner.submit`` spawns a worker process.

This is the smoke test that proves the IPC stack survives a real
``multiprocessing.Process`` round trip. It's slower than the in-proc tests
because of the spawn cost, so we keep just one case here.
"""

from __future__ import annotations

import time

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.thickness import ThicknessAlgorithm
from yj_studio.scene import LayerStore
from yj_studio.scene.layers import HorizonLayer, MeasurementLayer


def _wait_for(condition, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_runner_submit_round_trip(qapp) -> None:
    layer_store = LayerStore()
    runner = AlgorithmRunner(layer_store=layer_store)

    top = HorizonLayer(name="top", sample=np.full((6, 6), 100.0, dtype=np.float32))
    bottom = HorizonLayer(name="bottom", sample=np.full((6, 6), 110.0, dtype=np.float32))

    finished: list = []
    errors: list = []
    task = runner.submit(
        ThicknessAlgorithm,
        params={"depth_step_m": 10.0, "name": "thk"},
        input_layers={"top": top, "bottom": bottom},
        auto_attach_outputs=True,
    )
    task.finished.connect(lambda layers, summary: finished.append((layers, summary)))
    task.errored.connect(lambda msg, tb: errors.append((msg, tb)))

    def done() -> bool:
        qapp.processEvents()
        return bool(finished) or bool(errors)

    assert _wait_for(done, timeout_s=60.0), "Algorithm did not finish within timeout"
    assert not errors, errors
    layers, summary = finished[0]
    assert len(layers) == 1
    assert isinstance(layers[0], MeasurementLayer)
    # 100 m thickness (delta=10 samples * 10 m/sample)
    assert layers[0].values["mean_m"] == 100.0
    assert "thk" in (summary or "") or "mean" in (summary or "").lower()
