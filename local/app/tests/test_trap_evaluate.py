from __future__ import annotations

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.trap_evaluate import TrapEvaluateAlgorithm, rasterize_closure, volumetrics
from yj_studio.scene.layers import HorizonLayer, MeasurementLayer, TrapLayer, VolumeLayer


class _VolumeStore:
    def __init__(self, volumes: dict[str, np.ndarray]) -> None:
        self._volumes = volumes

    def get_volume(self, volume_id: str) -> np.ndarray:
        return self._volumes[volume_id]


def _trap() -> TrapLayer:
    boundary = np.asarray(
        [
            [1.0, 1.0, 4.0],
            [3.0, 1.0, 4.0],
            [3.0, 3.0, 4.0],
            [1.0, 3.0, 4.0],
            [1.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    return TrapLayer(name="Trap-1", boundary=boundary, score=0.8, attributes={"spill_level": 4.0})


def test_rasterize_closure_covers_rectangle_cells() -> None:
    rows, cols = rasterize_closure(_trap().boundary[:, :2], (6, 6))

    assert rows.size == 9
    assert set(zip(rows.tolist(), cols.tolist(), strict=False)) == {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    }


def test_volumetrics_matches_hand_calculation() -> None:
    top = np.zeros((6, 6), dtype=np.float32)
    bottom = np.full((6, 6), 4.0, dtype=np.float32)
    porosity = np.full((6, 6, 6), 0.2, dtype=np.float32)
    rows, cols = rasterize_closure(_trap().boundary[:, :2], top.shape)
    params = TrapEvaluateAlgorithm.input_schema(
        net_to_gross=0.5,
        water_saturation=0.25,
        cell_area_m2=10.0,
        z_step_m=2.0,
        formation_volume_factor=1.2,
    )

    values = volumetrics(rows, cols, top, bottom, porosity, params)

    assert values["cell_count"] == 9.0
    assert values["GRV"] == 9 * 4 * 2 * 10
    assert values["HCPV"] == values["GRV"] * 0.5 * 0.2 * 0.75
    assert values["STOIIP"] == values["HCPV"] / 1.2


def test_trap_evaluate_algorithm_outputs_measurement_layer() -> None:
    top = HorizonLayer(name="Top", sample=np.zeros((6, 6), dtype=np.float32))
    bottom = HorizonLayer(name="Bottom", sample=np.full((6, 6), 5.0, dtype=np.float32))
    porosity = np.full((6, 6, 8), 0.18, dtype=np.float32)
    por_layer = VolumeLayer(name="poro", volume_id="poro", shape=porosity.shape)
    runner = AlgorithmRunner()

    result = runner.run_sync(
        TrapEvaluateAlgorithm,
        params={"cell_area_m2": 20.0, "z_step_m": 1.0, "net_to_gross": 0.5},
        input_layers={"trap": _trap(), "top": top, "bottom": bottom, "porosity": por_layer},
        services={"volume_store": _VolumeStore({"poro": porosity})},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    measurement = result.output_layers[0]
    assert isinstance(measurement, MeasurementLayer)
    assert measurement.values["GRV"] > 0.0
    assert np.isclose(measurement.values["mean_phi"], 0.18)
