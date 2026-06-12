from __future__ import annotations

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.sandbody_extract import SandbodyExtractAlgorithm, summarize_body
from yj_studio.algorithms.builtin.connectivity import detect_bodies
from yj_studio.scene.layers import LithBodyLayer, VolumeLayer


class _VolumeStore:
    def __init__(self, volumes: dict[str, np.ndarray]) -> None:
        self._volumes = volumes

    def get_volume(self, volume_id: str) -> np.ndarray:
        return self._volumes[volume_id]


def test_summarize_body_reports_volume_and_mean_porosity() -> None:
    porosity = np.zeros((5, 5, 5), dtype=np.float32)
    porosity[1:3, 1:3, 1:3] = 0.25
    body = detect_bodies(porosity >= 0.2, min_voxels=1, top_k=1)[0]

    summary = summarize_body(body, porosity, cell_volume_m3=2.0)

    assert summary["voxel_count"] == 8
    assert summary["volume_m3"] == 16.0
    assert summary["mean_porosity"] == 0.25


def test_sandbody_extract_algorithm_uses_porosity_cutoff() -> None:
    porosity = np.zeros((6, 6, 6), dtype=np.float32)
    porosity[1:4, 1:3, 1:3] = 0.18
    runner = AlgorithmRunner()
    layer = VolumeLayer(name="poro", volume_id="poro", shape=porosity.shape)

    result = runner.run_sync(
        SandbodyExtractAlgorithm,
        params={"porosity_cutoff": 0.1, "min_voxels": 1, "cell_volume_m3": 3.0},
        input_layers={"porosity": layer},
        services={"volume_store": _VolumeStore({"poro": porosity})},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    sand = result.output_layers[0]
    assert isinstance(sand, LithBodyLayer)
    assert sand.metadata["voxel_count"] == 12
    assert sand.metadata["volume_m3"] == 36.0


def test_sandbody_extract_algorithm_uses_lithology_code() -> None:
    porosity = np.full((5, 5, 5), 0.12, dtype=np.float32)
    lithology = np.zeros((5, 5, 5), dtype=np.uint8)
    lithology[2:4, 2:4, 2:4] = 1
    runner = AlgorithmRunner()
    por_layer = VolumeLayer(name="poro", volume_id="poro", shape=porosity.shape)
    lith_layer = VolumeLayer(name="lith", volume_id="lith", shape=lithology.shape)

    result = runner.run_sync(
        SandbodyExtractAlgorithm,
        params={"use_lithology": True, "sand_code": 1, "min_voxels": 1},
        input_layers={"porosity": por_layer, "lithology": lith_layer},
        services={"volume_store": _VolumeStore({"poro": porosity, "lith": lithology})},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    assert result.output_layers[0].metadata["source_mode"] == "lithology"
    assert result.output_layers[0].metadata["voxel_count"] == 8
