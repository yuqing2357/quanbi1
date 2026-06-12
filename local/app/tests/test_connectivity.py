from __future__ import annotations

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.connectivity import ConnectivityAlgorithm, cuboid_mesh_from_bbox, detect_bodies
from yj_studio.scene.layers import LithBodyLayer, VolumeLayer


class _VolumeStore:
    def __init__(self, volumes: dict[str, np.ndarray]) -> None:
        self._volumes = volumes

    def get_volume(self, volume_id: str) -> np.ndarray:
        return self._volumes[volume_id]


def test_detect_bodies_finds_separate_components() -> None:
    binary = np.zeros((8, 8, 8), dtype=bool)
    binary[1:3, 1:3, 1:3] = True
    binary[5:7, 5:7, 5:7] = True

    bodies = detect_bodies(binary, min_voxels=1, top_k=10)

    assert len(bodies) == 2
    assert [body.voxel_count for body in bodies] == [8, 8]
    assert bodies[0].cells is not None and bodies[0].cells.shape == (8, 3)


def test_detect_bodies_filters_small_components() -> None:
    binary = np.zeros((6, 6, 6), dtype=bool)
    binary[1:3, 1:3, 1:3] = True
    binary[5, 5, 5] = True

    bodies = detect_bodies(binary, min_voxels=2, top_k=10)

    assert len(bodies) == 1
    assert bodies[0].voxel_count == 8


def test_detect_bodies_connectivity_changes_diagonal_linking() -> None:
    binary = np.zeros((4, 4, 4), dtype=bool)
    binary[1, 1, 1] = True
    binary[2, 2, 2] = True

    six = detect_bodies(binary, connectivity=1, min_voxels=1, top_k=10)
    twenty_six = detect_bodies(binary, connectivity=3, min_voxels=1, top_k=10)

    assert len(six) == 2
    assert len(twenty_six) == 1
    assert twenty_six[0].voxel_count == 2


def test_cuboid_mesh_from_bbox_is_visible_for_single_voxel() -> None:
    vertices, faces = cuboid_mesh_from_bbox((2, 2, 3, 3, 4, 4))

    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert vertices[:, 0].max() > vertices[:, 0].min()


def test_connectivity_algorithm_outputs_lith_body_layers() -> None:
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 1.0
    runner = AlgorithmRunner()
    layer = VolumeLayer(name="attr", volume_id="v", shape=volume.shape)

    result = runner.run_sync(
        ConnectivityAlgorithm,
        params={"threshold": 0.5, "min_voxels": 1},
        input_layers={"volume": layer},
        services={"volume_store": _VolumeStore({"v": volume})},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    body = result.output_layers[0]
    assert isinstance(body, LithBodyLayer)
    assert body.metadata["voxel_count"] == 8
