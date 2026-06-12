from __future__ import annotations

import numpy as np

from yj_studio.view.renderers.horizon_renderer import build_horizon_mesh


def test_build_horizon_mesh_skips_invalid_cells() -> None:
    sample = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    mask = np.ones(sample.shape, dtype=bool)
    mask[2, 2] = False

    mesh = build_horizon_mesh(sample, mask, stride=1)

    assert mesh.n_points == 9
    assert mesh.n_cells == 3


def test_build_horizon_mesh_stride() -> None:
    sample = np.arange(25, dtype=np.float32).reshape(5, 5)
    mask = np.ones(sample.shape, dtype=bool)

    mesh = build_horizon_mesh(sample, mask, stride=2)

    assert mesh.n_points == 9
    assert mesh.n_cells == 4


def test_build_horizon_mesh_preserves_sample_z() -> None:
    sample = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.ones(sample.shape, dtype=bool)

    mesh = build_horizon_mesh(sample, mask, stride=1)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_build_horizon_mesh_uses_display_z_when_count_is_known() -> None:
    sample = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.ones(sample.shape, dtype=bool)

    mesh = build_horizon_mesh(sample, mask, stride=1, z_count=6)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32))
