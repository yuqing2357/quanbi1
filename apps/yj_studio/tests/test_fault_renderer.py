from __future__ import annotations

import numpy as np
import pytest

from yj_studio.view.renderers.fault_renderer import build_fault_mesh


def test_build_fault_mesh_triangle_faces() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    mesh = build_fault_mesh(vertices, faces)

    assert mesh.n_points == 4
    assert mesh.n_cells == 2


def test_build_fault_mesh_preserves_vertex_z() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 4.0], [0.0, 1.0, 6.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    mesh = build_fault_mesh(vertices, faces)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([2.0, 4.0, 6.0], dtype=np.float32))


def test_build_fault_mesh_uses_display_z_when_count_is_known() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    mesh = build_fault_mesh(vertices, faces, z_count=6)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([4.0, 3.0, 2.0], dtype=np.float32))


def test_build_fault_mesh_culls_faces_outside_display_window() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 7.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    mesh = build_fault_mesh(vertices, faces, z_count=6)

    assert mesh.n_cells == 1
    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([4.0, 3.0, 2.0], dtype=np.float32))


def test_build_fault_mesh_rejects_bad_faces() -> None:
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.zeros((1, 4), dtype=np.int32)

    with pytest.raises(ValueError, match="faces"):
        build_fault_mesh(vertices, faces)
