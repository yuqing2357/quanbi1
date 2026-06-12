from __future__ import annotations

import numpy as np
import pytest

from yj_studio.view.renderers.lith_body_renderer import build_lith_body_mesh


def test_build_lith_body_mesh_triangle_faces() -> None:
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

    mesh = build_lith_body_mesh(vertices, faces)

    assert mesh.n_points == 4
    assert mesh.n_cells == 2


def test_build_lith_body_mesh_preserves_vertex_z() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 3.0], [0.0, 1.0, 5.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    mesh = build_lith_body_mesh(vertices, faces)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([1.0, 3.0, 5.0], dtype=np.float32))


def test_build_lith_body_mesh_uses_display_z_when_count_is_known() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 3.0], [0.0, 1.0, 5.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    mesh = build_lith_body_mesh(vertices, faces, z_count=7)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([5.0, 3.0, 1.0], dtype=np.float32))


def test_build_lith_body_mesh_rejects_bad_vertices() -> None:
    with pytest.raises(ValueError, match="vertices"):
        build_lith_body_mesh(np.zeros((3, 2), dtype=np.float32), np.zeros((1, 3), dtype=np.int32))
