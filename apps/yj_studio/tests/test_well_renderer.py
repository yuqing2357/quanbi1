from __future__ import annotations

import numpy as np
import pytest

from yj_studio.view.renderers.well_renderer import build_well_tube


def test_build_well_tube_from_two_points() -> None:
    trajectory = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32)

    mesh = build_well_tube(trajectory, radius=1.0)

    assert mesh.n_points > 0
    assert mesh.n_cells > 0


def test_build_well_tube_preserves_trajectory_z_bounds() -> None:
    trajectory = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 4.0]], dtype=np.float32)

    mesh = build_well_tube(trajectory, radius=1.0)

    z_min, z_max = mesh.bounds[4], mesh.bounds[5]
    assert z_min == pytest.approx(1.0, abs=1.1)
    assert z_max == pytest.approx(4.0, abs=1.1)


def test_build_well_tube_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="trajectory"):
        build_well_tube(np.zeros((2, 2), dtype=np.float32))
