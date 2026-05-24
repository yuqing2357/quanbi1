from __future__ import annotations

import numpy as np
import pytest

from yj_studio.view.renderers.well_log_renderer import build_well_log_points


def test_build_well_log_points_from_samples() -> None:
    samples = np.asarray(
        [[1.0, 2.0, 3.0, 0.12], [1.0, 2.0, 4.0, 0.14]],
        dtype=np.float32,
    )

    mesh = build_well_log_points(samples)

    assert mesh.n_points == 2
    assert mesh["value"].tolist() == pytest.approx([0.12, 0.14])


def test_build_well_log_points_preserves_sample_z() -> None:
    samples = np.asarray(
        [[1.0, 2.0, 3.0, 0.12], [1.0, 2.0, 4.0, 0.14]],
        dtype=np.float32,
    )

    mesh = build_well_log_points(samples)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([3.0, 4.0], dtype=np.float32))


def test_build_well_log_points_uses_display_z_when_count_is_known() -> None:
    samples = np.asarray(
        [[1.0, 2.0, 3.0, 0.12], [1.0, 2.0, 4.0, 0.14]],
        dtype=np.float32,
    )

    mesh = build_well_log_points(samples, z_count=8)

    np.testing.assert_allclose(mesh.points[:, 2], np.asarray([4.0, 3.0], dtype=np.float32))


def test_build_well_log_points_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="samples"):
        build_well_log_points(np.zeros((2, 3), dtype=np.float32))
