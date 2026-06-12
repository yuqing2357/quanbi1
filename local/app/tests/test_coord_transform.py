from __future__ import annotations

from yj_studio.data import CoordTransform


def test_full_depth_transform_round_trip() -> None:
    transform = CoordTransform(z_window_start=0.0, depth_step_to_sample=10.0)
    assert transform.depth_m_to_sample(1500.0) == 150.0
    assert transform.sample_to_depth_m(653.0) == 6530.0


def test_current_window_transform_round_trip() -> None:
    transform = CoordTransform(z_window_start=150.0, depth_step_to_sample=10.0)
    assert transform.depth_m_to_sample(1500.0) == 0.0
    assert transform.sample_to_depth_m(0.0) == 1500.0


def test_ijk_inline_xline_offsets() -> None:
    transform = CoordTransform(inline_origin=100.0, xline_origin=200.0)
    assert transform.ijk_to_inline_xline(1, 2, 3) == (101.0, 202.0, 3.0)
    assert transform.inline_xline_to_ijk(101, 202, 3) == (1.0, 2.0, 3.0)

