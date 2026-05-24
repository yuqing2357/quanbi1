from __future__ import annotations

import numpy as np

from yj_studio.view.renderers.volume_slice_renderer import build_slice_image, colorize_slice


def test_colorize_slice_returns_rgb_uint8() -> None:
    values = np.asarray([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
    image = colorize_slice(values, (0.0, 1.0), "gray")

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image[1, 1].tolist() == [0, 0, 0]


def test_build_z_slice_image_points() -> None:
    raw_slice = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = build_slice_image(raw_slice, (2, 3, 4), "z", 2, (0.0, 5.0), "gray")

    assert result.image.shape == (3, 2, 3)
    np.testing.assert_array_equal(
        result.points,
        np.asarray([[0, 0, 1], [1, 0, 1], [1, 2, 1], [0, 2, 1]], dtype=np.float32),
    )


def test_build_inline_slice_image_uses_display_z() -> None:
    raw_slice = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = build_slice_image(
        raw_slice,
        (2, 3, 4),
        "inline",
        1,
        (0.0, 11.0),
        "gray",
    )

    np.testing.assert_array_equal(
        result.points,
        np.asarray([[1, 0, 3], [1, 2, 3], [1, 2, 0], [1, 0, 0]], dtype=np.float32),
    )
