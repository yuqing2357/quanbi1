from __future__ import annotations

import numpy as np

from yj_studio.view.renderers.volume_slice_renderer import build_slice_image, colorize_slice


def test_colorize_slice_returns_rgb_uint8() -> None:
    values = np.asarray([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
    image = colorize_slice(values, (0.0, 1.0), "gray")

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image[1, 1].tolist() == [0, 0, 0]


def test_colorize_slice_with_display_mask_returns_rgba_uint8() -> None:
    values = np.asarray([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    display_mask = np.asarray([[True, False], [False, True]])
    image = colorize_slice(values, (0.0, 1.0), "gray", display_mask=display_mask)

    assert image.shape == (2, 2, 4)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[..., 3], display_mask.astype(np.uint8) * 255)
    assert image[0, 1, :3].tolist() == [0, 0, 0]
    assert image[1, 0, :3].tolist() == [0, 0, 0]


def test_colorize_lithology_uses_discrete_project_palette() -> None:
    values = np.asarray([[np.nan, 0.0, 1.0]], dtype=np.float32)
    image = colorize_slice(values, (-0.5, 1.5), "lithology_binary")

    assert image.shape == (1, 3, 3)
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[0, 1].tolist() == [47, 47, 47]
    assert image[0, 2].tolist() == [255, 221, 0]


def test_colorize_lithology_display_mask_is_transparent_outside_reservoir() -> None:
    values = np.asarray([[0.0, 1.0]], dtype=np.float32)
    display_mask = np.asarray([[False, True]])
    image = colorize_slice(values, (-0.5, 1.5), "lithology_binary", display_mask=display_mask)

    assert image.shape == (1, 2, 4)
    assert image[0, 0].tolist() == [0, 0, 0, 0]
    assert image[0, 1].tolist() == [255, 221, 0, 255]


def test_build_z_slice_image_points() -> None:
    raw_slice = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = build_slice_image(raw_slice, (2, 3, 4), "z", 2, (0.0, 5.0), "gray")

    assert result.image.shape == (3, 2, 3)
    np.testing.assert_array_equal(
        result.points,
        np.asarray([[0, 0, 1], [1, 0, 1], [1, 2, 1], [0, 2, 1]], dtype=np.float32),
    )


def test_build_z_slice_image_reverses_texture_rows_to_preserve_axis1_direction() -> None:
    raw_slice = np.zeros((2, 3), dtype=np.float32)
    raw_slice[1, 2] = 1.0  # high axis0, high axis1 corner

    result = build_slice_image(
        raw_slice,
        (2, 3, 4),
        "z",
        2,
        (-0.5, 1.5),
        "lithology_binary",
    )

    # VTK samples v=0 from the last image row.  Reversing image rows therefore
    # keeps high axis1 at the quad's high-axis1 edge instead of mirroring it.
    assert result.image[0, 1].tolist() == [255, 221, 0]
    assert result.image[-1, 1].tolist() == [47, 47, 47]


def test_build_z_slice_image_applies_display_mask() -> None:
    raw_slice = np.ones((2, 3), dtype=np.float32)
    display_mask = np.asarray([[True, False, True], [False, True, False]])
    result = build_slice_image(
        raw_slice,
        (2, 3, 4),
        "z",
        2,
        (0.0, 1.0),
        "gray",
        display_mask=display_mask,
    )

    assert result.image.shape == (3, 2, 4)
    np.testing.assert_array_equal(
        result.image[..., 3],
        display_mask.T[::-1, :].astype(np.uint8) * 255,
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
