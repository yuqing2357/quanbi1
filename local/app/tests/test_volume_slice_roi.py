from __future__ import annotations

import numpy as np

from yj_studio.view.renderers.volume_slice_renderer import build_slice_image, _slice_within_roi


def test_inline_slice_cropped_by_roi() -> None:
    nx, ny, nz = 4, 6, 8
    raw_slice = np.arange(ny * nz, dtype=np.float32).reshape(ny, nz)
    roi = (1, 2, 2, 4, 1, 5)
    image = build_slice_image(
        raw_slice,
        (nx, ny, nz),
        "inline",
        index=1,
        clim=(0.0, float(ny * nz - 1)),
        cmap="gray",
        roi=roi,
    )
    # Cropped image shape: (k1-k0+1, j1-j0+1) = (5, 3)
    assert image.image.shape[:2] == (5, 3)
    # Quad spans j0..j1 and k0..k1 on the inline plane at index=1
    quad = image.points
    assert quad.shape == (4, 3)
    j_min = float(quad[:, 1].min())
    j_max = float(quad[:, 1].max())
    assert j_min == 2.0 and j_max == 4.0


def test_z_slice_cropped_by_roi() -> None:
    nx, ny, nz = 4, 6, 8
    raw_slice = np.arange(nx * ny, dtype=np.float32).reshape(nx, ny)
    roi = (0, 2, 1, 3, 0, 7)
    image = build_slice_image(
        raw_slice,
        (nx, ny, nz),
        "z",
        index=4,
        clim=(0.0, float(nx * ny - 1)),
        cmap="gray",
        roi=roi,
    )
    assert image.image.shape[:2] == (3, 3)


def test_slice_within_roi_returns_false_outside_band() -> None:
    roi = (5, 10, 0, 20, 3, 30)
    assert _slice_within_roi("inline", 4, roi) is False
    assert _slice_within_roi("inline", 7, roi) is True
    assert _slice_within_roi("z", 2, roi) is False
    assert _slice_within_roi("z", 30, roi) is True
