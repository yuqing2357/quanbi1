"""Convert a 2D seismic slice into the RGB image format SAM3 expects.

SAM3 image preprocessing (``Sam3Processor.transform``) accepts either a PIL
image or a (3, H, W) tensor. We feed a numpy uint8 RGB array because that's
the cheapest reversible representation:

1. Stretch the slice values to [0, 255] using percentile clipping (robust to
   seismic outliers), with NaN preserved as a separate validity mask so the
   caller can mask out invalid regions in the returned image (filled with 0).
2. Stack the single grayscale channel three times - SAM3 was trained on
   natural images, so feeding a fake-RGB grayscale is the right default.

This is the same convention the legacy ``run_cigvis_web_*`` scripts used when
exporting screenshots, so colormap / aspect ratio downstream is consistent.
"""

from __future__ import annotations

import numpy as np


def stretch_to_uint8(
    slice2d: np.ndarray,
    *,
    clim: tuple[float, float] | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uint8_grayscale, finite_mask)`` for a (H, W) seismic slice.

    ``clim`` overrides percentile-based clipping. If neither is usable
    (e.g. all-NaN slice) the result is all zeros and ``finite_mask`` is all
    False.
    """

    values = np.asarray(slice2d, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return (
            np.zeros(values.shape, dtype=np.uint8),
            np.zeros(values.shape, dtype=bool),
        )
    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin, vmax = np.percentile(values[finite], percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(values[finite].min()), float(values[finite].max() or vmin + 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[finite] = np.clip(
        (values[finite] - vmin) / (vmax - vmin), 0.0, 1.0
    )
    return (normalized * 255.0).astype(np.uint8), finite


def slice_to_rgb_image(
    slice2d: np.ndarray,
    *,
    clim: tuple[float, float] | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Return an ``(H, W, 3)`` uint8 RGB image for SAM3.

    Invalid (NaN/inf) pixels are filled with 0 so they don't bias the
    pretrained image encoder. The caller is responsible for orienting the
    slice (e.g. transposing inline/xline) before calling this.
    """

    gray, _finite = stretch_to_uint8(slice2d, clim=clim, percentile=percentile)
    return np.stack([gray, gray, gray], axis=-1)
