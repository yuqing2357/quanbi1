from __future__ import annotations

import numpy as np


def stretch_to_uint8(
    slice2d: np.ndarray,
    *,
    clim: tuple[float, float] | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, np.ndarray]:
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
        vmin = float(values[finite].min())
        vmax = float(values[finite].max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[finite] = np.clip((values[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8), finite


def slice_to_rgb_image(
    slice2d: np.ndarray,
    *,
    clim: tuple[float, float] | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    gray, _finite = stretch_to_uint8(slice2d, clim=clim, percentile=percentile)
    return np.stack([gray, gray, gray], axis=-1)
