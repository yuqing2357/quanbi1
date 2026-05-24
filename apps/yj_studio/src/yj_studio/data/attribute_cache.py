from __future__ import annotations

import numpy as np


def estimate_clim(volume: np.ndarray, x_idx: int, y_idx: int, z_idx: int) -> list[float]:
    probes = [
        np.asarray(volume[x_idx, :, :], dtype=np.float32),
        np.asarray(volume[:, y_idx, :], dtype=np.float32),
        np.asarray(volume[:, :, z_idx], dtype=np.float32),
    ]
    merged = np.concatenate([probe.ravel() for probe in probes])
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return [0.0, 1.0]
    vmin, vmax = np.percentile(merged, [2.0, 98.0])
    return [float(vmin), float(vmax)]


def estimate_volume_clim(
    volume_key: str,
    volume: np.ndarray,
    x_idx: int,
    y_idx: int,
    z_idx: int,
) -> list[float]:
    if volume_key == "coherence":
        return [0.0, 1.0]
    if volume_key == "azimuth_deg":
        return [0.0, 360.0]
    if volume_key == "model_lithology":
        return [-0.5, 2.5]
    clim = estimate_clim(volume, x_idx, y_idx, z_idx)
    if volume_key.startswith("curvature_"):
        limit = max(abs(clim[0]), abs(clim[1]))
        return [-float(limit), float(limit)]
    return clim

