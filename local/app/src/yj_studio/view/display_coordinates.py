from __future__ import annotations

import numpy as np


def display_z(sample_z: np.ndarray | float, z_count: int | None) -> np.ndarray | float:
    """Convert depth sample coordinates to 3D display coordinates.

    The YJ source data uses sample index as depth: larger sample means deeper.
    The previous cigvis view reverses Z by default, so 3D display coordinates
    must place larger samples lower in the scene.
    """

    if z_count is None:
        return sample_z
    return float(z_count - 1) - sample_z


def sample_z(display_z_coord: np.ndarray | float, z_count: int | None) -> np.ndarray | float:
    """Convert 3D display Z back to source sample coordinates."""

    return display_z(display_z_coord, z_count)


def layer_z_count(metadata: dict[str, object]) -> int | None:
    shape = metadata.get("new_volume_shape") or metadata.get("source_shape")
    if isinstance(shape, list | tuple) and len(shape) >= 3:
        return int(shape[2])
    return None


def transform_points_z(points: np.ndarray, z_count: int | None) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    if z_count is not None and out.size:
        out[:, 2] = display_z(out[:, 2], z_count)
    return out


def transform_points_to_sample(points: np.ndarray, z_count: int | None) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    if z_count is not None and out.size:
        out[:, 2] = sample_z(out[:, 2], z_count)
    return out
