from __future__ import annotations

from typing import Any

import numpy as np

from yj_studio.scene.layer import BoundingBox

EMPTY_BOUNDS: BoundingBox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def array_shape(value: np.ndarray | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v) for v in value.shape]


def bbox_from_points(points: np.ndarray | None) -> BoundingBox:
    if points is None or points.size == 0:
        return EMPTY_BOUNDS
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return EMPTY_BOUNDS
    finite = np.all(np.isfinite(xyz[:, :3]), axis=1)
    if not np.any(finite):
        return EMPTY_BOUNDS
    pts = xyz[finite, :3]
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


def update_with_shape(payload: dict[str, Any], field_name: str, value: np.ndarray | None) -> None:
    payload[f"{field_name}_shape"] = array_shape(value)

