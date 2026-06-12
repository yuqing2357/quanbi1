from __future__ import annotations

from typing import TypeGuard

import numpy as np

from yj_studio.services.section_service import SectionAxis

from .layers import ArbitrarySectionLayer, FaultStickLayer, HorizonStickLayer, MeasurementLayer, PolygonLayer, TrapLayer

ManualGeometryLayer = (
    ArbitrarySectionLayer | PolygonLayer | HorizonStickLayer | FaultStickLayer | MeasurementLayer | TrapLayer
)
MANUAL_GEOMETRY_TYPES = (
    ArbitrarySectionLayer,
    PolygonLayer,
    HorizonStickLayer,
    FaultStickLayer,
    MeasurementLayer,
    TrapLayer,
)


def is_manual_geometry_layer(layer: object) -> TypeGuard[ManualGeometryLayer]:
    return isinstance(layer, MANUAL_GEOMETRY_TYPES)


def manual_geometry_points(layer: ManualGeometryLayer) -> np.ndarray | None:
    if isinstance(layer, ArbitrarySectionLayer):
        return as_world_points(layer.polyline)
    if isinstance(layer, PolygonLayer):
        return as_world_points(layer.vertices)
    if isinstance(layer, HorizonStickLayer):
        return as_world_points(layer.points)
    if isinstance(layer, FaultStickLayer):
        return as_world_points(layer.sticks)
    if isinstance(layer, MeasurementLayer):
        return as_world_points(layer.geometry)
    if isinstance(layer, TrapLayer):
        return as_world_points(layer.boundary)
    return None


def as_world_points(value: object) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return None
    if points.shape[1] < 3:
        points = np.column_stack(
            [points, np.zeros((points.shape[0], 3 - points.shape[1]), dtype=np.float32)]
        )
    return points[:, :3]


def project_points_to_section(
    points: np.ndarray | None,
    axis: SectionAxis,
    index: int,
    *,
    tolerance: float = 0.75,
) -> tuple[np.ndarray, np.ndarray] | None:
    if points is None:
        return None
    xyz = as_world_points(points)
    if xyz is None:
        return None
    axis_idx = {"inline": 0, "xline": 1, "z": 2}[axis]
    mask = np.abs(xyz[:, axis_idx] - float(index)) <= float(tolerance)
    if not np.any(mask):
        return None
    selected = xyz[mask]
    if axis == "inline":
        return selected[:, 1], selected[:, 2]
    if axis == "xline":
        return selected[:, 0], selected[:, 2]
    return selected[:, 0], selected[:, 1]
