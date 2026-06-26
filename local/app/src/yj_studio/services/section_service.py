from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yj_studio.data.volume_store import VolumeStore
from yj_studio.scene.layers import FaultSurfaceLayer, HorizonLayer, VolumeLayer, WellLayer

SectionAxis = Literal["inline", "xline", "z"]


@dataclass(frozen=True, slots=True)
class OrthogonalSection:
    axis: SectionAxis
    index: int
    values: np.ndarray
    x_label: str
    y_label: str
    extent: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SectionLine:
    x: np.ndarray
    y: np.ndarray
    label: str
    color: tuple[float, float, float, float]
    opacity: float


@dataclass(frozen=True, slots=True)
class SectionPoints:
    x: np.ndarray
    y: np.ndarray
    label: str
    color: tuple[float, float, float, float]
    opacity: float


def extract_orthogonal_section(
    volume_store: VolumeStore,
    layer: VolumeLayer,
    axis: SectionAxis,
    index: int,
    *,
    slice_volume_id: str | None = None,
) -> OrthogonalSection:
    """Extract one orthogonal 2D section using the same orientation as the 3D slices.

    ``slice_volume_id`` overrides which volume the pixels come from while keeping
    the layer's grid (shape/extent). Used by the rgt_overlay composite, whose own
    id has no raw slice — it borrows the lithology source's grid.
    """

    if layer.shape is None:
        raise ValueError("体数据图层需要有效尺寸。")
    clipped_index = _clip_index(axis, index, layer.shape)
    raw = volume_store.get_slice(slice_volume_id or layer.volume_id, axis, clipped_index)
    values = np.asarray(raw, dtype=np.float32).T
    nx, ny, nz = layer.shape
    if axis == "inline":
        return OrthogonalSection(axis, clipped_index, values, "Xline", "Sample", (0, ny - 1, nz - 1, 0))
    if axis == "xline":
        return OrthogonalSection(axis, clipped_index, values, "Inline", "Sample", (0, nx - 1, nz - 1, 0))
    return OrthogonalSection(axis, clipped_index, values, "Inline", "Xline", (0, nx - 1, ny - 1, 0))


def horizon_intersection(layer: HorizonLayer, axis: SectionAxis, index: int) -> SectionLine | None:
    """Project a horizon grid onto one orthogonal section."""

    if layer.sample is None:
        return None
    sample = np.asarray(layer.sample, dtype=np.float32)
    mask = np.isfinite(sample)
    if layer.mask is not None:
        mask &= np.asarray(layer.mask, dtype=bool)
    if axis == "inline":
        if not 0 <= index < sample.shape[0]:
            return None
        x = np.arange(sample.shape[1], dtype=np.float32)
        y = sample[index, :]
        valid = mask[index, :]
    elif axis == "xline":
        if not 0 <= index < sample.shape[1]:
            return None
        x = np.arange(sample.shape[0], dtype=np.float32)
        y = sample[:, index]
        valid = mask[:, index]
    else:
        return None
    if not np.any(valid):
        return None
    return SectionLine(
        x=x[valid],
        y=y[valid],
        label=layer.name,
        color=layer.color,
        opacity=float(layer.opacity),
    )


def well_intersection(
    layer: WellLayer,
    axis: SectionAxis,
    index: int,
    *,
    tolerance: float = 0.5,
) -> SectionLine | SectionPoints | None:
    """Project a well trajectory when it intersects the current section plane."""

    if layer.trajectory is None:
        return None
    points = np.asarray(layer.trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return None
    axis_idx = _axis_index(axis)
    if axis == "z":
        zmin = float(np.nanmin(points[:, 2]))
        zmax = float(np.nanmax(points[:, 2]))
        if not zmin - tolerance <= float(index) <= zmax + tolerance:
            return None
        x, y = _project_points(points[:1], axis)
        return SectionPoints(x=x, y=y, label=layer.name, color=layer.color, opacity=float(layer.opacity))
    if not np.any(np.abs(points[:, axis_idx] - float(index)) <= tolerance):
        return None
    x, y = _project_points(points, axis)
    return SectionLine(x=x, y=y, label=layer.name, color=layer.color, opacity=float(layer.opacity))


def fault_points_intersection(
    layer: FaultSurfaceLayer,
    axis: SectionAxis,
    index: int,
    *,
    tolerance: float = 1.5,
) -> SectionPoints | None:
    """Project fault vertices close to the current section plane as 2D points."""

    if layer.vertices is None:
        return None
    points = np.asarray(layer.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return None
    axis_idx = _axis_index(axis)
    mask = np.abs(points[:, axis_idx] - float(index)) <= tolerance
    if not np.any(mask):
        return None
    x, y = _project_points(points[mask], axis)
    return SectionPoints(x=x, y=y, label=layer.name, color=layer.color, opacity=float(layer.opacity))


def _clip_index(axis: SectionAxis, index: int, shape: tuple[int, int, int]) -> int:
    limit = {"inline": shape[0], "xline": shape[1], "z": shape[2]}[axis]
    return int(np.clip(int(index), 0, limit - 1))


def _axis_index(axis: SectionAxis) -> int:
    return {"inline": 0, "xline": 1, "z": 2}[axis]


def _project_points(points: np.ndarray, axis: SectionAxis) -> tuple[np.ndarray, np.ndarray]:
    if axis == "inline":
        return points[:, 1], points[:, 2]
    if axis == "xline":
        return points[:, 0], points[:, 2]
    return points[:, 0], points[:, 1]
