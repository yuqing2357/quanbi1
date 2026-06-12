from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True, slots=True)
class ArbitrarySectionData:
    polyline_xy: np.ndarray
    sampled_xy: np.ndarray
    distances: np.ndarray
    depths: np.ndarray
    values: np.ndarray


def sample_arbitrary_section(
    volume: np.ndarray,
    polyline_xy: np.ndarray,
    *,
    z_start: int = 0,
    z_end: int | None = None,
    horizontal_step: float = 1.0,
    max_trace_count: int = 900,
    order: int = 1,
) -> ArbitrarySectionData:
    """Sample a vertical seismic section along an XY polyline."""

    cube = np.asarray(volume)
    if cube.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {cube.shape}")
    if horizontal_step <= 0.0:
        raise ValueError("horizontal_step must be positive")
    if max_trace_count < 2:
        raise ValueError("max_trace_count must be >= 2")

    xy = _valid_polyline_xy(polyline_xy)
    sampled_xy, distances = resample_polyline_xy(
        xy,
        horizontal_step=horizontal_step,
        max_trace_count=max_trace_count,
    )
    z_values = _depth_indices(cube.shape[2], z_start, z_end)
    x_grid = np.broadcast_to(sampled_xy[:, 0], (z_values.size, sampled_xy.shape[0]))
    y_grid = np.broadcast_to(sampled_xy[:, 1], (z_values.size, sampled_xy.shape[0]))
    z_grid = np.broadcast_to(z_values[:, None], x_grid.shape)
    coords = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    values = map_coordinates(cube, coords, order=order, mode="nearest").reshape(x_grid.shape)
    return ArbitrarySectionData(
        polyline_xy=xy,
        sampled_xy=sampled_xy,
        distances=distances,
        depths=z_values.astype(np.float32),
        values=np.asarray(values, dtype=np.float32),
    )


def resample_polyline_xy(
    polyline_xy: np.ndarray,
    *,
    horizontal_step: float = 1.0,
    max_trace_count: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    xy = _valid_polyline_xy(polyline_xy)
    deltas = np.diff(xy, axis=0)
    segment_lengths = np.sqrt(np.sum(deltas * deltas, axis=1))
    total = float(np.sum(segment_lengths))
    if total <= 0.0:
        raise ValueError("polyline must contain at least two distinct points")
    trace_count = min(int(max_trace_count), max(2, int(np.floor(total / horizontal_step)) + 1))
    distances = np.linspace(0.0, total, trace_count, dtype=np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float32)
    sampled = np.empty((trace_count, 2), dtype=np.float32)
    for idx, distance in enumerate(distances):
        segment_idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment_idx = int(np.clip(segment_idx, 0, len(segment_lengths) - 1))
        left_distance = float(cumulative[segment_idx])
        length = float(segment_lengths[segment_idx])
        if length <= 0.0:
            frac = 0.0
        else:
            frac = (float(distance) - left_distance) / length
        sampled[idx] = xy[segment_idx] + (xy[segment_idx + 1] - xy[segment_idx]) * frac
    sampled[-1] = xy[-1]
    return sampled, distances


def _valid_polyline_xy(polyline_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(polyline_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] < 2:
        raise ValueError("polyline_xy must have shape (N, 2+) with at least two points")
    xy = xy[:, :2]
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if xy.shape[0] < 2:
        raise ValueError("polyline_xy must contain at least two finite points")
    return xy


def _depth_indices(z_count: int, z_start: int, z_end: int | None) -> np.ndarray:
    if z_count < 1:
        raise ValueError("volume z dimension must be >= 1")
    start = int(np.clip(int(z_start), 0, z_count - 1))
    end = z_count - 1 if z_end is None else int(np.clip(int(z_end), 0, z_count - 1))
    if end < start:
        start, end = end, start
    return np.arange(start, end + 1, dtype=np.float32)
