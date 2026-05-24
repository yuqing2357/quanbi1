from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yj_studio.data.volume_store import VolumeStore
from yj_studio.io.readers.layers_npz import load_layer_grid
from yj_studio.scene.layers import HorizonLayer, VolumeLayer


@dataclass(frozen=True, slots=True)
class HorizonHighPoint:
    inline: int
    xline: int
    sample: float


@dataclass(frozen=True, slots=True)
class HorizonSampleMap:
    title: str
    values: np.ndarray
    mask: np.ndarray
    cmap: str
    colorbar_label: str
    high_point: HorizonHighPoint | None = None


def ensure_horizon_arrays(layer: HorizonLayer) -> None:
    """Load lazy horizon arrays into the layer when needed."""

    if layer.sample is not None:
        return
    if layer.data_path is None:
        return
    grid = load_layer_grid(Path(layer.data_path))
    layer.sample = grid.sample
    layer.mask = grid.mask
    layer.metadata.update(grid.metadata)


def build_structure_map(layer: HorizonLayer) -> HorizonSampleMap:
    ensure_horizon_arrays(layer)
    if layer.sample is None:
        raise ValueError(f"层位 {layer.name} 没有样点网格。")
    sample = np.asarray(layer.sample, dtype=np.float32)
    mask = _valid_horizon_mask(layer, sample)
    values = np.where(mask, sample, np.nan).astype(np.float32)
    return HorizonSampleMap(
        title=f"{layer.name} 构造图",
        values=values,
        mask=mask,
        cmap="terrain",
        colorbar_label="采样值",
        high_point=find_horizon_high_point(layer),
    )


def find_horizon_high_point(layer: HorizonLayer) -> HorizonHighPoint:
    """Return the shallowest valid point on the horizon."""

    ensure_horizon_arrays(layer)
    if layer.sample is None:
        raise ValueError(f"层位 {layer.name} 没有样点网格。")
    sample = np.asarray(layer.sample, dtype=np.float32)
    mask = _valid_horizon_mask(layer, sample)
    if not np.any(mask):
        raise ValueError(f"层位 {layer.name} 没有有效样点。")
    values = np.where(mask, sample, np.inf)
    flat_index = int(np.nanargmin(values))
    inline, xline = np.unravel_index(flat_index, sample.shape)
    return HorizonHighPoint(
        inline=int(inline),
        xline=int(xline),
        sample=float(sample[inline, xline]),
    )


def sample_volume_along_horizon(
    volume_store: VolumeStore,
    volume_layer: VolumeLayer,
    horizon_layer: HorizonLayer,
) -> HorizonSampleMap:
    """Sample the active volume at each valid horizon grid position."""

    if volume_layer.shape is None:
        raise ValueError("请先加载体数据，再沿层采样。")
    ensure_horizon_arrays(horizon_layer)
    if horizon_layer.sample is None:
        raise ValueError(f"层位 {horizon_layer.name} 没有样点网格。")

    volume = np.asarray(volume_store.get_volume(volume_layer.volume_id), dtype=np.float32)
    nx = min(int(volume.shape[0]), int(horizon_layer.sample.shape[0]))
    ny = min(int(volume.shape[1]), int(horizon_layer.sample.shape[1]))
    nz = int(volume.shape[2])
    sample = np.asarray(horizon_layer.sample[:nx, :ny], dtype=np.float32)
    mask = _valid_horizon_mask(horizon_layer, sample, shape=(nx, ny))
    mask &= sample >= 0.0
    mask &= sample <= float(nz - 1)
    values = np.full((nx, ny), np.nan, dtype=np.float32)
    if np.any(mask):
        values[mask] = _interpolate_volume_z(volume, sample, mask)
    return HorizonSampleMap(
        title=f"{horizon_layer.name} 沿层图",
        values=values,
        mask=mask,
        cmap=str(volume_layer.cmap or "seismic"),
        colorbar_label=str(volume_layer.name or volume_layer.volume_id),
        high_point=find_horizon_high_point(horizon_layer),
    )


def _valid_horizon_mask(
    layer: HorizonLayer,
    sample: np.ndarray,
    *,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if shape is None:
        shape = sample.shape
    mask = np.isfinite(sample[: shape[0], : shape[1]])
    if layer.mask is not None:
        mask &= np.asarray(layer.mask[: shape[0], : shape[1]], dtype=bool)
    return mask


def _interpolate_volume_z(volume: np.ndarray, sample: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nz = int(volume.shape[2])
    z0 = np.floor(sample[mask]).astype(np.int64)
    z0 = np.clip(z0, 0, nz - 1)
    z1 = np.clip(z0 + 1, 0, nz - 1)
    frac = sample[mask] - z0.astype(np.float32)
    inline_indices, xline_indices = np.where(mask)
    lower = volume[inline_indices, xline_indices, z0]
    upper = volume[inline_indices, xline_indices, z1]
    return ((1.0 - frac) * lower + frac * upper).astype(np.float32)
