from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np

from yj_studio.config.defaults import DEFAULT_VOLUME_CACHE_SIZE
from yj_studio.io.readers.volume_npy import VolumeSpec

SliceAxis = Literal["inline", "xline", "z"]


class VolumeStore:
    """Memory-mapped volume registry with a small LRU cache."""

    def __init__(self, cache_size: int = DEFAULT_VOLUME_CACHE_SIZE) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self._cache_size = cache_size
        self._specs: dict[str, VolumeSpec] = {}
        self._cache: OrderedDict[str, np.memmap] = OrderedDict()

    @property
    def volume_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def register(self, spec: VolumeSpec) -> None:
        self._specs[spec.key] = spec

    def register_path(
        self,
        key: str,
        path: Path,
        *,
        label: str | None = None,
        cmap: str = "gray",
    ) -> None:
        self.register(
            VolumeSpec(
                key=key,
                path=path,
                label=label or key,
                cmap=cmap,
                filename=path.name,
            )
        )

    def spec(self, volume_id: str) -> VolumeSpec:
        return self._specs[volume_id]

    def get_volume(self, volume_id: str) -> np.memmap:
        if volume_id in self._cache:
            volume = self._cache.pop(volume_id)
            self._cache[volume_id] = volume
            return volume

        spec = self._specs.get(volume_id)
        if spec is None:
            raise KeyError(volume_id)
        volume = np.load(spec.path, mmap_mode="r")
        if volume.ndim != 3:
            raise ValueError(f"Volume must be 3D: {spec.path}, got shape {volume.shape}")
        self._cache[volume_id] = volume
        self._evict_if_needed()
        return volume

    def get_slice(self, volume_id: str, axis: SliceAxis, index: int) -> np.ndarray:
        volume = self.get_volume(volume_id)
        axis_index = {"inline": 0, "xline": 1, "z": 2}[axis]
        if not 0 <= index < volume.shape[axis_index]:
            raise IndexError(f"{axis} index {index} outside shape {volume.shape}")
        if axis == "inline":
            return np.asarray(volume[index, :, :])
        if axis == "xline":
            return np.asarray(volume[:, index, :])
        return np.asarray(volume[:, :, index])

    def shape(self, volume_id: str) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.get_volume(volume_id).shape)

    def clear(self) -> None:
        for volume in self._cache.values():
            _close_memmap(volume)
        self._cache.clear()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._cache_size:
            _, volume = self._cache.popitem(last=False)
            _close_memmap(volume)


def _close_memmap(volume: np.memmap) -> None:
    mmap_obj = getattr(volume, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()
