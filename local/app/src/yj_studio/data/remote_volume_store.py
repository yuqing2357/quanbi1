from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import numpy as np

from yj_studio.config.defaults import DEFAULT_VOLUME_CACHE_SIZE
from yj_studio.io.readers.volume_npy import VolumeSpec

from .volume_store import SliceAxis


@dataclass(frozen=True, slots=True)
class RemoteVolumeProxy:
    """Small shape-carrying proxy for code paths that expect a volume object."""

    store: "RemoteVolumeStore"
    volume_id: str
    shape: tuple[int, int, int]
    dtype: str
    ndim: int = 3

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError("Remote volumes only support orthogonal 2D slice access.")
        int_axes = [idx for idx, value in enumerate(item) if isinstance(value, int)]
        if len(int_axes) != 1:
            raise TypeError("Remote volumes require exactly one integer axis index.")
        axis_index = int_axes[0]
        axis: SliceAxis = ("inline", "xline", "z")[axis_index]  # type: ignore[assignment]
        for idx, value in enumerate(item):
            if idx == axis_index:
                continue
            if value != slice(None):
                raise TypeError("Remote volume slicing currently supports full orthogonal slices only.")
        return self.store.get_slice(self.volume_id, axis, int(item[axis_index]))


class RemoteVolumeStore:
    """Remote volume registry that fetches only requested 2D slices."""

    def __init__(
        self,
        server_url: str,
        *,
        timeout_s: float = 30.0,
        cache_size: int = DEFAULT_VOLUME_CACHE_SIZE,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._cache_size = max(1, int(cache_size))
        self._specs: dict[str, VolumeSpec] = {}
        self._volume_info: dict[str, dict[str, Any]] = {}
        self._slice_cache: OrderedDict[tuple[str, SliceAxis, int], np.ndarray] = OrderedDict()

    @property
    def volume_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def discover_specs(self) -> tuple[dict[str, VolumeSpec], list[str]]:
        payload = self._get_json("/volumes")
        if not isinstance(payload, list):
            raise ValueError("/volumes must return a list")
        specs: dict[str, VolumeSpec] = {}
        notes: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            volume_id = str(item.get("id", ""))
            if not volume_id:
                continue
            if not bool(item.get("exists", False)):
                notes.append(f"{volume_id}: remote file missing, skipped")
                continue
            shape = item.get("shape")
            if not _valid_shape(shape):
                notes.append(f"{volume_id}: missing remote shape, skipped")
                continue
            path_text = str(item.get("path", volume_id))
            filename = Path(path_text).name or volume_id
            specs[volume_id] = VolumeSpec(
                key=volume_id,
                path=Path(path_text),
                label=str(item.get("label", volume_id)),
                cmap=str(item.get("cmap") or "gray"),
                filename=filename,
            )
            self._volume_info[volume_id] = dict(item)
        for spec in specs.values():
            self.register(spec)
        return specs, notes

    def register(self, spec: VolumeSpec) -> None:
        self._specs[spec.key] = spec

    def spec(self, volume_id: str) -> VolumeSpec:
        return self._specs[volume_id]

    def info(self, volume_id: str) -> dict[str, Any]:
        if volume_id not in self._volume_info:
            self.discover_specs()
        return dict(self._volume_info[volume_id])

    def get_volume(self, volume_id: str) -> RemoteVolumeProxy:
        info = self._volume_info.get(volume_id)
        if info is None:
            self.discover_specs()
            info = self._volume_info.get(volume_id)
        if info is None:
            raise KeyError(volume_id)
        shape = info.get("shape")
        if not _valid_shape(shape):
            raise ValueError(f"Remote volume shape missing for {volume_id}")
        return RemoteVolumeProxy(
            store=self,
            volume_id=volume_id,
            shape=tuple(int(v) for v in shape),
            dtype=str(info.get("dtype", "")),
        )

    def get_slice(self, volume_id: str, axis: SliceAxis, index: int) -> np.ndarray:
        key = (volume_id, axis, int(index))
        if key in self._slice_cache:
            data = self._slice_cache.pop(key)
            self._slice_cache[key] = data
            return data

        query = urlencode({"volume_id": volume_id, "axis": axis, "index": int(index)})
        with urlopen(f"{self.server_url}/slice?{query}", timeout=self.timeout_s) as response:
            data = response.read()
        arr = np.load(BytesIO(data), allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"Remote slice must be 2D, got shape {arr.shape}")
        self._slice_cache[key] = np.asarray(arr)
        self._evict_if_needed()
        return self._slice_cache[key]

    def shape(self, volume_id: str) -> tuple[int, int, int]:
        return self.get_volume(volume_id).shape

    def clear(self) -> None:
        self._slice_cache.clear()

    def _get_json(self, path: str) -> Any:
        with urlopen(f"{self.server_url}{path}", timeout=self.timeout_s) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _evict_if_needed(self) -> None:
        while len(self._slice_cache) > self._cache_size:
            self._slice_cache.popitem(last=False)


def _valid_shape(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(isinstance(v, int) and v > 0 for v in value)
