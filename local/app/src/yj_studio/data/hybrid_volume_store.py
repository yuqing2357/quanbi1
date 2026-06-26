"""Hybrid volume registry: serve backgrounds from local disk when available.

The remote server owns the authoritative volume catalogue (shapes, colormap,
clim, voxel spacing). But the *pixel data* for a background slice is a pure
function of ``(volume_id, axis, index)`` — if the same volume file already lives
on the local disk there is no reason to stream it over HTTP every time.

``HybridVolumeStore`` keeps both a local :class:`VolumeStore` (memory-mapped,
never loads a whole volume into RAM) and a :class:`RemoteVolumeStore`. It fetches
the catalogue once from the server, then routes every per-volume call to whoever
owns that volume: local mmap if the file is present under the local data root,
remote streaming otherwise. Slice orientation is identical on both sides (the
server ``/slice`` endpoint and ``VolumeStore`` both return the raw orthogonal
slice with no transpose), so overlaying server-produced masks is unaffected by
which side served the background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yj_studio.io.readers.volume_npy import VolumeSpec

from .remote_volume_store import RemoteVolumeStore
from .volume_store import SliceAxis, VolumeStore

import logging

logger = logging.getLogger(__name__)


def resolve_local_volume_path(server_path: str, local_data_root: Path) -> Path | None:
    """Map a server-side absolute volume path onto the local data root.

    The server reports volume paths as absolute paths under *its* ``data_root``
    (e.g. ``/root/quanbi/data/reservoir/.../porosity_float16.npy``). We try, in
    order: the ``data``-relative tail joined onto ``local_data_root``; the bare
    filename directly under ``local_data_root``; and finally a recursive search
    for the filename. Returns the first existing path, or ``None``.
    """

    server = Path(server_path)
    parts = server.parts
    if "data" in parts:
        tail = Path(*parts[parts.index("data") + 1 :])
        candidate = local_data_root / tail
        if candidate.exists():
            return candidate
    direct = local_data_root / server.name
    if direct.exists():
        return direct
    if local_data_root.exists():
        for match in local_data_root.rglob(server.name):
            if match.is_file():
                return match
    return None


class HybridVolumeStore:
    """Route slice access to a local mmap store or a remote store per volume."""

    def __init__(
        self,
        remote_store: RemoteVolumeStore,
        local_data_root: Path,
        *,
        local_store: VolumeStore | None = None,
    ) -> None:
        self._remote = remote_store
        self._local = local_store if local_store is not None else VolumeStore()
        self._local_data_root = Path(local_data_root)
        # volume_id -> "local" | "remote"
        self._owner: dict[str, str] = {}
        self._specs: dict[str, VolumeSpec] = {}

    @property
    def volume_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def discover_specs(self) -> tuple[dict[str, VolumeSpec], list[str]]:
        specs, notes = self._remote.discover_specs()
        merged: dict[str, VolumeSpec] = {}
        for volume_id, spec in specs.items():
            # A composite render (e.g. rgt_overlay) has no file of its own; it is
            # rendered from its source volumes. Keep it remote/virtual and skip the
            # local-file probe (which would otherwise rglob the data tree in vain).
            if self._remote.info(volume_id).get("render"):
                self._owner[volume_id] = "remote"
                merged[volume_id] = spec
                logger.info("Hybrid volume %s is a composite render (no raw slices)", volume_id)
                continue
            local_path = resolve_local_volume_path(str(spec.path), self._local_data_root)
            if local_path is not None:
                local_spec = VolumeSpec(
                    key=volume_id,
                    path=local_path,
                    label=spec.label,
                    cmap=spec.cmap,
                    filename=local_path.name,
                )
                self._local.register(local_spec)
                self._owner[volume_id] = "local"
                merged[volume_id] = local_spec
                notes.append(f"{volume_id}: 使用本地体数据 {local_path}")
                logger.info("Hybrid volume %s served locally from %s", volume_id, local_path)
            else:
                self._owner[volume_id] = "remote"
                merged[volume_id] = spec
                logger.info("Hybrid volume %s served remotely (no local copy)", volume_id)
        self._specs.update(merged)
        return merged, notes

    def register(self, spec: VolumeSpec) -> None:
        self._specs[spec.key] = spec
        owner = self._owner.get(spec.key)
        if owner == "local":
            self._local.register(spec)
        elif owner == "remote":
            self._remote.register(spec)
        else:
            # Unknown volume registered before discovery: decide by local file.
            local_path = resolve_local_volume_path(str(spec.path), self._local_data_root)
            if local_path is not None and spec.path == local_path:
                self._owner[spec.key] = "local"
                self._local.register(spec)
            else:
                self._owner[spec.key] = "remote"
                self._remote.register(spec)

    def _store_for(self, volume_id: str):
        return self._local if self._owner.get(volume_id) == "local" else self._remote

    def spec(self, volume_id: str) -> VolumeSpec:
        return self._specs[volume_id]

    def info(self, volume_id: str) -> dict[str, Any]:
        # Catalogue metadata (shape/dtype/spacing/cmap/clim) is always the
        # server's authoritative copy, regardless of who serves the pixels.
        return self._remote.info(volume_id)

    def get_volume(self, volume_id: str):
        return self._store_for(volume_id).get_volume(volume_id)

    def get_slice(self, volume_id: str, axis: SliceAxis, index: int):
        return self._store_for(volume_id).get_slice(volume_id, axis, int(index))

    def shape(self, volume_id: str) -> tuple[int, int, int]:
        return self._store_for(volume_id).shape(volume_id)

    def clear(self) -> None:
        self._local.clear()
        self._remote.clear()
