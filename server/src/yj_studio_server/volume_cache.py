"""Server-side resident volume cache.

The server reads several large ``.npy`` volumes (seismic / lithology /
porosity, tens of GB each).  Without a cache every ``/slice`` request and every
SAM3 frame render reopens the file.  ``VolumeCache`` keeps each volume resident
across requests:

* ``preload_to_ram=True`` reads the full array into process memory at startup
  (one copy, regardless of how many requests/workers hit it).  If the RAM load
  fails (e.g. ``MemoryError``) the volume falls back to a memory-mapped handle
  that is still held open for reuse.
* ``preload_to_ram=False`` (or the on-demand path) holds a memory-mapped handle
  so repeated reads benefit from the OS page cache without committing RAM.

Loading is meant to run in a background thread (see the app lifespan) so the
server can answer ``/health`` immediately while volumes stream into memory.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VolumeCache:
    def __init__(
        self,
        data_root: str | Path,
        volumes: dict[str, dict[str, Any]] | None,
        *,
        preload_to_ram: bool = True,
        stage_dir: str | Path | None = None,
    ) -> None:
        self._data_root = Path(data_root)
        self._specs = dict(volumes or {})
        self._preload_to_ram = bool(preload_to_ram)
        # Optional RAM-backed staging dir (e.g. /dev/shm). When a volume's file
        # exists there it is mmapped from RAM ("shm" mode); otherwise we fall
        # back to the on-disk copy under data_root.
        self._stage_dir = Path(stage_dir) if stage_dir else None
        self._arrays: dict[str, np.ndarray] = {}
        self._modes: dict[str, str] = {}
        self._lock = threading.RLock()
        self._status: dict[str, dict[str, Any]] = {
            volume_id: {
                "volume_id": volume_id,
                "state": "pending",
                "mode": None,
                "error": None,
                "shape": None,
                "dtype": None,
            }
            for volume_id in self._specs
        }

    def _is_staged(self, volume_id: str) -> bool:
        spec = self._specs.get(volume_id)
        if spec is None or self._stage_dir is None:
            return False
        return (self._stage_dir / str(spec.get("path", ""))).exists()

    def _path(self, volume_id: str) -> Path | None:
        spec = self._specs.get(volume_id)
        if spec is None:
            return None
        rel = str(spec.get("path", ""))
        if not rel:
            # Virtual/composite volume (e.g. an rgt_overlay render): it has no
            # file of its own — the app renders it from its source volumes.
            return None
        if self._stage_dir is not None:
            staged = self._stage_dir / rel
            if staged.exists():
                return staged
        return self._data_root / rel

    def get(self, volume_id: str) -> tuple[np.ndarray, str]:
        """Return ``(array, mode)``; load a mmap handle on demand if not preloaded.

        ``mode`` is ``"ram"`` for a fully in-memory array or ``"mmap"`` for a
        memory-mapped handle.
        """
        with self._lock:
            arr = self._arrays.get(volume_id)
            if arr is not None:
                return arr, self._modes[volume_id]
        # Not published yet: open a mmap handle outside the lock (cheap — only
        # reads the header) so we never serialise slow loads behind the lock.
        arr, mode = self._load_one(volume_id, to_ram=False)
        with self._lock:
            if volume_id not in self._arrays:
                self._publish(volume_id, arr, mode)
            return self._arrays[volume_id], self._modes[volume_id]

    def preload_all(self, *, only: list[str] | None = None) -> None:
        ids = list(only) if only else list(self._specs)
        for volume_id in ids:
            with self._lock:
                current = self._status.get(volume_id, {})
                if current.get("state") == "ready" and current.get("mode") == "ram":
                    continue
            if self._path(volume_id) is None:
                # Virtual/composite volume: nothing to load (rendered on demand).
                with self._lock:
                    self._status.setdefault(volume_id, {"volume_id": volume_id})
                    self._status[volume_id].update(state="virtual", mode=None, error=None)
                continue
            try:
                arr, mode = self._load_one(volume_id, to_ram=self._preload_to_ram)
            except FileNotFoundError as exc:
                with self._lock:
                    self._status.setdefault(volume_id, {"volume_id": volume_id})
                    self._status[volume_id].update(state="missing", error=str(exc))
                logger.warning("volume preload skipped (missing): %s", volume_id)
                continue
            except Exception as exc:  # noqa: BLE001 - one bad volume must not kill the rest
                with self._lock:
                    self._status.setdefault(volume_id, {"volume_id": volume_id})
                    self._status[volume_id].update(
                        state="error", error=f"{type(exc).__name__}: {exc}"
                    )
                logger.exception("volume preload failed: %s", volume_id)
                continue
            with self._lock:
                self._publish(volume_id, arr, mode)
            logger.info(
                "volume ready: id=%s mode=%s shape=%s dtype=%s",
                volume_id,
                mode,
                tuple(arr.shape),
                arr.dtype,
            )

    def _load_one(self, volume_id: str, *, to_ram: bool) -> tuple[np.ndarray, str]:
        path = self._path(volume_id)
        if path is None:
            raise KeyError(f"Unknown volume: {volume_id}")
        if not path.exists():
            raise FileNotFoundError(f"Volume file not found: {path}")
        mmap_mode = "shm" if self._is_staged(volume_id) else "mmap"
        if to_ram:
            try:
                return np.load(path), "ram"
            except (MemoryError, OSError) as exc:
                logger.warning(
                    "RAM preload failed for %s (%s); falling back to %s",
                    volume_id,
                    exc,
                    mmap_mode,
                )
                return np.load(path, mmap_mode="r"), mmap_mode
        return np.load(path, mmap_mode="r"), mmap_mode

    def _publish(self, volume_id: str, arr: np.ndarray, mode: str) -> None:
        self._arrays[volume_id] = arr
        self._modes[volume_id] = mode
        path = self._path(volume_id)
        self._status[volume_id] = {
            "volume_id": volume_id,
            "state": "ready",
            "mode": mode,
            "error": None,
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "path": str(path) if path is not None else None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            volumes = [dict(self._status[v]) for v in self._specs]
            ready = sum(1 for v in self._specs if self._status[v].get("state") == "ready")
            ram = sum(1 for v in self._specs if self._status[v].get("mode") == "ram")
            shm = sum(1 for v in self._specs if self._status[v].get("mode") == "shm")
            return {
                "preload_to_ram": self._preload_to_ram,
                "stage_dir": str(self._stage_dir) if self._stage_dir is not None else None,
                "total": len(self._specs),
                "ready": ready,
                "ram_resident": ram,
                "shm_resident": shm,
                "volumes": volumes,
            }
