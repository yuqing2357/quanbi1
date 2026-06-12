"""Filesystem storage for target metadata, masks, and cell references."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from .model import GeoTarget, TargetFrame, TargetSet, frame_key, utc_now_iso


# Process-global registry of per-project locks. Every ``TargetStore`` pointing
# at the same on-disk project shares one lock, so concurrent jobs (batch
# children, interactive segmentation, future track jobs) serialise their
# read-modify-write of ``targets.json`` instead of clobbering each other or
# handing out duplicate ``next_seq`` ids. Keyed by the absolute project path.
#
# NOTE: this guards writers *within a single process* (the current
# ThreadPoolExecutor job queue). True multi-process GPU workers must either
# add a cross-process file lock or funnel all target writes through a single
# writer process — see docs/project_review_and_remediation.md §1.1.
_PROJECT_LOCKS: dict[str, threading.RLock] = {}
_PROJECT_LOCKS_GUARD = threading.Lock()


def _project_lock(key: str) -> threading.RLock:
    with _PROJECT_LOCKS_GUARD:
        lock = _PROJECT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROJECT_LOCKS[key] = lock
        return lock


class TargetStore:
    """Project-scoped target store.

    Layout:
        <root>/<project>/targets.json
        <root>/<project>/masks/<target_id>/<axis>_<index>.npy
        <root>/<project>/cells/<target_id>/<axis>_<index>.npy
        <root>/<project>/volumes/<target_id>_*.npy
    """

    def __init__(self, root: str | Path, project: str = "default", volume_id: str | None = None):
        self.root = Path(root)
        self.project = project or "default"
        self.volume_id = volume_id
        self.project_root = self.root / self.project

    @property
    def targets_path(self) -> Path:
        return self.project_root / "targets.json"

    @property
    def masks_dir(self) -> Path:
        return self.project_root / "masks"

    @property
    def cells_dir(self) -> Path:
        return self.project_root / "cells"

    @property
    def volumes_dir(self) -> Path:
        return self.project_root / "volumes"

    @property
    def exports_dir(self) -> Path:
        return self.project_root / "exports"

    def ensure_dirs(self) -> None:
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        self.volumes_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _lock(self) -> threading.RLock:
        return _project_lock(os.path.abspath(self.project_root))

    @contextmanager
    def mutate(self) -> Iterator[TargetSet]:
        """Atomic read-modify-write of this project's target set.

        Acquires the per-project lock, loads the current ``TargetSet``, yields
        it for in-place mutation, then saves on clean exit. If the body raises
        (e.g. a 404/400 HTTPException), the save is skipped and the lock is
        released. All server write paths MUST go through this instead of a bare
        ``load()`` ... ``save()`` so concurrent jobs cannot lose updates or
        allocate duplicate ids.
        """
        with self._lock:
            target_set = self.load()
            yield target_set
            self.save(target_set)

    def load(self) -> TargetSet:
        self.ensure_dirs()
        if not self.targets_path.exists():
            return TargetSet(project=self.project, volume_id=self.volume_id)
        payload = self.targets_path.read_text(encoding="utf-8")
        target_set = TargetSet.model_validate_json(payload)
        if not target_set.project:
            target_set.project = self.project
        if self.volume_id and not target_set.volume_id:
            target_set.volume_id = self.volume_id
        return target_set

    def save(self, target_set: TargetSet) -> None:
        self.ensure_dirs()
        target_set.project = target_set.project or self.project
        if self.volume_id and not target_set.volume_id:
            target_set.volume_id = self.volume_id
        target_set.updated_at = utc_now_iso()
        text = target_set.model_dump_json(indent=2)
        if os.name == "nt":
            self.targets_path.write_text(text, encoding="utf-8")
            return
        tmp_path = self.targets_path.with_suffix(".json.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(self.targets_path)

    def relative_mask_ref(self, target_id: str, axis: str, index: int) -> str:
        return f"masks/{target_id}/{axis}_{int(index)}.npy"

    def relative_cells_ref(self, target_id: str, axis: str, index: int) -> str:
        return f"cells/{target_id}/{axis}_{int(index)}.npy"

    def resolve_ref(self, ref: str | Path) -> Path:
        path = Path(ref)
        if path.is_absolute():
            return path
        return self.project_root / path

    def write_mask(self, target_id: str, axis: str, index: int, mask: np.ndarray) -> str:
        self.ensure_dirs()
        ref = self.relative_mask_ref(target_id, axis, index)
        path = self.resolve_ref(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(mask)
        if arr.ndim != 2:
            raise ValueError(f"Target mask must be 2D, got shape {arr.shape}")
        np.save(path, (arr > 0).astype(np.uint8, copy=False))
        return ref

    def read_mask(self, ref: str | Path, mmap_mode: str | None = None) -> np.ndarray:
        return np.load(self.resolve_ref(ref), mmap_mode=mmap_mode)

    def write_cells(
        self,
        target_id: str,
        axis: str,
        index: int,
        cells: np.ndarray,
    ) -> str:
        self.ensure_dirs()
        ref = self.relative_cells_ref(target_id, axis, index)
        path = self.resolve_ref(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(cells)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Cell ids must have shape (N, 3), got {arr.shape}")
        np.save(path, arr.astype(np.int32, copy=False))
        return ref

    def read_cells(self, ref: str | Path, mmap_mode: str | None = None) -> np.ndarray:
        return np.load(self.resolve_ref(ref), mmap_mode=mmap_mode)

    def write_mask3d_cache(self, target_id: str, masks: list[np.ndarray]) -> Path:
        self.ensure_dirs()
        path = self.volumes_dir / f"{target_id}_mask3d.npy"
        if not masks:
            np.save(path, np.zeros((0, 0, 0), dtype=np.uint8))
            return path
        shapes = {tuple(mask.shape) for mask in masks}
        if len(shapes) != 1:
            raise ValueError(f"Cannot stack masks with different shapes: {sorted(shapes)}")
        np.save(path, np.stack([(mask > 0).astype(np.uint8, copy=False) for mask in masks], axis=0))
        return path

    def write_target_mask3d_cache(self, target: GeoTarget) -> tuple[Path, int | None, int | None]:
        self.ensure_dirs()
        path = self.volumes_dir / f"{target.id}_mask3d.npy"
        frames = [frame for frame in target.frames.values() if frame.mask_ref]
        if not frames:
            np.save(path, np.zeros((0, 0, 0), dtype=np.uint8))
            return path, None, None

        index_lo = min(int(frame.index) for frame in frames)
        index_hi = max(int(frame.index) for frame in frames)
        sample = np.asarray(self.read_mask(frames[0].mask_ref))
        if sample.ndim != 2:
            raise ValueError(f"Target mask must be 2D, got shape {sample.shape}")
        height, width = sample.shape
        volume = np.zeros((index_hi - index_lo + 1, height, width), dtype=np.uint8)
        for frame in frames:
            mask = np.asarray(self.read_mask(frame.mask_ref))
            if mask.shape != (height, width):
                raise ValueError(
                    f"Cannot build mask3d for {target.id}; frame {frame.key} has shape {mask.shape}, "
                    f"expected {(height, width)}"
                )
            volume[int(frame.index) - index_lo] = (mask > 0).astype(np.uint8, copy=False)
        np.save(path, volume)
        return path, index_lo, index_hi

    def write_cells_union_cache(self, target_id: str, cell_refs: list[str]) -> Path:
        self.ensure_dirs()
        path = self.volumes_dir / f"{target_id}_cells.npy"
        arrays: list[np.ndarray] = []
        for ref in cell_refs:
            ref_path = self.resolve_ref(ref)
            if ref_path.exists():
                arr = np.asarray(np.load(ref_path), dtype=np.int32)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.size:
                    arrays.append(arr)
        if not arrays:
            np.save(path, np.zeros((0, 3), dtype=np.int32))
            return path
        merged = np.unique(np.concatenate(arrays, axis=0), axis=0)
        np.save(path, merged.astype(np.int32, copy=False))
        return path

    def target_to_summary(self, target: GeoTarget) -> dict[str, object]:
        return {
            "id": target.id,
            "name": target.name,
            "type": target.type,
            "status": target.status.value,
            "frame_range": target.frame_range,
            "frame_count": target.frame_count,
            "area_px": target.area_px,
            "score": target.score,
            "volume_id": target.volume_id,
            "updated_at": target.updated_at,
        }

    def frame_from_mask(
        self,
        *,
        target_id: str,
        axis: Literal["inline", "crossline", "timeslice"],
        index: int,
        mask: np.ndarray,
        score: float | None = None,
        origin: str = "sam3",
        image_ref: str | None = None,
    ) -> TargetFrame:
        mask_ref = self.write_mask(target_id, axis, index, mask)
        binary = np.asarray(mask) > 0
        ys, xs = np.nonzero(binary)
        if xs.size:
            bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            centroid = (float(xs.mean()), float(ys.mean()))
        else:
            bbox = None
            centroid = None
        return TargetFrame(
            axis=axis,
            index=int(index),
            mask_ref=mask_ref,
            bbox=bbox,
            centroid=centroid,
            area_px=int(binary.sum()),
            score=score,
            origin=origin,
            image_ref=image_ref,
        )

    def frame_from_cells(
        self,
        *,
        target_id: str,
        axis: Literal["inline", "crossline", "timeslice"],
        index: int,
        cells: np.ndarray,
        score: float | None = None,
        origin: str = "sam3_reservoir",
        image_ref: str | None = None,
    ) -> TargetFrame:
        cell_ids_ref = self.write_cells(target_id, axis, index, cells)
        arr = np.asarray(cells, dtype=np.int32)
        return TargetFrame(
            axis=axis,
            index=int(index),
            cell_ids_ref=cell_ids_ref,
            area_px=int(arr.shape[0]),
            score=score,
            origin=origin,
            image_ref=image_ref,
        )

    def add_single_frame_target(
        self,
        target_set: TargetSet,
        *,
        axis: Literal["inline", "crossline", "timeslice"],
        index: int,
        mask: np.ndarray,
        target_type: str = "unknown",
        score: float | None = None,
        source: str = "sam3",
        volume_id: str | None = None,
        image_ref: str | None = None,
    ) -> GeoTarget:
        target_id = target_set.new_id()
        frame = self.frame_from_mask(
            target_id=target_id,
            axis=axis,
            index=index,
            mask=mask,
            score=score,
            origin=source,
            image_ref=image_ref,
        )
        target = GeoTarget(
            id=target_id,
            type=target_type,
            volume_id=volume_id or target_set.volume_id,
            source=source,
            score=score,
        )
        target.add_frame(frame)
        target_set.add_target(target)
        return target

    def metadata_is_lightweight(self) -> bool:
        if not self.targets_path.exists():
            return True
        data = json.loads(self.targets_path.read_text(encoding="utf-8"))
        return not _contains_large_inline_array(data)


def get_frame(target: GeoTarget, axis: str, index: int) -> TargetFrame | None:
    return target.frames.get(frame_key(axis, int(index)))  # type: ignore[arg-type]


def _contains_large_inline_array(value: Any, *, threshold: int = 1024) -> bool:
    if isinstance(value, list):
        if len(value) > threshold:
            return True
        return any(_contains_large_inline_array(item, threshold=threshold) for item in value)
    if isinstance(value, dict):
        return any(_contains_large_inline_array(item, threshold=threshold) for item in value.values())
    return False
