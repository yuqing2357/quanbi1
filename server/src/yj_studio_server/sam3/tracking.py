"""Engine-agnostic core for multi-object cross-frame tracking.

Kept free of FastAPI so it can be unit-tested with a fake engine and a real
``TargetStore``. The HTTP/rendering glue lives in ``app._run_track_job``; this
module owns the part that matters for correctness: collecting per-object masks
across frames and persisting them as one ``GeoTarget`` per object (so the
obj_id ↔ target_id mapping keeps numbering consistent across the whole sweep).

See docs/project_review_and_remediation.md §2.1.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from ..targets import GeoTarget, TargetSet, TargetStatus, TargetStore


def collect_object_frames(
    engine: Any,
    frames_dir: Any,
    *,
    seeds: list[dict[str, Any]],
    seed_local: int,
    fwd_budget: int,
    back_budget: int,
    indices: list[int],
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """Drive the engine's video predictor and bucket masks per object.

    Returns ``{obj_id: {absolute_frame_index: mask (H,W) bool}}``. Frame-local
    indices yielded by the predictor are mapped back to absolute volume indices
    via ``indices`` (``indices[frame_idx_local]``). Empty masks are dropped.
    """
    engine.init_track_state(frames_dir)
    collected: dict[int, dict[int, np.ndarray]] = {int(s["obj_id"]): {} for s in seeds}
    for frame_idx_local, per_obj in engine.track_video(
        frames_dir,
        seeds=seeds,
        seed_local=seed_local,
        fwd_budget=fwd_budget,
        back_budget=back_budget,
    ):
        if cancelled is not None and cancelled():
            break
        if not 0 <= frame_idx_local < len(indices):
            continue
        abs_index = indices[frame_idx_local]
        for obj_id, mask in per_obj.items():
            mask_arr = np.asarray(mask, dtype=bool)
            if int(obj_id) in collected and mask_arr.any():
                collected[int(obj_id)][abs_index] = mask_arr
        if progress is not None:
            progress(sum(len(frames) for frames in collected.values()))
    return collected


def persist_tracked_targets(
    store: TargetStore,
    collected: dict[int, dict[int, np.ndarray]],
    *,
    seeds: Iterable[dict[str, Any]],
    target_axis: str,
    target_type: str,
    volume_id: str | None,
    image_axis_label: str | None = None,
    source: str = "sam3_video",
    gap_metadata: dict[int, dict[str, Any]] | None = None,
    link_resolver: Callable[
        [int, dict[int, np.ndarray], TargetSet, set[str]], str | None
    ]
    | None = None,
) -> dict[str, Any]:
    """Write one ``GeoTarget`` per seed object, atomically under the store lock.

    Each object's per-frame masks become ``TargetFrame``s with
    ``origin="propagated"``. Id allocation happens inside ``store.mutate()`` so
    concurrent jobs never collide. Returns a summary dict with the new target
    ids, their serialised payloads, and per-target frame counts.
    """
    axis_label = image_axis_label or target_axis
    targets_out_by_id: dict[str, dict[str, Any]] = {}
    target_ids: list[str] = []
    per_target_frames: dict[str, int] = {}
    obj_to_target_id: dict[int, str] = {}
    with store.mutate() as target_set:
        if volume_id and not target_set.volume_id:
            target_set.volume_id = volume_id
        linkable_target_ids = set(target_set.targets)
        for seed_obj in seeds:
            obj_id = int(seed_obj["obj_id"])
            frames = collected.get(obj_id, {})
            if not frames:
                continue
            linked_id = (
                link_resolver(obj_id, frames, target_set, linkable_target_ids)
                if link_resolver is not None
                else None
            )
            if linked_id and linked_id in target_set.targets:
                target_id = linked_id
                target = target_set.targets[target_id]
            else:
                target_id = target_set.new_id()
                target = GeoTarget(
                    id=target_id,
                    type=target_type,
                    volume_id=volume_id,
                    status=TargetStatus.ACTIVE,
                    source=source,
                )
            for abs_index in sorted(frames):
                frame = store.frame_from_mask(
                    target_id=target_id,
                    axis=target_axis,  # type: ignore[arg-type]
                    index=abs_index,
                    mask=frames[abs_index],
                    origin="propagated",
                    image_ref=f"{volume_id}:{axis_label}:{abs_index}",
                )
                target.add_frame(frame)
            _apply_gap_metadata(target, obj_id, gap_metadata.get(obj_id, {}) if gap_metadata else {})
            if target_id not in target_set.targets:
                target_set.add_target(target)
            elif target.type not in target_set.target_types:
                target_set.target_types.append(target.type)
            targets_out_by_id[target_id] = target.model_dump(mode="json")
            if target_id not in target_ids:
                target_ids.append(target_id)
            obj_to_target_id[obj_id] = target_id
            per_target_frames[target_id] = target.frame_count
    return {
        "target_ids": target_ids,
        "targets": list(targets_out_by_id.values()),
        "per_target_frames": per_target_frames,
        "obj_to_target_id": {str(k): v for k, v in obj_to_target_id.items()},
    }


def _apply_gap_metadata(target: GeoTarget, obj_id: int, gap: dict[str, Any]) -> None:
    tracking = dict(target.metadata.get("tracking", {}))
    tracking["last_obj_id"] = int(obj_id)
    if gap:
        tracking["last_gap"] = dict(gap)
        status_hint = str(gap.get("status_hint", "active"))
        if target.status in {TargetStatus.ACTIVE, TargetStatus.LOST}:
            target.status = TargetStatus.LOST if status_hint == "lost" else TargetStatus.ACTIVE
    target.metadata["tracking"] = tracking
