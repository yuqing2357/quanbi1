from __future__ import annotations

from typing import Any

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    if a_bool.shape != b_bool.shape:
        return 0.0
    union = np.logical_or(a_bool, b_bool).sum()
    if not union:
        return 0.0
    return float(np.logical_and(a_bool, b_bool).sum()) / float(union)


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def annotate_gaps(
    collected: dict[int, dict[int, np.ndarray]],
    indices: list[int],
    *,
    gap_limit: int = 5,
) -> dict[int, dict[str, Any]]:
    span = set(int(index) for index in indices)
    ordered = [int(index) for index in indices]
    out: dict[int, dict[str, Any]] = {}
    for obj_id, frames in collected.items():
        present = {int(index) for index in frames}
        missing = sorted(span - present)
        trailing = 0
        for index in reversed(ordered):
            if index in present:
                break
            trailing += 1
        out[int(obj_id)] = {
            "missing": missing,
            "missing_count": len(missing),
            "trailing_gap": trailing,
            "status_hint": "lost" if trailing > int(gap_limit) else "active",
        }
    return out


def link_targets_by_iou(
    existing: list[dict[str, Any]],
    candidate_frames: dict[int, np.ndarray],
    *,
    iou_thresh: float = 0.3,
    min_overlap_frames: int = 1,
) -> str | None:
    best_id: str | None = None
    best_iou = float(iou_thresh)
    for row in existing:
        target_id = str(row.get("target_id") or "")
        frames = row.get("frames")
        if not target_id or not isinstance(frames, dict):
            continue
        shared = sorted(set(frames) & set(candidate_frames))
        if len(shared) < int(min_overlap_frames):
            continue
        mean_iou = float(np.mean([mask_iou(frames[index], candidate_frames[index]) for index in shared]))
        if mean_iou > best_iou:
            best_id = target_id
            best_iou = mean_iou
    return best_id


def detect_merge_split(
    collected: dict[int, dict[int, np.ndarray]],
    indices: list[int],
    *,
    iou_merge: float = 0.5,
    persist_frames: int = 3,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    obj_ids = sorted(int(obj_id) for obj_id in collected)
    ordered = [int(index) for index in indices]

    for pos, first in enumerate(obj_ids):
        for second in obj_ids[pos + 1 :]:
            hot = [
                index
                for index in ordered
                if index in collected[first]
                and index in collected[second]
                and mask_iou(collected[first][index], collected[second][index]) > float(iou_merge)
            ]
            if len(hot) >= int(persist_frames):
                suggestions.append({"type": "merge", "obj_ids": [first, second], "frames": hot})

    for obj_id in obj_ids:
        multi = [
            index
            for index in ordered
            if index in collected[obj_id] and _component_count(collected[obj_id][index]) >= 2
        ]
        if len(multi) >= int(persist_frames):
            suggestions.append({"type": "split", "obj_id": obj_id, "frames": multi})
    return suggestions


def _component_count(mask: np.ndarray) -> int:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return 0
    try:
        from scipy import ndimage  # type: ignore

        return int(ndimage.label(mask_bool)[1])
    except Exception:  # noqa: BLE001 - scipy is optional for local smoke tests
        return _component_count_fallback(mask_bool)


def _component_count_fallback(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count
