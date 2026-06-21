"""Pure detection post-processing helpers (no FastAPI / torch import).

Kept separate from ``app.py`` so the box-selection logic that decides which
SAM3 detections survive can be unit-tested without standing up the server.
"""

from __future__ import annotations

from typing import Any


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    """IoU of two ``[x0, y0, x1, y1]`` boxes (orientation-agnostic)."""
    ax0, ax1 = sorted((float(a[0]), float(a[2])))
    ay0, ay1 = sorted((float(a[1]), float(a[3])))
    bx0, bx1 = sorted((float(b[0]), float(b[2])))
    by0, by1 = sorted((float(b[1]), float(b[3])))
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0.0 else 0.0


def select_box_matched_detections(
    detections: list[dict[str, Any]],
    boxes: list[list[float]],
) -> list[dict[str, Any]]:
    """Keep exactly the detection that best fills each user selection box.

    Powers the ``box_strict`` option: instead of surfacing the global top-K
    candidates, the result obeys the framed region — one target per box, the
    detection whose box has the highest IoU with the prompt (ties broken by
    score). Detections matching no box are dropped; if a box overlaps nothing,
    the globally best detection is returned so the user still gets a result.
    Detection ``box`` and prompt ``box`` share the server image's pixel xyxy
    space (the prompt box is fed to SAM3 unchanged), so IoU is directly valid.
    """
    if not boxes or not detections:
        return detections
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for prompt_box in boxes:
        if not prompt_box or len(prompt_box) < 4:
            continue
        best_i = -1
        best_key: tuple[float, float] | None = None
        for i, det in enumerate(detections):
            if i in used:
                continue
            det_box = det.get("box")
            if not det_box or len(det_box) < 4:
                continue
            iou = box_iou_xyxy(prompt_box, det_box)
            if iou <= 0.0:
                continue
            key = (iou, float(det.get("score", 0.0)))
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        if best_i >= 0:
            used.add(best_i)
            chosen.append(detections[best_i])
    if not chosen:
        chosen = [max(detections, key=lambda d: float(d.get("score", 0.0)))]
    return chosen


def limit_detections(
    detections: list[dict[str, Any]],
    *,
    box_strict: bool,
    boxes: list[list[float]],
    keep_top_k: int,
) -> list[dict[str, Any]]:
    """Apply either box-strict selection or the legacy top-K cap."""
    if box_strict and boxes:
        return select_box_matched_detections(detections, boxes)
    return detections[: max(1, int(keep_top_k))]
