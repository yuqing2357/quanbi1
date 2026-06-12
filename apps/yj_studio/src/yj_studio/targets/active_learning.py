"""Review queue and uncertainty scoring helpers for GeoTargets."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .model import GeoTarget, TargetSet, TargetStatus


REVIEWABLE_STATUSES: tuple[TargetStatus, ...] = (
    TargetStatus.ACTIVE,
    TargetStatus.LOST,
    TargetStatus.TO_REVIEW,
)


def target_uncertainty(target: GeoTarget) -> float:
    """Return a 0..1 priority score for human review.

    Higher means the target should be reviewed earlier. The score is intentionally
    simple and deterministic: uncertain model scores, unstable frame areas, lost
    status, and manual edits all increase priority.
    """
    score_values = _score_values(target)
    if score_values:
        mean_score = float(np.mean(score_values))
        model_uncertainty = 1.0 - _clip01(mean_score)
    else:
        model_uncertainty = 0.5

    areas = [float(frame.area_px) for frame in target.frames.values() if frame.area_px > 0]
    area_instability = 0.0
    if len(areas) >= 2:
        mean_area = float(np.mean(areas))
        if mean_area > 0:
            area_instability = _clip01(float(np.std(areas) / mean_area))

    status_bonus = 0.2 if target.status == TargetStatus.LOST else 0.0
    edit_bonus = 0.1 if target.edits else 0.0
    return _clip01(0.65 * model_uncertainty + 0.25 * area_instability + status_bonus + edit_bonus)


def review_queue(
    target_set: TargetSet,
    *,
    include_confirmed: bool = False,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in target_set.targets.values():
        if target.status == TargetStatus.DELETED and not include_deleted:
            continue
        if target.status in {TargetStatus.CONFIRMED, TargetStatus.REJECTED} and not include_confirmed:
            continue
        if target.status not in REVIEWABLE_STATUSES and not include_confirmed:
            continue
        uncertainty = target_uncertainty(target)
        rows.append(
            {
                "id": target.id,
                "name": target.name,
                "type": target.type,
                "status": target.status.value,
                "frame_range": target.frame_range,
                "frame_count": target.frame_count,
                "area_px": target.area_px,
                "score": target.score,
                "uncertainty": uncertainty,
                "updated_at": target.updated_at,
            }
        )
    rows.sort(key=lambda row: (-float(row["uncertainty"]), str(row["id"])))
    return rows


def _score_values(target: GeoTarget) -> list[float]:
    values: list[float] = []
    if target.score is not None and math.isfinite(float(target.score)):
        values.append(float(target.score))
    for frame in target.frames.values():
        if frame.score is not None and math.isfinite(float(frame.score)):
            values.append(float(frame.score))
    return values


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
