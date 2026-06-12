from __future__ import annotations

import numpy as np

from yj_studio.targets import GeoTarget, TargetSet, TargetStatus, review_queue, target_uncertainty
from yj_studio.targets.model import TargetFrame


def _target(target_id: str, *, score: float, areas: list[int], status: TargetStatus = TargetStatus.ACTIVE) -> GeoTarget:
    target = GeoTarget(id=target_id, type="trap", score=score, status=status)
    for index, area in enumerate(areas):
        target.add_frame(
            TargetFrame(
                axis="inline",
                index=index,
                area_px=area,
                score=score,
            )
        )
    return target


def test_target_uncertainty_prioritizes_low_score_and_unstable_area() -> None:
    confident = _target("T1", score=0.95, areas=[100, 102, 98])
    uncertain = _target("T2", score=0.45, areas=[20, 180, 30])

    assert target_uncertainty(uncertain) > target_uncertainty(confident)
    assert np.isclose(target_uncertainty(confident), target_uncertainty(confident))


def test_review_queue_sorts_by_uncertainty_and_skips_final_statuses() -> None:
    target_set = TargetSet(project="default")
    low = target_set.add_target(_target("T1", score=0.9, areas=[100, 100]))
    high = target_set.add_target(_target("T2", score=0.2, areas=[50, 150]))
    confirmed = target_set.add_target(_target("T3", score=0.1, areas=[50, 50], status=TargetStatus.CONFIRMED))
    rejected = target_set.add_target(_target("T4", score=0.1, areas=[50, 50], status=TargetStatus.REJECTED))

    rows = review_queue(target_set)

    assert [row["id"] for row in rows] == [high.id, low.id]
    assert confirmed.id not in {row["id"] for row in rows}
    assert rejected.id not in {row["id"] for row in rows}


def test_review_queue_can_include_confirmed_targets() -> None:
    target_set = TargetSet(project="default")
    target_set.add_target(_target("T1", score=0.9, areas=[100], status=TargetStatus.CONFIRMED))

    rows = review_queue(target_set, include_confirmed=True)

    assert [row["id"] for row in rows] == ["T1"]
