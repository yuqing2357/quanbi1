from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.sam3.detections import (  # noqa: E402
    box_iou_xyxy,
    limit_detections,
    select_box_matched_detections,
)


def _det(box, score):
    return {"box": list(box), "score": float(score), "mask": None}


def test_box_iou_basic() -> None:
    assert box_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # half-overlap along x: intersection 5x10=50, union 100+100-50=150
    assert abs(box_iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_strict_keeps_one_best_detection_per_box() -> None:
    box = [10, 10, 30, 30]
    # A high-overlap low-score detection and a no-overlap high-score detection.
    overlapping = _det([11, 11, 29, 29], score=0.3)
    far_away = _det([80, 80, 95, 95], score=0.99)
    chosen = select_box_matched_detections([far_away, overlapping], [box])
    assert chosen == [overlapping]  # obeys the box, not the global best score


def test_strict_maps_each_box_to_a_distinct_detection() -> None:
    box1 = [0, 0, 20, 20]
    box2 = [60, 60, 90, 90]
    det1 = _det([1, 1, 19, 19], 0.5)
    det2 = _det([61, 61, 89, 89], 0.6)
    chosen = select_box_matched_detections([det1, det2], [box1, box2])
    assert chosen == [det1, det2]


def test_strict_falls_back_to_best_score_when_no_overlap() -> None:
    box = [0, 0, 5, 5]
    far1 = _det([50, 50, 60, 60], 0.4)
    far2 = _det([70, 70, 80, 80], 0.8)
    chosen = select_box_matched_detections([far1, far2], [box])
    assert chosen == [far2]  # nothing overlaps -> keep the globally best


def test_limit_detections_switches_between_modes() -> None:
    box = [0, 0, 10, 10]
    inside = _det([1, 1, 9, 9], 0.2)
    extra = _det([1, 1, 9, 9], 0.9)
    detections = [extra, inside]
    # strict + boxes -> one per box
    strict = limit_detections(detections, box_strict=True, boxes=[box], keep_top_k=3)
    assert len(strict) == 1
    # legacy top-K when strict off
    capped = limit_detections(detections, box_strict=False, boxes=[box], keep_top_k=1)
    assert capped == [extra]
    # strict but no boxes -> behaves like top-K
    no_box = limit_detections(detections, box_strict=True, boxes=[], keep_top_k=2)
    assert no_box == [extra, inside]
