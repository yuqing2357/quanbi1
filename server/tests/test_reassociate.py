from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _src in (_REPO_ROOT / "server" / "src", _REPO_ROOT / "apps" / "yj_studio" / "src"):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from yj_studio_server.sam3.reassociate import (  # noqa: E402
    annotate_gaps,
    centroid,
    detect_merge_split,
    link_targets_by_iou,
    mask_iou,
)


def _mask(coords: list[tuple[int, int]], shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    arr = np.zeros(shape, dtype=bool)
    for y, x in coords:
        arr[y, x] = True
    return arr


def test_mask_iou_and_centroid() -> None:
    left = _mask([(0, 0), (0, 1), (1, 0), (1, 1)])
    shifted = _mask([(0, 1), (0, 2), (1, 1), (1, 2)])
    far = _mask([(3, 3)])

    assert mask_iou(left, left) == 1.0
    assert mask_iou(left, far) == 0.0
    assert round(mask_iou(left, shifted), 3) == 0.333
    assert centroid(left) == (0.5, 0.5)
    assert centroid(np.zeros((2, 2), dtype=bool)) is None


def test_annotate_gaps_marks_trailing_lost() -> None:
    collected = {
        1: {0: np.ones((2, 2), dtype=bool), 1: np.ones((2, 2), dtype=bool)},
        2: {0: np.ones((2, 2), dtype=bool), 3: np.ones((2, 2), dtype=bool)},
    }

    gaps = annotate_gaps(collected, [0, 1, 2, 3], gap_limit=1)

    assert gaps[1]["missing"] == [2, 3]
    assert gaps[1]["trailing_gap"] == 2
    assert gaps[1]["status_hint"] == "lost"
    assert gaps[2]["missing"] == [1, 2]
    assert gaps[2]["trailing_gap"] == 0
    assert gaps[2]["status_hint"] == "active"


def test_link_targets_by_iou_uses_shared_frames() -> None:
    base = _mask([(0, 0), (0, 1), (1, 0), (1, 1)])
    weak = _mask([(3, 3)])
    existing = [
        {"target_id": "T1", "frames": {5: base}},
        {"target_id": "T2", "frames": {5: weak}},
    ]

    assert link_targets_by_iou(existing, {5: base}, iou_thresh=0.8) == "T1"
    assert link_targets_by_iou(existing, {6: base}, iou_thresh=0.1) is None
    assert link_targets_by_iou(existing, {5: weak}, iou_thresh=0.8) == "T2"


def test_detect_merge_split_suggestions() -> None:
    overlap = np.ones((4, 4), dtype=bool)
    split = _mask([(0, 0), (3, 3)])
    collected = {
        1: {0: overlap, 1: overlap, 2: overlap},
        2: {0: overlap, 1: overlap, 2: overlap},
        3: {0: split, 1: split, 2: split},
    }

    suggestions = detect_merge_split(collected, [0, 1, 2], iou_merge=0.9, persist_frames=2)

    assert {"type": "merge", "obj_ids": [1, 2], "frames": [0, 1, 2]} in suggestions
    assert {"type": "split", "obj_id": 3, "frames": [0, 1, 2]} in suggestions
