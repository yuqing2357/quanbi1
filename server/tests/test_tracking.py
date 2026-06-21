"""Unit tests for the multi-object tracking core (review §2.1).

Runs without FastAPI / a real SAM3 model: a fake engine stands in for the
video predictor, and a real ``TargetStore`` verifies that each tracked object
becomes one ``GeoTarget`` with a stable id and propagated frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _src in (_REPO_ROOT / "server" / "src", _REPO_ROOT / "shared" / "src"):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from yj_studio_core.targets import TargetStore  # noqa: E402
from yj_studio_core.targets.model import GeoTarget, TargetSet, TargetStatus  # noqa: E402
from yj_studio_server.sam3.reassociate import link_targets_by_iou  # noqa: E402
from yj_studio_server.sam3.tracking import (  # noqa: E402
    chunked_track,
    collect_object_frames,
    mask_bbox_xyxy,
    persist_tracked_targets,
)


class FakeTrackEngine:
    """Stands in for SAM3Engine: yields one mask per object per frame."""

    def __init__(self, n_frames: int, hw: tuple[int, int]) -> None:
        self.n_frames = n_frames
        self.hw = hw
        self.init_calls = 0
        self.reset_calls = 0

    def init_track_state(self, frames_dir) -> None:  # noqa: ANN001
        self.init_calls += 1

    def reset_track_state(self) -> None:
        self.reset_calls += 1

    def track_video(self, frames_dir, *, seeds, seed_local, fwd_budget, back_budget, **_):  # noqa: ANN001
        h, w = self.hw
        obj_ids = [int(s["obj_id"]) for s in seeds]
        for frame_idx_local in range(self.n_frames):
            per_obj = {}
            for k, oid in enumerate(obj_ids):
                mask = np.zeros((h, w), dtype=bool)
                mask[min(k, h - 1), :] = True  # distinct row per object
                per_obj[oid] = mask
            yield frame_idx_local, per_obj


def _row_mask(row: int, hw=(8, 8)) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    m[row, 2:5] = True
    return m


def test_mask_bbox_xyxy_is_tight_or_none() -> None:
    assert mask_bbox_xyxy(np.zeros((4, 4), dtype=bool)) is None
    box = mask_bbox_xyxy(_row_mask(3))
    assert box == [2.0, 3.0, 5.0, 4.0]  # xmin,ymin,xmax,ymax


def test_chunked_track_relays_across_chunks_until_target_persists() -> None:
    """A target alive through frame 25 is tracked across multiple chunks, each
    re-seeded from the previous chunk's last bbox; memory bound = chunk_size."""
    seed = 10
    alive_until = 25  # forward target disappears after frame 25
    chunk_size = 6
    chunk_calls: list[tuple[int, int]] = []

    def run_chunk(chunk_indices, seed_local, boxes_by_obj):
        chunk_calls.append((chunk_indices[0], chunk_indices[-1]))
        assert len(chunk_indices) <= chunk_size + 1  # one chunk worth of frames
        out: dict[int, dict[int, np.ndarray]] = {1: {}}
        for idx in chunk_indices:
            if idx <= alive_until:  # forward-only test
                out[1][idx] = _row_mask(min(idx % 7, 7))
        return out

    collected = chunked_track(
        seed=seed,
        fwd=40,
        back=0,
        axis_len=200,
        seed_boxes={1: [2.0, 3.0, 5.0, 4.0]},
        chunk_size=chunk_size,
        run_chunk=run_chunk,
    )

    frames = sorted(collected[1])
    # Tracked contiguously from seed up to where the target disappeared.
    assert frames[0] == seed
    assert frames[-1] == alive_until
    # Multiple chunks were needed, and no single chunk exceeded the bound.
    assert len(chunk_calls) >= 2
    assert all((hi - lo + 1) <= chunk_size + 1 for lo, hi in chunk_calls)


def test_chunked_track_hard_caps_span_even_if_target_persists() -> None:
    """Free tracking stops at the per-direction cap, not at the volume edge.

    The target is present on every frame, but ``max_span=120`` bounds the sweep
    to 120 forward + 120 backward — no further pass is started past the cap."""
    def run_chunk(chunk_indices, seed_local, boxes_by_obj):
        out: dict[int, dict[int, np.ndarray]] = {1: {}}
        for idx in chunk_indices:
            out[1][idx] = _row_mask(3)  # alive everywhere, identical bbox
        return out

    collected = chunked_track(
        seed=200,
        fwd=500,
        back=500,
        axis_len=2000,
        seed_boxes={1: [2.0, 3.0, 5.0, 4.0]},
        chunk_size=10,
        run_chunk=run_chunk,
        max_span=120,
    )

    frames = sorted(collected[1])
    assert frames[0] == 200 - 120  # backward capped
    assert frames[-1] == 200 + 120  # forward capped
    assert len(frames) == 241


def test_chunked_track_drift_guard_stops_on_relay_mismatch() -> None:
    """A relay chunk whose re-detected anchor mask doesn't overlap the fed bbox
    is treated as drift: that direction stops and the hop is not merged in."""
    calls: list[int] = []

    def run_chunk(chunk_indices, seed_local, boxes_by_obj):
        calls.append(len(calls) + 1)
        # First chunk tracks the real target (row 3); a later relay chunk
        # "re-detects" a different region (row 6) -> zero IoU with the fed bbox.
        row = 3 if len(calls) == 1 else 6
        out: dict[int, dict[int, np.ndarray]] = {1: {}}
        for idx in chunk_indices:
            out[1][idx] = _row_mask(row)
        return out

    events: dict[str, object] = {}
    collected = chunked_track(
        seed=10,
        fwd=120,
        back=0,
        axis_len=400,
        seed_boxes={1: [2.0, 3.0, 5.0, 4.0]},
        chunk_size=6,
        run_chunk=run_chunk,
        drift_iou_min=0.1,
        events=events,
    )

    frames = sorted(collected[1])
    # Only the clean first chunk survives; the drifted relay is dropped.
    assert frames[0] == 10
    assert frames[-1] == 16  # seed + chunk_size, where the relay then drifted
    # Every surviving frame is the clean row-3 target, never the row-6 hop.
    assert all(mask_bbox_xyxy(collected[1][i]) == [2.0, 3.0, 5.0, 4.0] for i in frames)
    assert events.get("drift_stops")
    assert events["drift_stops"][0]["direction"] == "forward"


def test_chunked_track_stops_direction_when_target_absent_immediately() -> None:
    def run_chunk(chunk_indices, seed_local, boxes_by_obj):
        # Only the seed frame has a mask; nothing propagates outward.
        return {1: {chunk_indices[seed_local]: _row_mask(1)}}

    collected = chunked_track(
        seed=5,
        fwd=30,
        back=30,
        axis_len=100,
        seed_boxes={1: [2.0, 1.0, 5.0, 2.0]},
        chunk_size=8,
        run_chunk=run_chunk,
    )
    assert sorted(collected[1]) == [5]  # locked to the seed frame only


def test_target_sequence_resumes_after_highest_persisted_id() -> None:
    target_set = TargetSet(next_seq=1, targets={"T46": GeoTarget(id="T46")})

    assert target_set.next_seq == 47
    assert target_set.new_id() == "T47"


def test_track_assigns_one_consistent_id_per_object(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="vol")
    indices = [5, 6, 7, 8]
    seeds = [
        {"obj_id": 1, "box_xywh": [0.5, 0.5, 0.2, 0.2], "text": ""},
        {"obj_id": 2, "box_xywh": [0.3, 0.3, 0.1, 0.1], "text": ""},
    ]
    engine = FakeTrackEngine(n_frames=len(indices), hw=(4, 6))

    collected = collect_object_frames(
        engine,
        tmp_path / "frames",
        seeds=seeds,
        seed_local=1,
        fwd_budget=2,
        back_budget=1,
        indices=indices,
    )

    assert engine.init_calls == 1
    # The video session is always released after the sweep (OOM guard).
    assert engine.reset_calls == 1
    assert set(collected) == {1, 2}
    # frame-local indices mapped back to absolute volume indices
    assert sorted(collected[1]) == indices
    assert sorted(collected[2]) == indices

    summary = persist_tracked_targets(
        store,
        collected,
        seeds=seeds,
        target_axis="inline",
        target_type="trap",
        volume_id="vol",
    )

    # Each object -> exactly one target with a stable id; numbering is unique.
    assert summary["target_ids"] == ["T1", "T2"]
    assert summary["per_target_frames"] == {"T1": 4, "T2": 4}

    loaded = store.load()
    assert set(loaded.targets) == {"T1", "T2"}
    assert loaded.next_seq == 3

    t1 = loaded.targets["T1"]
    assert t1.type == "trap"
    assert t1.source == "sam3_video"
    assert t1.frame_count == 4
    frame = t1.frames["inline:5"]
    assert frame.origin == "propagated"
    assert frame.mask_ref == "masks/T1/inline_5.npy"
    stored = store.read_mask(frame.mask_ref)
    assert stored.shape == (4, 6)
    assert stored.sum() > 0


def test_track_skips_objects_with_no_frames(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="vol")
    collected = {1: {5: np.ones((3, 3), dtype=bool)}, 2: {}}
    seeds = [{"obj_id": 1}, {"obj_id": 2}]

    summary = persist_tracked_targets(
        store,
        collected,
        seeds=seeds,
        target_axis="inline",
        target_type="trap",
        volume_id="vol",
    )

    assert summary["target_ids"] == ["T1"]
    assert "T2" not in store.load().targets


def test_collect_drops_empty_masks_and_out_of_range_frames(tmp_path: Path) -> None:
    indices = [10, 11]
    seeds = [{"obj_id": 1, "box_xywh": [0.5, 0.5, 0.2, 0.2], "text": ""}]

    class PartialEngine(FakeTrackEngine):
        def track_video(self, frames_dir, *, seeds, seed_local, fwd_budget, back_budget, **_):  # noqa: ANN001
            yield 0, {1: np.ones((4, 4), dtype=bool)}     # kept -> index 10
            yield 1, {1: np.zeros((4, 4), dtype=bool)}    # empty -> dropped
            yield 9, {1: np.ones((4, 4), dtype=bool)}     # out of range -> ignored

    collected = collect_object_frames(
        PartialEngine(n_frames=2, hw=(4, 4)),
        tmp_path / "frames",
        seeds=seeds,
        seed_local=0,
        fwd_budget=1,
        back_budget=0,
        indices=indices,
    )

    assert sorted(collected[1]) == [10]


def test_persist_can_link_track_to_existing_target_and_mark_gap(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="vol")
    base_mask = np.ones((3, 3), dtype=bool)
    with store.mutate() as target_set:
        store.add_single_frame_target(
            target_set,
            axis="inline",
            index=5,
            mask=base_mask,
            target_type="trap",
            source="sam3_video",
            volume_id="vol",
        )

    def resolver(obj_id, frames, target_set, linkable_target_ids):  # noqa: ANN001
        existing = []
        for target_id in linkable_target_ids:
            target = target_set.targets[target_id]
            existing_frames = {
                frame.index: store.read_mask(frame.mask_ref) > 0
                for frame in target.frames.values()
                if frame.mask_ref
            }
            existing.append({"target_id": target_id, "frames": existing_frames})
        return link_targets_by_iou(existing, frames, iou_thresh=0.9)

    summary = persist_tracked_targets(
        store,
        {1: {5: base_mask, 6: base_mask}},
        seeds=[{"obj_id": 1}],
        target_axis="inline",
        target_type="trap",
        volume_id="vol",
        gap_metadata={1: {"missing": [7, 8], "trailing_gap": 2, "status_hint": "lost"}},
        link_resolver=resolver,
    )

    assert summary["target_ids"] == ["T1"]
    assert summary["obj_to_target_id"] == {"1": "T1"}
    loaded = store.load()
    assert loaded.next_seq == 2
    assert set(loaded.targets) == {"T1"}
    target = loaded.targets["T1"]
    assert target.status == TargetStatus.LOST
    assert target.source == "sam3_video"
    assert sorted(frame.index for frame in target.frames.values()) == [5, 6]
    assert target.metadata["tracking"]["last_gap"]["missing"] == [7, 8]
