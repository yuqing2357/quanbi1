from __future__ import annotations

import numpy as np

from yj_studio.ui.dialogs.target_visualization_dialog import (
    load_tracking_frames,
    mask3d_to_world,
    mask3d_world_origin,
    ordered_target_frames,
)
from yj_studio_core.targets import GeoTarget, TargetFrame


def test_ordered_target_frames_sorts_dominant_axis_by_index() -> None:
    target = GeoTarget(id="T1")
    target.add_frame(TargetFrame(axis="inline", index=9))
    target.add_frame(TargetFrame(axis="inline", index=7))
    target.add_frame(TargetFrame(axis="crossline", index=20))
    target.trajectory = ["inline:7", "inline:9", "crossline:20"]

    frames = ordered_target_frames(target)

    assert [(frame.axis, frame.index) for frame in frames] == [
        ("inline", 7),
        ("inline", 9),
    ]


def test_load_tracking_frames_keeps_server_image_orientation() -> None:
    target = GeoTarget(id="T1", volume_id="model_lithology")
    target.add_frame(TargetFrame(axis="inline", index=10))
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0, 3] = 1
    raw_slice = np.zeros((4, 3), dtype=np.float32)
    raw_slice[3, 0] = 1

    class TargetStore:
        def fetch_mask(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            return mask

    class VolumeStore:
        def get_slice(self, volume_id, axis, index):  # noqa: ANN001, ARG002
            assert axis == "inline"
            assert index == 10
            return raw_slice

    rows = load_tracking_frames(target, TargetStore(), VolumeStore())

    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0].mask, mask.astype(bool))
    assert rows[0].image[0, 3].tolist() == [255, 221, 0]


def test_load_tracking_frames_includes_missing_indices_as_empty_masks() -> None:
    target = GeoTarget(
        id="T1",
        volume_id="model_lithology",
        metadata={"tracking": {"last_gap": {"missing": [9, 11]}}},
    )
    target.add_frame(TargetFrame(axis="inline", index=10))

    class TargetStore:
        def fetch_mask(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            return np.ones((2, 3), dtype=np.uint8)

    rows = load_tracking_frames(target, TargetStore(), None)

    assert [row.frame.index for row in rows] == [9, 10, 11]
    assert [row.frame.origin for row in rows] == ["missing", "sam3", "missing"]
    assert [int(row.mask.sum()) for row in rows] == [0, 6, 0]


def test_mask3d_world_mapping_and_crop_origins_cover_all_axes() -> None:
    raw = np.zeros((2, 3, 4), dtype=bool)
    raw[1, 2, 3] = True

    assert np.argwhere(mask3d_to_world(raw, "inline")).tolist() == [[1, 3, 2]]
    assert np.argwhere(mask3d_to_world(raw, "crossline")).tolist() == [[3, 1, 2]]
    assert np.argwhere(mask3d_to_world(raw, "timeslice")).tolist() == [[3, 2, 1]]

    assert mask3d_world_origin("inline", 100, (1, 2, 3)) == (101, 3, 2)
    assert mask3d_world_origin("crossline", 100, (1, 2, 3)) == (3, 101, 2)
    assert mask3d_world_origin("timeslice", 100, (1, 2, 3)) == (3, 2, 101)
