from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SRC = _REPO_ROOT / "server" / "src"
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))

from yj_studio_server.sam3.validation import validate_sam3_payload  # noqa: E402


def test_validate_segment_payload_accepts_reasonable_request() -> None:
    validate_sam3_payload(
        {
            "volume_id": "seismic",
            "axis": "inline",
            "index": 10,
            "keep_top_k": 3,
            "confidence": 0.5,
            "prompts": {"boxes": [[1, 2, 3, 4]], "points": [[2, 3]]},
        },
        kind="segment",
    )


def test_validate_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        validate_sam3_payload({"confidence": 1.2}, kind="segment")


def test_validate_rejects_too_many_boxes() -> None:
    with pytest.raises(ValueError, match="boxes"):
        validate_sam3_payload({"prompts": {"boxes": [[0, 0, 1, 1], [1, 1, 2, 2]]}}, kind="segment", max_boxes=1)


def test_validate_rejects_oversized_track_window() -> None:
    with pytest.raises(ValueError, match="track frame window"):
        validate_sam3_payload({"index": {"seed": 10, "back": 10, "fwd": 10}}, kind="track", max_track_frames=5)


def test_validate_batch_counts_range() -> None:
    validate_sam3_payload({"start_index": 0, "end_index": 4, "step": 2}, kind="batch", max_batch_frames=3)
    with pytest.raises(ValueError, match="batch frame count"):
        validate_sam3_payload({"indices": [1, 2, 3, 4]}, kind="batch", max_batch_frames=3)


def test_validate_infer_volume_uses_batch_frame_limits() -> None:
    validate_sam3_payload({"indices": [1, 2]}, kind="infer_volume", max_batch_frames=2)
    with pytest.raises(ValueError, match="batch frame count"):
        validate_sam3_payload({"indices": [1, 2, 3]}, kind="infer_volume", max_batch_frames=2)
