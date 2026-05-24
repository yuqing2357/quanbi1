"""Mock-based pipeline test for SAM3PropagateAlgorithm.

We stub both the AIService (provides a fake video predictor) and the
volume store (returns synthetic slices). The fake predictor's
``init_state`` + ``add_prompt`` + ``model.propagate_in_video`` mirror the
real SAM3 contract enough that the algorithm's stitching logic is
exercised end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pytest

# Try to import PIL early; the propagation algorithm exports JPEGs via
# Pillow. If Pillow is missing the whole test module is skipped.
PIL = pytest.importorskip("PIL.Image")

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.ai.sam3_propagate import SAM3PropagateAlgorithm
from yj_studio.scene.layers import MaskLayer, VolumeLayer
from tests.test_sam3_segment_algorithm import _FakeVolumeStore


class _FakeVideoModel:
    """Yields a fixed boolean mask + score for every frame."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape

    def propagate_in_video(
        self,
        *,
        inference_state,
        start_frame_idx: int,
        max_frame_num_to_track: int,
        reverse: bool,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        step = -1 if reverse else 1
        seen_seed = False
        for i in range(max_frame_num_to_track + 1):
            frame_idx = start_frame_idx + i * step
            if frame_idx < 0:
                break
            mask = np.zeros(self.shape, dtype=bool)
            h, w = self.shape
            mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
            score = 0.9 if not seen_seed else 0.7
            seen_seed = True
            yield frame_idx, {1: {"masks": mask[None, None], "scores": np.array([score])}}


class _FakeVideoPredictor:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.model = _FakeVideoModel(shape)
        self.calls: list[tuple[str, Any]] = []

    def init_state(self, *, resource_path: str, async_loading_frames: bool, video_loader_type: str):
        self.calls.append(("init_state", resource_path))
        return {"resource": resource_path}

    def add_prompt(self, *, inference_state, frame_idx, text_str, points, point_labels,
                   boxes_xywh, box_labels, obj_id):
        self.calls.append(("add_prompt", (frame_idx, text_str, boxes_xywh, box_labels, obj_id)))
        return frame_idx, None


@dataclass
class _FakeAIServiceVideo:
    video_predictor: _FakeVideoPredictor

    def is_ready(self) -> bool:
        return True

    @property
    def state(self):
        class _S:
            value = "ready"

        return _S()

    def mark_busy(self, _m: str = "") -> None:
        pass

    def mark_ready(self, _m: str = "") -> None:
        pass


def test_propagate_stitches_3d_mask(qapp, tmp_path) -> None:
    nx, ny, nz = 16, 12, 8
    shape = (nx, ny, nz)
    volume_store = _FakeVolumeStore(shape)
    # Each exported JPEG is the transposed inline slice → (nz, ny)
    predictor = _FakeVideoPredictor(shape=(nz, ny))
    ai_service = _FakeAIServiceVideo(video_predictor=predictor)

    seed_mask = np.zeros((ny, nz), dtype=bool)
    seed_mask[3:6, 2:5] = True
    seed_layer = MaskLayer(
        name="seed",
        axis="inline",
        slice_index=5,
        mask=seed_mask,
    )
    volume_layer = VolumeLayer(name="seismic", volume_id="seismic", shape=shape)

    runner = AlgorithmRunner()
    result = runner.run_sync(
        SAM3PropagateAlgorithm,
        params={
            "forward_steps": 2,
            "backward_steps": 2,
            "text_prompt": "",
            "confidence_threshold": 0.1,
            "name_prefix": "Prop",
            "drop_low_confidence_frames": False,
        },
        input_layers={"volume": volume_layer, "seed_mask": seed_layer},
        services={
            "ai_service": ai_service,
            "volume_store": volume_store,
        },
    )

    assert result.ok, result.error
    layers = result.output_layers
    assert len(layers) == 1
    layer = layers[0]
    assert layer.mask is not None
    assert layer.mask.ndim == 3
    # 2 backward + seed + 2 forward = 5 frames
    assert layer.mask.shape[0] == 5
    assert layer.axis == "inline"
    assert layer.slice_index == 5

    init_calls = [c for c in predictor.calls if c[0] == "init_state"]
    add_calls = [c for c in predictor.calls if c[0] == "add_prompt"]
    assert len(init_calls) == 1
    assert len(add_calls) == 1
