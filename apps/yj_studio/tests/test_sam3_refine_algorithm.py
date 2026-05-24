from __future__ import annotations

from typing import Any

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.ai.sam3_refine import SAM3RefineAlgorithm
from yj_studio.scene.layers import MaskLayer, VolumeLayer
from tests.test_sam3_segment_algorithm import _FakeAIService, _FakeProcessor, _FakeVolumeStore


def test_refine_uses_mask_bbox_and_recovers_text_prompt(qapp) -> None:
    shape = (32, 24, 16)
    processor = _FakeProcessor(mask_shape=(16, 24))
    volume_store = _FakeVolumeStore(shape)
    ai_service = _FakeAIService(image_processor=processor)

    seed = MaskLayer(
        name="edited",
        axis="inline",
        slice_index=12,
        mask=_corner_mask(16, 24),
        metadata={"text_prompt": "channel sand"},
    )
    volume_layer = VolumeLayer(name="seismic", volume_id="seismic", shape=shape)

    runner = AlgorithmRunner()
    result = runner.run_sync(
        SAM3RefineAlgorithm,
        params={
            "text_prompt_override": "",
            "confidence_threshold": 0.3,
            "keep_top_k": 1,
            "pad_box_px": 0.0,
        },
        input_layers={"volume": volume_layer, "edited_mask": seed},
        services={"ai_service": ai_service, "volume_store": volume_store},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    layer = result.output_layers[0]
    assert layer.axis == "inline"
    assert layer.slice_index == 12
    assert layer.metadata["refined_from"] == seed.id
    assert layer.provenance["source"] == "ai.sam3.refine"

    # Refine should reuse the seed's text_prompt from metadata.
    text_calls = [args for kind, args in processor.calls if kind == "set_text_prompt"]
    assert text_calls == ["channel sand"]
    # And it should add exactly one box (the derived bbox).
    box_calls = [args for kind, args in processor.calls if kind == "add_geometric_prompt"]
    assert len(box_calls) == 1


def test_refine_rejects_empty_mask(qapp) -> None:
    shape = (8, 8, 8)
    processor = _FakeProcessor(mask_shape=(8, 8))
    volume_store = _FakeVolumeStore(shape)
    ai_service = _FakeAIService(image_processor=processor)

    seed = MaskLayer(
        name="empty",
        axis="inline",
        slice_index=0,
        mask=np.zeros((8, 8), dtype=bool),
    )
    volume_layer = VolumeLayer(name="seismic", volume_id="seismic", shape=shape)

    runner = AlgorithmRunner()
    result = runner.run_sync(
        SAM3RefineAlgorithm,
        params={},
        input_layers={"volume": volume_layer, "edited_mask": seed},
        services={"ai_service": ai_service, "volume_store": volume_store},
    )
    assert not result.ok
    assert "empty" in (result.error or "").lower()


def _corner_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[1:4, 1:4] = True
    return mask
