"""End-to-end pipeline test for SAM3SegmentAlgorithm using a fake processor.

We never load the real SAM3 model in CI / on developer machines — the
``Sam3Processor`` is duck-typed so a stub object with ``set_image``,
``set_text_prompt``, ``add_geometric_prompt``, ``set_confidence_threshold``
methods is enough to validate the data flow through the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yj_studio.algorithms import AlgorithmRunner
from yj_studio.algorithms.builtin.ai.sam3_segment import SAM3SegmentAlgorithm
from yj_studio.scene.layers import MaskLayer, VolumeLayer


@dataclass
class _FakeAIService:
    image_processor: Any
    state_value: str = "ready"

    def is_ready(self) -> bool:
        return True

    @property
    def state(self):
        class _State:
            value = "ready"

        return _State()

    def mark_busy(self, _message: str = "") -> None:
        pass

    def mark_ready(self, _message: str = "") -> None:
        pass


class _FakeProcessor:
    """Captures inputs and returns a deterministic boolean mask."""

    def __init__(self, mask_shape: tuple[int, int]) -> None:
        self.mask_shape = mask_shape
        self.calls: list[tuple[str, Any]] = []

    def set_confidence_threshold(self, threshold: float, state=None):  # noqa: D401
        self.calls.append(("set_confidence_threshold", threshold))
        return state

    def set_image(self, image, state=None):
        self.calls.append(("set_image", (getattr(image, "size", None))))
        return {"backbone_out": {}, "geometric_prompt": object()}

    def set_text_prompt(self, prompt: str, state):
        self.calls.append(("set_text_prompt", prompt))
        state["text"] = prompt
        return self._with_results(state)

    def add_geometric_prompt(self, box, label, state):
        self.calls.append(("add_geometric_prompt", (tuple(box), label)))
        return self._with_results(state)

    def _with_results(self, state):
        h, w = self.mask_shape
        mask = np.zeros((1, 1, h, w), dtype=bool)
        mask[0, 0, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        state["masks"] = mask
        state["scores"] = np.array([0.7], dtype=np.float32)
        state["boxes"] = np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=np.float32)
        return state


class _FakeVolumeStore:
    """Tiny VolumeStore-shaped object returning fixed slices."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.shape = shape
        self.calls: list[tuple[str, str, int]] = []

    def get_slice(self, volume_id: str, axis: str, index: int) -> np.ndarray:
        self.calls.append((volume_id, axis, index))
        nx, ny, nz = self.shape
        if axis == "inline":
            return np.linspace(-1.0, 1.0, num=ny * nz, dtype=np.float32).reshape(ny, nz)
        if axis == "xline":
            return np.linspace(-1.0, 1.0, num=nx * nz, dtype=np.float32).reshape(nx, nz)
        return np.linspace(-1.0, 1.0, num=nx * ny, dtype=np.float32).reshape(nx, ny)


def test_sam3_segment_pipeline_with_fakes(qapp) -> None:
    shape = (32, 24, 16)
    # SAM3 preprocesses to a fixed resolution; for the fake we use the
    # transposed slice shape (z, ny) for inline = (16, 24).
    processor = _FakeProcessor(mask_shape=(16, 24))
    volume_store = _FakeVolumeStore(shape)
    ai_service = _FakeAIService(image_processor=processor)

    runner = AlgorithmRunner()
    volume_layer = VolumeLayer(name="seismic", volume_id="seismic", shape=shape)
    result = runner.run_sync(
        SAM3SegmentAlgorithm,
        params={
            "axis": "inline",
            "slice_index": 10,
            "text_prompt": "salt body",
            "boxes": [(2.0, 3.0, 10.0, 11.0)],
            "points": [(5.0, 5.0)],
            "confidence_threshold": 0.3,
            "keep_top_k": 5,
            "name_prefix": "TEST",
        },
        input_layers={"volume": volume_layer},
        services={"ai_service": ai_service, "volume_store": volume_store},
    )

    assert result.ok, result.error
    assert len(result.output_layers) == 1
    layer = result.output_layers[0]
    assert isinstance(layer, MaskLayer)
    assert layer.axis == "inline"
    assert layer.slice_index == 10
    assert layer.confidence == 0.7
    # Processor was called with the expected sequence
    calls = [c[0] for c in processor.calls]
    assert calls.count("set_image") == 1
    assert calls.count("set_text_prompt") == 1
    # 1 user box + 1 point-as-box prompt
    assert calls.count("add_geometric_prompt") == 2
    # Volume store was hit once with the requested slice
    assert volume_store.calls == [("seismic", "inline", 10)]


def test_sam3_segment_requires_services(qapp) -> None:
    runner = AlgorithmRunner()
    volume_layer = VolumeLayer(name="seismic", volume_id="seismic", shape=(4, 5, 6))
    result = runner.run_sync(
        SAM3SegmentAlgorithm,
        params={"axis": "inline", "slice_index": 0, "text_prompt": "salt"},
        input_layers={"volume": volume_layer},
        services={},
    )
    assert not result.ok
    assert "service" in (result.error or "").lower()
