from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import EMPTY_BOUNDS, update_with_shape


@dataclass(slots=True)
class MaskLayer(Layer):
    kind: ClassVar[str] = "mask"

    mask: np.ndarray | None = None
    confidence: np.ndarray | float | None = None
    axis: str | None = None
    slice_index: int | None = None
    data_path: str | None = None

    def bounding_box(self) -> BoundingBox:
        if self.mask is None or self.mask.size == 0:
            return EMPTY_BOUNDS
        coords = np.argwhere(np.asarray(self.mask) > 0)
        if coords.size == 0:
            return EMPTY_BOUNDS
        while coords.shape[1] < 3:
            coords = np.column_stack([coords, np.zeros(coords.shape[0])])
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {
                "axis": self.axis,
                "slice_index": self.slice_index,
                "data_path": self.data_path,
                "confidence_is_scalar": isinstance(self.confidence, float),
                "confidence": self.confidence if isinstance(self.confidence, float) else None,
            }
        )
        update_with_shape(payload, "mask", self.mask)
        if isinstance(self.confidence, np.ndarray):
            update_with_shape(payload, "confidence", self.confidence)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaskLayer":
        return cls(
            **_base_kwargs(data),
            axis=data.get("axis"),
            slice_index=data.get("slice_index"),
            data_path=data.get("data_path"),
            confidence=data.get("confidence"),
        )
