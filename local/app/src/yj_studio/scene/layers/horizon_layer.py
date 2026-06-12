from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import EMPTY_BOUNDS, update_with_shape


@dataclass(slots=True)
class HorizonLayer(Layer):
    kind: ClassVar[str] = "horizon"

    sample: np.ndarray | None = None
    mask: np.ndarray | None = None
    data_path: str | None = None

    def bounding_box(self) -> BoundingBox:
        if self.sample is None or self.sample.size == 0:
            return EMPTY_BOUNDS
        valid = np.isfinite(self.sample)
        if self.mask is not None:
            valid &= self.mask.astype(bool)
        if not np.any(valid):
            return EMPTY_BOUNDS
        rows, cols = np.where(valid)
        z_values = self.sample[valid]
        return (
            float(np.min(rows)),
            float(np.max(rows)),
            float(np.min(cols)),
            float(np.max(cols)),
            float(np.nanmin(z_values)),
            float(np.nanmax(z_values)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload["data_path"] = self.data_path
        update_with_shape(payload, "sample", self.sample)
        update_with_shape(payload, "mask", self.mask)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HorizonLayer":
        return cls(**_base_kwargs(data), data_path=data.get("data_path"))
