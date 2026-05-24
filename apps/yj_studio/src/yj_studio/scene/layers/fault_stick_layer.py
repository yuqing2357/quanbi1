from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class FaultStickLayer(Layer):
    kind: ClassVar[str] = "fault_stick"

    sticks: np.ndarray | None = None

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.sticks)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        update_with_shape(payload, "sticks", self.sticks)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FaultStickLayer":
        return cls(**_base_kwargs(data))
