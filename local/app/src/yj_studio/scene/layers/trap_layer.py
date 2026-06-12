from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class TrapLayer(Layer):
    kind: ClassVar[str] = "trap"

    boundary: np.ndarray | None = None
    score: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.boundary)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update({"score": self.score, "attributes": dict(self.attributes)})
        update_with_shape(payload, "boundary", self.boundary)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrapLayer":
        return cls(
            **_base_kwargs(data),
            score=data.get("score"),
            attributes=dict(data.get("attributes", {})),
        )
