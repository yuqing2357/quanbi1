from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class PolygonLayer(Layer):
    kind: ClassVar[str] = "polygon"

    vertices: np.ndarray | None = None
    closed: bool = True

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.vertices)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload["closed"] = self.closed
        update_with_shape(payload, "vertices", self.vertices)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolygonLayer":
        return cls(**_base_kwargs(data), closed=bool(data.get("closed", True)))
