from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class FaultSurfaceLayer(Layer):
    kind: ClassVar[str] = "fault_surface"

    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    data_path: str | None = None

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.vertices)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload["data_path"] = self.data_path
        update_with_shape(payload, "vertices", self.vertices)
        update_with_shape(payload, "faces", self.faces)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FaultSurfaceLayer":
        return cls(**_base_kwargs(data), data_path=data.get("data_path"))
