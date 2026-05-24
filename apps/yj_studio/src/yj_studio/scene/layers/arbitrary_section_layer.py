from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class ArbitrarySectionLayer(Layer):
    kind: ClassVar[str] = "arbitrary_section"

    polyline: np.ndarray | None = None
    image: np.ndarray | None = None
    distances: np.ndarray | None = None
    depths: np.ndarray | None = None
    axis_label: str = "distance"
    data_path: str | None = None

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.polyline)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update({"axis_label": self.axis_label, "data_path": self.data_path})
        update_with_shape(payload, "polyline", self.polyline)
        update_with_shape(payload, "image", self.image)
        update_with_shape(payload, "distances", self.distances)
        update_with_shape(payload, "depths", self.depths)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArbitrarySectionLayer":
        return cls(
            **_base_kwargs(data),
            axis_label=str(data.get("axis_label", "distance")),
            data_path=data.get("data_path"),
        )
