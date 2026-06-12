from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class LithBodyLayer(Layer):
    kind: ClassVar[str] = "lith_body"

    class_value: int = 0
    class_name: str = ""
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    data_path: str | None = None

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.vertices)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {
                "class_value": self.class_value,
                "class_name": self.class_name,
                "data_path": self.data_path,
            }
        )
        update_with_shape(payload, "vertices", self.vertices)
        update_with_shape(payload, "faces", self.faces)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LithBodyLayer":
        return cls(
            **_base_kwargs(data),
            class_value=int(data.get("class_value", 0)),
            class_name=str(data.get("class_name", "")),
            data_path=data.get("data_path"),
        )
