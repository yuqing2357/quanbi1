from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class WellLayer(Layer):
    kind: ClassVar[str] = "well"

    trajectory: np.ndarray | None = None
    head_position: tuple[float, float, float] | None = None
    well_name: str = ""

    def bounding_box(self) -> BoundingBox:
        if self.trajectory is not None:
            return bbox_from_points(self.trajectory)
        if self.head_position is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        x, y, z = self.head_position
        return (x, x, y, y, z, z)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {
                "well_name": self.well_name,
                "head_position": list(self.head_position) if self.head_position else None,
            }
        )
        update_with_shape(payload, "trajectory", self.trajectory)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WellLayer":
        head = data.get("head_position")
        return cls(
            **_base_kwargs(data),
            well_name=str(data.get("well_name", "")),
            head_position=tuple(head) if head is not None else None,
        )
