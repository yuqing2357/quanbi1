from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape

WellLogMode = Literal["por", "perm", "lith"]


@dataclass(slots=True)
class WellLogLayer(Layer):
    kind: ClassVar[str] = "well_log"

    well_name: str = ""
    mode: WellLogMode = "por"
    samples: np.ndarray | None = None
    value_column: str = ""

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.samples)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {"well_name": self.well_name, "mode": self.mode, "value_column": self.value_column}
        )
        update_with_shape(payload, "samples", self.samples)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WellLogLayer":
        return cls(
            **_base_kwargs(data),
            well_name=str(data.get("well_name", "")),
            mode=data.get("mode", "por"),
            value_column=str(data.get("value_column", "")),
        )
