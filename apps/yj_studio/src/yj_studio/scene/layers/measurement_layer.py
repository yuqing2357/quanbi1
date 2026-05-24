from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import bbox_from_points, update_with_shape


@dataclass(slots=True)
class MeasurementLayer(Layer):
    kind: ClassVar[str] = "measurement"

    geometry: np.ndarray | None = None
    values: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)

    def bounding_box(self) -> BoundingBox:
        return bbox_from_points(self.geometry)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update({"values": dict(self.values), "units": dict(self.units)})
        update_with_shape(payload, "geometry", self.geometry)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementLayer":
        # ``geometry`` is restored separately by the cross-process layer
        # payload (see algorithms.serialization); to_dict only carries its
        # shape so projects can reference an external .npy. Both code paths
        # later set ``geometry`` if a real array exists.
        return cls(
            **_base_kwargs(data),
            values=dict(data.get("values", {})),
            units=dict(data.get("units", {})),
        )
