from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs


@dataclass(slots=True)
class AnnotationLayer(Layer):
    kind: ClassVar[str] = "annotation"

    items: list[dict[str, Any]] = field(default_factory=list)

    def bounding_box(self) -> BoundingBox:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload["items"] = list(self.items)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationLayer":
        return cls(**_base_kwargs(data), items=list(data.get("items", [])))
