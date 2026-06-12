from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol
from uuid import uuid4

Color = tuple[float, float, float, float]
BoundingBox = tuple[float, float, float, float, float, float]


class LayerVisitor(Protocol):
    def visit_layer(self, layer: "Layer") -> Any: ...


@dataclass(slots=True)
class Layer:
    """Pure scene/domain object without renderer or VTK state."""

    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    color: Color = (1.0, 1.0, 1.0, 1.0)
    opacity: float = 1.0
    visible: bool = True
    locked: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "layer"

    def bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "name": self.name,
            "color": list(self.color),
            "opacity": float(self.opacity),
            "visible": bool(self.visible),
            "locked": bool(self.locked),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer":
        return cls(**_base_kwargs(data))

    def accept(self, visitor: LayerVisitor) -> Any:
        method = getattr(visitor, f"visit_{self.kind}", None)
        if method is None:
            method = visitor.visit_layer
        return method(self)


def _base_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(data.get("id") or uuid4()),
        "name": str(data["name"]),
        "color": tuple(data.get("color", (1.0, 1.0, 1.0, 1.0))),
        "opacity": float(data.get("opacity", 1.0)),
        "visible": bool(data.get("visible", True)),
        "locked": bool(data.get("locked", False)),
        "provenance": dict(data.get("provenance", {})),
        "metadata": dict(data.get("metadata", {})),
    }

