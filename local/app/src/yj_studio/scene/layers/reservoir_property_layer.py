"""Property-coloring layer riding on top of a ``ReservoirGridLayer``.

Each instance picks one property keyword (LITHOLOGIES, PORO, ...) from
the grid and binds it to a colormap + value range. Multiple
``ReservoirPropertyLayer`` instances may share one grid — toggling
their ``visible`` flag in the layer tree is how the user switches
between displays without re-loading the grid.

The actual property array lives on the ``ReservoirGrid`` in the
registry. We carry only the metadata the renderer needs to look it
up and shade it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import EMPTY_BOUNDS


@dataclass(slots=True)
class ReservoirPropertyLayer(Layer):
    kind: ClassVar[str] = "reservoir_property"

    grid_layer_id: str = ""        # id of the owning ReservoirGridLayer
    grid_id: str = ""              # ReservoirRegistry id for quick lookup
    property_name: str = ""        # e.g. "LITHOLOGIES" or "PORO"
    is_integer: bool = False       # True → categorical cmap; False → continuous
    cmap: str = "viridis"
    clim: tuple[float, float] | None = None
    # When True the renderer hides cells whose property value matches
    # one of these. Useful for hiding "lith=0 = unfilled/background"
    # categories without touching the underlying grid.
    hidden_values: tuple[float, ...] = ()

    def bounding_box(self) -> BoundingBox:
        # Property layers don't carry geometry — they piggyback on the
        # grid layer. Returning EMPTY_BOUNDS keeps them from inflating
        # the scene bbox; callers that want a real bbox should query
        # the parent grid layer.
        return EMPTY_BOUNDS

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {
                "grid_layer_id": self.grid_layer_id,
                "grid_id": self.grid_id,
                "property_name": self.property_name,
                "is_integer": self.is_integer,
                "cmap": self.cmap,
                "clim": list(self.clim) if self.clim is not None else None,
                "hidden_values": list(self.hidden_values),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReservoirPropertyLayer":
        clim = data.get("clim")
        return cls(
            **_base_kwargs(data),
            grid_layer_id=str(data.get("grid_layer_id", "")),
            grid_id=str(data.get("grid_id", "")),
            property_name=str(data.get("property_name", "")),
            is_integer=bool(data.get("is_integer", False)),
            cmap=str(data.get("cmap", "viridis")),
            clim=tuple(float(v) for v in clim) if clim is not None else None,
            hidden_values=tuple(float(v) for v in data.get("hidden_values", ())),
        )
