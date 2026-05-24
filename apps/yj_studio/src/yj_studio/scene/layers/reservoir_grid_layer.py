"""Layer pointing at a loaded reservoir grid (corner-point Petrel grid).

The grid itself — geometry, ACTNUM, ZCORN memmap, cached corners,
property arrays — is too big and too live-state to serialise into a
slotted dataclass. We hold an opaque ``grid_id`` string instead and
keep the real data in a ``ReservoirRegistry`` service (see
``yj_studio.reservoir.registry``).

A scene typically carries:

- one ``ReservoirGridLayer`` per loaded model (defines whether to
  show the grid at all, whether to draw the wireframe, whether to use
  the downsampled overview vs. fine geometry, etc.)
- zero or more ``ReservoirPropertyLayer`` referencing the same
  ``grid_id`` — each picks one cell property (LITHOLOGIES, PORO, ...)
  and a colormap, and rides on top of the grid for shading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import EMPTY_BOUNDS


@dataclass(slots=True)
class ReservoirGridLayer(Layer):
    kind: ClassVar[str] = "reservoir_grid"

    grid_id: str = ""
    master_path: str = ""
    shape: tuple[int, int, int] | None = None
    bounds: BoundingBox = field(default_factory=lambda: EMPTY_BOUNDS)

    show_wireframe: bool = True
    show_inactive: bool = False
    # When True the 3D overview renders from the (2,2,4)-downsampled
    # super-grid; when False it uses fine-resolution chunks (slower,
    # but exact). IJK slice rendering always uses fine resolution
    # regardless of this flag.
    use_downsampled: bool = True
    # If set, the renderer skips cells outside this index box —
    # analogous to VolumeLayer.roi but in cell index space.
    roi: tuple[int, int, int, int, int, int] | None = None

    def bounding_box(self) -> BoundingBox:
        return self.bounds

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {
                "grid_id": self.grid_id,
                "master_path": self.master_path,
                "shape": list(self.shape) if self.shape is not None else None,
                "bounds": list(self.bounds),
                "show_wireframe": self.show_wireframe,
                "show_inactive": self.show_inactive,
                "use_downsampled": self.use_downsampled,
                "roi": list(self.roi) if self.roi is not None else None,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReservoirGridLayer":
        shape = data.get("shape")
        bounds = data.get("bounds")
        roi = data.get("roi")
        return cls(
            **_base_kwargs(data),
            grid_id=str(data.get("grid_id", "")),
            master_path=str(data.get("master_path", "")),
            shape=tuple(int(v) for v in shape) if shape is not None else None,
            bounds=tuple(float(v) for v in bounds) if bounds is not None else EMPTY_BOUNDS,
            show_wireframe=bool(data.get("show_wireframe", True)),
            show_inactive=bool(data.get("show_inactive", False)),
            use_downsampled=bool(data.get("use_downsampled", True)),
            roi=tuple(int(v) for v in roi) if roi is not None else None,
        )
