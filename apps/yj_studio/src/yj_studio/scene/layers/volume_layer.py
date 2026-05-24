from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs


ROIBox = tuple[int, int, int, int, int, int]
"""ROI clipping box: (i_min, i_max, j_min, j_max, k_min, k_max), inclusive both ends."""


@dataclass(slots=True)
class VolumeLayer(Layer):
    kind: ClassVar[str] = "volume"

    volume_id: str = ""
    shape: tuple[int, int, int] | None = None
    clim: tuple[float, float] | None = None
    cmap: str = "Petrel"
    slice_indices: dict[str, int] = field(default_factory=dict)
    roi: ROIBox | None = None

    def bounding_box(self) -> BoundingBox:
        if self.shape is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            0.0,
            float(self.shape[0] - 1),
            0.0,
            float(self.shape[1] - 1),
            0.0,
            float(self.shape[2] - 1),
        )

    def effective_roi(self) -> ROIBox | None:
        """Return the ROI clamped against ``shape``, or None if it covers everything."""

        if self.shape is None:
            return self.roi
        if self.roi is None:
            return None
        nx, ny, nz = self.shape
        i0, i1, j0, j1, k0, k1 = self.roi
        i0 = max(0, min(int(i0), nx - 1))
        i1 = max(i0, min(int(i1), nx - 1))
        j0 = max(0, min(int(j0), ny - 1))
        j1 = max(j0, min(int(j1), ny - 1))
        k0 = max(0, min(int(k0), nz - 1))
        k1 = max(k0, min(int(k1), nz - 1))
        if (i0, i1, j0, j1, k0, k1) == (0, nx - 1, 0, ny - 1, 0, nz - 1):
            return None
        return (i0, i1, j0, j1, k0, k1)

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        payload.update(
            {
                "volume_id": self.volume_id,
                "shape": list(self.shape) if self.shape is not None else None,
                "clim": list(self.clim) if self.clim is not None else None,
                "cmap": self.cmap,
                "slice_indices": dict(self.slice_indices),
                "roi": list(self.roi) if self.roi is not None else None,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VolumeLayer":
        shape = data.get("shape")
        clim = data.get("clim")
        roi = data.get("roi")
        return cls(
            **_base_kwargs(data),
            volume_id=str(data.get("volume_id", "")),
            shape=tuple(shape) if shape is not None else None,
            clim=tuple(clim) if clim is not None else None,
            cmap=str(data.get("cmap", "Petrel")),
            slice_indices=dict(data.get("slice_indices", {})),
            roi=tuple(int(v) for v in roi) if roi is not None else None,
        )
