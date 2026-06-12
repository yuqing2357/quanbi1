"""A set of cell IJK triples selected from a ``ReservoirGridLayer``.

Produced by:
- SAM3 mask reverse-lookup on a workbench frame (one IJK plane).
- SAM3 video propagation across a propagation axis (a 3D body).
- (future) Manual cell-picking tools.

The layer carries the cell-id list itself — it doesn't depend on the
SAM3 mask staying alive — so selections survive after the workbench
is closed. Renderers paint the listed cells with the layer's colour;
ROI clipping and any other geometry concerns are inherited from the
owning grid via the registry lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from yj_studio.scene.layer import BoundingBox, Layer, _base_kwargs

from ._helpers import EMPTY_BOUNDS


@dataclass(slots=True)
class ReservoirSelectionLayer(Layer):
    kind: ClassVar[str] = "reservoir_selection"

    grid_layer_id: str = ""        # parent ReservoirGridLayer id
    grid_id: str = ""              # ReservoirRegistry id for quick lookup
    # The selected cells as a (N, 3) int32 array of (i, j, k) triples,
    # sorted lexicographically. Always the source of truth — renderers
    # iterate over these to build their meshes.
    cell_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int32)
    )
    # Provenance: which axis the selection came from (i / j / k / None),
    # and which index range it spans. For single-frame selections lo==hi.
    source_axis: str | None = None
    source_index_lo: int | None = None
    source_index_hi: int | None = None
    # SAM3 score / confidence if applicable; informational only.
    score: float | None = None

    def __post_init__(self) -> None:
        # Normalise cell_ids on construction so equality / hashing /
        # rendering can rely on a canonical form.
        arr = np.asarray(self.cell_ids, dtype=np.int32)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"cell_ids must be (N, 3), got {arr.shape}")
        # Drop duplicates and sort for stable downstream behaviour.
        if arr.shape[0] > 0:
            arr = np.unique(arr, axis=0)
        self.cell_ids = arr

    @property
    def n_cells(self) -> int:
        return int(self.cell_ids.shape[0])

    def bounding_box(self) -> BoundingBox:
        # Selections don't carry geometry; the parent grid does. The
        # renderer pulls cell corners from the registry on demand.
        return EMPTY_BOUNDS

    def to_dict(self) -> dict[str, Any]:
        payload = Layer.to_dict(self)
        cell_payload = [] if self.metadata.get("external_cells_ref") else self.cell_ids.tolist()
        payload.update(
            {
                "grid_layer_id": self.grid_layer_id,
                "grid_id": self.grid_id,
                # Remote target selections keep the full cell list on the
                # server and only hold it in memory for display.
                "cell_ids": cell_payload,
                "source_axis": self.source_axis,
                "source_index_lo": self.source_index_lo,
                "source_index_hi": self.source_index_hi,
                "score": self.score,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReservoirSelectionLayer":
        raw = data.get("cell_ids", [])
        arr = np.asarray(raw, dtype=np.int32) if raw else np.zeros((0, 3), dtype=np.int32)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 3)
        return cls(
            **_base_kwargs(data),
            grid_layer_id=str(data.get("grid_layer_id", "")),
            grid_id=str(data.get("grid_id", "")),
            cell_ids=arr,
            source_axis=data.get("source_axis"),
            source_index_lo=data.get("source_index_lo"),
            source_index_hi=data.get("source_index_hi"),
            score=data.get("score"),
        )
