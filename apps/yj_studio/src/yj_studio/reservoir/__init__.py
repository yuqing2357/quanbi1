"""Reservoir-grid runtime (Petrel corner-point grid).

This package owns ``ReservoirGrid`` — the in-memory representation of
a corner-point reservoir model loaded from GRDECL files. Geometry
(ZCORN) is memmapped and materialised on demand, so the whole model
fits well under a gigabyte of RSS even when the source file is several
GB.

Public API:

    from yj_studio.reservoir import ReservoirGrid

    grid = ReservoirGrid.load_from_master(Path("F:/1234.GRDECL"))
    corners = grid.corners_for_k_chunk(0, 32)
    poro = grid.property("PORO")
    print(grid.active.sum(), "active cells")
"""

from .grid import ReservoirGrid
from .registry import ReservoirRegistry
from .roi import ROI, default_roi
from .seismic_mapping import SeismicIndexTransform

__all__ = [
    "ReservoirGrid",
    "ReservoirRegistry",
    "ROI",
    "default_roi",
    "SeismicIndexTransform",
]
