"""Block-downsample a ReservoirGrid for the 3D overview render.

The full grid is too big to push through VTK as one ExplicitStructuredGrid
(140 M cells, 48 M active on our reference model). For the 3D overview
we aggregate (di, dj, dk)-sized blocks into single super-cells:

  - geometry: the super-cell's 8 corners are picked from the outermost
    corners of the block (lowest-K + lowest-IJ corner ... highest-K +
    highest-IJ corner). This preserves the corner-point grid feel —
    super-cells follow horizons and faults instead of being axis-aligned
    boxes.
  - active: OR over the block. A super-cell is shown iff at least one
    underlying cell was active.
  - integer property (e.g. LITHOLOGIES): majority vote over active
    cells. Falls back to 0 when no active cells in block.
  - float property (e.g. PORO): mean over active cells. Falls back to
    NaN when no active cells.

The default block size is (2, 2, 4) which on a 372x343x1076 grid
gives 186x172x269 = ~8.6 M super-cells, ~2.9 M active. Comfortable
for VTK.

The output ``DownsampledGrid`` is fully materialised — geometry and
all aggregated properties are computed once and held in memory. Cost
on the reference grid: ~200 MB for the super-cell corners array
(8.6M × 8 × 3 × float32) plus tens of MB per property.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .grid import ReservoirGrid

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DownsampledGrid:
    """A block-aggregated view of a ``ReservoirGrid``.

    ``shape`` is the super-cell grid size; ``block`` is the (di, dj,
    dk) block size used. ``source_shape`` keeps the original
    (nx, ny, nz) so callers can map back if needed.
    """

    source_shape: tuple[int, int, int]
    block: tuple[int, int, int]
    shape: tuple[int, int, int]
    corners: np.ndarray            # (Nx, Ny, Nz, 8, 3) float32
    active: np.ndarray             # (Nx, Ny, Nz) bool
    int_properties: dict[str, np.ndarray] = field(default_factory=dict)
    float_properties: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def total_super_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def active_super_cells(self) -> int:
        return int(self.active.sum())


# ECLIPSE cell corner slots (from io.grdecl.zcorn_cache.cell_corners):
#   0..3 = lower-K face, ordering (SW, SE, NW, NE) in IJ
#   4..7 = upper-K face, ordering (SW, SE, NW, NE) in IJ
# When aggregating a block we want the super-cell's 8 corners to be
# the spatial extremes of the block. For lower-K we use the lowest-K
# cell of the block; for upper-K we use the highest-K cell.
# Within a face we use the cell at the appropriate IJ extreme.
_BLOCK_CORNER_PICK = [
    # (slot_in_super, di_in_block, dj_in_block, dk_in_block, slot_in_source_cell)
    (0, 0, 0, 0, 0),    # super lowK-SW = source lowK-SW of (i0, j0, k0)
    (1, 1, 0, 0, 1),    # super lowK-SE = source lowK-SE of (i1, j0, k0)
    (2, 0, 1, 0, 2),    # super lowK-NW = source lowK-NW of (i0, j1, k0)
    (3, 1, 1, 0, 3),    # super lowK-NE = source lowK-NE of (i1, j1, k0)
    (4, 0, 0, 1, 4),    # super hiK-SW
    (5, 1, 0, 1, 5),
    (6, 0, 1, 1, 6),
    (7, 1, 1, 1, 7),
]


def downsample(
    grid: ReservoirGrid,
    block: tuple[int, int, int] = (2, 2, 4),
    int_properties: Iterable[str] | None = None,
    float_properties: Iterable[str] | None = None,
    progress_cb=None,
) -> DownsampledGrid:
    """Aggregate ``grid`` into super-cells.

    The output covers ``ceil(nx/di) × ceil(ny/dj) × ceil(nz/dk)`` super-
    cells; trailing partial blocks at the high-i/j/k edges are kept
    and use the available cells (super-cell's "high" corners come from
    the last real cell in the partial block).
    """

    di, dj, dk = block
    nx, ny, nz = grid.shape
    Nx = (nx + di - 1) // di
    Ny = (ny + dj - 1) // dj
    Nz = (nz + dk - 1) // dk

    logger.info(
        "Downsampling reservoir grid %s by block %s → %s (%.1fM super-cells)",
        grid.shape, block, (Nx, Ny, Nz), Nx * Ny * Nz / 1e6,
    )

    corners = np.empty((Nx, Ny, Nz, 8, 3), dtype=np.float32)
    active_super = np.zeros((Nx, Ny, Nz), dtype=bool)

    # Decide which properties to aggregate. By default, take everything
    # the grid has loaded.
    int_props = list(int_properties) if int_properties is not None else [
        name for name, arr in grid.properties.items() if np.issubdtype(arr.dtype, np.integer)
    ]
    float_props = list(float_properties) if float_properties is not None else [
        name for name, arr in grid.properties.items() if np.issubdtype(arr.dtype, np.floating)
    ]

    int_arr_out: dict[str, np.ndarray] = {
        name: np.zeros((Nx, Ny, Nz), dtype=np.int32) for name in int_props
    }
    float_arr_out: dict[str, np.ndarray] = {
        name: np.full((Nx, Ny, Nz), np.nan, dtype=np.float32) for name in float_props
    }

    # Walk K-chunk by K-chunk so we touch each ZCORN page only once.
    # A super-cell's geometry depends on cells from kk to kk+dk-1; we
    # iterate K_super and pull the matching source-K range.
    t0 = time.time()

    active_full = grid.active != 0

    for K_super in range(Nz):
        k0 = K_super * dk
        k1 = min(k0 + dk, nz)
        # Need source corners for [k0, k1). The grid's chunk cache is
        # aligned to k_chunk (default 32); a (di, dj, 4) block fits
        # comfortably inside one chunk most of the time. Just ask the
        # grid for the [k0, k1) range — the cache will dedup.
        chunk_k0, chunk_k1 = grid.chunk_for_k(k0)
        if k1 > chunk_k1:
            # rare: block straddles a chunk boundary, fetch the larger one
            chunk_k1 = grid.chunk_for_k(k1 - 1)[1]
        chunk_corners = grid.corners_for_k_chunk(chunk_k0, chunk_k1)
        # Source corners for our K slab: (nx, ny, k1-k0, 8, 3)
        src_corners = chunk_corners[:, :, k0 - chunk_k0 : k1 - chunk_k0]
        src_active = active_full[:, :, k0:k1]

        # Property slabs (same K range)
        int_slabs = {n: grid.properties[n][:, :, k0:k1] for n in int_props}
        float_slabs = {n: grid.properties[n][:, :, k0:k1] for n in float_props}

        for J_super in range(Ny):
            j0 = J_super * dj
            j1 = min(j0 + dj, ny)
            for I_super in range(Nx):
                i0 = I_super * di
                i1 = min(i0 + di, nx)

                block_active = src_active[i0:i1, j0:j1, :]
                if not block_active.any():
                    # Inactive super-cell. Still need its geometry so
                    # the corners array is well-defined; copy from the
                    # block's spatial corners anyway.
                    pass
                else:
                    active_super[I_super, J_super, K_super] = True

                # --- corners: pick the 8 extreme cells in the block
                for (super_slot, di_off, dj_off, dk_off, src_slot) in _BLOCK_CORNER_PICK:
                    src_i = min(i0 + di_off * (i1 - i0 - 1), i1 - 1)
                    src_j = min(j0 + dj_off * (j1 - j0 - 1), j1 - 1)
                    src_k_local = min(dk_off * (k1 - k0 - 1), (k1 - k0) - 1)
                    corners[I_super, J_super, K_super, super_slot] = src_corners[
                        src_i - 0, src_j - 0, src_k_local, src_slot
                    ]
                # (Note: src_i / src_j include i0/j0 already, so the
                #  index into src_corners is absolute, not block-local.
                #  We treat src_corners as covering the whole nx, ny
                #  grid which is the case.)

                # --- int properties: majority vote on active cells
                if block_active.any():
                    for name, slab in int_slabs.items():
                        vals = slab[i0:i1, j0:j1, :][block_active]
                        # bincount with a small range works fine for
                        # LITHOLOGIES which is 0..few.
                        if vals.size > 0:
                            min_v = int(vals.min())
                            shifted = (vals - min_v).astype(np.int32)
                            counts = np.bincount(shifted)
                            int_arr_out[name][I_super, J_super, K_super] = int(counts.argmax()) + min_v

                    for name, slab in float_slabs.items():
                        vals = slab[i0:i1, j0:j1, :][block_active]
                        if vals.size > 0:
                            float_arr_out[name][I_super, J_super, K_super] = float(vals.mean())

        if progress_cb is not None and (K_super % max(Nz // 20, 1) == 0):
            progress_cb((K_super + 1) / Nz, f"Downsampling K {K_super + 1}/{Nz}")

    if progress_cb is not None:
        progress_cb(1.0, "Downsampling done")

    logger.info(
        "Downsample done in %.2fs (%d active super-cells of %d)",
        time.time() - t0, int(active_super.sum()), Nx * Ny * Nz,
    )

    return DownsampledGrid(
        source_shape=grid.shape,
        block=block,
        shape=(Nx, Ny, Nz),
        corners=corners,
        active=active_super,
        int_properties=int_arr_out,
        float_properties=float_arr_out,
    )
