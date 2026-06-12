"""Reservoir Region-of-Interest helpers.

A ROI is just an IJK index box ``(i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)``
(half-open: ``i_lo <= i < i_hi``). It's stored on
``ReservoirGridLayer.roi`` and acts as the **single shared frame** for
every downstream operation:

- 2D section views clamp their data window and index controls to the ROI
- 3D renders only the cells inside the ROI
- SAM3 segmentation runs on rendered ROI snapshots (so pixel<->cell is
  stable across i/j/k frames — what video propagation needs)
- cell reverse-lookup is restricted to ROI cells

Two convenience APIs:

- ``default_roi(grid)`` — tight ROI around the active cells; used the
  first time a grid is loaded so users see something sensible without
  having to drag a box.
- ``roi_for_axis(roi, axis)`` — for an axis (i/j/k), returns the index
  range of the *moving* dimension (the one we slice along) and the
  index ranges of the two *fixed* dimensions (used to extract cell
  quads).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .grid import ReservoirGrid


# Half-open ``(i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)``.
ROI = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class ROIDecomposition:
    """ROI projected onto one section axis.

    ``moving_range`` is the index range of the section's slicing
    dimension (e.g. for axis='i', this is ``(i_lo, i_hi)``).
    ``fixed_i / fixed_j / fixed_k`` are the other two dimensions'
    half-open ranges, used to extract cell quads inside the ROI.
    """

    axis: str
    moving_range: tuple[int, int]
    fixed_i: tuple[int, int]
    fixed_j: tuple[int, int]
    fixed_k: tuple[int, int]


def default_roi(grid: "ReservoirGrid") -> ROI:
    """Tight ROI around all active cells. Used as the initial ROI for
    a freshly-loaded grid."""

    active = grid.active != 0
    if not active.any():
        nx, ny, nz = grid.shape
        return (0, nx, 0, ny, 0, nz)
    i_any = active.any(axis=(1, 2))
    j_any = active.any(axis=(0, 2))
    k_any = active.any(axis=(0, 1))
    i_idx = np.where(i_any)[0]
    j_idx = np.where(j_any)[0]
    k_idx = np.where(k_any)[0]
    return (
        int(i_idx.min()), int(i_idx.max()) + 1,
        int(j_idx.min()), int(j_idx.max()) + 1,
        int(k_idx.min()), int(k_idx.max()) + 1,
    )


def clamp_roi(roi: ROI, grid_shape: tuple[int, int, int]) -> ROI:
    """Clamp a ROI to grid bounds; raise if it inverts."""

    nx, ny, nz = grid_shape
    il, ih, jl, jh, kl, kh = roi
    il = max(0, min(int(il), nx))
    ih = max(il + 1, min(int(ih), nx))
    jl = max(0, min(int(jl), ny))
    jh = max(jl + 1, min(int(jh), ny))
    kl = max(0, min(int(kl), nz))
    kh = max(kl + 1, min(int(kh), nz))
    return (il, ih, jl, jh, kl, kh)


def decompose(roi: ROI, axis: str) -> ROIDecomposition:
    """Project a ROI onto one section axis."""

    il, ih, jl, jh, kl, kh = roi
    if axis == "i":
        return ROIDecomposition(
            axis="i",
            moving_range=(il, ih),
            fixed_i=(il, ih),    # unused but kept for symmetry
            fixed_j=(jl, jh),
            fixed_k=(kl, kh),
        )
    if axis == "j":
        return ROIDecomposition(
            axis="j",
            moving_range=(jl, jh),
            fixed_i=(il, ih),
            fixed_j=(jl, jh),
            fixed_k=(kl, kh),
        )
    return ROIDecomposition(
        axis="k",
        moving_range=(kl, kh),
        fixed_i=(il, ih),
        fixed_j=(jl, jh),
        fixed_k=(kl, kh),
    )


def roi_xy_bounds(
    grid: "ReservoirGrid", roi: ROI
) -> tuple[float, float, float, float]:
    """Return the local-xy bounding box of the ROI's pillar envelope.

    Used to seed 2D K-section axes. Pillars at indices [i_lo..i_hi]
    × [j_lo..j_hi] (one extra column/row over cells since pillars
    flank cells).
    """

    il, ih, jl, jh, _kl, _kh = roi
    pillars = grid.coord[il : ih + 1, jl : jh + 1, :]
    xs = np.concatenate([pillars[..., 0].ravel(), pillars[..., 3].ravel()])
    ys = np.concatenate([pillars[..., 1].ravel(), pillars[..., 4].ravel()])
    return (float(xs.min()), float(xs.max()),
            float(ys.min()), float(ys.max()))


def roi_z_bounds(
    grid: "ReservoirGrid", roi: ROI
) -> tuple[float, float]:
    """Return the local-z (depth in metres, positive-downward) range
    of the ROI envelope.

    Result is cached on the grid keyed by ROI tuple — the same ROI
    is reused for every frame in a video propagation, so we don't
    want to walk K-chunks 50× for the same answer.
    """

    cache = grid._roi_z_cache
    if roi in cache:
        return cache[roi]

    il, ih, jl, jh, kl, kh = roi
    # Read ZCORN directly — its values are pillar z-corners, which is
    # exactly what we want for a depth envelope. This avoids running
    # the full corner-point geometry rebuild (cell_corners) just to
    # extract z, saving ~18s on a typical ROI.
    # ZCORN layout: (2*nx, 2*ny, 2*nz), each cell occupies its 8
    # neighbouring corner positions. The half-open ROI in cell
    # indices maps to ZCORN ranges [2*lo : 2*hi].
    z_slab = np.asarray(
        grid.zcorn[2*il : 2*ih, 2*jl : 2*jh, 2*kl : 2*kh]
    )
    if z_slab.size == 0:
        result = (0.0, 0.0)
    else:
        result = (float(z_slab.min()), float(z_slab.max()))
    cache[roi] = result
    return result
