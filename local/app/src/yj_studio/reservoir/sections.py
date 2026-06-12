"""Extract Petrel-style IJK section slabs from a ReservoirGrid.

A *section* is the 2D set of cell quadrilaterals along one of the
three grid axes:

- K-section (horizontal): fix ``k``; cells are indexed by ``(i, j)``,
  drawn in the local-xy plane (top view of the layer).
- I-section: fix ``i``; cells are indexed by ``(j, k)``, drawn with
  local-y on the horizontal and sample on the vertical.
- J-section: fix ``j``; cells are indexed by ``(i, k)``, drawn with
  local-x on the horizontal and sample on the vertical.

Each section yields a flat list of cell quads + a cell-id buffer so
downstream consumers (the matplotlib renderer for now; later the
SAM3 mask reverse-lookup) can map a pixel/click back to its source
``(i, j, k)`` triple. Cell ids are encoded as a single int32 via the
``ravel_index`` helper.

A K-section only needs one K-layer's geometry, so it's cheap. An I-
or J-section needs every K-chunk in the grid (potentially all ~1076
layers); we fetch them through ``ReservoirGrid.corners_for_k_chunk``,
which is LRU-cached so subsequent sections through nearby planes are
fast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from yj_studio.io.grdecl.zcorn_cache import cell_corners

from .grid import ReservoirGrid
from .roi import ROI
from .seismic_mapping import SeismicIndexTransform

logger = logging.getLogger(__name__)

SectionAxis = Literal["i", "j", "k"]

# Drop section quads whose extent along the vertical axis exceeds
# this many units. On vertical (I/J) sections the units are sample
# index (~10 m / unit), so the default 50 ≈ 500 m kills "long spike"
# cells that fly off to the model bottom — same artefact we filter in
# the 3D renderer. On a K section the units are metres, so this
# threshold is much looser than typical cell spacing and almost never
# trips. Tune per dataset if needed.
_MAX_QUAD_EXTENT_BY_AXIS: dict[str, float] = {"i": 50.0, "j": 50.0, "k": 5000.0}


def _filter_outlier_quads(
    quads: np.ndarray,
    cell_ids: np.ndarray,
    active: np.ndarray,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop quads whose vertical extent indicates a malformed cell."""

    if quads.shape[0] == 0:
        return quads, cell_ids, active
    threshold = _MAX_QUAD_EXTENT_BY_AXIS.get(axis, 5000.0)
    ys = quads[..., 1]
    extent = ys.max(axis=1) - ys.min(axis=1)
    keep = extent <= threshold
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        logger.info(
            "Section %s: dropped %d outlier quads (vertical extent > %.0f)",
            axis, n_dropped, threshold,
        )
    return quads[keep], cell_ids[keep], active[keep]


@dataclass(slots=True)
class CellSection:
    """A 2D arrangement of cell quads on one IJK plane.

    ``quads`` are in local-xy or (axis, sample) coordinates depending
    on the section axis. ``cell_ids`` lets you reverse-look-up which
    (i, j, k) a pixel came from.
    """

    axis: SectionAxis
    fixed_index: int            # the i / j / k that was held constant
    grid_shape: tuple[int, int, int]
    # Float32 (N, 4, 2) array of quadrilateral vertices in section
    # coords. For K-sections that's (local_x, local_y); for I/J-
    # sections it's (along-axis, sample_index_neg) where the y axis
    # has been flipped so deeper sample draws lower.
    quads: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2), dtype=np.float32))
    # Same length as quads, packed (i, j, k) triples as int32.
    cell_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    # Same length as quads, the cell's active flag.
    active: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))

    @property
    def n_cells(self) -> int:
        return int(self.quads.shape[0])


def extract_k_section(
    grid: ReservoirGrid,
    k: int,
    *,
    active_only: bool = True,
) -> CellSection:
    """Return the cell quads of a fixed-K horizontal layer.

    The quads are in the grid's local-xy frame (the same frame Petrel
    stores in COORD). Each cell becomes one quad with corners ordered
    (SW, SE, NE, NW) for matplotlib PolyCollection.
    """

    nx, ny, nz = grid.shape
    if not 0 <= k < nz:
        raise IndexError(f"k={k} out of range [0, {nz})")

    layer = grid.corners_for_k_layer(k)    # (nx, ny, 8, 3)
    # Take the lower-K face (slots 0..3 in the cell_corners output)
    # for the cell's footprint; for K-sections low or high face is
    # geometrically the same in xy (only z differs).
    face = layer[:, :, 0:4, 0:2]    # (nx, ny, 4, 2) — corners SW SE NW NE
    # Reorder to SW, SE, NE, NW for matplotlib's CCW expectation.
    face = face[..., [0, 1, 3, 2], :]
    quads = face.reshape(-1, 4, 2)

    ij = np.indices((nx, ny), dtype=np.int32).reshape(2, -1).T    # (nx*ny, 2)
    cell_ids = np.empty((ij.shape[0], 3), dtype=np.int32)
    cell_ids[:, 0:2] = ij
    cell_ids[:, 2] = k

    active = (grid.active[:, :, k] != 0).ravel()

    if active_only:
        mask = active
        quads = quads[mask]
        cell_ids = cell_ids[mask]
        active = active[mask]

    return CellSection(
        axis="k",
        fixed_index=k,
        grid_shape=grid.shape,
        quads=quads.astype(np.float32, copy=False),
        cell_ids=cell_ids,
        active=active,
    )


def extract_i_section(
    grid: ReservoirGrid,
    i: int,
    *,
    transform: SeismicIndexTransform | None = None,
    active_only: bool = True,
) -> CellSection:
    """Return cell quads along a fixed-I (inline-like) slice.

    Horizontal axis: local-y at the cell centre. Vertical axis: the
    cell's z corner depths, optionally transformed to seismic sample
    indices via ``transform`` (so the section's vertical scale matches
    other 2D seismic sections).
    """

    nx, ny, nz = grid.shape
    if not 0 <= i < nx:
        raise IndexError(f"i={i} out of range [0, {nx})")

    # Compute corners only for the (1, ny) i-strip we need, all K in
    # one pass. Feed the ZCORN memmap directly — cell_corners' ix_
    # fancy-indexing only touches the z-values for our strip, so the
    # OS pages in tens of MB instead of materialising the full 4 GB.
    sub = cell_corners(
        grid.zcorn, grid.coord, grid.spec,
        k_offset=0,
        i_range=(i, i + 1),
        j_range=(0, ny),
    )    # (1, ny, nz, 8, 3)
    sub = sub[0]    # (ny, nz, 8, 3)

    # Quad ordering CCW: lowK-S, lowK-N, hiK-N, hiK-S.
    # Average the SE/SW (or NE/NW) corner pairs that straddle the i
    # direction since we collapsed it.
    lowS = 0.5 * (sub[..., 0, 1:3] + sub[..., 1, 1:3])    # (ny, nz, 2) (y, z)
    lowN = 0.5 * (sub[..., 2, 1:3] + sub[..., 3, 1:3])
    hiN = 0.5 * (sub[..., 6, 1:3] + sub[..., 7, 1:3])
    hiS = 0.5 * (sub[..., 4, 1:3] + sub[..., 5, 1:3])
    quad = np.stack([lowS, lowN, hiN, hiS], axis=2)    # (ny, nz, 4, 2)

    jk = np.indices((ny, nz), dtype=np.int32).reshape(2, -1).T
    cell_ids = np.empty((jk.shape[0], 3), dtype=np.int32)
    cell_ids[:, 0] = i
    cell_ids[:, 1] = jk[:, 0]
    cell_ids[:, 2] = jk[:, 1]

    active = (grid.active[i, :, :] != 0).ravel()
    quads = quad.reshape(-1, 4, 2)

    if active_only:
        m = active
        quads = quads[m]; cell_ids = cell_ids[m]; active = active[m]

    # Convert the z column (vertical axis) to sample index and flip
    # sign so deeper draws lower in the matplotlib axes.
    if transform is not None and quads.size > 0:
        quads[..., 1] = quads[..., 1] / transform.z_step
    if quads.size > 0:
        quads[..., 1] = -quads[..., 1]

    quads, cell_ids, active = _filter_outlier_quads(quads, cell_ids, active, "i")

    return CellSection(
        axis="i",
        fixed_index=i,
        grid_shape=grid.shape,
        quads=quads.astype(np.float32, copy=False),
        cell_ids=cell_ids,
        active=active,
    )


def extract_j_section(
    grid: ReservoirGrid,
    j: int,
    *,
    transform: SeismicIndexTransform | None = None,
    active_only: bool = True,
) -> CellSection:
    """Return cell quads along a fixed-J slice. Mirror of ``extract_i_section``."""

    nx, ny, nz = grid.shape
    if not 0 <= j < ny:
        raise IndexError(f"j={j} out of range [0, {ny})")

    # Compute corners only for the (nx, 1) j-strip; see extract_i_section
    # for the rationale (avoid spending K-chunk time on cells we discard).
    sub = cell_corners(
        grid.zcorn, grid.coord, grid.spec,
        k_offset=0,
        i_range=(0, nx),
        j_range=(j, j + 1),
    )    # (nx, 1, nz, 8, 3)
    sub = sub[:, 0, :, :, :]    # (nx, nz, 8, 3)

    # On a J-section we average the SW/NW pair (low-i side) and the
    # SE/NE pair (high-i side) to collapse the y dimension.
    lowW = 0.5 * (sub[..., 0, ::2] + sub[..., 2, ::2])    # take cols 0,2 = (x, z)
    lowE = 0.5 * (sub[..., 1, ::2] + sub[..., 3, ::2])
    hiE = 0.5 * (sub[..., 5, ::2] + sub[..., 7, ::2])
    hiW = 0.5 * (sub[..., 4, ::2] + sub[..., 6, ::2])
    # Quad ordering CCW: lowW, lowE, hiE, hiW
    quad = np.stack([lowW, lowE, hiE, hiW], axis=2)    # (nx, nz, 4, 2)

    ik = np.indices((nx, nz), dtype=np.int32).reshape(2, -1).T
    cell_ids = np.empty((ik.shape[0], 3), dtype=np.int32)
    cell_ids[:, 0] = ik[:, 0]
    cell_ids[:, 1] = j
    cell_ids[:, 2] = ik[:, 1]

    active = (grid.active[:, j, :] != 0).ravel()
    quads = quad.reshape(-1, 4, 2)

    if active_only:
        m = active
        quads = quads[m]; cell_ids = cell_ids[m]; active = active[m]

    if transform is not None and quads.size > 0:
        quads[..., 1] = quads[..., 1] / transform.z_step
    if quads.size > 0:
        quads[..., 1] = -quads[..., 1]

    quads, cell_ids, active = _filter_outlier_quads(quads, cell_ids, active, "j")

    return CellSection(
        axis="j",
        fixed_index=j,
        grid_shape=grid.shape,
        quads=quads.astype(np.float32, copy=False),
        cell_ids=cell_ids,
        active=active,
    )


def clip_to_roi(section: CellSection, roi: ROI) -> CellSection:
    """Return a copy of ``section`` keeping only cells inside the ROI.

    Cheap (~one boolean mask) and lets the section view + SAM3 use
    the same per-cell coordinate space as before, just shorter. The
    grouping into a separate step (vs. building ROI awareness into
    every extract_* function) keeps the extract code straightforward.
    """

    if section.n_cells == 0:
        return section
    il, ih, jl, jh, kl, kh = roi
    ids = section.cell_ids
    mask = (
        (ids[:, 0] >= il) & (ids[:, 0] < ih)
        & (ids[:, 1] >= jl) & (ids[:, 1] < jh)
        & (ids[:, 2] >= kl) & (ids[:, 2] < kh)
    )
    if mask.all():
        return section
    return CellSection(
        axis=section.axis,
        fixed_index=section.fixed_index,
        grid_shape=section.grid_shape,
        quads=section.quads[mask],
        cell_ids=ids[mask],
        active=section.active[mask],
    )


def values_for_section(
    section: CellSection, property_array: np.ndarray
) -> np.ndarray:
    """Look up a per-cell property at the section's cell ids."""

    if section.n_cells == 0:
        return np.empty((0,), dtype=property_array.dtype)
    i = section.cell_ids[:, 0]
    j = section.cell_ids[:, 1]
    k = section.cell_ids[:, 2]
    return property_array[i, j, k]
