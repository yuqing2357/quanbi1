"""Render reservoir 2D sections on demand from native corner-point columns.

Pure compute (numpy + scipy.spatial.cKDTree); no Qt, no FastAPI, no GRDECL at
runtime. Inputs are the native column arrays plus the offline precompute
(``column_centers_axis.npy`` / ``column_valid.npy`` / ``column_geometry.json``
written by tools/precompute_reservoir_columns.py).

Sections are produced in the seismic-axis frame so they co-register 1:1 with the
dense seismic volume: lateral steps are whole seismic indices (12.5 m), and the
section position ``index`` is a seismic axis index. The vertical axis is sampled
in metres (default 2 m — finer than seismic's 10 m, ~native cell size).

Sampling rules (decided 2026-06-18):
  * lithology -> interval aggregation: a depth band is "target" if ANY native
    target cell (class 1/2) overlaps it (preserves thin sands; point sampling
    drops ~half of sub-metre beds).
  * porosity  -> nearest native z_center (~one native cell per 2 m band).
  * lateral   -> nearest native column (Voronoi); columns are ~50 m so 12.5 m
    lateral sampling replicates each column ~4x (intentional, co-registers with
    seismic — no lateral information gain).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INLINE = 0  # section at fixed axis0, lateral varies along axis1
XLINE = 1   # section at fixed axis1, lateral varies along axis0

# native lithology encoding: 0 = background (gravel, zero-porosity), 1/2 = target
# (sandstone/mud, porous); negative = null.
_TARGET_CLASSES = (1, 2)


@dataclass(frozen=True)
class ReservoirGeometry:
    nx: int
    ny: int
    nz: int
    axis0_spacing_m: float
    axis1_spacing_m: float
    sample_spacing_m: float


@dataclass
class SectionResult:
    """A rendered section. ``values`` is (n_depth, n_lateral); row 0 = shallow."""
    values: np.ndarray          # lithology uint8 (0/1) or porosity float32 (NaN=nodata)
    valid: np.ndarray           # bool (n_depth, n_lateral): True where reservoir data exists
    depths_m: np.ndarray        # (n_depth,) depth of each row, metres
    prop: str                   # "lithology" | "porosity"
    axis: int
    index: int


def _nearest_indices(z_sorted: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Index into z_sorted (ascending) of the value nearest each depth."""
    pos = np.clip(np.searchsorted(z_sorted, depths, side="left"), 0, len(z_sorted) - 1)
    left = np.clip(pos - 1, 0, len(z_sorted) - 1)
    choose_left = np.abs(depths - z_sorted[left]) < np.abs(depths - z_sorted[pos])
    return np.where(choose_left, left, pos)


class NativeColumnRenderer:
    def __init__(self, native_dir, *, max_column_dist_idx: float = 4.0):
        self.dir = Path(native_dir)
        geom = json.loads((self.dir / "column_geometry.json").read_text())
        nx, ny, nz = geom["shape"]
        self.geom = ReservoirGeometry(
            nx=int(nx), ny=int(ny), nz=int(nz),
            axis0_spacing_m=float(geom["axis0_spacing_m"]),
            axis1_spacing_m=float(geom["axis1_spacing_m"]),
            sample_spacing_m=float(geom["sample_spacing_m"]),
        )
        self.max_column_dist_idx = float(max_column_dist_idx)

        self._lith = np.load(self.dir / "lithology_native_i_j_k.npy", mmap_mode="r")
        self._poro = np.load(self.dir / "porosity_native_i_j_k.npy", mmap_mode="r")
        self._act = np.load(self.dir / "actnum_native_i_j_k.npy", mmap_mode="r")
        self._z = np.load(self.dir / "z_center_native_i_j_k.npy", mmap_mode="r")

        centers = np.load(self.dir / "column_centers_axis.npy")           # (nx, ny, 2)
        valid = np.load(self.dir / "column_valid.npy")                    # (nx, ny)
        self._valid_ij = np.argwhere(valid).astype(np.int32)              # (M, 2)
        from scipy.spatial import cKDTree
        self._tree = cKDTree(centers[valid].astype(np.float32))

        # default depth window from a strided sample of active z-centres
        zs = np.asarray(self._z[::8, ::8, :]).astype(np.float32)
        acts = np.asarray(self._act[::8, ::8, :]) > 0
        zv = zs[acts & np.isfinite(zs)]
        self._z_lo = float(np.floor(zv.min())) if zv.size else 0.0
        self._z_hi = float(np.ceil(zv.max())) if zv.size else float(self.geom.nz)

    # -- per-column profiles ------------------------------------------------
    def _column_arrays(self, ni: int, nj: int):
        z = np.asarray(self._z[ni, nj, :]).astype(np.float64)
        act = np.asarray(self._act[ni, nj, :]) > 0
        lith = np.asarray(self._lith[ni, nj, :])
        poro = np.asarray(self._poro[ni, nj, :]).astype(np.float32)
        valid = act & (np.isfinite(poro) | (lith >= 0)) & np.isfinite(z)
        if valid.sum() < 2:
            return None
        order = np.argsort(z[valid])
        return (z[valid][order], lith[valid][order], poro[valid][order])

    def _cell_boundaries(self, z_sorted: np.ndarray) -> np.ndarray:
        mid = (z_sorted[:-1] + z_sorted[1:]) * 0.5
        b = np.empty(z_sorted.size + 1)
        b[1:-1] = mid
        b[0] = z_sorted[0] - (z_sorted[1] - z_sorted[0]) * 0.5
        b[-1] = z_sorted[-1] + (z_sorted[-1] - z_sorted[-2]) * 0.5
        return b

    def _lith_profile(self, arrs, depths, step):
        z_sorted, lith_sorted, _ = arrs
        b = self._cell_boundaries(z_sorted)
        present = (depths - step * 0.5 < b[-1]) & (depths + step * 0.5 > b[0])
        is_tgt_cell = np.isin(lith_sorted, _TARGET_CLASSES)
        nd = depths.size
        diff = np.zeros(nd + 1, np.int32)
        if is_tgt_cell.any():
            clo = b[:-1][is_tgt_cell] - step * 0.5
            chi = b[1:][is_tgt_cell] + step * 0.5
            i0 = np.clip(np.searchsorted(depths, clo, side="right"), 0, nd)
            i1 = np.clip(np.searchsorted(depths, chi, side="left"), 0, nd)
            np.add.at(diff, i0, 1)
            np.add.at(diff, i1, -1)
        tgt = (np.cumsum(diff[:-1]) > 0) & present
        return tgt.astype(np.uint8), present

    def _poro_profile(self, arrs, depths, step):
        z_sorted, _, poro_sorted = arrs
        b = self._cell_boundaries(z_sorted)
        present = (depths - step * 0.5 < b[-1]) & (depths + step * 0.5 > b[0])
        k = _nearest_indices(z_sorted, depths)
        val = poro_sorted[k]
        out = np.where(present & np.isfinite(val), val, np.float32(np.nan)).astype(np.float32)
        return out, present

    # -- public render ------------------------------------------------------
    def render_section(
        self,
        axis: int,
        index: int,
        n_lateral: int,
        *,
        prop: str = "lithology",
        v_step_m: float = 2.0,
        depth_min_m: float | None = None,
        depth_max_m: float | None = None,
    ) -> SectionResult:
        if axis not in (INLINE, XLINE):
            raise ValueError(f"axis must be INLINE(0) or XLINE(1), got {axis}")
        if prop not in ("lithology", "porosity"):
            raise ValueError(f"prop must be 'lithology' or 'porosity', got {prop!r}")

        lat = np.arange(n_lateral, dtype=np.float32)
        if axis == INLINE:        # fixed axis0=index, vary axis1
            query = np.column_stack([np.full(n_lateral, float(index), np.float32), lat])
        else:                     # fixed axis1=index, vary axis0
            query = np.column_stack([lat, np.full(n_lateral, float(index), np.float32)])
        dist, nearest = self._tree.query(query.astype(np.float32), workers=-1)
        cols = self._valid_ij[nearest]                       # (n_lateral, 2)
        in_support = dist <= self.max_column_dist_idx

        d_lo = self._z_lo if depth_min_m is None else float(depth_min_m)
        d_hi = self._z_hi if depth_max_m is None else float(depth_max_m)
        depths = (np.arange(d_lo + v_step_m * 0.5, d_hi, v_step_m)).astype(np.float64)
        nd = depths.size

        if prop == "lithology":
            values = np.zeros((nd, n_lateral), np.uint8)
        else:
            values = np.full((nd, n_lateral), np.nan, np.float32)
        valid = np.zeros((nd, n_lateral), bool)

        uniq, inverse = np.unique(cols, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        for g in range(uniq.shape[0]):
            lateral_idx = np.flatnonzero((inverse == g) & in_support)
            if lateral_idx.size == 0:
                continue
            arrs = self._column_arrays(int(uniq[g, 0]), int(uniq[g, 1]))
            if arrs is None:
                continue
            if prop == "lithology":
                prof, present = self._lith_profile(arrs, depths, v_step_m)
            else:
                prof, present = self._poro_profile(arrs, depths, v_step_m)
            values[:, lateral_idx] = prof[:, None]
            valid[:, lateral_idx] = present[:, None]

        return SectionResult(values=values, valid=valid, depths_m=depths.astype(np.float32),
                             prop=prop, axis=axis, index=int(index))
