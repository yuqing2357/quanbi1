"""Calibrate world→seismic-sample mapping for the reservoir grid.

The simplified mapping (X0=Y0=0, dx=dy=12.5) gets the *scale* right
but leaves the reservoir mirrored / offset in the seismic frame.
This script pins down the true affine transform by fitting Petrel
native cell centres against their resampled positions in
``lithology_points_seismic_vis.npy``.

How it works:

1. Load the GRDECL grid → compute (x_center_m, y_center_m, z_center_m)
   per cell column (we use k=mid-K so corner-point pinches don't bias).
2. Load ``z_center_native_i_j_k.npy`` and ``actnum_native_i_j_k.npy``
   to find which (i, j, k) cells are actually filled in the vis output.
3. Load ``lithology_points_seismic_vis.npy`` (N, 4) = (axis0, axis1,
   sample, value). The first 497384 rows correspond to the active
   cells walked with ``stride=85``.
4. Match vis-points to native cells by sample/z proximity, then do
   least-squares on:
       axis0 = a * x_m + b * y_m + c
       axis1 = d * x_m + e * y_m + f
   This handles arbitrary 2D rotation + translation + flip.
5. Emit the parameters as code we can drop into SeismicIndexTransform.

Run via:
    python tools\\calibrate_reservoir_transform.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from project_paths import (
    RESERVOIR_GRDECL_ROOT,
    RESERVOIR_NUMPY_ROOT,
    add_app_src_to_path,
)

add_app_src_to_path()

import numpy as np

from yj_studio.reservoir import ReservoirGrid


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) >= 2 else RESERVOIR_GRDECL_ROOT

    # ---- find files ----
    master = None
    for p in root.glob("*.GRDECL"):
        up = p.name.upper()
        if "_COORD" not in up and "_ZCORN" not in up and "_ACTNUM" not in up:
            master = p
            break
    if master is None:
        print(f"!! no master GRDECL in {root}")
        sys.exit(2)
    print(f"master: {master}")

    model_dir = RESERVOIR_NUMPY_ROOT
    vis_path = model_dir / "lithology_points_seismic_vis.npy"
    zc_path = model_dir / "z_center_native_i_j_k.npy"
    actnum_path = model_dir / "actnum_native_i_j_k.npy"
    for p in (vis_path, zc_path, actnum_path):
        if not p.exists():
            print(f"!! missing: {p}")
            sys.exit(2)

    # ---- load grid (for cell centre xy in native frame) ----
    print()
    print("==== load ReservoirGrid ====")
    grid = ReservoirGrid.load_from_master(master)
    nx, ny, nz = grid.shape
    print(f"  shape: {grid.shape}")

    # ---- load reference arrays ----
    print()
    print("==== load reference numpy ====")
    actnum_native = np.load(actnum_path)
    zc_native = np.load(zc_path)
    vis = np.load(vis_path)
    print(f"  actnum native: {actnum_native.shape} {actnum_native.dtype}")
    print(f"  z_center:      {zc_native.shape} {zc_native.dtype}")
    print(f"  vis points:    {vis.shape} {vis.dtype}")

    # vis layout: (N, 4) = (axis0, axis1, sample, value)
    # Per metadata.json: lith_stride = 85, lith_valid_cells = 42277621,
    # vis_points = 497384 = ceil(42277621 / 85). The vis order is the
    # ravel order of active cells in (i, j, k).
    print()
    print("==== identify active cells walked in vis ====")
    active_mask = actnum_native != 0
    active_count = int(active_mask.sum())
    expected_vis = (active_count + 85 - 1) // 85
    print(f"  active count (actnum): {active_count:,}")
    print(f"  vis row count:         {vis.shape[0]:,}")
    print(f"  ceil(active/85):       {expected_vis:,}")

    # Get the (i, j, k) of all active cells in the same order vis walked
    # them. ravel order matters — metadata's `correction_note` mentions
    # axis 0/1 were swapped after raster build, so we should be careful
    # about i-fast vs j-fast. Try the most common ordering first: C-order
    # of (i, j, k) — i.e. k slowest, then j, then i fastest. (No, that's
    # F-order. In numpy default C-order, i is slowest.)
    # Empirically: ravel the active mask in C-order, take every 85th
    # entry, and we should align with vis.
    flat_active_idx = np.argwhere(active_mask)    # (active_count, 3) i,j,k
    sampled = flat_active_idx[::85]
    n_match = min(sampled.shape[0], vis.shape[0])
    sampled = sampled[:n_match]
    vis_match = vis[:n_match]
    print(f"  matched pairs: {n_match:,}")

    # ---- compute world xy centre for each matched cell ----
    print()
    print("==== compute cell-centre world xy from grid ====")
    t0 = time.time()
    # We need a cell xy centre. Cheapest: average the 4 pillar tops at
    # the cell's 4 IJ corners. (Top vs bot ignored — we only want xy
    # near the cell column's xy footprint.)
    coord = grid.coord    # (nx+1, ny+1, 6) float32 — (x_top, y_top, z_top, x_bot, y_bot, z_bot)
    pillar_xy_top = coord[..., 0:2]    # (nx+1, ny+1, 2)
    # Average over the 4 IJ corners of cell (i, j):
    #   pillars (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    # Build a vectorised lookup.
    i_arr = sampled[:, 0]
    j_arr = sampled[:, 1]
    p00 = pillar_xy_top[i_arr,   j_arr,   :]
    p10 = pillar_xy_top[i_arr+1, j_arr,   :]
    p01 = pillar_xy_top[i_arr,   j_arr+1, :]
    p11 = pillar_xy_top[i_arr+1, j_arr+1, :]
    cell_xy_world = (p00 + p10 + p01 + p11) / 4.0    # (n_match, 2)
    print(f"  computed in {time.time() - t0:.2f}s")
    print(f"  world x range: [{cell_xy_world[:, 0].min():.1f}, {cell_xy_world[:, 0].max():.1f}]")
    print(f"  world y range: [{cell_xy_world[:, 1].min():.1f}, {cell_xy_world[:, 1].max():.1f}]")
    print(f"  vis axis0 range: [{vis_match[:, 0].min():.1f}, {vis_match[:, 0].max():.1f}]")
    print(f"  vis axis1 range: [{vis_match[:, 1].min():.1f}, {vis_match[:, 1].max():.1f}]")

    # ---- least-squares fit: (axis0, axis1) = A @ (x_m, y_m) + b ----
    print()
    print("==== least-squares fit ====")
    X = cell_xy_world.astype(np.float64)
    A0 = vis_match[:, 0].astype(np.float64)
    A1 = vis_match[:, 1].astype(np.float64)
    # Augment X with constant column for offset
    M = np.column_stack([X, np.ones(X.shape[0])])    # (n, 3)
    # Solve for axis0 = a*x + b*y + c
    sol0, res0, *_ = np.linalg.lstsq(M, A0, rcond=None)
    sol1, res1, *_ = np.linalg.lstsq(M, A1, rcond=None)
    a, b, c = sol0
    d, e, f = sol1
    print(f"  axis0 = {a:+.6f} * x_m  +  {b:+.6f} * y_m  +  {c:+.4f}")
    print(f"  axis1 = {d:+.6f} * x_m  +  {e:+.6f} * y_m  +  {f:+.4f}")

    # ---- residuals ----
    pred0 = M @ sol0
    pred1 = M @ sol1
    err0 = pred0 - A0
    err1 = pred1 - A1
    print(f"  residual axis0: rms={np.sqrt((err0**2).mean()):.3f}  max|err|={np.abs(err0).max():.3f}")
    print(f"  residual axis1: rms={np.sqrt((err1**2).mean()):.3f}  max|err|={np.abs(err1).max():.3f}")

    # ---- print code-ready output ----
    print()
    print("==== copy into SeismicIndexTransform ====")
    print(f"# 2D affine: (axis0, axis1) = ([{a:.6f} {b:.6f}; {d:.6f} {e:.6f}]) @ (x, y) + ({c:.4f}, {f:.4f})")
    print(f"AXIS0_A = {a!r}")
    print(f"AXIS0_B = {b!r}")
    print(f"AXIS0_C = {c!r}")
    print(f"AXIS1_D = {d!r}")
    print(f"AXIS1_E = {e!r}")
    print(f"AXIS1_F = {f!r}")
    print(f"Z_STEP = 10.0  # from metadata sample_index = z_depth_m / 10.0")

    # ---- sanity check on a known active cell ----
    print()
    print("==== sanity check ====")
    # Use the cell we know from earlier smoke tests: (194, 237, 50)
    if 194 < nx and 237 < ny:
        i, j = 194, 237
        p00 = pillar_xy_top[i,   j,   :]
        p10 = pillar_xy_top[i+1, j,   :]
        p01 = pillar_xy_top[i,   j+1, :]
        p11 = pillar_xy_top[i+1, j+1, :]
        xy = (p00 + p10 + p01 + p11) / 4.0
        x_m, y_m = float(xy[0]), float(xy[1])
        axis0_pred = a * x_m + b * y_m + c
        axis1_pred = d * x_m + e * y_m + f
        print(f"  cell (i=194, j=237):")
        print(f"    world (x, y) = ({x_m:.2f}, {y_m:.2f})")
        print(f"    predicted (axis0, axis1) = ({axis0_pred:.2f}, {axis1_pred:.2f})")
        print(f"    expected axis0 ∈ [201, 1451],  axis1 ∈ [0, 1097]")


if __name__ == "__main__":
    main()
