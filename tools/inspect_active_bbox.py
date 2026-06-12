"""Report several candidate bounding boxes for fixing the section view.

Compares:
  A. Full COORD pillar bbox (what we use now)
  B. Bbox over only the pillars touched by at least one active cell
  C. Per-axis bbox (the tightest container that still works for every
     index along that axis)

For each, prints the data extent and what fraction of the current
A bbox it occupies.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from project_paths import DEFAULT_RESERVOIR_MASTER, add_app_src_to_path

add_app_src_to_path()

import numpy as np

from yj_studio.reservoir import ReservoirGrid


def main() -> None:
    grid = ReservoirGrid.load_from_master(DEFAULT_RESERVOIR_MASTER)
    nx, ny, nz = grid.shape
    coord = grid.coord    # (nx+1, ny+1, 6)
    active = grid.active != 0    # (nx, ny, nz)

    # ---- A: full pillar bbox
    xs_all = np.concatenate([coord[..., 0].ravel(), coord[..., 3].ravel()])
    ys_all = np.concatenate([coord[..., 1].ravel(), coord[..., 4].ravel()])
    A = (float(xs_all.min()), float(xs_all.max()),
         float(ys_all.min()), float(ys_all.max()))
    A_area = (A[1] - A[0]) * (A[3] - A[2])
    print("==== A: full pillar bbox ====")
    print(f"  x: [{A[0]:.0f}, {A[1]:.0f}]  span {A[1]-A[0]:.0f}")
    print(f"  y: [{A[2]:.0f}, {A[3]:.0f}]  span {A[3]-A[2]:.0f}")
    print(f"  area = {A_area:.0f} m²")

    # ---- B: bbox over pillars touched by any active cell
    # A cell (i, j, *) touches pillars (i, j), (i+1, j), (i, j+1), (i+1, j+1).
    # So we mark active pillars wherever an active cell uses them.
    has_active_ij = active.any(axis=2)    # (nx, ny)
    pillar_active = np.zeros((nx + 1, ny + 1), dtype=bool)
    pillar_active[:-1, :-1] |= has_active_ij
    pillar_active[1:,  :-1] |= has_active_ij
    pillar_active[:-1, 1:]  |= has_active_ij
    pillar_active[1:,  1:]  |= has_active_ij
    xs_b = np.concatenate([coord[..., 0][pillar_active], coord[..., 3][pillar_active]])
    ys_b = np.concatenate([coord[..., 1][pillar_active], coord[..., 4][pillar_active]])
    B = (float(xs_b.min()), float(xs_b.max()),
         float(ys_b.min()), float(ys_b.max()))
    B_area = (B[1] - B[0]) * (B[3] - B[2])
    print()
    print("==== B: active-only pillar bbox ====")
    print(f"  x: [{B[0]:.0f}, {B[1]:.0f}]  span {B[1]-B[0]:.0f}")
    print(f"  y: [{B[2]:.0f}, {B[3]:.0f}]  span {B[3]-B[2]:.0f}")
    print(f"  area = {B_area:.0f} m²")
    print(f"  area fraction of A = {100*B_area/A_area:.1f}%")

    # ---- C: per-axis tightest bbox
    # K section needs to enclose all (i, j) where any K layer has active.
    # That's the same as has_active_ij above.
    pillar_for_k = pillar_active    # equivalent
    # I section needs to enclose all j of any active cell, for any i.
    # The horizontal axis of an I section is local y. The widest j range
    # we'll see across all i is the j range where some i,k is active.
    # In data terms: project active to j → take all j with anything.
    has_j = active.any(axis=(0, 2))    # (ny,)
    has_i = active.any(axis=(1, 2))    # (nx,)
    j_range = np.where(has_j)[0]
    i_range = np.where(has_i)[0]
    print()
    print(f"==== C: per-axis info ====")
    print(f"  any-active i indices: [{i_range.min()}, {i_range.max()}]")
    print(f"  any-active j indices: [{j_range.min()}, {j_range.max()}]")
    print()
    print("Recommendation: B (active-only pillar bbox) is usually the")
    print("best default — same bbox for every i/j/k, but skips the empty")
    print("padding of inactive border pillars.")


if __name__ == "__main__":
    main()
