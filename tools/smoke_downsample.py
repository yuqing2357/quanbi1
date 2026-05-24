"""Smoke-test downsampling pipeline on the F:\\ grid.

Loads ReservoirGrid (fast — cache hit), then downsamples with the
default (2, 2, 4) block. Reports timing and sanity-checks the result.

Run via:
    E:\\miniconda\\envs\\py312\\python.exe tools\\smoke_downsample.py F:\\
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "apps" / "yj_studio" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from yj_studio.reservoir import ReservoirGrid
from yj_studio.reservoir.downsample import downsample


def _progress(frac: float, msg: str) -> None:
    print(f"  [{frac * 100:5.1f}%] {msg}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: smoke_downsample.py <master.GRDECL or dir>")
        sys.exit(2)
    master = Path(sys.argv[1])
    if master.is_dir():
        cands = [p for p in master.glob("*.GRDECL")
                 if "_COORD" not in p.name.upper()
                 and "_ZCORN" not in p.name.upper()
                 and "_ACTNUM" not in p.name.upper()]
        master = cands[0]
    print(f"master: {master}")

    print()
    print("==== load ReservoirGrid ====")
    t0 = time.time()
    grid = ReservoirGrid.load_from_master(master)
    print(f"  loaded in {time.time() - t0:.2f}s")

    print()
    print("==== downsample (2, 2, 4) ====")
    t0 = time.time()
    ds = downsample(grid, block=(2, 2, 4), progress_cb=_progress)
    elapsed = time.time() - t0
    print(f"  downsample took {elapsed:.1f}s")

    print()
    print("==== sanity check ====")
    print(f"  super-cell shape: {ds.shape}")
    print(f"  total super-cells: {ds.total_super_cells:,}")
    print(f"  active super-cells: {ds.active_super_cells:,} "
          f"({100*ds.active_super_cells/ds.total_super_cells:.2f}%)")
    print(f"  corners: {ds.corners.shape} {ds.corners.dtype} "
          f"({ds.corners.nbytes / 1024**2:.1f} MB)")
    print(f"  int properties: {list(ds.int_properties.keys())}")
    for name, arr in ds.int_properties.items():
        vals = arr[ds.active]
        print(f"    {name:15s} shape={arr.shape} range=[{vals.min()}, {vals.max()}]")
    print(f"  float properties: {list(ds.float_properties.keys())}")
    for name, arr in ds.float_properties.items():
        vals = arr[ds.active]
        print(f"    {name:15s} shape={arr.shape} "
              f"mean={vals.mean():.4f}  range=[{vals.min():.4f}, {vals.max():.4f}]")

    # Pick an active super-cell and print its geometry
    print()
    print("==== sample active super-cell ====")
    active_idx = np.argwhere(ds.active)
    if len(active_idx):
        I, J, K = active_idx[len(active_idx) // 2]
        cell = ds.corners[I, J, K]
        print(f"  super-cell (I={I}, J={J}, K={K}):")
        for c, xyz in enumerate(cell):
            print(f"    corner {c}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})")
        dx = float(cell[:, 0].max() - cell[:, 0].min())
        dy = float(cell[:, 1].max() - cell[:, 1].min())
        dz = float(cell[:, 2].max() - cell[:, 2].min())
        print(f"  size: dx={dx:.2f}  dy={dy:.2f}  dz={dz:.2f}")
        print(f"  (should be roughly 2x ij size, 4x k size of a single cell)")

    print()
    print("==== DONE ====")


if __name__ == "__main__":
    main()
