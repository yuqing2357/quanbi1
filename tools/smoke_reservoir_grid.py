"""Smoke-test ReservoirGrid end-to-end on the bundled GRDECL files.

Loads the grid (ZCORN cache hit if already built), checks shapes,
exercises the chunk cache, prints a few cells from different K
chunks to confirm the LRU caching path doesn't return stale data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from project_paths import add_app_src_to_path, reservoir_master_from_arg

add_app_src_to_path()

import numpy as np

from yj_studio.reservoir import ReservoirGrid


def _progress(frac: float, msg: str) -> None:
    print(f"  [{frac * 100:5.1f}%] {msg}")


def main() -> None:
    master = reservoir_master_from_arg(sys.argv[1] if len(sys.argv) >= 2 else None)
    print(f"master: {master}")

    print()
    print("==== load_from_master ====")
    t0 = time.time()
    grid = ReservoirGrid.load_from_master(master, progress_cb=_progress)
    print(f"  loaded in {time.time() - t0:.2f}s")
    print(f"  shape:      {grid.shape}")
    print(f"  active:     {int(grid.active.sum()):,} / {grid.active.size:,}")
    print(f"  coord:      {grid.coord.shape} {grid.coord.dtype}")
    print(f"  zcorn mmap: {grid.zcorn.shape} {grid.zcorn.dtype}")
    print(f"  properties: {grid.property_names()}")
    for name, arr in grid.properties.items():
        print(f"    {name:15s} shape={arr.shape} dtype={arr.dtype} "
              f"range=[{arr.min()}, {arr.max()}]")

    print()
    print("==== K-chunk geometry ====")
    for k_target in (0, 100, 500, 1000):
        if k_target >= grid.spec.nz:
            continue
        k0, k1 = grid.chunk_for_k(k_target)
        t0 = time.time()
        corners = grid.corners_for_k_chunk(k0, k1)
        dt = time.time() - t0
        print(f"  k={k_target} → chunk [{k0}, {k1})  "
              f"shape={corners.shape}  built in {dt:.2f}s")

    print()
    print("==== LRU cache hit (should be instant) ====")
    t0 = time.time()
    _ = grid.corners_for_k_chunk(0, grid.chunk_for_k(0)[1])
    print(f"  re-fetched chunk [0, k1): {(time.time() - t0) * 1000:.1f} ms")

    print()
    print("==== single K-layer access ====")
    layer = grid.corners_for_k_layer(50)
    print(f"  k=50 layer shape={layer.shape}")
    # Spot-check: pick an active cell in this layer
    active_in_layer = np.argwhere(grid.active[:, :, 50] != 0)
    if len(active_in_layer) > 0:
        i, j = active_in_layer[len(active_in_layer) // 2]
        cell = layer[i, j]
        print(f"  active cell (i={i}, j={j}, k=50) corners:")
        for c, xyz in enumerate(cell):
            print(f"    corner {c}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})")
        dz = cell[4:8, 2].mean() - cell[0:4, 2].mean()
        print(f"  dz = {dz:.2f} m")

    print()
    print("==== DONE ====")


if __name__ == "__main__":
    main()
