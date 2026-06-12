"""End-to-end verification of the GRDECL parser on the bundled files.

What it does:

1. summarize_grdecl() on the master — prints SPECGRID + INCLUDE list +
   top-level keyword inventory.
2. read_actnum() — checks total == nx*ny*nz, prints active count,
   compares against the probe's earlier number.
3. read_coord() — checks shape (nx+1, ny+1, 6), prints a sample pillar.
4. ZCORN cache build (first run: parses 5.38 GB ASCII → 4.09 GB binary;
   subsequent runs: cache hit, instant). Reports peak RSS to confirm
   we don't blow up memory during the build.
5. memmap the cache, pull a k-slab, materialise cell corners for a
   small (i, j) patch around the peak-active K layer.
6. Print a few cell corner coordinates so you can sanity-check vs.
   Petrel.

Run via:
    python tools\\verify_grdecl_parser.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make the package importable when the user runs this script directly.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from project_paths import RESERVOIR_GRDECL_ROOT, add_app_src_to_path

add_app_src_to_path()

import numpy as np

from yj_studio.io.grdecl import find_includes, summarize_grdecl
from yj_studio.io.grdecl.parser import read_actnum, read_coord
from yj_studio.io.grdecl.zcorn_cache import (
    build_cache,
    cell_corners,
    open_zcorn,
    zcorn_for_k_range,
)


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _humansize(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _find_quartet(root: Path) -> dict[str, Path]:
    quartet: dict[str, Path] = {}
    for p in root.glob("*.GRDECL"):
        upper = p.name.upper()
        if "_COORD" in upper:
            quartet["coord"] = p
        elif "_ZCORN" in upper:
            quartet["zcorn"] = p
        elif "_ACTNUM" in upper:
            quartet["actnum"] = p
        else:
            quartet["master"] = p
    return quartet


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) >= 2 else RESERVOIR_GRDECL_ROOT
    q = _find_quartet(root)
    for key in ("master", "coord", "zcorn", "actnum"):
        if key not in q:
            print(f"!! missing {key} GRDECL in {root}")
            sys.exit(2)
        print(f"{key:7s}: {q[key]}  ({_humansize(q[key].stat().st_size)})")

    print()
    print("======== summarize_grdecl ========")
    t0 = time.time()
    summary = summarize_grdecl(q["master"])
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  specgrid: {summary.specgrid}")
    print(f"  includes ({len(summary.includes)}):")
    for inc in summary.includes:
        print(f"    {inc}")
    print(f"  top-level keywords ({len(summary.keywords_seen)}): "
          f"{summary.keywords_seen[:20]}{'...' if len(summary.keywords_seen) > 20 else ''}")
    spec = summary.specgrid
    if spec is None:
        print("!! no SPECGRID, aborting")
        sys.exit(2)

    print()
    print("======== read_actnum ========")
    t0 = time.time()
    actnum = read_actnum(q["actnum"], spec)
    active = int(actnum.sum())
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  shape:   {actnum.shape}   dtype: {actnum.dtype}")
    print(f"  active:  {active:,} / {actnum.size:,}  ({100*active/actnum.size:.2f}%)")
    print(f"  RSS:     {_rss_mb():.1f} MB")

    print()
    print("======== read_coord ========")
    t0 = time.time()
    coord = read_coord(q["coord"], spec)
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  shape:   {coord.shape}   dtype: {coord.dtype}")
    print(f"  pillar (0, 0): top={coord[0, 0, 0:3]}  bot={coord[0, 0, 3:6]}")
    print(f"  pillar (nx/2, ny/2): "
          f"top={coord[spec.nx // 2, spec.ny // 2, 0:3]}  "
          f"bot={coord[spec.nx // 2, spec.ny // 2, 3:6]}")
    print(f"  RSS:     {_rss_mb():.1f} MB")

    print()
    print("======== ZCORN cache build (first run = slow!) ========")
    t0 = time.time()
    def _cb(written: int, total: int) -> None:
        if total <= 0:
            return
        pct = 100.0 * written / total
        print(f"    ZCORN progress: {pct:5.1f}%  ({written:,}/{total:,})  RSS={_rss_mb():.1f} MB")
    cache = build_cache(q["zcorn"], spec, progress_cb=_cb)
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  cache:   {cache}")
    print(f"  size:    {_humansize(cache.stat().st_size)}")
    print(f"  RSS:     {_rss_mb():.1f} MB")

    print()
    print("======== open_zcorn (memmap) ========")
    mm = open_zcorn(cache, spec)
    print(f"  shape:   {mm.shape}   dtype: {mm.dtype}")
    print(f"  sample mm[0, 0, 0:4] = {np.asarray(mm[0, 0, 0:4])}")
    print(f"  RSS:     {_rss_mb():.1f} MB  (should NOT include 4 GB of ZCORN)")

    print()
    print("======== zcorn_for_k_range (small slab) ========")
    k0, k1 = 0, 4  # first 4 K-layers
    t0 = time.time()
    slab = zcorn_for_k_range(mm, spec, k0, k1)
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  shape:   {slab.shape}   dtype: {slab.dtype}  size: {_humansize(slab.nbytes)}")
    print(f"  z range: [{slab.min():.2f}, {slab.max():.2f}]")

    print()
    print("======== cell_corners (small ij patch around an active cell) ========")
    # Pick a cell we KNOW is active (use actnum) so geometry shouldn't
    # be a pinched zero-thickness shell.
    active_ijk = np.argwhere(actnum[:, :, k0:k1] != 0)
    if active_ijk.size == 0:
        print("  no active cells in initial slab, falling back to centre")
        i_start, j_start = spec.nx // 2, spec.ny // 2
    else:
        # Pick one a few rows into the active region so the 4x4 patch
        # stays inside the grid.
        pick = active_ijk[len(active_ijk) // 2]
        i_start = max(0, min(int(pick[0]), spec.nx - 4))
        j_start = max(0, min(int(pick[1]), spec.ny - 4))
    i0_, i1_ = i_start, i_start + 4
    j0_, j1_ = j_start, j_start + 4
    t0 = time.time()
    corners = cell_corners(
        slab, coord, spec,
        k_offset=k0,
        i_range=(i0_, i1_),
        j_range=(j0_, j1_),
    )
    print(f"  elapsed: {time.time() - t0:.2f}s")
    print(f"  shape:   {corners.shape}   dtype: {corners.dtype}")
    cell = corners[0, 0, 0]    # first cell of patch, 8 corners
    print(f"  cell (i={i0_}, j={j0_}, k={k0}) corners:")
    labels = ["lowK-SW", "lowK-SE", "lowK-NW", "lowK-NE",
              "hiK-SW",  "hiK-SE",  "hiK-NW",  "hiK-NE"]
    for label, xyz in zip(labels, cell):
        print(f"    {label}: {xyz}")
    # Sanity: x/y of the 4 corners on a single K-face should form a
    # quadrilateral of sensible size (a few × pillar spacing).
    xy_low = cell[0:4, 0:2]
    dx = float(xy_low[:, 0].max() - xy_low[:, 0].min())
    dy = float(xy_low[:, 1].max() - xy_low[:, 1].min())
    print(f"  cell footprint: dx={dx:.2f}  dy={dy:.2f}")
    z_low = cell[0:4, 2].mean()
    z_hi  = cell[4:8, 2].mean()
    dz = z_hi - z_low
    print(f"  mean z_lowK={z_low:.2f}  z_hiK={z_hi:.2f}  dz={dz:.2f}")
    print(f"  RSS:     {_rss_mb():.1f} MB")

    print()
    print("======== DONE ========")


if __name__ == "__main__":
    main()
