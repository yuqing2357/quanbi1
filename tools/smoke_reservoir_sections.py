"""Smoke-test the IJK section extractors on the bundled GRDECL grid.

Loads ReservoirGrid (cache hits), extracts one K / I / J section,
verifies shapes, prints quad bounds and a few sample cells.

Run via:
    python tools\\smoke_reservoir_sections.py data\\reservoir\\grdecl
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

from yj_studio.reservoir import ReservoirGrid, SeismicIndexTransform
from yj_studio.reservoir.sections import (
    extract_i_section,
    extract_j_section,
    extract_k_section,
    values_for_section,
)


def main() -> None:
    master = reservoir_master_from_arg(sys.argv[1] if len(sys.argv) >= 2 else None)
    print(f"master: {master}")

    grid = ReservoirGrid.load_from_master(master)
    print(f"grid loaded: shape={grid.shape}, active={int(grid.active.sum()):,}")
    transform = SeismicIndexTransform()

    # Choose interior indices so the section has real content.
    nx, ny, nz = grid.shape
    i_test, j_test, k_test = nx // 2, ny // 2, nz // 2

    for label, fn, idx in [
        ("K", lambda: extract_k_section(grid, k_test), k_test),
        ("I", lambda: extract_i_section(grid, i_test, transform=transform), i_test),
        ("J", lambda: extract_j_section(grid, j_test, transform=transform), j_test),
    ]:
        print()
        print(f"==== {label}-section @ idx={idx} ====")
        t0 = time.time()
        sec = fn()
        dt = time.time() - t0
        print(f"  elapsed: {dt:.2f}s")
        print(f"  n_cells: {sec.n_cells:,}")
        if sec.n_cells == 0:
            print("  !! empty section, skipping")
            continue
        q = sec.quads
        print(f"  quad shape: {q.shape}, dtype: {q.dtype}")
        print(f"  horiz range: [{q[..., 0].min():.2f}, {q[..., 0].max():.2f}]")
        print(f"  vert  range: [{q[..., 1].min():.2f}, {q[..., 1].max():.2f}]")
        print(f"  first cell quad:")
        for c, xy in enumerate(q[0]):
            print(f"    corner {c}: ({xy[0]:.2f}, {xy[1]:.2f})")
        print(f"  first cell id: {tuple(sec.cell_ids[0])}")

        # Look up LITHOLOGIES on this section
        if grid.has_property("LITHOLOGIES"):
            vals = values_for_section(sec, grid.property("LITHOLOGIES"))
            unique, counts = np.unique(vals, return_counts=True)
            print(f"  LITHOLOGIES counts: {dict(zip(unique.tolist(), counts.tolist()))}")

    print()
    print("==== DONE ====")


if __name__ == "__main__":
    main()
