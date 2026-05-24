"""Smoke-test SAM3 offscreen rendering.

Loads the reservoir grid, builds a small ROI, renders three I-frames
and checks:
- output shape is identical across frames (SAM3 needs this)
- pixel values look like the cell colours we expect
- cell_id grid maps back to valid IJK inside the ROI

Saves the three frames as PNGs in tools/_sam3_smoke_out/ so the user
can eyeball them.

Run via:
    E:\\miniconda\\envs\\py312\\python.exe tools\\smoke_sam3_render.py F:\\
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "apps" / "yj_studio" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from PIL import Image

from yj_studio.reservoir import ReservoirGrid, default_roi
from yj_studio.reservoir.sam3_render import render_roi_section


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: smoke_sam3_render.py <dir>")
        sys.exit(2)
    root = Path(sys.argv[1])
    master = next(p for p in root.glob("*.GRDECL")
                  if "_COORD" not in p.name.upper()
                  and "_ZCORN" not in p.name.upper()
                  and "_ACTNUM" not in p.name.upper())

    grid = ReservoirGrid.load_from_master(master)
    print(f"grid: {grid.shape}, active={int(grid.active.sum()):,}")

    # A typical user-drawn ROI is small (a target zone, not the
    # whole grid). Simulate that by tightening to a centred sub-cube
    # — this is the realistic perf regime for SAM3.
    big = default_roi(grid)
    il, ih, jl, jh, kl, kh = big
    def shrink(lo, hi, frac=0.4):
        mid = (lo + hi) // 2
        half = max(1, int((hi - lo) * frac / 2))
        return mid - half, mid + half
    j_lo, j_hi = shrink(jl, jh)
    k_lo, k_hi = shrink(kl, kh)
    roi = (il, ih, j_lo, j_hi, k_lo, k_hi)
    print(f"ROI (narrow, simulated user-drawn): {roi}")

    out_dir = HERE / "_sam3_smoke_out"
    out_dir.mkdir(exist_ok=True)

    # Three I frames spaced through the ROI.
    i_lo, i_hi = roi[0], roi[1]
    samples = [i_lo + (i_hi - i_lo) // 4 * k for k in (1, 2, 3)]
    print(f"sampling i = {samples}")

    last_shape = None
    for idx in samples:
        t0 = time.time()
        frame = render_roi_section(grid, "i", idx, roi)
        dt = time.time() - t0
        print()
        print(f"i={idx}: rendered in {dt:.2f}s")
        print(f"  image shape: {frame.image.shape}, dtype={frame.image.dtype}")
        print(f"  cell_id shape: {frame.cell_id_grid.shape}")
        print(f"  unique pixel colours: ~{len(np.unique(frame.image.reshape(-1, 3), axis=0))}")
        valid_pixels = (frame.cell_id_grid[..., 0] >= 0).sum()
        print(f"  pixels with a cell-id: {valid_pixels:,} / {frame.image.shape[0] * frame.image.shape[1]:,}")

        if last_shape is not None and frame.image.shape != last_shape:
            print(f"  !! SHAPE MISMATCH vs previous frame ({last_shape})")
        last_shape = frame.image.shape

        png_path = out_dir / f"i_{idx:04d}.png"
        Image.fromarray(frame.image).save(png_path)
        print(f"  saved: {png_path}")

    # Sanity: pick a non-background pixel from the middle frame and
    # confirm the cell-id is inside the ROI.
    frame = render_roi_section(grid, "i", samples[1], roi)
    cid = frame.cell_id_grid
    valid = cid[..., 0] >= 0
    if valid.any():
        ys, xs = np.where(valid)
        idx_pick = len(ys) // 2
        py, px = int(ys[idx_pick]), int(xs[idx_pick])
        ijk = tuple(int(v) for v in cid[py, px])
        il, ih, jl, jh, kl, kh = roi
        in_roi = (il <= ijk[0] < ih) and (jl <= ijk[1] < jh) and (kl <= ijk[2] < kh)
        print()
        print(f"pixel ({px}, {py}) → cell {ijk}; inside ROI: {in_roi}")

    print()
    print(f"==== DONE: open {out_dir} to eyeball the frames ====")


if __name__ == "__main__":
    main()
