#!/usr/bin/env python
"""Compute the fixed RGT stretch span [lo, hi] to pin in config (model_rgt).

A fixed span keeps each horizon ONE colour across every tracked frame; per-slice
rescaling would make the same layer drift colour between frames and hurt SAM3
tracking. The server computes a stable span lazily from a strided subsample when
``rgt_span`` is null, but pinning it makes the value explicit and reproducible.

By default the span is the body-restricted percentile (the exact tonal range the
QC review used): RGT values intersected with the lithology body, then the 2/98
percentile. Pass --whole-volume to ignore lithology and use all finite RGT.

Usage (on the server):
  python server/scripts/compute_rgt_span.py \
      --rgt   data/seismic/rgt_out/rgt_vol.npy \
      --lith  data/reservoir/npy_625x625x2_v3/lithology_binary_uint8.npy
Paste the printed [lo, hi] into config/server.yaml under volumes.model_rgt.rgt_span.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/src"))

from yj_studio_core.reservoir import compute_rgt_span  # noqa: E402

# RGT grid is the model's 2x-lateral coarsening (handled in the renderer); when
# intersecting with lithology we stride the model body to the RGT grid.
_LATERAL_STRIDE = 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgt", type=Path, required=True, help="rgt_vol.npy")
    p.add_argument("--lith", type=Path, default=None, help="lithology_binary_uint8.npy")
    p.add_argument("--percentile", type=float, nargs=2, default=(2.0, 98.0))
    p.add_argument("--stride", type=int, default=2, help="extra subsample stride on the RGT grid")
    p.add_argument("--whole-volume", action="store_true", help="ignore lithology, use all finite RGT")
    args = p.parse_args()

    rgt = np.load(args.rgt, mmap_mode="r")
    s = max(1, int(args.stride))
    sub = np.asarray(rgt[::s, ::s, ::s], dtype=np.float32)

    if args.lith is not None and not args.whole_volume:
        lith = np.load(args.lith, mmap_mode="r")
        # Map the strided RGT sample back to model indices to pick the body mask.
        gi = (np.arange(sub.shape[0]) * s * _LATERAL_STRIDE).clip(0, lith.shape[0] - 1)
        gj = (np.arange(sub.shape[1]) * s * _LATERAL_STRIDE).clip(0, lith.shape[1] - 1)
        gk = (np.linspace(0, lith.shape[2] - 1, sub.shape[2])).round().astype(int)
        body = np.asarray(lith[np.ix_(gi, gj, gk)]) > 0
        vals = sub[body]
        scope = f"body-restricted ({body.mean() * 100:.1f}% of subsample is body)"
        if vals.size == 0:
            print("WARNING: no body voxels in subsample; falling back to whole volume")
            vals = sub
            scope = "whole volume (fallback)"
    else:
        vals = sub
        scope = "whole volume"

    lo, hi = compute_rgt_span(vals, percentile=tuple(args.percentile))
    print(f"rgt file        : {args.rgt}")
    print(f"subsample shape : {sub.shape}  (stride {s})")
    print(f"scope           : {scope}")
    print(f"percentile      : {tuple(args.percentile)}")
    print(f"rgt_span        : [{lo:.6g}, {hi:.6g}]")
    print()
    print("Paste into config/server.yaml -> volumes.model_rgt.rgt_span:")
    print(f"    rgt_span: [{lo:.6g}, {hi:.6g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
