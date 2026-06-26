#!/usr/bin/env python
"""Crop the full seismic volume down to the reservoir model's footprint.

The reservoir model (``npy_625x625x2_v3``) was built as an anisotropic
node-aligned refinement of a *sub-cube* of the seismic volume.  Its
``metadata.json`` records exactly which seismic indices that sub-cube spans
(``seismic_index_bounds_inclusive``), so cropping is loss-free and unambiguous:
the crop is the seismic data co-located with the reservoir model, at native
seismic resolution.

  reservoir_shape[ax] == (span - 1) * scale[ax] + 1,  span = hi - lo + 1

This script reads those bounds from the metadata (never hard-codes them, so it
keeps working if a v4 model re-crops a different window), verifies the seismic
shape matches ``seismic_shape``, and writes ``seismic[lo:hi+1, ...]`` to a new
.npy.  The output keeps the seismic's native sampling (coarser than the model);
upsampling to the model grid for per-voxel fusion is a separate step.

Usage (on the server, in the yjstudio-server env):

    python server/scripts/crop_seismic_to_reservoir.py
    python server/scripts/crop_seismic_to_reservoir.py --root /root/quanbi
    python server/scripts/crop_seismic_to_reservoir.py --dry-run   # verify only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_AXES = ("axis0", "axis1", "sample")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/quanbi"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="reservoir metadata.json (default: <root>/data/reservoir/npy_625x625x2_v3/metadata.json)",
    )
    parser.add_argument(
        "--seismic",
        type=Path,
        default=None,
        help="full seismic .npy (default: <root>/data/seismic/YJ-ALL-SEISMIC.npy)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .npy (default: <seismic_dir>/YJ-SEISMIC-RESERVOIR-CROP.npy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify geometry and print the crop, but do not write the output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output after validating the source geometry",
    )
    args = parser.parse_args()

    meta_path = args.metadata or (
        args.root / "data/reservoir/npy_625x625x2_v3/metadata.json"
    )
    seismic_path = args.seismic or (args.root / "data/seismic/YJ-ALL-SEISMIC.npy")
    out_path = args.out or (seismic_path.parent / "YJ-SEISMIC-RESERVOIR-CROP.npy")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bounds = meta["seismic_index_bounds_inclusive"]
    scale = dict(zip(_AXES, meta["scale_axis0_axis1_sample"]))
    res_shape = tuple(meta["shape"])
    seis_shape_meta = tuple(meta["seismic_shape"])

    # Inclusive bounds -> Python slices; verify the refinement identity per axis.
    slices: list[slice] = []
    for i, ax in enumerate(_AXES):
        lo, hi = int(bounds[ax][0]), int(bounds[ax][1])
        span = hi - lo + 1
        refined = (span - 1) * int(scale[ax]) + 1
        if refined != res_shape[i]:
            raise SystemExit(
                f"geometry mismatch on {ax}: seismic span {span} refined to "
                f"{refined} but model shape is {res_shape[i]}"
            )
        print(
            f"geometry {ax}: span={span} scale={scale[ax]} refined={refined} "
            f"model={res_shape[i]} match={refined == res_shape[i]}"
        )
        if hi + 1 > seis_shape_meta[i]:
            raise SystemExit(
                f"{ax} upper bound {hi} exceeds seismic_shape {seis_shape_meta[i]}"
            )
        slices.append(slice(lo, hi + 1))

    seismic = np.load(seismic_path, mmap_mode="r")
    if any(actual < expected for actual, expected in zip(seismic.shape, seis_shape_meta)):
        raise SystemExit(
            f"seismic shape {tuple(seismic.shape)} does not contain metadata grid "
            f"{seis_shape_meta}; wrong file or stale metadata"
        )
    if tuple(seismic.shape) != seis_shape_meta:
        print(
            f"note: source shape {tuple(seismic.shape)} is a superset of metadata "
            f"grid {seis_shape_meta}; only the recorded inclusive bounds are used"
        )

    out_shape = tuple(s.stop - s.start for s in slices)
    gb = int(np.prod(out_shape)) * seismic.dtype.itemsize / 2**30
    print(f"seismic     : {tuple(seismic.shape)} {seismic.dtype}  {seismic_path}")
    print(f"crop slice  : [{slices[0].start}:{slices[0].stop}, "
          f"{slices[1].start}:{slices[1].stop}, {slices[2].start}:{slices[2].stop}]")
    print(f"crop shape  : {out_shape} {seismic.dtype}  (~{gb:.1f} GB)")
    print(f"model shape : {res_shape}  (refinement {meta['scale_axis0_axis1_sample']})")

    if args.dry_run:
        print("dry-run: geometry verified, no output written.")
        return 0

    if out_path.exists() and not args.overwrite:
        existing = np.load(out_path, mmap_mode="r")
        if tuple(existing.shape) == out_shape and existing.dtype == seismic.dtype:
            print(
                f"existing output is already valid: {out_path} "
                f"{tuple(existing.shape)} {existing.dtype}"
            )
            return 0
        raise SystemExit(
            f"output already exists with shape={existing.shape} dtype={existing.dtype}; "
            "pass --overwrite to replace it"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=seismic.dtype, shape=out_shape
    )
    # Copy a slab at a time along axis0 to keep peak RAM at one slab, not the
    # whole crop.
    s1, s2 = slices[1], slices[2]
    for k, a0 in enumerate(range(slices[0].start, slices[0].stop)):
        out[k] = seismic[a0, s1, s2]
    out.flush()
    del out
    print(f"wrote       : {out_path}  ({gb:.1f} GB)")
    print("Next: add it to config (e.g. id 'seismic_reservoir') and restart the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
