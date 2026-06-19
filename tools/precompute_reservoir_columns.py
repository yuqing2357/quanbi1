"""Offline precompute for the runtime reservoir renderer.

Reads the GRDECL COORD geometry ONCE and writes small npy/json artifacts so the
server-side renderer never needs to parse GRDECL at runtime:

  column_centers_axis.npy  (nx, ny, 2) float32  - each native column centre in
                                                   seismic-axis index coords
  column_valid.npy         (nx, ny)    bool      - columns with any active/valid cell
  column_geometry.json                            - axis spacings, sample spacing, shape

Run on the server (the GRDECL parser lives under local/app/src).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/quanbi/tools")
import create_reservoir_3x_direct_numpy as ref

NUMPY = Path("/root/quanbi/data/reservoir/numpy")
GRDECL = Path("/root/quanbi/data/reservoir/grdecl")

transform = ref.transform_from_metadata(NUMPY / "metadata.json")
master = ref.find_master_grdecl(GRDECL)
spec = ref.find_specgrid(master)
if spec is None:
    raise SystemExit(f"SPECGRID not found in {master}")
coord = ref.read_coord(master.with_name(master.stem + "_COORD.GRDECL"), spec)
a0, a1 = ref.column_centers_axis(coord, transform)
del coord
centers = np.stack([np.asarray(a0, np.float32), np.asarray(a1, np.float32)], axis=-1)

lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
if not (lith.shape == poro.shape == act.shape == (spec.nx, spec.ny, spec.nz)):
    raise SystemExit("native array / SPECGRID shape mismatch")
valid = (
    np.asarray(act).any(axis=2)
    | (np.asarray(lith) >= 0).any(axis=2)
    | np.isfinite(np.asarray(poro)).any(axis=2)
)

geom = {
    "shape": [int(spec.nx), int(spec.ny), int(spec.nz)],
    "axis0_spacing_m": float(transform.axis0_spacing),
    "axis1_spacing_m": float(transform.axis1_spacing),
    "sample_spacing_m": float(transform.sample_spacing),
    "note": "column_centers_axis.npy is in seismic-axis INDEX units; "
            "multiply by axis*_spacing_m for metres.",
}

np.save(NUMPY / "column_centers_axis.npy", centers)
np.save(NUMPY / "column_valid.npy", valid)
with open(NUMPY / "column_geometry.json", "w") as fh:
    json.dump(geom, fh, indent=2)

print("centers:", centers.shape, centers.dtype)
print("valid columns:", int(valid.sum()), "/", valid.size)
print("axis spacing (idx units, m):", transform.axis0_spacing, transform.axis1_spacing,
      "| sample:", transform.sample_spacing)
print("wrote:", NUMPY / "column_centers_axis.npy", NUMPY / "column_valid.npy",
      NUMPY / "column_geometry.json")
