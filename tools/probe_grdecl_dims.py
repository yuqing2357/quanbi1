"""Read-only: measure the native GRDECL grid's lateral cell size from COORD,
both raw (all columns) and restricted to ACTIVE columns."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/quanbi/tools")
import create_reservoir_3x_direct_numpy as ref

GRDECL = Path("/root/quanbi/data/reservoir/grdecl")
NUMPY = Path("/root/quanbi/data/reservoir/numpy")
SRC_META = NUMPY / "metadata.json"

transform = ref.transform_from_metadata(SRC_META)
master = ref.find_master_grdecl(GRDECL)
spec = ref.find_specgrid(master)
print("SPECGRID nx,ny,nz =", (spec.nx, spec.ny, spec.nz))
coord = ref.read_coord(master.with_name(master.stem + "_COORD.GRDECL"), spec)

xp = (coord[..., 0] + coord[..., 3]) * 0.5
yp = (coord[..., 1] + coord[..., 4]) * 0.5
xc = (xp[:-1, :-1] + xp[1:, :-1] + xp[:-1, 1:] + xp[1:, 1:]) * 0.25
yc = (yp[:-1, :-1] + yp[1:, :-1] + yp[:-1, 1:] + yp[1:, 1:]) * 0.25
wx, wy = transform.local_to_world(xc, yc)

d0 = np.hypot(np.diff(wx, axis=0), np.diff(wy, axis=0))   # (nx-1, ny)
d1 = np.hypot(np.diff(wx, axis=1), np.diff(wy, axis=1))   # (nx, ny-1)

actnum = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
active = np.asarray(actnum).any(axis=2)                    # (nx, ny) active column mask
print("active columns:", int(active.sum()), "/", active.size)

a0 = active[:-1, :] & active[1:, :]
a1 = active[:, :-1] & active[:, 1:]


def stats(name, arr):
    arr = arr[np.isfinite(arr) & (arr > 0)]
    p = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"{name}: n={arr.size}")
    print(f"    min={p[0]:.2f}  p1={p[1]:.2f}  p5={p[2]:.2f}  p25={p[3]:.2f}  "
          f"p50={p[4]:.2f}  p75={p[5]:.2f}  p95={p[6]:.2f}  p99={p[7]:.2f}  max={p[8]:.2f}")
    print(f"    fraction in 30-90 m: {np.mean((arr>=30)&(arr<=90)):.1%}")


print("\n--- RAW (all columns) ---")
stats("axis0 (i) width m", d0)
stats("axis1 (j) width m", d1)
print("\n--- ACTIVE columns only ---")
stats("axis0 (i) width m", d0[a0])
stats("axis1 (j) width m", d1[a1])
