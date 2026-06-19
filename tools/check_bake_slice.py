"""Validate the baker on one data-rich inline before the full bake: render the
section at the locked 6.25/2 m grid using bake_reservoir_npy's own functions and
draw a 3-colour image (white=nodata gray=bg yellow=target)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/quanbi/tools")
import bake_reservoir_npy as bake
from project_paths import DIAGNOSTICS_ROOT

NUMPY = Path("/root/quanbi/data/reservoir/numpy")
OUT = DIAGNOSTICS_ROOT / "bake_check"
OUT.mkdir(parents=True, exist_ok=True)
from PIL import Image

AXIS0 = 479  # seismic-index inline near the most target-rich native column

geom = json.loads((NUMPY / "column_geometry.json").read_text())
sp0, sp1, spz = geom["axis0_spacing_m"], geom["axis1_spacing_m"], geom["sample_spacing_m"]
centers = np.load(NUMPY / "column_centers_axis.npy")
valid = np.load(NUMPY / "column_valid.npy")
valid_ij = np.argwhere(valid).astype(np.int32)
tree = cKDTree(centers[valid].astype(np.float32))
lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")

a1v = centers[..., 1][valid]
a1_lo, a1_hi = int(np.floor(a1v.min())), int(np.ceil(a1v.max()))
zs = np.asarray(z[::2, ::2, :]); acts = np.asarray(act[::2, ::2, :]) > 0
zv = zs[acts & np.isfinite(zs)]
s_lo = int(np.floor((zv.min() - 20) / spz)); s_hi = int(np.ceil((zv.max() + 20) / spz))
sz = 5; v_step = spz / sz
NZ = (s_hi - s_lo) * sz
depths = (s_lo + np.arange(NZ) / sz) * spz
N1 = (a1_hi - a1_lo) * 2
p1 = (a1_lo + np.arange(N1) / 2).astype(np.float32)

dist, nn = tree.query(np.column_stack([np.full(N1, float(AXIS0), np.float32), p1]), workers=-1)
cols = valid_ij[nn]; inside = dist <= bake.MAX_DIST_IDX
lith2d = np.zeros((N1, NZ), np.uint8)
pres2d = np.zeros((N1, NZ), bool)
uniq, inv = np.unique(cols[inside], axis=0, return_inverse=True)
inv = inv.ravel(); positions = np.flatnonzero(inside)
for g in range(uniq.shape[0]):
    cells = positions[inv == g]
    arr = bake.column_arrays(lith, poro, act, z, int(uniq[g, 0]), int(uniq[g, 1]))
    if arr is None:
        continue
    zc, lcol, pcol = arr
    b = bake.cell_boundaries(zc)
    present = (depths - v_step * 0.5 < b[-1]) & (depths + v_step * 0.5 > b[0])
    lp = bake.lith_profile(zc, lcol, depths, v_step)
    lith2d[cells] = lp
    pres2d[cells] = present

rgb = np.full((NZ, N1, 3), 255, np.uint8)            # depth vertical
img = np.transpose(np.stack([lith2d, pres2d.astype(np.uint8)], -1), (1, 0, 2))
L = img[..., 0]; P = img[..., 1].astype(bool)
rgb[P & (L == 0)] = (150, 150, 150)
rgb[L == 1] = (245, 214, 45)
# downscale for viewing
h, w = rgb.shape[:2]
s = max(1, int(max(h, w) / 1400))
Image.fromarray(rgb[::s, ::s], "RGB").save(OUT / f"inline_{AXIS0}_baked.png")
print(f"axis0={AXIS0} section (NZ={NZ}, N1={N1}) target_frac={(L[P]==1).mean():.1%} "
      f"present_frac={P.mean():.1%} -> {OUT}")
