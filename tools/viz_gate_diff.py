"""Show EXACTLY what changes between two gates on one inline: gate44 | gate125 |
gate125 with newly-filled pixels highlighted red. Proves the bulk interior is
identical and only thin gap columns (+edge) change."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/quanbi/tools")
import bake_reservoir_npy as bake
from project_paths import DIAGNOSTICS_ROOT

NUMPY = Path("/root/quanbi/data/reservoir/numpy")
OUT = DIAGNOSTICS_ROOT / "gate_compare"
OUT.mkdir(parents=True, exist_ok=True)
centers = np.load(NUMPY / "column_centers_axis.npy")
valid = np.load(NUMPY / "column_valid.npy")
vij = np.argwhere(valid).astype(np.int32)
tree = cKDTree(centers[valid].astype(np.float32))
lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")
V_STEP = 2.0

AX0 = 479
near0 = (np.abs(centers[..., 0] - AX0) < 3) & valid
a1 = centers[..., 1][near0]
lat = np.arange(int(np.floor(a1.min())) - 20, int(np.ceil(a1.max())) + 20)
depths = np.arange(3000.0, 4700.0, V_STEP)
nlat, nd = lat.size, depths.size

dist, nn = tree.query(np.column_stack([np.full(nlat, float(AX0)), lat.astype(np.float32)]).astype(np.float32))
cols = vij[nn]
lith2d = np.zeros((nlat, nd), np.uint8)
pres2d = np.zeros((nlat, nd), bool)
uniq, inv = np.unique(cols, axis=0, return_inverse=True)
inv = inv.ravel()
for g in range(uniq.shape[0]):
    cells = np.flatnonzero(inv == g)
    arr = bake.column_arrays(lith, poro, act, z, int(uniq[g, 0]), int(uniq[g, 1]))
    if arr is None:
        continue
    zc, lcol, pcol = arr
    b = bake.cell_boundaries(zc)
    present = (depths - V_STEP * 0.5 < b[-1]) & (depths + V_STEP * 0.5 > b[0])
    lith2d[cells] = bake.lith_profile(zc, lcol, depths, V_STEP)
    pres2d[cells] = present


def rgb_at(gate):
    keep = (dist <= gate)[:, None] & pres2d
    img = np.full((nlat, nd, 3), 255, np.uint8)
    img[keep & (lith2d == 0)] = (150, 150, 150)
    img[keep & (lith2d == 1)] = (245, 214, 45)
    return np.transpose(img, (1, 0, 2)), keep


A, keepA = rgb_at(3.5)
B, keepB = rgb_at(125 / 12.5)
newly = np.transpose(keepB & ~keepA, (1, 0))      # (nd, nlat)
C = B.copy()
C[newly] = (220, 30, 30)

# stats
d = dist
print(f"laterals: total={nlat}  dist<=3.5(unaffected)={int((d<=3.5).sum())}  "
      f"3.5<dist<=12(interior gaps)={int(((d>3.5)&(d<=12)).sum())}  "
      f"12<dist<=10idx_gate(edge)={int(((d>3.5)&(d<=10)).sum())}")
changed_pix = int(newly.sum()); total_pix = nd * nlat
print(f"pixels changed gate44->gate125: {changed_pix}/{total_pix} = {changed_pix/total_pix:.2%}")


def label(img, txt):
    s = max(1, int(img.shape[1] / 700))
    im = Image.fromarray(img[::s, ::s], "RGB")
    band = Image.new("RGB", (im.width, 22), (20, 20, 20))
    ImageDraw.Draw(band).text((6, 5), txt, fill=(255, 255, 255))
    out = Image.new("RGB", (im.width, im.height + 22), (20, 20, 20))
    out.paste(band, (0, 0)); out.paste(im, (0, 22))
    return np.asarray(out)


pa, pb, pc = label(A, "gate=44m"), label(B, "gate=125m"), label(C, "gate=125m (red=newly filled)")
h = min(pa.shape[0], pb.shape[0], pc.shape[0])
gap = np.full((h, 14, 3), 60, np.uint8)
Image.fromarray(np.concatenate([pa[:h], gap, pb[:h], gap, pc[:h]], axis=1), "RGB").save(OUT / "gate_diff_inline479.png")
print("DONE -> gate_diff_inline479.png")
