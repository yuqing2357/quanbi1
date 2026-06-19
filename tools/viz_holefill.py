"""Demonstrate '44 m + fill enclosed holes': tight 44 m footprint defines the
body edge; interior holes (enclosed white stripes) are filled by their nearest
valid column at ANY distance; the exterior (connected to the grid border) stays
nodata. Panels: gate44 | 44+holefill | holefill (red=newly filled)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/quanbi/tools")
import bake_reservoir_npy as bake
from project_paths import DIAGNOSTICS_ROOT

NUMPY = Path("/root/quanbi/data/reservoir/numpy")
RES = Path("/root/quanbi/data/reservoir/npy_625x625x2_v3")
OUT = DIAGNOSTICS_ROOT / "gate_compare"
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((RES / "metadata.json").read_text())
N0, N1, _ = meta["shape"]
ORG0 = meta["seismic_index_origin"]["axis0"]
ORG1 = meta["seismic_index_origin"]["axis1"]

centers = np.load(NUMPY / "column_centers_axis.npy")
valid = np.load(NUMPY / "column_valid.npy")
vij = np.argwhere(valid).astype(np.int32)
tree = cKDTree(centers[valid].astype(np.float32))
lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")
V_STEP, TIGHT = 2.0, 3.5

# 2D lateral footprint over the whole baked grid (axis0 x axis1)
p0 = ORG0 + np.arange(N0) / 2.0
p1 = ORG1 + np.arange(N1) / 2.0
xx, yy = np.meshgrid(p0.astype(np.float32), p1.astype(np.float32), indexing="ij")
dmap, _ = tree.query(np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32), workers=-1)
dmap = dmap.reshape(N0, N1)
tight = dmap <= TIGHT
filled = binary_fill_holes(tight)
holes = filled & ~tight
print(f"footprint: tight cells={int(tight.sum())}  filled cells={int(filled.sum())}  "
      f"holes filled={int(holes.sum())}  ({holes.sum()/max(tight.sum(),1):.2%} of body)")
hd = dmap[holes]
if hd.size:
    print(f"hole nearest-col dist: p50={np.percentile(hd,50)*12.5:.0f}m p95={np.percentile(hd,95)*12.5:.0f}m "
          f"max={hd.max()*12.5:.0f}m  (these are the fill reaches)")

# inline 479 render over the body window
o0 = (479 - ORG0) * 2
row_t, row_f = tight[o0], filled[o0]
have = np.flatnonzero(row_f)
o1lo, o1hi = int(have.min()) - 40, int(have.max()) + 40
o1 = np.arange(max(0, o1lo), min(N1, o1hi))
axis1 = ORG1 + o1 / 2.0
depths = np.arange(3000.0, 4700.0, V_STEP)
nlat, nd = o1.size, depths.size
dist, nn = tree.query(np.column_stack([np.full(nlat, 479.0), axis1.astype(np.float32)]).astype(np.float32))
cols = vij[nn]
lith2d = np.zeros((nlat, nd), np.uint8); pres2d = np.zeros((nlat, nd), bool)
uniq, inv = np.unique(cols, axis=0, return_inverse=True); inv = inv.ravel()
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

kt = row_t[o1][:, None] & pres2d
kf = row_f[o1][:, None] & pres2d
newly = np.transpose(kf & ~kt, (1, 0))


def img_of(keep):
    im = np.full((nlat, nd, 3), 255, np.uint8)
    im[keep & (lith2d == 0)] = (150, 150, 150)
    im[keep & (lith2d == 1)] = (245, 214, 45)
    return np.transpose(im, (1, 0, 2))


A = img_of(kt); B = img_of(kf); C = B.copy(); C[newly] = (220, 30, 30)


def label(img, txt):
    s = max(1, int(img.shape[1] / 740))
    im = Image.fromarray(img[::s, ::s], "RGB")
    band = Image.new("RGB", (im.width, 22), (20, 20, 20))
    ImageDraw.Draw(band).text((6, 5), txt, fill=(255, 255, 255))
    o = Image.new("RGB", (im.width, im.height + 22), (20, 20, 20))
    o.paste(band, (0, 0)); o.paste(im, (0, 22)); return np.asarray(o)


pa, pb, pc = label(A, "44m only (stripes)"), label(B, "44m + fill holes"), label(C, "red = newly filled holes")
h = min(pa.shape[0], pb.shape[0], pc.shape[0])
gap = np.full((h, 14, 3), 60, np.uint8)
out_img = Image.fromarray(np.concatenate([pa[:h], gap, pb[:h], gap, pc[:h]], axis=1), "RGB")
out_img.thumbnail((1500, 1500))
out_img.save(OUT / "holefill_inline479.png")
print("DONE -> holefill_inline479.png", out_img.size)
