"""Read-only: render a reservoir lithology z-slice (map view) in the seismic-axis
frame, side by side with the seismic amplitude slice at the same depth & region.
Shows the reservoir's true ~50 m lateral granularity vs seismic 12.5 m.
white=nodata, gray=background(lith0), yellow=target(lith1/2)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/quanbi/tools")
import create_reservoir_3x_direct_numpy as ref
from project_paths import DIAGNOSTICS_ROOT

NUMPY = Path("/root/quanbi/data/reservoir/numpy")
GRDECL = Path("/root/quanbi/data/reservoir/grdecl")
SRC_META = NUMPY / "metadata.json"
SEIS = Path("/root/quanbi/data/seismic/YJ-ALL-SEISMIC.npy")
OUT = DIAGNOSTICS_ROOT / "zslice_demo"
OUT.mkdir(parents=True, exist_ok=True)

from PIL import Image

WHITE = np.array([255, 255, 255], np.uint8)
GRAY = np.array([150, 150, 150], np.uint8)
YELLOW = np.array([245, 214, 45], np.uint8)

# axis0/axis1 reservoir footprint in seismic-index units (from metadata point bounds)
A0_LO, A0_HI = 201, 1451
A1_LO, A1_HI = 0, 1097

transform = ref.transform_from_metadata(SRC_META)
master = ref.find_master_grdecl(GRDECL)
spec = ref.find_specgrid(master)
coord = ref.read_coord(master.with_name(master.stem + "_COORD.GRDECL"), spec)
a0c, a1c = ref.column_centers_axis(coord, transform)
del coord

lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")

# auto-pick a depth with lots of target
ls = np.asarray(lith[::4, ::4, :])
zs = np.asarray(z[::4, ::4, :]).astype(np.float32)
tgt = (ls == 1) | (ls == 2)
depth = float(np.median(zs[tgt]))
sample_idx = int(round(depth / transform.sample_spacing))
print(f"depth = {depth:.0f} m  -> seismic sample idx {sample_idx}")

# per native column: nearest k to this depth, its lith, and distance
zf = np.asarray(z).astype(np.float32)
af = np.asarray(act) > 0
dz = np.abs(zf - depth)
dz[~af] = np.inf
k_at = np.argmin(dz, axis=2)
dist_at = np.take_along_axis(dz, k_at[..., None], axis=2)[..., 0]
lith_at = np.take_along_axis(np.asarray(lith), k_at[..., None], axis=2)[..., 0]
col_ok = dist_at <= 3.0  # cell center within 3 m of the requested depth

# KD-tree of valid column centers
tree, valid_ij = ref.build_column_tree(a0c, a1c, act, lith, np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r"))

# render reservoir on the seismic 1x grid (12.5 m) over the footprint
ax0 = np.arange(A0_LO, A0_HI)
ax1 = np.arange(A1_LO, A1_HI)
xx, yy = np.meshgrid(ax0.astype(np.float32), ax1.astype(np.float32), indexing="ij")
dist, nearest = tree.query(np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32), workers=-1)
sel = valid_ij[nearest]
si, sj = sel[:, 0], sel[:, 1]
lv = lith_at[si, sj]
ok = col_ok[si, sj] & (dist <= 3.5)
H, W = xx.shape
res_rgb = np.empty((H, W, 3), np.uint8)
res_rgb[...] = WHITE
flat = res_rgb.reshape(-1, 3)
flat[ok & (lv == 0)] = GRAY
flat[ok & ((lv == 1) | (lv == 2))] = YELLOW
res_rgb = flat.reshape(H, W, 3)
print(f"reservoir map shape {res_rgb.shape}  target px frac {(lv[ok] > 0).mean():.2%}")
res_clean = res_rgb.copy()  # color only, no grid lines (used for the zoom)

# draw white cell boundaries where the nearest native column (i,j) changes
col_id = sel.reshape(H, W, 2).astype(np.int64)
cid = col_id[..., 0] * 100000 + col_id[..., 1]
okm = ok.reshape(H, W)
bnd = np.zeros((H, W), bool)
bnd[1:, :] |= (cid[1:, :] != cid[:-1, :]) & okm[1:, :] & okm[:-1, :]
bnd[:, 1:] |= (cid[:, 1:] != cid[:, :-1]) & okm[:, 1:] & okm[:, :-1]
res_rgb[bnd] = WHITE

# seismic slice over the same axis region & depth
seis = np.load(SEIS, mmap_mode="r")
print("seismic shape", seis.shape)
si0, si1 = min(A0_HI, seis.shape[0]), min(A1_HI, seis.shape[1])
ssl = np.asarray(seis[A0_LO:si0, A1_LO:si1, min(sample_idx, seis.shape[2] - 1)]).astype(np.float32)
lo, hi = np.percentile(ssl[np.isfinite(ssl)], [2, 98])
sg = np.clip((ssl - lo) / max(hi - lo, 1e-6), 0, 1)
seis_rgb = (np.repeat((sg * 255).astype(np.uint8)[..., None], 3, axis=2))


def up(img, target=1300):
    h, w = img.shape[:2]
    s = max(1, int(max(h, w) / target))
    small = img[::s, ::s]
    return small


def save(img, name):
    Image.fromarray(img, "RGB").save(OUT / name)


# pad seismic to reservoir shape if needed for stacking
def fit(a, shape):
    out = np.zeros(shape + (3,), np.uint8)
    h = min(a.shape[0], shape[0]); w = min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    return out


seis_fit = fit(seis_rgb, (H, W))
gap = np.zeros((up(res_rgb).shape[0], 16, 3), np.uint8)
r_u = up(res_rgb); s_u = up(seis_fit)
hh = min(r_u.shape[0], s_u.shape[0]); ww = min(r_u.shape[1], s_u.shape[1])
combo = np.concatenate([r_u[:hh, :ww], gap[:hh], s_u[:hh, :ww]], axis=1)
save(combo, "zslice_reservoir_vs_seismic.png")

# zoom: small window so both grids are countable; grid lines on BOTH panels.
WIN = 20   # half-window in seismic-index px (12.5 m) -> 40 px = 500 m
F = 20     # upscale factor (each 12.5 m cell -> 20 px)
ys, xs = np.where((res_rgb == YELLOW).all(-1))
if xs.size:
    ci, cj = int(np.median(ys)), int(np.median(xs))
else:
    ci, cj = H // 2, W // 2
r0, r1 = max(0, ci - WIN), min(H, ci + WIN)
c0, c1 = max(0, cj - WIN), min(W, cj + WIN)
res_z = res_clean[r0:r1, c0:c1]
seis_z = seis_fit[r0:r1, c0:c1]
cid_z = cid[r0:r1, c0:c1]
h0, w0 = res_z.shape[:2]


def upz(img):
    return np.repeat(np.repeat(img, F, axis=0), F, axis=1)


RZ = upz(res_z)
SZ = upz(seis_z)

# seismic: white grid line at every 12.5 m cell boundary
for k in range(0, h0 + 1):
    SZ[min(k * F, SZ.shape[0] - 1), :] = WHITE
for k in range(0, w0 + 1):
    SZ[:, min(k * F, SZ.shape[1] - 1)] = WHITE

# reservoir: white line only where the native column (i,j) changes (~50 m cells)
hb = np.zeros((h0, w0), bool)
vb = np.zeros((h0, w0), bool)
hb[1:, :] = cid_z[1:, :] != cid_z[:-1, :]
vb[:, 1:] = cid_z[:, 1:] != cid_z[:, :-1]
for ri, cjj in zip(*np.where(hb)):
    RZ[ri * F, cjj * F:(cjj + 1) * F] = WHITE
for ri, cjj in zip(*np.where(vb)):
    RZ[ri * F:(ri + 1) * F, cjj * F] = WHITE

hh = min(RZ.shape[0], SZ.shape[0]); ww = min(RZ.shape[1], SZ.shape[1])
gap2 = np.zeros((hh, 16, 3), np.uint8)
save(np.concatenate([RZ[:hh, :ww], gap2, SZ[:hh, :ww]], axis=1), "zslice_zoom_reservoir_vs_seismic.png")
print(f"zoom window {h0}x{w0} px (~{h0*12.5:.0f} m): reservoir ~{h0*12.5/50:.0f} cells, seismic {h0} cells")
print("DONE ->", OUT)
