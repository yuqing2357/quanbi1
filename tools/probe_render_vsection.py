"""Read-only: vertical-section demo. Reservoir inline section (native-column
render) with cell boundaries vs seismic same section, same physical window &
scale, both with grid lines. Shows reservoir ~50 m lateral / ~2 m vertical
(finer vertically than seismic's 10 m).
white=nodata gray=bg(lith0) yellow=target(lith1/2)."""
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

HWIN_M, VWIN_M, MPP = 600.0, 240.0, 0.8     # lateral window, depth window, display m/px

transform = ref.transform_from_metadata(SRC_META)
master = ref.find_master_grdecl(GRDECL)
spec = ref.find_specgrid(master)
coord = ref.read_coord(master.with_name(master.stem + "_COORD.GRDECL"), spec)
a0c, a1c = ref.column_centers_axis(coord, transform)
del coord

lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")
poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")

# pick the column with the most target cells -> defines inline + window center
lt = np.asarray(lith)
tgt_cnt = ((lt == 1) | (lt == 2)).sum(axis=2)
ti, tj = np.unravel_index(int(np.argmax(tgt_cnt)), tgt_cnt.shape)
a0_idx = float(a0c[ti, tj]); a1_idx = float(a1c[ti, tj])
zc_col = np.asarray(z[ti, tj]); tcol = (lt[ti, tj] == 1) | (lt[ti, tj] == 2)
d_center = float(np.median(zc_col[tcol]))
print(f"inline col=({ti},{tj}) axis0={a0_idx:.1f} axis1={a1_idx:.1f} depth~{d_center:.0f}m  targets={int(tgt_cnt[ti,tj])}")

tree, valid_ij = ref.build_column_tree(a0c, a1c, act, lith, poro)

# reservoir render at MPP, fixed inline axis0
nlat = int(HWIN_M / MPP); ndep = int(VWIN_M / MPP)
a1_vals = a1_idx + (np.arange(nlat) - nlat / 2) * (MPP / transform.axis1_spacing)
depths = d_center + (np.arange(ndep) - ndep / 2) * MPP
x = np.full(nlat, a0_idx, np.float32)
dist, nearest = tree.query(np.column_stack([x, a1_vals]).astype(np.float32), workers=-1)
sel = valid_ij[nearest]
lith2d = np.zeros((ndep, nlat), np.uint8)
cellid = np.full((ndep, nlat), -1, np.int64)
okcol = dist <= 3.5
for c in range(nlat):
    if not okcol[c]:
        continue
    ni, nj = int(sel[c, 0]), int(sel[c, 1])
    zcol = np.asarray(z[ni, nj]); acol = np.asarray(act[ni, nj]) > 0
    lcol = np.asarray(lith[ni, nj]); pcol = np.asarray(poro[ni, nj])
    valid = acol & (np.isfinite(pcol) | (lcol >= 0))
    k = ref.nearest_profile_indices(zcol, depths.astype(np.float32), valid)
    lv = lcol[k]; av = acol[k]
    dz = np.abs(zcol[k] - depths)
    pres = av & (dz <= 3.0)
    lb = (av & ((lv == 1) | (lv == 2))).astype(np.uint8)
    col_uid = np.int64(ni) * 100000 + nj
    cid = np.where(pres, col_uid * 2000 + k.astype(np.int64), np.int64(-1))
    lith2d[:, c] = np.where(pres, lb, 0)
    cellid[:, c] = cid
    lith2d[~pres, c] = 0

res = np.empty((ndep, nlat, 3), np.uint8); res[...] = WHITE
present = cellid >= 0
res[present & (lith2d == 0)] = GRAY
res[present & (lith2d == 1)] = YELLOW
# white cell boundaries where native cell id changes
b = np.zeros((ndep, nlat), bool)
b[1:, :] |= (cellid[1:, :] != cellid[:-1, :]) & present[1:, :] & present[:-1, :]
b[:, 1:] |= (cellid[:, 1:] != cellid[:, :-1]) & present[:, 1:] & present[:, :-1]
res[b] = WHITE
print(f"reservoir section {res.shape}  target frac {(lith2d[present]==1).mean():.1%}")

# seismic same section/window, upscale to MPP, draw 12.5x10 grid
seis = np.load(SEIS, mmap_mode="r")
a1_lo = int(round(a1_idx - HWIN_M / 2 / transform.axis1_spacing))
a1_hi = int(round(a1_idx + HWIN_M / 2 / transform.axis1_spacing))
s_lo = int(round((d_center - VWIN_M / 2) / transform.sample_spacing))
s_hi = int(round((d_center + VWIN_M / 2) / transform.sample_spacing))
a0i = int(round(a0_idx))
sb = np.asarray(seis[a0i, max(0, a1_lo):a1_hi, max(0, s_lo):s_hi]).astype(np.float32)  # (lat, depth)
sb = sb.T  # (depth, lat)
lo, hi = np.percentile(sb[np.isfinite(sb)], [2, 98])
sg = np.clip((sb - lo) / max(hi - lo, 1e-6), 0, 1)
sgray = np.repeat((sg * 255).astype(np.uint8)[..., None], 3, 2)
fv = int(round(transform.sample_spacing / MPP)); fh = int(round(transform.axis1_spacing / MPP))
SZ = np.repeat(np.repeat(sgray, fv, 0), fh, 1)
for kk in range(0, sgray.shape[0] + 1):
    SZ[min(kk * fv, SZ.shape[0] - 1), :] = WHITE
for kk in range(0, sgray.shape[1] + 1):
    SZ[:, min(kk * fh, SZ.shape[1] - 1)] = WHITE
print(f"seismic section {sgray.shape} -> {SZ.shape}")

hh = min(res.shape[0], SZ.shape[0]); ww = min(res.shape[1], SZ.shape[1])
gap = np.zeros((hh, 16, 3), np.uint8)
combo = np.concatenate([res[:hh, :ww], gap, SZ[:hh, :ww]], axis=1)
Image.fromarray(combo, "RGB").save(OUT / "vsection_reservoir_vs_seismic.png")
print(f"window {HWIN_M:.0f}m x {VWIN_M:.0f}m @ {MPP} m/px | LEFT=reservoir RIGHT=seismic -> {OUT}")
