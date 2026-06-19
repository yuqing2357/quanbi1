"""Render the SAME reservoir section under several support-distance gates and
stack them for comparison. Shows how the gate fills interior stripes vs bleeds
the exterior. Lithology only (white=nodata gray=bg yellow=target)."""
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

geom = json.loads((NUMPY / "column_geometry.json").read_text())
spz = geom["sample_spacing_m"]
depth0_default = None
centers = np.load(NUMPY / "column_centers_axis.npy")
valid = np.load(NUMPY / "column_valid.npy")
valid_ij = np.argwhere(valid).astype(np.int32)
tree = cKDTree(centers[valid].astype(np.float32))
lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")
act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")

# reservoir footprint origin in seismic idx (axis1, sample) for depth mapping
a1v = centers[..., 1][valid]
ORG1 = int(np.floor(a1v.min()))
zs = np.asarray(z[::2, ::2, :]); acts = np.asarray(act[::2, ::2, :]) > 0
zv = zs[acts & np.isfinite(zs)]
S_LO = int(np.floor((zv.min() - 20) / spz))
DEPTH0 = S_LO * spz
ORG0 = int(np.floor(centers[..., 0][valid].min()))

GATES = [3.5, 6.0, 10.0, 30.0, 80.0]   # idx units (x12.5 = m)
V_STEP = 2.0


def render_section(axis, idx, lat_lo, lat_hi, ss_lo, ss_hi):
    """Return per-lateral nearest-column lith profile + present + dist."""
    depths = np.arange(ss_lo * 10.0, ss_hi * 10.0, V_STEP)
    nd = depths.size
    lat = np.arange(lat_lo, lat_hi)
    if axis == 0:
        q = np.column_stack([np.full(lat.size, float(idx)), lat.astype(np.float32)])
    else:
        q = np.column_stack([lat.astype(np.float32), np.full(lat.size, float(idx))])
    dist, nn = tree.query(q.astype(np.float32))
    cols = valid_ij[nn]
    nlat = lat.size
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
        lp = bake.lith_profile(zc, lcol, depths, V_STEP)
        lith2d[cells] = lp
        pres2d[cells] = present
    return lith2d, pres2d, dist, depths


def panel(lith2d, pres2d, dist, gate, width=1200):
    keep = (dist <= gate)[:, None] & pres2d
    img = np.full(lith2d.shape + (3,), 255, np.uint8)   # (nlat, nd)
    img[keep & (lith2d == 0)] = (150, 150, 150)
    img[keep & (lith2d == 1)] = (245, 214, 45)
    sec = np.transpose(img, (1, 0, 2))                  # (nd, nlat)
    s = max(1, int(sec.shape[1] / width))
    sec = sec[::s, ::s]
    pil = Image.fromarray(sec, "RGB")
    band = Image.new("RGB", (pil.width, 22), (20, 20, 20))
    ImageDraw.Draw(band).text((6, 5), f"gate = {gate:.1f} idx = {gate*12.5:.0f} m", fill=(255, 255, 255))
    out = Image.new("RGB", (pil.width, pil.height + 22), (20, 20, 20))
    out.paste(band, (0, 0)); out.paste(pil, (0, 22))
    return np.asarray(out)


def make(name, axis, idx, lat_lo, lat_hi, ss_lo, ss_hi):
    L, P, D, _ = render_section(axis, idx, lat_lo, lat_hi, ss_lo, ss_hi)
    panels = [panel(L, P, D, g) for g in GATES]
    w = max(p.shape[1] for p in panels)
    sep = np.full((6, w, 3), 60, np.uint8)
    rows = []
    for p in panels:
        if p.shape[1] < w:
            p = np.pad(p, ((0, 0), (0, w - p.shape[1]), (0, 0)), constant_values=20)
        rows.append(p); rows.append(sep)
    Image.fromarray(np.concatenate(rows[:-1], axis=0), "RGB").save(OUT / f"{name}.png")
    inter = int(((D > 3.5) & (D <= 12)).sum()); ext = int((D > 70).sum())
    print(f"{name}: axis={axis} idx={idx} lat[{lat_lo},{lat_hi}] | laterals with dist in(3.5,12]={inter} "
          f"(interior gaps), dist>70={ext} (exterior)")


# inline 479: frame the body axis1 extent (+margin to show exterior) from native columns
near0 = (np.abs(centers[..., 0] - 479) < 3) & valid
a1 = centers[..., 1][near0]
c1lo, c1hi = int(np.floor(a1.min())) - 30, int(np.ceil(a1.max())) + 30
make("inline479_gates", 0, 479, c1lo, c1hi, 300, 470)

# xline through a target-rich axis1; frame its axis0 extent from native columns
tcol = (((np.asarray(lith) == 1) | (np.asarray(lith) == 2)).any(axis=2)) & valid
ax1 = int(round(np.median(centers[..., 1][tcol])))
near1 = (np.abs(centers[..., 1] - ax1) < 3) & valid
a0 = centers[..., 0][near1]
c0lo, c0hi = int(np.floor(a0.min())) - 30, int(np.ceil(a0.max())) + 30
make("xline_gates", 1, ax1, c0lo, c0hi, 300, 470)
print("DONE -> top-to-bottom gates:", [f"{g*12.5:.0f}m" for g in GATES])
