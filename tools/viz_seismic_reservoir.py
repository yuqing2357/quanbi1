"""Co-registered zoom: seismic | reservoir lithology | reservoir porosity, at the
same inline + window, each with white grid lines at its own cell boundaries.
Reservoir read from the v3 6.25/6.25/2 m volume; seismic 12.5/12.5/10 m."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from project_paths import DIAGNOSTICS_ROOT

RES = Path("/root/quanbi/data/reservoir/npy_625x625x2_v3")
SEIS = Path("/root/quanbi/data/seismic/YJ-ALL-SEISMIC.npy")
OUT = DIAGNOSTICS_ROOT / "bake_check"
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((RES / "metadata.json").read_text())
org0 = meta["seismic_index_origin"]["axis0"]
org1 = meta["seismic_index_origin"]["axis1"]
orgs = meta["seismic_index_origin"]["sample"]      # depth0 = orgs*10
depth0 = orgs * 10.0

lith = np.load(RES / "lithology_binary_uint8.npy", mmap_mode="r")
poro = np.load(RES / "porosity_float16.npy", mmap_mode="r")
seis = np.load(SEIS, mmap_mode="r")

AX0 = 479                          # seismic inline
o0 = (AX0 - org0) * 2              # reservoir axis0 output index

# find a target-rich centre on this inline
Lf = np.asarray(lith[o0])
ys, xs = np.where(Lf == 1)
o1c, kc = int(np.median(ys)), int(np.median(xs))
c1c = org1 + o1c // 2                          # seismic axis1 centre
ssc = int(round((depth0 + kc * 2) / 10.0))     # seismic sample centre

# window: 24 seismic axis1 idx (=300 m), 10 seismic samples (=100 m)
s1_lo, s1_hi = c1c - 12, c1c + 12
ss_lo, ss_hi = ssc - 5, ssc + 5
o1_lo, o1_hi = (s1_lo - org1) * 2, (s1_hi - org1) * 2
k_lo, k_hi = (ss_lo * 10 - int(depth0)) // 2, (ss_hi * 10 - int(depth0)) // 2
print(f"inline axis0={AX0} (o0={o0})  axis1 seismic[{s1_lo},{s1_hi}]  sample[{ss_lo},{ss_hi}]  depth~{ss_lo*10}-{ss_hi*10}m")

# --- extract sub-sections (rows=depth, cols=lateral) ---
seis_sec = np.asarray(seis[AX0, s1_lo:s1_hi, ss_lo:ss_hi]).T.astype(np.float32)   # (10,24)
lith_sec = np.asarray(lith[o0, o1_lo:o1_hi, k_lo:k_hi]).T                          # (50,48)
poro_sec = np.asarray(poro[o0, o1_lo:o1_hi, k_lo:k_hi]).T.astype(np.float32)       # (50,48)

WHITE = np.array([255, 255, 255], np.uint8)


def color_seismic(a):
    lo, hi = np.percentile(a[np.isfinite(a)], [2, 98])
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.repeat((g * 255).astype(np.uint8)[..., None], 3, 2)


def color_lith(L, present):
    rgb = np.full(L.shape + (3,), 255, np.uint8)
    rgb[present & (L == 0)] = (150, 150, 150)
    rgb[L == 1] = (245, 214, 45)
    return rgb


def color_poro(P):
    stops = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
             (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    v = np.clip(P / 0.30, 0, 1)
    rgb = np.full(P.shape + (3,), 255, np.uint8)
    fin = np.isfinite(P)
    out = np.zeros(P.shape + (3,), np.float32)
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]; x1, c1 = stops[i + 1]
        m = (v >= x0) & (v <= x1)
        t = (v[m] - x0) / (x1 - x0)
        for ch in range(3):
            out[m, ch] = c0[ch] + t * (c1[ch] - c0[ch])
    rgb[fin] = out[fin].astype(np.uint8)
    return rgb


def grid(img, fx, fy):
    up = np.repeat(np.repeat(img, fy, 0), fx, 1)
    for r in range(0, img.shape[0] + 1):
        up[min(r * fy, up.shape[0] - 1), :] = WHITE
    for c in range(0, img.shape[1] + 1):
        up[:, min(c * fx, up.shape[1] - 1)] = WHITE
    return up


present = np.isfinite(poro_sec)
S = grid(color_seismic(seis_sec), 42, 35)                 # 24*42=1008 x 10*35=350
L = grid(color_lith(lith_sec, present), 21, 7)            # 48*21=1008 x 50*7=350
P = grid(color_poro(poro_sec), 21, 7)
h = min(S.shape[0], L.shape[0], P.shape[0])
gap = np.zeros((h, 18, 3), np.uint8)
combo = np.concatenate([S[:h], gap, L[:h], gap, P[:h]], axis=1)
Image.fromarray(combo, "RGB").save(OUT / "trio_seismic_lith_poro.png")
print(f"panels seismic{S.shape} lith{L.shape} poro{P.shape}  target_frac={(lith_sec[present]==1).mean():.1%}")
print("ORDER: LEFT=seismic(12.5x10m)  MID=lithology(6.25x2m)  RIGHT=porosity(6.25x2m) -> trio_seismic_lith_poro.png")
