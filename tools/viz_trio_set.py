"""A set of co-registered seismic | lithology | porosity comparison trios from the
baked reservoir npy (6.25/6.25/2 m) and seismic (12.5/12.5/10 m). White grid lines
at each volume's cell boundaries (grid auto-off when cells get too small)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from project_paths import DIAGNOSTICS_ROOT

RES = Path("/root/quanbi/data/reservoir/npy_625x625x2_v3")
SEIS = Path("/root/quanbi/data/seismic/YJ-ALL-SEISMIC.npy")
OUT = DIAGNOSTICS_ROOT / "trio_set"
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((RES / "metadata.json").read_text())
org0 = meta["seismic_index_origin"]["axis0"]
org1 = meta["seismic_index_origin"]["axis1"]
depth0 = meta["seismic_index_origin"]["sample"] * 10.0
N0, N1, NZ = meta["shape"]

lith = np.load(RES / "lithology_binary_uint8.npy", mmap_mode="r")
poro = np.load(RES / "porosity_float16.npy", mmap_mode="r")
seis = np.load(SEIS, mmap_mode="r")
SA0, SA1, SAS = seis.shape
WHITE = np.array([255, 255, 255], np.uint8)


def color_seismic(a):
    f = np.isfinite(a)
    lo, hi = np.percentile(a[f], [2, 98]) if f.any() else (0, 1)
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
    out = np.zeros(P.shape + (3,), np.float32)
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]; x1, c1 = stops[i + 1]
        m = (v >= x0) & (v <= x1)
        t = (v[m] - x0) / (x1 - x0)
        for ch in range(3):
            out[m, ch] = c0[ch] + t * (c1[ch] - c0[ch])
    rgb = np.full(P.shape + (3,), 255, np.uint8)
    fin = np.isfinite(P)
    rgb[fin] = out[fin].astype(np.uint8)
    return rgb


def up_grid(img, fx, fy, draw):
    up = np.repeat(np.repeat(img, fy, 0), fx, 1)
    if draw:
        for r in range(img.shape[0] + 1):
            up[min(r * fy, up.shape[0] - 1), :] = WHITE
        for c in range(img.shape[1] + 1):
            up[:, min(c * fx, up.shape[1] - 1)] = WHITE
    return up


def trio(name, axis, idx, lat_lo, lat_hi, ss_lo, ss_hi):
    k_lo = int((ss_lo * 10 - depth0) // 2); k_hi = int((ss_hi * 10 - depth0) // 2)
    if axis == 0:   # inline: fixed axis0=idx, lateral=axis1
        o0 = (idx - org0) * 2
        o1a, o1b = (lat_lo - org1) * 2, (lat_hi - org1) * 2
        Ls = np.asarray(lith[o0, o1a:o1b, k_lo:k_hi]).T
        Ps = np.asarray(poro[o0, o1a:o1b, k_lo:k_hi]).T.astype(np.float32)
        Ss = np.asarray(seis[idx, lat_lo:lat_hi, ss_lo:ss_hi]).T.astype(np.float32)
    else:           # xline: fixed axis1=idx, lateral=axis0
        o1 = (idx - org1) * 2
        o0a, o0b = (lat_lo - org0) * 2, (lat_hi - org0) * 2
        Ls = np.asarray(lith[o0a:o0b, o1, k_lo:k_hi]).T
        Ps = np.asarray(poro[o0a:o0b, o1, k_lo:k_hi]).T.astype(np.float32)
        Ss = np.asarray(seis[lat_lo:lat_hi, idx, ss_lo:ss_hi]).T.astype(np.float32)
    width_m = (lat_hi - lat_lo) * 12.5
    mpp = width_m / 1100.0
    fxr = max(1, round(6.25 / mpp)); fyr = max(1, round(2.0 / mpp))
    fxs = max(1, round(12.5 / mpp)); fys = max(1, round(10.0 / mpp))
    draw = fxr >= 5
    pres = np.isfinite(Ps)
    S = up_grid(color_seismic(Ss), fxs, fys, draw)
    L = up_grid(color_lith(Ls, pres), fxr, fyr, draw)
    P = up_grid(color_poro(Ps), fxr, fyr, draw)
    h = min(S.shape[0], L.shape[0], P.shape[0])
    gap = np.zeros((h, 18, 3), np.uint8)
    combo = np.concatenate([S[:h], gap, L[:h], gap, P[:h]], axis=1)
    Image.fromarray(combo, "RGB").save(OUT / f"{name}.png")
    tf = (Ls[pres] == 1).mean() if pres.any() else 0
    print(f"{name}: axis={axis} idx={idx} win {width_m:.0f}m x {(ss_hi-ss_lo)*10}m  "
          f"grid={draw} present={pres.mean():.0%} target={tf:.0%}")


def section_masks(axis, idx):
    if axis == 0:
        o0 = (idx - org0) * 2
        L = np.asarray(lith[o0])
        P = np.isfinite(np.asarray(poro[o0]).astype(np.float32))
    else:
        o1 = (idx - org1) * 2
        L = np.asarray(lith[:, o1, :])
        P = np.isfinite(np.asarray(poro[:, o1, :]).astype(np.float32))
    return L, P  # (n_lat_res, NZ)


def pick_center(axis, idx, want, present_min=0.8, hl_seis=12, hs_seis=5):
    """Slide a window over the section; return (lat_center, ss_center) whose
    target fraction is closest to `want` among windows with present>=present_min."""
    L, P = section_masks(axis, idx)
    nlat, ndep = L.shape
    hl, hs = hl_seis * 2, hs_seis * 5
    org = org1 if axis == 0 else org0
    best = None
    for sc in range(hl, nlat - hl, 16):
        for kc in range(hs, ndep - hs, 25):
            pm = P[sc - hl:sc + hl, kc - hs:kc + hs]
            if pm.mean() < present_min:
                continue
            win = L[sc - hl:sc + hl, kc - hs:kc + hs]
            tf = (win[pm] == 1).mean()
            score = abs(tf - want)
            if best is None or score < best[0]:
                best = (score, sc, kc, tf)
    if best is None:
        return None
    _, sc, kc, tf = best
    return org + sc // 2, int(round((depth0 + kc * 2) / 10)), tf


INLINE = 479
a = pick_center(0, INLINE, 0.82); b = pick_center(0, INLINE, 0.45)
laA, ssA = a[0], a[1]
laB, ssB = b[0], b[1]
XLINE = laA
c = pick_center(1, XLINE, 0.7)
laC, ssC = (c[0], c[1]) if c else (928, ssA)
print(f"A inline{INLINE}@(axis1={laA},samp={ssA},tf={a[2]:.0%})  "
      f"B mixed@(axis1={laB},samp={ssB},tf={b[2]:.0%})  C xline{XLINE}@(axis0={laC},samp={ssC})")

trio("A_inline_interior_zoom", 0, INLINE, laA - 12, laA + 12, ssA - 5, ssA + 5)
trio("B_inline_mixed_zoom", 0, INLINE, laB - 16, laB + 16, ssB - 7, ssB + 7)
trio("C_xline_interior_zoom", 1, XLINE, laC - 12, laC + 12, ssC - 5, ssC + 5)
trio("D_inline_wide", 0, INLINE, laA - 90, laA + 90, ssA - 28, ssA + 32)
trio("E_xline_wide", 1, XLINE, laC - 90, laC + 90, ssC - 28, ssC + 32)
print("DONE -> ORDER per image: LEFT=seismic MID=lithology RIGHT=porosity")
