"""Read-only: quantify how many thin target beds 1 m POINT sampling drops,
vs INTERVAL aggregation, on the native reservoir columns.

Cell vertical boundaries are taken as midpoints between adjacent z_centers
(this is exactly what 'nearest z_center' sampling implies); thickness derived
from that. No ZCORN parse needed for this question — see note in chat.
Target = native lithology in {1,2} (porous classes); background = 0.
"""
import sys

import numpy as np

D = "/root/quanbi/data/reservoir/numpy"
lith = np.load(f"{D}/lithology_native_i_j_k.npy", mmap_mode="r")
act = np.load(f"{D}/actnum_native_i_j_k.npy", mmap_mode="r")
z = np.load(f"{D}/z_center_native_i_j_k.npy", mmap_mode="r")
nx, ny, nz = lith.shape

STRIDE = 2          # sample every 2nd column in i and j
STEP = 1.0          # render vertical step (m)

tgt_total = 0
tgt_thick_total = 0.0
tgt_dropped_pt = 0           # target cells never hit by 1 m point sampling
tgt_thin_total = 0          # target cells thinner than STEP
tgt_thin_dropped_pt = 0
cap_thick_pt = 0.0          # target thickness "seen" by point sampling
cap_thick_iv = 0.0          # target thickness "seen" by interval aggregation
cols_with_drop = 0
cols_done = 0
thin_hist = np.zeros(6, np.int64)  # <0.5, 0.5-1, 1-2, 2-3, 3-4, >=4 m

for i in range(0, nx, STRIDE):
    li = np.asarray(lith[i])          # (ny, nz)
    ai = np.asarray(act[i]) > 0
    zi = np.asarray(z[i]).astype(np.float64)
    for j in range(0, ny, STRIDE):
        valid = ai[j] & (li[j] >= 0)
        if valid.sum() < 3:
            continue
        zc = zi[j][valid]
        lv = li[j][valid]
        order = np.argsort(zc)
        zc = zc[order]
        lv = lv[order]
        m = zc.size
        # boundaries = midpoints, outer extrapolated
        mid = (zc[:-1] + zc[1:]) * 0.5
        b = np.empty(m + 1)
        b[1:-1] = mid
        b[0] = zc[0] - (zc[1] - zc[0]) * 0.5
        b[-1] = zc[-1] + (zc[-1] - zc[-2]) * 0.5
        thick = np.diff(b)                       # per-cell thickness
        is_tgt = (lv == 1) | (lv == 2)
        if not is_tgt.any():
            cols_done += 1
            continue
        # thickness histogram of target cells
        tt = thick[is_tgt]
        thin_hist[0] += int((tt < 0.5).sum())
        thin_hist[1] += int(((tt >= 0.5) & (tt < 1)).sum())
        thin_hist[2] += int(((tt >= 1) & (tt < 2)).sum())
        thin_hist[3] += int(((tt >= 2) & (tt < 3)).sum())
        thin_hist[4] += int(((tt >= 3) & (tt < 4)).sum())
        thin_hist[5] += int((tt >= 4).sum())

        tgt_total += int(is_tgt.sum())
        tgt_thick_total += float(tt.sum())
        tgt_thin_total += int((tt < STEP).sum())

        # 1 m point sampling: sample centers across the column
        depths = np.arange(b[0] + STEP * 0.5, b[-1], STEP)
        if depths.size == 0:
            cols_done += 1
            continue
        cell_of = np.clip(np.searchsorted(b, depths, side="right") - 1, 0, m - 1)
        sampled_tgt = is_tgt[cell_of]
        cap_thick_pt += float(sampled_tgt.sum()) * STEP
        hit = np.zeros(m, bool)
        hit[np.unique(cell_of)] = True
        dropped = is_tgt & ~hit
        nd = int(dropped.sum())
        tgt_dropped_pt += nd
        tgt_thin_dropped_pt += int((dropped & (thick < STEP)).sum())
        if nd:
            cols_with_drop += 1

        # interval aggregation: band [d-step/2, d+step/2] -> target if any target cell overlaps.
        # bands tile the axis, so capture = (# bands whose interval contains any target cell)*step
        # a band centered at d is target if any target boundary-interval overlaps [d-.5,d+.5]
        tgt_lo = b[:-1][is_tgt]
        tgt_hi = b[1:][is_tgt]
        band_lo = depths - STEP * 0.5
        band_hi = depths + STEP * 0.5
        # band target if exists target cell with lo<band_hi and hi>band_lo
        any_tgt = np.zeros(depths.size, bool)
        for clo, chi in zip(tgt_lo, tgt_hi):
            any_tgt |= (clo < band_hi) & (chi > band_lo)
        cap_thick_iv += float(any_tgt.sum()) * STEP
        cols_done += 1

print(f"columns processed: {cols_done}  (stride {STRIDE})")
print(f"target cells: {tgt_total}   true target thickness sum: {tgt_thick_total/1000:.1f} km")
print()
print("target-cell thickness distribution:")
labels = ["<0.5m", "0.5-1m", "1-2m", "2-3m", "3-4m", ">=4m"]
for lab, c in zip(labels, thin_hist):
    print(f"   {lab:>7}: {c:>9}  ({c/max(tgt_total,1):.1%})")
print()
print(f"target cells thinner than {STEP} m: {tgt_thin_total} ({tgt_thin_total/max(tgt_total,1):.1%})")
print(f"--- 1 m POINT sampling ---")
print(f"   target cells DROPPED (never sampled): {tgt_dropped_pt} ({tgt_dropped_pt/max(tgt_total,1):.1%})")
print(f"   of which thin (<{STEP}m): {tgt_thin_dropped_pt} ({tgt_thin_dropped_pt/max(tgt_thin_total,1):.1%} of thin)")
print(f"   columns with >=1 dropped target bed: {cols_with_drop}")
print(f"   target thickness captured / true: {cap_thick_pt/max(tgt_thick_total,1e-9):.3f}")
print(f"--- 1 m INTERVAL aggregation (any target in band) ---")
print(f"   target cells dropped: 0 (bands tile the axis)")
print(f"   target thickness captured / true: {cap_thick_iv/max(tgt_thick_total,1e-9):.3f}")
