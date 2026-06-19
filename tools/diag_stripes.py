"""Diagnose the vertical nodata stripes: along inline axis0=479, measure the
nearest-native-column distance vs axis1, and inspect the native column layout
near that inline (are there gaps > the support gate?)."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

NUMPY = Path("/root/quanbi/data/reservoir/numpy")
geom = json.loads((NUMPY / "column_geometry.json").read_text())
centers = np.load(NUMPY / "column_centers_axis.npy")   # (nx,ny,2) seismic-idx
valid = np.load(NUMPY / "column_valid.npy")
vij = np.argwhere(valid)
pts = centers[valid]
tree = cKDTree(pts.astype(np.float32))

AX0 = 479.0
GATE = 3.5
org1 = -6
# baked lateral grid along this inline (6.25 m => /2 of seismic idx)
N1 = 2206
p1 = org1 + np.arange(N1) / 2.0
dist, nn = tree.query(np.column_stack([np.full(N1, AX0, np.float32), p1.astype(np.float32)]))
nod = dist > GATE
print(f"inline axis0={AX0}: nodata cols {nod.sum()}/{N1} ({nod.mean():.1%})")
print(f"nearest-dist along axis1: p50={np.percentile(dist,50):.2f} p90={np.percentile(dist,90):.2f} "
      f"p99={np.percentile(dist,99):.2f} max={dist.max():.2f} (idx units, *12.5=m)")

# runs of nodata (stripe widths in baked columns)
runs = []
i = 0
while i < N1:
    if nod[i]:
        j = i
        while j < N1 and nod[j]:
            j += 1
        runs.append((i, j - i))
        i = j
    else:
        i += 1
# only count stripes flanked by data (interior), not the big exterior tails
interior = [(s, w) for (s, w) in runs if s > 0 and s + w < N1 and not nod[max(0, s - 1)] is True]
print(f"total nodata runs={len(runs)}  widths(baked cols): "
      f"{sorted(set(w for _, w in runs))[:12]}")
print(f"interior-ish stripe count (width<=6): {sum(1 for _,w in runs if w<=6)}")

# native column layout near this inline: columns with |axis0-479|<perp
for perp in (2.0, 3.5, 5.0):
    near = np.abs(pts[:, 0] - AX0) <= perp
    a1 = np.sort(pts[near, 1])
    if a1.size < 3:
        print(f"perp<={perp}: only {a1.size} columns"); continue
    d = np.diff(a1)
    d = d[d > 1e-3]
    print(f"perp<={perp} idx: {near.sum()} cols  axis1-spacing p50={np.median(d):.2f} "
          f"p95={np.percentile(d,95):.2f} max={d.max():.2f}  (gaps>{2*GATE}: {(d>2*GATE).sum()})")

# how far is the nearest column in axis0 from this inline, per lateral?
a0_of_nearest = pts[nn, 0]
print(f"axis0 offset of nearest column from {AX0}: "
      f"p50={np.percentile(np.abs(a0_of_nearest-AX0),50):.2f} "
      f"p95={np.percentile(np.abs(a0_of_nearest-AX0),95):.2f} max={np.abs(a0_of_nearest-AX0).max():.2f}")
