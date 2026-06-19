"""Read-only probe of native GRDECL grid geometry: is it a regular box or
corner-point (variable thickness / folded layers)?"""
import numpy as np

D = "/root/quanbi/data/reservoir/numpy"
z = np.load(f"{D}/z_center_native_i_j_k.npy", mmap_mode="r")  # [nx,ny,nz]
act = np.load(f"{D}/actnum_native_i_j_k.npy", mmap_mode="r")
nx, ny, nz = z.shape
print("shape (nx,ny,nz):", z.shape)

# 1) within-column thickness: is dz constant per column?
print("\n--- per-column cell thickness (dz = z[k+1]-z[k]) ---")
for (i, j) in [(nx//2, ny//2), (nx//3, ny//3), (2*nx//3, 2*ny//3)]:
    col = np.asarray(z[i, j, :]).astype(np.float64)
    a = np.asarray(act[i, j, :]) > 0
    if a.sum() < 2:
        print(f"  col({i},{j}): <2 active cells"); continue
    cv = col[a]
    dz = np.diff(np.sort(cv))
    dz = dz[dz > 0]
    print(f"  col({i},{j}): active={a.sum()}  z[{cv.min():.1f},{cv.max():.1f}]  "
          f"dz min/med/max = {dz.min():.2f}/{np.median(dz):.2f}/{dz.max():.2f} m")

# 2) folding: for a fixed logical layer k, how much does z vary across (i,j)?
print("\n--- folding: z spread across columns at a fixed logical layer k ---")
for k in [nz//4, nz//2, 3*nz//4]:
    plane = np.asarray(z[:, :, k]).astype(np.float64)
    a = np.asarray(act[:, :, k]) > 0
    if a.sum() < 10:
        print(f"  k={k}: <10 active"); continue
    pv = plane[a]
    print(f"  k={k}: active_cols={a.sum()}  z range across map = "
          f"{pv.min():.1f}..{pv.max():.1f} m  (spread {pv.max()-pv.min():.0f} m)")

# 3) overall: global dz distribution over a sample of active cells
print("\n--- global thickness distribution (sampled) ---")
zs = np.asarray(z[::7, ::7, :]).astype(np.float64)
acts = np.asarray(act[::7, ::7, :]) > 0
dz_all = []
for i in range(zs.shape[0]):
    for j in range(zs.shape[1]):
        cv = zs[i, j, acts[i, j]]
        if cv.size >= 2:
            d = np.diff(np.sort(cv))
            dz_all.append(d[d > 0])
if dz_all:
    dz_all = np.concatenate(dz_all)
    print(f"  n={dz_all.size}  dz p5/p50/p95 = "
          f"{np.percentile(dz_all,5):.2f}/{np.percentile(dz_all,50):.2f}/{np.percentile(dz_all,95):.2f} m  "
          f"min={dz_all.min():.2f} max={dz_all.max():.2f}")
