"""Cross-check lithology binary encoding against porosity (read-only)."""
import numpy as np

DIRECT = "/root/quanbi/data/reservoir/numpy_3x_direct"
NATIVE = "/root/quanbi/data/reservoir/numpy/lithology_native_i_j_k.npy"
NATIVE_PORO = "/root/quanbi/data/reservoir/numpy/porosity_native_i_j_k.npy"

dl = np.load(f"{DIRECT}/lithology_binary_3x_uint8.npy", mmap_mode="r")
dp = np.load(f"{DIRECT}/porosity_3x_float16.npy", mmap_mode="r")
sl = dl[::5, ::5, ::3]
sp = dp[::5, ::5, ::3].astype(np.float32)
fin = np.isfinite(sp)
for v in (0, 1):
    m = (sl == v) & fin
    if m.any():
        pv = sp[m]
        print(f"binary lith=={v}: n={int(m.sum())} poro mean={pv.mean():.4f} median={np.median(pv):.4f} p90={np.percentile(pv,90):.4f}")

# native 3-class distribution + porosity per native class (native grid is small)
nl = np.load(NATIVE)
npo = np.load(NATIVE_PORO).astype(np.float32)
print("\nnative lith dtype", nl.dtype, "shape", nl.shape)
vals, counts = np.unique(nl, return_counts=True)
print("native lith values:", {int(v): int(c) for v, c in zip(vals, counts)})
finn = np.isfinite(npo)
for v in vals:
    m = (nl == v) & finn
    if m.any():
        pv = npo[m]
        print(f"native lith=={int(v)}: n={int(m.sum())} poro mean={pv.mean():.4f} median={np.median(pv):.4f}")
