"""Read-only inspection + slice rendering for numpy_3x_direct vs numpy_3x.

Does NOT touch the API service. Memory-maps the volumes, computes subsampled
stats, auto-picks representative slices that actually contain reservoir data,
and renders a 3-colour composite (white=nodata, gray=non-target, yellow=target)
side by side (direct | old). Saves PNGs to OUT_DIR.
"""
from __future__ import annotations

import json
import os

import numpy as np

DIRECT = "/root/quanbi/data/reservoir/numpy_3x_direct"
OLD = "/root/quanbi/data/reservoir/numpy_3x"
OUT_DIR = "/root/quanbi/runtime/inspect_3x_direct"
os.makedirs(OUT_DIR, exist_ok=True)

WHITE = np.array([255, 255, 255], np.uint8)
GRAY = np.array([150, 150, 150], np.uint8)
YELLOW = np.array([245, 214, 45], np.uint8)

try:
    from PIL import Image

    HAVE_PIL = True
except Exception as exc:  # noqa: BLE001
    HAVE_PIL = False
    print("PIL MISSING:", exc)


def load(path):
    return np.load(path, mmap_mode="r")


def save_png(rgb, name):
    if HAVE_PIL:
        Image.fromarray(rgb, "RGB").save(os.path.join(OUT_DIR, name))
    else:
        np.save(os.path.join(OUT_DIR, name + ".npy"), rgb)


def downsample_rgb(rgb, max_dim=1400):
    h, w = rgb.shape[:2]
    step = max(1, int(max(h, w) / max_dim))
    return rgb[::step, ::step]


def composite(lith2d, poro2d):
    """lith2d uint8 {0,1}; poro2d float (NaN=nodata). Returns RGB (H,W,3)."""
    lith2d = np.asarray(lith2d)
    poro2d = np.asarray(poro2d)
    valid = np.isfinite(poro2d.astype(np.float32))
    rgb = np.empty(lith2d.shape + (3,), np.uint8)
    rgb[...] = WHITE
    rgb[valid & (lith2d == 0)] = GRAY
    rgb[lith2d == 1] = YELLOW
    return rgb


def section_image(rgb):
    # transpose so depth (axis -1 of the slice) is vertical, shallow on top
    return np.transpose(rgb, (1, 0, 2))


def isolated_fraction(mask2d):
    m = mask2d.astype(bool)
    if m.sum() == 0:
        return 0.0
    neigh = np.zeros_like(m)
    neigh[:-1] |= m[1:]
    neigh[1:] |= m[:-1]
    neigh[:, :-1] |= m[:, 1:]
    neigh[:, 1:] |= m[:, :-1]
    isolated = m & ~neigh
    return float(isolated.sum()) / float(m.sum())


def subsample_stats(lith, poro, label):
    sl = lith[::5, ::5, ::3]
    sp = poro[::5, ::5, ::3].astype(np.float32)
    vals, counts = np.unique(sl, return_counts=True)
    total = sl.size
    finite = np.isfinite(sp)
    print(f"\n[{label}] subsampled stride(5,5,3) n={total}")
    print("  lith values:", {int(v): int(c) for v, c in zip(vals, counts)})
    print(f"  lith target(==1) frac: {float((sl == 1).sum()) / total:.4%}")
    print(f"  poro NaN frac: {float((~finite).sum()) / total:.4%}")
    if finite.any():
        fv = sp[finite]
        print(f"  poro finite: min={fv.min():.4f} max={fv.max():.4f} mean={fv.mean():.4f}")
    # populated range per axis (full-index) from target mask
    tgt = sl == 1
    rng = {}
    for ax, step, full in ((0, 5, lith.shape[0]), (1, 5, lith.shape[1]), (2, 3, lith.shape[2])):
        prof = tgt.any(axis=tuple(a for a in (0, 1, 2) if a != ax))
        idx = np.where(prof)[0]
        if idx.size:
            rng[ax] = (int(idx.min() * step), int(idx.max() * step))
        else:
            rng[ax] = (0, full - 1)
    print("  target populated full-index range per axis:", rng)
    return rng


def pick_indices(rng):
    lo, hi = rng
    span = hi - lo
    return [int(lo + span * f) for f in (0.35, 0.5, 0.65)]


def main():
    dl, dp = load(f"{DIRECT}/lithology_binary_3x_uint8.npy"), load(f"{DIRECT}/porosity_3x_float16.npy")
    ol, op = load(f"{OLD}/lithology_binary_3x_uint8.npy"), load(f"{OLD}/porosity_3x_float16.npy")
    print("direct lith", dl.shape, dl.dtype, "| poro", dp.shape, dp.dtype)
    print("old    lith", ol.shape, ol.dtype, "| poro", op.shape, op.dtype)
    assert dl.shape == ol.shape == dp.shape == op.shape

    rng = subsample_stats(dl, dp, "DIRECT")
    subsample_stats(ol, op, "OLD")

    axes = {"inline": 0, "xline": 1, "depth": 2}
    report = []
    for name, ax in axes.items():
        for idx in pick_indices(rng[ax]):
            if ax == 0:
                d_rgb = composite(dl[idx], dp[idx]); o_rgb = composite(ol[idx], op[idx]); sect = True
            elif ax == 1:
                d_rgb = composite(dl[:, idx], dp[:, idx]); o_rgb = composite(ol[:, idx], op[:, idx]); sect = True
            else:
                d_rgb = composite(dl[:, :, idx], dp[:, :, idx]); o_rgb = composite(ol[:, :, idx], op[:, :, idx]); sect = False
            d_iso = isolated_fraction((d_rgb == YELLOW).all(-1))
            o_iso = isolated_fraction((o_rgb == YELLOW).all(-1))
            d_tf = float((d_rgb == YELLOW).all(-1).mean())
            o_tf = float((o_rgb == YELLOW).all(-1).mean())
            if sect:
                d_rgb, o_rgb = section_image(d_rgb), section_image(o_rgb)
            d_rgb, o_rgb = downsample_rgb(d_rgb), downsample_rgb(o_rgb)
            gap = np.full((d_rgb.shape[0], 12, 3), 0, np.uint8)
            combo = np.concatenate([d_rgb, gap, o_rgb], axis=1)
            fn = f"{name}_{idx}.png"
            save_png(combo, fn)
            report.append((fn, d_tf, o_tf, d_iso, o_iso))
            print(f"  {name} idx={idx}: target frac direct={d_tf:.4%} old={o_tf:.4%} | "
                  f"isolated(speckle) direct={d_iso:.4%} old={o_iso:.4%} -> {fn} (LEFT=direct RIGHT=old)")

    with open(os.path.join(OUT_DIR, "report.json"), "w") as fh:
        json.dump({"slices": report, "have_pil": HAVE_PIL}, fh, indent=2)
    print("\nDONE. PNGs in", OUT_DIR)


if __name__ == "__main__":
    main()
