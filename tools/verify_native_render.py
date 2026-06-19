"""Read-only verification of on-demand native-column slice rendering.

Proves two things, reusing the EXACT mapping/profile primitives from
create_reservoir_3x_direct_numpy.py (so the per-slice renderer is provably the
same logic as the dense rasteriser):

  (A) Correctness: rendering a slice at scale-3 sampling and applying the same
      support mask reproduces the dense numpy_3x_direct volume slice
      bit-for-bit (reports any mismatch).
  (B) Detail gain: rendering the same slice at fine 1 m vertical sampling
      recovers laminae that the 3.33 m dense raster smears. Saves 3-colour
      composites (white=nodata, gray=present non-target, yellow=target) with
      LEFT=on-demand 1 m, RIGHT=dense 3.33 m.

Does NOT touch the API service. Run from the project root on the server.

Colour note: in THIS dataset the binary target = porous native classes {1,2};
native 0 (gravel) is zero-porosity background. So yellow=native{1,2},
gray=present native 0, white=nodata. (Body-mesh names are misleading; porosity
cross-check confirmed the encoding.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import create_reservoir_3x_direct_numpy as ref  # noqa: E402
from project_paths import DIAGNOSTICS_ROOT

ROOT = Path("/root/quanbi")
NUMPY_DIR = ROOT / "data/reservoir/numpy"
GRDECL_DIR = ROOT / "data/reservoir/grdecl"
DIRECT_DIR = ROOT / "data/reservoir/numpy_3x_direct"
SUPPORT_DIR = ROOT / "data/reservoir/numpy_3x"
SRC_META = NUMPY_DIR / "metadata.json"
REF_3X_META = SUPPORT_DIR / "metadata.json"
OUT = DIAGNOSTICS_ROOT / "verify_native_render"
SCALE = 3
MAX_COL_DIST = 1.5            # rasteriser fallback gate (1x index units)
DETAIL_MAX_COL_DIST = 3.5     # ~native column spacing (~3.3 idx ≈ 40 m); fills Voronoi gaps for the viz

WHITE = np.array([255, 255, 255], np.uint8)
GRAY = np.array([150, 150, 150], np.uint8)
YELLOW = np.array([245, 214, 45], np.uint8)

try:
    from PIL import Image

    HAVE_PIL = True
except Exception as exc:  # noqa: BLE001
    HAVE_PIL = False
    print("PIL MISSING:", exc)


def log(msg):
    print(msg, flush=True)


def build_mapping():
    transform = ref.transform_from_metadata(SRC_META)
    bbox = ref.bbox_from_metadata(REF_3X_META)
    lith, poro, actnum, z_center = ref.load_native_arrays(NUMPY_DIR)
    master = ref.find_master_grdecl(GRDECL_DIR)
    spec = ref.find_specgrid(master)
    if spec is None:
        raise RuntimeError(f"SPECGRID not found in {master}")
    if tuple(lith.shape) != (spec.nx, spec.ny, spec.nz):
        raise ValueError(f"native {lith.shape} != SPECGRID {(spec.nx, spec.ny, spec.nz)}")
    coord_path = master.with_name(master.stem + "_COORD.GRDECL")
    log(f"reading COORD {coord_path}")
    coord = ref.read_coord(coord_path, spec)
    a0, a1 = ref.column_centers_axis(coord, transform)
    del coord
    tree, valid_ij = ref.build_column_tree(a0, a1, actnum, lith, poro)
    log(f"native {lith.shape}  valid_columns={valid_ij.shape[0]}  "
        f"sample_spacing={transform.sample_spacing}m  bbox={bbox}")
    return transform, bbox, lith, poro, actnum, z_center, tree, valid_ij


def section_query(fixed_axis, fixed_idx, bbox):
    """Return (x, y) seismic-axis-index query points for an inline/xline slice."""
    if fixed_axis == 0:  # inline: axis0 fixed, vary axis1 (j)
        n = (bbox.j1 - bbox.j0) * SCALE
        x = np.full(n, bbox.i0 + (fixed_idx + 0.5) / SCALE, np.float32)
        y = (bbox.j0 + (np.arange(n, dtype=np.float32) + 0.5) / SCALE).astype(np.float32)
    else:  # xline: axis1 fixed, vary axis0 (i)
        n = (bbox.i1 - bbox.i0) * SCALE
        y = np.full(n, bbox.j0 + (fixed_idx + 0.5) / SCALE, np.float32)
        x = (bbox.i0 + (np.arange(n, dtype=np.float32) + 0.5) / SCALE).astype(np.float32)
    return x, y, n


def render_scale3(fixed_axis, fixed_idx, m):
    """Render slice at scale-3 sampling using the reference build_profiles."""
    transform, bbox, lith, poro, actnum, z_center, tree, valid_ij = m
    x, y, n = section_query(fixed_axis, fixed_idx, bbox)
    K = (bbox.k1 - bbox.k0) * SCALE
    depths = (bbox.k0 + (np.arange(K, dtype=np.float32) + 0.5) / SCALE) * transform.sample_spacing
    dist, nearest = tree.query(np.column_stack([x, y]).astype(np.float32), workers=-1)
    sel = valid_ij[nearest]
    lith2d = np.zeros((n, K), np.uint8)
    poro2d = np.full((n, K), np.nan, np.float16)
    uniq, inv = np.unique(sel, axis=0, return_inverse=True)
    inv = inv.ravel()
    for g, (ni, nj) in enumerate(uniq):
        cols = np.flatnonzero(inv == g)
        lp, pp = ref.build_profiles(int(ni), int(nj), depths, lith, poro, actnum, z_center)
        lith2d[cols] = lp
        poro2d[cols] = pp
    return lith2d, poro2d


def render_fine(fixed_axis, fixed_idx, m, step_m=1.0):
    """Render slice at fine vertical sampling; return lith_bin, present masks."""
    transform, bbox, lith, poro, actnum, z_center, tree, valid_ij = m
    x, y, n = section_query(fixed_axis, fixed_idx, bbox)
    z_lo = bbox.k0 * transform.sample_spacing
    z_hi = bbox.k1 * transform.sample_spacing
    depths = np.arange(z_lo, z_hi, step_m, dtype=np.float32)
    K = depths.shape[0]
    dist, nearest = tree.query(np.column_stack([x, y]).astype(np.float32), workers=-1)
    sel = valid_ij[nearest]
    in_support = dist <= DETAIL_MAX_COL_DIST
    lith_bin = np.zeros((n, K), np.uint8)
    present = np.zeros((n, K), bool)
    uniq, inv = np.unique(sel, axis=0, return_inverse=True)
    inv = inv.ravel()
    for g, (ni, nj) in enumerate(uniq):
        cols = np.flatnonzero(inv == g)
        z = np.asarray(z_center[ni, nj, :])
        active_col = np.asarray(actnum[ni, nj, :]) > 0
        lith_col = np.asarray(lith[ni, nj, :])
        poro_col = np.asarray(poro[ni, nj, :])
        valid_col = active_col & (np.isfinite(poro_col) | (lith_col >= 0))
        k = ref.nearest_profile_indices(z, depths, valid_col)
        lv = np.asarray(lith[ni, nj, k])
        active = np.asarray(actnum[ni, nj, k]) > 0
        pres = active & ((lv >= 0) | np.isfinite(np.asarray(poro[ni, nj, k])))
        lb = (active & (lv > 0)).astype(np.uint8)
        for c in cols:
            present[c] = pres
            lith_bin[c] = lb
    present &= in_support[:, None]
    lith_bin[~present] = 0
    return lith_bin, present


def composite_from_dense(lith2d, poro2d):
    valid = np.isfinite(poro2d.astype(np.float32))
    rgb = np.empty(lith2d.shape + (3,), np.uint8)
    rgb[...] = WHITE
    rgb[valid & (lith2d == 0)] = GRAY
    rgb[lith2d == 1] = YELLOW
    return rgb


def composite_from_fine(lith_bin, present):
    rgb = np.empty(lith_bin.shape + (3,), np.uint8)
    rgb[...] = WHITE
    rgb[present & (lith_bin == 0)] = GRAY
    rgb[lith_bin == 1] = YELLOW
    return rgb


def to_section(rgb):
    # (lateral, depth, 3) -> (depth, lateral, 3): depth vertical, shallow on top
    return np.transpose(rgb, (1, 0, 2))


def resize_h(img, target_h):
    if not HAVE_PIL or img.shape[0] == target_h:
        return img
    im = Image.fromarray(img, "RGB").resize((img.shape[1], target_h), Image.NEAREST)
    return np.asarray(im)


def downscale(rgb, max_dim=1500):
    h, w = rgb.shape[:2]
    step = max(1, int(max(h, w) / max_dim))
    return rgb[::step, ::step]


def save_png(rgb, name):
    if HAVE_PIL:
        Image.fromarray(rgb, "RGB").save(OUT / name)
    else:
        np.save(OUT / (name + ".npy"), rgb)


def pick_indices(bbox):
    """Pick inline/xline out-indices that contain reservoir, from the cheap point file."""
    pts = np.load(NUMPY_DIR / "lithology_points_seismic_vis.npy")  # [N,4] axis0,axis1,sample,value
    val = pts[:, 3]
    tgt = pts[(val == 1) | (val == 2)]
    a0 = tgt[:, 0]
    a1 = tgt[:, 1]

    def out_idx(a, start, hi):
        return [int(np.clip(round((np.percentile(a, p) - start) * SCALE - 0.5), 0, hi - 1))
                for p in (40, 50, 60)]

    inlines = out_idx(a0, bbox.i0, (bbox.i1 - bbox.i0) * SCALE)
    xlines = out_idx(a1, bbox.j0, (bbox.j1 - bbox.j0) * SCALE)
    return inlines, xlines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = build_mapping()
    bbox = m[1]
    inlines, xlines = pick_indices(bbox)
    log(f"picked inlines={inlines} xlines={xlines}")

    dense_lith = np.load(DIRECT_DIR / "lithology_binary_3x_uint8.npy", mmap_mode="r")
    dense_poro = np.load(DIRECT_DIR / "porosity_3x_float16.npy", mmap_mode="r")
    sup_poro = np.load(SUPPORT_DIR / "porosity_3x_float16.npy", mmap_mode="r")

    report = {"correctness": [], "detail": []}

    # (A) correctness on 2 inlines + 1 xline
    checks = [("inline", 0, inlines[0]), ("inline", 0, inlines[2]), ("xline", 1, xlines[1])]
    for name, axis, idx in checks:
        ol, op = render_scale3(axis, idx, m)
        if axis == 0:
            dl = np.asarray(dense_lith[idx]); dp = np.asarray(dense_poro[idx]); sp = np.asarray(sup_poro[idx])
        else:
            dl = np.asarray(dense_lith[:, idx]); dp = np.asarray(dense_poro[:, idx]); sp = np.asarray(sup_poro[:, idx])
        support = np.isfinite(sp.astype(np.float32))
        ol = ol.copy(); op = op.copy()
        ol[~support] = 0
        op[~support] = np.float16(np.nan)
        lith_mismatch = int(np.count_nonzero(ol != dl))
        both_nan = np.isnan(op.astype(np.float32)) & np.isnan(dp.astype(np.float32))
        poro_mismatch = int(np.count_nonzero((~both_nan) & (op.astype(np.float32) != dp.astype(np.float32))))
        total = int(dl.size)
        log(f"[A] {name} idx={idx}: shape={dl.shape} lith_mismatch={lith_mismatch}/{total} "
            f"poro_mismatch={poro_mismatch}/{total}")
        report["correctness"].append(
            {"name": name, "idx": idx, "shape": list(dl.shape),
             "lith_mismatch": lith_mismatch, "poro_mismatch": poro_mismatch, "total": total})

    # (B) detail at 1 m on 1 inline + 1 xline
    for name, axis, idx in [("inline", 0, inlines[1]), ("xline", 1, xlines[1])]:
        lb, pres = render_fine(axis, idx, m, step_m=1.0)
        fine_rgb = to_section(composite_from_fine(lb, pres))
        ol, op = render_scale3(axis, idx, m)
        if axis == 0:
            sp = np.asarray(sup_poro[idx])
        else:
            sp = np.asarray(sup_poro[:, idx])
        support = np.isfinite(sp.astype(np.float32))
        dl = np.asarray(dense_lith[idx] if axis == 0 else dense_lith[:, idx])
        dp = np.asarray(dense_poro[idx] if axis == 0 else dense_poro[:, idx])
        dense_rgb = to_section(composite_from_dense(dl, dp))
        dense_rgb = resize_h(dense_rgb, fine_rgb.shape[0])
        fine_rgb = downscale(fine_rgb); dense_rgb = downscale(dense_rgb)
        h = min(fine_rgb.shape[0], dense_rgb.shape[0])
        gap = np.zeros((h, 14, 3), np.uint8)
        combo = np.concatenate([fine_rgb[:h], gap, dense_rgb[:h]], axis=1)
        fn = f"{name}_{idx}_fine_vs_dense.png"
        save_png(combo, fn)
        ty = float((lb == 1).mean())
        td = float((dl == 1).mean())
        log(f"[B] {name} idx={idx}: 1m_target_frac={ty:.4%} dense_target_frac={td:.4%} "
            f"fine_K={lb.shape[1]} dense_K={dl.shape[1]} -> {fn} (LEFT=1m RIGHT=dense)")
        report["detail"].append({"name": name, "idx": idx, "png": fn,
                                  "fine_K": lb.shape[1], "dense_K": dl.shape[1]})

    with open(OUT / "report.json", "w") as fh:
        json.dump({"have_pil": HAVE_PIL, **report}, fh, indent=2)
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
