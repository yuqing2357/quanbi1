"""Bake the reservoir model to a dense .npy co-registered with the seismic volume.

Grid: anisotropic refine of the seismic grid -- axis0 x2, axis1 x2, sample x5
=> 6.25 x 6.25 x 2 m, NODE-ALIGNED (a reservoir section lands on every seismic
line AND at the midpoint between lines; depths land on every seismic sample and
4 levels between). Cropped to the reservoir footprint in the seismic frame.

No GRDECL at runtime: reads the native column arrays + the offline precompute
(column_centers_axis.npy / column_valid.npy / column_geometry.json).

Sampling (decided 2026-06-18):
  lithology = interval aggregation over each 2 m band (any native target cell
              {1,2} overlapping -> target; preserves thin sands)
  porosity  = nearest native z_center
  lateral   = nearest valid native column (Voronoi)
  support   = native logical (i,j) topology; enclosed holes filled before
              mapping to the regular grid, exterior unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, distance_transform_edt

import create_reservoir_3x_direct_numpy as direct

NUMPY = Path("data/reservoir/numpy")
OUT = Path("data/reservoir/npy_625x625x2_v3")
SEISMIC = Path("data/seismic/YJ-ALL-SEISMIC_depth_0_653.npy")
GRDECL = Path("data/reservoir/grdecl")
SCALE = (2, 2, 5)            # axis0, axis1, sample
BBOX_PADDING_IDX = 3.5       # crop padding in seismic-index units (12.5 m each)
TARGET_CLASSES = (1, 2)


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def aligned_node_axis(lo, hi, source_count, scale, *, padding=0.0):
    """Return an inclusive, source-node-aligned refined axis."""
    if source_count < 1 or scale < 1:
        raise ValueError("source_count and scale must be positive")
    source_lo = max(0, math.floor(float(lo) - float(padding)))
    source_hi = min(source_count - 1, math.ceil(float(hi) + float(padding)))
    if source_lo > source_hi:
        raise ValueError("extent does not intersect the source grid")
    count = (source_hi - source_lo) * scale + 1
    coords = source_lo + np.arange(count, dtype=np.float64) / scale
    return source_lo, source_hi, coords


def find_zcorn_cache(grdecl_dir, expected_bytes):
    matches = sorted((Path(grdecl_dir) / ".yj_cache").glob("*.zcorn.f32"))
    for path in matches:
        if path.stat().st_size == expected_bytes:
            return path
    raise FileNotFoundError(
        f"No valid ZCORN float32 cache in {Path(grdecl_dir) / '.yj_cache'} "
        f"(expected {expected_bytes} bytes). Build it with tools/verify_grdecl_parser.py."
    )


def active_zcorn_depth_bounds(cache_path, act):
    """Return exact corner-depth bounds across ACTNUM-active cells."""
    nx, ny, nz = act.shape
    zcorn = np.memmap(
        cache_path,
        dtype=np.float32,
        mode="r",
        shape=(2 * nx, 2 * ny, 2 * nz),
        order="F",
    )
    ii = 2 * np.arange(nx, dtype=np.int64)
    jj = 2 * np.arange(ny, dtype=np.int64)
    z_lo = math.inf
    z_hi = -math.inf
    for k0 in range(0, nz, 32):
        k1 = min(k0 + 32, nz)
        active = np.asarray(act[:, :, k0:k1]) > 0
        if not active.any():
            continue
        base_k = 2 * np.arange(k0, k1, dtype=np.int64)
        for di, dj, dk in (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ):
            values = np.asarray(zcorn[np.ix_(ii + di, jj + dj, base_k + dk)])
            selected = values[active]
            z_lo = min(z_lo, float(selected.min()))
            z_hi = max(z_hi, float(selected.max()))
    if not math.isfinite(z_lo) or not math.isfinite(z_hi):
        raise RuntimeError("No ACTNUM-active ZCORN values found")
    return z_lo, z_hi


def native_column_polygon_footprint(
    pillar_axis0,
    pillar_axis1,
    valid,
    p0,
    p1,
):
    """Rasterize the union of native column quadrilaterals."""
    native_support = binary_fill_holes(np.asarray(valid, dtype=bool))
    origin0, origin1 = float(p0[0]), float(p1[0])
    step0 = float(p0[1] - p0[0])
    step1 = float(p1[1] - p1[0])
    image = Image.new("1", (len(p1), len(p0)), 0)
    draw = ImageDraw.Draw(image)
    for i, j in np.argwhere(native_support):
        corners = np.asarray(
            (
                (pillar_axis1[i, j], pillar_axis0[i, j]),
                (pillar_axis1[i + 1, j], pillar_axis0[i + 1, j]),
                (pillar_axis1[i + 1, j + 1], pillar_axis0[i + 1, j + 1]),
                (pillar_axis1[i, j + 1], pillar_axis0[i, j + 1]),
            ),
            dtype=np.float64,
        )
        if not np.isfinite(corners).all():
            continue
        pixels = [
            ((axis1 - origin1) / step1, (axis0 - origin0) / step0)
            for axis1, axis0 in corners
        ]
        draw.polygon(pixels, fill=1)
    return np.asarray(image, dtype=bool), native_support


def load_pillar_geometry(grdecl_dir, source_metadata):
    master = direct.find_master_grdecl(grdecl_dir)
    spec = direct.find_specgrid(master)
    coord_path = master.with_name(master.stem + "_COORD.GRDECL")
    coord = direct.read_coord(coord_path, spec)
    transform = direct.transform_from_metadata(source_metadata)
    top_world_x, top_world_y = transform.local_to_world(
        coord[..., 0], coord[..., 1]
    )
    bottom_world_x, bottom_world_y = transform.local_to_world(
        coord[..., 3], coord[..., 4]
    )
    return (
        transform.world_to_axis0(top_world_x).astype(np.float32),
        transform.world_to_axis1(top_world_y).astype(np.float32),
        transform.world_to_axis0(bottom_world_x).astype(np.float32),
        transform.world_to_axis1(bottom_world_y).astype(np.float32),
        coord[..., 2].astype(np.float32),
        coord[..., 5].astype(np.float32),
    )


def active_column_depth_bounds(z, act):
    """Approximate active column top/base from native z-centre profiles."""
    nx, ny, _ = z.shape
    z_lo = np.full((nx, ny), np.nan, dtype=np.float32)
    z_hi = np.full((nx, ny), np.nan, dtype=np.float32)
    for i in range(nx):
        zi = np.asarray(z[i], dtype=np.float32)
        ai = np.asarray(act[i]) > 0
        for j in range(ny):
            values = np.sort(zi[j, ai[j] & np.isfinite(zi[j])])
            if values.size == 0:
                continue
            if values.size == 1:
                z_lo[i, j] = values[0]
                z_hi[i, j] = values[0]
            else:
                z_lo[i, j] = values[0] - 0.5 * (values[1] - values[0])
                z_hi[i, j] = values[-1] + 0.5 * (values[-1] - values[-2])
    return z_lo, z_hi


def fill_column_depth_holes(z_lo, z_hi, support):
    """Fill depth bounds only for enclosed logical support holes."""
    missing = support & ~(np.isfinite(z_lo) & np.isfinite(z_hi))
    if not missing.any():
        return z_lo, z_hi
    valid = np.isfinite(z_lo) & np.isfinite(z_hi)
    if not valid.any():
        raise RuntimeError("No valid column depth bounds")
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    out_lo = z_lo.copy()
    out_hi = z_hi.copy()
    out_lo[missing] = z_lo[nearest[0][missing], nearest[1][missing]]
    out_hi[missing] = z_hi[nearest[0][missing], nearest[1][missing]]
    return out_lo, out_hi


def representative_pillar_axes(pillar_geometry, column_z_lo, column_z_hi, valid):
    """Interpolate shared pillars at neighboring columns' mean active depth."""
    (
        axis0_top,
        axis1_top,
        axis0_bottom,
        axis1_bottom,
        z_top,
        z_bottom,
    ) = pillar_geometry
    column_mid = (column_z_lo + column_z_hi) * 0.5
    column_mid = np.where(valid, column_mid, np.nan)
    pillar_depth_sum = np.zeros(z_top.shape, dtype=np.float64)
    pillar_depth_count = np.zeros(z_top.shape, dtype=np.int16)
    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        target = pillar_depth_sum[di:di + valid.shape[0], dj:dj + valid.shape[1]]
        count = pillar_depth_count[di:di + valid.shape[0], dj:dj + valid.shape[1]]
        finite = np.isfinite(column_mid)
        target[finite] += column_mid[finite]
        count[finite] += 1
    fallback = (z_top + z_bottom) * 0.5
    pillar_depth = np.divide(
        pillar_depth_sum,
        pillar_depth_count,
        out=fallback.astype(np.float64),
        where=pillar_depth_count > 0,
    )
    dz = z_bottom - z_top
    fraction = np.divide(
        pillar_depth - z_top,
        dz,
        out=np.zeros_like(pillar_depth, dtype=np.float64),
        where=np.abs(dz) > 1e-9,
    )
    axis0 = axis0_top + fraction * (axis0_bottom - axis0_top)
    axis1 = axis1_top + fraction * (axis1_bottom - axis1_top)
    return axis0.astype(np.float32), axis1.astype(np.float32)


def cell_boundaries(z):
    mid = (z[:-1] + z[1:]) * 0.5
    b = np.empty(z.size + 1)
    b[1:-1] = mid
    b[0] = z[0] - (z[1] - z[0]) * 0.5
    b[-1] = z[-1] + (z[-1] - z[-2]) * 0.5
    return b


def nearest_idx(z, depths):
    pos = np.clip(np.searchsorted(z, depths, side="left"), 0, len(z) - 1)
    left = np.clip(pos - 1, 0, len(z) - 1)
    cl = np.abs(depths - z[left]) < np.abs(depths - z[pos])
    return np.where(cl, left, pos)


def lith_profile(z, lith, depths, step):
    b = cell_boundaries(z)
    present = (depths - step * 0.5 < b[-1]) & (depths + step * 0.5 > b[0])
    tcell = np.isin(lith, TARGET_CLASSES)
    nd = depths.size
    diff = np.zeros(nd + 1, np.int32)
    if tcell.any():
        clo = b[:-1][tcell] - step * 0.5
        chi = b[1:][tcell] + step * 0.5
        i0 = np.clip(np.searchsorted(depths, clo, side="right"), 0, nd)
        i1 = np.clip(np.searchsorted(depths, chi, side="left"), 0, nd)
        np.add.at(diff, i0, 1)
        np.add.at(diff, i1, -1)
    return ((np.cumsum(diff[:-1]) > 0) & present).astype(np.uint8)


def poro_profile(z, poro, depths, step):
    b = cell_boundaries(z)
    present = (depths - step * 0.5 < b[-1]) & (depths + step * 0.5 > b[0])
    val = poro[nearest_idx(z, depths)]
    return np.where(present & np.isfinite(val), val, np.nan).astype(np.float16)


def column_arrays(lith, poro, act, z, ni, nj):
    zc = np.asarray(z[ni, nj, :]).astype(np.float64)
    a = np.asarray(act[ni, nj, :]) > 0
    lc = np.asarray(lith[ni, nj, :])
    pc = np.asarray(poro[ni, nj, :]).astype(np.float32)
    valid = a & (np.isfinite(pc) | (lc >= 0)) & np.isfinite(zc)
    if valid.sum() < 2:
        return None
    order = np.argsort(zc[valid])
    return zc[valid][order], lc[valid][order], pc[valid][order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--chunk-axis0", type=int, default=4)
    ap.add_argument("--max-axis0-chunks", type=int, default=None)
    ap.add_argument(
        "--bbox-padding-idx",
        "--max-dist-idx",
        dest="bbox_padding_idx",
        type=float,
        default=BBOX_PADDING_IDX,
        help="padding around the native-column bounding box in seismic-index units",
    )
    ap.add_argument("--seismic", type=Path, default=SEISMIC)
    ap.add_argument("--grdecl-dir", type=Path, default=GRDECL)
    ap.add_argument("--zcorn-cache", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    geom = json.loads((NUMPY / "column_geometry.json").read_text())
    nx, ny, nz = geom["shape"]
    sp0, sp1, spz = geom["axis0_spacing_m"], geom["axis1_spacing_m"], geom["sample_spacing_m"]
    centers = np.load(NUMPY / "column_centers_axis.npy")
    valid = np.load(NUMPY / "column_valid.npy")
    valid_ij = np.argwhere(valid).astype(np.int32)
    tree = cKDTree(centers[valid].astype(np.float32))
    lith = np.load(NUMPY / "lithology_native_i_j_k.npy", mmap_mode="r")
    poro = np.load(NUMPY / "porosity_native_i_j_k.npy", mmap_mode="r")
    act = np.load(NUMPY / "actnum_native_i_j_k.npy", mmap_mode="r")
    z = np.load(NUMPY / "z_center_native_i_j_k.npy", mmap_mode="r")
    seismic = np.load(args.seismic, mmap_mode="r")
    if seismic.ndim != 3:
        raise SystemExit(f"seismic volume must be 3D, got {seismic.shape}")
    seismic_shape = tuple(int(v) for v in seismic.shape)
    del seismic

    a0v = centers[..., 0][valid]; a1v = centers[..., 1][valid]
    s0, s1, sz = SCALE
    a0_lo, a0_hi, p0 = aligned_node_axis(
        a0v.min(), a0v.max(), seismic_shape[0], s0, padding=args.bbox_padding_idx
    )
    a1_lo, a1_hi, p1 = aligned_node_axis(
        a1v.min(), a1v.max(), seismic_shape[1], s1, padding=args.bbox_padding_idx
    )
    expected_zcorn_bytes = 8 * nx * ny * nz * np.dtype(np.float32).itemsize
    zcorn_cache = args.zcorn_cache or find_zcorn_cache(args.grdecl_dir, expected_zcorn_bytes)
    z_min_m, z_max_m = active_zcorn_depth_bounds(zcorn_cache, act)
    s_lo, s_hi, sample_coords = aligned_node_axis(
        z_min_m / spz, z_max_m / spz, seismic_shape[2], sz
    )

    N0, N1, NZ = len(p0), len(p1), len(sample_coords)
    v_step = spz / sz
    depths = sample_coords * spz
    p0 = p0.astype(np.float32)
    p1 = p1.astype(np.float32)

    log(f"native {nx}x{ny}x{nz}  valid_cols={valid_ij.shape[0]}")
    log(f"seismic shape={seismic_shape} spacing=({sp0:g},{sp1:g},{spz:g})m")
    log(f"active ZCORN depth={z_min_m:.2f}..{z_max_m:.2f}m")
    log(f"seismic-idx bbox: axis0[{a0_lo},{a0_hi}] axis1[{a1_lo},{a1_hi}] sample[{s_lo},{s_hi}]")
    log(f"output shape {(N0, N1, NZ)}  spacing (6.25,6.25,2)  depth {depths[0]:.0f}..{depths[-1]:.0f}m")
    voxels = N0 * N1 * NZ
    log(f"voxels={voxels/1e9:.2f}e9  lith={voxels/1024**3:.1f}GiB  poro={voxels*2/1024**3:.1f}GiB")

    metadata_base = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grid": "anisotropic refinement of the cropped seismic grid; node-aligned",
        "scale_axis0_axis1_sample": list(SCALE),
        "voxel_spacing_m": {"axis0": sp0 / s0, "axis1": sp1 / s1, "sample": spz / sz},
        "shape": [N0, N1, NZ],
        "seismic_shape": list(seismic_shape),
        "seismic_index_bounds_inclusive": {
            "axis0": [a0_lo, a0_hi],
            "axis1": [a1_lo, a1_hi],
            "sample": [s_lo, s_hi],
        },
        "seismic_index_origin": {"axis0": a0_lo, "axis1": a1_lo, "sample": s_lo},
        "index_mapping": "seismic_axis = origin + output_index / scale (node-aligned, inclusive endpoints)",
        "depth_range_m": [float(depths[0]), float(depths[-1])],
        "source_active_zcorn_depth_range_m": [z_min_m, z_max_m],
        "zcorn_cache": str(zcorn_cache),
        "bbox_padding_idx": args.bbox_padding_idx,
        "support_method": (
            "binary_fill_holes(column_valid) in native logical (i,j), then "
            "interpolate shared COORD pillars at neighboring columns' mean "
            "active depth and rasterize native quadrilaterals; inside values "
            "use nearest valid column"
        ),
        "estimated_bytes": {
            "lithology_uint8": voxels,
            "porosity_float16": voxels * 2,
            "total": voxels * 3,
        },
    }
    if args.dry_run:
        print(json.dumps(metadata_base, indent=2))
        return

    # Determine support from the native logical topology. This avoids false
    # exterior channels caused by gaps in a spatial distance threshold.
    log("computing lateral footprint from native logical topology...")
    gx, gy = np.meshgrid(p0.astype(np.float32), p1, indexing="ij")
    _, nfull = tree.query(
        np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32),
        workers=-1,
    )
    del gx, gy
    nn_full = nfull.reshape(N0, N1).astype(np.int32); del nfull
    pillar_geometry = load_pillar_geometry(
        args.grdecl_dir,
        NUMPY / "metadata.json",
    )
    column_z_lo, column_z_hi = active_column_depth_bounds(z, act)
    logical_support = binary_fill_holes(valid)
    column_z_lo, column_z_hi = fill_column_depth_holes(
        column_z_lo, column_z_hi, logical_support
    )
    pillar_axis0, pillar_axis1 = representative_pillar_axes(
        pillar_geometry, column_z_lo, column_z_hi, logical_support
    )
    footprint, native_support = native_column_polygon_footprint(
        pillar_axis0,
        pillar_axis1,
        valid,
        p0,
        p1,
    )
    native_holes = int((native_support & ~valid).sum())
    log(
        f"footprint: body cols={int(footprint.sum())}; "
        f"native logical holes filled={native_holes}; exterior unchanged"
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lpath, ppath = out / "lithology_binary_uint8.npy", out / "porosity_float16.npy"
    for p in (lpath, ppath):
        part = p.with_suffix(p.suffix + ".partial")
        if (p.exists() or part.exists()) and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite")
        p.unlink(missing_ok=True); part.unlink(missing_ok=True)
    lith_out = open_memmap(lpath.with_suffix(".npy.partial"), mode="w+", dtype=np.uint8, shape=(N0, N1, NZ))
    poro_out = open_memmap(ppath.with_suffix(".npy.partial"), mode="w+", dtype=np.float16, shape=(N0, N1, NZ))

    nchunks = math.ceil(N0 / args.chunk_axis0)
    if args.max_axis0_chunks:
        nchunks = min(nchunks, args.max_axis0_chunks)
    t0 = time.time()
    for ci, c0 in enumerate(range(0, N0, args.chunk_axis0), 1):
        if args.max_axis0_chunks and ci > args.max_axis0_chunks:
            break
        c1 = min(c0 + args.chunk_axis0, N0)
        cn = c1 - c0
        cols = valid_ij[nn_full[c0:c1].ravel()]
        inside = footprint[c0:c1].ravel()
        lc = np.zeros((cn * N1, NZ), np.uint8)
        pc = np.full((cn * N1, NZ), np.nan, np.float16)
        if inside.any():
            uniq, inv = np.unique(cols[inside], axis=0, return_inverse=True)
            inv = inv.ravel(); positions = np.flatnonzero(inside)
            for g in range(uniq.shape[0]):
                cells = positions[inv == g]
                arr = column_arrays(lith, poro, act, z, int(uniq[g, 0]), int(uniq[g, 1]))
                if arr is None:
                    continue
                zc, lcol, pcol = arr
                lc[cells] = lith_profile(zc, lcol, depths, v_step)
                pc[cells] = poro_profile(zc, pcol, depths, v_step)
        lith_out[c0:c1] = lc.reshape(cn, N1, NZ)
        poro_out[c0:c1] = pc.reshape(cn, N1, NZ)
        el = (time.time() - t0) / 60
        log(f"chunk {ci}/{nchunks} (axis0 {c0}:{c1}) inside={int(inside.sum())}/{inside.size} "
            f"elapsed={el:.1f}min eta={el/ci*(nchunks-ci):.1f}min")
        if ci % 40 == 0:                      # periodic flush: keep dirty pages bounded
            lith_out.flush(); poro_out.flush()

    lith_out.flush(); poro_out.flush()
    del lith_out, poro_out
    meta = {
        **metadata_base,
        "sampling": {"lithology": "interval aggregation (target if any class 1/2 cell overlaps the 2 m band)",
                     "porosity": "nearest native z_center", "lateral": "nearest native column",
                     "support": "native COORD quadrilaterals at representative active "
                                "depth after filling enclosed logical holes; exterior unchanged"},
        "lithology_binary_rule": {"0": 0, "1": 1, "2": 1, "null_or_inactive": 0},
        "source": "data/reservoir/numpy native columns + column_centers precompute",
        "partial": bool(args.max_axis0_chunks),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    if not args.max_axis0_chunks:
        lpath.with_suffix(".npy.partial").replace(lpath)
        ppath.with_suffix(".npy.partial").replace(ppath)
        log(f"DONE full bake -> {out}")
    else:
        log(f"PARTIAL bake ({args.max_axis0_chunks} chunks) left as .partial in {out}")


if __name__ == "__main__":
    main()
