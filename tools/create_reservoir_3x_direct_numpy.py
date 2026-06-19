from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_SRC_CANDIDATES = [
    PROJECT_ROOT / "local" / "app" / "src",
    PROJECT_ROOT / "apps" / "yj_studio" / "src",
    *sorted(PROJECT_ROOT.glob("apps*/yj_studio/src")),
]
for app_src in APP_SRC_CANDIDATES:
    if (app_src / "yj_studio" / "io" / "grdecl" / "parser.py").exists():
        if str(app_src) not in sys.path:
            sys.path.insert(0, str(app_src))
        break

from yj_studio.io.grdecl.parser import find_specgrid, read_coord  # noqa: E402


SOURCE_NUMPY_DIR = Path("data/reservoir/numpy")
SOURCE_GRDECL_DIR = Path("data/reservoir/grdecl")
OUT_DIR = Path("data/reservoir/numpy_3x_direct")
REFERENCE_3X_METADATA = Path("data/reservoir/numpy_3x/metadata.json")
SUPPORT_DIR = Path("data/reservoir/numpy_3x")


@dataclass(frozen=True)
class BBox:
    i0: int
    i1: int
    j0: int
    j1: int
    k0: int
    k1: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.i1 - self.i0, self.j1 - self.j0, self.k1 - self.k0)


@dataclass(frozen=True)
class AxisTransform:
    axis0_origin: float
    axis1_origin: float
    axis0_spacing: float
    axis1_spacing: float
    sample_spacing: float
    map_origin_x: float = 0.0
    map_origin_y: float = 0.0
    map_x_unit_x: float = 1.0
    map_x_unit_y: float = 0.0
    map_y_unit_x: float = 0.0
    map_y_unit_y: float = 1.0

    def world_to_axis0(self, world_x: np.ndarray) -> np.ndarray:
        return (world_x - self.axis0_origin) / self.axis0_spacing

    def world_to_axis1(self, world_y: np.ndarray) -> np.ndarray:
        return (world_y - self.axis1_origin) / self.axis1_spacing

    def local_to_world(self, local_x: np.ndarray, local_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        world_x = self.map_origin_x + local_x * self.map_x_unit_x + local_y * self.map_y_unit_x
        world_y = self.map_origin_y + local_x * self.map_x_unit_y + local_y * self.map_y_unit_y
        return world_x, world_y


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def gib(nbytes: int | float) -> float:
    return float(nbytes) / (1024.0**3)


def find_master_grdecl(root: Path) -> Path:
    candidates = [
        p
        for p in root.glob("*.GRDECL")
        if "_COORD" not in p.name.upper()
        and "_ZCORN" not in p.name.upper()
        and "_ACTNUM" not in p.name.upper()
    ]
    if not candidates:
        raise FileNotFoundError(f"No master GRDECL found in {root}")
    return candidates[0]


def bbox_from_metadata(path: Path) -> BBox:
    payload = load_json(path)
    raw = payload.get("bbox_1x_half_open")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain bbox_1x_half_open")
    return BBox(
        i0=int(raw["i0"]),
        i1=int(raw["i1"]),
        j0=int(raw["j0"]),
        j1=int(raw["j1"]),
        k0=int(raw["k0"]),
        k1=int(raw["k1"]),
    )


def transform_from_metadata(path: Path) -> AxisTransform:
    payload = load_json(path)
    transform = payload.get("seismic_index_transform") or {}
    axis0 = str(transform.get("axis0_index", ""))
    axis1 = str(transform.get("axis1_index", ""))
    sample = str(transform.get("sample_index", ""))
    axis0_origin, axis0_spacing = _parse_linear_transform(axis0, "world_x", 630200.0, 12.5)
    axis1_origin, axis1_spacing = _parse_linear_transform(axis1, "world_y", 4154988.0, 12.5)
    _, sample_spacing = _parse_linear_transform(sample, "z_depth_m", 0.0, 10.0)
    map_origin_x, map_origin_y, map_x_unit_x, map_x_unit_y, map_y_unit_x, map_y_unit_y = _parse_mapaxes(
        payload.get("mapaxes")
    )
    return AxisTransform(
        axis0_origin=axis0_origin,
        axis1_origin=axis1_origin,
        axis0_spacing=axis0_spacing,
        axis1_spacing=axis1_spacing,
        sample_spacing=sample_spacing,
        map_origin_x=map_origin_x,
        map_origin_y=map_origin_y,
        map_x_unit_x=map_x_unit_x,
        map_x_unit_y=map_x_unit_y,
        map_y_unit_x=map_y_unit_x,
        map_y_unit_y=map_y_unit_y,
    )


def _parse_linear_transform(
    text: str,
    variable: str,
    default_origin: float,
    default_spacing: float,
) -> tuple[float, float]:
    pattern = rf"\(\s*{re.escape(variable)}\s*-\s*([-+0-9.eE]+)\s*\)\s*/\s*([-+0-9.eE]+)"
    match = re.search(pattern, text)
    if not match:
        return default_origin, default_spacing
    return float(match.group(1)), float(match.group(2))


def _parse_mapaxes(value: object) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        return 0.0, 0.0, 1.0, 0.0, 0.0, 1.0
    raw = value["values"]
    if len(raw) != 6:
        return 0.0, 0.0, 1.0, 0.0, 0.0, 1.0
    x1, y1, x2, y2, x3, y3 = (float(v) for v in raw)
    x_axis = np.asarray([x3 - x2, y3 - y2], dtype=np.float64)
    y_axis = np.asarray([x1 - x2, y1 - y2], dtype=np.float64)
    x_len = float(np.linalg.norm(x_axis))
    y_len = float(np.linalg.norm(y_axis))
    if x_len <= 0.0 or y_len <= 0.0:
        return x2, y2, 1.0, 0.0, 0.0, 1.0
    x_unit = x_axis / x_len
    y_unit = y_axis / y_len
    return x2, y2, float(x_unit[0]), float(x_unit[1]), float(y_unit[0]), float(y_unit[1])


def load_native_arrays(source_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "lithology": source_dir / "lithology_native_i_j_k.npy",
        "porosity": source_dir / "porosity_native_i_j_k.npy",
        "actnum": source_dir / "actnum_native_i_j_k.npy",
        "z_center": source_dir / "z_center_native_i_j_k.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing native reservoir arrays:\n" + "\n".join(missing))
    lith = np.load(paths["lithology"], mmap_mode="r")
    poro = np.load(paths["porosity"], mmap_mode="r")
    actnum = np.load(paths["actnum"], mmap_mode="r")
    z_center = np.load(paths["z_center"], mmap_mode="r")
    if not (lith.shape == poro.shape == actnum.shape == z_center.shape):
        raise ValueError(
            "Native array shape mismatch: "
            f"lith={lith.shape}, poro={poro.shape}, actnum={actnum.shape}, z={z_center.shape}"
        )
    return lith, poro, actnum, z_center


def load_support_volume(support_dir: Path) -> np.ndarray | None:
    porosity_path = support_dir / "porosity_3x_float16.npy"
    if not porosity_path.exists():
        return None
    return np.load(porosity_path, mmap_mode="r")


def column_centers_axis(coord: np.ndarray, transform: AxisTransform) -> tuple[np.ndarray, np.ndarray]:
    pillars = coord[:, :, :]
    x_pillar = (pillars[..., 0] + pillars[..., 3]) * 0.5
    y_pillar = (pillars[..., 1] + pillars[..., 4]) * 0.5
    x_center = (
        x_pillar[:-1, :-1]
        + x_pillar[1:, :-1]
        + x_pillar[:-1, 1:]
        + x_pillar[1:, 1:]
    ) * 0.25
    y_center = (
        y_pillar[:-1, :-1]
        + y_pillar[1:, :-1]
        + y_pillar[:-1, 1:]
        + y_pillar[1:, 1:]
    ) * 0.25
    world_x, world_y = transform.local_to_world(x_center, y_center)
    return transform.world_to_axis0(world_x), transform.world_to_axis1(world_y)


def build_column_tree(
    axis0: np.ndarray,
    axis1: np.ndarray,
    actnum: np.ndarray,
    lith: np.ndarray,
    poro: np.ndarray,
) -> tuple[cKDTree, np.ndarray]:
    valid = (
        (np.asarray(actnum).any(axis=2))
        | (np.asarray(lith) >= 0).any(axis=2)
        | np.isfinite(np.asarray(poro)).any(axis=2)
    )
    ij = np.argwhere(valid)
    if ij.size == 0:
        raise RuntimeError("No valid native reservoir columns were found.")
    points = np.column_stack([axis0[valid], axis1[valid]]).astype(np.float32, copy=False)
    return cKDTree(points), ij.astype(np.int32, copy=False)


def output_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / "lithology_binary_3x_uint8.npy", out_dir / "porosity_3x_float16.npy"


def prepare_output(path: Path, overwrite: bool, dtype: np.dtype, shape: tuple[int, int, int]) -> np.memmap:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        if not overwrite:
            raise FileExistsError(f"{path} or {partial} exists; pass --overwrite to replace it")
        path.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)


def close_memmap(arr: np.memmap) -> None:
    arr.flush()
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def nearest_profile_indices(z_profile: np.ndarray, depths_m: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    z = np.asarray(z_profile, dtype=np.float32)
    finite = np.isfinite(z)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    if int(finite.sum()) < 2:
        return np.zeros(depths_m.shape, dtype=np.int32)
    valid_indices = np.flatnonzero(finite)
    z_valid = z[valid_indices]
    order = np.argsort(z_valid)
    z_sorted = z_valid[order]
    idx_sorted = valid_indices[order]
    pos = np.searchsorted(z_sorted, depths_m, side="left")
    pos = np.clip(pos, 0, len(z_sorted) - 1)
    left = np.clip(pos - 1, 0, len(z_sorted) - 1)
    choose_left = np.abs(depths_m - z_sorted[left]) < np.abs(depths_m - z_sorted[pos])
    best = np.where(choose_left, left, pos)
    return idx_sorted[best].astype(np.int32, copy=False)


def build_profiles(
    native_i: int,
    native_j: int,
    depths_m: np.ndarray,
    lith: np.ndarray,
    poro: np.ndarray,
    actnum: np.ndarray,
    z_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lith_col = np.asarray(lith[native_i, native_j, :])
    poro_col = np.asarray(poro[native_i, native_j, :])
    active_col = np.asarray(actnum[native_i, native_j, :]) > 0
    valid_col = active_col & (np.isfinite(poro_col) | (lith_col >= 0))
    k_idx = nearest_profile_indices(z_center[native_i, native_j, :], depths_m, valid_col)
    lith_values = np.asarray(lith[native_i, native_j, k_idx])
    poro_values = np.asarray(poro[native_i, native_j, k_idx])
    active = np.asarray(actnum[native_i, native_j, k_idx]) > 0
    lith_profile = (active & np.isfinite(lith_values) & (lith_values > 0)).astype(np.uint8)
    poro_profile = np.where(
        active & np.isfinite(poro_values),
        poro_values.astype(np.float16, copy=False),
        np.float16(0.0),
    )
    return lith_profile, poro_profile


def create_direct_3x(args: argparse.Namespace) -> dict[str, Any]:
    source_metadata = load_json(args.source_metadata)
    reference_bbox = bbox_from_metadata(args.reference_3x_metadata)
    bbox = args.bbox or reference_bbox
    transform = transform_from_metadata(args.source_metadata)
    out_shape = tuple(dim * args.scale for dim in bbox.shape)
    support_start = (
        (bbox.i0 - reference_bbox.i0) * args.scale,
        (bbox.j0 - reference_bbox.j0) * args.scale,
        (bbox.k0 - reference_bbox.k0) * args.scale,
    )

    log(f"Using bbox={asdict(bbox)} scale={args.scale} out_shape={out_shape}")
    log(
        "Effective voxel spacing: "
        f"axis0={transform.axis0_spacing / args.scale:.12g}m, "
        f"axis1={transform.axis1_spacing / args.scale:.12g}m, "
        f"sample={transform.sample_spacing / args.scale:.12g}m"
    )
    if args.dry_run:
        return build_metadata(args, bbox, transform, out_shape, source_metadata, dry_run=True)

    lith, poro, actnum, z_center = load_native_arrays(args.source_numpy_dir)
    support = load_support_volume(args.support_dir)
    if support is None:
        log("No support porosity volume found; falling back to max-column-distance support.")
    else:
        support_stop = tuple(start + size for start, size in zip(support_start, out_shape, strict=True))
        if any(start < 0 for start in support_start) or any(stop > dim for stop, dim in zip(support_stop, support.shape, strict=True)):
            raise ValueError(
                f"Support volume shape {support.shape} cannot cover bbox {asdict(bbox)} "
                f"with local support range {support_start}->{support_stop}"
            )
        log(f"Using continuous support mask from {args.support_dir / 'porosity_3x_float16.npy'}")
    master = args.master_grdecl or find_master_grdecl(args.grdecl_dir)
    spec = find_specgrid(master)
    if spec is None:
        raise RuntimeError(f"SPECGRID not found in {master}")
    if tuple(lith.shape) != (spec.nx, spec.ny, spec.nz):
        raise ValueError(f"Native arrays {lith.shape} do not match SPECGRID {(spec.nx, spec.ny, spec.nz)}")

    coord_path = args.coord_grdecl or master.with_name(master.stem + "_COORD.GRDECL")
    if not coord_path.exists():
        raise FileNotFoundError(f"COORD GRDECL not found: {coord_path}")

    log(f"Reading COORD: {coord_path}")
    coord = read_coord(coord_path, spec)
    axis0_center, axis1_center = column_centers_axis(coord, transform)
    del coord

    log("Building native column KD-tree")
    tree, valid_ij = build_column_tree(axis0_center, axis1_center, actnum, lith, poro)
    del axis0_center, axis1_center

    lith_path, poro_path = output_paths(args.out_dir)
    lith_out = prepare_output(lith_path, args.overwrite, np.dtype(np.uint8), out_shape)
    poro_out = prepare_output(poro_path, args.overwrite, np.dtype(np.float16), out_shape)

    depths_m = (bbox.k0 + (np.arange(out_shape[2], dtype=np.float32) + 0.5) / args.scale) * transform.sample_spacing
    y_coords = bbox.j0 + (np.arange(out_shape[1], dtype=np.float32) + 0.5) / args.scale

    total_chunks = math.ceil(out_shape[0] / args.chunk_axis0)
    if args.max_axis0_chunks is not None:
        total_chunks = min(total_chunks, args.max_axis0_chunks)
    t0 = time.time()
    for chunk_idx, out_i0 in enumerate(range(0, out_shape[0], args.chunk_axis0), start=1):
        if args.max_axis0_chunks is not None and chunk_idx > args.max_axis0_chunks:
            break
        out_i1 = min(out_i0 + args.chunk_axis0, out_shape[0])
        x_coords = bbox.i0 + (np.arange(out_i0, out_i1, dtype=np.float32) + 0.5) / args.scale
        xi, yj = np.meshgrid(x_coords, y_coords, indexing="ij")
        query = np.column_stack([xi.ravel(), yj.ravel()]).astype(np.float32, copy=False)
        distances, nearest = tree.query(query, workers=args.query_workers)
        if support is None:
            inside = distances <= args.max_column_distance
            support_chunk = None
        else:
            support_i0 = support_start[0] + out_i0
            support_i1 = support_start[0] + out_i1
            support_j0 = support_start[1]
            support_j1 = support_start[1] + out_shape[1]
            support_k0 = support_start[2]
            support_k1 = support_start[2] + out_shape[2]
            support_chunk = np.isfinite(np.asarray(support[support_i0:support_i1, support_j0:support_j1, support_k0:support_k1]))
            inside = support_chunk.reshape(query.shape[0], out_shape[2]).any(axis=1)
        selected_ij = valid_ij[nearest]

        n_cols = query.shape[0]
        lith_chunk = np.zeros((n_cols, out_shape[2]), dtype=np.uint8)
        poro_chunk = np.full((n_cols, out_shape[2]), np.nan, dtype=np.float16)

        if inside.any():
            unique, inverse = np.unique(selected_ij[inside], axis=0, return_inverse=True)
            inside_positions = np.flatnonzero(inside)
            for group_idx, (native_i, native_j) in enumerate(unique):
                cols = inside_positions[inverse == group_idx]
                lith_profile, poro_profile = build_profiles(
                    int(native_i),
                    int(native_j),
                    depths_m,
                    lith,
                    poro,
                    actnum,
                    z_center,
                )
                lith_chunk[cols, :] = lith_profile
                poro_chunk[cols, :] = poro_profile

        if support_chunk is not None:
            support_flat = support_chunk.reshape(n_cols, out_shape[2])
            lith_chunk[~support_flat] = 0
            poro_chunk[~support_flat] = np.nan

        chunk_shape = (out_i1 - out_i0, out_shape[1], out_shape[2])
        lith_out[out_i0:out_i1, :, :] = lith_chunk.reshape(chunk_shape)
        poro_out[out_i0:out_i1, :, :] = poro_chunk.reshape(chunk_shape)

        elapsed = max(time.time() - t0, 1e-6)
        pct = 100.0 * chunk_idx / total_chunks
        log(
            f"Wrote axis0 chunk {chunk_idx}/{total_chunks} "
            f"({pct:.1f}%, inside_columns={int(inside.sum())}/{n_cols}, "
            f"support_voxels={int(support_chunk.sum()) if support_chunk is not None else 'n/a'}, "
            f"elapsed={elapsed / 60.0:.1f} min)"
        )

    close_memmap(lith_out)
    close_memmap(poro_out)
    lith_path.with_name(lith_path.name + ".partial").replace(lith_path)
    poro_path.with_name(poro_path.name + ".partial").replace(poro_path)

    metadata = build_metadata(args, bbox, transform, out_shape, source_metadata, dry_run=False)
    metadata["outputs"] = {
        lith_path.name: {
            "dtype": "uint8",
            "shape": list(out_shape),
            "bytes": lith_path.stat().st_size,
            "gib": gib(lith_path.stat().st_size),
        },
        poro_path.name: {
            "dtype": "float16",
            "shape": list(out_shape),
            "bytes": poro_path.stat().st_size,
            "gib": gib(poro_path.stat().st_size),
        },
    }
    write_json(args.out_dir / "metadata.json", metadata)
    return metadata


def build_metadata(
    args: argparse.Namespace,
    bbox: BBox,
    transform: AxisTransform,
    out_shape: tuple[int, int, int],
    source_metadata: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "direct native reservoir column resampling to 3x regular numpy grid; "
            "does not read or repeat the 1x seismic-aligned reservoir volumes"
        ),
        "dry_run": dry_run,
        "scale": int(args.scale),
        "source": {
            "grdecl_dir": str(args.grdecl_dir),
            "master_grdecl": str(args.master_grdecl or find_master_grdecl(args.grdecl_dir)),
            "native_numpy_dir": str(args.source_numpy_dir),
            "support_dir": str(args.support_dir),
            "native_arrays": [
                "lithology_native_i_j_k.npy",
                "porosity_native_i_j_k.npy",
                "actnum_native_i_j_k.npy",
                "z_center_native_i_j_k.npy",
            ],
            "explicitly_not_used": [
                "lithology_volume_seismic.npy",
                "porosity_volume_seismic.npy",
            ],
        },
        "bbox_1x_half_open": asdict(bbox),
        "bbox_1x_shape": list(bbox.shape),
        "shape": list(out_shape),
        "full_3x_start_index": [bbox.i0 * args.scale, bbox.j0 * args.scale, bbox.k0 * args.scale],
        "output_index_mapping": (
            "axis_index_1x = bbox_1x_start + (output_index + 0.5) / scale; "
            "nearest native reservoir column is selected in axis0/axis1; "
            "nearest native k is selected by z_center depth"
        ),
        "voxel_spacing_m": {
            "axis0": transform.axis0_spacing / args.scale,
            "axis1": transform.axis1_spacing / args.scale,
            "sample": transform.sample_spacing / args.scale,
        },
        "resampling": {
            "xy_method": "nearest native reservoir column in seismic axis coordinates",
            "z_method": "nearest valid native z_center in the selected column",
            "support_method": (
                "continuous runtime support from old 3x porosity finite mask; "
                "native attributes fill only inside this support"
            ),
            "max_column_distance_1x_samples": float(args.max_column_distance),
            "chunk_axis0": int(args.chunk_axis0),
            "mapaxes": {
                "origin": [transform.map_origin_x, transform.map_origin_y],
                "x_unit": [transform.map_x_unit_x, transform.map_x_unit_y],
                "y_unit": [transform.map_y_unit_x, transform.map_y_unit_y],
            },
        },
        "lithology_binary_rule": {
            "0": 0,
            "1": 1,
            "2": 1,
            "null_or_inactive": 0,
        },
        "porosity_rule": "native float32 sampled to float16; inactive/outside is NaN",
        "source_metadata_subset": {
            "seismic_index_transform": source_metadata.get("seismic_index_transform"),
            "mapaxes": source_metadata.get("mapaxes"),
            "crop": source_metadata.get("crop"),
            "grid": source_metadata.get("grid"),
        },
    }


def parse_bbox(values: list[str] | None) -> BBox | None:
    if values is None:
        return None
    if len(values) != 6:
        raise argparse.ArgumentTypeError("--bbox expects six integers: i0 i1 j0 j1 k0 k1")
    nums = [int(v) for v in values]
    return BBox(nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create 3x reservoir numpy volumes by resampling native GRDECL-derived "
            "properties directly, without repeating the existing 1x seismic-aligned volumes."
        )
    )
    parser.add_argument("--source-numpy-dir", type=Path, default=SOURCE_NUMPY_DIR)
    parser.add_argument("--grdecl-dir", type=Path, default=SOURCE_GRDECL_DIR)
    parser.add_argument("--master-grdecl", type=Path, default=None)
    parser.add_argument("--coord-grdecl", type=Path, default=None)
    parser.add_argument("--source-metadata", type=Path, default=SOURCE_NUMPY_DIR / "metadata.json")
    parser.add_argument("--reference-3x-metadata", type=Path, default=REFERENCE_3X_METADATA)
    parser.add_argument("--support-dir", type=Path, default=SUPPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--chunk-axis0", type=int, default=8)
    parser.add_argument("--query-workers", type=int, default=-1)
    parser.add_argument("--max-column-distance", type=float, default=1.5)
    parser.add_argument("--max-axis0-chunks", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--bbox",
        nargs=6,
        metavar=("I0", "I1", "J0", "J1", "K0", "K1"),
        default=None,
        help="1x half-open bbox. Defaults to data/reservoir/numpy_3x/metadata.json.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.bbox = parse_bbox(args.bbox)
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.chunk_axis0 < 1:
        raise ValueError("--chunk-axis0 must be >= 1")
    if args.max_column_distance <= 0.0:
        raise ValueError("--max-column-distance must be > 0")
    if args.max_axis0_chunks is not None and args.max_axis0_chunks < 1:
        raise ValueError("--max-axis0-chunks must be >= 1")

    metadata = create_direct_3x(args)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
