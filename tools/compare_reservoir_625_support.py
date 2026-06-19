"""Compare legacy distance-gate and native-topology reservoir footprints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

from bake_reservoir_npy import (
    active_column_depth_bounds,
    fill_column_depth_holes,
    load_pillar_geometry,
    native_column_polygon_footprint,
    representative_pillar_axes,
)


def transect_gap_stats(mask: np.ndarray, axis: int) -> dict[str, int]:
    """Count false runs bracketed by support on 1-D transects."""
    lines = mask if axis == 1 else mask.T
    line_count = 0
    gap_count = 0
    gap_cells = 0
    max_gap = 0
    for line in lines:
        occupied = np.flatnonzero(line)
        if occupied.size < 2:
            continue
        interior = ~line[occupied[0]:occupied[-1] + 1]
        if not interior.any():
            continue
        padded = np.pad(interior, 1, constant_values=False)
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        stops = np.flatnonzero(padded[:-1] & ~padded[1:])
        widths = stops - starts
        line_count += 1
        gap_count += int(widths.size)
        gap_cells += int(widths.sum())
        max_gap = max(max_gap, int(widths.max()))
    return {
        "transects_with_internal_gaps": line_count,
        "internal_gap_runs": gap_count,
        "internal_gap_cells": gap_cells,
        "max_internal_gap_width_cells": max_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--grdecl-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    shape = tuple(int(value) for value in metadata["shape"])
    origin = metadata["seismic_index_origin"]
    scale = metadata["scale_axis0_axis1_sample"]
    gate = float(metadata.get("max_column_dist_idx", metadata.get("bbox_padding_idx", 3.5)))
    centers = np.load(args.native_dir / "column_centers_axis.npy")
    valid = np.load(args.native_dir / "column_valid.npy")
    act = np.load(args.native_dir / "actnum_native_i_j_k.npy", mmap_mode="r")
    z = np.load(args.native_dir / "z_center_native_i_j_k.npy", mmap_mode="r")
    p0 = origin["axis0"] + np.arange(shape[0], dtype=np.float32) / scale[0]
    p1 = origin["axis1"] + np.arange(shape[1], dtype=np.float32) / scale[1]

    valid_tree = cKDTree(centers[valid].astype(np.float32))
    gx, gy = np.meshgrid(p0, p1, indexing="ij")
    distance, _ = valid_tree.query(
        np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32),
        workers=-1,
    )
    old_tight = distance.reshape(shape[:2]) <= gate
    old = binary_fill_holes(old_tight)
    del gx, gy, distance

    pillar_geometry = load_pillar_geometry(
        args.grdecl_dir,
        args.native_dir / "metadata.json",
    )
    column_z_lo, column_z_hi = active_column_depth_bounds(z, act)
    column_z_lo, column_z_hi = fill_column_depth_holes(
        column_z_lo, column_z_hi, binary_fill_holes(valid)
    )
    pillar_axis0, pillar_axis1 = representative_pillar_axes(
        pillar_geometry,
        column_z_lo,
        column_z_hi,
        binary_fill_holes(valid),
    )
    new, native_support = native_column_polygon_footprint(
        pillar_axis0,
        pillar_axis1,
        valid,
        p0,
        p1,
    )
    added = new & ~old
    removed = old & ~new

    # A valid topology fix should fill logical holes but never turn native
    # exterior cells into support.
    result = {
        "shape_xy": list(shape[:2]),
        "old_columns": int(old.sum()),
        "new_columns": int(new.sum()),
        "added_columns": int(added.sum()),
        "removed_columns": int(removed.sum()),
        "native_valid_columns": int(valid.sum()),
        "native_logical_support_columns": int(native_support.sum()),
        "native_holes_filled": int((native_support & ~valid).sum()),
        "old_axis0_sections": transect_gap_stats(old, axis=1),
        "new_axis0_sections": transect_gap_stats(new, axis=1),
        "old_axis1_sections": transect_gap_stats(old, axis=0),
        "new_axis1_sections": transect_gap_stats(new, axis=0),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    rgb = np.zeros(shape[:2] + (3,), dtype=np.uint8)
    rgb[old & new] = (160, 160, 160)
    rgb[added] = (245, 214, 45)
    rgb[removed] = (220, 40, 40)
    image = Image.fromarray(np.transpose(rgb, (1, 0, 2)), "RGB")
    image.thumbnail((2200, 1800), Image.Resampling.NEAREST)
    image.save(args.out_dir / "support_diff.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
