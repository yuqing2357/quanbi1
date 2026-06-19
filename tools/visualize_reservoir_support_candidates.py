"""Visualize old/new reservoir support on transects with the largest changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

from bake_reservoir_npy import (
    active_column_depth_bounds,
    fill_column_depth_holes,
    load_pillar_geometry,
    native_column_polygon_footprint,
    representative_pillar_axes,
)


def gap_cells(line: np.ndarray) -> int:
    occupied = np.flatnonzero(line)
    if occupied.size < 2:
        return 0
    return int((~line[occupied[0]:occupied[-1] + 1]).sum())


def render_support(
    old_line: np.ndarray,
    new_line: np.ndarray,
    title: str,
    *,
    scale_x: int = 3,
    height: int = 160,
) -> Image.Image:
    width = old_line.size * scale_x
    out = Image.new("RGB", (width, height + 34), (16, 16, 16))
    pixels = np.zeros((height, old_line.size, 3), dtype=np.uint8)
    pixels[:, old_line & new_line] = (150, 150, 150)
    pixels[:, new_line & ~old_line] = (245, 214, 45)
    pixels[:, old_line & ~new_line] = (220, 50, 50)
    image = Image.fromarray(pixels, "RGB").resize(
        (width, height), Image.Resampling.NEAREST
    )
    out.paste(image, (0, 34))
    ImageDraw.Draw(out).text((8, 11), title, fill=(255, 255, 255))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--grdecl-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
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

    gx, gy = np.meshgrid(p0, p1, indexing="ij")
    tree = cKDTree(centers[valid].astype(np.float32))
    distance, _ = tree.query(
        np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32),
        workers=-1,
    )
    old = binary_fill_holes(distance.reshape(shape[:2]) <= gate)
    del gx, gy, distance
    pillar_geometry = load_pillar_geometry(
        args.grdecl_dir, args.native_dir / "metadata.json"
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
    new, _ = native_column_polygon_footprint(
        pillar_axis0,
        pillar_axis1,
        valid,
        p0,
        p1,
    )

    records = []
    for axis, axis_name in ((0, "axis0"), (1, "axis1")):
        line_count = shape[axis]
        scored = []
        for index in range(line_count):
            old_line = old[index] if axis == 0 else old[:, index]
            new_line = new[index] if axis == 0 else new[:, index]
            score = (
                gap_cells(old_line) - gap_cells(new_line),
                int(np.count_nonzero(old_line ^ new_line)),
            )
            scored.append((score, index))
        selected = [
            index
            for _, index in sorted(scored, reverse=True)[: args.count]
        ]
        rows = []
        for index in selected:
            old_line = old[index] if axis == 0 else old[:, index]
            new_line = new[index] if axis == 0 else new[:, index]
            coordinate = origin[axis_name] + index / scale[axis]
            title = (
                f"{axis_name} output={index} seismic={coordinate:.2f} | "
                f"old gaps={gap_cells(old_line)} new gaps={gap_cells(new_line)}"
            )
            rows.append(render_support(old_line, new_line, title))
            records.append(
                {
                    "axis": axis_name,
                    "output_index": index,
                    "seismic_coordinate": coordinate,
                    "old_gap_cells": gap_cells(old_line),
                    "new_gap_cells": gap_cells(new_line),
                    "changed_cells": int(np.count_nonzero(old_line ^ new_line)),
                }
            )
        width = max(row.width for row in rows)
        sheet = Image.new("RGB", (width, sum(row.height for row in rows)), (0, 0, 0))
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(args.out_dir / f"{axis_name}_support_candidates.png")

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "legend": {
                    "gray": "old and new support",
                    "yellow": "added by polygon support",
                    "red": "removed by polygon support",
                    "black": "nodata in both",
                },
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
