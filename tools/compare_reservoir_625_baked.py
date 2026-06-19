"""Compare actual support and sections of two baked reservoir volumes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_reservoir_625_trios import color_lithology, color_porosity


def support_xy(poro: np.ndarray, chunk_axis0: int = 32) -> np.ndarray:
    result = np.zeros(poro.shape[:2], dtype=bool)
    for i0 in range(0, poro.shape[0], chunk_axis0):
        i1 = min(i0 + chunk_axis0, poro.shape[0])
        result[i0:i1] = np.isfinite(
            np.asarray(poro[i0:i1], dtype=np.float32)
        ).any(axis=2)
    return result


def gap_stats(mask: np.ndarray, axis: int) -> dict[str, int]:
    lines = mask if axis == 0 else mask.T
    line_count = gap_runs = gap_cells = max_width = 0
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
        gap_runs += int(widths.size)
        gap_cells += int(widths.sum())
        max_width = max(max_width, int(widths.max()))
    return {
        "sections_with_internal_gaps": line_count,
        "gap_runs": gap_runs,
        "gap_cells": gap_cells,
        "max_gap_width_cells": max_width,
    }


def fit(rgb: np.ndarray, size: tuple[int, int] = (650, 650)) -> Image.Image:
    image = Image.fromarray(rgb, "RGB")
    image.thumbnail(size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", size, (20, 20, 20))
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return canvas


def titled(image: Image.Image, title: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + 38), (10, 10, 10))
    out.paste(image, (0, 38))
    ImageDraw.Draw(out).text((8, 13), title, fill=(255, 255, 255))
    return out


def render_section(
    old_lith: np.ndarray,
    old_poro: np.ndarray,
    new_lith: np.ndarray,
    new_poro: np.ndarray,
    axis: int,
    index: int,
    path: Path,
) -> dict[str, object]:
    old_l = np.take(old_lith, index, axis=axis)
    old_p = np.take(old_poro, index, axis=axis).astype(np.float32)
    new_l = np.take(new_lith, index, axis=axis)
    new_p = np.take(new_poro, index, axis=axis).astype(np.float32)
    if axis in (0, 1):
        old_l, old_p = old_l.T, old_p.T
        new_l, new_p = new_l.T, new_p.T
    old_valid = np.isfinite(old_p)
    new_valid = np.isfinite(new_p)
    panels = [
        titled(fit(color_lithology(old_l, old_valid)), "V2 LITHOLOGY"),
        titled(fit(color_lithology(new_l, new_valid)), "V3 LITHOLOGY"),
        titled(fit(color_porosity(old_p)), "V2 POROSITY"),
        titled(fit(color_porosity(new_p)), "V3 POROSITY"),
    ]
    gap = 12
    out = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * 3, panels[0].height),
        (0, 0, 0),
    )
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    out.save(path)
    return {
        "axis": ("axis0", "axis1", "sample")[axis],
        "index": index,
        "old_support_pixels": int(old_valid.sum()),
        "new_support_pixels": int(new_valid.sum()),
        "added_pixels": int((new_valid & ~old_valid).sum()),
        "removed_pixels": int((old_valid & ~new_valid).sum()),
        "file": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    old_lith = np.load(args.old_dir / "lithology_binary_uint8.npy", mmap_mode="r")
    old_poro = np.load(args.old_dir / "porosity_float16.npy", mmap_mode="r")
    new_lith = np.load(args.new_dir / "lithology_binary_uint8.npy", mmap_mode="r")
    new_poro = np.load(args.new_dir / "porosity_float16.npy", mmap_mode="r")
    if not (old_lith.shape == old_poro.shape == new_lith.shape == new_poro.shape):
        raise SystemExit("old/new volume shapes differ")

    old_support = support_xy(old_poro)
    new_support = support_xy(new_poro)
    changed = old_support ^ new_support
    records = []
    for axis in (0, 1):
        scores = changed.sum(axis=1 - axis)
        selected = np.argsort(scores)[-args.count:][::-1]
        for index in selected:
            records.append(
                render_section(
                    old_lith,
                    old_poro,
                    new_lith,
                    new_poro,
                    axis,
                    int(index),
                    args.out_dir
                    / f"{('axis0', 'axis1')[axis]}_index_{int(index)}.png",
                )
            )

    summary = {
        "shape": list(old_lith.shape),
        "old_support_columns": int(old_support.sum()),
        "new_support_columns": int(new_support.sum()),
        "added_columns": int((new_support & ~old_support).sum()),
        "removed_columns": int((old_support & ~new_support).sum()),
        "old_axis0_gap_stats": gap_stats(old_support, axis=0),
        "new_axis0_gap_stats": gap_stats(new_support, axis=0),
        "old_axis1_gap_stats": gap_stats(old_support, axis=1),
        "new_axis1_gap_stats": gap_stats(new_support, axis=1),
        "sections": records,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
