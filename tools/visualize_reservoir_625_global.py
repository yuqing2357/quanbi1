"""Render full-extent co-registered seismic/lithology/porosity sections.

For one representative section on each axis, writes:
  * overview: screen-sized full section;
  * cell_grid: high-resolution image with every reservoir voxel outlined and
    seismic cells outlined at their native 12.5/12.5/10 m spacing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_reservoir_625_trios import (
    color_lithology,
    color_porosity,
    color_seismic,
    interpolated_slice,
)


WHITE = np.asarray((255, 255, 255), dtype=np.uint8)


def draw_grid(
    rgb: np.ndarray,
    *,
    cell_pixels: int,
    grid_step_x: int,
    grid_step_y: int,
) -> np.ndarray:
    enlarged = np.repeat(np.repeat(rgb, cell_pixels, axis=0), cell_pixels, axis=1)
    step_x = cell_pixels * grid_step_x
    step_y = cell_pixels * grid_step_y
    enlarged[::step_y, :] = WHITE
    enlarged[:, ::step_x] = WHITE
    enlarged[-1, :] = WHITE
    enlarged[:, -1] = WHITE
    return enlarged


def fit_image(rgb: np.ndarray, max_width: int, max_height: int) -> Image.Image:
    image = Image.fromarray(rgb, "RGB")
    ratio = min(max_width / image.width, max_height / image.height, 1.0)
    if ratio < 1.0:
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return image


def titled(image: Image.Image, title: str) -> Image.Image:
    header = 44
    out = Image.new("RGB", (image.width, image.height + header), (12, 12, 12))
    out.paste(image, (0, header))
    ImageDraw.Draw(out).text((10, 15), title, fill=(255, 255, 255))
    return out


def compose(path: Path, panels: list[Image.Image]) -> None:
    gap = 16
    height = min(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for panel in panels:
        out.paste(panel.crop((0, 0, panel.width, height)), (x, 0))
        x += panel.width + gap
    out.save(path)


def resized_seismic(
    seismic: np.ndarray,
    *,
    axis: int,
    coordinate: float,
    bounds: dict[str, list[int]],
    shape: tuple[int, int],
) -> np.ndarray:
    section = interpolated_slice(seismic, axis, coordinate)
    a0_lo, a0_hi = bounds["axis0"]
    a1_lo, a1_hi = bounds["axis1"]
    s_lo, s_hi = bounds["sample"]
    if axis == 0:
        section = section[a1_lo:a1_hi + 1, s_lo:s_hi + 1].T
    elif axis == 1:
        section = section[a0_lo:a0_hi + 1, s_lo:s_hi + 1].T
    else:
        section = section[a0_lo:a0_hi + 1, a1_lo:a1_hi + 1]
    image = Image.fromarray(section.astype(np.float32), mode="F")
    image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir-dir", type=Path, required=True)
    parser.add_argument("--seismic", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cell-pixels", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.reservoir_dir / "metadata.json").read_text())
    origin = metadata["seismic_index_origin"]
    bounds = metadata["seismic_index_bounds_inclusive"]
    scale = metadata["scale_axis0_axis1_sample"]
    lith = np.load(args.reservoir_dir / "lithology_binary_uint8.npy", mmap_mode="r")
    poro = np.load(args.reservoir_dir / "porosity_float16.npy", mmap_mode="r")
    seismic = np.load(args.seismic, mmap_mode="r")

    selected = {
        "axis0": lith.shape[0] // 2,
        "axis1": lith.shape[1] // 2,
        "sample": lith.shape[2] // 2,
    }
    records = []
    for axis, axis_name in enumerate(("axis0", "axis1", "sample")):
        output_index = selected[axis_name]
        coordinate = float(origin[axis_name]) + output_index / float(scale[axis])
        lith_section = np.take(lith, output_index, axis=axis)
        poro_section = np.take(poro, output_index, axis=axis).astype(np.float32)
        if axis in (0, 1):
            lith_section = lith_section.T
            poro_section = poro_section.T
            seismic_grid = (2, 5)
        else:
            seismic_grid = (2, 2)
        present = np.isfinite(poro_section)

        seismic_section = resized_seismic(
            seismic,
            axis=axis,
            coordinate=coordinate,
            bounds=bounds,
            shape=lith_section.shape,
        )
        seismic_rgb = color_seismic(seismic_section)
        lith_rgb = color_lithology(lith_section, present)
        poro_rgb = color_porosity(poro_section)

        overview_panels = [
            titled(fit_image(seismic_rgb, 1500, 1300), f"SEISMIC | {axis_name}={coordinate:.2f}"),
            titled(fit_image(lith_rgb, 1500, 1300), f"LITHOLOGY | output index={output_index}"),
            titled(fit_image(poro_rgb, 1500, 1300), "POROSITY | 0.00..0.30"),
        ]
        overview_name = f"{axis_name}_global_overview.png"
        compose(args.out_dir / overview_name, overview_panels)

        seismic_grid_rgb = draw_grid(
            seismic_rgb,
            cell_pixels=args.cell_pixels,
            grid_step_x=seismic_grid[0],
            grid_step_y=seismic_grid[1],
        )
        lith_grid_rgb = draw_grid(
            lith_rgb,
            cell_pixels=args.cell_pixels,
            grid_step_x=1,
            grid_step_y=1,
        )
        poro_grid_rgb = draw_grid(
            poro_rgb,
            cell_pixels=args.cell_pixels,
            grid_step_x=1,
            grid_step_y=1,
        )
        grid_name = f"{axis_name}_global_cell_grid.png"
        compose(
            args.out_dir / grid_name,
            [
                titled(Image.fromarray(seismic_grid_rgb), f"SEISMIC | {axis_name}={coordinate:.2f}"),
                titled(Image.fromarray(lith_grid_rgb), f"LITHOLOGY | output index={output_index}"),
                titled(Image.fromarray(poro_grid_rgb), "POROSITY | 0.00..0.30"),
            ],
        )
        records.append(
            {
                "axis": axis_name,
                "output_index": output_index,
                "seismic_coordinate": coordinate,
                "overview": overview_name,
                "cell_grid": grid_name,
                "section_shape": list(lith_section.shape),
                "present_fraction": float(present.mean()),
                "target_fraction_among_present": (
                    float((lith_section[present] == 1).mean()) if present.any() else 0.0
                ),
            }
        )
        print(json.dumps(records[-1]), flush=True)

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "reservoir_dir": str(args.reservoir_dir),
                "seismic": str(args.seismic),
                "voxel_spacing_m": metadata["voxel_spacing_m"],
                "panel_order": ["seismic", "lithology", "porosity"],
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
