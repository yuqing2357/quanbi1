"""Render 15 co-registered seismic/lithology/porosity examples.

Five examples are selected for each reservoir axis. Every panel is a local
zoom, so each source cell can be outlined with a visible one-pixel white grid.
Reservoir coordinates are mapped back to the seismic grid through metadata;
half-line and intermediate-depth positions use linear interpolation between
the two adjacent seismic slices.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WHITE = np.asarray((255, 255, 255), dtype=np.uint8)
BLACK = np.asarray((0, 0, 0), dtype=np.uint8)


def color_seismic(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    lo, hi = np.percentile(values[finite], [2, 98]) if finite.any() else (0.0, 1.0)
    scaled = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    gray = (scaled * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def color_lithology(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    rgb = np.full(values.shape + (3,), (28, 28, 28), dtype=np.uint8)
    rgb[present & (values == 0)] = (150, 150, 150)
    rgb[present & (values == 1)] = (245, 214, 45)
    return rgb


def color_porosity(values: np.ndarray) -> np.ndarray:
    stops = (
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    )
    scaled = np.clip(values / 0.30, 0.0, 1.0)
    work = np.zeros(values.shape + (3,), dtype=np.float32)
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:], strict=True):
        mask = (scaled >= x0) & (scaled <= x1)
        ratio = (scaled[mask] - x0) / (x1 - x0)
        for channel in range(3):
            work[mask, channel] = c0[channel] + ratio * (c1[channel] - c0[channel])
    rgb = np.full(values.shape + (3,), (28, 28, 28), dtype=np.uint8)
    finite = np.isfinite(values)
    rgb[finite] = work[finite].astype(np.uint8)
    return rgb


def cell_grid(rgb: np.ndarray, factor_x: int, factor_y: int) -> np.ndarray:
    enlarged = np.repeat(np.repeat(rgb, factor_y, axis=0), factor_x, axis=1)
    enlarged[::factor_y, :] = WHITE
    enlarged[:, ::factor_x] = WHITE
    enlarged[-1, :] = WHITE
    enlarged[:, -1] = WHITE
    return enlarged


def panel(rgb: np.ndarray, title: str) -> Image.Image:
    image = Image.fromarray(rgb, "RGB")
    out = Image.new("RGB", (image.width, image.height + 42), (12, 12, 12))
    out.paste(image, (0, 42))
    ImageDraw.Draw(out).text((10, 14), title, fill=(255, 255, 255))
    return out


def compose(path: Path, panels: list[Image.Image]) -> None:
    gap = 16
    height = min(image.height for image in panels)
    width = sum(image.width for image in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in panels:
        out.paste(image.crop((0, 0, image.width, height)), (x, 0))
        x += image.width + gap
    out.save(path)


def interpolated_slice(volume: np.ndarray, axis: int, coordinate: float) -> np.ndarray:
    lo = int(math.floor(coordinate))
    hi = min(lo + 1, volume.shape[axis] - 1)
    lo = max(0, min(lo, volume.shape[axis] - 1))
    weight = float(coordinate - lo)
    a = np.take(volume, lo, axis=axis).astype(np.float32)
    if hi == lo or weight <= 1e-8:
        return a
    b = np.take(volume, hi, axis=axis).astype(np.float32)
    return a * (1.0 - weight) + b * weight


def five_supported_indices(lith: np.ndarray, poro: np.ndarray, axis: int) -> list[int]:
    strides = [59, 44, 57]
    strides[axis] = 1
    slices = tuple(slice(None, None, stride) for stride in strides)
    present = np.isfinite(np.asarray(poro[slices], dtype=np.float32))
    target = (np.asarray(lith[slices]) == 1) & present
    reduce_axes = tuple(value for value in range(3) if value != axis)
    score = target.sum(axis=reduce_axes)
    candidates = np.flatnonzero(score > 0)
    if candidates.size < 5:
        score = present.sum(axis=reduce_axes)
        candidates = np.flatnonzero(score > 0)
    if candidates.size == 0:
        raise RuntimeError(f"No supported samples found for axis {axis}")
    positions = np.linspace(0, candidates.size - 1, 5)
    return [int(candidates[int(round(value))]) for value in positions]


def centered_window(
    target: np.ndarray,
    present: np.ndarray,
    height: int,
    width: int,
) -> tuple[slice, slice]:
    points = np.argwhere(target & present)
    if points.size == 0:
        points = np.argwhere(present)
    if points.size:
        center_y, center_x = np.median(points, axis=0)
    else:
        center_y, center_x = np.asarray(target.shape, dtype=np.float64) / 2.0
    y0 = int(round(center_y)) - height // 2
    x0 = int(round(center_x)) - width // 2
    y0 = max(0, min(y0, target.shape[0] - height))
    x0 = max(0, min(x0, target.shape[1] - width))
    return slice(y0, y0 + height), slice(x0, x0 + width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir-dir", type=Path, required=True)
    parser.add_argument("--seismic", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.reservoir_dir / "metadata.json").read_text())
    origin = metadata["seismic_index_origin"]
    scale = metadata["scale_axis0_axis1_sample"]
    lith = np.load(args.reservoir_dir / "lithology_binary_uint8.npy", mmap_mode="r")
    poro = np.load(args.reservoir_dir / "porosity_float16.npy", mmap_mode="r")
    seismic = np.load(args.seismic, mmap_mode="r")

    records: list[dict[str, object]] = []
    axis_names = ("axis0", "axis1", "sample")
    for axis, axis_name in enumerate(axis_names):
        selected = five_supported_indices(lith, poro, axis)
        for number, output_index in enumerate(selected, start=1):
            seismic_coordinate = float(origin[axis_name]) + output_index / float(scale[axis])
            reservoir_slice = np.take(lith, output_index, axis=axis)
            porosity_slice = np.take(poro, output_index, axis=axis).astype(np.float32)
            present = np.isfinite(porosity_slice)

            if axis in (0, 1):
                # Native slice axes are (lateral, depth); display depth vertically.
                lith_display = reservoir_slice.T
                poro_display = porosity_slice.T
                present_display = present.T
                rs_y, rs_x = centered_window(
                    lith_display == 1, present_display, height=50, width=48
                )
                lith_zoom = lith_display[rs_y, rs_x]
                poro_zoom = poro_display[rs_y, rs_x]
                present_zoom = present_display[rs_y, rs_x]

                seismic_full = interpolated_slice(seismic, axis, seismic_coordinate).T
                lateral_origin = origin["axis1" if axis == 0 else "axis0"]
                seismic_x0 = int(lateral_origin + rs_x.start / scale[1 if axis == 0 else 0])
                seismic_y0 = int(origin["sample"] + rs_y.start / scale[2])
                seismic_zoom = seismic_full[
                    seismic_y0:seismic_y0 + 10,
                    seismic_x0:seismic_x0 + 24,
                ]
                seismic_rgb = cell_grid(color_seismic(seismic_zoom), 24, 30)
                lith_rgb = cell_grid(color_lithology(lith_zoom, present_zoom), 12, 6)
                poro_rgb = cell_grid(color_porosity(poro_zoom), 12, 6)
            else:
                rs_y, rs_x = centered_window(
                    reservoir_slice == 1, present, height=48, width=48
                )
                lith_zoom = reservoir_slice[rs_y, rs_x]
                poro_zoom = porosity_slice[rs_y, rs_x]
                present_zoom = present[rs_y, rs_x]

                seismic_full = interpolated_slice(seismic, axis, seismic_coordinate)
                seismic_y0 = int(origin["axis0"] + rs_y.start / scale[0])
                seismic_x0 = int(origin["axis1"] + rs_x.start / scale[1])
                seismic_zoom = seismic_full[
                    seismic_y0:seismic_y0 + 24,
                    seismic_x0:seismic_x0 + 24,
                ]
                seismic_rgb = cell_grid(color_seismic(seismic_zoom), 24, 24)
                lith_rgb = cell_grid(color_lithology(lith_zoom, present_zoom), 12, 12)
                poro_rgb = cell_grid(color_porosity(poro_zoom), 12, 12)

            filename = f"{axis_name}_{number:02d}_index_{output_index}.png"
            compose(
                args.out_dir / filename,
                [
                    panel(seismic_rgb, f"SEISMIC | {axis_name}={seismic_coordinate:.2f}"),
                    panel(lith_rgb, f"LITHOLOGY | output index={output_index}"),
                    panel(poro_rgb, "POROSITY | 0.00..0.30"),
                ],
            )
            record = {
                "axis": axis_name,
                "example": number,
                "output_index": output_index,
                "seismic_coordinate": seismic_coordinate,
                "file": filename,
                "present_fraction": float(present_zoom.mean()),
                "target_fraction_among_present": (
                    float((lith_zoom[present_zoom] == 1).mean()) if present_zoom.any() else 0.0
                ),
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    summary = {
        "reservoir_dir": str(args.reservoir_dir),
        "seismic": str(args.seismic),
        "shape": list(lith.shape),
        "voxel_spacing_m": metadata["voxel_spacing_m"],
        "grid": "one-pixel white line at every displayed source cell boundary",
        "panel_order": ["seismic", "lithology", "porosity"],
        "examples": records,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DONE: {len(records)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
