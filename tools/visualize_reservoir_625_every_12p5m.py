"""Encode full reservoir/seismic section sequences at exactly 12.5 m steps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from visualize_reservoir_625_trios import (
    color_lithology,
    color_porosity,
    color_seismic,
    interpolated_slice,
)


FRAME_SIZE = (1920, 720)
PANEL_SIZE = (624, 650)
HEADER = 50
GAP = 16


def resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(values.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)


def resize_nearest(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(values, size, interpolation=cv2.INTER_NEAREST)


def reservoir_slice(
    volume: np.ndarray,
    *,
    axis: int,
    output_coordinate: float,
    categorical: bool,
) -> np.ndarray:
    if categorical:
        index = int(np.clip(round(output_coordinate), 0, volume.shape[axis] - 1))
        return np.take(volume, index, axis=axis)
    lo = int(np.clip(math.floor(output_coordinate), 0, volume.shape[axis] - 1))
    hi = min(lo + 1, volume.shape[axis] - 1)
    weight = float(output_coordinate - lo)
    a = np.take(volume, lo, axis=axis).astype(np.float32)
    if hi == lo or weight <= 1e-8:
        return a
    b = np.take(volume, hi, axis=axis).astype(np.float32)
    return a * (1.0 - weight) + b * weight


def crop_seismic(
    section: np.ndarray,
    *,
    axis: int,
    bounds: dict[str, list[int]],
) -> np.ndarray:
    a0_lo, a0_hi = bounds["axis0"]
    a1_lo, a1_hi = bounds["axis1"]
    s_lo, s_hi = bounds["sample"]
    if axis == 0:
        return section[a1_lo:a1_hi + 1, s_lo:s_hi + 1].T
    if axis == 1:
        return section[a0_lo:a0_hi + 1, s_lo:s_hi + 1].T
    return section[a0_lo:a0_hi + 1, a1_lo:a1_hi + 1]


def section_frame(
    *,
    axis: int,
    axis_name: str,
    seismic_coordinate: float,
    position_m: float,
    output_coordinate: float,
    lith: np.ndarray,
    poro: np.ndarray,
    seismic: np.ndarray,
    bounds: dict[str, list[int]],
) -> np.ndarray:
    lsec = reservoir_slice(
        lith, axis=axis, output_coordinate=output_coordinate, categorical=True
    )
    psec = reservoir_slice(
        poro, axis=axis, output_coordinate=output_coordinate, categorical=False
    )
    if axis in (0, 1):
        lsec = lsec.T
        psec = psec.T
    present = np.isfinite(psec)

    ssec = crop_seismic(
        interpolated_slice(seismic, axis, seismic_coordinate),
        axis=axis,
        bounds=bounds,
    )
    ssmall = resize_float(ssec, PANEL_SIZE)
    lsmall = resize_nearest(lsec, PANEL_SIZE)
    psmall = resize_float(psec, PANEL_SIZE)
    present_small = resize_nearest(present.astype(np.uint8), PANEL_SIZE).astype(bool)

    panels = (
        color_seismic(ssmall),
        color_lithology(lsmall, present_small),
        color_porosity(psmall),
    )
    titles = (
        f"SEISMIC | {axis_name}={seismic_coordinate:.2f}",
        f"LITHOLOGY | output={output_coordinate:.2f}",
        "POROSITY | 0.00..0.30",
    )
    frame = Image.new("RGB", FRAME_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(frame)
    x = 0
    for panel_rgb, title in zip(panels, titles, strict=True):
        panel = Image.fromarray(panel_rgb, "RGB")
        frame.paste(panel, (x, HEADER))
        draw.text((x + 8, 16), title, fill=(255, 255, 255))
        x += PANEL_SIZE[0] + GAP
    draw.text(
        (8, 700),
        f"position={position_m:.1f} m | exact frame spacing=12.5 m",
        fill=(255, 255, 255),
    )
    return np.asarray(frame, dtype=np.uint8)


def frame_positions(
    *,
    axis: int,
    origin: dict[str, int],
    bounds: dict[str, list[int]],
    scale: list[int],
    reservoir_spacing: dict[str, float],
) -> list[dict[str, float]]:
    axis_name = ("axis0", "axis1", "sample")[axis]
    if axis in (0, 1):
        lo, hi = bounds[axis_name]
        seismic_coordinates = np.arange(lo, hi + 1, dtype=np.float64)
        positions_m = seismic_coordinates * 12.5
    else:
        lo_m = bounds["sample"][0] * 10.0
        hi_m = bounds["sample"][1] * 10.0
        positions_m = np.arange(lo_m, hi_m + 1e-6, 12.5, dtype=np.float64)
        seismic_coordinates = positions_m / 10.0
    output_coordinates = (
        seismic_coordinates - float(origin[axis_name])
    ) * float(scale[axis])
    return [
        {
            "position_m": float(position_m),
            "seismic_coordinate": float(seismic_coordinate),
            "output_coordinate": float(output_coordinate),
            "nearest_output_index": int(round(output_coordinate)),
            "nearest_output_position_m": (
                float(origin[axis_name]) * (12.5 if axis in (0, 1) else 10.0)
                + int(round(output_coordinate)) * float(reservoir_spacing[axis_name])
            ),
        }
        for position_m, seismic_coordinate, output_coordinate in zip(
            positions_m, seismic_coordinates, output_coordinates, strict=True
        )
    ]


def contact_sheet(frames: list[np.ndarray], labels: list[str], path: Path) -> None:
    thumb_size = (480, 180)
    cols = 4
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * (thumb_size[1] + 24)), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        image = Image.fromarray(frame, "RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_size[0]
        y = (index // cols) * (thumb_size[1] + 24)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + thumb_size[1] + 4), label, fill=(255, 255, 255))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir-dir", type=Path, required=True)
    parser.add_argument("--seismic", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--axes", nargs="+", choices=("axis0", "axis1", "sample"),
                        default=("axis0", "axis1", "sample"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.reservoir_dir / "metadata.json").read_text())
    origin = metadata["seismic_index_origin"]
    bounds = metadata["seismic_index_bounds_inclusive"]
    scale = metadata["scale_axis0_axis1_sample"]
    spacing = metadata["voxel_spacing_m"]
    lith = np.load(args.reservoir_dir / "lithology_binary_uint8.npy", mmap_mode="r")
    poro = np.load(args.reservoir_dir / "porosity_float16.npy", mmap_mode="r")
    seismic = np.load(args.seismic, mmap_mode="r")

    summary: dict[str, object] = {
        "step_m": 12.5,
        "fps": args.fps,
        "frame_size": list(FRAME_SIZE),
        "axes": {},
    }
    for axis_name in args.axes:
        axis = ("axis0", "axis1", "sample").index(axis_name)
        records = frame_positions(
            axis=axis,
            origin=origin,
            bounds=bounds,
            scale=scale,
            reservoir_spacing=spacing,
        )
        video_path = args.out_dir / f"{axis_name}_every_12p5m.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            FRAME_SIZE,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")

        contact_indices = set(
            int(round(value))
            for value in np.linspace(0, len(records) - 1, min(20, len(records)))
        )
        contact_frames: list[np.ndarray] = []
        contact_labels: list[str] = []
        for index, record in enumerate(records):
            frame = section_frame(
                axis=axis,
                axis_name=axis_name,
                seismic_coordinate=record["seismic_coordinate"],
                position_m=record["position_m"],
                output_coordinate=record["output_coordinate"],
                lith=lith,
                poro=poro,
                seismic=seismic,
                bounds=bounds,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if index in contact_indices:
                contact_frames.append(frame)
                contact_labels.append(f"{record['position_m']:.1f} m")
            if index % 25 == 0 or index == len(records) - 1:
                print(f"{axis_name}: {index + 1}/{len(records)}", flush=True)
        writer.release()

        contact_name = f"{axis_name}_contact_sheet.png"
        contact_sheet(contact_frames, contact_labels, args.out_dir / contact_name)
        manifest_name = f"{axis_name}_frames.json"
        (args.out_dir / manifest_name).write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["axes"][axis_name] = {
            "frame_count": len(records),
            "video": video_path.name,
            "contact_sheet": contact_name,
            "manifest": manifest_name,
            "first_position_m": records[0]["position_m"],
            "last_position_m": records[-1]["position_m"],
        }

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
