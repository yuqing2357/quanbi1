"""Verify a baked 6.25 m x 6.25 m x 2 m reservoir numpy pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chunk-axis0", type=int, default=32)
    args = parser.parse_args()

    metadata = json.loads((args.directory / "metadata.json").read_text())
    lith = np.load(args.directory / "lithology_binary_uint8.npy", mmap_mode="r")
    poro = np.load(args.directory / "porosity_float16.npy", mmap_mode="r")
    expected_shape = tuple(int(value) for value in metadata["shape"])
    if lith.shape != expected_shape or poro.shape != expected_shape:
        raise SystemExit(
            f"shape mismatch: metadata={expected_shape}, lith={lith.shape}, poro={poro.shape}"
        )
    if lith.dtype != np.uint8 or poro.dtype != np.float16:
        raise SystemExit(f"dtype mismatch: lith={lith.dtype}, poro={poro.dtype}")

    origin = metadata["seismic_index_origin"]
    scale = metadata["scale_axis0_axis1_sample"]
    axis_names = ("axis0", "axis1", "sample")
    endpoint = {
        name: float(origin[name]) + (expected_shape[index] - 1) / float(scale[index])
        for index, name in enumerate(axis_names)
    }
    expected_endpoint = {
        name: float(metadata["seismic_index_bounds_inclusive"][name][1])
        for name in axis_names
    }
    if endpoint != expected_endpoint:
        raise SystemExit(f"endpoint mismatch: calculated={endpoint}, metadata={expected_endpoint}")

    finite_count = 0
    target_count = 0
    target_without_porosity = 0
    lith_min = 255
    lith_max = 0
    poro_min = np.inf
    poro_max = -np.inf
    for i0 in range(0, expected_shape[0], args.chunk_axis0):
        i1 = min(i0 + args.chunk_axis0, expected_shape[0])
        lchunk = np.asarray(lith[i0:i1])
        pchunk = np.asarray(poro[i0:i1], dtype=np.float32)
        finite = np.isfinite(pchunk)
        target = lchunk == 1
        finite_count += int(finite.sum())
        target_count += int(target.sum())
        target_without_porosity += int((target & ~finite).sum())
        lith_min = min(lith_min, int(lchunk.min()))
        lith_max = max(lith_max, int(lchunk.max()))
        if finite.any():
            poro_min = min(poro_min, float(pchunk[finite].min()))
            poro_max = max(poro_max, float(pchunk[finite].max()))

    if (lith_min, lith_max) != (0, 1):
        raise SystemExit(f"unexpected lithology range: {(lith_min, lith_max)}")
    if target_without_porosity:
        raise SystemExit(f"{target_without_porosity} target voxels have no porosity support")

    result = {
        "shape": list(expected_shape),
        "dtype": {"lithology": str(lith.dtype), "porosity": str(poro.dtype)},
        "seismic_endpoint": endpoint,
        "voxel_spacing_m": metadata["voxel_spacing_m"],
        "voxel_count": int(np.prod(expected_shape, dtype=np.int64)),
        "finite_porosity_voxels": finite_count,
        "target_lithology_voxels": target_count,
        "target_without_porosity": target_without_porosity,
        "lithology_range": [lith_min, lith_max],
        "porosity_range": [poro_min, poro_max],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
