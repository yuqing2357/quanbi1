"""Diagnose full-depth nodata stripes in a baked reservoir volume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, label
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--baked-dir", type=Path, required=True)
    parser.add_argument("--chunk-axis0", type=int, default=16)
    args = parser.parse_args()

    metadata = json.loads((args.baked_dir / "metadata.json").read_text())
    shape = tuple(int(value) for value in metadata["shape"])
    origin = metadata["seismic_index_origin"]
    scale = metadata["scale_axis0_axis1_sample"]
    gate = float(
        metadata.get("max_column_dist_idx", metadata.get("bbox_padding_idx", 3.5))
    )

    centers = np.load(args.native_dir / "column_centers_axis.npy")
    valid = np.load(args.native_dir / "column_valid.npy")
    tree = cKDTree(centers[valid].astype(np.float32))

    p0 = origin["axis0"] + np.arange(shape[0], dtype=np.float32) / scale[0]
    p1 = origin["axis1"] + np.arange(shape[1], dtype=np.float32) / scale[1]
    gx, gy = np.meshgrid(p0, p1, indexing="ij")
    distance, _ = tree.query(
        np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32),
        workers=-1,
    )
    tight = distance.reshape(shape[:2]) <= gate
    filled = binary_fill_holes(tight)
    del gx, gy, distance

    lith = np.load(args.baked_dir / "lithology_binary_uint8.npy", mmap_mode="r")
    poro = np.load(args.baked_dir / "porosity_float16.npy", mmap_mode="r")
    poro_support = np.zeros(shape[:2], dtype=bool)
    lith_target = np.zeros(shape[:2], dtype=bool)
    for i0 in range(0, shape[0], args.chunk_axis0):
        i1 = min(i0 + args.chunk_axis0, shape[0])
        poro_support[i0:i1] = np.isfinite(
            np.asarray(poro[i0:i1], dtype=np.float32)
        ).any(axis=2)
        lith_target[i0:i1] = (np.asarray(lith[i0:i1]) == 1).any(axis=2)
        print(f"scan {i1}/{shape[0]}", flush=True)

    missing_from_bake = filled & ~poro_support
    lith_hidden_by_poro = lith_target & ~poro_support

    # What remains empty after topological filling is connected to the image
    # border under the chosen 2-D topology. Narrow exterior channels are the
    # likely source of section-spanning stripes.
    exterior_empty = ~filled
    labels, count = label(exterior_empty)
    border_labels = np.unique(
        np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
        )
    )
    border_labels = border_labels[border_labels != 0]
    connected_to_border = np.isin(labels, border_labels)

    # Interior-looking columns: exterior-connected in 2-D, but bracketed by
    # filled support along axis0 or axis1. These produce full-depth stripes in
    # vertical sections despite not being binary_fill_holes() holes.
    bracket_axis0 = np.zeros(shape[:2], dtype=bool)
    bracket_axis1 = np.zeros(shape[:2], dtype=bool)
    bracket_axis0[1:-1] = filled[:-2] & filled[2:]
    bracket_axis1[:, 1:-1] = filled[:, :-2] & filled[:, 2:]
    one_pixel_channels = connected_to_border & (bracket_axis0 | bracket_axis1)

    result = {
        "shape_xy": list(shape[:2]),
        "gate_idx": gate,
        "tight_columns": int(tight.sum()),
        "filled_columns": int(filled.sum()),
        "topological_holes_filled": int((filled & ~tight).sum()),
        "actual_porosity_support_columns": int(poro_support.sum()),
        "filled_but_all_poro_nan": int(missing_from_bake.sum()),
        "lith_target_columns_hidden_by_poro_mask": int(lith_hidden_by_poro.sum()),
        "exterior_components": int(count),
        "one_pixel_exterior_channels_bracketed_by_support": int(one_pixel_channels.sum()),
        "one_pixel_channel_axis0_indices": np.flatnonzero(
            one_pixel_channels.any(axis=1)
        ).astype(int).tolist()[:100],
        "one_pixel_channel_axis1_indices": np.flatnonzero(
            one_pixel_channels.any(axis=0)
        ).astype(int).tolist()[:100],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
