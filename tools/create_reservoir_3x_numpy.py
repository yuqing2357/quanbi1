from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np


SOURCE_DIR = Path("data/reservoir/numpy")
OUT_DIR = Path("data/reservoir/numpy_3x")
LITH_SOURCE = "lithology_volume_seismic.npy"
PORO_SOURCE = "porosity_volume_seismic.npy"


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

    def as_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.i0, self.i1),
            slice(self.j0, self.j1),
            slice(self.k0, self.k1),
        )


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def gib(nbytes: int | float) -> float:
    return float(nbytes) / (1024.0 ** 3)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def scan_bbox(
    lith: np.ndarray,
    poro: np.ndarray,
    chunk_size: int,
) -> BBox:
    if lith.shape != poro.shape:
        raise ValueError(f"Source shapes differ: lith={lith.shape}, poro={poro.shape}")

    i_min = lith.shape[0]
    i_max = -1
    j_min = lith.shape[1]
    j_max = -1
    k_min = lith.shape[2]
    k_max = -1

    t0 = time.time()
    for i0 in range(0, lith.shape[0], chunk_size):
        i1 = min(i0 + chunk_size, lith.shape[0])
        poro_block = poro[i0:i1, :, :]
        lith_block = lith[i0:i1, :, :]

        valid = np.isfinite(poro_block) | (np.isfinite(lith_block) & (lith_block >= 0))
        if not valid.any():
            continue

        valid_i = valid.any(axis=(1, 2))
        valid_j = valid.any(axis=(0, 2))
        valid_k = valid.any(axis=(0, 1))

        i_hits = np.flatnonzero(valid_i)
        j_hits = np.flatnonzero(valid_j)
        k_hits = np.flatnonzero(valid_k)
        i_min = min(i_min, i0 + int(i_hits[0]))
        i_max = max(i_max, i0 + int(i_hits[-1]))
        j_min = min(j_min, int(j_hits[0]))
        j_max = max(j_max, int(j_hits[-1]))
        k_min = min(k_min, int(k_hits[0]))
        k_max = max(k_max, int(k_hits[-1]))

        if i0 == 0 or (i1 % (chunk_size * 8) == 0):
            log(f"Scanned axis0 {i1}/{lith.shape[0]} in {time.time() - t0:.1f}s")

    if i_max < i_min:
        raise RuntimeError("No valid reservoir voxels were found.")
    return BBox(i_min, i_max + 1, j_min, j_max + 1, k_min, k_max + 1)


def chunks_for_bbox(bbox: BBox, chunk_size: int) -> Iterator[tuple[int, int]]:
    for i0 in range(bbox.i0, bbox.i1, chunk_size):
        yield i0, min(i0 + chunk_size, bbox.i1)


def repeat_3d(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return np.array(arr, copy=True)
    out = arr
    for axis in range(3):
        out = np.repeat(out, scale, axis=axis)
    return out


def prepare_output_path(path: Path, overwrite: bool) -> Path:
    partial = path.with_name(path.name + ".partial")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        path.unlink()
    if partial.exists():
        if not overwrite:
            raise FileExistsError(f"{partial} exists; pass --overwrite to replace it")
        partial.unlink()
    return partial


def close_memmap(arr: np.memmap) -> None:
    arr.flush()
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def output_paths(out_dir: Path, scale: int) -> tuple[Path, Path, Path, Path]:
    lith_final = out_dir / f"lithology_binary_{scale}x_uint8.npy"
    poro_final = out_dir / f"porosity_{scale}x_float16.npy"
    return (
        lith_final,
        poro_final,
        lith_final.with_name(lith_final.name + ".partial"),
        poro_final.with_name(poro_final.name + ".partial"),
    )


def build_metadata(
    lith: np.ndarray,
    poro: np.ndarray,
    bbox: BBox,
    out_dir: Path,
    scale: int,
    chunk_size: int,
    workers: int,
    source_metadata: dict,
) -> dict:
    lith_final, poro_final, _, _ = output_paths(out_dir, scale)
    out_shape = tuple(dim * scale for dim in bbox.shape)
    lith_bytes = lith_final.stat().st_size
    poro_bytes = poro_final.stat().st_size
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "repeat upsample from existing seismic-aligned reservoir numpy volumes",
        "scale": scale,
        "source": {
            "directory": str(SOURCE_DIR),
            "lithology": LITH_SOURCE,
            "porosity": PORO_SOURCE,
            "shape": list(lith.shape),
            "lithology_dtype": str(lith.dtype),
            "porosity_dtype": str(poro.dtype),
        },
        "bbox_1x_half_open": asdict(bbox),
        "bbox_1x_shape": list(bbox.shape),
        "shape": list(out_shape),
        "full_3x_start_index": [bbox.i0 * scale, bbox.j0 * scale, bbox.k0 * scale],
        "output_index_mapping": (
            "source_1x_index = bbox_1x_start + floor(output_index / scale)"
        ),
        "voxel_spacing_m": {
            "axis0": 12.5 / scale,
            "axis1": 12.5 / scale,
            "sample": 10.0 / scale,
        },
        "lithology_binary_rule": {
            "0": 0,
            "1": 1,
            "2": 1,
            "null_or_nan": 0,
        },
        "porosity_rule": "float32 source cast to float16; NaN is preserved",
        "outputs": {
            lith_final.name: {
                "dtype": "uint8",
                "shape": list(out_shape),
                "bytes": lith_bytes,
                "gib": gib(lith_bytes),
            },
            poro_final.name: {
                "dtype": "float16",
                "shape": list(out_shape),
                "bytes": poro_bytes,
                "gib": gib(poro_bytes),
            },
        },
        "conversion": {
            "chunk_size_1x_axis0": chunk_size,
            "workers": workers,
        },
        "source_metadata_subset": {
            "seismic_index_transform": source_metadata.get("seismic_index_transform"),
            "mapaxes": source_metadata.get("mapaxes"),
            "crop": source_metadata.get("crop"),
        },
    }


def finalize_existing_partials(
    lith: np.ndarray,
    poro: np.ndarray,
    bbox: BBox,
    out_dir: Path,
    scale: int,
    chunk_size: int,
    workers: int,
    overwrite: bool,
    source_metadata: dict,
) -> dict:
    lith_final, poro_final, lith_partial, poro_partial = output_paths(out_dir, scale)
    for final, partial in ((lith_final, lith_partial), (poro_final, poro_partial)):
        if partial.exists():
            if final.exists():
                if not overwrite and final.stat().st_size == partial.stat().st_size:
                    log(f"{final.name} already exists; leaving {partial.name} in place")
                    continue
                if not overwrite:
                    raise FileExistsError(
                        f"{final} exists; pass --overwrite to replace it"
                    )
                final.unlink()
            log(f"Finalizing {partial.name} -> {final.name}")
            partial.replace(final)
        elif not final.exists():
            raise FileNotFoundError(f"Neither {final} nor {partial} exists")

    metadata = build_metadata(
        lith=lith,
        poro=poro,
        bbox=bbox,
        out_dir=out_dir,
        scale=scale,
        chunk_size=chunk_size,
        workers=workers,
        source_metadata=source_metadata,
    )
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def convert(
    lith: np.ndarray,
    poro: np.ndarray,
    bbox: BBox,
    out_dir: Path,
    scale: int,
    chunk_size: int,
    workers: int,
    overwrite: bool,
    source_metadata: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lith_final, poro_final, _, _ = output_paths(out_dir, scale)
    lith_partial = prepare_output_path(lith_final, overwrite)
    poro_partial = prepare_output_path(poro_final, overwrite)

    out_shape = tuple(dim * scale for dim in bbox.shape)
    log(f"Creating output memmaps with shape={out_shape}")
    lith_out = np.lib.format.open_memmap(
        lith_partial, mode="w+", dtype=np.uint8, shape=out_shape
    )
    poro_out = np.lib.format.open_memmap(
        poro_partial, mode="w+", dtype=np.float16, shape=out_shape
    )

    def build_chunk(i0: int, i1: int) -> tuple[int, np.ndarray, np.ndarray]:
        block = (slice(i0, i1), slice(bbox.j0, bbox.j1), slice(bbox.k0, bbox.k1))
        lith_block = lith[block]
        poro_block = np.asarray(poro[block], dtype=np.float16)

        # Lithology storage rule requested by the user:
        # original 0 -> 0, original 1/2 -> 1, null/outside -> 0.
        lith_binary = (np.isfinite(lith_block) & (lith_block > 0)).astype(np.uint8)
        lith_up = repeat_3d(lith_binary, scale)
        poro_up = repeat_3d(poro_block, scale)
        return i0, lith_up, poro_up

    chunk_ranges = list(chunks_for_bbox(bbox, chunk_size))
    total_chunks = len(chunk_ranges)
    max_pending = max(1, workers)
    done_chunks = 0
    t0 = time.time()
    next_chunk = iter(chunk_ranges)
    pending = set()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while pending or done_chunks < total_chunks:
            while len(pending) < max_pending:
                try:
                    i0, i1 = next(next_chunk)
                except StopIteration:
                    break
                pending.add(executor.submit(build_chunk, i0, i1))

            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                i0, lith_up, poro_up = future.result()
                out_i0 = (i0 - bbox.i0) * scale
                out_i1 = out_i0 + lith_up.shape[0]
                lith_out[out_i0:out_i1, :, :] = lith_up
                poro_out[out_i0:out_i1, :, :] = poro_up
                done_chunks += 1

                del lith_up, poro_up
                if done_chunks == 1 or done_chunks % 5 == 0 or done_chunks == total_chunks:
                    elapsed = max(time.time() - t0, 1e-6)
                    pct = 100.0 * done_chunks / total_chunks
                    log(
                        f"Wrote chunk {done_chunks}/{total_chunks} "
                        f"({pct:.1f}%, elapsed {elapsed / 60.0:.1f} min)"
                    )

    close_memmap(lith_out)
    close_memmap(poro_out)
    del lith_out, poro_out

    lith_partial.replace(lith_final)
    poro_partial.replace(poro_final)

    metadata = build_metadata(
        lith=lith,
        poro=poro,
        bbox=bbox,
        out_dir=out_dir,
        scale=scale,
        chunk_size=chunk_size,
        workers=workers,
        source_metadata=source_metadata,
    )
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create cropped 3x reservoir numpy volumes for SAM workflows."
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--bbox-chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, min(3, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-existing-partials", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.chunk_size < 1 or args.bbox_chunk_size < 1:
        raise ValueError("--chunk-size and --bbox-chunk-size must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    global SOURCE_DIR
    SOURCE_DIR = args.source_dir
    lith_path = args.source_dir / LITH_SOURCE
    poro_path = args.source_dir / PORO_SOURCE
    source_metadata = load_json(args.source_dir / "metadata.json")

    log(f"Opening {lith_path}")
    lith = np.load(lith_path, mmap_mode="r")
    log(f"Opening {poro_path}")
    poro = np.load(poro_path, mmap_mode="r")

    log("Scanning minimal valid reservoir bbox")
    bbox = scan_bbox(lith, poro, args.bbox_chunk_size)
    out_shape = tuple(dim * args.scale for dim in bbox.shape)
    voxels = math.prod(out_shape)
    log(f"bbox_1x={asdict(bbox)} shape_1x={bbox.shape}")
    log(f"shape_{args.scale}x={out_shape} voxels={voxels:,}")
    log(
        "Expected output sizes: "
        f"lithology uint8={gib(voxels):.2f} GiB, "
        f"porosity float16={gib(voxels * 2):.2f} GiB, "
        f"total={gib(voxels * 3):.2f} GiB"
    )

    if args.dry_run:
        return

    if args.finalize_existing_partials:
        metadata = finalize_existing_partials(
            lith=lith,
            poro=poro,
            bbox=bbox,
            out_dir=args.out_dir,
            scale=args.scale,
            chunk_size=args.chunk_size,
            workers=args.workers,
            overwrite=args.overwrite,
            source_metadata=source_metadata,
        )
    else:
        metadata = convert(
            lith=lith,
            poro=poro,
            bbox=bbox,
            out_dir=args.out_dir,
            scale=args.scale,
            chunk_size=args.chunk_size,
            workers=args.workers,
            overwrite=args.overwrite,
            source_metadata=source_metadata,
        )
    total_gib = sum(item["gib"] for item in metadata["outputs"].values())
    log(f"Done. Wrote {args.out_dir} ({total_gib:.2f} GiB)")


if __name__ == "__main__":
    main()
