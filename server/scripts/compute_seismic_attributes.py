#!/usr/bin/env python
"""Compute crop-grid seismic attributes and write the multichannel contract."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import hilbert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPECTED_CROP_SHAPE = (1480, 1101, 566)
MODEL_SHAPE = (2959, 2201, 2826)
SCALE = (2, 2, 5)
ORIGIN = (204, 0, 88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/quanbi"))
    parser.add_argument("--chunk-axis0", type=int, default=8)
    parser.add_argument("--window", type=int, nargs=3, default=(1, 5, 9))
    parser.add_argument("--sample-per-chunk", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.chunk_axis0 <= 0:
        raise SystemExit("--chunk-axis0 must be positive")
    window = tuple(int(v) for v in args.window)
    if any(v <= 0 or v % 2 == 0 for v in window):
        raise SystemExit("--window values must be positive odd integers")

    source_path = args.root / "data/seismic/YJ-SEISMIC-RESERVOIR-CROP.npy"
    attrs_dir = args.root / "data/seismic/attrs"
    qc_dir = attrs_dir / "qc"
    attrs_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    cos_path = attrs_dir / "cosphase_f16.npy"
    coherence_path = attrs_dir / "coherence_f16.npy"
    stats_path = attrs_dir / "stats.json"
    spec_path = attrs_dir / "channel_spec.json"
    finals = (cos_path, coherence_path, stats_path, spec_path)
    if not args.overwrite and any(path.exists() for path in finals):
        existing = [str(path) for path in finals if path.exists()]
        raise SystemExit(f"refusing to overwrite existing outputs: {existing}")

    source = np.load(source_path, mmap_mode="r")
    if tuple(source.shape) != EXPECTED_CROP_SHAPE:
        raise SystemExit(f"crop shape {source.shape} != {EXPECTED_CROP_SHAPE}")
    cos_partial = cos_path.with_suffix(".partial.npy")
    coherence_partial = coherence_path.with_suffix(".partial.npy")
    variance_path = attrs_dir / "_coherence_variance_f32.partial.npy"
    for path in (cos_partial, coherence_partial, variance_path):
        if path.exists():
            path.unlink()

    cos_out = np.lib.format.open_memmap(
        cos_partial, mode="w+", dtype=np.float16, shape=source.shape
    )
    variance_out = np.lib.format.open_memmap(
        variance_path, mode="w+", dtype=np.float32, shape=source.shape
    )
    rng = np.random.default_rng(20260622)
    cos_samples: list[np.ndarray] = []
    variance_samples: list[np.ndarray] = []
    source_nan_count = 0
    source_count = int(np.prod(source.shape))
    halo = window[0] // 2
    total_chunks = (source.shape[0] + args.chunk_axis0 - 1) // args.chunk_axis0

    for chunk_no, start in enumerate(
        range(0, source.shape[0], args.chunk_axis0), start=1
    ):
        stop = min(start + args.chunk_axis0, source.shape[0])
        read_start = max(0, start - halo)
        read_stop = min(source.shape[0], stop + halo)
        slab = np.array(
            source[read_start:read_stop], dtype=np.float32, copy=True
        )
        core_start = start - read_start
        core_stop = core_start + (stop - start)
        source_nan_count += int(np.isnan(slab[core_start:core_stop]).sum())
        np.nan_to_num(slab, copy=False)

        analytic = hilbert(slab, axis=2)
        cosphase = ((np.cos(np.angle(analytic[core_start:core_stop])) + 1.0) * 0.5)
        cosphase = np.clip(cosphase, 0.0, 1.0).astype(np.float32, copy=False)
        cos_out[start:stop] = cosphase.astype(np.float16)

        mean = uniform_filter(slab, size=window, mode="nearest")
        mean_sq = uniform_filter(slab * slab, size=window, mode="nearest")
        variance = np.clip(
            mean_sq[core_start:core_stop] - mean[core_start:core_stop] ** 2,
            0.0,
            None,
        )
        variance_out[start:stop] = variance
        cos_samples.append(_sample_values(cosphase, args.sample_per_chunk, rng))
        variance_samples.append(_sample_values(variance, args.sample_per_chunk, rng))
        if chunk_no % 10 == 0 or chunk_no == total_chunks:
            cos_out.flush()
            variance_out.flush()
        print(
            f"pass1 {chunk_no}/{total_chunks}: axis0 [{start}:{stop})",
            flush=True,
        )

    cos_out.flush()
    variance_out.flush()
    del cos_out
    del variance_out

    cos_stats = _sample_stats(np.concatenate(cos_samples))
    variance_stats = _sample_stats(np.concatenate(variance_samples))
    low = variance_stats["p005"]
    high = variance_stats["p995"]
    if not high > low:
        raise SystemExit(f"invalid coherence normalization bounds: {low}, {high}")

    raw_variance = np.load(variance_path, mmap_mode="r")
    coherence_out = np.lib.format.open_memmap(
        coherence_partial, mode="w+", dtype=np.float16, shape=source.shape
    )
    coherence_samples: list[np.ndarray] = []
    for chunk_no, start in enumerate(
        range(0, source.shape[0], args.chunk_axis0), start=1
    ):
        stop = min(start + args.chunk_axis0, source.shape[0])
        variance = np.asarray(raw_variance[start:stop], dtype=np.float32)
        coherence = np.clip((variance - low) / (high - low), 0.0, 1.0)
        coherence_out[start:stop] = coherence.astype(np.float16)
        coherence_samples.append(
            _sample_values(coherence, args.sample_per_chunk, rng)
        )
        if chunk_no % 20 == 0 or chunk_no == total_chunks:
            coherence_out.flush()
        print(
            f"pass2 {chunk_no}/{total_chunks}: axis0 [{start}:{stop})",
            flush=True,
        )
    coherence_out.flush()
    del coherence_out
    del raw_variance

    os.replace(cos_partial, cos_path)
    os.replace(coherence_partial, coherence_path)
    variance_path.unlink()
    coherence_stats = _sample_stats(np.concatenate(coherence_samples))
    now = datetime.now(timezone.utc).isoformat()
    stats = {
        "created_at_utc": now,
        "source": {
            "path": str(source_path.relative_to(args.root)),
            "size_bytes": source_path.stat().st_size,
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "nan_count": source_nan_count,
            "nan_fraction": source_nan_count / source_count,
        },
        "cosphase": {
            **cos_stats,
            "dtype": "float16",
            "range": [0.0, 1.0],
            "method": "cos(angle(hilbert(trace))) then (x+1)/2",
            "scale": 0.5,
            "offset": 0.5,
        },
        "coherence": {
            **coherence_stats,
            "dtype": "float16",
            "range": [0.0, 1.0],
            "method": "normalized local amplitude variance (discontinuity)",
            "window_axis0_axis1_sample": list(window),
            "raw_variance": variance_stats,
            "normalization_p005": low,
            "normalization_p995": high,
            "scale": 1.0 / (high - low),
            "offset": -low / (high - low),
        },
    }
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_channel_spec(spec_path, now)
    _write_attribute_qc(cos_path, coherence_path, qc_dir)
    print(f"wrote: {cos_path}", flush=True)
    print(f"wrote: {coherence_path}", flush=True)
    print(f"wrote: {stats_path}", flush=True)
    print(f"wrote: {spec_path}", flush=True)
    return 0


def _sample_values(
    values: np.ndarray, requested: int, rng: np.random.Generator
) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= requested:
        return flat.astype(np.float32, copy=True)
    indices = rng.integers(0, flat.size, size=requested)
    return flat[indices].astype(np.float32, copy=True)


def _sample_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SystemExit("attribute sample contains no finite values")
    p005, p995 = np.percentile(finite, [0.5, 99.5])
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p005": float(p005),
        "p995": float(p995),
        "sample_count": int(finite.size),
    }


def _write_channel_spec(path: Path, created_at: str) -> None:
    payload = {
        "created_at_utc": created_at,
        "grid_model_shape": list(MODEL_SHAPE),
        "grid_model_spacing_m": [6.25, 6.25, 2.0],
        "grid_seismic_crop_shape": list(EXPECTED_CROP_SHAPE),
        "grid_seismic_crop_spacing_m": [12.5, 12.5, 10.0],
        "scale": list(SCALE),
        "origin_in_full_seismic": list(ORIGIN),
        "channels": [
            {
                "name": "lithology",
                "source": "model_lithology",
                "grid": "model",
                "path": "data/reservoir/npy_625x625x2_v3/lithology_binary_uint8.npy",
                "norm": "as_is",
                "dtype": "uint8",
            },
            {
                "name": "porosity",
                "source": "model_porosity",
                "grid": "model",
                "path": "data/reservoir/npy_625x625x2_v3/porosity_float16.npy",
                "norm": "clip01_porosity",
                "dtype": "float16",
            },
            {
                "name": "cosphase",
                "source": "seismic_crop",
                "grid": "seismic_crop",
                "path": "data/seismic/attrs/cosphase_f16.npy",
                "norm": "as_is",
                "dtype": "float16",
            },
            {
                "name": "coherence",
                "source": "seismic_crop",
                "grid": "seismic_crop",
                "path": "data/seismic/attrs/coherence_f16.npy",
                "norm": "as_is",
                "dtype": "float16",
            },
        ],
        "stats_path": "data/seismic/attrs/stats.json",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_attribute_qc(cos_path: Path, coherence_path: Path, qc_dir: Path) -> None:
    cosphase = np.load(cos_path, mmap_mode="r")
    coherence = np.load(coherence_path, mmap_mode="r")
    for index in (148, 740, 1331):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        axes[0].imshow(
            np.asarray(cosphase[index]).T,
            cmap="gray",
            origin="upper",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        axes[0].set_title(f"cosphase inline {index}")
        axes[1].imshow(
            np.asarray(coherence[index]).T,
            cmap="magma",
            origin="upper",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title(f"discontinuity inline {index}")
        for axis in axes:
            axis.set_xlabel("axis1")
            axis.set_ylabel("sample")
        fig.savefig(qc_dir / f"attributes_inline_{index}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
