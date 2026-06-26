#!/usr/bin/env python
"""Create visual and numeric QC for reservoir/seismic node alignment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, sobel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/src"))

from yj_studio_core.multichannel import resample_nodes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/quanbi"))
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--min-orientation-score",
        type=float,
        default=0.53,
        help="minimum mean cos^2 agreement of lithology and seismic local normals",
    )
    args = parser.parse_args()

    model_dir = args.root / "data/reservoir/npy_625x625x2_v3"
    lith_path = model_dir / "lithology_binary_uint8.npy"
    crop_path = args.root / "data/seismic/YJ-SEISMIC-RESERVOIR-CROP.npy"
    qc_dir = args.root / "data/seismic/attrs/qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    lith = np.load(lith_path, mmap_mode="r")
    crop = np.load(crop_path, mmap_mode="r")
    expected_crop = tuple((int(v) - 1) // s + 1 for v, s in zip(lith.shape, (2, 2, 5)))
    if tuple(crop.shape) != expected_crop:
        raise SystemExit(f"crop shape {crop.shape} != expected {expected_crop}")

    indices = args.indices or _default_node_indices(lith.shape[0], crop.shape[0], 5)
    reports: list[dict[str, object]] = []
    for model_index in indices:
        if not 0 <= model_index < lith.shape[0]:
            raise SystemExit(f"inline index outside model: {model_index}")
        seismic_index = model_index // 2
        lith_slice = np.asarray(lith[model_index], dtype=np.float32)
        seismic_slice = np.array(crop[seismic_index], dtype=np.float32, copy=True)
        seismic_slice = np.nan_to_num(seismic_slice, copy=False)
        seismic_up = resample_nodes(seismic_slice, lith_slice.shape)
        report = _alignment_metrics(lith_slice, seismic_up)
        report.update(
            model_inline=int(model_index),
            seismic_crop_inline=int(seismic_index),
            lithology_fraction=float(np.mean(lith_slice > 0.5)),
        )
        reports.append(report)
        out_path = qc_dir / f"align_inline_{model_index}.png"
        _plot_qc(lith_slice, seismic_up, model_index, seismic_index, out_path)
        print(
            f"inline model={model_index} seismic={seismic_index} "
            f"corr={report['edge_gradient_correlation']:.6f} "
            f"lift={report['boundary_gradient_lift']:.3f} "
            f"orientation={report['edge_seismic_orientation_cos2']:.6f} "
            f"-> {out_path}",
            flush=True,
        )

    correlations = [float(r["edge_gradient_correlation"]) for r in reports]
    orientation_scores = [
        float(r["edge_seismic_orientation_cos2"]) for r in reports
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lithology_path": str(lith_path.relative_to(args.root)),
        "seismic_crop_path": str(crop_path.relative_to(args.root)),
        "model_shape": list(lith.shape),
        "seismic_crop_shape": list(crop.shape),
        "scale": [2, 2, 5],
        "gradient_strength_note": (
            "Diagnostic only: reflector peaks/troughs can have low first-derivative "
            "magnitude even when their local orientation is aligned."
        ),
        "mean_edge_gradient_correlation": float(np.mean(correlations)),
        "orientation_random_baseline": 0.5,
        "minimum_required_orientation_score": args.min_orientation_score,
        "mean_edge_seismic_orientation_cos2": float(np.mean(orientation_scores)),
        "alignment_gate_passed": bool(
            all(value > args.min_orientation_score for value in orientation_scores)
        ),
        "slices": reports,
    }
    report_path = qc_dir / "alignment_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report: {report_path}", flush=True)
    if not summary["alignment_gate_passed"]:
        print("FAIL: one or more orientation scores missed the threshold.")
        return 2
    print("PASS: orientation alignment gate passed; inspect the overlay PNGs.")
    return 0


def _default_node_indices(model_size: int, seismic_size: int, count: int) -> list[int]:
    coarse = np.linspace(
        max(1, round((seismic_size - 1) * 0.1)),
        min(seismic_size - 2, round((seismic_size - 1) * 0.9)),
        count,
    )
    return [int(round(value)) * 2 for value in coarse]


def _alignment_metrics(lithology: np.ndarray, seismic: np.ndarray) -> dict[str, float | int]:
    binary = lithology > 0.5
    lith_float = binary.astype(np.float32)
    lith_dy = sobel(lith_float, axis=0, mode="nearest")
    lith_dx = sobel(lith_float, axis=1, mode="nearest")
    lith_gradient = np.hypot(lith_dy, lith_dx)
    raw_boundary = lith_gradient > 0
    boundary = binary_dilation(lith_gradient > 0, iterations=2)
    seismic_smooth = gaussian_filter(seismic, sigma=(1.0, 2.0), mode="nearest")
    seismic_dy = sobel(seismic_smooth, axis=0, mode="nearest")
    seismic_dx = sobel(seismic_smooth, axis=1, mode="nearest")
    seismic_gradient = np.hypot(seismic_dy, seismic_dx)
    finite = np.isfinite(seismic_gradient)
    boundary &= finite
    background = finite & ~boundary
    if boundary.sum() == 0 or background.sum() == 0:
        return {
            "edge_pixels": int(boundary.sum()),
            "edge_gradient_correlation": 0.0,
            "boundary_gradient_mean": 0.0,
            "background_gradient_mean": 0.0,
            "boundary_gradient_lift": 0.0,
            "orientation_pixels": 0,
            "edge_seismic_orientation_cos2": 0.0,
            "edge_seismic_orientation_excess_over_random": -0.5,
        }
    edge_values = seismic_gradient[boundary]
    background_values = seismic_gradient[background]
    edge_indicator = boundary[finite].astype(np.float32)
    gradients = seismic_gradient[finite].astype(np.float32)
    correlation = float(np.corrcoef(edge_indicator, gradients)[0, 1])
    edge_mean = float(np.mean(edge_values))
    background_mean = float(np.mean(background_values))
    gradient_floor = float(np.percentile(seismic_gradient[finite], 25.0))
    orientation_pixels = raw_boundary & finite & (seismic_gradient > gradient_floor)
    dot = (
        lith_dy[orientation_pixels] * seismic_dy[orientation_pixels]
        + lith_dx[orientation_pixels] * seismic_dx[orientation_pixels]
    )
    denominator = (
        lith_gradient[orientation_pixels] ** 2
        * seismic_gradient[orientation_pixels] ** 2
    )
    orientation = (dot * dot) / np.maximum(denominator, 1e-12)
    return {
        "edge_pixels": int(boundary.sum()),
        "edge_gradient_correlation": correlation if np.isfinite(correlation) else 0.0,
        "boundary_gradient_mean": edge_mean,
        "background_gradient_mean": background_mean,
        "boundary_gradient_lift": edge_mean / max(background_mean, 1e-12),
        "orientation_pixels": int(orientation_pixels.sum()),
        "edge_seismic_orientation_cos2": float(np.mean(orientation)),
        "edge_seismic_orientation_excess_over_random": float(
            np.mean(orientation) - 0.5
        ),
    }


def _plot_qc(
    lithology: np.ndarray,
    seismic: np.ndarray,
    model_index: int,
    seismic_index: int,
    out_path: Path,
) -> None:
    finite = seismic[np.isfinite(seismic)]
    limit = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    limit = max(limit, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    axes[0].imshow(lithology.T, cmap="gray", origin="upper", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title(f"Lithology model inline {model_index}")
    axes[1].imshow(
        seismic.T,
        cmap="gray",
        origin="upper",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )
    axes[1].set_title(f"Seismic crop inline {seismic_index}, node-resampled")
    axes[2].imshow(
        seismic.T,
        cmap="gray",
        origin="upper",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].contour(
        lithology.T,
        levels=[0.5],
        colors=["#ff3b30"],
        linewidths=0.45,
    )
    axes[2].set_title("Lithology boundary over seismic")
    for axis in axes:
        axis.set_xlabel("axis1")
        axis.set_ylabel("sample")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
