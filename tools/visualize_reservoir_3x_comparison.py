from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


OLD_DIR = Path("data/reservoir/numpy_3x")
NEW_DIR = Path("data/reservoir/numpy_3x_direct")
OUT_DIR = Path("data/results/reservoir_3x_direct_comparison")


def load_metadata(root: Path) -> dict:
    path = root / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def open_volumes(root: Path) -> tuple[np.ndarray, np.ndarray]:
    lith = np.load(root / "lithology_binary_3x_uint8.npy", mmap_mode="r")
    poro = np.load(root / "porosity_3x_float16.npy", mmap_mode="r")
    return lith, poro


def parse_positions(values: list[str] | None, size: int) -> list[int]:
    if not values:
        return [size // 4, size // 2, (size * 3) // 4]
    out: list[int] = []
    for value in values:
        text = value.strip()
        if text.endswith("%"):
            frac = float(text[:-1]) / 100.0
            idx = int(round(frac * (size - 1)))
        else:
            idx = int(text)
        out.append(max(0, min(size - 1, idx)))
    return out


def downsample_for_display(arr: np.ndarray, max_dim: int) -> np.ndarray:
    if max(arr.shape) <= max_dim:
        return np.asarray(arr)
    stride = int(np.ceil(max(arr.shape) / max_dim))
    return np.asarray(arr[::stride, ::stride])


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(volume[index, :, :])
    if axis == 1:
        return np.asarray(volume[:, index, :])
    if axis == 2:
        return np.asarray(volume[:, :, index])
    raise ValueError(f"axis must be 0, 1 or 2, got {axis}")


def robust_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def classify_lithology(lith: np.ndarray, poro: np.ndarray) -> np.ndarray:
    support = np.isfinite(np.asarray(poro, dtype=np.float32))
    out = np.zeros(np.asarray(lith).shape, dtype=np.uint8)
    out[support] = 1
    out[support & (np.asarray(lith) > 0)] = 2
    return out


def render_lithology(
    old: np.ndarray,
    new: np.ndarray,
    old_poro: np.ndarray,
    new_poro: np.ndarray,
    axis_name: str,
    index: int,
    out_path: Path,
    max_dim: int,
) -> None:
    old_ds = downsample_for_display(classify_lithology(old, old_poro), max_dim)
    new_ds = downsample_for_display(classify_lithology(new, new_poro), max_dim)
    support_diff = (new_ds > 0).astype(np.int16) - (old_ds > 0).astype(np.int16)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    lith_cmap = ListedColormap(["#ffffff", "#9b9b9b", "#ffe600"])
    axes[0].imshow(old_ds.T, origin="lower", cmap=lith_cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[0].set_title("old repeat 3x lithology")
    axes[1].imshow(new_ds.T, origin="lower", cmap=lith_cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title("new direct 3x lithology")
    axes[2].imshow(support_diff.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    axes[2].set_title("support new - old")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Lithology comparison | {axis_name}={index}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_porosity(old: np.ndarray, new: np.ndarray, axis_name: str, index: int, out_path: Path, max_dim: int) -> None:
    old_ds = downsample_for_display(old.astype(np.float32), max_dim)
    new_ds = downsample_for_display(new.astype(np.float32), max_dim)
    vmin, vmax = robust_limits(old_ds, new_ds)
    diff = np.abs(new_ds - old_ds)
    diff_max = float(np.nanpercentile(diff[np.isfinite(diff)], 99.0)) if np.isfinite(diff).any() else 1.0
    if diff_max <= 0.0 or not np.isfinite(diff_max):
        diff_max = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    axes[0].imshow(old_ds.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("old repeat 3x porosity")
    axes[1].imshow(new_ds.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("new direct 3x porosity")
    axes[2].imshow(diff.T, origin="lower", cmap="magma", vmin=0.0, vmax=diff_max, interpolation="nearest")
    axes[2].set_title("|new - old|")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Porosity comparison | {axis_name}={index}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def iter_requests(shape: tuple[int, int, int], args: argparse.Namespace) -> Iterable[tuple[int, str, int]]:
    axes = [(0, "axis0"), (1, "axis1"), (2, "sample")]
    for axis, name in axes:
        positions = parse_positions(getattr(args, f"{name}_positions"), shape[axis])
        for index in positions:
            yield axis, name, index


def main() -> None:
    parser = argparse.ArgumentParser(description="Render old-vs-new reservoir 3x comparison slices.")
    parser.add_argument("--old-dir", type=Path, default=OLD_DIR)
    parser.add_argument("--new-dir", type=Path, default=NEW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-dim", type=int, default=1800)
    parser.add_argument("--axis0-positions", nargs="*", default=None, help="Indices or percentages, e.g. 25%% 50%% 75%%.")
    parser.add_argument("--axis1-positions", nargs="*", default=None)
    parser.add_argument("--sample-positions", nargs="*", default=None)
    args = parser.parse_args()

    old_lith, old_poro = open_volumes(args.old_dir)
    new_lith, new_poro = open_volumes(args.new_dir)
    if old_lith.shape != new_lith.shape or old_poro.shape != new_poro.shape or old_lith.shape != old_poro.shape:
        raise ValueError(
            f"Shape mismatch: old_lith={old_lith.shape}, new_lith={new_lith.shape}, "
            f"old_poro={old_poro.shape}, new_poro={new_poro.shape}"
        )

    summary = {
        "old_dir": str(args.old_dir),
        "new_dir": str(args.new_dir),
        "shape": list(old_lith.shape),
        "old_metadata_method": load_metadata(args.old_dir).get("method"),
        "new_metadata_method": load_metadata(args.new_dir).get("method"),
        "images": [],
    }
    for axis, axis_name, index in iter_requests(old_lith.shape, args):
        old_lith_slice = take_slice(old_lith, axis, index)
        new_lith_slice = take_slice(new_lith, axis, index)
        old_poro_slice = take_slice(old_poro, axis, index)
        new_poro_slice = take_slice(new_poro, axis, index)

        lith_path = args.out_dir / f"lithology_{axis_name}_{index}.png"
        poro_path = args.out_dir / f"porosity_{axis_name}_{index}.png"
        render_lithology(
            old_lith_slice,
            new_lith_slice,
            old_poro_slice,
            new_poro_slice,
            axis_name,
            index,
            lith_path,
            args.max_dim,
        )
        render_porosity(old_poro_slice, new_poro_slice, axis_name, index, poro_path, args.max_dim)
        summary["images"].extend([str(lith_path), str(poro_path)])
        print(lith_path)
        print(poro_path)

    write_path = args.out_dir / "summary.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(write_path)


if __name__ == "__main__":
    main()
