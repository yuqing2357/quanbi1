"""Aligned, on-demand slices from a logical multichannel volume."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_AXIS_ALIASES = {
    "axis0": 0,
    "inline": 0,
    "i": 0,
    "axis1": 1,
    "xline": 1,
    "crossline": 1,
    "j": 1,
    "axis2": 2,
    "sample": 2,
    "depth": 2,
    "z": 2,
}


def resample_nodes(src2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Linearly resample a 2D node grid while preserving endpoints and nodes."""

    src = np.asarray(src2d, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"src2d must be 2D, got shape {src.shape}")
    oh, ow = (int(out_hw[0]), int(out_hw[1]))
    if oh <= 0 or ow <= 0:
        raise ValueError(f"out_hw must be positive, got {out_hw}")
    if src.shape == (oh, ow):
        return src.copy()

    y = _node_coordinates(src.shape[0], oh)
    y0 = np.floor(y).astype(np.intp)
    y1 = np.minimum(y0 + 1, src.shape[0] - 1)
    wy = (y - y0).astype(np.float32)
    tmp = src[y0, :] * (1.0 - wy[:, None]) + src[y1, :] * wy[:, None]

    x = _node_coordinates(src.shape[1], ow)
    x0 = np.floor(x).astype(np.intp)
    x1 = np.minimum(x0 + 1, src.shape[1] - 1)
    wx = (x - x0).astype(np.float32)
    return tmp[:, x0] * (1.0 - wx[None, :]) + tmp[:, x1] * wx[None, :]


def extract_multichannel_slice(
    spec: str | Path | Mapping[str, Any],
    axis: str | int,
    index: int,
    *,
    project_root: str | Path | None = None,
) -> np.ndarray:
    """Return an aligned ``float32 (C, H, W)`` slice on the model grid."""

    payload, root = _load_spec(spec, project_root)
    model_shape = tuple(int(v) for v in payload["grid_model_shape"])
    scale = tuple(int(v) for v in payload["scale"])
    if len(model_shape) != 3 or len(scale) != 3:
        raise ValueError("grid_model_shape and scale must each have three values")

    axis_id = _axis_number(axis)
    idx = int(index)
    if not 0 <= idx < model_shape[axis_id]:
        raise IndexError(
            f"index {idx} outside axis {axis_id} range [0, {model_shape[axis_id]})"
        )
    out_hw = tuple(model_shape[i] for i in range(3) if i != axis_id)
    seismic_shape = tuple((model_shape[i] - 1) // scale[i] + 1 for i in range(3))

    slices: list[np.ndarray] = []
    for channel in payload["channels"]:
        path = Path(str(channel["path"]))
        if not path.is_absolute():
            path = root / path
        volume = np.load(path, mmap_mode="r")
        grid = str(channel["grid"])
        if grid == "model":
            expected_shape = model_shape
            source_index = idx
        elif grid == "seismic_crop":
            expected_shape = seismic_shape
            source_index = idx // scale[axis_id]
        else:
            raise ValueError(f"unsupported channel grid {grid!r}")
        if tuple(volume.shape) != expected_shape:
            raise ValueError(
                f"{channel['name']} shape {tuple(volume.shape)} != {expected_shape}"
            )

        selector: list[int | slice] = [slice(None), slice(None), slice(None)]
        selector[axis_id] = source_index
        channel_slice = np.asarray(volume[tuple(selector)], dtype=np.float32)
        if grid == "seismic_crop":
            channel_slice = resample_nodes(channel_slice, out_hw)
        channel_slice = _normalise(channel_slice, channel, payload)
        slices.append(channel_slice)

    if not slices:
        raise ValueError("channel_spec contains no channels")
    return np.stack(slices, axis=0).astype(np.float32, copy=False)


def _node_coordinates(source_size: int, output_size: int) -> np.ndarray:
    if source_size <= 0:
        raise ValueError("source dimensions must be positive")
    if source_size == 1:
        return np.zeros(output_size, dtype=np.float32)
    if output_size == 1:
        return np.zeros(1, dtype=np.float32)
    if (output_size - 1) % (source_size - 1) == 0:
        factor = (output_size - 1) // (source_size - 1)
        return np.arange(output_size, dtype=np.float32) / factor
    return np.linspace(0.0, source_size - 1, output_size, dtype=np.float32)


def _axis_number(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis in (0, 1, 2):
            return axis
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    try:
        return _AXIS_ALIASES[str(axis).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported axis {axis!r}") from exc


def _load_spec(
    spec: str | Path | Mapping[str, Any],
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(spec, Mapping):
        if project_root is None:
            raise ValueError("project_root is required when spec is a mapping")
        return dict(spec), Path(project_root).expanduser().resolve()

    spec_path = Path(spec).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        root = _find_project_root(spec_path)
    return payload, root


def _find_project_root(spec_path: Path) -> Path:
    for parent in spec_path.parents:
        if (parent / "data").is_dir():
            return parent
    raise ValueError(
        f"could not infer project root from {spec_path}; pass project_root explicitly"
    )


def _normalise(
    values: np.ndarray,
    channel: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> np.ndarray:
    norm = str(channel.get("norm", "as_is"))
    if norm == "as_is":
        return values
    if norm == "clip01_porosity":
        finite = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(finite, 0.0, 1.0)
    if norm == "stats_linear":
        stats = spec.get("stats", {}).get(channel["name"], {})
        scale = float(stats.get("scale", 1.0))
        offset = float(stats.get("offset", 0.0))
        return np.clip(values * scale + offset, 0.0, 1.0)
    raise ValueError(f"unsupported norm {norm!r} for channel {channel['name']!r}")
