from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_VOXEL_SPACING: tuple[float, float, float] = (1.0, 1.0, 1.0)


def resolve_voxel_spacing(volume_spec: dict[str, Any] | None) -> tuple[tuple[float, float, float], str]:
    """Return effective physical voxel spacing as ``(dx, dy, dz)``.

    Server volume config may specify an already-effective
    ``voxel_spacing``/``voxel_spacing_m`` triplet or metadata-style
    ``axis0``/``axis1``/``sample`` fields. A base spacing plus a downsample
    factor is still accepted for other derived volumes.
    """

    spec = dict(volume_spec or {})
    spacing = _spacing_triplet(
        spec.get("voxel_spacing_m")
        or spec.get("voxel_spacing")
        or spec.get("spacing_m")
        or spec.get("spacing")
    )
    source = "config"
    if spacing is None:
        spacing = _dx_dy_dz(spec)
    if spacing is None:
        spacing = DEFAULT_VOXEL_SPACING
        source = "default"

    factor = _downsample_triplet(spec.get("downsample_factor") or spec.get("downsample"))
    if factor is not None:
        spacing = tuple(float(a) * float(b) for a, b in zip(spacing, factor, strict=True))
        source = f"{source}+downsample"
    return spacing, source


def mask_volume_stats(
    mask: np.ndarray,
    voxel_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
) -> dict[str, float | int | tuple[float, float, float]]:
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"mask3d must have shape (D, H, W), got {arr.shape}")
    spacing = tuple(float(v) for v in voxel_spacing)
    voxel_count = int(np.count_nonzero(arr > 0))
    cell_volume = float(spacing[0] * spacing[1] * spacing[2])
    return {
        "voxel_count": voxel_count,
        "voxel_spacing": spacing,
        "voxel_volume": cell_volume,
        "volume_m3": float(voxel_count) * cell_volume,
    }


def _spacing_triplet(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        return _dx_dy_dz(value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return values if all(v > 0.0 for v in values) else None


def _dx_dy_dz(value: dict[str, Any]) -> tuple[float, float, float] | None:
    axis_spacing = _axis0_axis1_sample(value)
    if axis_spacing is not None:
        return axis_spacing
    try:
        spacing = (float(value["dx"]), float(value["dy"]), float(value["dz"]))
    except (KeyError, TypeError, ValueError):
        return None
    return spacing if all(v > 0.0 for v in spacing) else None


def _axis0_axis1_sample(value: dict[str, Any]) -> tuple[float, float, float] | None:
    try:
        spacing = (float(value["axis0"]), float(value["axis1"]), float(value["sample"]))
    except (KeyError, TypeError, ValueError):
        return None
    return spacing if all(v > 0.0 for v in spacing) else None


def _downsample_triplet(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        factor = float(value)
        return (factor, factor, factor) if factor > 0.0 else None
    return _spacing_triplet(value)
