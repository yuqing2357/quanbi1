"""Map sample/voxel indices to real-world physical coordinates for display.

The volume arrays are addressed by sample index, but the user wants axes to read
in metres (and, for the vertical axis, absolute depth). The physical metadata is
already carried on each ``VolumeLayer``:

* ``metadata["voxel_spacing"]`` – ``(dx, dy, dz)`` metres per voxel along
  ``(inline=axis0, xline=axis1, sample)`` (resolved server-side from
  ``voxel_spacing_m``).
* ``metadata["grid_reference"]["depth_range_m"]`` – ``[top_m, bottom_m]`` so the
  sample axis can show absolute depth (sample 0 = ``top_m``).

These helpers ONLY translate the numbers shown on the axes; they never rescale
geometry, so the existing aspect ratio / view proportions are preserved. When no
spacing is known (default ``(1, 1, 1)``) the helpers report "no physical mapping"
and callers fall back to plain sample-index labels.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Section axis -> (horizontal direction, vertical direction). Mirrors the
# orientation used by services.section_service.extract_orthogonal_section.
SECTION_DIRECTIONS: dict[str, tuple[str, str]] = {
    "inline": ("xline", "sample"),
    "xline": ("inline", "sample"),
    "z": ("inline", "xline"),
}

# Direction -> index into the (dx, dy, dz) spacing / (nx, ny, nz) shape triplet.
_DIRECTION_INDEX: dict[str, int] = {"inline": 0, "xline": 1, "sample": 2}
_DIRECTION_LABEL: dict[str, str] = {
    "inline": "Inline",
    "xline": "Xline",
    "sample": "深度 Z",
}


def voxel_spacing(metadata: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    """Effective ``(dx, dy, dz)`` metres per voxel, or ``None`` if not meaningful.

    Returns ``None`` for missing/invalid spacing and for the default unit spacing
    ``(1, 1, 1)`` so callers keep the legacy sample-index display in that case.
    """
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("voxel_spacing")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    try:
        spacing = tuple(float(v) for v in raw[:3])
    except (TypeError, ValueError):
        return None
    if any(not np.isfinite(v) or v <= 0.0 for v in spacing):
        return None
    if spacing == (1.0, 1.0, 1.0):
        return None
    return spacing  # type: ignore[return-value]


def depth_origin_m(metadata: Mapping[str, Any] | None) -> float | None:
    """Depth of sample 0 in metres, or ``None`` when no datum is known."""
    if not isinstance(metadata, Mapping):
        return None
    candidates: list[Any] = []
    grid = metadata.get("grid_reference")
    if isinstance(grid, Mapping):
        candidates.append(grid.get("depth_range_m"))
    candidates.append(metadata.get("depth_range_m"))
    for value in candidates:
        if isinstance(value, (list, tuple)) and value:
            try:
                return float(value[0])
            except (TypeError, ValueError):
                continue
    return None


def _direction_to_physical(direction: str, spacing: tuple[float, float, float], origin: float):
    """Return ``(scale, offset)`` so ``physical = offset + index * scale``."""
    scale = spacing[_DIRECTION_INDEX[direction]]
    offset = origin if direction == "sample" else 0.0
    return scale, offset


def _axis_label(direction: str) -> str:
    return f"{_DIRECTION_LABEL[direction]} (m)"


def apply_section_axis_units(
    axes: Any,
    section_axis: str,
    metadata: Mapping[str, Any] | None,
    *,
    fallback_x_label: str,
    fallback_y_label: str,
) -> bool:
    """Relabel a matplotlib section's ticks/labels in metres (depth for sample).

    The image ``extent`` stays in index units and ``aspect="equal"`` is untouched,
    so only the displayed numbers change — geometry/proportions are preserved.
    Returns ``True`` when physical units were applied, ``False`` when the axes were
    left in sample-index units (no usable spacing).
    """
    from matplotlib.ticker import FuncFormatter

    spacing = voxel_spacing(metadata)
    directions = SECTION_DIRECTIONS.get(section_axis)
    if spacing is None or directions is None:
        axes.set_xlabel(fallback_x_label)
        axes.set_ylabel(fallback_y_label)
        return False

    origin = depth_origin_m(metadata) or 0.0
    x_dir, y_dir = directions
    x_scale, x_offset = _direction_to_physical(x_dir, spacing, origin)
    y_scale, y_offset = _direction_to_physical(y_dir, spacing, origin)

    axes.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{x_offset + value * x_scale:,.0f}")
    )
    axes.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{y_offset + value * y_scale:,.0f}")
    )
    axes.set_xlabel(_axis_label(x_dir))
    axes.set_ylabel(_axis_label(y_dir))
    return True


def volume_axes_ranges(
    shape: tuple[int, int, int] | None,
    metadata: Mapping[str, Any] | None,
) -> tuple[list[float], tuple[str, str, str]] | None:
    """Physical ranges + titles for a 3D bounds box, or ``None`` if no spacing.

    ``ranges`` is ``[xmin, xmax, ymin, ymax, zmin, zmax]`` in metres, matching the
    mesh's index-space bounds ``[0, nx-1] x [0, ny-1] x [0, nz-1]``. The z axis is
    display-flipped (``display_z(k) = (nz-1) - k``), so mesh ``z=0`` is the deepest
    sample and ``z=nz-1`` the shallowest — the labels follow that so depth still
    increases downward. The mesh itself is left in index space (proportions kept);
    only the tick labels read in metres.
    """
    spacing = voxel_spacing(metadata)
    if spacing is None or shape is None or len(shape) < 3:
        return None
    try:
        nx, ny, nz = (int(s) for s in shape[:3])
    except (TypeError, ValueError):
        return None
    if min(nx, ny, nz) < 1:
        return None
    dx, dy, dz = spacing
    origin = depth_origin_m(metadata) or 0.0
    z_at_mesh_min = origin + (nz - 1) * dz  # deepest sample (mesh z = 0)
    z_at_mesh_max = origin                  # shallowest sample (mesh z = nz-1)
    ranges = [
        0.0,
        (nx - 1) * dx,
        0.0,
        (ny - 1) * dy,
        z_at_mesh_min,
        z_at_mesh_max,
    ]
    titles = (_axis_label("inline"), _axis_label("xline"), _axis_label("sample"))
    return ranges, titles
