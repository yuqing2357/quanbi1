"""Coordinate mapping between a cropped/refined volume and seismic indices."""

from __future__ import annotations

from typing import Any, Mapping


AXES = ("axis0", "axis1", "sample")
SECTION_AXIS = {"inline": "axis0", "xline": "axis1", "z": "sample"}


def grid_reference_from_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(mapping or {})
    nested = source.get("grid_reference")
    if isinstance(nested, Mapping):
        source = {**source, **dict(nested)}

    origin = _triplet(source.get("seismic_index_origin"), default=(0.0, 0.0, 0.0))
    scale = _triplet(
        source.get("scale_axis0_axis1_sample"),
        default=(1.0, 1.0, 1.0),
    )
    if any(value <= 0 for value in scale):
        raise ValueError(f"Volume grid scale must be positive, got {scale}")
    return {
        "reference": "seismic_index",
        "seismic_index_origin": dict(zip(AXES, origin, strict=True)),
        "scale_axis0_axis1_sample": list(scale),
        "index_mapping": source.get(
            "index_mapping",
            "seismic_axis = origin + local_index / scale",
        ),
    }


def local_to_seismic_index(
    mapping: Mapping[str, Any] | None,
    axis: str,
    index: int | float,
) -> float:
    axis_name = SECTION_AXIS[str(axis)]
    reference = grid_reference_from_mapping(mapping)
    origin = float(reference["seismic_index_origin"][axis_name])
    scale = float(reference["scale_axis0_axis1_sample"][AXES.index(axis_name)])
    return origin + float(index) / scale


def seismic_to_local_index(
    mapping: Mapping[str, Any] | None,
    axis: str,
    seismic_index: int | float,
) -> int:
    axis_name = SECTION_AXIS[str(axis)]
    reference = grid_reference_from_mapping(mapping)
    origin = float(reference["seismic_index_origin"][axis_name])
    scale = float(reference["scale_axis0_axis1_sample"][AXES.index(axis_name)])
    return int(round((float(seismic_index) - origin) * scale))


def _triplet(value: object, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        return tuple(float(value.get(axis, default[pos])) for pos, axis in enumerate(AXES))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(float(value[pos]) for pos in range(3))
    return default
