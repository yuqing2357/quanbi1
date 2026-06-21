from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from yj_studio.view.physical_axes import (
    apply_section_axis_units,
    depth_origin_m,
    volume_axes_ranges,
    voxel_spacing,
)

# Reservoir model spacing (axis0/inline, axis1/xline, sample) + depth datum.
RESERVOIR_META = {
    "voxel_spacing": [6.25, 6.25, 2.0],
    "grid_reference": {"depth_range_m": [880.0, 6530.0]},
}


def test_voxel_spacing_rejects_default_and_invalid() -> None:
    assert voxel_spacing(RESERVOIR_META) == (6.25, 6.25, 2.0)
    assert voxel_spacing({"voxel_spacing": [1.0, 1.0, 1.0]}) is None
    assert voxel_spacing({"voxel_spacing": [6.25, 6.25, 0.0]}) is None
    assert voxel_spacing({}) is None
    assert voxel_spacing(None) is None


def test_depth_origin_reads_grid_reference_or_top_level() -> None:
    assert depth_origin_m(RESERVOIR_META) == 880.0
    assert depth_origin_m({"depth_range_m": [120.0, 500.0]}) == 120.0
    assert depth_origin_m({"voxel_spacing": [6.25, 6.25, 2.0]}) is None


def _tick_text(axes, which: str, index_value: float) -> str:
    formatter = (axes.xaxis if which == "x" else axes.yaxis).get_major_formatter()
    return formatter(index_value)


def test_inline_section_labels_depth_in_metres() -> None:
    axes = Figure().add_subplot(111)
    applied = apply_section_axis_units(
        axes,
        "inline",
        RESERVOIR_META,
        fallback_x_label="Xline",
        fallback_y_label="Sample",
    )
    assert applied is True
    # Inline section: x=Xline (6.25 m), y=depth (880 + k*2)
    assert _tick_text(axes, "x", 100) == "625"
    assert _tick_text(axes, "y", 0) == "880"
    assert _tick_text(axes, "y", 500) == "1,880"
    assert axes.get_ylabel() == "深度 Z (m)"
    assert axes.get_xlabel() == "Xline (m)"


def test_z_section_labels_both_axes_in_metres_no_depth_offset() -> None:
    axes = Figure().add_subplot(111)
    apply_section_axis_units(
        axes,
        "z",
        RESERVOIR_META,
        fallback_x_label="Inline",
        fallback_y_label="Xline",
    )
    assert _tick_text(axes, "x", 8) == "50"   # 8 * 6.25
    assert _tick_text(axes, "y", 8) == "50"
    assert axes.get_xlabel() == "Inline (m)"
    assert axes.get_ylabel() == "Xline (m)"


def test_section_falls_back_to_sample_index_without_spacing() -> None:
    axes = Figure().add_subplot(111)
    applied = apply_section_axis_units(
        axes,
        "inline",
        {"voxel_spacing": [1.0, 1.0, 1.0]},
        fallback_x_label="Xline",
        fallback_y_label="Sample",
    )
    assert applied is False
    assert axes.get_xlabel() == "Xline"
    assert axes.get_ylabel() == "Sample"


def test_volume_axes_ranges_flips_depth_for_display_z() -> None:
    result = volume_axes_ranges((2959, 2201, 2826), RESERVOIR_META)
    assert result is not None
    ranges, titles = result
    # x,y are 0..(n-1)*spacing
    assert ranges[0] == 0.0 and ranges[1] == (2959 - 1) * 6.25
    assert ranges[2] == 0.0 and ranges[3] == (2201 - 1) * 6.25
    # mesh z=0 is deepest (shallowest sample flipped to top), so zmin label is deepest
    assert ranges[4] == 880.0 + (2826 - 1) * 2.0  # deepest
    assert ranges[5] == 880.0                      # shallowest
    assert titles == ("Inline (m)", "Xline (m)", "深度 Z (m)")


def test_volume_axes_ranges_none_without_spacing() -> None:
    assert volume_axes_ranges((10, 10, 10), {"voxel_spacing": [1.0, 1.0, 1.0]}) is None
    assert volume_axes_ranges(None, RESERVOIR_META) is None
