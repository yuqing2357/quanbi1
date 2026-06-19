from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from bake_reservoir_npy import aligned_node_axis, native_column_polygon_footprint
from compare_reservoir_625_support import transect_gap_stats


def test_half_step_axis_contains_source_nodes_and_midpoints():
    lo, hi, coords = aligned_node_axis(2.2, 5.1, 10, 2)

    assert (lo, hi) == (2, 6)
    np.testing.assert_allclose(
        coords,
        [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    )


def test_axis_is_clipped_to_seismic_bounds_and_includes_last_node():
    lo, hi, coords = aligned_node_axis(-3.0, 20.0, 4, 5)

    assert (lo, hi) == (0, 3)
    assert len(coords) == 16
    assert coords[0] == 0.0
    assert coords[-1] == 3.0


def test_polygon_footprint_fills_enclosed_native_hole_but_not_exterior():
    axis0, axis1 = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    valid = np.zeros((5, 5), dtype=bool)
    valid[1:4, 1:4] = True
    valid[2, 2] = False

    footprint, native_support = native_column_polygon_footprint(
        axis0.astype(np.float32),
        axis1.astype(np.float32),
        valid,
        np.arange(6, dtype=np.float32),
        np.arange(6, dtype=np.float32),
    )

    assert native_support[2, 2]
    assert footprint[2, 2]
    assert not footprint[0, 0]
    assert not footprint[5, 5]


def test_transect_gap_stats_detects_bracketed_crack_only():
    mask = np.zeros((3, 7), dtype=bool)
    mask[0, 1:6] = True
    mask[1, 1:6] = True
    mask[1, 3] = False
    mask[2, 4:7] = True

    stats = transect_gap_stats(mask, axis=1)

    assert stats == {
        "transects_with_internal_gaps": 1,
        "internal_gap_runs": 1,
        "internal_gap_cells": 1,
        "max_internal_gap_width_cells": 1,
    }
