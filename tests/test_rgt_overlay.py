from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared/src"))

from yj_studio_core.reservoir import (  # noqa: E402
    INLINE,
    XLINE,
    RgtRenderParams,
    compute_rgt_span,
    extract_rgt_slice,
    render_rgt_section,
)
from yj_studio_core.reservoir.rgt_overlay import _turbo_map  # noqa: E402


def _toy_section():
    """(n_long=4, n_sample=6) with one sand column, one mud, one no-data."""
    n_long, n_sample = 4, 6
    lith = np.zeros((n_long, n_sample), np.float32)
    poro = np.zeros((n_long, n_sample), np.float32)
    lith[0, :] = 1.0          # sand column
    poro[0, :] = 0.2
    # column 1 stays mud (lith 0, finite poro)
    poro[2, :] = np.nan       # no-data column
    poro[3, :] = np.nan
    rgt = np.linspace(0, 1, (n_long // 2) * (n_sample // 5) or 1).astype(np.float32)
    rgt = np.tile(np.linspace(0, 1, n_sample, dtype=np.float32), (n_long, 1))
    return lith, poro, rgt


def test_output_shape_dtype_orientation() -> None:
    lith, poro, rgt = _toy_section()
    rgb = render_rgt_section(lith, poro, rgt)
    assert rgb.dtype == np.uint8
    # (n_sample, n_long, 3)
    assert rgb.shape == (lith.shape[1], lith.shape[0], 3)


def test_three_regions_get_distinct_base_colours() -> None:
    lith, poro, rgt = _toy_section()
    p = RgtRenderParams(smooth=False)  # raw labels, no gaussian bleed
    rgb = render_rgt_section(lith, poro, rgt, params=p)
    # columns map to axis-1 of the output
    nodata_px = rgb[:, 3, :]          # column 3 is NaN
    mud_px = rgb[:, 1, :]             # column 1 is mud
    np.testing.assert_array_equal(np.unique(nodata_px.reshape(-1, 3), axis=0), [[255, 255, 255]])
    np.testing.assert_array_equal(np.unique(mud_px.reshape(-1, 3), axis=0), [[0, 0, 0]])
    # sand column remains a coloured target, not black/white background
    sand_px = rgb[:, 0, :]
    assert not np.all(sand_px == 0)
    assert not np.all(sand_px == 255)


def test_render_is_deterministic() -> None:
    lith, poro, rgt = _toy_section()
    a = render_rgt_section(lith, poro, rgt)
    b = render_rgt_section(lith, poro, rgt)
    np.testing.assert_array_equal(a, b)


def test_fixed_span_decouples_colour_from_per_slice_stats() -> None:
    """Same RGT structure but different value offset must render identically
    under a fixed span (the SAM3-tracking requirement)."""
    lith, poro, rgt = _toy_section()
    span = compute_rgt_span(rgt)
    a = render_rgt_section(lith, poro, rgt, rgt_span=span)
    b = render_rgt_section(lith, poro, rgt + 10.0, rgt_span=(span[0] + 10.0, span[1] + 10.0))
    np.testing.assert_array_equal(a, b)
    # Holding the span fixed while the absolute RGT values shift DOES change the
    # colour (the whole point of pinning a span) — guard that too. smooth=False
    # keeps the lone sand column from being washed out on this tiny toy section.
    raw = RgtRenderParams(smooth=False)
    c = render_rgt_section(lith, poro, rgt, params=raw, rgt_span=span)
    d = render_rgt_section(lith, poro, rgt + 10.0, params=raw, rgt_span=span)
    assert not np.array_equal(c, d)  # different absolute values, same span -> different colour


def test_extract_rgt_slice_halves_index() -> None:
    vol = np.arange(6 * 5 * 4, dtype=np.float32).reshape(6, 5, 4)
    np.testing.assert_array_equal(extract_rgt_slice(vol, INLINE, 1300 % 6 * 2), vol[(1300 % 6 * 2) // 2])
    np.testing.assert_array_equal(extract_rgt_slice(vol, INLINE, 4), vol[2])
    np.testing.assert_array_equal(extract_rgt_slice(vol, XLINE, 6), vol[:, 3, :])


def test_turbo_lut_endpoints() -> None:
    lut = _turbo_map(np.array([0.0, 1.0], np.float32))
    # turbo starts dark blue, ends dark red
    assert lut[0, 2] > lut[0, 0]      # blue dominant at 0
    assert lut[1, 0] > lut[1, 2]      # red dominant at 1


def test_params_from_mapping_partial_and_ignores_unknown() -> None:
    p = RgtRenderParams.from_mapping({"alpha": 0.5, "sigma_lateral": 3.0, "bogus": 9})
    assert p.alpha == 0.5
    assert p.sigma_lateral == 3.0
    assert p.sigma_depth == RgtRenderParams().sigma_depth  # default preserved
