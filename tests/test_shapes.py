"""Orientation-aware morphology matching for hand-drawn templates."""

from __future__ import annotations

import numpy as np

from yj_studio_core.shapes import (
    match_template_in_foreground,
    match_template_to_candidates,
    rasterize_polygon,
    shape_similarity,
    slice_foreground,
)


def _wedge(width: int, height: int, *, left_high: bool) -> np.ndarray:
    """A right-triangle wedge filling the slice.

    ``left_high`` → tall on the left, tapering to the right (the classic
    "左高右低" morphology). The mirror has the same outline flipped in x.
    """
    if left_high:
        pts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    else:
        pts = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    return rasterize_polygon(pts, height, width, normalized=True)


def test_rasterize_polygon_fills_triangle_interior():
    mask = _wedge(100, 100, left_high=True)
    # Bottom-left corner is inside the wedge; top-right corner is outside.
    assert mask[95, 5]
    assert not mask[5, 95]
    # Roughly half the slice is filled for a corner-to-corner triangle.
    frac = mask.mean()
    assert 0.4 < frac < 0.6


def test_rasterize_polygon_rejects_degenerate_input():
    assert not rasterize_polygon([(0.1, 0.1), (0.2, 0.2)], 32, 32).any()
    assert not rasterize_polygon([], 32, 32).any()


def test_shape_similarity_is_translation_and_scale_invariant():
    template = _wedge(120, 120, left_high=True)
    # Same wedge, smaller and shoved into a corner of a larger slice.
    small = _wedge(40, 40, left_high=True)
    candidate = np.zeros((200, 200), dtype=bool)
    candidate[10:50, 150:190] = small
    score = shape_similarity(template, candidate)
    assert score > 0.85


def test_shape_similarity_penalises_mirror_orientation():
    template = _wedge(120, 120, left_high=True)
    same = _wedge(120, 120, left_high=True)
    mirror = _wedge(120, 120, left_high=False)
    same_score = shape_similarity(template, same)
    mirror_score = shape_similarity(template, mirror)
    assert same_score > 0.95
    # Mirror must score clearly lower — orientation is preserved, not invariant.
    assert mirror_score < same_score - 0.2


def test_shape_similarity_empty_masks_score_zero():
    template = _wedge(64, 64, left_high=True)
    assert shape_similarity(template, np.zeros((64, 64), dtype=bool)) == 0.0
    assert shape_similarity(np.zeros((64, 64), dtype=bool), template) == 0.0


def test_match_ranks_same_orientation_above_mirror_and_respects_top_k():
    template = _wedge(100, 100, left_high=True)
    candidates = [
        _wedge(100, 100, left_high=False),  # 0: mirror
        np.zeros((100, 100), dtype=bool),   # 1: empty
        _wedge(100, 100, left_high=True),   # 2: exact
    ]
    ranked = match_template_to_candidates(template, candidates, top_k=2)
    assert len(ranked) == 2
    # Best match is the exact wedge; the empty mask is filtered to the bottom.
    assert ranked[0][0] == 2
    assert ranked[0][1] > ranked[1][1]
    assert all(idx != 1 for idx, _ in ranked)


def test_slice_foreground_excludes_background():
    # Binary lithology-style slice: 0 = background, 1 = body.
    slice2d = np.zeros((50, 50), dtype=np.float32)
    slice2d[10:30, 12:28] = 1.0
    fg = slice_foreground(slice2d)
    assert fg[20, 20]  # inside the body
    assert not fg[0, 0]  # background corner
    assert fg.sum() == 20 * 16


def test_slice_foreground_treats_nan_as_background():
    slice2d = np.full((20, 20), np.nan, dtype=np.float32)
    slice2d[5:15, 5:15] = 1.0
    fg = slice_foreground(slice2d)
    assert fg.sum() == 100
    assert not fg[0, 0]


def test_match_in_foreground_finds_structure_not_background():
    # A reservoir-style slice: empty background with one wedge-shaped body in the
    # lower-right; the rest is black (no content).
    h = w = 200
    fg = np.zeros((h, w), dtype=bool)
    body = _wedge(60, 60, left_high=True)
    fg[120:180, 120:180] = body
    template = _wedge(80, 80, left_high=True)

    matches = match_template_in_foreground(template, fg, top_k=3, min_score=0.3)
    assert matches, "expected at least one match on the body"
    # The top match must overlap the body region, never the empty background.
    top = matches[0]
    x0, y0, x1, y1 = top["box"]
    assert x1 >= 110 and y1 >= 110  # window sits over the lower-right body
    assert np.asarray(top["mask"]).any()
    # Every returned mask lies within the foreground (no background hits).
    for match in matches:
        m = np.asarray(match["mask"], dtype=bool)
        assert np.logical_and(m, ~fg).sum() == 0


def test_match_in_foreground_empty_when_no_content():
    template = _wedge(40, 40, left_high=True)
    assert match_template_in_foreground(template, np.zeros((80, 80), dtype=bool)) == []
