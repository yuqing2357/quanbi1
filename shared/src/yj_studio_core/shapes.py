"""Orientation-aware shape matching for hand-drawn morphology templates.

The desktop "形态模板识别" feature lets the user sketch a *shape* (e.g. a
left-high / right-low wedge) and search the current 2D section for structures
of a similar morphology. A hand drawing has no texture or amplitude, so it can
only be matched against the *geometry* of candidate regions — not via image
embeddings. This module supplies the geometry half:

  * :func:`rasterize_polygon` turns the drawn polygon into a binary template.
  * :func:`shape_similarity` scores one candidate mask against the template.
  * :func:`match_template_to_candidates` ranks a pool of candidate masks.

Design constraints that shape the metric:

* **Translation + scale invariant** — the target may sit anywhere and at any
  size, so every mask is cropped to its bounding box and resampled to a fixed
  square grid before comparison.
* **Orientation preserving** — "左高右低" must NOT match its mirror image. We
  therefore do *not* search over horizontal/vertical flips (that would make a
  mirrored wedge score high) and do *not* use rotation-invariant descriptors
  such as Hu moments (which collapse left/right wedges into one). The column /
  row fill profiles below are direction-sensitive: mirroring reverses a profile
  and drives the score down.

Pure NumPy on purpose: this runs on the GPU server (no matplotlib there) and in
local tests (no cv2 there), so it avoids both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable

import numpy as np

__all__ = [
    "rasterize_polygon",
    "shape_similarity",
    "match_template_to_candidates",
    "slice_foreground",
    "match_template_in_foreground",
]


def rasterize_polygon(
    points: Sequence[Sequence[float]],
    height: int,
    width: int,
    *,
    normalized: bool = True,
) -> np.ndarray:
    """Rasterize a polygon to a boolean mask of shape ``(height, width)``.

    ``points`` is a sequence of ``(x, y)`` vertices in drawing order. When
    ``normalized`` is true the coordinates are in ``[0, 1]`` (``x`` along width,
    ``y`` along height) and are scaled to pixel units; otherwise they are taken
    as raw pixel coordinates. Fewer than 3 vertices yields an all-False mask.

    Uses an even-odd scanline fill evaluated at pixel centres, so the result is
    independent of vertex winding and memory-light even on full-resolution
    seismic slices (one pass per row, vectorised over edges).
    """
    h = int(height)
    w = int(width)
    mask = np.zeros((h, w), dtype=bool)
    if h <= 0 or w <= 0:
        return mask
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return mask
    if normalized:
        pts = pts.copy()
        pts[:, 0] *= float(w)
        pts[:, 1] *= float(h)

    px = pts[:, 0]
    py = pts[:, 1]
    # Previous vertex for each edge (i-1 -> i), closing the polygon.
    pxj = np.roll(px, 1)
    pyj = np.roll(py, 1)

    y_lo = max(0, int(np.floor(py.min())))
    y_hi = min(h, int(np.ceil(py.max())) + 1)
    for row in range(y_lo, y_hi):
        yc = row + 0.5
        # Edges straddling this scanline (half-open test avoids double-counting
        # a vertex that lies exactly on the line).
        crosses = ((py <= yc) & (pyj > yc)) | ((pyj <= yc) & (py > yc))
        if not crosses.any():
            continue
        x_int = px[crosses] + (yc - py[crosses]) / (pyj[crosses] - py[crosses]) * (
            pxj[crosses] - px[crosses]
        )
        x_int.sort()
        # Fill between consecutive intersection pairs.
        for k in range(0, len(x_int) - 1, 2):
            left = x_int[k]
            right = x_int[k + 1]
            lo = int(np.ceil(left - 0.5))
            hi = int(np.floor(right - 0.5))
            if hi >= lo:
                mask[row, max(0, lo) : min(w, hi + 1)] = True
    return mask


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1), int(y1)


def _resample_to_grid(mask: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """Crop ``mask`` to its bounding box and nearest-resample to ``n x n``.

    Returns ``(grid, aspect)`` where ``aspect`` is the original bbox width/height
    ratio (kept separately because the square resample discards it). An empty
    mask yields an all-False grid and ``aspect == 1.0``.
    """
    bbox = _bbox(mask)
    if bbox is None:
        return np.zeros((n, n), dtype=bool), 1.0
    x0, y0, x1, y1 = bbox
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = crop.shape
    rows = np.clip((np.arange(n) * ch / n).astype(int), 0, ch - 1)
    cols = np.clip((np.arange(n) * cw / n).astype(int), 0, cw - 1)
    grid = crop[np.ix_(rows, cols)]
    aspect = float(cw) / float(ch) if ch > 0 else 1.0
    return grid, aspect


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _profile_similarity(a_grid: np.ndarray, b_grid: np.ndarray, axis: int) -> float:
    """1 - mean |fill profile difference| along ``axis`` (both in [0, 1]).

    Summing the resampled mask along one axis gives a direction-sensitive
    silhouette profile: a left-high wedge's column profile rises left-to-right,
    its mirror falls — so a flipped candidate scores low. Values are normalised
    by the grid size so the difference stays in ``[0, 1]``.
    """
    n = a_grid.shape[axis]
    if n == 0:
        return 0.0
    pa = a_grid.sum(axis=axis).astype(np.float64) / n
    pb = b_grid.sum(axis=axis).astype(np.float64) / n
    return float(1.0 - np.mean(np.abs(pa - pb)))


# Weights blend silhouette overlap (IoU) with direction-sensitive profiles and
# the bbox aspect ratio. IoU dominates; the profiles enforce left/right + up/down
# orientation; aspect recovers the wide-vs-tall distinction the square resample
# throws away.
_W_IOU = 0.5
_W_COL = 0.2
_W_ROW = 0.15
_W_ASPECT = 0.15


def shape_similarity(
    template_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    grid: int = 64,
) -> float:
    """Orientation-aware morphology similarity in ``[0, 1]`` (1 = identical).

    Both masks are cropped to their bounding boxes and resampled to a common
    ``grid x grid`` square (translation + scale invariant). The score blends the
    resampled IoU, direction-sensitive column/row fill profiles, and the bbox
    aspect ratio. It is intentionally NOT invariant to mirroring or rotation, so
    a left-high-right-low wedge does not match its mirror image.
    """
    t = np.asarray(template_mask, dtype=bool)
    c = np.asarray(candidate_mask, dtype=bool)
    if not t.any() or not c.any():
        return 0.0
    n = max(8, int(grid))
    t_grid, t_aspect = _resample_to_grid(t, n)
    c_grid, c_aspect = _resample_to_grid(c, n)

    iou = _iou(t_grid, c_grid)
    col_sim = _profile_similarity(t_grid, c_grid, axis=0)
    row_sim = _profile_similarity(t_grid, c_grid, axis=1)
    aspect_sim = min(t_aspect, c_aspect) / max(t_aspect, c_aspect) if max(t_aspect, c_aspect) > 0 else 1.0

    score = (
        _W_IOU * iou
        + _W_COL * col_sim
        + _W_ROW * row_sim
        + _W_ASPECT * aspect_sim
    )
    return float(max(0.0, min(1.0, score)))


def match_template_to_candidates(
    template_mask: np.ndarray,
    candidates: Iterable[np.ndarray],
    *,
    top_k: int | None = None,
    min_score: float = 0.0,
    grid: int = 64,
) -> list[tuple[int, float]]:
    """Rank ``candidates`` by morphology similarity to ``template_mask``.

    Returns ``[(candidate_index, score), ...]`` sorted by descending score,
    keeping only entries with ``score >= min_score`` and, if ``top_k`` is set,
    at most ``top_k`` of them. Indices refer to the position in ``candidates``.
    """
    scored: list[tuple[int, float]] = []
    for index, mask in enumerate(candidates):
        score = shape_similarity(template_mask, mask, grid=grid)
        if score >= min_score:
            scored.append((index, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if top_k is not None and top_k > 0:
        scored = scored[: int(top_k)]
    return scored


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's between-class-variance threshold for a 1-D array of finite values."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:
        return vmin
    hist, edges = np.histogram(vals, bins=256, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return vmin
    weight = np.cumsum(hist).astype(np.float64)
    mu = np.cumsum(hist * centers)
    mu_total = mu[-1]
    denom = weight * (total - weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * weight - mu) ** 2 / denom
    between[~np.isfinite(between)] = 0.0
    return float(centers[int(np.argmax(between))])


def slice_foreground(slice2d: np.ndarray, *, max_categories: int = 6) -> np.ndarray:
    """Boolean "has content" mask for a reservoir-model slice.

    Template search must ignore the empty background (the black region around the
    model) and only consider real structures. Reservoir models follow the
    convention *low value = background / non-reservoir, high value = body*:

    * categorical/segmented slices (few distinct values, e.g. a 0/1 lithology
      cube) → foreground is everything above the lowest category (the body);
    * continuous slices (e.g. porosity) → foreground is everything above an
      Otsu threshold.

    Non-finite samples (NaN fill outside the model) are always background.
    """
    arr = np.asarray(slice2d, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=bool)
    vals = arr[finite]
    uniques = np.unique(vals)
    if uniques.size <= 1:
        # Only one finite value: it is the body; background is the non-finite
        # (NaN) fill around the model, so every finite sample is foreground.
        return finite
    if uniques.size <= max_categories:
        threshold = float(uniques.min())
    else:
        threshold = _otsu_threshold(vals)
    return finite & (arr > threshold)


# Sliding-window scales as a fraction of the slice's shorter side, plus the
# stride (also a fraction of the window). Multiple scales make the search
# size-independent; a coarse stride keeps it well under a second on big slices.
_WINDOW_SCALES = (0.08, 0.13, 0.20, 0.30)
_STRIDE_FRAC = 0.5


def match_template_in_foreground(
    template_mask: np.ndarray,
    foreground: np.ndarray,
    *,
    top_k: int = 8,
    min_score: float = 0.0,
    min_fill: float = 0.04,
    max_fill: float = 0.92,
    nms_iou: float = 0.3,
    grid: int = 48,
    scales: Sequence[float] = _WINDOW_SCALES,
    stride_frac: float = _STRIDE_FRAC,
) -> list[dict[str, Any]]:
    """Find local regions whose foreground silhouette matches ``template_mask``.

    Slides multi-scale windows over the ``foreground`` content mask; at each
    window the local foreground silhouette is scored against the template with
    the orientation-aware :func:`shape_similarity`. Windows that are empty
    (``< min_fill``) or solid interior (``> max_fill``) are skipped — so the
    background never produces a hit and uniform interiors don't masquerade as
    shapes. Overlapping hits are de-duplicated by box NMS.

    Returns ``[{"mask", "score", "box"}]`` (best first, ``top_k`` max), where
    ``mask`` is the matched structure's foreground pixels embedded in a
    full-slice boolean array and ``box`` is the ``[x0, y0, x1, y1]`` window.
    """
    template = np.asarray(template_mask, dtype=bool)
    fg = np.asarray(foreground, dtype=bool)
    if not template.any() or not fg.any():
        return []
    h, w = fg.shape
    short = min(h, w)
    hits: list[tuple[float, tuple[int, int, int, int]]] = []
    for scale in scales:
        win = int(round(scale * short))
        if win < 8:
            continue
        stride = max(1, int(round(win * stride_frac)))
        for y0 in range(0, max(1, h - win + 1), stride):
            for x0 in range(0, max(1, w - win + 1), stride):
                y1 = min(h, y0 + win)
                x1 = min(w, x0 + win)
                sub = fg[y0:y1, x0:x1]
                fill = float(sub.mean())
                if fill < min_fill or fill > max_fill:
                    continue
                score = shape_similarity(template, sub, grid=grid)
                if score >= min_score:
                    hits.append((score, (x0, y0, x1 - 1, y1 - 1)))

    hits.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, tuple[int, int, int, int]]] = []
    for score, box in hits:
        if any(_box_iou(box, kb) >= nms_iou for _s, kb in kept):
            continue
        kept.append((score, box))
        if len(kept) >= max(1, int(top_k)):
            break

    results: list[dict[str, Any]] = []
    for score, (x0, y0, x1, y1) in kept:
        mask = np.zeros((h, w), dtype=bool)
        mask[y0 : y1 + 1, x0 : x1 + 1] = fg[y0 : y1 + 1, x0 : x1 + 1]
        results.append(
            {"mask": mask, "score": float(score), "box": [float(x0), float(y0), float(x1), float(y1)]}
        )
    return results


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0 + 1), max(0, iy1 - iy0 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0
