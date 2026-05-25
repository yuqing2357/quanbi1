"""Render a reservoir section to a SAM3-ready image array.

SAM3 (image + video predictor) needs RGB inputs whose pixel grid is
**identical across frames** — otherwise the cross-frame appearance
model used by video propagation falls apart. We can't reuse the user-
facing matplotlib canvas for this: that canvas honours zoom/pan and
the figure size adapts to the dock; the resulting pixels shift as
soon as the user touches anything.

Instead, this module renders straight to an offscreen Agg canvas with
a deterministic bbox = the ROI's physical envelope. Shape is
``(H, W, 3) uint8`` and is the same for every (axis, index) inside
the same ROI.

Output details:

- Canvas size: a fixed ``(image_w, image_h)`` driven by ROI aspect.
  Defaults give SAM3 ~512px on the short side and stay under 2048
  on the long side, which keeps the model fast without losing
  cell-level detail on a typical ROI.
- Colour: LITHOLOGIES discrete palette (greys / yellows / cyans),
  matching the on-screen view so user-supplied prompts mean the same
  thing in both places.
- Axes have no spines/ticks/labels, so the image is pure data.

This is the function the SAM3 workbench widget feeds into the image
processor and (in step 5) into the video predictor frame-by-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from .grid import ReservoirGrid
from .palettes import palette_for
from .roi import ROI, roi_xy_bounds, roi_z_bounds
from .sections import (
    CellSection,
    clip_to_roi,
    extract_i_section,
    extract_j_section,
    values_for_section,
)
from .seismic_mapping import SeismicIndexTransform


# Use Agg explicitly — we don't want Qt or any interactive backend.
matplotlib.use("Agg", force=False)


# Target short-side and long-side caps. The actual canvas is sized to
# preserve the ROI's data aspect ratio (after z-exaggeration) so cells
# don't get stretched. 768x2560 leaves room for SAM3 to see cell-level
# detail in the vertical direction without sending GPU memory through
# the roof.
_TARGET_SHORT_PX = 1024
_MAX_LONG_PX = 3200
_DPI = 200


@dataclass(slots=True)
class SAM3Frame:
    """One frame's image plus the per-pixel cell-id lookup table.

    ``image`` is what we hand SAM3. ``cell_id_grid`` lets us reverse
    a pixel mask back to its source ``(i, j, k)`` triple without
    re-running matplotlib's transforms. Both arrays share the same
    ``(H, W)`` shape; pixels that don't sit inside any cell get
    ``(-1, -1, -1)`` in the cell-id grid.
    """

    image: np.ndarray            # (H, W, 3) uint8
    cell_id_grid: np.ndarray     # (H, W, 3) int32, -1 outside cells
    axis: str
    index: int
    roi: ROI
    # Mapping data coords → pixel coords (so prompts the user draws in
    # data space can be expressed as pixel boxes for SAM3).
    data_bbox: tuple[float, float, float, float]    # xmin, xmax, ymin, ymax


def render_roi_section(
    grid: ReservoirGrid,
    axis: str,
    index: int,
    roi: ROI,
    *,
    transform: SeismicIndexTransform | None = None,
    image_short_px: int = _TARGET_SHORT_PX,
    image_max_long_px: int = _MAX_LONG_PX,
    z_exaggeration: float = 10.0,
    build_cell_id_grid: bool = True,
) -> SAM3Frame:
    """Render one section frame inside the ROI to a SAM3-ready image.

    ``axis`` is 'i' or 'j' (K isn't supported as a propagation axis
    for SAM3 in this project). ``index`` is the slice index along
    that axis. ``roi`` defines the cube and therefore the fixed
    pixel grid.

    If ``build_cell_id_grid`` is False we skip the (slightly slow)
    cell-id rasterisation — useful when callers only need the image.
    """

    if axis not in {"i", "j"}:
        raise ValueError(f"axis must be 'i' or 'j', got {axis!r}")

    transform = transform or SeismicIndexTransform()

    # 1) Pull the section, clip to ROI.
    if axis == "i":
        section = extract_i_section(grid, index, transform=transform)
    else:
        section = extract_j_section(grid, index, transform=transform)
    section = clip_to_roi(section, roi)

    # 2) Compute the fixed data bbox (ROI physical envelope; same
    #    every frame so SAM3 sees a stable pixel grid).
    x0, x1, y0_xy, y1_xy = roi_xy_bounds(grid, roi)
    z_min, z_max = roi_z_bounds(grid, roi)
    v_lo = -z_max / transform.z_step
    v_hi = -z_min / transform.z_step
    if axis == "i":
        bbox = (y0_xy, y1_xy, v_lo, v_hi)
    else:
        bbox = (x0, x1, v_lo, v_hi)

    # 3) Pick a canvas size that preserves the bbox aspect ratio
    #    AFTER applying z-exaggeration. The reservoir is extremely
    #    flat (lateral kilometres × hundreds of metres depth), so we
    #    stretch the vertical so SAM3 sees cell-level detail in the
    #    short axis. The data isn't actually scaled — only the bbox
    #    used for axes limits is — so cell shapes get visually tall,
    #    same as in Petrel's default section view.
    bw_raw = max(bbox[1] - bbox[0], 1e-6)
    bh_raw = max(bbox[3] - bbox[2], 1e-6)
    bh_eff = bh_raw * float(z_exaggeration)
    bw = bw_raw
    bh = bh_eff
    if bw >= bh:
        long_px = min(image_max_long_px, max(image_short_px, int(round(image_short_px * bw / bh))))
        w_px = long_px
        h_px = max(image_short_px, int(round(long_px * bh / bw)))
    else:
        long_px = min(image_max_long_px, max(image_short_px, int(round(image_short_px * bh / bw))))
        h_px = long_px
        w_px = max(image_short_px, int(round(long_px * bw / bh)))

    # 4) Build the offscreen figure: no spines/ticks, axes fills the
    #    full canvas, fixed data window.
    figure = Figure(figsize=(w_px / _DPI, h_px / _DPI), dpi=_DPI)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))    # fill
    axes.set_xlim(bbox[0], bbox[1])
    axes.set_ylim(bbox[2], bbox[3])
    axes.set_aspect("auto")
    axes.set_axis_off()

    # 5) Paint cells. Use the LITHOLOGIES palette if available so the
    #    SAM3 frame looks like the on-screen view.
    if section.n_cells > 0:
        face_colors = _face_colors(grid, section)
        coll = PolyCollection(
            section.quads,
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0.0,
            antialiased=True,
        )
        axes.add_collection(coll)

    # ROI 物理外框：每帧固定，标记 SAM3 看到的视野范围。
    # 用 axes-fraction 坐标画，避免被 data-space 缩放吃掉。线宽用
    # 像素换算，保证不同 DPI 下视觉宽度一致。
    border_lw_px = max(2.0, h_px / 400.0)
    for spine_xy in (
        [(0, 0), (1, 0)], [(1, 0), (1, 1)],
        [(1, 1), (0, 1)], [(0, 1), (0, 0)],
    ):
        axes.plot(
            [spine_xy[0][0], spine_xy[1][0]],
            [spine_xy[0][1], spine_xy[1][1]],
            transform=axes.transAxes,
            color="black",
            linewidth=border_lw_px,
            solid_capstyle="butt",
            clip_on=False,
        )

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    image = np.ascontiguousarray(buf[..., :3])

    cell_id_grid = (
        _rasterize_cell_ids(section, bbox, image.shape[:2])
        if build_cell_id_grid else
        np.full((image.shape[0], image.shape[1], 3), -1, dtype=np.int32)
    )

    return SAM3Frame(
        image=image,
        cell_id_grid=cell_id_grid,
        axis=axis,
        index=index,
        roi=roi,
        data_bbox=bbox,
    )


def _face_colors(grid: ReservoirGrid, section: CellSection) -> np.ndarray:
    """Lithology-coloured face colours for each cell quad.

    Falls back to flat grey if LITHOLOGIES isn't present — keeps SAM3
    happy (it just needs a real image) even on unusual properties.
    """
    if grid.has_property("LITHOLOGIES"):
        vals = values_for_section(section, grid.property("LITHOLOGIES"))
        cmap = palette_for("LITHOLOGIES")
        if cmap is not None:
            return cmap(np.clip(vals.astype(np.int32), 0, cmap.N - 1))
    n = section.n_cells
    out = np.empty((n, 4), dtype=np.float32)
    out[:] = (0.7, 0.7, 0.75, 1.0)
    return out


def _rasterize_cell_ids(
    section: CellSection,
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Build a (H, W, 3) int32 array where each pixel holds the
    ``(i, j, k)`` of the cell that covers it.

    Implementation: for every quad, compute its pixel bbox via the
    same affine that matplotlib used; flood the bbox pixels with the
    cell's id. Quads overlap rarely on a single section (corner-point
    grids are space-filling), so a simple last-writer-wins fill is
    sufficient. Pixels not hit by any quad stay (-1, -1, -1).

    This is O(N_pixels_in_quad × N_cells); on a 1500 × 500 SAM3 image
    it runs in a few hundred milliseconds — fast enough to invalidate
    on every frame change.
    """
    h, w = image_shape
    xmin, xmax, ymin, ymax = bbox
    bw = max(xmax - xmin, 1e-6)
    bh = max(ymax - ymin, 1e-6)
    out = np.full((h, w, 3), -1, dtype=np.int32)

    if section.n_cells == 0:
        return out

    # All quads' pixel bboxes (rough; we don't do per-pixel inclusion
    # — quad overlap is negligible on a corner-point cross-section).
    q = section.quads     # (N, 4, 2)
    q_xmin = q[..., 0].min(axis=1)
    q_xmax = q[..., 0].max(axis=1)
    q_ymin = q[..., 1].min(axis=1)
    q_ymax = q[..., 1].max(axis=1)

    px_x0 = np.clip(((q_xmin - xmin) / bw * w).astype(np.int32), 0, w - 1)
    px_x1 = np.clip(((q_xmax - xmin) / bw * w).astype(np.int32) + 1, 0, w)
    # Y axis flips: data y increases up, image row increases down.
    px_y0 = np.clip((h - (q_ymax - ymin) / bh * h).astype(np.int32), 0, h - 1)
    px_y1 = np.clip((h - (q_ymin - ymin) / bh * h).astype(np.int32) + 1, 0, h)

    ids = section.cell_ids    # (N, 3) int32
    for idx in range(section.n_cells):
        y0, y1, x0, x1 = px_y0[idx], px_y1[idx], px_x0[idx], px_x1[idx]
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = ids[idx]
    return out
