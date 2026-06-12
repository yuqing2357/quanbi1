"""2D Petrel-style cell-section view for a ``ReservoirGrid``.

Each section displays one of the three IJK planes (K = horizontal
layer view, I/J = vertical along-axis slabs) as a matplotlib
PolyCollection — every active cell is rendered as a quadrilateral
coloured by the chosen property (LITHOLOGIES, PORO, ...).

The view is intentionally lean — it doesn't pull in the seismic/
horizon/well overlays that ``View2DSection`` carries; reservoir
sections live in the grid's own IJK frame, not the seismic
sample-index frame. Coordinate alignment with seismic happens
through ``SeismicIndexTransform`` (the vertical axis on I/J
sections is sample-index, so it stacks naturally next to seismic
sections drawn the same way).

Cell-id buffer is kept on the canvas so a future click handler can
map ``(x, y) → (i, j, k)`` cheaply — that's the entry point the SAM3
reverse-lookup will use in Phase 11.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from yj_studio.reservoir import ROI, ReservoirGrid, SeismicIndexTransform
from yj_studio.reservoir.palettes import palette_for
from yj_studio.reservoir.roi import (
    decompose as decompose_roi,
    default_roi,
    roi_xy_bounds,
    roi_z_bounds,
)
from yj_studio.reservoir.sections import (
    CellSection,
    clip_to_roi,
    extract_i_section,
    extract_j_section,
    values_for_section,
)


# Reservoir sections only support I/J — the user's workflow never
# enters from the K (depth) direction. If we ever need a K-layer
# overview (top-down map), add it as a separate widget rather than
# folding it back into this view; the affordances (sliding axis,
# ROI box, SAM3 propagation tunnel) are I/J-specific.
_AXIS_LABELS = {
    "i": "I 剖面 (inline)",
    "j": "J 剖面 (xline)",
}
_AXIS_AXIS_NAMES = {
    "i": ("Y (m)", "Sample"),
    "j": ("X (m)", "Sample"),
}

# Fixed rendering geometry. These define the pixel grid SAM3 will
# see and must NOT change once a section is rendered.
#
# Per-axis canvas sizes (data aspect ratios are very different):
#   K layer  ≈ square footprint (~19 km × ~14 km) → near-square canvas
#   I/J      ≈ wide vertical slabs (~14 km lateral × few-hundred sample
#              depth, roughly 30:1 → 14:1 after using active sample range)
#              → wide-short canvas, otherwise SAM3 gets a few-pixel-tall
#              data band drowning in whitespace.
_FIGURE_DPI = 100
_FIGURE_SIZE_BY_AXIS = {
    "i": (14.0, 5.0),     # 1400 × 500
    "j": (14.0, 5.0),     # 1400 × 500
}
# Axes rect ≈ as tight as we can go and still keep axis labels visible.
# 4 % left, 4 % right; 12 % bottom for x-label, 10 % top for title.
_AXES_RECT = (0.04, 0.12, 0.92, 0.78)


class ViewReservoirSection(QWidget):
    """2D cell-quad section of a reservoir grid."""

    section_updated = pyqtSignal(str, str, int)
    cell_clicked = pyqtSignal(int, int, int)    # (i, j, k)
    # Fires when the user finishes drawing a ROI rectangle. Carries
    # the new ROI tuple ``(i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)``.
    # Listeners (main_window / layer store) push the value onto
    # ReservoirGridLayer.roi and propagate to other open sections.
    roi_changed = pyqtSignal(tuple)
    # Emitted after the user finishes drawing a rectangle. Carries the
    # newly defined ROI and the axis ('i' or 'j') the rectangle was
    # drawn on, so main_window can spin up a SAM3 workbench bound to
    # that ROI and propagation axis.
    roi_drawn = pyqtSignal(tuple, str)

    def __init__(
        self,
        grid: ReservoirGrid,
        *,
        axis: str = "i",
        index: int | None = None,
        property_name: str | None = None,
        transform: SeismicIndexTransform | None = None,
        roi: ROI | None = None,
        section_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.section_id = section_id or str(uuid4())
        self._grid = grid
        self._transform = transform or SeismicIndexTransform()
        # ROI defines the shared cube every operation works inside —
        # axes window, allowed index range, and (eventually) SAM3
        # render target. Defaults to the active-cell bbox so users see
        # something sensible immediately.
        self._roi: ROI = roi if roi is not None else default_roi(grid)
        self._axis = axis
        self._index = self._default_index(axis) if index is None else int(index)
        self._property_name = property_name or self._default_property()

        self._section: CellSection | None = None
        self._collection: PolyCollection | None = None
        self._selector: RectangleSelector | None = None

        self._build_ui()
        self._sync_roi_button_enabled()
        self._render()

    # ------------------------------------------------------------------ public

    @property
    def axis(self) -> str:
        return self._axis

    @property
    def index(self) -> int:
        return self._index

    @property
    def title(self) -> str:
        return f"{self._grid.master_path.name if hasattr(self._grid, 'master_path') else 'Reservoir'} · {_AXIS_LABELS[self._axis]} {self._index}"

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 0)

        controls.addWidget(QLabel("剖面:"))
        self._axis_combo = QComboBox()
        for key, label in _AXIS_LABELS.items():
            self._axis_combo.addItem(label, key)
        self._axis_combo.setCurrentIndex(list(_AXIS_LABELS).index(self._axis))
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        controls.addWidget(self._axis_combo)

        controls.addWidget(QLabel("索引:"))
        self._index_spin = QSpinBox()
        self._index_spin.setRange(self._axis_min(), self._axis_max())
        self._index_spin.setValue(self._index)
        self._index_spin.valueChanged.connect(self._on_index_changed)
        controls.addWidget(self._index_spin)

        controls.addWidget(QLabel("属性:"))
        self._prop_combo = QComboBox()
        self._prop_combo.addItem("(无)", "")
        for name in self._grid.property_names():
            self._prop_combo.addItem(name, name)
        if self._property_name:
            ix = self._prop_combo.findData(self._property_name)
            if ix >= 0:
                self._prop_combo.setCurrentIndex(ix)
        self._prop_combo.currentIndexChanged.connect(self._on_property_changed)
        controls.addWidget(self._prop_combo)

        # ROI tools — "切西瓜": draw a rectangle on the current section
        # to clip i+j (on K layer) or j+k / i+k (on I/J sections).
        # Successive boxes intersect, so the user can progressively
        # narrow the cube. Reset returns to the active-cell bbox.
        self._roi_button = QToolButton()
        self._roi_button.setText("画框 ROI")
        self._roi_button.setCheckable(True)
        self._roi_button.setToolTip(
            "在剖面上拖矩形定义 ROI。后续在其他剖面再拖会取交集，逐步缩小目标体。"
        )
        self._roi_button.toggled.connect(self._on_roi_button_toggled)
        controls.addWidget(self._roi_button)

        self._roi_reset_button = QToolButton()
        self._roi_reset_button.setText("重置 ROI")
        self._roi_reset_button.setToolTip("恢复 ROI 为整个活动单元包围盒")
        self._roi_reset_button.clicked.connect(self._on_roi_reset)
        controls.addWidget(self._roi_reset_button)

        controls.addStretch(1)
        layout.addLayout(controls)

        # QScrollArea holds the canvas at a fixed pixel size. We rebuild
        # the figure + canvas every time the axis changes (so K can use
        # near-square pixels and I/J can use wide-short ones), then swap
        # them into the scroll area. The scroll widget itself is created
        # once here.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._scroll, stretch=1)
        self._build_canvas_for_axis()

    def _build_canvas_for_axis(self) -> None:
        """(Re)create figure + canvas sized for the current axis.

        Figure size is fixed at construction in matplotlib (no
        setSize), so to switch K ↔ I/J pixel grids we throw away the
        old figure and make a new one. The SAM3-facing helpers
        (``axes_pixel_rect``, ``to_image_array``) automatically pick
        up the new dimensions because they read _FIGURE_SIZE_BY_AXIS
        live.
        """

        size = _FIGURE_SIZE_BY_AXIS[self._axis]
        self._figure = Figure(figsize=size, dpi=_FIGURE_DPI)
        self._axes = self._figure.add_axes(_AXES_RECT)
        self._canvas = FigureCanvasQTAgg(self._figure)
        canvas_px_w = int(size[0] * _FIGURE_DPI)
        canvas_px_h = int(size[1] * _FIGURE_DPI)
        self._canvas.setFixedSize(QSize(canvas_px_w, canvas_px_h))
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        # Mouse wheel zoom (centred on the cursor) + middle-mouse pan.
        # Both work in data coordinates so the cell-at-pixel reverse
        # lookup keeps working at any zoom level — matplotlib stores
        # data coords on the event and we never touch the underlying
        # quad positions.
        self._canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
        self._canvas.mpl_connect("button_press_event", self._on_pan_press)
        self._canvas.mpl_connect("button_release_event", self._on_pan_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_pan_move)
        self._pan_anchor: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None
        # Rectangle selector for ROI drawing. Inactive by default so
        # plain clicks still go to cell_clicked; the user activates it
        # via the "画框 ROI" toggle. matplotlib's RectangleSelector
        # has to be re-created with each new axes.
        self._selector = RectangleSelector(
            self._axes,
            self._on_rect_drawn,
            useblit=True,
            interactive=False,
            button=[1],     # left mouse only
            props=dict(facecolor="yellow", edgecolor="orange", alpha=0.25),
        )
        self._selector.set_active(False)
        # If the toggle is currently on (axis switch while drawing),
        # reactivate.
        if getattr(self, "_roi_button", None) and self._roi_button.isChecked():
            self._selector.set_active(True)
        # takeWidget detaches the previous canvas; Qt will delete it
        # later as it goes out of scope.
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._scroll.setWidget(self._canvas)

    # ------------------------------------------------------------------ slots

    def _on_axis_changed(self, _idx: int) -> None:
        new_axis = self._axis_combo.currentData()
        if new_axis == self._axis:
            return
        self._axis = new_axis
        self._index = self._default_index(new_axis)
        self._index_spin.blockSignals(True)
        self._index_spin.setRange(self._axis_min(), self._axis_max())
        self._index_spin.setValue(self._index)
        self._index_spin.blockSignals(False)
        # Different axes use different canvas pixel sizes — rebuild.
        self._build_canvas_for_axis()
        # ROI drawing only makes sense on I/J sections (sliding axis
        # interpretation). Disable on K layer.
        self._sync_roi_button_enabled()
        self._render(reset_view=True)

    def _sync_roi_button_enabled(self) -> None:
        # Both I and J support ROI drawing; this is kept as a no-op
        # for now so the future "K layer view" addition can flip the
        # button enabled state without restructuring the call sites.
        self._roi_button.setEnabled(True)
        self._roi_button.setToolTip(
            "在剖面上拖矩形定义 ROI（限定其余两轴范围，"
            "沿剖面轴贯穿整个网格，作为后续 SAM3 视频追踪的隧道）。"
        )

    def _on_index_changed(self, value: int) -> None:
        self._index = int(value)
        self._render()

    def _on_property_changed(self, _idx: int) -> None:
        self._property_name = self._prop_combo.currentData() or None
        self._render()

    def _on_canvas_scroll(self, event) -> None:
        """Zoom in/out centred on the cursor (mouse wheel)."""
        if event.inaxes is not self._axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = self._axes.get_xlim()
        y0, y1 = self._axes.get_ylim()
        cx, cy = float(event.xdata), float(event.ydata)
        # Scale interval around (cx, cy).
        self._axes.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        self._axes.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        self._canvas.draw_idle()

    def _on_pan_press(self, event) -> None:
        """Middle-button drag starts a pan."""
        if event.button != 2 or event.inaxes is not self._axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._pan_anchor = (
            float(event.xdata), float(event.ydata),
            self._axes.get_xlim(), self._axes.get_ylim(),
        )

    def _on_pan_move(self, event) -> None:
        if self._pan_anchor is None or event.inaxes is not self._axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax, ay, (x0, x1), (y0, y1) = self._pan_anchor
        dx = float(event.xdata) - ax
        dy = float(event.ydata) - ay
        self._axes.set_xlim(x0 - dx, x1 - dx)
        self._axes.set_ylim(y0 - dy, y1 - dy)
        self._canvas.draw_idle()

    def _on_pan_release(self, event) -> None:
        if event.button == 2:
            self._pan_anchor = None

    def _on_roi_button_toggled(self, on: bool) -> None:
        """Enable/disable the rectangle selector for ROI drawing."""
        if self._selector is not None:
            self._selector.set_active(on)

    def _on_roi_reset(self) -> None:
        """Restore the default active-cell ROI."""
        new_roi = default_roi(self._grid)
        if new_roi != self._roi:
            self._roi = new_roi
            self.roi_changed.emit(new_roi)
            self.set_roi(new_roi)    # local refresh

    def _on_rect_drawn(self, click, release) -> None:
        """Compute a new ROI from the rectangle the user just drew.

        The rectangle clips the *two* axes visible on this section:
            K layer → clips i and j; k range stays as before
            I section → clips j and k; i range stays
            J section → clips i and k; j range stays

        The result is intersected with the current ROI so successive
        boxes progressively narrow the cube ("切西瓜"). The reset
        button restores the full active-cell ROI.
        """
        if click.xdata is None or release.xdata is None:
            return
        x0, x1 = sorted((float(click.xdata), float(release.xdata)))
        y0, y1 = sorted((float(click.ydata), float(release.ydata)))
        new_roi = self._roi_from_rect(x0, x1, y0, y1)
        if new_roi is None:
            return
        # Auto-disable the selector after one draw, mirroring how
        # screenshot tools behave. User can re-enable with the toggle.
        self._roi_button.setChecked(False)
        if new_roi != self._roi:
            # Apply locally first (so set_roi's no-op early-out doesn't
            # swallow the update), then broadcast so other views and
            # the layer follow.
            self.set_roi(new_roi)
            self.roi_changed.emit(new_roi)
        # Always announce the draw — even when the new ROI equals the
        # existing one (e.g. the user redraws the same box on purpose).
        # main_window listens to spawn a SAM3 workbench bound to this
        # ROI + propagation axis.
        self.roi_drawn.emit(new_roi, self._axis)

    def _roi_from_rect(
        self, x0: float, x1: float, y0: float, y1: float
    ) -> ROI | None:
        """Map a 2D rectangle on an I/J section to a "sliding tunnel" ROI.

        The rectangle clips two of the three IJK axes; the third — the
        section's own slicing axis — is forced to span the full grid.
        That's the "video propagation tunnel" the user wants: pick a
        cross-section of interest on I (j,k box), then slide along i
        like watching a video.

        - I section → clip j and k, i = (0, nx).
        - J section → clip i and k, j = (0, ny).
        - K layer → not supported (button disabled), but return None
          defensively if it ever gets here.

        Returns None if no cells fall in the rectangle.
        """
        if self._axis not in {"i", "j"}:
            return None
        if self._section is None or self._section.n_cells == 0:
            return None
        q = self._section.quads
        centroids = q.mean(axis=1)
        in_rect = (
            (centroids[:, 0] >= x0) & (centroids[:, 0] <= x1)
            & (centroids[:, 1] >= y0) & (centroids[:, 1] <= y1)
        )
        if not in_rect.any():
            return None
        ids = self._section.cell_ids[in_rect]
        nx, ny, nz = self._grid.shape
        if self._axis == "i":
            j_lo, j_hi = int(ids[:, 1].min()), int(ids[:, 1].max()) + 1
            k_lo, k_hi = int(ids[:, 2].min()), int(ids[:, 2].max()) + 1
            return (0, nx, j_lo, j_hi, k_lo, k_hi)
        # j section
        i_lo, i_hi = int(ids[:, 0].min()), int(ids[:, 0].max()) + 1
        k_lo, k_hi = int(ids[:, 2].min()), int(ids[:, 2].max()) + 1
        return (i_lo, i_hi, 0, ny, k_lo, k_hi)

    def set_roi(self, roi: ROI) -> None:
        """Update the active ROI and re-render.

        Called by the layer-watcher when the user redraws the ROI in
        the 3D view, or by external code (e.g. SAM3 export) that
        wants the section to reframe.
        """

        if roi == self._roi:
            return
        self._roi = roi
        # Reclamp the current index into the new ROI's axis range.
        lo, hi = self._axis_range()
        self._index = max(lo, min(self._index, hi - 1))
        self._index_spin.blockSignals(True)
        self._index_spin.setRange(lo, max(lo, hi - 1))
        self._index_spin.setValue(self._index)
        self._index_spin.blockSignals(False)
        self._render(reset_view=True)

    def _on_canvas_click(self, event) -> None:
        if event.inaxes is not self._axes:
            return
        if self._section is None or self._section.n_cells == 0:
            return
        if event.xdata is None or event.ydata is None:
            return
        cell_id = self._cell_at(event.xdata, event.ydata)
        if cell_id is not None:
            i, j, k = cell_id
            self.cell_clicked.emit(int(i), int(j), int(k))

    # ------------------------------------------------------------------ render

    def _render(self, *, reset_view: bool = False) -> None:
        # Decide *before* clearing whether we should preserve the
        # user's zoom/pan. "First render" (no section drawn yet) and
        # explicit reset_view both fall back to the ROI bbox; any
        # other re-render keeps the current axes window so a panned/
        # zoomed view survives an index change.
        keep_view = (not reset_view) and (self._section is not None)
        prev_xlim = self._axes.get_xlim() if keep_view else None
        prev_ylim = self._axes.get_ylim() if keep_view else None
        self._axes.clear()

        section = self._extract()
        self._section = section

        if keep_view:
            self._axes.set_xlim(prev_xlim)
            self._axes.set_ylim(prev_ylim)
        else:
            xmin, xmax, ymin, ymax = self._fixed_data_window(section)
            self._axes.set_xlim(xmin, xmax)
            self._axes.set_ylim(ymin, ymax)
        # aspect='auto' lets matplotlib stretch the data to the axes
        # rect — that's the Petrel-style z-exaggeration users expect.
        self._axes.set_aspect("auto")

        if section.n_cells == 0:
            self._axes.text(0.5, 0.5, "该剖面无活动单元",
                            transform=self._axes.transAxes,
                            ha="center", va="center")
        else:
            face_colors = self._face_colors(section)
            coll = PolyCollection(
                section.quads,
                facecolors=face_colors,
                edgecolors=(0.2, 0.2, 0.2, 0.4),
                linewidths=0.2,
                antialiased=False,
            )
            self._axes.add_collection(coll)
            self._collection = coll

        hlabel, vlabel = _AXIS_AXIS_NAMES[self._axis]
        self._axes.set_xlabel(hlabel)
        self._axes.set_ylabel(vlabel)
        self._axes.set_title(
            f"{_AXIS_LABELS[self._axis]} = {self._index} "
            f"({section.n_cells:,} 活动单元)"
        )

        self._canvas.draw_idle()
        self.section_updated.emit(self.section_id, self._axis, self._index)

    def _extract(self) -> CellSection:
        if self._axis == "i":
            section = extract_i_section(self._grid, self._index,
                                        transform=self._transform)
        else:
            section = extract_j_section(self._grid, self._index,
                                        transform=self._transform)
        return clip_to_roi(section, self._roi)

    def _face_colors(self, section: CellSection) -> np.ndarray:
        if (
            self._property_name
            and self._grid.has_property(self._property_name)
        ):
            arr = self._grid.property(self._property_name)
            vals = values_for_section(section, arr)
            if np.issubdtype(arr.dtype, np.integer):
                # Categorical: prefer a project-specific palette if we
                # have one (LITHOLOGIES); fall back to tab10 otherwise.
                cmap = palette_for(self._property_name) or colormaps["tab10"]
                ncolors = cmap.N
                idx = np.clip(vals.astype(np.int32), 0, ncolors - 1)
                return cmap(idx)
            # Continuous: viridis over data range
            cmap = colormaps["viridis"]
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            norm = Normalize(vmin=vmin, vmax=vmax)
            return cmap(norm(vals))

        # No property → flat grey-ish
        out = np.empty((section.n_cells, 4), dtype=np.float32)
        out[:] = (0.7, 0.7, 0.75, 0.9)
        return out

    # ------------------------------------------------------------------ SAM3 hooks

    def axes_pixel_rect(self) -> tuple[int, int, int, int]:
        """Return the axes' pixel rect ``(x0, y0, x1, y1)`` in canvas coords.

        Origin is top-left of the canvas, matching what SAM3 / PIL
        expect when slicing an image array. Both x and y are pixel
        indices into ``to_image_array()``.
        """

        w_in, h_in = _FIGURE_SIZE_BY_AXIS[self._axis]
        canvas_w = int(w_in * _FIGURE_DPI)
        canvas_h = int(h_in * _FIGURE_DPI)
        left, bottom, width, height = _AXES_RECT
        x0 = int(round(left * canvas_w))
        y1 = int(round((1.0 - bottom) * canvas_h))
        x1 = int(round((left + width) * canvas_w))
        y0 = int(round((1.0 - (bottom + height)) * canvas_h))
        return x0, y0, x1, y1

    def data_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Convert a data-coordinate point to canvas pixel coords.

        Reflects the *current* section's tight data window. The window
        is recomputed on every render, so call this only on the
        section you're actually viewing — Phase 11's SAM3 offscreen
        renderer will provide its own stable mapping.
        """

        if self._section is None:
            return 0.0, 0.0
        xmin, xmax, ymin, ymax = self._fixed_data_window(self._section)
        x0, y0, x1, y1 = self.axes_pixel_rect()
        fx = (x - xmin) / (xmax - xmin) if xmax > xmin else 0.0
        fy = (y - ymin) / (ymax - ymin) if ymax > ymin else 0.0
        px = x0 + fx * (x1 - x0)
        py = y1 - fy * (y1 - y0)
        return px, py

    def pixel_to_data(self, px: float, py: float) -> tuple[float, float]:
        """Inverse of ``data_to_pixel``."""

        if self._section is None:
            return 0.0, 0.0
        xmin, xmax, ymin, ymax = self._fixed_data_window(self._section)
        x0, y0, x1, y1 = self.axes_pixel_rect()
        if x1 == x0 or y1 == y0:
            return (xmin, ymin)
        fx = (px - x0) / (x1 - x0)
        fy = (y1 - py) / (y1 - y0)
        return (xmin + fx * (xmax - xmin),
                ymin + fy * (ymax - ymin))

    def to_image_array(self) -> np.ndarray:
        """Render the current canvas to ``(H, W, 3) uint8`` for SAM3.

        Shape is deterministic — ``H = figsize[1] * dpi`` and
        ``W = figsize[0] * dpi`` regardless of window size.
        """

        self._canvas.draw()
        buf = np.asarray(self._canvas.buffer_rgba())
        # buffer_rgba is (H, W, 4) uint8; drop alpha for SAM3.
        return np.ascontiguousarray(buf[..., :3])

    # ------------------------------------------------------------------ helpers

    def _fixed_data_window(
        self, _section: CellSection
    ) -> tuple[float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax)`` data extent.

        Derived from the ROI's pillar envelope (XY) plus the ROI's
        z extent — NOT from the current section's data. This is
        deliberate: every i/j/k inside the same ROI shares the same
        axes window, so SAM3 video propagation, click-to-cell lookup,
        and visual comparison across slices all stay pixel-stable.

        Trade-off: a particular i may not fill the whole window (its
        active cells may be only part of the ROI footprint). That's
        the SAM3-safe choice — users tighten the ROI in the 3D view
        to focus on a smaller target.
        """

        x0, x1, y0, y1 = roi_xy_bounds(self._grid, self._roi)
        z_min, z_max = roi_z_bounds(self._grid, self._roi)
        # Vertical axis on I/J sections is -sample (deeper is more
        # negative); convert depth metres → sample index.
        v_lo = -z_max / self._transform.z_step
        v_hi = -z_min / self._transform.z_step

        # 5 % padding so cells don't touch the axes border.
        pad = 0.05
        if self._axis == "i":
            dy_h = (y1 - y0) * pad
            dy_v = (v_hi - v_lo) * pad
            return (y0 - dy_h, y1 + dy_h, v_lo - dy_v, v_hi + dy_v)
        # j section
        dx = (x1 - x0) * pad
        dy_v = (v_hi - v_lo) * pad
        return (x0 - dx, x1 + dx, v_lo - dy_v, v_hi + dy_v)

    def _axis_range(self, axis: str | None = None) -> tuple[int, int]:
        """ROI-clamped half-open index range ``(lo, hi)`` for an axis."""

        ax = axis or self._axis
        dec = decompose_roi(self._roi, ax)
        return dec.moving_range

    def _axis_max(self) -> int:
        lo, hi = self._axis_range()
        return max(lo, hi - 1)

    def _axis_min(self) -> int:
        lo, _hi = self._axis_range()
        return lo

    def _default_index(self, axis: str) -> int:
        lo, hi = self._axis_range(axis)
        return (lo + hi) // 2

    def _default_property(self) -> str | None:
        for cand in ("LITHOLOGIES", "PORO"):
            if self._grid.has_property(cand):
                return cand
        names = self._grid.property_names()
        return names[0] if names else None

    def _cell_at(self, x: float, y: float) -> tuple[int, int, int] | None:
        """Reverse-lookup: which cell does point (x, y) fall inside?

        Brute-force over quads. For ~150k quads on an I-section this is
        ~10 ms — fine for click handling. A spatial index would let us
        do tens of thousands of these per second; not needed yet.
        """
        if self._section is None:
            return None
        q = self._section.quads    # (N, 4, 2)
        # Quick bbox filter
        xmin = q[..., 0].min(axis=1)
        xmax = q[..., 0].max(axis=1)
        ymin = q[..., 1].min(axis=1)
        ymax = q[..., 1].max(axis=1)
        cand = np.where(
            (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        )[0]
        for idx in cand:
            if _point_in_quad(x, y, q[idx]):
                return tuple(int(v) for v in self._section.cell_ids[idx])    # type: ignore[return-value]
        return None


def _point_in_quad(x: float, y: float, quad: np.ndarray) -> bool:
    """Ray-cast point-in-polygon for a 4-vertex CCW quad."""
    inside = False
    n = quad.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = quad[i]
        xj, yj = quad[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside
