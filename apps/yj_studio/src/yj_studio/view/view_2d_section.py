from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from yj_studio.config.styles import LITH_COLORS
from yj_studio.data.volume_store import VolumeStore
from yj_studio.io.readers.fault_mesh import load_fault_mesh
from yj_studio.io.readers.layers_npz import load_layer_grid
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import (
    ArbitrarySectionLayer,
    FaultStickLayer,
    FaultSurfaceLayer,
    HorizonLayer,
    HorizonStickLayer,
    MaskLayer,
    MeasurementLayer,
    PolygonLayer,
    VolumeLayer,
    WellLayer,
    WellLogLayer,
)
from yj_studio.scene.manual_geometry import is_manual_geometry_layer, manual_geometry_points, project_points_to_section
from yj_studio.services.section_service import (
    SectionAxis,
    extract_orthogonal_section,
    fault_points_intersection,
    horizon_intersection,
    well_intersection,
)
from yj_studio.services.view_sync_service import ViewSyncService
from yj_studio.ui.text import measurement_value_label, section_axis_label
from yj_studio.view.highlight import highlight_color, is_layer_highlighted, selected_well_names


class View2DSection(QWidget):
    """Matplotlib-backed orthogonal section view."""

    section_updated = pyqtSignal(str, str, int)

    def __init__(
        self,
        layer_store: LayerStore,
        volume_store: VolumeStore,
        sync_service: ViewSyncService,
        *,
        volume_layer_id: str,
        axis: SectionAxis,
        index: int,
        section_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.section_id = section_id or str(uuid4())
        self.axis = axis
        self.index = int(index)
        self._layer_store = layer_store
        self._volume_store = volume_store
        self._sync_service = sync_service
        self._volume_layer_id = volume_layer_id
        self._current_extent: tuple[float, float, float, float] | None = None

        self._figure = Figure(figsize=(5, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("button_press_event", self._on_button_press)
        self._canvas.mpl_connect("button_release_event", self._on_button_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("key_press_event", self._on_key_press)
        self._canvas.mpl_connect("key_release_event", self._on_key_release)

        layer_store.layer_added.connect(self._on_layer_added)
        layer_store.layer_changed.connect(self._on_layer_changed)
        layer_store.layer_removed.connect(self._on_layer_removed)
        layer_store.selection_changed.connect(self._on_selection_changed)
        sync_service.subscribe(f"slice.{axis}_position", self._on_slice_synced)
        self.refresh()

    @property
    def title(self) -> str:
        return f"{section_axis_label(self.axis)} {self.index}"

    def set_index(self, index: int, *, publish: bool = False) -> None:
        if int(index) == self.index:
            return
        self.index = int(index)
        self.refresh()
        self.section_updated.emit(self.section_id, self.title, self.index)
        if publish:
            self._sync_service.publish(f"slice.{self.axis}_position", self.index, self)

    def refresh(self) -> None:
        self._axes.clear()
        volume_layer = self._volume_layer()
        if volume_layer is None:
            self._draw_message("未加载体数据")
            return
        try:
            section = extract_orthogonal_section(
                self._volume_store,
                volume_layer,
                self.axis,
                self.index,
            )
        except Exception as exc:
            self._draw_message(str(exc))
            return
        self.index = section.index
        self._current_extent = section.extent
        vmin, vmax = volume_layer.clim if volume_layer.clim is not None else (None, None)
        # ``aspect="equal"`` locks one data unit on X to one data unit on Y so
        # the seismic slice keeps its shape when the user resizes the dock /
        # main window. The previous ``"auto"`` value let the image stretch to
        # fill whatever space the canvas had, which was visually noisy.
        self._axes.imshow(
            section.values,
            cmap=_matplotlib_cmap_name(volume_layer.cmap),
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="equal",
            extent=section.extent,
        )
        self._draw_overlays()
        self._axes.set_title(self.title)
        self._axes.set_xlabel(section.x_label)
        self._axes.set_ylabel(section.y_label)
        self._canvas.draw_idle()

    def _draw_overlays(self) -> None:
        selected_ids = set(self._layer_store.selection)
        selected_wells = selected_well_names(self._layer_store)
        for layer in self._layer_store.iter_layers():
            if not layer.visible:
                continue
            highlighted = is_layer_highlighted(layer, selected_ids, selected_wells)
            if isinstance(layer, HorizonLayer):
                _ensure_horizon_arrays(layer)
                line = horizon_intersection(layer, self.axis, self.index)
                if line is not None:
                    self._axes.plot(
                        line.x,
                        line.y,
                        color=highlight_color(line.color, highlighted),
                        alpha=max(0.25, line.opacity),
                        linewidth=3.0 if highlighted else 1.4,
                        zorder=6 if highlighted else 3,
                    )
            elif isinstance(layer, WellLayer):
                item = well_intersection(layer, self.axis, self.index)
                if item is not None:
                    if hasattr(item, "x") and item.x.size == 1:
                        self._axes.scatter(
                            item.x,
                            item.y,
                            color=highlight_color(item.color, highlighted),
                            alpha=item.opacity,
                            s=54 if highlighted else 24,
                            zorder=7 if highlighted else 4,
                        )
                    else:
                        self._axes.plot(
                            item.x,
                            item.y,
                            color=highlight_color(item.color, highlighted),
                            alpha=item.opacity,
                            linewidth=3.0 if highlighted else 1.2,
                            zorder=7 if highlighted else 4,
                        )
            elif isinstance(layer, FaultSurfaceLayer):
                _ensure_fault_arrays(layer)
                points = fault_points_intersection(layer, self.axis, self.index)
                if points is not None:
                    self._axes.scatter(
                        points.x,
                        points.y,
                        color=highlight_color(points.color, highlighted),
                        alpha=max(0.2, points.opacity),
                        s=8 if highlighted else 2,
                        zorder=6 if highlighted else 3,
                    )
            elif isinstance(layer, WellLogLayer):
                self._draw_well_log(layer, highlighted=highlighted)
            elif isinstance(layer, MaskLayer):
                if layer.axis == self.axis and layer.slice_index == self.index and layer.mask is not None:
                    self._axes.imshow(
                        _mask_rgba(layer.mask, highlight_color(layer.color, highlighted), float(layer.opacity)),
                        origin="upper",
                        aspect="equal",
                        extent=self._current_extent,
                        interpolation="nearest",
                        zorder=8 if highlighted else 5,
                    )
            elif is_manual_geometry_layer(layer):
                self._draw_manual_geometry(layer, highlighted=highlighted)

    def _draw_well_log(self, layer: WellLogLayer, *, highlighted: bool = False) -> None:
        if layer.samples is None:
            return
        samples = np.asarray(layer.samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 4:
            return
        axis_index = {"inline": 0, "xline": 1, "z": 2}[self.axis]
        mask = np.abs(samples[:, axis_index] - float(self.index)) <= 0.5
        if self.axis == "z":
            zmin = float(self.index) - 0.5
            zmax = float(self.index) + 0.5
            mask = (samples[:, 2] >= zmin) & (samples[:, 2] <= zmax)
        if not np.any(mask):
            return
        selected = samples[mask]
        if self.axis == "inline":
            x, y = selected[:, 1], selected[:, 2]
        elif self.axis == "xline":
            x, y = selected[:, 0], selected[:, 2]
        else:
            x, y = selected[:, 0], selected[:, 1]
        values = selected[:, 3]
        if self.axis == "z":
            self._draw_well_log_points(x, y, values, layer, highlighted=highlighted)
        elif layer.mode == "lith":
            self._draw_lith_log_column(x, y, values, layer, highlighted=highlighted)
        else:
            self._draw_numeric_log_bar(x, y, values, layer, highlighted=highlighted)

    def _draw_well_log_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        layer: WellLogLayer,
        *,
        highlighted: bool = False,
    ) -> None:
        self._axes.scatter(
            x,
            y,
            c=values,
            cmap=_matplotlib_cmap_name(str(layer.metadata.get("cmap", "viridis"))),
            s=float(layer.metadata.get("point_size", 7.0)) * (2.0 if highlighted else 1.0),
            alpha=float(layer.opacity),
            edgecolors="yellow" if highlighted else "none",
            linewidths=0.8 if highlighted else 0.0,
            zorder=7 if highlighted else 4,
        )

    def _draw_lith_log_column(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        layer: WellLogLayer,
        *,
        highlighted: bool = False,
    ) -> None:
        x_base = float(np.nanmedian(x))
        colors = [_lith_color(value) for value in values]
        self._axes.scatter(
            np.full_like(y, x_base),
            y,
            c=colors,
            marker="s",
            s=float(layer.metadata.get("point_size", 7.0)) * (4.0 if highlighted else 2.4),
            alpha=float(layer.opacity),
            edgecolors="yellow" if highlighted else "none",
            linewidths=0.8 if highlighted else 0.0,
            zorder=8 if highlighted else 5,
        )

    def _draw_numeric_log_bar(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        layer: WellLogLayer,
        *,
        highlighted: bool = False,
    ) -> None:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        if not np.any(finite):
            return
        x = x[finite]
        y = y[finite]
        values = values[finite]
        x_base = float(np.nanmedian(x))
        span = _horizontal_span(self._current_extent)
        bar_width = max(span * 0.006, 0.8)
        bar_left = x_base + max(span * 0.006, 0.8)
        height = _sample_height(y)
        colors = _numeric_colors(values, layer)
        self._axes.barh(
            y,
            np.full(y.shape, bar_width, dtype=np.float32),
            height=height,
            left=np.full(y.shape, bar_left, dtype=np.float32),
            color=colors,
            edgecolor="yellow" if highlighted else "none",
            linewidth=0.8 if highlighted else 0.0,
            alpha=float(layer.opacity),
            zorder=8 if highlighted else 5,
        )

    def _draw_manual_geometry(self, layer, *, highlighted: bool = False) -> None:
        points = manual_geometry_points(layer)
        projected = project_points_to_section(points, self.axis, self.index)
        if projected is None:
            return
        x, y = projected
        if x.size == 0:
            return
        color = highlight_color(layer.color, highlighted)
        if isinstance(layer, PolygonLayer) and layer.closed and x.size >= 3:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        if x.size >= 2:
            self._axes.plot(
                x,
                y,
                color=color,
                alpha=max(0.3, float(layer.opacity)),
                linewidth=2.8 if highlighted else 1.6,
                zorder=9 if highlighted else 6,
            )
        self._axes.scatter(
            x,
            y,
            color=color,
            alpha=max(0.4, float(layer.opacity)),
            s=40 if highlighted else 18,
            zorder=10 if highlighted else 7,
        )
        if isinstance(layer, MeasurementLayer) and layer.values:
            label = _measurement_label(layer)
            if label:
                mid = len(x) // 2
                self._axes.text(
                    float(x[mid]),
                    float(y[mid]),
                    label,
                    color=color,
                    fontsize=9 if highlighted else 8,
                    ha="left",
                    va="bottom",
                    zorder=11,
                )

    def _draw_message(self, message: str) -> None:
        self._axes.clear()
        self._current_extent = None
        self._axes.text(0.5, 0.5, message, ha="center", va="center", transform=self._axes.transAxes)
        self._axes.set_axis_off()
        self._canvas.draw_idle()

    def _on_layer_changed(self, layer_id: str, _field: str) -> None:
        if layer_id == self._volume_layer_id:
            self.refresh()
            return
        layer = self._layer_store.get(layer_id)
        if isinstance(layer, (HorizonLayer, FaultSurfaceLayer, ArbitrarySectionLayer, PolygonLayer, HorizonStickLayer, FaultStickLayer, MeasurementLayer, MaskLayer, WellLayer, WellLogLayer)):
            self.refresh()

    def _on_layer_added(self, layer_id: str) -> None:
        try:
            layer = self._layer_store.get(layer_id)
        except KeyError:
            return
        if isinstance(layer, (HorizonLayer, FaultSurfaceLayer, ArbitrarySectionLayer, PolygonLayer, HorizonStickLayer, FaultStickLayer, MeasurementLayer, MaskLayer, WellLayer, WellLogLayer)):
            if layer.visible:
                self.refresh()

    def _on_layer_removed(self, _layer_id: str) -> None:
        self.refresh()

    def _on_selection_changed(self, _layer_ids: list[str]) -> None:
        self.refresh()

    def _on_button_press(self, event) -> None:
        if event.dblclick:
            self._forward_tool_event("on_mouse_press", event)
            self._forward_tool_event("on_mouse_double_click", event)
            return
        self._forward_tool_event("on_mouse_press", event)

    def _on_button_release(self, event) -> None:
        self._forward_tool_event("on_mouse_release", event)

    def _on_mouse_move(self, event) -> None:
        self._forward_tool_event("on_mouse_move", event)

    def _on_key_press(self, event) -> None:
        self._forward_tool_event("on_key_press", event)

    def _on_key_release(self, event) -> None:
        self._forward_tool_event("on_key_release", event)

    def _forward_tool_event(self, method_name: str, event) -> bool:
        manager = getattr(self, "tool_manager", None)
        if manager is None:
            return False
        return bool(manager.forward(method_name, self, event))

    def _on_slice_synced(self, _topic: str, value: object, origin: object | None) -> None:
        if origin is self:
            return
        self.set_index(int(value))

    def _volume_layer(self) -> VolumeLayer | None:
        layer = self._layer_store.get(self._volume_layer_id)
        if isinstance(layer, VolumeLayer):
            return layer
        return None


def _ensure_horizon_arrays(layer: HorizonLayer) -> None:
    if layer.sample is not None:
        return
    if layer.data_path is None:
        return
    grid = load_layer_grid(Path(layer.data_path))
    layer.sample = grid.sample
    layer.mask = grid.mask
    layer.metadata.update(grid.metadata)


def _ensure_fault_arrays(layer: FaultSurfaceLayer) -> None:
    if layer.vertices is not None:
        return
    if layer.data_path is None:
        return
    mesh = load_fault_mesh(Path(layer.data_path))
    layer.vertices = mesh.vertices
    layer.faces = mesh.faces
    layer.metadata.update(mesh.metadata)


def _matplotlib_cmap_name(cmap: str) -> str:
    return {"Petrel": "seismic", "petrel": "seismic"}.get(cmap, cmap)


def _lith_color(value: float) -> tuple[float, float, float, float]:
    try:
        lith_class = int(round(float(value)))
    except (TypeError, ValueError):
        lith_class = -1
    return to_rgba(LITH_COLORS.get(lith_class, "#d62728"))


def _numeric_colors(values: np.ndarray, layer: WellLogLayer) -> np.ndarray:
    clim = layer.metadata.get("clim")
    if isinstance(clim, list | tuple) and len(clim) >= 2:
        vmin = float(clim[0])
        vmax = float(clim[1])
    else:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = (float(v) for v in np.nanpercentile(finite, [2.0, 98.0]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap_name = _matplotlib_cmap_name(str(layer.metadata.get("cmap", "viridis")))
    return colormaps[cmap_name](normalized)


def _horizontal_span(extent: tuple[float, float, float, float] | None) -> float:
    if extent is None:
        return 100.0
    return max(abs(float(extent[1]) - float(extent[0])), 1.0)


def _sample_height(y: np.ndarray) -> float:
    unique = np.unique(np.asarray(y[np.isfinite(y)], dtype=np.float32))
    if unique.size < 2:
        return 1.0
    spacing = float(np.nanmedian(np.diff(np.sort(unique))))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return 1.0
    return max(spacing * 0.9, 0.4)


def _mask_rgba(mask: np.ndarray, color: tuple[float, float, float, float], opacity: float) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    rgba = np.zeros(values.shape + (4,), dtype=np.float32)
    rgba[..., :3] = np.asarray(color[:3], dtype=np.float32)
    rgba[..., 3] = np.clip(values, 0.0, 1.0) * float(opacity)
    return np.transpose(rgba, (1, 0, 2))


def _measurement_label(layer: MeasurementLayer) -> str:
    if "area" in layer.values:
        unit = layer.units.get("area", "")
        return f"{measurement_value_label('area')} {layer.values['area']:.2f}{_unit_suffix(unit)}"
    if "distance" in layer.values:
        unit = layer.units.get("distance", "")
        return f"{measurement_value_label('distance')} {layer.values['distance']:.2f}{_unit_suffix(unit)}"
    if "thickness" in layer.values:
        unit = layer.units.get("thickness", "")
        return f"{measurement_value_label('thickness')} {layer.values['thickness']:.2f}{_unit_suffix(unit)}"
    return ""


def _unit_suffix(unit: str) -> str:
    return f" {unit}" if unit else ""
