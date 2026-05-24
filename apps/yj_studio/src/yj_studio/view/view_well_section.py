from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from yj_studio.config.styles import LITH_COLORS
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import WellLayer, WellLogLayer
from yj_studio.services import WellSectionData
from yj_studio.view.highlight import highlight_color, is_layer_highlighted, selected_well_names


class ViewWellSection(QWidget):
    """Matplotlib-backed connected-well section inside the desktop application."""

    def __init__(
        self,
        data: WellSectionData,
        layer_store: LayerStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self._layer_store = layer_store
        self._figure = Figure(figsize=(7, 4), tight_layout=True)
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
        layer_store.layer_changed.connect(self._on_layer_changed)
        layer_store.selection_changed.connect(self._on_selection_changed)
        self.refresh()

    @property
    def title(self) -> str:
        return f"Well {'-'.join(self.data.names[:3])}"

    def refresh(self) -> None:
        self._axes.clear()
        self._draw_seismic()
        self._draw_wells()
        self._apply_section_limits()
        self._axes.set_title(" -> ".join(self.data.names))
        self._axes.set_xlabel("Distance")
        self._axes.set_ylabel("Depth (m)")
        self._canvas.draw_idle()

    def _draw_seismic(self) -> None:
        if self.data.seismic.size == 0 or self.data.distances.size == 0:
            return
        finite = self.data.seismic[np.isfinite(self.data.seismic)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [2.0, 98.0])
        else:
            vmin, vmax = -1.0, 1.0
        extent = (
            float(np.nanmin(self.data.distances)),
            float(np.nanmax(self.data.distances)),
            float(np.nanmax(self.data.depths_m)),
            float(np.nanmin(self.data.depths_m)),
        )
        self._axes.imshow(
            self.data.seismic,
            cmap="seismic",
            vmin=float(vmin),
            vmax=float(vmax),
            origin="upper",
            aspect="auto",
            extent=extent,
        )

    def _draw_wells(self) -> None:
        total = max(float(np.nanmax(self.data.distances)) if self.data.distances.size else 1.0, 1.0)
        bar_width = max(total * 0.002, 0.2)
        bar_left_offset = max(total * 0.004, 0.35)
        selected_ids = set(self._layer_store.selection)
        selected_wells = selected_well_names(self._layer_store)
        for well in self.data.wells:
            layer = self._well_layer(well.layer_id)
            highlighted = layer is not None and is_layer_highlighted(layer, selected_ids, selected_wells)
            self._axes.axvline(
                well.distance,
                color=highlight_color((0.0, 0.0, 0.0, 1.0), highlighted),
                linewidth=2.4 if highlighted else 1.0,
                alpha=0.9 if highlighted else 0.65,
                zorder=9 if highlighted else 6,
            )
            self._axes.text(
                well.distance,
                0.0,
                well.name,
                ha="center",
                va="bottom",
                fontsize=11 if highlighted else 9,
                color=highlight_color((0.0, 0.0, 0.0, 1.0), highlighted),
                clip_on=False,
                zorder=10,
            )
            for log_layer in well.logs:
                if not log_layer.visible:
                    continue
                log_highlighted = is_layer_highlighted(log_layer, selected_ids, selected_wells)
                samples = log_layer.samples
                if samples is None or samples.size == 0:
                    continue
                values = samples[:, 3]
                depths_m = samples[:, 2] * 10.0
                if log_layer.mode == "lith":
                    self._axes.scatter(
                        np.full_like(depths_m, well.distance),
                        depths_m,
                        c=[_lith_color(value) for value in values],
                        marker="s",
                        s=24 if log_highlighted else 12,
                        alpha=float(log_layer.opacity),
                        edgecolors="yellow" if log_highlighted else "none",
                        linewidths=0.8 if log_highlighted else 0.0,
                        zorder=10 if log_highlighted else 7,
                    )
                else:
                    self._axes.barh(
                        depths_m,
                        np.full(depths_m.shape, bar_width, dtype=np.float32),
                        height=_sample_height(depths_m),
                        left=np.full(
                            depths_m.shape,
                            well.distance + bar_left_offset,
                            dtype=np.float32,
                        ),
                        color=_numeric_colors(values, log_layer),
                        edgecolor="yellow" if log_highlighted else "none",
                        linewidth=0.8 if log_highlighted else 0.0,
                        alpha=max(float(log_layer.opacity), 0.55),
                        zorder=10 if log_highlighted else 7,
                    )

    def _apply_section_limits(self) -> None:
        distances = [float(well.distance) for well in self.data.wells]
        if self.data.distances.size:
            distances.extend(float(value) for value in self.data.distances[np.isfinite(self.data.distances)])
        if distances:
            xmin = min(distances)
            xmax = max(distances)
            span = max(xmax - xmin, 1.0)
            padding = max(span * 0.05, 1.0)
            self._axes.set_xlim(xmin - padding, xmax + padding)
        if self.data.depths_m.size:
            finite_depths = self.data.depths_m[np.isfinite(self.data.depths_m)]
            if finite_depths.size:
                self._axes.set_ylim(float(np.nanmax(finite_depths)), float(np.nanmin(finite_depths)))

    def _well_layer(self, layer_id: str) -> WellLayer | None:
        try:
            layer = self._layer_store.get(layer_id)
        except KeyError:
            return None
        if isinstance(layer, WellLayer):
            return layer
        return None

    def _on_layer_changed(self, layer_id: str, _field: str) -> None:
        try:
            layer = self._layer_store.get(layer_id)
        except KeyError:
            return
        if isinstance(layer, (WellLayer, WellLogLayer)):
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
    return colormaps[str(layer.metadata.get("cmap", "viridis"))](normalized)


def _sample_height(y: np.ndarray) -> float:
    unique = np.unique(np.asarray(y[np.isfinite(y)], dtype=np.float32))
    if unique.size < 2:
        return 1.0
    spacing = float(np.nanmedian(np.diff(np.sort(unique))))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return 1.0
    return max(spacing * 0.9, 0.4)
