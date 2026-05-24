from __future__ import annotations

import numpy as np

from yj_studio.scene.layers import MeasurementLayer
from yj_studio.tools._geometry import polygon_area
from yj_studio.tools._helpers import event_left_button, event_world_point, next_layer_name, tool_layer_store, tool_notify
from yj_studio.tools.tool import InteractionTool


class MeasureTool(InteractionTool):
    def __init__(self) -> None:
        super().__init__(id="measure", label="Measure", icon="ruler", cursor="crosshair")
        self._points: list[tuple[float, float, float]] = []

    def activate(self, view) -> None:
        self._points = []

    def deactivate(self, view) -> None:
        self._points = []

    def on_mouse_press(self, view, event) -> bool:
        if not event_left_button(event):
            return False
        point = event_world_point(view, event, prefer_mouse=False)
        if point is None:
            return False
        self._points.append(point)
        tool_notify(view, f"Measure points: {len(self._points)}")
        return True

    def on_mouse_double_click(self, view, event) -> bool:
        layer_store = tool_layer_store(view)
        if layer_store is None or len(self._points) < 2:
            self._points = []
            return False
        geometry = np.asarray(self._points, dtype=np.float32)
        values, units = self._measure_values(view, geometry)
        layer = MeasurementLayer(
            name=next_layer_name(layer_store, "Measurement"),
            geometry=geometry,
            values=values,
            units=units,
            color=(0.95, 0.95, 0.2, 1.0),
            opacity=1.0,
            visible=True,
            metadata={"tool": "measure"},
            provenance={"source": "manual"},
        )
        layer_store.add(layer)
        layer_store.select([layer.id])
        tool_notify(view, self._measure_message(values))
        self._points = []
        return True

    def on_key_press(self, view, event) -> bool:
        key = getattr(event, "key", "")
        if key in {"escape", "esc"}:
            self._points = []
            tool_notify(view, "Measurement cleared")
            return True
        return False

    def _measure_values(self, view, geometry: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
        if geometry.shape[0] == 2:
            distance = float(np.linalg.norm(geometry[1] - geometry[0]))
            thickness = float(abs(float(geometry[1, 2]) - float(geometry[0, 2])))
            return (
                {"distance": distance, "thickness": thickness},
                {"distance": "grid", "thickness": "grid"},
            )
        axis = getattr(view, "axis", None)
        if axis == "inline":
            poly = geometry[:, [1, 2]]
        elif axis == "xline":
            poly = geometry[:, [0, 2]]
        else:
            poly = geometry[:, :2]
        area = polygon_area(poly)
        return ({"area": area}, {"area": "grid^2"})

    def _measure_message(self, values: dict[str, float]) -> str:
        if "area" in values:
            return f"Area: {values['area']:.3f}"
        if "distance" in values:
            return f"Distance: {values['distance']:.3f}"
        return "Measurement completed"
