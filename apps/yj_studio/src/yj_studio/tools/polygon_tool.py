from __future__ import annotations

import numpy as np

from yj_studio.scene.layers import PolygonLayer
from yj_studio.tools._helpers import event_left_button, event_world_point, next_layer_name, tool_layer_store, tool_notify
from yj_studio.tools.tool import InteractionTool


class PolygonTool(InteractionTool):
    def __init__(self) -> None:
        super().__init__(id="polygon", label="Polygon", icon="polygon", cursor="crosshair")
        self._points: list[tuple[float, float, float]] = []

    def activate(self, view) -> None:
        self._points = []

    def deactivate(self, view) -> None:
        self._points = []

    def on_mouse_press(self, view, event) -> bool:
        if not event_left_button(event):
            return False
        point = event_world_point(view, event)
        if point is None:
            return False
        self._points.append(point)
        tool_notify(view, f"Polygon points: {len(self._points)}")
        return True

    def on_mouse_double_click(self, view, event) -> bool:
        layer_store = tool_layer_store(view)
        if layer_store is None:
            self._points = []
            return False
        if len(self._points) < 3:
            tool_notify(view, "Polygon needs at least 3 points")
            self._points = []
            return True
        vertices = np.asarray(self._points, dtype=np.float32)
        layer = PolygonLayer(
            name=next_layer_name(layer_store, "Polygon"),
            vertices=vertices,
            closed=True,
            color=(1.0, 0.75, 0.15, 0.85),
            opacity=0.85,
            visible=True,
            metadata={"tool": "polygon"},
            provenance={"source": "manual"},
        )
        layer_store.add(layer)
        layer_store.select([layer.id])
        tool_notify(view, f"Created {layer.name}")
        self._points = []
        return True

    def on_key_press(self, view, event) -> bool:
        key = getattr(event, "key", "")
        if key in {"escape", "esc"}:
            self._points = []
            tool_notify(view, "Polygon cleared")
            return True
        return False
