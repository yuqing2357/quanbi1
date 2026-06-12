from __future__ import annotations

import numpy as np

from yj_studio.scene.layers import HorizonStickLayer
from yj_studio.tools._helpers import event_left_button, event_world_point, next_layer_name, tool_layer_store, tool_notify
from yj_studio.tools.tool import InteractionTool


class _BaseStickTool(InteractionTool):
    layer_cls = HorizonStickLayer
    points_field = "points"
    layer_key = "stick"
    layer_title = "层位杆"
    layer_color = (0.95, 0.7, 0.1, 0.95)

    def __init__(self, *, tool_id: str, label: str, icon: str) -> None:
        super().__init__(id=tool_id, label=label, icon=icon, cursor="crosshair")
        self._layer_id: str | None = None

    def activate(self, view) -> None:
        self._layer_id = None

    def deactivate(self, view) -> None:
        self._layer_id = None

    def on_mouse_press(self, view, event) -> bool:
        if not event_left_button(event):
            return False
        point = event_world_point(view, event)
        if point is None:
            return False
        self._append_point(view, point)
        return True

    def on_mouse_double_click(self, view, event) -> bool:
        if self._layer_id is None:
            return False
        tool_notify(view, f"{self.label} 已完成")
        self._layer_id = None
        return True

    def on_key_press(self, view, event) -> bool:
        key = getattr(event, "key", "")
        if key in {"escape", "esc"}:
            self._layer_id = None
            tool_notify(view, f"{self.label} 已清除")
            return True
        return False

    def _append_point(self, view, point: tuple[float, float, float]) -> None:
        layer_store = tool_layer_store(view)
        if layer_store is None:
            return
        layer = self._layer(layer_store)
        points = getattr(layer, self.points_field, None)
        if points is None or np.asarray(points).size == 0:
            new_points = np.asarray([point], dtype=np.float32)
        else:
            new_points = np.vstack([np.asarray(points, dtype=np.float32), np.asarray(point, dtype=np.float32)])
        layer_store.update(layer.id, **{self.points_field: new_points})
        layer_store.select([layer.id])
        tool_notify(view, f"{self.label} 点数：{len(new_points)}")

    def _layer(self, layer_store):
        if self._layer_id is not None:
            try:
                return layer_store.get(self._layer_id)
            except KeyError:
                self._layer_id = None
        layer = self.layer_cls(
            name=next_layer_name(layer_store, self.layer_title),
            color=self.layer_color,
            opacity=self.layer_color[3],
            visible=True,
            metadata={"tool": self.layer_key},
            provenance={"source": "manual"},
        )
        layer_store.add(layer)
        self._layer_id = layer.id
        return layer


class HorizonStickTool(_BaseStickTool):
    def __init__(self) -> None:
        super().__init__(tool_id="horizon_stick", label="层位杆", icon="pin")
