from __future__ import annotations

from yj_studio.tools._helpers import (
    event_left_button,
    ensure_mask_layer,
    paint_disk,
    slice_context_from_event,
    tool_layer_store,
    tool_notify,
)
from yj_studio.tools.tool import InteractionTool


class _MaskEditTool(InteractionTool):
    layer_key = "paint_mask"
    paint_value = 1
    mask_color = (1.0, 0.2, 0.2, 0.35)

    def __init__(self, *, tool_id: str, label: str, icon: str) -> None:
        super().__init__(id=tool_id, label=label, icon=icon, cursor="crosshair")
        self._painting = False
        self.radius = 5

    def activate(self, view) -> None:
        self._painting = False

    def deactivate(self, view) -> None:
        self._painting = False

    def on_mouse_press(self, view, event) -> bool:
        if not event_left_button(event):
            return False
        if not self._paint(view, event, prefer_mouse=False):
            return False
        self._painting = True
        return True

    def on_mouse_move(self, view, event) -> bool:
        if not self._painting:
            return False
        self._paint(view, event, prefer_mouse=True)
        return True

    def on_mouse_release(self, view, event) -> bool:
        if not self._painting:
            return False
        self._paint(view, event, prefer_mouse=True)
        self._painting = False
        return True

    def on_key_press(self, view, event) -> bool:
        key = getattr(event, "key", "")
        if key in {"escape", "esc"}:
            self._painting = False
            tool_notify(view, "Painting cancelled")
            return True
        return False

    def _paint(self, view, event, *, prefer_mouse: bool) -> bool:
        layer_store = tool_layer_store(view)
        if layer_store is None:
            return False
        context = slice_context_from_event(view, event, prefer_mouse=prefer_mouse)
        if context is None:
            tool_notify(view, "Load a volume before painting")
            return False
        mask_layer = ensure_mask_layer(
            layer_store,
            tool_name=self.layer_key,
            axis=context.axis,
            slice_index=context.slice_index,
            shape=context.shape,
            color=self.mask_color,
            opacity=self.mask_color[3],
        )
        if mask_layer.mask is None:
            return False
        paint_disk(mask_layer.mask, context.row, context.col, int(self.radius), int(self.paint_value))
        layer_store.update(mask_layer.id, mask=mask_layer.mask)
        return True


class BrushTool(_MaskEditTool):
    def __init__(self) -> None:
        super().__init__(tool_id="brush", label="Brush", icon="brush")
