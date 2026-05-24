from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from pyvistaqt import QtInteractor


class QtVTKView(QtInteractor):
    """Base PyVista/VTK widget used by 3D views."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setMinimumSize(640, 480)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.store_click_position()
        if self._forward_tool_event("on_mouse_press", event):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self.store_mouse_position()
        if self._forward_tool_event("on_mouse_move", event):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.store_click_position()
        if self._forward_tool_event("on_mouse_release", event):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.store_click_position()
        handled_press = self._forward_tool_event("on_mouse_press", event)
        handled_double = self._forward_tool_event("on_mouse_double_click", event)
        if handled_press or handled_double:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._forward_tool_event("on_key_press", event):
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if self._forward_tool_event("on_key_release", event):
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _forward_tool_event(self, method_name: str, event) -> bool:
        manager = getattr(self, "tool_manager", None)
        if manager is None:
            return False
        return bool(manager.forward(method_name, self, event))
