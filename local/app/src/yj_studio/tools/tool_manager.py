from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import Qt

from .tool import InteractionTool


class ToolManager(QObject):
    active_tool_changed = pyqtSignal(str)
    message_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._tools: dict[str, InteractionTool] = {}
        self._active_tool_id: str | None = None
        self._views: list[Any] = []
        self._services: dict[str, Any] = {}

    def register_service(self, name: str, service: Any) -> None:
        """Expose a long-lived backend service to interactive tools.
        Tools can pull it via ``view.tool_manager.service('ai_service')``.
        """

        self._services[name] = service

    def service(self, name: str) -> Any | None:
        return self._services.get(name)

    def register(self, tool: InteractionTool) -> None:
        if tool.id in self._tools:
            raise ValueError(f"Tool already registered: {tool.id}")
        self._tools[tool.id] = tool
        if self._active_tool_id is None and tool.enabled:
            self._active_tool_id = tool.id

    def attach_view(self, view: Any) -> None:
        if view in self._views:
            return
        self._views.append(view)
        setattr(view, "tool_manager", self)
        active = self.active_tool
        if active is not None:
            active.activate(view)
            self._apply_cursor(view, active.cursor)

    def detach_view(self, view: Any) -> None:
        if view not in self._views:
            return
        active = self.active_tool
        if active is not None:
            active.deactivate(view)
        self._views = [item for item in self._views if item is not view]
        self._apply_cursor(view, "arrow")

    def activate(self, tool_id: str, view: Any = None) -> None:
        tool = self._tools[tool_id]
        if not tool.enabled:
            raise RuntimeError(f"工具不可用：{tool_id}")
        active = self.active_tool
        if view is not None and view not in self._views:
            self.attach_view(view)
        if active is not None and active.id != tool_id:
            for attached_view in self._views:
                active.deactivate(attached_view)
        self._active_tool_id = tool_id
        targets = [view] if view is not None else list(self._views)
        if not targets and view is not None:
            targets = [view]
        for target in targets:
            tool.activate(target)
            self._apply_cursor(target, tool.cursor)
        self.active_tool_changed.emit(tool_id)
        if not targets:
            self.message_requested.emit(f"工具已激活：{tool.label}")

    def set_active(self, tool_id: str, view: Any = None) -> None:
        self.activate(tool_id, view=view)

    @property
    def active_tool(self) -> InteractionTool | None:
        if self._active_tool_id is None:
            return None
        return self._tools[self._active_tool_id]

    def tools(self) -> tuple[InteractionTool, ...]:
        return tuple(self._tools.values())

    def get(self, tool_id: str) -> InteractionTool:
        return self._tools[tool_id]

    def forward(self, method_name: str, view: Any, event: Any) -> bool:
        tool = self.active_tool
        if tool is None:
            return False
        method = getattr(tool, method_name, None)
        if method is None:
            return False
        result = method(view, event)
        return bool(result)

    def notify(self, message: str) -> None:
        self.message_requested.emit(message)

    def _apply_cursor(self, view: Any, cursor_name: str) -> None:
        cursor = _cursor_for_name(cursor_name)
        if hasattr(view, "setCursor"):
            view.setCursor(cursor)
        canvas = getattr(view, "_canvas", None)
        if canvas is not None and hasattr(canvas, "setCursor"):
            canvas.setCursor(cursor)


def _cursor_for_name(cursor_name: str) -> QCursor:
    mapping = {
        "arrow": Qt.CursorShape.ArrowCursor,
        "crosshair": Qt.CursorShape.CrossCursor,
        "pointinghand": Qt.CursorShape.PointingHandCursor,
        "size_all": Qt.CursorShape.SizeAllCursor,
        "ibeam": Qt.CursorShape.IBeamCursor,
        "wait": Qt.CursorShape.WaitCursor,
        "closedhand": Qt.CursorShape.ClosedHandCursor,
    }
    return QCursor(mapping.get(cursor_name, Qt.CursorShape.ArrowCursor))
