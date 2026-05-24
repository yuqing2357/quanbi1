from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from yj_studio.tools import ToolManager


class ToolPaletteDock(QDockWidget):
    """Vertical list of interaction tools."""

    def __init__(self, tool_manager: ToolManager, parent: QWidget | None = None) -> None:
        super().__init__("Tools", parent)
        self._tool_manager = tool_manager
        self._buttons: dict[str, QToolButton] = {}

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for tool in tool_manager.tools():
            button = QToolButton(content)
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setText(tool.label)
            button.setToolTip(tool.label if tool.enabled else f"{tool.label} (Phase 2)")
            button.setIcon(_tool_icon(self, tool.icon))
            button.setEnabled(tool.enabled)
            button.clicked.connect(lambda checked=False, tool_id=tool.id: self._tool_manager.set_active(tool_id))
            layout.addWidget(button)
            self._group.addButton(button)
            self._buttons[tool.id] = button

        layout.addStretch(1)
        self.setWidget(content)

        tool_manager.active_tool_changed.connect(self._sync_active)
        self._sync_active(tool_manager.active_tool.id if tool_manager.active_tool is not None else "")

    def _sync_active(self, tool_id: str) -> None:
        for current_id, button in self._buttons.items():
            button.setChecked(current_id == tool_id)


def _tool_icon(widget: QWidget, icon_name: str):
    style = widget.style()
    mapping = {
        "navigation": QStyle.StandardPixmap.SP_BrowserReload,
        "crosshair": QStyle.StandardPixmap.SP_DialogYesButton,
        "square": QStyle.StandardPixmap.SP_FileDialogDetailedView,
        "polygon": QStyle.StandardPixmap.SP_FileIcon,
        "brush": QStyle.StandardPixmap.SP_ArrowUp,
        "eraser": QStyle.StandardPixmap.SP_TrashIcon,
        "pin": QStyle.StandardPixmap.SP_ArrowRight,
        "pen": QStyle.StandardPixmap.SP_ArrowLeft,
        "ruler": QStyle.StandardPixmap.SP_DialogApplyButton,
        "ban": QStyle.StandardPixmap.SP_DialogCancelButton,
    }
    standard = mapping.get(icon_name)
    if standard is None:
        return style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
    return style.standardIcon(standard)
