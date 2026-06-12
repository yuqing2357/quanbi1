from __future__ import annotations

from yj_studio.tools.brush_tool import _MaskEditTool


class EraserTool(_MaskEditTool):
    paint_value = 0
    mask_color = (0.2, 0.6, 1.0, 0.25)

    def __init__(self) -> None:
        super().__init__(tool_id="eraser", label="橡皮", icon="eraser")
