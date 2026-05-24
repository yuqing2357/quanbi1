from __future__ import annotations

from yj_studio.scene.layers import FaultStickLayer
from yj_studio.tools.horizon_stick_tool import _BaseStickTool


class FaultStickTool(_BaseStickTool):
    layer_cls = FaultStickLayer
    points_field = "sticks"
    layer_key = "fault_stick"
    layer_title = "Fault Stick"
    layer_color = (0.2, 0.85, 0.8, 0.95)

    def __init__(self) -> None:
        super().__init__(tool_id="fault_stick", label="Fault Stick", icon="pen")
