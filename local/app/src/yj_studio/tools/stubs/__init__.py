from __future__ import annotations

from yj_studio.tools._helpers import tool_notify
from yj_studio.tools.tool import InteractionTool


class _PhaseTwoStubTool(InteractionTool):
    def __init__(self, *, tool_id: str, label: str) -> None:
        super().__init__(id=tool_id, label=label, icon="ban", cursor="arrow", enabled=False)

    def on_mouse_press(self, view, event) -> bool:
        tool_notify(view, f"{self.label} 将在第二阶段提供")
        return True


class FillTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="fill", label="填充")


class ConnectedComponentTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="connected_component", label="连通域")


class ThresholdTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="threshold", label="阈值")


class RegionGrowTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="region_grow", label="区域生长")


class SnapTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="snap", label="吸附")


class ContourTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="contour", label="等值线")


class HorizonAutotrackTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="horizon_autotrack", label="层位自动追踪")
