from __future__ import annotations

from yj_studio.tools._helpers import tool_notify
from yj_studio.tools.tool import InteractionTool


class _PhaseTwoStubTool(InteractionTool):
    def __init__(self, *, tool_id: str, label: str) -> None:
        super().__init__(id=tool_id, label=label, icon="ban", cursor="arrow", enabled=False)

    def on_mouse_press(self, view, event) -> bool:
        tool_notify(view, f"{self.label} will be available in Phase 2")
        return True


class FillTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="fill", label="Fill")


class ConnectedComponentTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="connected_component", label="Connected Component")


class ThresholdTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="threshold", label="Threshold")


class RegionGrowTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="region_grow", label="Region Grow")


class SnapTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="snap", label="Snap")


class ContourTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="contour", label="Contour")


class HorizonAutotrackTool(_PhaseTwoStubTool):
    def __init__(self) -> None:
        super().__init__(tool_id="horizon_autotrack", label="Horizon Autotrack")
