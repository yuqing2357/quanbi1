from __future__ import annotations

from .ai_prompt_tools import AIBoxPromptTool, AIPointPromptTool
from .box_pick_tool import BoxPickTool
from .brush_tool import BrushTool
from .eraser_tool import EraserTool
from .fault_stick_tool import FaultStickTool
from .horizon_stick_tool import HorizonStickTool
from .measure_tool import MeasureTool
from .navigation_tool import NavigationTool
from .point_pick_tool import PointPickTool
from .polygon_tool import PolygonTool
from .stubs import (
    ConnectedComponentTool,
    ContourTool,
    FillTool,
    HorizonAutotrackTool,
    RegionGrowTool,
    SnapTool,
    ThresholdTool,
)


def build_default_tools():
    return [
        NavigationTool(),
        PointPickTool(),
        BoxPickTool(),
        PolygonTool(),
        BrushTool(),
        EraserTool(),
        HorizonStickTool(),
        FaultStickTool(),
        MeasureTool(),
        AIPointPromptTool(),
        AIBoxPromptTool(),
        FillTool(),
        ConnectedComponentTool(),
        ThresholdTool(),
        RegionGrowTool(),
        SnapTool(),
        ContourTool(),
        HorizonAutotrackTool(),
    ]
