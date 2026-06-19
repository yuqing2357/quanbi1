from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yj_studio.app import create_application
from PyQt6.QtCore import Qt
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import VolumeLayer
from yj_studio.tools import ToolManager, build_default_tools
from yj_studio.tools._helpers import display_world_point, event_left_button, sample_world_point


def test_default_tool_catalog() -> None:
    tools = build_default_tools()
    assert len(tools) == 17
    assert tools[0].id == "navigation"
    assert "measure" not in {tool.id for tool in tools}
    assert sum(1 for tool in tools if tool.enabled) == 10
    assert sum(1 for tool in tools if not tool.enabled) == 7


def test_tool_manager_tracks_active_tool_and_views() -> None:
    app = create_application([])
    manager = ToolManager()
    for tool in build_default_tools()[:2]:
        manager.register(tool)

    class DummyView:
        def __init__(self) -> None:
            self.cursor = None

        def setCursor(self, cursor) -> None:  # noqa: N802
            self.cursor = cursor

    view = DummyView()
    manager.attach_view(view)
    assert view.tool_manager is manager
    assert manager.active_tool is not None
    assert manager.active_tool.id == "navigation"

    manager.activate("point_pick")
    assert manager.active_tool is not None
    assert manager.active_tool.id == "point_pick"
    app.quit()


def test_event_left_button_accepts_matplotlib_and_qt_events() -> None:
    class MatplotlibEvent:
        button = 1

    class QtEvent:
        def button(self):
            return Qt.MouseButton.LeftButton

    class QtButtonsEvent:
        def buttons(self):
            return Qt.MouseButton.LeftButton

    class RightButtonEvent:
        button = 3

    assert event_left_button(MatplotlibEvent())
    assert event_left_button(QtEvent())
    assert event_left_button(QtButtonsEvent())
    assert not event_left_button(RightButtonEvent())


def test_tool_coordinate_helpers_convert_between_sample_and_display_z() -> None:
    store = LayerStore()
    store.add(VolumeLayer(name="Volume", volume_id="v", shape=(2, 3, 4)))

    class View:
        layer_store = store

    assert display_world_point(View(), (1.0, 2.0, 0.0)) == (1.0, 2.0, 3.0)
    assert sample_world_point(View(), (1.0, 2.0, 3.0)) == (1.0, 2.0, 0.0)
