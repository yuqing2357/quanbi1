from __future__ import annotations

from typing import Any

import numpy as np

from yj_studio.scene.layers import FaultSurfaceLayer, HorizonLayer, WellLayer, WellLogLayer
from yj_studio.scene.manual_geometry import is_manual_geometry_layer, manual_geometry_points, project_points_to_section
from yj_studio.services.section_service import fault_points_intersection, horizon_intersection, well_intersection
from yj_studio.tools._geometry import bbox_corners, point_in_rect, points_in_rect
from yj_studio.tools._helpers import (
    event_display_xy,
    event_left_button,
    project_world_point,
    tool_layer_store,
    tool_notify,
    view_rect_from_events,
)
from yj_studio.tools.point_pick_tool import _well_log_points
from yj_studio.tools.tool import InteractionTool


class BoxPickTool(InteractionTool):
    def __init__(self) -> None:
        super().__init__(id="box_pick", label="Box Pick", icon="square", cursor="crosshair")
        self._start: tuple[float, float] | None = None

    def deactivate(self, view: Any) -> None:
        self._start = None

    def on_mouse_press(self, view, event) -> bool:
        if not event_left_button(event):
            return False
        pos = _event_xy(view, event)
        if pos is None:
            return False
        self._start = pos
        return True

    def on_mouse_move(self, view, event) -> bool:
        return self._start is not None

    def on_mouse_release(self, view, event) -> bool:
        if self._start is None:
            return False
        end = _event_xy(view, event)
        if end is None:
            self._start = None
            return False
        layer_store = tool_layer_store(view)
        if layer_store is None:
            self._start = None
            return False
        rect = view_rect_from_events(self._start, end)
        selected = _select_layers(view, layer_store, rect)
        if selected:
            layer_store.select(selected)
            tool_notify(view, f"Selected {len(selected)} layers")
        else:
            layer_store.select([])
            tool_notify(view, "Selection cleared")
        self._start = None
        return True


def _select_layers(view, layer_store, rect: tuple[float, float, float, float]) -> list[str]:
    axis = getattr(view, "axis", None)
    index = int(getattr(view, "index", 0))
    selected: list[str] = []
    for layer in layer_store.iter_layers():
        if not layer.visible:
            continue
        if axis in {"inline", "xline", "z"} and _section_hits(layer, axis, index, rect):
            selected.append(layer.id)
            continue
        if _projected_bbox_hits(view, layer.bounding_box(), rect):
            selected.append(layer.id)
    return selected


def _section_hits(layer, axis: str, index: int, rect: tuple[float, float, float, float]) -> bool:
    if isinstance(layer, HorizonLayer):
        line = horizon_intersection(layer, axis, index)
        if line is not None:
            return points_in_rect(np.column_stack([line.x, line.y]), rect)
    if isinstance(layer, WellLayer):
        line = well_intersection(layer, axis, index)
        if line is not None:
            return points_in_rect(np.column_stack([line.x, line.y]), rect)
    if isinstance(layer, FaultSurfaceLayer):
        points = fault_points_intersection(layer, axis, index)
        if points is not None:
            return points_in_rect(np.column_stack([points.x, points.y]), rect)
    if isinstance(layer, WellLogLayer):
        points = _well_log_points(layer, axis, index)
        if points is not None:
            x, y = points
            return points_in_rect(np.column_stack([x, y]), rect)
    if is_manual_geometry_layer(layer):
        points = manual_geometry_points(layer)
        projected = project_points_to_section(points, axis, index)
        if projected is not None:
            x, y = projected
            return points_in_rect(np.column_stack([x, y]), rect)
    return False


def _projected_bbox_hits(view, bbox: tuple[float, float, float, float, float, float], rect: tuple[float, float, float, float]) -> bool:
    corners = bbox_corners(bbox)
    projected = [project_world_point(view, tuple(point)) for point in corners]
    hits = [point for point in projected if point is not None and point_in_rect(point, rect)]
    return bool(hits)


def _event_xy(view, event) -> tuple[float, float] | None:
    xdata = getattr(event, "xdata", None)
    ydata = getattr(event, "ydata", None)
    if xdata is not None and ydata is not None:
        return float(xdata), float(ydata)
    if hasattr(event, "inaxes"):
        return None
    return event_display_xy(view, event)
