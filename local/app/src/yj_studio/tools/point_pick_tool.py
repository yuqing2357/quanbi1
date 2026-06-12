from __future__ import annotations

import numpy as np

from yj_studio.scene.layers import FaultSurfaceLayer, HorizonLayer, WellLayer, WellLogLayer
from yj_studio.scene.manual_geometry import is_manual_geometry_layer, manual_geometry_points
from yj_studio.services.section_service import fault_points_intersection, horizon_intersection, well_intersection
from yj_studio.tools._geometry import distance_point_to_bbox, distance_point_to_points, distance_point_to_polyline
from yj_studio.tools._helpers import event_display_xy, event_world_point, tool_layer_store, tool_notify
from yj_studio.tools.tool import InteractionTool
from yj_studio.view.picker import Picker


class PointPickTool(InteractionTool):
    def __init__(self) -> None:
        super().__init__(id="point_pick", label="点选", icon="crosshair", cursor="crosshair")
        self._picker = Picker()

    def on_mouse_press(self, view, event) -> bool:
        layer_store = tool_layer_store(view)
        if layer_store is None:
            return False
        if not _looks_like_pick_event(event):
            return False

        selected_id = self._pick_layer_id(view, event)
        if selected_id is None:
            return False

        layer_store.select([selected_id])
        layer = layer_store.get(selected_id)
        tool_notify(view, f"已选中 {layer.name}")
        return True

    def _pick_layer_id(self, view, event) -> str | None:
        layer_store = tool_layer_store(view)
        if layer_store is None:
            return None

        picked = None
        screen_xy = event_display_xy(view, event)
        if screen_xy is not None and hasattr(view, "pick_click_position"):
            try:
                picked = self._picker.pick(view, (int(screen_xy[0]), int(screen_xy[1])))
            except Exception:
                picked = None
        if picked is not None and picked.layer_id:
            return picked.layer_id

        world = event_world_point(view, event)
        if world is None:
            return None

        axis = getattr(view, "axis", None)
        index = int(getattr(view, "index", 0))
        best_id: str | None = None
        best_score = float("inf")
        for layer in layer_store.iter_layers():
            if not layer.visible:
                continue
            score = _layer_distance(view, layer, world, axis=axis, index=index)
            if score < best_score:
                best_score = score
                best_id = layer.id
        return best_id if best_id is not None and best_score <= 25.0 else None


def _layer_distance(view, layer, world: tuple[float, float, float], *, axis: str | None, index: int) -> float:
    if axis in {"inline", "xline", "z"}:
        if isinstance(layer, HorizonLayer):
            line = horizon_intersection(layer, axis, index)
            if line is not None:
                return _distance_to_xy(line.x, line.y, world, axis)
        if isinstance(layer, WellLayer):
            line = well_intersection(layer, axis, index)
            if line is not None:
                return _distance_to_xy(line.x, line.y, world, axis)
        if isinstance(layer, FaultSurfaceLayer):
            points = fault_points_intersection(layer, axis, index)
            if points is not None:
                return _distance_to_xy(points.x, points.y, world, axis)
        if isinstance(layer, WellLogLayer):
            points = _well_log_points(layer, axis, index)
            if points is not None:
                return _distance_to_xy(points[0], points[1], world, axis)
    if isinstance(layer, WellLayer) and layer.trajectory is not None:
        return distance_point_to_polyline(world, layer.trajectory)
    if isinstance(layer, WellLogLayer) and layer.samples is not None:
        samples = np.asarray(layer.samples, dtype=np.float32)
        if samples.ndim == 2 and samples.shape[1] >= 3:
            return distance_point_to_points(world, samples[:, :3])
    if is_manual_geometry_layer(layer):
        points = manual_geometry_points(layer)
        if points is not None:
            return distance_point_to_points(world, points)
    bbox = layer.bounding_box()
    return distance_point_to_bbox(world, bbox)


def _well_log_points(layer: WellLogLayer, axis: str, index: int) -> tuple[np.ndarray, np.ndarray] | None:
    if layer.samples is None:
        return None
    samples = np.asarray(layer.samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 4:
        return None
    axis_index = {"inline": 0, "xline": 1, "z": 2}[axis]
    mask = np.abs(samples[:, axis_index] - float(index)) <= 0.5
    if axis == "z":
        zmin = float(index) - 0.5
        zmax = float(index) + 0.5
        mask = (samples[:, 2] >= zmin) & (samples[:, 2] <= zmax)
    if not np.any(mask):
        return None
    selected = samples[mask]
    if axis == "inline":
        return selected[:, 1], selected[:, 2]
    if axis == "xline":
        return selected[:, 0], selected[:, 2]
    return selected[:, 0], selected[:, 1]


def _distance_to_xy(x: np.ndarray, y: np.ndarray, world: tuple[float, float, float], axis: str) -> float:
    if x.size == 0:
        return float("inf")
    if axis == "inline":
        point = np.asarray([world[1], world[2]], dtype=np.float32)
    elif axis == "xline":
        point = np.asarray([world[0], world[2]], dtype=np.float32)
    else:
        point = np.asarray([world[0], world[1]], dtype=np.float32)
    pts = np.column_stack([x, y]).astype(np.float32)
    delta = pts - point
    return float(np.sqrt(np.sum(delta * delta, axis=1)).min())


def _looks_like_pick_event(event) -> bool:
    return _left_or_unknown(event)


def _left_or_unknown(event) -> bool:
    from yj_studio.tools._helpers import event_left_button

    return event_left_button(event)
