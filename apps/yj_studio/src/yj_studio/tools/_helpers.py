from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt

from yj_studio.data.volume_store import VolumeStore
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import MaskLayer, VolumeLayer
from yj_studio.view.display_coordinates import display_z, sample_z


@dataclass(slots=True)
class SliceContext:
    axis: str
    slice_index: int
    shape: tuple[int, int]
    row: int
    col: int
    world_xyz: tuple[float, float, float]


def tool_layer_store(view: Any) -> LayerStore | None:
    return getattr(view, "layer_store", None)


def tool_volume_store(view: Any) -> VolumeStore | None:
    return getattr(view, "volume_store", None)


def tool_notify(view: Any, message: str) -> None:
    manager = getattr(view, "tool_manager", None)
    if manager is not None and hasattr(manager, "notify"):
        manager.notify(message)
        return
    callback = getattr(view, "status_message", None)
    if callable(callback):
        callback(message)


def active_volume_layer(view: Any) -> VolumeLayer | None:
    layer_store = tool_layer_store(view)
    if layer_store is None:
        return None
    for layer in layer_store.iter_by_type(VolumeLayer):
        if layer.shape is not None:
            return layer
    return None


def active_z_count(view: Any) -> int | None:
    layer = active_volume_layer(view)
    if layer is None or layer.shape is None:
        return None
    return int(layer.shape[2])


def display_world_point(view: Any, point: tuple[float, float, float]) -> tuple[float, float, float]:
    z_count = active_z_count(view)
    return (float(point[0]), float(point[1]), float(display_z(float(point[2]), z_count)))


def sample_world_point(view: Any, point: tuple[float, float, float]) -> tuple[float, float, float]:
    z_count = active_z_count(view)
    return (float(point[0]), float(point[1]), float(sample_z(float(point[2]), z_count)))


def event_world_point(view: Any, event: Any, *, prefer_mouse: bool = False) -> tuple[float, float, float] | None:
    xdata = getattr(event, "xdata", None)
    ydata = getattr(event, "ydata", None)
    axis = getattr(view, "axis", None)
    if xdata is not None and ydata is not None and axis in {"inline", "xline", "z"}:
        index = int(getattr(view, "index", 0))
        if axis == "inline":
            return (float(index), float(xdata), float(ydata))
        if axis == "xline":
            return (float(xdata), float(index), float(ydata))
        return (float(xdata), float(ydata), float(index))

    picked = _pick_world_point_from_event(view, event)
    if picked is not None:
        return picked

    if xdata is not None and ydata is not None and hasattr(view, "data"):
        return (float(xdata), 0.0, float(ydata))
    return None


def event_screen_xy(event: Any) -> tuple[float, float] | None:
    position = getattr(event, "position", None)
    if callable(position):
        p = position()
        return float(p.x()), float(p.y())
    pos = getattr(event, "pos", None)
    if callable(pos):
        p = pos()
        return float(p.x()), float(p.y())
    x = getattr(event, "x", None)
    y = getattr(event, "y", None)
    if x is not None and y is not None:
        return float(x), float(y)
    return None


def event_display_xy(view: Any, event: Any) -> tuple[float, float] | None:
    point = event_screen_xy(event)
    if point is None:
        return None
    if _view_uses_vtk_display_coordinates(view):
        height = float(view.height()) if hasattr(view, "height") else 0.0
        return float(point[0]), max(0.0, height - 1.0 - float(point[1]))
    return point


def event_left_button(event: Any) -> bool:
    left_value = _button_value(Qt.MouseButton.LeftButton)
    buttons = getattr(event, "buttons", None)
    if callable(buttons):
        buttons = buttons()
    if buttons is not None:
        buttons_value = _button_value(buttons)
        if buttons_value is not None and left_value is not None:
            return bool(buttons_value & left_value)
    button = getattr(event, "button", None)
    if callable(button):
        button = button()
    if button is None:
        return False
    button_value = _button_value(button)
    return button_value is not None and left_value is not None and button_value == left_value


def _button_value(value: Any) -> int | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def is_double_click(event: Any) -> bool:
    return bool(getattr(event, "dblclick", False))


def _pick_world_point_from_event(view: Any, event: Any) -> tuple[float, float, float] | None:
    display_xy = event_display_xy(view, event)
    renderer = _renderer_for_view(view)
    if display_xy is None or renderer is None:
        return None
    try:
        import vtk

        picker = vtk.vtkCellPicker()
        picked = picker.Pick(float(display_xy[0]), float(display_xy[1]), 0.0, renderer)
        if int(picked) <= 0:
            return None
        picked = picker.GetPickPosition()
        xyz = tuple(float(value) for value in picked[:3])
        if all(np.isfinite(value) for value in xyz):
            return sample_world_point(view, xyz)
    except Exception:
        return None
    return None


def _renderer_for_view(view: Any):
    renderer = getattr(view, "renderer", None)
    if renderer is not None:
        return renderer
    renderers = getattr(view, "renderers", None)
    if renderers:
        return renderers[0]
    return None


def _view_uses_vtk_display_coordinates(view: Any) -> bool:
    return _renderer_for_view(view) is not None


def slice_context_from_event(view: Any, event: Any, *, prefer_mouse: bool = False) -> SliceContext | None:
    volume_layer = active_volume_layer(view)
    if volume_layer is None or volume_layer.shape is None:
        return None
    world_xyz = event_world_point(view, event, prefer_mouse=prefer_mouse)
    if world_xyz is None:
        return None
    nx, ny, nz = volume_layer.shape
    axis = getattr(view, "axis", None)
    index = getattr(view, "index", None)
    if axis in {"inline", "xline", "z"} and index is not None:
        return _context_for_axis(axis, int(index), volume_layer.shape, world_xyz, event)

    slice_indices = volume_layer.slice_indices or {}
    candidates = [
        ("inline", abs(float(world_xyz[0]) - float(slice_indices.get("inline", nx // 2))), int(round(world_xyz[0]))),
        ("xline", abs(float(world_xyz[1]) - float(slice_indices.get("xline", ny // 2))), int(round(world_xyz[1]))),
        ("z", abs(float(world_xyz[2]) - float(slice_indices.get("z", nz // 2))), int(round(world_xyz[2]))),
    ]
    axis_name, _, slice_index = min(candidates, key=lambda item: item[1])
    return _context_for_axis(axis_name, slice_index, volume_layer.shape, world_xyz, event)


def _context_for_axis(
    axis: str,
    slice_index: int,
    shape: tuple[int, int, int],
    world_xyz: tuple[float, float, float],
    event: Any,
) -> SliceContext:
    nx, ny, nz = shape
    if axis == "inline":
        row = int(round(world_xyz[1] if getattr(event, "xdata", None) is None else float(event.xdata)))
        col = int(round(world_xyz[2] if getattr(event, "ydata", None) is None else float(event.ydata)))
        return SliceContext(axis, slice_index, (ny, nz), row, col, world_xyz)
    if axis == "xline":
        row = int(round(world_xyz[0] if getattr(event, "xdata", None) is None else float(event.xdata)))
        col = int(round(world_xyz[2] if getattr(event, "ydata", None) is None else float(event.ydata)))
        return SliceContext(axis, slice_index, (nx, nz), row, col, world_xyz)
    row = int(round(world_xyz[0] if getattr(event, "xdata", None) is None else float(event.xdata)))
    col = int(round(world_xyz[1] if getattr(event, "ydata", None) is None else float(event.ydata)))
    return SliceContext(axis, slice_index, (nx, ny), row, col, world_xyz)


def ensure_mask_layer(
    layer_store: LayerStore,
    *,
    tool_name: str,
    axis: str,
    slice_index: int,
    shape: tuple[int, int],
    color: tuple[float, float, float, float],
    opacity: float,
) -> MaskLayer:
    for layer in layer_store.iter_by_type(MaskLayer):
        if layer.axis == axis and layer.slice_index == slice_index and layer.metadata.get("tool") == tool_name:
            if layer.mask is None or layer.mask.shape != shape:
                layer.mask = np.zeros(shape, dtype=np.uint8)
            return layer
    layer = MaskLayer(
        name=f"{_tool_display_name(tool_name)} {axis} {slice_index}",
        color=color,
        opacity=opacity,
        visible=True,
        axis=axis,
        slice_index=slice_index,
        mask=np.zeros(shape, dtype=np.uint8),
        metadata={"tool": tool_name, "shape": list(shape)},
        provenance={"source": "manual"},
    )
    layer_store.add(layer)
    return layer


def paint_disk(mask: np.ndarray, row: int, col: int, radius: int, value: int) -> None:
    if mask.ndim != 2:
        return
    rows, cols = mask.shape
    r0 = max(0, row - radius)
    r1 = min(rows, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(cols, col + radius + 1)
    rr, cc = np.ogrid[r0:r1, c0:c1]
    circle = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    patch = mask[r0:r1, c0:c1]
    patch[circle] = value


def project_world_point(view: Any, point: tuple[float, float, float]) -> tuple[float, float] | None:
    renderer = getattr(view, "renderer", None)
    if renderer is None:
        renderers = getattr(view, "renderers", None)
        if renderers:
            renderer = renderers[0]
    if renderer is None:
        return None
    try:
        display_point = display_world_point(view, point)
        renderer.SetWorldPoint(float(display_point[0]), float(display_point[1]), float(display_point[2]), 1.0)
        renderer.WorldToDisplay()
        display = renderer.GetDisplayPoint()
        return float(display[0]), float(display[1])
    except Exception:
        return None


def view_rect_from_events(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float, float, float]:
    x0, y0 = start
    x1, y1 = end
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def next_layer_name(layer_store: LayerStore, prefix: str) -> str:
    count = 0
    for layer in layer_store.iter_layers():
        if layer.name.startswith(prefix):
            count += 1
    return f"{prefix} {count + 1}"


def _tool_display_name(tool_name: str) -> str:
    return {
        "paint_mask": "掩膜",
        "fill": "填充",
        "connected_component": "连通域",
        "threshold": "阈值",
        "region_grow": "区域生长",
        "snap": "吸附",
        "contour": "等值线",
        "horizon_autotrack": "层位自动追踪",
        "ai_point_prompt": "AI 点提示",
        "ai_box_prompt": "AI 框提示",
        "measure": "测量",
        "polygon": "多边形",
        "brush": "画笔",
        "eraser": "橡皮",
        "horizon_stick": "层位杆",
        "fault_stick": "断层杆",
    }.get(tool_name, tool_name.replace("_", " "))
