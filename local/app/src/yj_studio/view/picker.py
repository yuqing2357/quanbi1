from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import vtk

from yj_studio.view.display_coordinates import sample_z

PickMode = Literal["cell", "point", "prop"]


@dataclass(frozen=True, slots=True)
class PickResult:
    world_xyz: tuple[float, float, float] | None = None
    layer_id: str | None = None
    picked_type: PickMode | None = None
    picked_cell_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Picker:
    def pick(
        self,
        view: Any,
        screen_xy: tuple[int, int],
        modes: tuple[PickMode, ...] = ("cell", "point", "prop"),
    ) -> PickResult:
        renderer = _renderer_for_view(view)
        if renderer is None:
            return PickResult()
        picker = _picker_for_modes(modes)
        x, y = int(screen_xy[0]), int(screen_xy[1])
        picked = picker.Pick(x, y, 0.0, renderer)
        if int(picked) <= 0:
            return PickResult()
        world_xyz = _sample_world_point(view, tuple(float(value) for value in picker.GetPickPosition()))
        picked_actor = picker.GetActor() if hasattr(picker, "GetActor") else None
        layer_id = _layer_id_from_actor(view, picked_actor)
        picked_cell_id = int(picker.GetCellId()) if hasattr(picker, "GetCellId") else None
        picked_type: PickMode | None = None
        if "cell" in modes and picked_cell_id is not None and picked_cell_id >= 0:
            picked_type = "cell"
        elif "point" in modes:
            picked_type = "point"
        elif "prop" in modes:
            picked_type = "prop"
        return PickResult(
            world_xyz=world_xyz,
            layer_id=layer_id,
            picked_type=picked_type,
            picked_cell_id=picked_cell_id if picked_cell_id is not None and picked_cell_id >= 0 else None,
            extra={"actor_name": _actor_name(view, picked_actor)},
        )


def _picker_for_modes(modes: tuple[PickMode, ...]) -> vtk.vtkAbstractPicker:
    if "cell" in modes:
        return vtk.vtkCellPicker()
    if "point" in modes:
        return vtk.vtkPointPicker()
    return vtk.vtkPropPicker()


def _renderer_for_view(view: Any):
    renderer = getattr(view, "renderer", None)
    if renderer is not None:
        return renderer
    renderers = getattr(view, "renderers", None)
    if renderers:
        return renderers[0]
    return None


def _layer_id_from_actor(view: Any, actor: Any) -> str | None:
    actor_name = _actor_name(view, actor)
    if actor_name is None:
        return None
    if actor_name.endswith("-line"):
        actor_name = actor_name[:-5]
    elif actor_name.endswith("-points"):
        actor_name = actor_name[:-7]
    for prefix in (
        "horizon-",
        "fault-",
        "lith-body-",
        "well-tube-",
        "well-head-",
        "well-label-",
        "well-log-",
        "arbitrary_section-",
        "polygon-",
        "horizon_stick-",
        "fault_stick-",
        "measurement-",
    ):
        if actor_name.startswith(prefix):
            return actor_name.removeprefix(prefix)
    if actor_name.startswith("volume-slice-"):
        active = _active_volume_layer_id(view)
        return active
    return actor_name


def _active_volume_layer_id(view: Any) -> str | None:
    layer_store = getattr(view, "layer_store", None)
    if layer_store is None:
        return None
    from yj_studio.scene.layers import VolumeLayer

    for layer in layer_store.iter_by_type(VolumeLayer):
        return layer.id
    return None


def _active_z_count(view: Any) -> int | None:
    layer_store = getattr(view, "layer_store", None)
    if layer_store is None:
        return None
    from yj_studio.scene.layers import VolumeLayer

    for layer in layer_store.iter_by_type(VolumeLayer):
        if layer.shape is not None:
            return int(layer.shape[2])
    return None


def _sample_world_point(view: Any, point: tuple[float, float, float]) -> tuple[float, float, float]:
    z_count = _active_z_count(view)
    return (float(point[0]), float(point[1]), float(sample_z(float(point[2]), z_count)))


def _actor_name(view: Any, actor: Any) -> str | None:
    if actor is None:
        return None
    actor_id = _actor_identity(actor)
    for name, candidate in getattr(view, "actors", {}).items():
        if candidate is actor:
            return str(name)
        if _actor_identity(candidate) == actor_id:
            return str(name)
    return None


def _actor_identity(actor: Any) -> str | None:
    if actor is None:
        return None
    getter = getattr(actor, "GetAddressAsString", None)
    if callable(getter):
        try:
            return str(getter(""))
        except Exception:
            return None
    return None
