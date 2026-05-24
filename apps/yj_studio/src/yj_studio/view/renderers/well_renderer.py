from __future__ import annotations

import numpy as np
import pyvista as pv

from yj_studio.scene.layers import WellLayer
from yj_studio.view.display_coordinates import transform_points_z
from yj_studio.view.highlight import highlight_color, highlight_opacity


class WellRenderer:
    """Render well trajectory, head marker, and name label."""

    def __init__(self, plotter, *, tube_radius: float = 1.5, head_radius: float = 5.0) -> None:
        self._plotter = plotter
        self._tube_radius = tube_radius
        self._head_radius = head_radius
        self._actor_names: dict[str, tuple[str, str, str]] = {}

    def render(self, layer: WellLayer, *, highlighted: bool = False, z_count: int | None = None) -> None:
        names = self._actor_name_tuple(layer)
        if not layer.visible:
            self.clear(layer.id)
            return
        if layer.trajectory is None or layer.head_position is None:
            return

        self.clear(layer.id, render=False)
        color = highlight_color(layer.color, highlighted)
        trajectory_mesh = build_well_tube(
            layer.trajectory,
            radius=self._tube_radius * (1.8 if highlighted else 1.0),
            z_count=z_count,
        )
        if trajectory_mesh.n_points:
            self._plotter.add_mesh(
                trajectory_mesh,
                name=names[0],
                color=color,
                opacity=highlight_opacity(layer.opacity, highlighted),
                smooth_shading=True,
                lighting=False,
                pickable=True,
            )

        head_position = transform_points_z(np.asarray([layer.head_position], dtype=np.float32), z_count)[0]
        head_mesh = pv.Sphere(
            radius=self._head_radius * (1.6 if highlighted else 1.0),
            center=head_position,
        )
        self._plotter.add_mesh(
            head_mesh,
            name=names[1],
            color=color,
            opacity=1.0,
            smooth_shading=True,
            lighting=False,
            pickable=True,
        )
        self._plotter.add_point_labels(
            np.asarray([head_position], dtype=np.float32),
            [layer.well_name or layer.name],
            name=names[2],
            font_size=14 if highlighted else 10,
            text_color="yellow" if highlighted else "white",
            point_color=color,
            point_size=0,
            shape_opacity=0.0,
            always_visible=True,
        )
        self._plotter.render()

    def clear(self, layer_id: str, *, render: bool = True) -> None:
        names = self._actor_names.get(layer_id, _actor_names(layer_id))
        for name in names:
            self._plotter.remove_actor(name, reset_camera=False, render=False)
        if render:
            self._plotter.render()

    def _actor_name_tuple(self, layer: WellLayer) -> tuple[str, str, str]:
        names = _actor_names(layer.id)
        self._actor_names[layer.id] = names
        return names


def build_well_tube(
    trajectory: np.ndarray,
    *,
    radius: float = 1.5,
    z_count: int | None = None,
) -> pv.PolyData:
    """Build a tube mesh around a well trajectory polyline."""

    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"trajectory must have shape (N, 3), got {points.shape}")
    if points.shape[0] < 2:
        return pv.PolyData()
    points = transform_points_z(points, z_count)
    polyline = pv.lines_from_points(points, close=False)
    return polyline.tube(radius=radius, n_sides=12)


def _actor_names(layer_id: str) -> tuple[str, str, str]:
    return (f"well-tube-{layer_id}", f"well-head-{layer_id}", f"well-label-{layer_id}")
