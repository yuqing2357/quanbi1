from __future__ import annotations

import numpy as np
import pyvista as pv

from yj_studio.scene.manual_geometry import manual_geometry_points
from yj_studio.scene.layers import (
    ArbitrarySectionLayer,
    FaultStickLayer,
    HorizonStickLayer,
    MeasurementLayer,
    PolygonLayer,
)
from yj_studio.view.display_coordinates import transform_points_z
from yj_studio.view.highlight import highlight_color, highlight_opacity


class ManualGeometryRenderer:
    """Render small manually created interpretation geometry."""

    def __init__(self, plotter) -> None:
        self._plotter = plotter
        self._actor_names: dict[str, str] = {}

    def render(
        self,
        layer: ArbitrarySectionLayer | PolygonLayer | HorizonStickLayer | FaultStickLayer | MeasurementLayer,
        *,
        highlighted: bool = False,
        z_count: int | None = None,
    ) -> None:
        actor_name = self._actor_name(layer)
        if not layer.visible:
            self.clear(layer.id)
            return
        points = manual_geometry_points(layer)
        if points is None or points.shape[0] == 0:
            self.clear(layer.id)
            return
        display_points = transform_points_z(points, z_count)
        mesh = _line_mesh(display_points, close=bool(getattr(layer, "closed", False)))
        point_mesh = pv.PolyData(display_points)
        color = highlight_color(layer.color, highlighted)
        self.clear(layer.id, render=False)
        if mesh.n_points >= 2:
            self._plotter.add_mesh(
                mesh,
                name=f"{actor_name}-line",
                color=color,
                opacity=highlight_opacity(layer.opacity, highlighted),
                line_width=4 if highlighted else 2,
                lighting=False,
                pickable=True,
            )
        self._plotter.add_mesh(
            point_mesh,
            name=f"{actor_name}-points",
            color=color,
            opacity=1.0,
            render_points_as_spheres=True,
            point_size=12 if highlighted else 8,
            lighting=False,
            pickable=True,
        )
        self._plotter.render()

    def clear(self, layer_id: str, *, render: bool = True) -> None:
        actor_name = self._actor_names.get(layer_id, f"manual-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.remove_actor(f"{actor_name}-line", reset_camera=False, render=False)
        self._plotter.remove_actor(f"{actor_name}-points", reset_camera=False, render=False)
        if render:
            self._plotter.render()

    def _actor_name(self, layer) -> str:
        actor_name = f"{layer.kind}-{layer.id}"
        self._actor_names[layer.id] = actor_name
        return actor_name


def _line_mesh(points: np.ndarray, *, close: bool) -> pv.PolyData:
    if points.shape[0] < 2:
        return pv.PolyData()
    return pv.lines_from_points(points, close=close)
