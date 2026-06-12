from __future__ import annotations

import numpy as np
import pyvista as pv

from yj_studio.scene.layers import WellLogLayer
from yj_studio.view.display_coordinates import transform_points_z
from yj_studio.view.highlight import highlight_opacity


class WellLogRenderer:
    """Render well log samples as colored points along the well path."""

    def __init__(self, plotter, *, point_size: float = 7.0) -> None:
        self._plotter = plotter
        self._point_size = point_size
        self._actor_names: dict[str, str] = {}

    def render(self, layer: WellLogLayer, *, highlighted: bool = False, z_count: int | None = None) -> None:
        actor_name = self._actor_name(layer)
        if not layer.visible:
            self.clear(layer.id)
            return
        if layer.samples is None:
            return
        mesh = build_well_log_points(layer.samples, z_count=z_count)
        if mesh.n_points == 0:
            return
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            scalars="value",
            cmap=str(layer.metadata.get("cmap", _default_cmap(layer.mode))),
            clim=tuple(layer.metadata.get("clim", _default_clim(layer.mode))),
            opacity=highlight_opacity(layer.opacity, highlighted),
            render_points_as_spheres=True,
            point_size=float(layer.metadata.get("point_size", self._point_size))
            * (1.8 if highlighted else 1.0),
            lighting=False,
            pickable=True,
        )
        self._plotter.render()

    def clear(self, layer_id: str) -> None:
        actor_name = self._actor_names.get(layer_id, f"well-log-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()

    def _actor_name(self, layer: WellLogLayer) -> str:
        actor_name = f"well-log-{layer.id}"
        self._actor_names[layer.id] = actor_name
        return actor_name


def build_well_log_points(samples: np.ndarray, *, z_count: int | None = None) -> pv.PolyData:
    """Build point geometry from [inline, xline, sample, value] rows."""

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError(f"samples must have shape (N, 4), got {samples.shape}")
    if samples.size == 0:
        return pv.PolyData()
    finite = np.all(np.isfinite(samples), axis=1)
    if not np.any(finite):
        return pv.PolyData()
    clean = samples[finite]
    mesh = pv.PolyData(transform_points_z(clean[:, :3], z_count))
    mesh["value"] = clean[:, 3]
    return mesh


def _default_cmap(mode: str) -> str:
    if mode == "lith":
        return "tab10"
    if mode == "perm":
        return "plasma"
    return "viridis"


def _default_clim(mode: str) -> tuple[float, float]:
    if mode == "lith":
        return (-0.5, 5.5)
    if mode == "perm":
        return (0.0, 1000.0)
    return (0.0, 0.35)
