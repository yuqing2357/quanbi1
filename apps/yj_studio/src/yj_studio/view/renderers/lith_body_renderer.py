from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from yj_studio.io.readers.lith_body import load_lith_body_mesh
from yj_studio.scene.layers import LithBodyLayer
from yj_studio.view.display_coordinates import layer_z_count, transform_points_z
from yj_studio.view.highlight import highlight_color, highlight_opacity


class LithBodyRenderer:
    """Render lazy lithology body meshes."""

    def __init__(self, plotter) -> None:
        self._plotter = plotter
        self._actor_names: dict[str, str] = {}

    def render(self, layer: LithBodyLayer, *, highlighted: bool = False, z_count: int | None = None) -> None:
        actor_name = self._actor_name(layer)
        if not layer.visible:
            self.clear(layer.id)
            return
        _ensure_lith_body_arrays(layer)
        if layer.vertices is None or layer.faces is None:
            return
        mesh = build_lith_body_mesh(layer.vertices, layer.faces, z_count=layer_z_count(layer.metadata) or z_count)
        if mesh.n_cells == 0:
            return
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            color=highlight_color(layer.color, highlighted),
            opacity=highlight_opacity(layer.opacity, highlighted),
            smooth_shading=True,
            lighting=False,
            pickable=True,
        )
        self._plotter.render()

    def clear(self, layer_id: str) -> None:
        actor_name = self._actor_names.get(layer_id, f"lith-body-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()

    def _actor_name(self, layer: LithBodyLayer) -> str:
        actor_name = f"lith-body-{layer.id}"
        self._actor_names[layer.id] = actor_name
        return actor_name


def build_lith_body_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    z_count: int | None = None,
) -> pv.PolyData:
    """Build a PyVista triangle mesh from lithology body vertices/faces."""

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")
    if vertices.size == 0 or faces.size == 0:
        return pv.PolyData()
    vertices = transform_points_z(vertices, z_count)
    face_prefix = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    vtk_faces = np.hstack([face_prefix, faces]).ravel()
    return pv.PolyData(vertices, vtk_faces)


def _ensure_lith_body_arrays(layer: LithBodyLayer) -> None:
    if layer.vertices is not None and layer.faces is not None:
        return
    if layer.data_path is None:
        return
    mesh = load_lith_body_mesh(
        Path(layer.data_path),
        class_value=layer.class_value,
        class_name=layer.class_name,
    )
    layer.vertices = mesh.vertices
    layer.faces = mesh.faces
    layer.metadata.update(mesh.metadata)
