from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from yj_studio.io.readers.layers_npz import load_layer_grid
from yj_studio.scene.layers import HorizonLayer
from yj_studio.view.display_coordinates import display_z, layer_z_count
from yj_studio.view.highlight import highlight_color, highlight_opacity


class HorizonRenderer:
    """Render lazy HorizonLayer objects as decimated PyVista surfaces."""

    def __init__(self, plotter, stride: int = 8) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._plotter = plotter
        self._stride = stride
        self._actor_names: dict[str, str] = {}

    def render(self, layer: HorizonLayer, *, highlighted: bool = False, z_count: int | None = None) -> None:
        actor_name = self._actor_name(layer)
        if not layer.visible:
            self.clear(layer.id)
            return
        _ensure_horizon_arrays(layer)
        if layer.sample is None:
            return
        mesh = build_horizon_mesh(
            layer.sample,
            layer.mask,
            stride=self._stride,
            z_count=layer_z_count(layer.metadata) or z_count,
        )
        if mesh.n_cells == 0:
            return
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            color=highlight_color(layer.color, highlighted),
            opacity=highlight_opacity(layer.opacity, highlighted),
            smooth_shading=False,
            lighting=False,
            pickable=True,
        )
        self._plotter.render()

    def clear(self, layer_id: str) -> None:
        actor_name = self._actor_names.get(layer_id, f"horizon-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()

    def _actor_name(self, layer: HorizonLayer) -> str:
        actor_name = f"horizon-{layer.id}"
        self._actor_names[layer.id] = actor_name
        return actor_name


def build_horizon_mesh(
    sample: np.ndarray,
    mask: np.ndarray | None,
    *,
    stride: int = 8,
    z_count: int | None = None,
) -> pv.PolyData:
    """Build a decimated quad mesh, skipping cells with invalid corners."""

    if stride < 1:
        raise ValueError("stride must be >= 1")
    sample_view = np.asarray(sample, dtype=np.float32)[::stride, ::stride]
    if mask is None:
        valid = np.isfinite(sample_view)
    else:
        valid = np.asarray(mask, dtype=bool)[::stride, ::stride] & np.isfinite(sample_view)

    nx, ny = sample_view.shape
    x_index = np.arange(0, sample.shape[0], stride, dtype=np.float32)[:nx]
    y_index = np.arange(0, sample.shape[1], stride, dtype=np.float32)[:ny]
    x_grid, y_grid = np.meshgrid(x_index, y_index, indexing="ij")
    points = np.column_stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.nan_to_num(display_z(sample_view, z_count), nan=0.0).ravel(),
        ]
    ).astype(np.float32)

    faces: list[int] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if not (valid[i, j] and valid[i + 1, j] and valid[i + 1, j + 1] and valid[i, j + 1]):
                continue
            p0 = i * ny + j
            p1 = (i + 1) * ny + j
            p2 = (i + 1) * ny + (j + 1)
            p3 = i * ny + (j + 1)
            faces.extend([4, p0, p1, p2, p3])
    if not faces:
        return pv.PolyData()
    return pv.PolyData(points, np.asarray(faces, dtype=np.int64))


def _ensure_horizon_arrays(layer: HorizonLayer) -> None:
    if layer.sample is not None:
        return
    if layer.data_path is None:
        return
    grid = load_layer_grid(Path(layer.data_path))
    layer.sample = grid.sample
    layer.mask = grid.mask
    layer.metadata.update(grid.metadata)
