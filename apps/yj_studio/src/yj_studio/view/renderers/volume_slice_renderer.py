from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyvista as pv

from yj_studio.data.volume_store import SliceAxis, VolumeStore
from yj_studio.scene.layers import VolumeLayer
from yj_studio.view.display_coordinates import display_z

try:
    from matplotlib import colormaps
except Exception:  # pragma: no cover - optional fallback
    colormaps = None

TextureAxis = Literal["inline", "xline", "z"]


@dataclass(frozen=True, slots=True)
class SliceImage:
    image: np.ndarray
    points: np.ndarray


class VolumeSliceRenderer:
    """Render three orthogonal volume slices as textured planes."""

    ACTOR_NAMES = {
        "inline": "volume-slice-inline",
        "xline": "volume-slice-xline",
        "z": "volume-slice-z",
    }

    def __init__(self, plotter, volume_store: VolumeStore) -> None:
        self._plotter = plotter
        self._volume_store = volume_store
        self._layer: VolumeLayer | None = None

    def set_layer(self, layer: VolumeLayer) -> None:
        self._layer = layer
        self.render_all()

    def render_all(self) -> None:
        if self._layer is None or self._layer.shape is None:
            return
        if not self._layer.visible:
            self.clear()
            return
        for axis in ("inline", "xline", "z"):
            self.render_axis(axis)
        self._plotter.render()

    def render_axis(self, axis: TextureAxis) -> None:
        if self._layer is None or self._layer.shape is None:
            return
        layer = self._layer
        index = _slice_index(layer, axis)
        roi = layer.effective_roi()
        if roi is not None and not _slice_within_roi(axis, index, roi):
            self._plotter.remove_actor(
                self.ACTOR_NAMES[axis], reset_camera=False, render=False
            )
            return
        raw_slice = self._volume_store.get_slice(layer.volume_id, axis, index)
        slice_image = build_slice_image(
            raw_slice,
            layer.shape,
            axis,
            index,
            layer.clim,
            layer.cmap,
            roi=roi,
        )
        mesh = _quad(slice_image.points)
        texture = pv.numpy_to_texture(slice_image.image)
        actor_name = self.ACTOR_NAMES[axis]
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            texture=texture,
            lighting=False,
            pickable=True,
            show_edges=False,
        )

    def clear(self) -> None:
        for actor_name in self.ACTOR_NAMES.values():
            self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()


def build_slice_image(
    raw_slice: np.ndarray,
    volume_shape: tuple[int, int, int],
    axis: SliceAxis,
    index: int,
    clim: tuple[float, float] | None,
    cmap: str,
    roi: tuple[int, int, int, int, int, int] | None = None,
) -> SliceImage:
    """Return a colorized image and its 3D quad points for one slice.

    ``roi`` is an optional clipping box ``(i0, i1, j0, j1, k0, k1)`` inclusive on
    both ends; when given, the slice is cropped and the quad shrunk accordingly.
    """

    nx, ny, nz = volume_shape
    if roi is None:
        i0, i1, j0, j1, k0, k1 = 0, nx - 1, 0, ny - 1, 0, nz - 1
    else:
        i0, i1, j0, j1, k0, k1 = roi

    if axis == "inline":
        values = np.asarray(raw_slice, dtype=np.float32).T  # shape (nz, ny)
        values = values[k0 : k1 + 1, j0 : j1 + 1]
        # Spatial quad places shallow (k0) at the top after display_z flip, but
        # VTK's texture v-axis maps v=0 to the LAST image row; without flipping
        # the texture along sample axis the seismic image renders upside-down
        # relative to horizons/wells/faults. Flip rows here to match the quad.
        values = values[::-1, :]
        points = np.asarray(
            [
                [index, j0, display_z(float(k0), nz)],
                [index, j1, display_z(float(k0), nz)],
                [index, j1, display_z(float(k1), nz)],
                [index, j0, display_z(float(k1), nz)],
            ],
            dtype=np.float32,
        )
    elif axis == "xline":
        values = np.asarray(raw_slice, dtype=np.float32).T  # shape (nz, nx)
        values = values[k0 : k1 + 1, i0 : i1 + 1]
        values = values[::-1, :]  # match flipped display Z, see inline branch
        points = np.asarray(
            [
                [i0, index, display_z(float(k0), nz)],
                [i1, index, display_z(float(k0), nz)],
                [i1, index, display_z(float(k1), nz)],
                [i0, index, display_z(float(k1), nz)],
            ],
            dtype=np.float32,
        )
    else:
        values = np.asarray(raw_slice, dtype=np.float32).T  # shape (ny, nx)
        values = values[j0 : j1 + 1, i0 : i1 + 1]
        z_pos = display_z(float(index), nz)
        points = np.asarray(
            [[i0, j0, z_pos], [i1, j0, z_pos], [i1, j1, z_pos], [i0, j1, z_pos]],
            dtype=np.float32,
        )
    return SliceImage(image=colorize_slice(values, clim, cmap), points=points)


def _slice_within_roi(axis: SliceAxis, index: int, roi: tuple[int, int, int, int, int, int]) -> bool:
    i0, i1, j0, j1, k0, k1 = roi
    if axis == "inline":
        return i0 <= index <= i1
    if axis == "xline":
        return j0 <= index <= j1
    return k0 <= index <= k1


def colorize_slice(values: np.ndarray, clim: tuple[float, float] | None, cmap: str) -> np.ndarray:
    finite = np.isfinite(values)
    if clim is None:
        if np.any(finite):
            vmin, vmax = np.percentile(values[finite], [2.0, 98.0])
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = clim
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[finite] = np.clip((values[finite] - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)

    cmap_name = _matplotlib_cmap_name(cmap)
    if colormaps is None:
        rgb = np.repeat((normalized * 255.0).astype(np.uint8)[..., None], 3, axis=2)
        rgb[~finite] = 0
        return rgb

    rgba = colormaps[cmap_name](normalized)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~finite] = 0
    return rgb


def _slice_index(layer: VolumeLayer, axis: SliceAxis) -> int:
    if layer.shape is None:
        return 0
    default_index = {"inline": layer.shape[0] // 2, "xline": layer.shape[1] // 2, "z": layer.shape[2] // 2}[axis]
    index = int(layer.slice_indices.get(axis, default_index))
    limit = {"inline": layer.shape[0], "xline": layer.shape[1], "z": layer.shape[2]}[axis]
    return int(np.clip(index, 0, limit - 1))


def _matplotlib_cmap_name(cmap: str) -> str:
    aliases = {
        "Petrel": "seismic",
        "petrel": "seismic",
    }
    name = aliases.get(cmap, cmap)
    if colormaps is not None and name not in colormaps:
        return "gray"
    return name


def _quad(points: np.ndarray) -> pv.PolyData:
    mesh = pv.PolyData(points, np.asarray([4, 0, 1, 2, 3]))
    mesh.active_texture_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    return mesh
