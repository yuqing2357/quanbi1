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
        mask_slice = self._display_mask_slice(layer, axis, index)
        slice_image = build_slice_image(
            raw_slice,
            layer.shape,
            axis,
            index,
            layer.clim,
            _display_cmap(layer),
            roi=roi,
            display_mask=mask_slice,
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

    def _display_mask_slice(
        self, layer: VolumeLayer, axis: SliceAxis, index: int
    ) -> np.ndarray | None:
        mask_volume_id = _model_mask_volume_id(layer.volume_id)
        if mask_volume_id is None or mask_volume_id not in self._volume_store.volume_ids:
            return None
        try:
            mask_slice = self._volume_store.get_slice(mask_volume_id, axis, index)
        except (KeyError, IndexError, ValueError):
            return None
        return np.isfinite(mask_slice)

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
    display_mask: np.ndarray | None = None,
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
        mask = _prepare_display_mask(display_mask)
        values = values[k0 : k1 + 1, j0 : j1 + 1]
        if mask is not None:
            mask = mask[k0 : k1 + 1, j0 : j1 + 1]
        # Spatial quad places shallow (k0) at the top after display_z flip, but
        # VTK's texture v-axis maps v=0 to the LAST image row; without flipping
        # the texture along sample axis the seismic image renders upside-down
        # relative to horizons/wells/faults. Flip rows here to match the quad.
        values = values[::-1, :]
        if mask is not None:
            mask = mask[::-1, :]
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
        mask = _prepare_display_mask(display_mask)
        values = values[k0 : k1 + 1, i0 : i1 + 1]
        if mask is not None:
            mask = mask[k0 : k1 + 1, i0 : i1 + 1]
        values = values[::-1, :]  # match flipped display Z, see inline branch
        if mask is not None:
            mask = mask[::-1, :]
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
        mask = _prepare_display_mask(display_mask)
        values = values[j0 : j1 + 1, i0 : i1 + 1]
        if mask is not None:
            mask = mask[j0 : j1 + 1, i0 : i1 + 1]
        z_pos = display_z(float(index), nz)
        points = np.asarray(
            [[i0, j0, z_pos], [i1, j0, z_pos], [i1, j1, z_pos], [i0, j1, z_pos]],
            dtype=np.float32,
        )
    return SliceImage(image=colorize_slice(values, clim, cmap, display_mask=mask), points=points)


def _prepare_display_mask(display_mask: np.ndarray | None) -> np.ndarray | None:
    if display_mask is None:
        return None
    # Match the value orientation created from raw_slice in build_slice_image().
    return np.asarray(display_mask, dtype=bool).T


def _model_mask_volume_id(volume_id: str) -> str | None:
    if volume_id == "model_lithology":
        return "model_porosity"
    if volume_id == "model_porosity":
        return volume_id
    return None


def _slice_within_roi(axis: SliceAxis, index: int, roi: tuple[int, int, int, int, int, int]) -> bool:
    i0, i1, j0, j1, k0, k1 = roi
    if axis == "inline":
        return i0 <= index <= i1
    if axis == "xline":
        return j0 <= index <= j1
    return k0 <= index <= k1


def colorize_slice(
    values: np.ndarray,
    clim: tuple[float, float] | None,
    cmap: str,
    display_mask: np.ndarray | None = None,
) -> np.ndarray:
    if _is_lithology_cmap(cmap):
        return _colorize_lithology(values, display_mask=display_mask)

    finite = np.isfinite(values)
    visible = finite
    use_alpha = display_mask is not None
    if display_mask is not None:
        mask = np.asarray(display_mask, dtype=bool)
        if mask.shape == values.shape:
            visible = finite & mask
        else:
            use_alpha = False
    if clim is None:
        if np.any(visible):
            vmin, vmax = np.percentile(values[visible], [2.0, 98.0])
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = clim
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[visible] = np.clip(
        (values[visible] - float(vmin)) / (float(vmax) - float(vmin)),
        0.0,
        1.0,
    )

    cmap_name = _matplotlib_cmap_name(cmap)
    if colormaps is None:
        rgb = np.repeat((normalized * 255.0).astype(np.uint8)[..., None], 3, axis=2)
        rgb[~visible] = 0
        if use_alpha:
            alpha = np.zeros(values.shape, dtype=np.uint8)
            alpha[visible] = 255
            return np.dstack((rgb, alpha))
        return rgb

    rgba = colormaps[cmap_name](normalized)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~visible] = 0
    if use_alpha:
        alpha = np.zeros(values.shape, dtype=np.uint8)
        alpha[visible] = 255
        return np.dstack((rgb, alpha))
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


def _display_cmap(layer: VolumeLayer) -> str:
    if layer.volume_id == "model_lithology":
        return "lithology_binary"
    return layer.cmap


def _is_lithology_cmap(cmap: str) -> bool:
    return str(cmap).strip().lower() in {"lithology", "lithology_binary", "yj_lithology"}


def _colorize_lithology(
    values: np.ndarray,
    *,
    display_mask: np.ndarray | None = None,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    visible = np.isfinite(data)
    use_alpha = False
    if display_mask is not None:
        mask = np.asarray(display_mask, dtype=bool)
        if mask.shape == data.shape:
            visible &= mask
            use_alpha = True

    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    class_values = np.zeros(data.shape, dtype=np.int16)
    class_values[visible] = np.rint(data[visible]).astype(np.int16, copy=False)
    rgb[visible & (class_values == 0)] = (47, 47, 47)
    rgb[visible & (class_values >= 1)] = (255, 221, 0)
    if not use_alpha:
        return rgb
    alpha = np.zeros(data.shape, dtype=np.uint8)
    alpha[visible] = 255
    return np.dstack((rgb, alpha))


def _quad(points: np.ndarray) -> pv.PolyData:
    mesh = pv.PolyData(points, np.asarray([4, 0, 1, 2, 3]))
    mesh.active_texture_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    return mesh
