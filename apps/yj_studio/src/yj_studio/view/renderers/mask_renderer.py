from __future__ import annotations

import numpy as np
import pyvista as pv

from yj_studio.scene.layers import MaskLayer
from yj_studio.view.display_coordinates import display_z, layer_z_count
from yj_studio.view.highlight import highlight_color, highlight_opacity


class MaskRenderer:
    """Render 2D mask layers as textured planes.

    Honours :attr:`MaskLayer.confidence`: where confidence falls below the
    layer's ``low_confidence_threshold`` metadata (default 0.5), the mask is
    drawn with a desaturated outline-only style so the user can tell SAM3
    is uncertain about that region. Confidence may be either a scalar (one
    value for the whole mask) or a 2D array matching ``mask.shape``.
    """

    def __init__(self, plotter) -> None:
        self._plotter = plotter
        self._actor_names: dict[str, str] = {}

    def render(
        self,
        layer: MaskLayer,
        *,
        highlighted: bool = False,
        z_count: int | None = None,
    ) -> None:
        actor_name = self._actor_name(layer)
        if not layer.visible or layer.mask is None:
            self.clear(layer.id)
            return
        mask = np.asarray(layer.mask)
        if mask.ndim != 2:
            # 3D masks (e.g. SAM3 propagation output) are not yet handled by
            # this renderer; skip silently — the LayerTree still lists them
            # and downstream algorithms can still consume them.
            self.clear(layer.id)
            return
        z_count_for_axis = layer_z_count(layer.metadata) or z_count
        mesh = build_mask_mesh(layer, z_count=z_count_for_axis)
        if mesh.n_cells == 0:
            return
        texture = pv.numpy_to_texture(build_mask_texture(layer))
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            texture=texture,
            opacity=highlight_opacity(layer.opacity, highlighted),
            color=highlight_color(layer.color, highlighted),
            lighting=False,
            pickable=True,
        )
        self._plotter.render()

    def clear(self, layer_id: str) -> None:
        actor_name = self._actor_names.get(layer_id, f"mask-{layer_id}")
        self._plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._plotter.render()

    def _actor_name(self, layer: MaskLayer) -> str:
        actor_name = f"mask-{layer.id}"
        self._actor_names[layer.id] = actor_name
        return actor_name


def build_mask_mesh(layer: MaskLayer, *, z_count: int | None = None) -> pv.PolyData:
    """Quad covering the slice the mask lives on, with the same display_z
    flip used by the volume slice renderer so masks line up with horizons.
    """

    mask = np.asarray(layer.mask)
    if mask.ndim != 2 or mask.size == 0:
        return pv.PolyData()
    rows, cols = mask.shape
    axis = layer.axis or "z"
    index = float(layer.slice_index or 0)
    if axis == "inline":
        # rows -> xline, cols -> z. Apply display_z to z so the texture
        # rests on top of the matching seismic inline slice.
        z0 = float(display_z(0.0, z_count))
        z1 = float(display_z(float(cols - 1), z_count))
        points = np.asarray(
            [
                [index, 0, z0],
                [index, rows - 1, z0],
                [index, rows - 1, z1],
                [index, 0, z1],
            ],
            dtype=np.float32,
        )
    elif axis == "xline":
        z0 = float(display_z(0.0, z_count))
        z1 = float(display_z(float(cols - 1), z_count))
        points = np.asarray(
            [
                [0, index, z0],
                [rows - 1, index, z0],
                [rows - 1, index, z1],
                [0, index, z1],
            ],
            dtype=np.float32,
        )
    else:
        z = float(display_z(index, z_count))
        points = np.asarray(
            [
                [0, 0, z],
                [rows - 1, 0, z],
                [rows - 1, cols - 1, z],
                [0, cols - 1, z],
            ],
            dtype=np.float32,
        )
    mesh = pv.PolyData(points, np.asarray([4, 0, 1, 2, 3]))
    mesh.active_texture_coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    return mesh


def build_mask_texture(layer: MaskLayer) -> np.ndarray:
    """Encode the mask + confidence overlay into an RGBA texture.

    Three pixel categories:

    1. ``mask == 0``                              -> fully transparent
    2. ``mask == 1`` and confidence >= threshold  -> layer colour, opaque
    3. ``mask == 1`` and confidence  < threshold  -> red-tinted, half alpha,
       so the user immediately sees which regions SAM3 was unsure about.

    The orientation matches :func:`build_mask_mesh`: the slice is flipped
    along the z-axis to match the display_z convention applied to the
    volume slice renderer (see issue with seismic z being upside down).
    """

    mask = np.asarray(layer.mask, dtype=bool)
    if mask.ndim != 2:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    rows, cols = mask.shape

    threshold = float(layer.metadata.get("low_confidence_threshold", 0.5))
    confidence = layer.confidence
    if isinstance(confidence, np.ndarray) and confidence.shape == mask.shape:
        conf_map = confidence.astype(np.float32)
    else:
        scalar = float(confidence) if isinstance(confidence, (int, float)) else 1.0
        conf_map = np.full(mask.shape, scalar, dtype=np.float32)

    color = np.asarray(layer.color[:3], dtype=np.float32)
    low_conf_color = np.asarray([1.0, 0.2, 0.2], dtype=np.float32)

    rgba = np.zeros((rows, cols, 4), dtype=np.float32)
    high = mask & (conf_map >= threshold)
    low = mask & (conf_map < threshold)
    rgba[high, :3] = color
    rgba[high, 3] = float(layer.opacity)
    rgba[low, :3] = low_conf_color
    rgba[low, 3] = float(layer.opacity) * 0.5

    # Flip along axis 1 (the "z" axis in inline / xline orientation, the
    # "xline" axis for z-slice; see build_mask_mesh layout). For z-slice
    # the second axis is not the depth axis so flipping has no semantic
    # effect on alignment but keeps the texture/quad mapping consistent.
    rgba = rgba[:, ::-1, :]
    return (np.clip(rgba * 255.0, 0.0, 255.0)).astype(np.uint8)
