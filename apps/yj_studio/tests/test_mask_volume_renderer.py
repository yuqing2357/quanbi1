from __future__ import annotations

import numpy as np

from yj_studio.scene.layers import MaskLayer
from yj_studio.view.renderers.mask_renderer import build_mask_volume_mesh


def test_build_mask_volume_mesh_inline_maps_image_order_to_world() -> None:
    mask = np.zeros((2, 3, 4), dtype=np.uint8)  # frame, sample_z, xline
    mask[1, 2, 3] = 1
    layer = MaskLayer(
        name="T1",
        mask=mask,
        axis="inline",
        slice_index=10,
        metadata={"mask3d_index_lo": 10},
    )

    mesh = build_mask_volume_mesh(layer, z_count=6)
    bounds = mesh.bounds

    assert mesh.n_cells > 0
    assert bounds[0] <= 11.0 <= bounds[1]  # inline index_lo + frame
    assert bounds[2] <= 3.0 <= bounds[3]   # xline trace
    assert bounds[4] <= 3.0 <= bounds[5]   # display_z around sample 2 in z_count 6


def test_build_mask_volume_mesh_crossline_maps_image_order_to_world() -> None:
    mask = np.zeros((2, 3, 4), dtype=np.uint8)  # frame, sample_z, inline
    mask[1, 2, 3] = 1
    layer = MaskLayer(
        name="T1",
        mask=mask,
        axis="xline",
        slice_index=20,
        metadata={"mask3d_index_lo": 20},
    )

    mesh = build_mask_volume_mesh(layer, z_count=6)
    bounds = mesh.bounds

    assert mesh.n_cells > 0
    assert bounds[0] <= 3.0 <= bounds[1]    # inline trace
    assert bounds[2] <= 21.0 <= bounds[3]   # xline index_lo + frame
    assert bounds[4] <= 3.0 <= bounds[5]


def test_build_mask_volume_mesh_timeslice_maps_image_order_to_world() -> None:
    mask = np.zeros((2, 3, 4), dtype=np.uint8)  # frame, xline, inline
    mask[1, 2, 3] = 1
    layer = MaskLayer(
        name="T1",
        mask=mask,
        axis="z",
        slice_index=30,
        metadata={"mask3d_index_lo": 30},
    )

    mesh = build_mask_volume_mesh(layer, z_count=40)
    bounds = mesh.bounds

    assert mesh.n_cells > 0
    assert bounds[0] <= 3.0 <= bounds[1]    # inline
    assert bounds[2] <= 2.0 <= bounds[3]    # xline
    assert bounds[4] <= 8.0 <= bounds[5]    # display_z around sample 31 in z_count 40
