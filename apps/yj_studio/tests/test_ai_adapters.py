from __future__ import annotations

import numpy as np

from yj_studio.ai.adapters import (
    build_mask_layer,
    decode_sam3_masks,
    slice_to_rgb_image,
    stretch_to_uint8,
)
from yj_studio.scene.layers import MaskLayer


def test_stretch_to_uint8_robust_to_outliers() -> None:
    arr = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 100.0, 5.0],
        ],
        dtype=np.float32,
    )
    out, finite = stretch_to_uint8(arr, percentile=(2.0, 98.0))
    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert finite.all()
    # The outlier pulls clipping but doesn't escape [0, 255].
    assert out.min() == 0
    assert out.max() == 255


def test_stretch_to_uint8_all_nan_returns_zeros() -> None:
    arr = np.full((4, 5), np.nan, dtype=np.float32)
    out, finite = stretch_to_uint8(arr)
    assert out.shape == arr.shape
    assert not finite.any()
    assert (out == 0).all()


def test_slice_to_rgb_image_shape_and_dtype() -> None:
    arr = np.linspace(0.0, 1.0, num=12, dtype=np.float32).reshape(3, 4)
    rgb = slice_to_rgb_image(arr)
    assert rgb.shape == (3, 4, 3)
    assert rgb.dtype == np.uint8


def test_decode_sam3_masks_from_numpy_state() -> None:
    state = {
        "masks": np.array([[[[True, False], [False, True]]]], dtype=bool),
        "scores": np.array([0.87], dtype=np.float32),
        "boxes": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    }
    detections = decode_sam3_masks(state)
    assert len(detections) == 1
    assert detections[0]["mask"].shape == (2, 2)
    assert detections[0]["mask"][0, 0]
    assert detections[0]["score"] == 0.87
    assert detections[0]["box"] == (1.0, 2.0, 3.0, 4.0)


def test_build_mask_layer_attaches_provenance_and_score() -> None:
    layer = build_mask_layer(
        np.array([[True, False], [False, True]]),
        name="seg",
        axis="inline",
        slice_index=100,
        score=0.42,
    )
    assert isinstance(layer, MaskLayer)
    assert layer.axis == "inline"
    assert layer.slice_index == 100
    assert layer.confidence == 0.42
    assert layer.provenance["source"] == "ai.sam3"
    assert layer.metadata["score"] == 0.42
