from __future__ import annotations

from yj_studio.view.view_sam3_workbench import _data_bbox_to_pixel_window


def test_data_bbox_to_pixel_window_maps_inside_view() -> None:
    window = _data_bbox_to_pixel_window(
        frame_bbox=(0.0, 100.0, 0.0, 200.0),
        view_bbox=(25.0, 75.0, 50.0, 150.0),
        image_shape=(200, 100),
    )
    assert window == (25.0, 75.0, 50.0, 150.0)


def test_data_bbox_to_pixel_window_clamps_to_image() -> None:
    window = _data_bbox_to_pixel_window(
        frame_bbox=(0.0, 100.0, 0.0, 200.0),
        view_bbox=(-10.0, 110.0, -20.0, 220.0),
        image_shape=(200, 100),
    )
    assert window == (0.0, 100.0, 0.0, 200.0)


def test_data_bbox_to_pixel_window_rejects_degenerate_views() -> None:
    window = _data_bbox_to_pixel_window(
        frame_bbox=(0.0, 100.0, 0.0, 200.0),
        view_bbox=(10.0, 10.5, 20.0, 21.0),
        image_shape=(200, 100),
    )
    assert window is None
