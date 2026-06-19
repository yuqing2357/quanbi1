from yj_studio.scene.layers import VolumeLayer
import numpy as np

from yj_studio.view.view_2d_section import (
    _lithology_rgba,
    _mask_centroid_data_coords,
    _mask_rgba,
    _slice_sync_index_for_layer,
    _volume_interpolation,
)


def test_reservoir_model_sections_use_nearest_interpolation():
    lithology = VolumeLayer(name="岩性模型", volume_id="model_lithology")
    porosity = VolumeLayer(name="孔隙度模型", volume_id="model_porosity")

    assert _volume_interpolation(lithology) == "nearest"
    assert _volume_interpolation(porosity) == "nearest"


def test_seismic_sections_keep_antialiased_display():
    seismic = VolumeLayer(name="地震体数据", volume_id="seismic")

    assert _volume_interpolation(seismic) == "antialiased"


def test_lithology_rgba_maps_invalid_zero_and_one():
    values = np.asarray([[np.nan, 0.0, 1.0]], dtype=np.float32)

    rgba = _lithology_rgba(values)

    assert rgba[0, 0].tolist() == [0, 0, 0, 255]
    assert rgba[0, 1].tolist() == [47, 47, 47, 255]
    assert rgba[0, 2].tolist() == [255, 221, 0, 255]


def test_lithology_rgba_uses_display_mask_for_black_na_area():
    values = np.asarray([[0.0, 1.0]], dtype=np.float32)
    display_mask = np.asarray([[False, True]])

    rgba = _lithology_rgba(values, display_mask)

    assert rgba[0, 0].tolist() == [0, 0, 0, 255]
    assert rgba[0, 1].tolist() == [255, 221, 0, 255]


def test_mask_rgba_keeps_top_sample_at_top_of_image():
    layer_mask = np.zeros((4, 5), dtype=bool)
    layer_mask[:, 0] = True

    rgba = _mask_rgba(layer_mask, (1.0, 0.0, 0.0, 0.5), 0.5)

    assert rgba.shape == (5, 4, 4)
    assert np.all(rgba[0, :, 3] == 0.5)
    assert not np.any(rgba[1:, :, 3])


def test_mask_centroid_uses_imshow_upper_origin_extent():
    image_mask = np.zeros((5, 4), dtype=bool)
    image_mask[0, 2] = True

    assert _mask_centroid_data_coords(image_mask, (0, 3, 4, 0)) == (2.0, 0.0)

    image_mask[:] = False
    image_mask[4, 1] = True

    assert _mask_centroid_data_coords(image_mask, (0, 3, 4, 0)) == (1.0, 4.0)


def test_slice_sync_ignores_reservoir_index_for_seismic_layer():
    seismic = VolumeLayer(name="地震体数据", volume_id="seismic", shape=(1684, 1451, 1201))
    value = {"index": 2226, "volume_id": "model_lithology", "volume_layer_id": "lith"}

    assert (
        _slice_sync_index_for_layer(
            value,
            seismic,
            "inline",
            volume_layer_id="seismic-layer",
        )
        is None
    )


def test_slice_sync_accepts_matching_in_bounds_payload():
    lithology = VolumeLayer(
        name="岩性模型",
        volume_id="model_lithology",
        shape=(4452, 2796, 1443),
    )
    value = {"index": 2226, "volume_id": "model_lithology", "volume_layer_id": "lith"}

    assert (
        _slice_sync_index_for_layer(
            value,
            lithology,
            "inline",
            volume_layer_id="lith",
        )
        == 2226
    )


def test_slice_sync_maps_reservoir_index_to_seismic_index():
    seismic = VolumeLayer(
        name="地震体数据",
        volume_id="seismic",
        shape=(1684, 1451, 1201),
        metadata={
            "grid_reference": {
                "seismic_index_origin": {"axis0": 0, "axis1": 0, "sample": 0},
                "scale_axis0_axis1_sample": [1, 1, 1],
            }
        },
    )
    value = {
        "index": 1485,
        "volume_id": "model_lithology",
        "volume_layer_id": "lith",
        "seismic_index": 946.5,
    }

    assert _slice_sync_index_for_layer(
        value,
        seismic,
        "inline",
        volume_layer_id="seismic-layer",
    ) == 946


def test_slice_sync_maps_seismic_index_to_reservoir_index():
    lithology = VolumeLayer(
        name="岩性模型",
        volume_id="model_lithology",
        shape=(2959, 2201, 2826),
        metadata={
            "grid_reference": {
                "seismic_index_origin": {"axis0": 204, "axis1": 0, "sample": 88},
                "scale_axis0_axis1_sample": [2, 2, 5],
            }
        },
    )
    value = {
        "index": 946,
        "volume_id": "seismic",
        "volume_layer_id": "seismic-layer",
        "seismic_index": 946.0,
    }

    assert _slice_sync_index_for_layer(
        value,
        lithology,
        "inline",
        volume_layer_id="lith",
    ) == 1484


def test_legacy_slice_sync_int_still_checks_layer_bounds():
    seismic = VolumeLayer(name="地震体数据", volume_id="seismic", shape=(1684, 1451, 1201))

    assert (
        _slice_sync_index_for_layer(2226, seismic, "inline", volume_layer_id="seismic-layer")
        is None
    )
    assert _slice_sync_index_for_layer(100, seismic, "inline", volume_layer_id="seismic-layer") == 100
