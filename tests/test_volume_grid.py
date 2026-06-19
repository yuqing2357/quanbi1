from yj_studio_core.volume_grid import (
    grid_reference_from_mapping,
    local_to_seismic_index,
    seismic_to_local_index,
)


REFERENCE = {
    "seismic_index_origin": {"axis0": 204, "axis1": 0, "sample": 88},
    "scale_axis0_axis1_sample": [2, 2, 5],
}


def test_grid_reference_preserves_crop_origin_and_scale():
    reference = grid_reference_from_mapping(REFERENCE)

    assert reference["seismic_index_origin"] == {"axis0": 204.0, "axis1": 0.0, "sample": 88.0}
    assert reference["scale_axis0_axis1_sample"] == [2.0, 2.0, 5.0]


def test_local_and_seismic_indices_round_trip_on_grid_nodes():
    assert local_to_seismic_index(REFERENCE, "inline", 1484) == 946.0
    assert seismic_to_local_index(REFERENCE, "inline", 946.0) == 1484
    assert local_to_seismic_index(REFERENCE, "z", 1492) == 386.4
