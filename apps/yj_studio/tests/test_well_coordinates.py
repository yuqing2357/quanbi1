from __future__ import annotations

from io import StringIO
from pathlib import Path

from yj_studio.io.readers.well_coordinates import load_well_coordinates


class FakeCsvPath:
    def __init__(self, text: str) -> None:
        self.text = text

    def exists(self) -> bool:
        return True

    def open(self, *_args, **_kwargs):
        return StringIO(self.text)


def test_load_well_coordinates_filters_inside_rows() -> None:
    text = (
        "normalized_name,chosen_wellbore,inline_index_float,crossline_index_float,"
        "inline_number,crossline_number,matched_wells_csv_name,inside_new_seismic_depth_0_654\n"
        "a,well_a,10.5,20.25,100,200,well_a,True\n"
        "b,well_b,11.5,21.25,101,201,well_b,False\n"
    )

    wells = load_well_coordinates(FakeCsvPath(text))  # type: ignore[arg-type]

    assert len(wells) == 1
    assert wells[0].name == "well_a"
    assert wells[0].inline_index == 10.5
    assert wells[0].xline_index == 20.25
    assert wells[0].inline_number == 100
    assert wells[0].matched_csv_name == "well_a"


def test_missing_well_coordinate_file_raises() -> None:
    try:
        load_well_coordinates(Path("missing.csv"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")

