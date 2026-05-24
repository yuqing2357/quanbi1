from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WellCoordinate:
    name: str
    inline_index: float
    xline_index: float
    inline_number: int | None
    xline_number: int | None
    matched_csv_name: str | None
    metadata: dict[str, Any]


def load_well_coordinates(coords_csv: Path, *, inside_only: bool = True) -> list[WellCoordinate]:
    """Load well head coordinates already aligned to the YJ seismic index grid."""

    if not coords_csv.exists():
        raise FileNotFoundError(coords_csv)

    wells: list[WellCoordinate] = []
    with coords_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if inside_only and _inside_flag(row) != "True":
                continue
            name = (row.get("chosen_wellbore") or row.get("normalized_name") or "").strip()
            if not name:
                continue
            inline_text = row.get("inline_index_float") or row.get("inline_index")
            xline_text = row.get("crossline_index_float") or row.get("crossline_index")
            if not inline_text or not xline_text:
                continue
            wells.append(
                WellCoordinate(
                    name=name,
                    inline_index=float(inline_text),
                    xline_index=float(xline_text),
                    inline_number=_optional_int(row.get("inline_number")),
                    xline_number=_optional_int(row.get("crossline_number")),
                    matched_csv_name=(row.get("matched_wells_csv_name") or "").strip() or None,
                    metadata=dict(row),
                )
            )
    return wells


def _inside_flag(row: dict[str, str]) -> str | None:
    return row.get("inside_new_seismic_depth_0_654") or row.get("inside_current_window_150_654")


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None

