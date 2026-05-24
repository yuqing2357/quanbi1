from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yj_studio.config.defaults import DEFAULT_Z_WINDOW_START
from yj_studio.io.readers.well_coordinates import WellCoordinate, load_well_coordinates
from yj_studio.io.readers.well_logs import WellDepthRange, resolve_well_depth_range


@dataclass(frozen=True, slots=True)
class WellRecord:
    name: str
    head_position: tuple[float, float, float]
    trajectory: np.ndarray
    matched_csv_name: str | None
    metadata: dict[str, str]


class WellRepository:
    """Build simple vertical well trajectories from aligned well-head coordinates."""

    def __init__(self, records: list[WellRecord]) -> None:
        self._records = records
        self._by_name = {record.name: record for record in records}

    @classmethod
    def from_coordinates_csv(
        cls,
        coords_csv: Path,
        *,
        z_count: int,
        inside_only: bool = True,
        log_roots: list[Path] | None = None,
        z_window_start: float = DEFAULT_Z_WINDOW_START,
    ) -> "WellRepository":
        """Create well records using the same DEPT-to-sample mapping as the data bundle.

        For the full-depth YJ volume, the processed-data summary states:
        sample_index = DEPT / 10.0. That is z_window_start=0.0.
        """

        coordinates = load_well_coordinates(coords_csv, inside_only=inside_only)
        records: list[WellRecord] = []
        for item in coordinates:
            depth_range = None
            if log_roots is not None and item.matched_csv_name:
                depth_range = resolve_well_depth_range(
                    item.matched_csv_name,
                    log_roots,
                    z_count=z_count,
                    z_window_start=z_window_start,
                )
                if depth_range is None:
                    continue
            records.append(
                _record_from_coordinate(item, z_count=z_count, depth_range=depth_range)
            )
        return cls(records)

    def iter_records(self) -> tuple[WellRecord, ...]:
        return tuple(self._records)

    def get(self, name: str) -> WellRecord:
        return self._by_name[name]

    def __len__(self) -> int:
        return len(self._records)


def _record_from_coordinate(
    coordinate: WellCoordinate,
    *,
    z_count: int,
    depth_range: WellDepthRange | None,
) -> WellRecord:
    x = float(coordinate.inline_index)
    y = float(coordinate.xline_index)
    if depth_range is None:
        z_top = 0.0
        z_bottom = float(max(0, z_count - 1))
        depth_metadata = {"depth_source": "fallback_full_window", "depth_sample_count": "0"}
    else:
        z_top = float(max(0.0, min(float(z_count - 1), depth_range.min_sample)))
        z_bottom = float(max(0.0, min(float(z_count - 1), depth_range.max_sample)))
        depth_metadata = {
            "depth_source": "|".join(str(path) for path in depth_range.source_paths),
            "depth_sample_count": str(depth_range.sample_count),
            "z_top": f"{z_top:.3f}",
            "z_bottom": f"{z_bottom:.3f}",
        }
    trajectory = np.asarray([[x, y, z_top], [x, y, z_bottom]], dtype=np.float32)
    metadata = {k: str(v) for k, v in coordinate.metadata.items()}
    metadata.update(depth_metadata)
    return WellRecord(
        name=coordinate.name,
        head_position=(x, y, z_top),
        trajectory=trajectory,
        matched_csv_name=coordinate.matched_csv_name,
        metadata=metadata,
    )
