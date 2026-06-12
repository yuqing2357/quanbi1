from __future__ import annotations

from io import StringIO
from pathlib import Path

from yj_studio.data import WellRepository
from yj_studio.io.readers.well_logs import WellDepthRange


class FakeCsvPath:
    def exists(self) -> bool:
        return True

    def open(self, *_args, **_kwargs):
        return StringIO(
            "chosen_wellbore,inline_index_float,crossline_index_float,"
            "matched_wells_csv_name,inside_new_seismic_depth_0_654\n"
            "well_a,10.0,20.0,well_a,True\n"
        )


def test_well_repository_builds_vertical_trajectory() -> None:
    repository = WellRepository.from_coordinates_csv(FakeCsvPath(), z_count=654)  # type: ignore[arg-type]
    record = repository.get("well_a")

    assert len(repository) == 1
    assert record.head_position == (10.0, 20.0, 0.0)
    assert record.trajectory.tolist() == [[10.0, 20.0, 0.0], [10.0, 20.0, 653.0]]


def test_well_repository_uses_real_log_depth_range(monkeypatch) -> None:
    def fake_resolve(_csv_name, _log_roots, *, z_count, z_window_start):
        assert z_count == 654
        assert z_window_start == 0.0
        return WellDepthRange(
            min_sample=150.0,
            max_sample=321.5,
            sample_count=42,
            source_paths=(Path("por/well_a.csv"),),
        )

    monkeypatch.setattr("yj_studio.data.well_repository.resolve_well_depth_range", fake_resolve)

    repository = WellRepository.from_coordinates_csv(
        FakeCsvPath(),  # type: ignore[arg-type]
        z_count=654,
        log_roots=[Path("por")],
    )
    record = repository.get("well_a")

    assert record.head_position == (10.0, 20.0, 150.0)
    assert record.trajectory.tolist() == [[10.0, 20.0, 150.0], [10.0, 20.0, 321.5]]
    assert record.metadata["depth_sample_count"] == "42"


def test_well_repository_can_use_current_window_offset(monkeypatch) -> None:
    def fake_resolve(_csv_name, _log_roots, *, z_count, z_window_start):
        assert z_window_start == 150.0
        return WellDepthRange(
            min_sample=0.0,
            max_sample=171.5,
            sample_count=42,
            source_paths=(Path("por/well_a.csv"),),
        )

    monkeypatch.setattr("yj_studio.data.well_repository.resolve_well_depth_range", fake_resolve)

    repository = WellRepository.from_coordinates_csv(
        FakeCsvPath(),  # type: ignore[arg-type]
        z_count=504,
        log_roots=[Path("por")],
        z_window_start=150.0,
    )

    assert repository.get("well_a").trajectory.tolist() == [[10.0, 20.0, 0.0], [10.0, 20.0, 171.5]]


def test_well_repository_skips_wells_without_real_depth(monkeypatch) -> None:
    monkeypatch.setattr(
        "yj_studio.data.well_repository.resolve_well_depth_range",
        lambda *_args, **_kwargs: None,
    )

    repository = WellRepository.from_coordinates_csv(
        FakeCsvPath(),  # type: ignore[arg-type]
        z_count=654,
        log_roots=[Path("por")],
    )

    assert len(repository) == 0
