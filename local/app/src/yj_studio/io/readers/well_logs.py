from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yj_studio.config.defaults import DEFAULT_Z_WINDOW_START, DEPTH_STEP_TO_SAMPLE


@dataclass(frozen=True, slots=True)
class WellDepthRange:
    min_sample: float
    max_sample: float
    sample_count: int
    source_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class WellLogSamples:
    samples: np.ndarray
    source_path: Path
    value_column: str


def load_depth_samples(
    log_path: Path,
    *,
    z_count: int,
    z_window_start: float = DEFAULT_Z_WINDOW_START,
    depth_step_to_sample: float = DEPTH_STEP_TO_SAMPLE,
) -> np.ndarray:
    """Load valid sample indices from the DEPT column in a well-log CSV."""

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    samples: list[float] = []
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "DEPT" not in reader.fieldnames:
            return np.asarray([], dtype=np.float32)
        for record in reader:
            depth_text = (record.get("DEPT") or "").strip()
            if not depth_text:
                continue
            try:
                depth_m = float(depth_text)
            except ValueError:
                continue
            sample = depth_m / depth_step_to_sample - z_window_start
            if 0.0 <= sample < float(z_count):
                samples.append(float(sample))
    return np.asarray(samples, dtype=np.float32)


def load_log_samples(
    log_path: Path,
    *,
    inline_index: float,
    xline_index: float,
    value_column: str,
    z_count: int,
    z_window_start: float = DEFAULT_Z_WINDOW_START,
    depth_step_to_sample: float = DEPTH_STEP_TO_SAMPLE,
) -> WellLogSamples:
    """Load log samples as [inline, xline, sample, value] rows."""

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    rows: list[list[float]] = []
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "DEPT" not in reader.fieldnames:
            return WellLogSamples(np.empty((0, 4), dtype=np.float32), log_path, value_column)
        if value_column not in reader.fieldnames:
            return WellLogSamples(np.empty((0, 4), dtype=np.float32), log_path, value_column)
        for record in reader:
            depth_text = (record.get("DEPT") or "").strip()
            value_text = (record.get(value_column) or "").strip()
            if not depth_text or not value_text:
                continue
            try:
                depth_m = float(depth_text)
                value = float(value_text)
            except ValueError:
                continue
            sample = depth_m / depth_step_to_sample - z_window_start
            if 0.0 <= sample < float(z_count):
                rows.append([float(inline_index), float(xline_index), float(sample), value])
    return WellLogSamples(np.asarray(rows, dtype=np.float32), log_path, value_column)


def resolve_well_depth_range(
    csv_name: str,
    log_roots: list[Path],
    *,
    z_count: int,
    z_window_start: float = DEFAULT_Z_WINDOW_START,
    depth_step_to_sample: float = DEPTH_STEP_TO_SAMPLE,
) -> WellDepthRange | None:
    """Find the union of valid depth samples for one well across log directories."""

    if not csv_name:
        return None

    sample_arrays: list[np.ndarray] = []
    source_paths: list[Path] = []
    for root in log_roots:
        path = root / f"{csv_name}.csv"
        if not path.exists():
            continue
        samples = load_depth_samples(
            path,
            z_count=z_count,
            z_window_start=z_window_start,
            depth_step_to_sample=depth_step_to_sample,
        )
        if samples.size == 0:
            continue
        sample_arrays.append(samples)
        source_paths.append(path)

    if not sample_arrays:
        return None
    merged = np.concatenate(sample_arrays)
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return None
    return WellDepthRange(
        min_sample=float(np.nanmin(finite)),
        max_sample=float(np.nanmax(finite)),
        sample_count=int(finite.size),
        source_paths=tuple(source_paths),
    )
