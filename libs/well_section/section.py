from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Iterable
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
import plotly.graph_objects as go
from skimage import filters, measure, morphology

Z_WINDOW_START = 0.0
DEPTH_STEP_TO_SAMPLE = 10.0
SEISMIC_EVENT_MAX_LINES_PER_POLARITY = 35
SEISMIC_EVENT_MIN_LENGTH = 24
SEISMIC_EVENT_MAX_SLOPE = 3.5
SEISMIC_EVENT_VESSELNESS_PERCENTILE = 95.5
SEISMIC_EVENT_AMPLITUDE_PERCENTILE = 70.0
SEISMIC_EVENT_SATO_SIGMAS = (1, 2, 3, 4)
SEISMIC_WIGGLE_MAX_TRACES = 120
SEISMIC_WIGGLE_AMPLITUDE_FRACTION = 0.45
DL_POR_HIDDEN_SIZE = 64
DL_POR_NUM_LAYERS = 2
DL_POR_EPOCHS = 100
DL_LOCAL_EPOCH_MAX = 600
DL_LOCAL_REFERENCE_AREA = 12000
DL_LOCAL_LABEL_WEIGHT_MAX_SCALE = 3.0
DL_LOCAL_DROPOUT = 0.0
DL_LOCAL_WEIGHT_DECAY = 0.0
DL_OUTPUT_RESOLUTION = "fixed-depth"
DL_FIXED_GRID_STEP_M = 5.0
DL_WELL_GRID_MIN_STEP_M = 5.0
DL_WELL_GRID_MAX_STEP_M = 5.0
DL_LITH_GRID_MIN_STEP_M = 5.0
DL_LITH_GRID_MAX_STEP_M = 5.0
DL_WELL_GRID_MAX_DEPTH_POINTS = 1800
DL_FAULT_BARRIER_PERCENTILE = 88.0
DL_FAULT_BARRIER_POWER = 1.7
DL_FAULT_BARRIER_STRENGTH = 0.35
DL_FAULT_MIN_SMOOTH_WEIGHT = 0.08
DL_LABEL_CONFLICT_MIN_WEIGHT = 0.25
DL_LABEL_CONFLICT_LITH_DEPTH_WINDOW_M = 12.0
DL_LABEL_CONFLICT_CONTINUOUS_DEPTH_WINDOW_M = 18.0
DL_LABEL_CONFLICT_CONTINUOUS_SCALE_FACTOR = 1.5
DL_POR_LR = 2e-3
DL_POR_LOG_LOSS_WEIGHT = 18.0
DL_POR_SMOOTH_LOSS_WEIGHT = 0.06
DL_POR_NEIGHBOR_WEIGHT = 0.70
DL_POR_SUPERVISION_RADIUS = 2
DL_LITH_CLASS_LOSS_WEIGHT = 16.0
TG_LAYER_NAME = "tg"

LITH_COLORS = {
    0: "#8f8f8f",
    1: "#f2c84b",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#17becf",
}

POR_STYLES = ("条形", "点状", "曲线")

SEISMIC_COLOR_OPTIONS = {
    "RdBu": ("RdBu", True),
    "Greys": ("Greys", False),
    "Seismic": ("RdBu", False),
    "Viridis": ("Viridis", False),
    "Cividis": ("Cividis", False),
    "Jet": ("Jet", False),
}

SEISMIC_DISPLAY_OPTIONS = ("彩色", "波形", "彩色+波形")

_LOCAL_DL_BACKEND: dict[str, object] = {
    "server": None,
    "thread": None,
    "url": None,
}
_LOCAL_DL_BACKEND_LOCK = threading.Lock()
_LOCAL_DL_PROGRESS: dict[str, dict[str, object]] = {}
_LOCAL_DL_PROGRESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class WellInfo:
    name: str
    csv_name: str
    inline: float
    crossline: float


@dataclass(frozen=True)
class SectionWell:
    info: WellInfo
    distance: float
    log: np.ndarray


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _sample_to_depth(sample: float) -> float:
    return (float(sample) + Z_WINDOW_START) * DEPTH_STEP_TO_SAMPLE


def _samples_to_depth(samples: np.ndarray | list[float]) -> np.ndarray:
    return (np.asarray(samples, dtype=np.float32) + np.float32(Z_WINDOW_START)) * np.float32(
        DEPTH_STEP_TO_SAMPLE
    )


def load_well_infos(coords_csv: Path, allowed_wells: Iterable[str] | None = None) -> dict[str, WellInfo]:
    allowed = set(allowed_wells) if allowed_wells is not None else None
    infos: dict[str, WellInfo] = {}
    with coords_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("chosen_wellbore") or "").strip()
            csv_name = (row.get("matched_wells_csv_name") or "").strip()
            if not name or not csv_name:
                continue
            inside_flag = row.get("inside_new_seismic_depth_0_654", row.get("inside_current_window_150_654"))
            if inside_flag != "True":
                continue
            if allowed is not None and name not in allowed:
                continue
            infos[name] = WellInfo(
                name=name,
                csv_name=csv_name,
                inline=float(row["inline_index"]),
                crossline=float(row["crossline_index"]),
            )
    return infos


def _load_log(path: Path, value_column: str, z_count: int) -> np.ndarray:
    points = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            depth_text = (row.get("DEPT") or "").strip()
            value_text = (row.get(value_column) or "").strip()
            if not depth_text or not value_text:
                continue
            try:
                depth = float(depth_text)
                value = float(value_text)
            except ValueError:
                continue
            sample = depth / DEPTH_STEP_TO_SAMPLE - Z_WINDOW_START
            if 0.0 <= sample < float(z_count):
                points.append((sample, value))
    return np.asarray(points, dtype=np.float32)


def _build_section_wells(
    selected_wells: list[str],
    coords_csv: Path,
    log_dir: Path,
    value_column: str,
    z_count: int,
) -> list[SectionWell]:
    infos = load_well_infos(coords_csv, selected_wells)
    section_wells: list[SectionWell] = []
    distance = 0.0
    previous: WellInfo | None = None
    for name in selected_wells:
        if name not in infos:
            continue
        info = infos[name]
        log_path = log_dir / f"{info.csv_name}.csv"
        if not log_path.exists():
            continue
        log = _load_log(log_path, value_column, z_count)
        if log.size == 0:
            continue
        if previous is not None:
            distance += float(
                np.hypot(info.inline - previous.inline, info.crossline - previous.crossline)
            )
        section_wells.append(SectionWell(info=info, distance=distance, log=log))
        previous = info
    return section_wells


def _build_companion_wells(
    base_wells: list[SectionWell],
    log_dir: Path,
    value_column: str,
    z_count: int,
) -> list[SectionWell]:
    section_wells: list[SectionWell] = []
    for well in base_wells:
        log_path = log_dir / f"{well.info.csv_name}.csv"
        if not log_path.exists():
            continue
        log = _load_log(log_path, value_column, z_count)
        if log.size == 0:
            continue
        section_wells.append(SectionWell(info=well.info, distance=well.distance, log=log))
    return section_wells


def _sample_layer_along_section(layer_path: Path, section_wells: list[SectionWell]) -> tuple[list[float], list[float]]:
    data = np.load(layer_path)
    sample = data["sample"].astype(np.float32)
    mask = data["mask"].astype(bool)
    xs: list[float] = []
    zs: list[float] = []
    for left, right in zip(section_wells[:-1], section_wells[1:]):
        segment_len = max(2, int(round(right.distance - left.distance)))
        for idx in range(segment_len):
            t = idx / float(segment_len)
            inline = left.info.inline * (1.0 - t) + right.info.inline * t
            crossline = left.info.crossline * (1.0 - t) + right.info.crossline * t
            ii = int(np.clip(round(inline), 0, sample.shape[0] - 1))
            jj = int(np.clip(round(crossline), 0, sample.shape[1] - 1))
            xs.append(left.distance * (1.0 - t) + right.distance * t)
            zs.append(float(sample[ii, jj]) if mask[ii, jj] else np.nan)
    xs.append(section_wells[-1].distance)
    last = section_wells[-1]
    ii = int(np.clip(round(last.info.inline), 0, sample.shape[0] - 1))
    jj = int(np.clip(round(last.info.crossline), 0, sample.shape[1] - 1))
    zs.append(float(sample[ii, jj]) if mask[ii, jj] else np.nan)
    return xs, zs


def _sample_layer_at_points(
    layer_path: Path,
    inlines: np.ndarray,
    crosslines: np.ndarray,
) -> np.ndarray:
    data = np.load(layer_path)
    sample = data["sample"].astype(np.float32)
    mask = data["mask"].astype(bool)
    out = np.full(inlines.shape, np.nan, dtype=np.float32)
    for idx, (inline, crossline) in enumerate(zip(inlines, crosslines)):
        ii = int(np.clip(round(float(inline)), 0, sample.shape[0] - 1))
        jj = int(np.clip(round(float(crossline)), 0, sample.shape[1] - 1))
        if mask[ii, jj]:
            out[idx] = float(sample[ii, jj])
    return out


def _section_sample_points(section_wells: list[SectionWell], max_points: int = 900) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances: list[float] = []
    inlines: list[float] = []
    crosslines: list[float] = []
    total = max(section_wells[-1].distance, 1.0)
    for left, right in zip(section_wells[:-1], section_wells[1:]):
        segment_distance = max(right.distance - left.distance, 1.0)
        count = max(2, int(round(max_points * segment_distance / total)))
        for idx in range(count):
            t = idx / float(count)
            distances.append(left.distance * (1.0 - t) + right.distance * t)
            inlines.append(left.info.inline * (1.0 - t) + right.info.inline * t)
            crosslines.append(left.info.crossline * (1.0 - t) + right.info.crossline * t)
    last = section_wells[-1]
    distances.append(last.distance)
    inlines.append(last.info.inline)
    crosslines.append(last.info.crossline)
    return (
        np.asarray(distances, dtype=np.float32),
        np.asarray(inlines, dtype=np.float32),
        np.asarray(crosslines, dtype=np.float32),
    )


def _sample_seismic_section_with_geometry(
    seismic_path: Path,
    section_wells: list[SectionWell],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seismic = np.load(seismic_path, mmap_mode="r")
    distances, inlines, crosslines = _section_sample_points(section_wells)
    ii = np.clip(np.rint(inlines).astype(np.int64), 0, seismic.shape[0] - 1)
    jj = np.clip(np.rint(crosslines).astype(np.int64), 0, seismic.shape[1] - 1)
    section = np.asarray(seismic[ii, jj, :], dtype=np.float32).T
    samples = np.arange(seismic.shape[2], dtype=np.float32)
    return distances, _samples_to_depth(samples), section, inlines, crosslines


def _sample_seismic_section(seismic_path: Path, section_wells: list[SectionWell]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, depths, section, _inlines, _crosslines = _sample_seismic_section_with_geometry(
        seismic_path,
        section_wells,
    )
    return distances, depths, section


def _tg_layer_path(layer_dir: Path) -> Path | None:
    path = layer_dir / f"{TG_LAYER_NAME}.npz"
    return path if path.exists() else None


def _boundary_depth_at_distances(
    query_distances: np.ndarray | list[float],
    boundary_distances: np.ndarray | None,
    boundary_depths: np.ndarray | None,
) -> np.ndarray:
    query = np.asarray(query_distances, dtype=np.float32)
    if boundary_distances is None or boundary_depths is None:
        return np.full(query.shape, np.nan, dtype=np.float32)
    bx = np.asarray(boundary_distances, dtype=np.float32)
    by = np.asarray(boundary_depths, dtype=np.float32)
    finite = np.isfinite(bx) & np.isfinite(by)
    if int(np.count_nonzero(finite)) < 2:
        return np.full(query.shape, np.nan, dtype=np.float32)
    order = np.argsort(bx[finite])
    sorted_x = bx[finite][order]
    sorted_y = by[finite][order]
    return np.interp(query, sorted_x, sorted_y, left=sorted_y[0], right=sorted_y[-1]).astype(np.float32)


def _apply_tg_boundary_to_seismic(
    depths: np.ndarray,
    section: np.ndarray,
    tg_depths: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if tg_depths is None:
        return depths, section
    cropped = np.asarray(section, dtype=np.float32).copy()
    for col_idx, tg_depth in enumerate(np.asarray(tg_depths, dtype=np.float32)):
        if np.isfinite(tg_depth):
            cropped[depths > tg_depth, col_idx] = np.nan
    row_keep = np.any(np.isfinite(cropped), axis=1)
    if np.any(row_keep):
        return depths[row_keep], cropped[row_keep, :]
    return depths, cropped


def _crop_section_wells_to_tg(
    section_wells: list[SectionWell],
    tg_distances: np.ndarray | None,
    tg_depths: np.ndarray | None,
) -> list[SectionWell]:
    if tg_distances is None or tg_depths is None:
        return section_wells
    cropped_wells: list[SectionWell] = []
    for well in section_wells:
        boundary_depth = _boundary_depth_at_distances([well.distance], tg_distances, tg_depths)[0]
        if not np.isfinite(boundary_depth):
            cropped_wells.append(well)
            continue
        boundary_sample = float(boundary_depth) / DEPTH_STEP_TO_SAMPLE - Z_WINDOW_START
        keep = well.log[:, 0] <= boundary_sample + 1e-3
        if np.any(keep):
            cropped_wells.append(
                SectionWell(
                    info=well.info,
                    distance=well.distance,
                    log=well.log[keep],
                )
            )
    return cropped_wells


def _mask_depths_below_boundary(
    xs: list[float],
    depth_values: np.ndarray,
    tg_distances: np.ndarray | None,
    tg_depths: np.ndarray | None,
) -> np.ndarray:
    masked = np.asarray(depth_values, dtype=np.float32).copy()
    if tg_distances is None or tg_depths is None or masked.size == 0:
        return masked
    boundary = _boundary_depth_at_distances(xs, tg_distances, tg_depths)
    below = np.isfinite(masked) & np.isfinite(boundary) & (masked > boundary)
    masked[below] = np.nan
    return masked


def _seismic_color_setting(name: str) -> tuple[str, bool]:
    return SEISMIC_COLOR_OPTIONS.get(name, SEISMIC_COLOR_OPTIONS["RdBu"])


def _seismic_display_setting(name: str) -> str:
    legacy_map = {
        "color": "彩色",
        "wiggle": "波形",
        "color+wiggle": "彩色+波形",
    }
    if name in SEISMIC_DISPLAY_OPTIONS:
        return name
    return legacy_map.get(name, "彩色")


def _seismic_display_has_color(name: str) -> bool:
    return _seismic_display_setting(name) in {"彩色", "彩色+波形"}


def _seismic_display_has_wiggle(name: str) -> bool:
    return _seismic_display_setting(name) in {"波形", "彩色+波形"}


def _add_seismic_background(
    fig: go.Figure,
    seismic_path: Path,
    section_wells: list[SectionWell],
    seismic_colorscale: str,
    seismic_display: str,
) -> None:
    distances, depths, section = _sample_seismic_section(seismic_path, section_wells)
    _add_sampled_seismic_background(fig, distances, depths, section, seismic_colorscale, seismic_display)


def _add_sampled_seismic_background(
    fig: go.Figure,
    distances: np.ndarray,
    depths: np.ndarray,
    section: np.ndarray,
    seismic_colorscale: str,
    seismic_display: str,
) -> None:
    finite = section[np.isfinite(section)]
    if finite.size == 0:
        return
    vmin, vmax = np.percentile(finite, [2.0, 98.0])
    limit = float(max(abs(vmin), abs(vmax), 1e-6))
    colorscale, reversescale = _seismic_color_setting(seismic_colorscale)
    fig.add_trace(
        go.Heatmap(
            x=distances,
            y=depths,
            z=section,
            colorscale=colorscale,
            zmin=-limit,
            zmax=limit,
            reversescale=reversescale,
            colorbar={"title": "seismic", "x": 1.02, "y": 0.76, "len": 0.34},
            name="seismic",
            visible=_seismic_display_has_color(seismic_display),
            hovertemplate="distance=%{x:.1f}<br>depth=%{y:.1f} m<br>amp=%{z:.3f}<extra></extra>",
            meta={"section_seismic_color": True},
        )
    )
    _add_seismic_event_candidates(fig, distances, depths, section)
    _add_seismic_wiggle_traces(
        fig,
        distances,
        depths,
        section,
        visible=_seismic_display_has_wiggle(seismic_display),
    )


def _section_payload_json(
    distances: np.ndarray | None,
    depths: np.ndarray | None,
    section: np.ndarray | None,
) -> str:
    if distances is None or depths is None or section is None:
        return "null"
    payload = {
        "distances": np.asarray(distances, dtype=np.float32).tolist(),
        "depths": np.asarray(depths, dtype=np.float32).tolist(),
        "section": np.asarray(section, dtype=np.float32).tolist(),
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=True, separators=(",", ":"))


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_response_payload(
    *,
    ok: bool,
    result: dict | None = None,
    error: str | None = None,
) -> bytes:
    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    return json.dumps(
        {"ok": ok, "result": json_safe(result), "error": error},
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _dl_progress_update(
    job_id: str | None,
    *,
    state: str,
    kind: str | None = None,
    phase: str | None = None,
    current_epoch: int | None = None,
    total_epochs: int | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    if not job_id:
        return
    now = time.time()
    with _LOCAL_DL_PROGRESS_LOCK:
        current = dict(_LOCAL_DL_PROGRESS.get(job_id, {}))
        current.update(
            {
                "job_id": job_id,
                "state": state,
                "updated_at": now,
            }
        )
        if kind is not None:
            current["kind"] = kind
        if phase is not None:
            current["phase"] = phase
        if current_epoch is not None:
            current["current_epoch"] = int(current_epoch)
        if total_epochs is not None:
            current["total_epochs"] = int(total_epochs)
        if progress is not None:
            current["progress"] = float(np.clip(progress, 0.0, 100.0))
        if message is not None:
            current["message"] = message
        if error is not None:
            current["error"] = error
        _LOCAL_DL_PROGRESS[job_id] = current


def _dl_progress_snapshot(job_id: str | None) -> dict[str, object] | None:
    if not job_id:
        return None
    with _LOCAL_DL_PROGRESS_LOCK:
        current = _LOCAL_DL_PROGRESS.get(job_id)
        return dict(current) if current is not None else None


def _prepare_subset_for_dl(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = payload.get("subset") or {}
    x = np.asarray(subset.get("subX") or [], dtype=np.float32)
    y = np.asarray(subset.get("subY") or [], dtype=np.float32)
    z = np.asarray(subset.get("subZ") or [], dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 2:
        raise ValueError("Invalid subset payload for local DL modeling.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"Subset shape mismatch: z={tuple(z.shape)}, expected={(int(y.size), int(x.size))}."
        )
    if x.size < 2 or y.size < 4:
        raise ValueError("Selected box is too small for local DL modeling.")
    return x, y, z


def _prepare_local_seeds(payload: dict, kind: str) -> list[dict]:
    seeds = payload.get("seeds") or []
    prepared: list[dict] = []
    for seed in seeds:
        anchor_x = float(seed.get("anchorX", np.nan))
        depths = np.asarray(seed.get("depths") or [], dtype=np.float32)
        values = np.asarray(seed.get("values") or [], dtype=np.float32)
        well_name = str(seed.get("wellName") or kind.upper())
        if not np.isfinite(anchor_x) or depths.size == 0 or values.size == 0:
            continue
        size = min(depths.size, values.size)
        mask = np.isfinite(depths[:size]) & np.isfinite(values[:size])
        if not np.any(mask):
            continue
        ordered = np.argsort(depths[:size][mask])
        prepared.append(
            {
                "anchor_x": anchor_x,
                "depths": depths[:size][mask][ordered],
                "values": values[:size][mask][ordered],
                "well_name": well_name,
            }
        )
    if not prepared:
        raise ValueError(f"No {kind.upper()} labels were found inside the selected box.")
    return prepared


def _median_positive_depth_step(seeds: list[dict]) -> float | None:
    diffs: list[float] = []
    for seed in seeds:
        depths = np.asarray(seed["depths"], dtype=np.float32)
        if depths.size < 2:
            continue
        local_diffs = np.diff(np.sort(depths))
        diffs.extend(float(diff) for diff in local_diffs if np.isfinite(diff) and diff > 1e-6)
    if not diffs:
        return None
    return float(np.median(np.asarray(diffs, dtype=np.float32)))


def _build_dl_depth_axis(sub_y: np.ndarray, seeds: list[dict], kind: str) -> tuple[np.ndarray, dict[str, float | int | str]]:
    y_min = float(np.nanmin(sub_y))
    y_max = float(np.nanmax(sub_y))
    if y_max <= y_min:
        return sub_y.astype(np.float32), {
            "resolution_mode": "seismic",
            "depth_step_m": 0.0,
            "depth_points": int(sub_y.size),
        }

    seismic_step = float(np.median(np.diff(np.sort(sub_y)))) if sub_y.size > 1 else DEPTH_STEP_TO_SAMPLE
    seed_step = _median_positive_depth_step(seeds)
    if seed_step is None:
        seed_step = seismic_step

    if DL_OUTPUT_RESOLUTION == "fixed-depth":
        step = float(DL_FIXED_GRID_STEP_M)
    elif kind == "lith":
        min_step = DL_LITH_GRID_MIN_STEP_M
        max_step = DL_LITH_GRID_MAX_STEP_M
        step = float(np.clip(seed_step, min_step, max_step))
    else:
        min_step = DL_WELL_GRID_MIN_STEP_M
        max_step = DL_WELL_GRID_MAX_STEP_M
        step = float(np.clip(seed_step, min_step, max_step))

    count = int(np.floor((y_max - y_min) / step)) + 1
    if count > DL_WELL_GRID_MAX_DEPTH_POINTS:
        count = int(DL_WELL_GRID_MAX_DEPTH_POINTS)
        depth_axis = np.linspace(y_min, y_max, count, dtype=np.float32)
        step = float((y_max - y_min) / max(count - 1, 1))
    else:
        depth_axis = y_min + np.arange(count, dtype=np.float32) * np.float32(step)
        if float(depth_axis[-1]) < y_max:
            depth_axis = np.append(depth_axis, np.float32(y_max))
    return depth_axis.astype(np.float32), {
        "resolution_mode": DL_OUTPUT_RESOLUTION,
        "depth_step_m": float(step),
        "seed_depth_step_m": float(seed_step),
        "seismic_depth_step_m": float(seismic_step),
        "depth_points": int(depth_axis.size),
    }


def _resample_section_depth_axis(sub_y: np.ndarray, sub_z: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    order = np.argsort(sub_y)
    source_y = sub_y[order].astype(np.float32)
    source_z = sub_z[order, :].astype(np.float32)
    out = np.empty((target_y.size, source_z.shape[1]), dtype=np.float32)
    for trace_idx in range(source_z.shape[1]):
        trace = source_z[:, trace_idx]
        finite = np.isfinite(source_y) & np.isfinite(trace)
        if int(finite.sum()) < 2:
            out[:, trace_idx] = np.nan
            continue
        out[:, trace_idx] = np.interp(
            target_y.astype(np.float32),
            source_y[finite],
            trace[finite],
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
    return out


def _prepare_dl_model_grid(
    sub_x: np.ndarray,
    sub_y: np.ndarray,
    sub_z: np.ndarray,
    seeds: list[dict],
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if DL_OUTPUT_RESOLUTION == "seismic":
        return sub_x, sub_y, sub_z, {
            "resolution_mode": "seismic",
            "depth_step_m": float(np.median(np.diff(np.sort(sub_y)))) if sub_y.size > 1 else 0.0,
            "depth_points": int(sub_y.size),
        }
    model_y, resolution_meta = _build_dl_depth_axis(sub_y, seeds, kind)
    if model_y.size == sub_y.size and np.allclose(model_y, sub_y):
        return sub_x, sub_y, sub_z, resolution_meta
    model_z = _resample_section_depth_axis(sub_y, sub_z, model_y)
    return sub_x, model_y, model_z, resolution_meta


def _preprocess_local_seismic(section: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    clip_min, clip_max = np.percentile(section[np.isfinite(section)], [1.0, 99.0])
    clipped = np.clip(np.nan_to_num(section, nan=0.0), clip_min, clip_max)
    trace_major = clipped.T.astype(np.float32)
    trace_mean = trace_major.mean(axis=1, keepdims=True)
    trace_std = np.maximum(trace_major.std(axis=1, keepdims=True), 1e-6)
    processed = (trace_major - trace_mean) / trace_std
    return processed.astype(np.float32), {
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
    }


def _seed_label_confidences(seeds: list[dict], kind: str) -> tuple[list[np.ndarray], dict[str, float | int]]:
    confidences = [np.ones(np.asarray(seed["values"]).shape, dtype=np.float32) for seed in seeds]
    items: list[tuple[int, int, float, float]] = []
    for seed_idx, seed in enumerate(seeds):
        depths = np.asarray(seed["depths"], dtype=np.float32)
        values = np.asarray(seed["values"], dtype=np.float32)
        for value_idx, (depth, value) in enumerate(zip(depths, values)):
            if np.isfinite(depth) and np.isfinite(value):
                items.append((seed_idx, value_idx, float(depth), float(value)))

    if len(seeds) < 2 or len(items) < 2:
        return confidences, {
            "label_conflict_mean": 0.0,
            "label_conflict_max": 0.0,
            "label_conflict_points": 0,
        }

    all_values = np.asarray([item[3] for item in items], dtype=np.float32)
    global_scale = float(np.nanpercentile(all_values, 75) - np.nanpercentile(all_values, 25))
    global_scale = max(global_scale, float(np.nanstd(all_values)), 1e-6)
    depth_window = (
        DL_LABEL_CONFLICT_LITH_DEPTH_WINDOW_M
        if kind == "lith"
        else DL_LABEL_CONFLICT_CONTINUOUS_DEPTH_WINDOW_M
    )

    conflict_values: list[float] = []
    for seed_idx, value_idx, depth, value in items:
        neighbors = [
            other_value
            for other_seed_idx, _other_value_idx, other_depth, other_value in items
            if other_seed_idx != seed_idx and abs(other_depth - depth) <= depth_window
        ]
        if not neighbors:
            continue
        neighbor_values = np.asarray(neighbors, dtype=np.float32)
        if kind == "lith":
            value_class = int(round(value))
            neighbor_classes = np.rint(neighbor_values).astype(np.int32)
            conflict = float(np.mean(neighbor_classes != value_class))
        else:
            median_value = float(np.nanmedian(neighbor_values))
            scale = max(global_scale * DL_LABEL_CONFLICT_CONTINUOUS_SCALE_FACTOR, 1e-6)
            conflict = float(np.clip(abs(value - median_value) / scale, 0.0, 1.0))
        confidence = float(1.0 - (1.0 - DL_LABEL_CONFLICT_MIN_WEIGHT) * np.clip(conflict, 0.0, 1.0))
        confidences[seed_idx][value_idx] = min(confidences[seed_idx][value_idx], confidence)
        conflict_values.append(1.0 - confidence)

    if not conflict_values:
        return confidences, {
            "label_conflict_mean": 0.0,
            "label_conflict_max": 0.0,
            "label_conflict_points": 0,
        }
    return confidences, {
        "label_conflict_mean": float(np.mean(conflict_values)),
        "label_conflict_max": float(np.max(conflict_values)),
        "label_conflict_points": int(np.sum(np.asarray(conflict_values) > 1e-6)),
    }


def _build_local_supervision(
    sub_x: np.ndarray,
    sub_y: np.ndarray,
    seeds: list[dict],
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    targets = np.zeros((sub_x.size, sub_y.size), dtype=np.float32)
    mask = np.zeros((sub_x.size, sub_y.size), dtype=bool)
    weights = np.zeros((sub_x.size, sub_y.size), dtype=np.float32)
    conflict_grid = np.zeros((sub_x.size, sub_y.size), dtype=np.float32)
    label_confidences, conflict_meta = _seed_label_confidences(seeds, kind)
    for seed_idx, seed in enumerate(seeds):
        trace_center = int(np.argmin(np.abs(sub_x - float(seed["anchor_x"]))))
        seed_depths = np.asarray(seed["depths"], dtype=np.float32)
        seed_values = np.asarray(seed["values"], dtype=np.float32)
        seed_confidences = label_confidences[seed_idx]
        for value_idx, (depth, value) in enumerate(zip(seed_depths, seed_values)):
            depth_idx = int(np.argmin(np.abs(sub_y - float(depth))))
            label_confidence = float(seed_confidences[value_idx]) if value_idx < seed_confidences.size else 1.0
            label_conflict = float(1.0 - label_confidence)
            left = max(0, trace_center - DL_POR_SUPERVISION_RADIUS)
            right = min(sub_x.size, trace_center + DL_POR_SUPERVISION_RADIUS + 1)
            for trace_idx in range(left, right):
                lateral = abs(trace_idx - trace_center)
                trace_weight = 1.0 if lateral == 0 else DL_POR_NEIGHBOR_WEIGHT
                trace_weight *= label_confidence
                targets[trace_idx, depth_idx] = float(value)
                mask[trace_idx, depth_idx] = True
                weights[trace_idx, depth_idx] = max(weights[trace_idx, depth_idx], trace_weight)
                conflict_grid[trace_idx, depth_idx] = max(conflict_grid[trace_idx, depth_idx], label_conflict)
    if not np.any(mask):
        raise ValueError("No valid supervision points were found inside the selected box.")
    return targets, mask, weights, conflict_grid, conflict_meta


def _label_conflict_payload(model_x: np.ndarray, model_y: np.ndarray, conflict_grid: np.ndarray) -> dict[str, object]:
    if conflict_grid.size == 0:
        return {"x": [], "y": [], "z": []}
    return {
        "x": model_x.astype(float).tolist(),
        "y": model_y.astype(float).tolist(),
        "z": conflict_grid.T.astype(float).tolist(),
    }


def _vertical_smooth2d(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 3:
        return values.astype(np.float32)
    padded = np.pad(values, ((0, 0), (1, 1)), mode="edge")
    return (
        0.25 * padded[:, :-2]
        + 0.50 * padded[:, 1:-1]
        + 0.25 * padded[:, 2:]
    ).astype(np.float32)


def _build_local_fault_aware_smoothing(seismic_proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = np.abs(np.diff(seismic_proc, axis=0))
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        weight = np.ones_like(diff, dtype=np.float32)
        barrier = np.zeros_like(diff, dtype=np.float32)
        return weight, barrier

    scale = float(np.median(finite)) + 1e-6
    base_similarity = np.exp(-diff / scale).astype(np.float32)

    jump = _vertical_smooth2d(diff.astype(np.float32))
    low = float(np.median(finite))
    high = float(np.percentile(finite, DL_FAULT_BARRIER_PERCENTILE))
    denom = max(high - low, 1e-6)
    barrier = np.clip((jump - low) / denom, 0.0, 1.0).astype(np.float32)
    barrier = np.power(barrier, DL_FAULT_BARRIER_POWER).astype(np.float32)

    barrier_gate = np.clip(1.0 - DL_FAULT_BARRIER_STRENGTH * barrier, 0.0, 1.0)
    weight = np.clip(base_similarity * barrier_gate, DL_FAULT_MIN_SMOOTH_WEIGHT, 1.0)
    return weight.astype(np.float32), barrier.astype(np.float32)


def _fault_barrier_payload(model_x: np.ndarray, model_y: np.ndarray, barrier: np.ndarray) -> dict[str, object]:
    if barrier.size == 0 or model_x.size < 2:
        return {"x": [], "y": [], "z": []}
    barrier_x = ((model_x[:-1] + model_x[1:]) * 0.5).astype(np.float32)
    return {
        "x": barrier_x.astype(float).tolist(),
        "y": model_y.astype(float).tolist(),
        "z": barrier.T.astype(float).tolist(),
    }


def _local_dl_training_config(sub_x: np.ndarray, sub_y: np.ndarray) -> dict[str, float | int]:
    area = max(1, int(sub_x.size) * int(sub_y.size))
    area_scale = float(np.sqrt(area / float(DL_LOCAL_REFERENCE_AREA)))
    epoch_scale = max(1.0, area_scale)
    epochs = int(round(DL_POR_EPOCHS * epoch_scale / 10.0) * 10)
    epochs = int(np.clip(epochs, DL_POR_EPOCHS, DL_LOCAL_EPOCH_MAX))
    label_loss_scale = float(np.clip(np.sqrt(epoch_scale), 1.0, DL_LOCAL_LABEL_WEIGHT_MAX_SCALE))
    return {
        "area": area,
        "area_scale": area_scale,
        "epochs": epochs,
        "label_loss_scale": label_loss_scale,
    }


def _predict_local_continuous_with_gru(payload: dict, kind: str) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 1, 8)))

    import torch
    from torch import nn

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    class LocalPorGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.GRU(
                input_size=1,
                hidden_size=DL_POR_HIDDEN_SIZE,
                num_layers=DL_POR_NUM_LAYERS,
                batch_first=True,
                bidirectional=True,
                dropout=DL_LOCAL_DROPOUT if DL_POR_NUM_LAYERS > 1 else 0.0,
            )
            latent_size = DL_POR_HIDDEN_SIZE * 2
            self.decoder = nn.GRU(
                input_size=latent_size,
                hidden_size=DL_POR_HIDDEN_SIZE,
                num_layers=1,
                batch_first=True,
            )
            self.recon_head = nn.Linear(DL_POR_HIDDEN_SIZE, 1)
            self.log_head = nn.Sequential(
                nn.Linear(latent_size, DL_POR_HIDDEN_SIZE),
                nn.GELU(),
                nn.Linear(DL_POR_HIDDEN_SIZE, 1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent, _ = self.encoder(x)
            decoded, _ = self.decoder(latent)
            recon = self.recon_head(decoded)
            pred = self.log_head(latent)
            return recon, pred

    job_id = str(payload.get("job_id") or "")
    sub_x, sub_y, sub_z = _prepare_subset_for_dl(payload)
    seeds = _prepare_local_seeds(payload, kind)
    model_x, model_y, model_z, resolution_meta = _prepare_dl_model_grid(sub_x, sub_y, sub_z, seeds, kind)
    training_config = _local_dl_training_config(model_x, model_y)
    epochs = int(training_config["epochs"])
    log_loss_weight = float(DL_POR_LOG_LOSS_WEIGHT * float(training_config["label_loss_scale"]))
    seismic_proc, _seis_stats = _preprocess_local_seismic(model_z)
    targets_raw, target_mask, target_weight, label_conflict, conflict_meta = _build_local_supervision(
        model_x,
        model_y,
        seeds,
        kind,
    )
    valid_output_mask_np = np.isfinite(model_z.T).astype(bool)
    target_mask = target_mask & valid_output_mask_np
    if not np.any(target_mask):
        raise ValueError("No valid supervision points remain above TG inside the selected box.")
    smooth_weight, fault_barrier = _build_local_fault_aware_smoothing(seismic_proc)

    seed_values = np.concatenate([np.asarray(seed["values"], dtype=np.float32) for seed in seeds])
    if kind == "perm":
        seed_values_proc = 0.5 * np.log(np.maximum(seed_values, 1e-6))
        targets_proc_raw = 0.5 * np.log(np.maximum(targets_raw, 1e-6))
    else:
        seed_values_proc = seed_values
        targets_proc_raw = targets_raw
    value_mean = float(seed_values_proc.mean())
    value_std = float(max(seed_values_proc.std(), 1e-6))
    targets_proc = np.where(target_mask, (targets_proc_raw - value_mean) / value_std, 0.0).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalPorGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DL_POR_LR, weight_decay=DL_LOCAL_WEIGHT_DECAY)
    mse = nn.MSELoss()

    x_tensor = torch.from_numpy(seismic_proc[:, :, None]).to(device)
    seismic_target = x_tensor
    log_target = torch.from_numpy(targets_proc[:, :, None]).to(device)
    log_mask = torch.from_numpy(target_mask[:, :, None]).to(device)
    log_weight = torch.from_numpy(target_weight[:, :, None]).to(device)
    smooth_weight_t = torch.from_numpy(smooth_weight[:, :, None]).to(device)
    valid_output_mask_t = torch.from_numpy(valid_output_mask_np[:, :, None]).to(device)
    smooth_valid_mask_t = valid_output_mask_t[1:] & valid_output_mask_t[:-1]

    start_time = time.perf_counter()
    _dl_progress_update(
        job_id,
        state="running",
        kind=kind,
        phase="training",
        current_epoch=0,
        total_epochs=epochs,
        progress=0.0,
        message=f"Training local {kind.upper()} BiGRU: 0/{epochs}",
    )
    progress_update_interval = max(1, epochs // 100)
    for epoch_idx in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seismic_recon, log_pred = model(x_tensor)
        recon_error = (seismic_recon - seismic_target) ** 2
        recon_loss = torch.where(valid_output_mask_t, recon_error, torch.zeros_like(recon_error)).sum()
        recon_loss = recon_loss / torch.clamp(valid_output_mask_t.sum(), min=1)
        log_error = (log_pred - log_target) ** 2
        weighted_error = torch.where(log_mask, log_error * log_weight, torch.zeros_like(log_error))
        denom = torch.clamp(log_weight[log_mask].sum(), min=1.0)
        log_loss = weighted_error.sum() / denom
        smooth_error = smooth_weight_t * torch.abs(log_pred[1:] - log_pred[:-1])
        smooth_loss = torch.where(smooth_valid_mask_t, smooth_error, torch.zeros_like(smooth_error)).sum()
        smooth_loss = smooth_loss / torch.clamp(smooth_valid_mask_t.sum(), min=1)
        loss = recon_loss + log_loss_weight * log_loss + DL_POR_SMOOTH_LOSS_WEIGHT * smooth_loss
        loss.backward()
        optimizer.step()
        current_epoch = epoch_idx + 1
        if current_epoch == epochs or current_epoch % progress_update_interval == 0:
            _dl_progress_update(
                job_id,
                state="running",
                kind=kind,
                phase="training",
                current_epoch=current_epoch,
                total_epochs=epochs,
                progress=100.0 * current_epoch / max(epochs, 1),
                message=f"Training local {kind.upper()} BiGRU: {current_epoch}/{epochs}",
            )

    model.eval()
    _dl_progress_update(
        job_id,
        state="running",
        kind=kind,
        phase="rendering",
        current_epoch=epochs,
        total_epochs=epochs,
        progress=100.0,
        message=f"Rendering local {kind.upper()} prediction.",
    )
    with torch.no_grad():
        _seismic_recon, log_pred = model(x_tensor)
    pred_proc = log_pred.squeeze(-1).cpu().numpy().astype(np.float32)
    pred_raw = pred_proc * value_std + value_mean
    if kind == "perm":
        pred_raw = np.exp(2.0 * pred_raw)
    for trace_idx, depth_idx in np.argwhere(target_mask):
        pred_raw[trace_idx, depth_idx] = targets_raw[trace_idx, depth_idx]
    pred_raw[~valid_output_mask_np] = np.nan

    return {
        "kind": kind,
        "x": model_x.astype(float).tolist(),
        "y": model_y.astype(float).tolist(),
        "z": pred_raw.T.astype(float).tolist(),
        "seeds": [
            {
                "anchorX": float(seed["anchor_x"]),
                "depths": np.asarray(seed["depths"], dtype=float).tolist(),
                "values": np.asarray(seed["values"], dtype=float).tolist(),
                "wellName": seed["well_name"],
            }
            for seed in seeds
        ],
        "seismic": {
            "subX": model_x.astype(float).tolist(),
            "subY": model_y.astype(float).tolist(),
            "subZ": model_z.astype(float).tolist(),
            "faultBarrier": _fault_barrier_payload(model_x, model_y, fault_barrier),
            "labelConflict": _label_conflict_payload(model_x, model_y, label_conflict),
        },
        "meta": {
            "model": f"BiGRU local {kind.upper()}",
            "device": str(device),
            **resolution_meta,
            "fault_aware_smoothing": True,
            "fault_barrier_percentile": float(DL_FAULT_BARRIER_PERCENTILE),
            "label_conflict_aware": True,
            **conflict_meta,
            "epochs": epochs,
            "base_epochs": int(DL_POR_EPOCHS),
            "hidden_size": int(DL_POR_HIDDEN_SIZE),
            "num_layers": int(DL_POR_NUM_LAYERS),
            "label_loss_weight": log_loss_weight,
            "selected_area": int(training_config["area"]),
            "grid_shape": [int(model_x.size), int(model_y.size)],
            "source_grid_shape": [int(sub_x.size), int(sub_y.size)],
            "seed_wells": len(seeds),
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
    }


def _predict_local_lith_with_gru(payload: dict) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", str(min(os.cpu_count() or 1, 8)))

    import torch
    from torch import nn

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    class LocalLithGRU(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.encoder = nn.GRU(
                input_size=1,
                hidden_size=DL_POR_HIDDEN_SIZE,
                num_layers=DL_POR_NUM_LAYERS,
                batch_first=True,
                bidirectional=True,
                dropout=DL_LOCAL_DROPOUT if DL_POR_NUM_LAYERS > 1 else 0.0,
            )
            latent_size = DL_POR_HIDDEN_SIZE * 2
            self.decoder = nn.GRU(
                input_size=latent_size,
                hidden_size=DL_POR_HIDDEN_SIZE,
                num_layers=1,
                batch_first=True,
            )
            self.recon_head = nn.Linear(DL_POR_HIDDEN_SIZE, 1)
            self.class_head = nn.Sequential(
                nn.Linear(latent_size, DL_POR_HIDDEN_SIZE),
                nn.GELU(),
                nn.Linear(DL_POR_HIDDEN_SIZE, num_classes),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent, _ = self.encoder(x)
            decoded, _ = self.decoder(latent)
            recon = self.recon_head(decoded)
            logits = self.class_head(latent)
            return recon, logits

    job_id = str(payload.get("job_id") or "")
    sub_x, sub_y, sub_z = _prepare_subset_for_dl(payload)
    seeds = _prepare_local_seeds(payload, "lith")
    model_x, model_y, model_z, resolution_meta = _prepare_dl_model_grid(sub_x, sub_y, sub_z, seeds, "lith")
    training_config = _local_dl_training_config(model_x, model_y)
    epochs = int(training_config["epochs"])
    class_loss_weight = float(DL_LITH_CLASS_LOSS_WEIGHT * float(training_config["label_loss_scale"]))
    seismic_proc, _seis_stats = _preprocess_local_seismic(model_z)
    targets_raw, target_mask, target_weight, label_conflict, conflict_meta = _build_local_supervision(
        model_x,
        model_y,
        seeds,
        "lith",
    )
    valid_output_mask_np = np.isfinite(model_z.T).astype(bool)
    target_mask = target_mask & valid_output_mask_np
    if not np.any(target_mask):
        raise ValueError("No valid supervision points remain above TG inside the selected box.")
    smooth_weight, fault_barrier = _build_local_fault_aware_smoothing(seismic_proc)

    class_values = sorted({int(round(float(value))) for seed in seeds for value in seed["values"]})
    class_to_index = {value: idx for idx, value in enumerate(class_values)}
    index_to_class = {idx: value for value, idx in class_to_index.items()}
    targets_index = np.full_like(targets_raw, -1, dtype=np.int64)
    for trace_idx, depth_idx in np.argwhere(target_mask):
        targets_index[trace_idx, depth_idx] = class_to_index[int(round(float(targets_raw[trace_idx, depth_idx])))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalLithGRU(len(class_values)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DL_POR_LR, weight_decay=DL_LOCAL_WEIGHT_DECAY)
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss(reduction="none")

    x_tensor = torch.from_numpy(seismic_proc[:, :, None]).to(device)
    seismic_target = x_tensor
    target_index_t = torch.from_numpy(targets_index).to(device)
    target_mask_t = torch.from_numpy(target_mask).to(device)
    target_weight_t = torch.from_numpy(target_weight).to(device)
    smooth_weight_t = torch.from_numpy(smooth_weight[:, :, None]).to(device)
    valid_output_mask_t = torch.from_numpy(valid_output_mask_np[:, :, None]).to(device)
    smooth_valid_mask_t = valid_output_mask_t[1:] & valid_output_mask_t[:-1]

    start_time = time.perf_counter()
    _dl_progress_update(
        job_id,
        state="running",
        kind="lith",
        phase="training",
        current_epoch=0,
        total_epochs=epochs,
        progress=0.0,
        message=f"Training local LITH BiGRU: 0/{epochs}",
    )
    progress_update_interval = max(1, epochs // 100)
    for epoch_idx in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seismic_recon, lith_logits = model(x_tensor)
        recon_error = (seismic_recon - seismic_target) ** 2
        recon_loss = torch.where(valid_output_mask_t, recon_error, torch.zeros_like(recon_error)).sum()
        recon_loss = recon_loss / torch.clamp(valid_output_mask_t.sum(), min=1)
        logits_flat = lith_logits.reshape(-1, lith_logits.shape[-1])
        target_flat = target_index_t.reshape(-1)
        mask_flat = target_mask_t.reshape(-1)
        weight_flat = target_weight_t.reshape(-1)
        class_error = cross_entropy(logits_flat, torch.clamp(target_flat, min=0))
        class_error = torch.where(mask_flat, class_error * weight_flat, torch.zeros_like(class_error))
        denom = torch.clamp(weight_flat[mask_flat].sum(), min=1.0)
        class_loss = class_error.sum() / denom
        class_prob = torch.softmax(lith_logits, dim=-1)
        smooth_error = smooth_weight_t * torch.abs(class_prob[1:] - class_prob[:-1])
        smooth_loss = torch.where(smooth_valid_mask_t, smooth_error, torch.zeros_like(smooth_error)).sum()
        smooth_loss = smooth_loss / torch.clamp(smooth_valid_mask_t.sum(), min=1)
        loss = recon_loss + class_loss_weight * class_loss + DL_POR_SMOOTH_LOSS_WEIGHT * smooth_loss
        loss.backward()
        optimizer.step()
        current_epoch = epoch_idx + 1
        if current_epoch == epochs or current_epoch % progress_update_interval == 0:
            _dl_progress_update(
                job_id,
                state="running",
                kind="lith",
                phase="training",
                current_epoch=current_epoch,
                total_epochs=epochs,
                progress=100.0 * current_epoch / max(epochs, 1),
                message=f"Training local LITH BiGRU: {current_epoch}/{epochs}",
            )

    model.eval()
    _dl_progress_update(
        job_id,
        state="running",
        kind="lith",
        phase="rendering",
        current_epoch=epochs,
        total_epochs=epochs,
        progress=100.0,
        message="Rendering local LITH prediction.",
    )
    with torch.no_grad():
        _seismic_recon, lith_logits = model(x_tensor)
    pred_index = torch.argmax(lith_logits, dim=-1).cpu().numpy().astype(np.int64)
    pred_class = np.vectorize(index_to_class.get)(pred_index).astype(np.float32)
    for trace_idx, depth_idx in np.argwhere(target_mask):
        pred_class[trace_idx, depth_idx] = targets_raw[trace_idx, depth_idx]
    pred_class[~valid_output_mask_np] = np.nan

    return {
        "kind": "lith",
        "x": model_x.astype(float).tolist(),
        "y": model_y.astype(float).tolist(),
        "z": pred_class.T.astype(float).tolist(),
        "seeds": [
            {
                "anchorX": float(seed["anchor_x"]),
                "depths": np.asarray(seed["depths"], dtype=float).tolist(),
                "values": np.asarray(seed["values"], dtype=float).tolist(),
                "wellName": seed["well_name"],
            }
            for seed in seeds
        ],
        "seismic": {
            "subX": model_x.astype(float).tolist(),
            "subY": model_y.astype(float).tolist(),
            "subZ": model_z.astype(float).tolist(),
            "faultBarrier": _fault_barrier_payload(model_x, model_y, fault_barrier),
            "labelConflict": _label_conflict_payload(model_x, model_y, label_conflict),
        },
        "meta": {
            "model": "BiGRU local LITH",
            "device": str(device),
            **resolution_meta,
            "fault_aware_smoothing": True,
            "fault_barrier_percentile": float(DL_FAULT_BARRIER_PERCENTILE),
            "label_conflict_aware": True,
            **conflict_meta,
            "epochs": epochs,
            "base_epochs": int(DL_POR_EPOCHS),
            "hidden_size": int(DL_POR_HIDDEN_SIZE),
            "num_layers": int(DL_POR_NUM_LAYERS),
            "class_loss_weight": class_loss_weight,
            "selected_area": int(training_config["area"]),
            "grid_shape": [int(model_x.size), int(model_y.size)],
            "source_grid_shape": [int(sub_x.size), int(sub_y.size)],
            "seed_wells": len(seeds),
            "classes": class_values,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
    }


def _predict_with_local_gru(payload: dict) -> dict:
    kind = str(payload.get("kind") or "").strip().lower()
    if kind == "lith":
        return _predict_local_lith_with_gru(payload)
    if kind == "perm":
        return _predict_local_continuous_with_gru(payload, "perm")
    if kind == "por":
        return _predict_local_continuous_with_gru(payload, "por")
    raise ValueError(f"Unsupported local DL kind: {kind!r}")


class _LocalDlRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args) -> None:
        return

    def _send_json(self, code: int, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD" and payload:
            self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self._send_json(204, b"")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/local-dl-status":
            self._send_json(404, _json_response_payload(ok=False, error="Unknown local DL endpoint."))
            return
        query = parse_qs(parsed.query)
        job_id = (query.get("job_id") or [""])[0]
        status = _dl_progress_snapshot(job_id)
        if status is None:
            self._send_json(404, _json_response_payload(ok=False, error="Unknown local DL job."))
            return
        self._send_json(
            200,
            json.dumps(
                {"ok": True, "status": status},
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
        )

    def do_POST(self) -> None:
        if self.path not in {"/local-dl-model", "/local-dl-por"}:
            self._send_json(404, _json_response_payload(ok=False, error="Unknown local DL endpoint."))
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        job_id = ""
        try:
            payload = json.loads(raw.decode("utf-8"))
            job_id = str(payload.get("job_id") or "")
            if self.path == "/local-dl-por" and "kind" not in payload:
                payload["kind"] = "por"
            kind = str(payload.get("kind") or "por").strip().lower()
            _dl_progress_update(
                job_id,
                state="running",
                kind=kind,
                phase="preparing",
                progress=0.0,
                message=f"Preparing local {kind.upper()} DL modeling.",
            )
            result = _predict_with_local_gru(payload)
        except Exception as exc:
            _dl_progress_update(
                job_id,
                state="failed",
                phase="failed",
                progress=0.0,
                message="Local DL modeling failed.",
                error=str(exc),
            )
            self._send_json(
                500,
                _json_response_payload(ok=False, error=str(exc)),
            )
            return
        _dl_progress_update(
            job_id,
            state="done",
            phase="done",
            progress=100.0,
            message="Local DL modeling complete.",
        )
        self._send_json(200, _json_response_payload(ok=True, result=result))


def _ensure_local_dl_backend() -> str:
    with _LOCAL_DL_BACKEND_LOCK:
        url = _LOCAL_DL_BACKEND.get("url")
        if isinstance(url, str) and url:
            return url
        port = _pick_free_local_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), _LocalDlRequestHandler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="well-section-local-dl-backend",
            daemon=True,
        )
        thread.start()
        url = f"http://127.0.0.1:{port}/local-dl-model"
        _LOCAL_DL_BACKEND["server"] = server
        _LOCAL_DL_BACKEND["thread"] = thread
        _LOCAL_DL_BACKEND["url"] = url
        print(f"[well section] Local DL backend started: {url}")
        return url


def _add_seismic_wiggle_traces(
    fig: go.Figure,
    distances: np.ndarray,
    depths: np.ndarray,
    section: np.ndarray,
    visible: bool,
) -> None:
    finite = section[np.isfinite(section)]
    if finite.size == 0 or distances.size < 2:
        return

    step = max(1, int(np.ceil(distances.size / SEISMIC_WIGGLE_MAX_TRACES)))
    trace_indices = np.arange(0, distances.size, step, dtype=np.int64)
    if trace_indices[-1] != distances.size - 1:
        trace_indices = np.append(trace_indices, distances.size - 1)

    spacing = float(np.nanmedian(np.diff(distances[trace_indices]))) if trace_indices.size > 1 else 1.0
    spacing = max(spacing, 1e-6)
    amp_limit = float(np.nanpercentile(np.abs(finite), 98.0))
    if amp_limit <= 1e-6:
        return
    wiggle_scale = spacing * SEISMIC_WIGGLE_AMPLITUDE_FRACTION / amp_limit

    for order, x_idx in enumerate(trace_indices):
        base_x = float(distances[x_idx])
        amplitude = np.nan_to_num(section[:, x_idx].astype(np.float32), nan=0.0)
        wiggle_x = base_x + amplitude * wiggle_scale
        positive_x = base_x + np.maximum(amplitude, 0.0) * wiggle_scale

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([np.full(depths.size, base_x, dtype=np.float32), positive_x[::-1]]),
                y=np.concatenate([depths, depths[::-1]]),
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                fill="toself",
                fillcolor="rgba(0,0,0,0.82)",
                name="seismic wiggle positive fill",
                legendgroup="seismic wiggle",
                showlegend=False,
                hoverinfo="skip",
                visible=visible,
                meta={"seismic_wiggle": True, "wiggle_fill": True},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=wiggle_x,
                y=depths,
                mode="lines",
                line={"color": "black", "width": 1},
                name="seismic wiggle",
                legendgroup="seismic wiggle",
                showlegend=order == 0,
                visible=visible,
                hovertemplate=(
                    "seismic wiggle<br>distance="
                    f"{base_x:.1f}<br>depth=%{{y:.1f}} m<extra></extra>"
                ),
                meta={"seismic_wiggle": True, "wiggle_line": True},
            )
        )


def _normalize_positive_signal(values: np.ndarray) -> np.ndarray:
    signal = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    signal = np.maximum(signal, 0.0)
    finite = signal[signal > 0.0]
    if finite.size == 0:
        return signal
    high = float(np.percentile(finite, 99.0))
    if high <= 1e-6:
        return signal
    return np.clip(signal / high, 0.0, 1.0)


def _smooth_event_track(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values
    padded = np.pad(values.astype(np.float32), (2, 2), mode="edge")
    return (
        padded[:-4] * 0.0625
        + padded[1:-3] * 0.25
        + padded[2:-2] * 0.375
        + padded[3:-1] * 0.25
        + padded[4:] * 0.0625
    )


def _tracks_from_skeleton(
    distances: np.ndarray,
    depths: np.ndarray,
    skeleton: np.ndarray,
    vesselness: np.ndarray,
    *,
    min_length: int,
    max_slope: float,
) -> list[dict[str, np.ndarray | float]]:
    tracks: list[dict[str, np.ndarray | float]] = []
    labels = measure.label(skeleton, connectivity=2)
    for region in measure.regionprops(labels, intensity_image=vesselness):
        coords = region.coords
        if coords.shape[0] < min_length:
            continue

        columns = np.unique(coords[:, 1])
        if columns.size < min_length:
            continue

        y_indices = []
        x_indices = []
        for col in columns:
            rows = coords[coords[:, 1] == col, 0]
            if rows.size == 0:
                continue
            x_indices.append(int(col))
            y_indices.append(float(np.median(rows)))

        x_indices_arr = np.asarray(x_indices, dtype=np.int64)
        y_indices_arr = np.asarray(y_indices, dtype=np.float32)
        if x_indices_arr.size < min_length:
            continue

        gaps = np.diff(x_indices_arr)
        slopes = np.abs(np.diff(y_indices_arr)) / np.maximum(gaps, 1)
        if slopes.size and float(np.percentile(slopes, 90.0)) > max_slope:
            continue

        y_indices_arr = _smooth_event_track(y_indices_arr)
        x = distances[np.clip(x_indices_arr, 0, distances.size - 1)]
        y = depths[np.clip(np.rint(y_indices_arr).astype(np.int64), 0, depths.size - 1)]
        tracks.append(
            {
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "score": float(region.intensity_mean) * float(np.log1p(x.size)),
            }
        )
    return tracks


def _sato_centerline_tracks(
    distances: np.ndarray,
    depths: np.ndarray,
    signal: np.ndarray,
) -> list[dict[str, np.ndarray | float]]:
    normalized = _normalize_positive_signal(signal)
    positive = normalized[normalized > 0.0]
    if positive.size == 0:
        return []

    vesselness = filters.sato(
        normalized,
        sigmas=SEISMIC_EVENT_SATO_SIGMAS,
        black_ridges=False,
        mode="reflect",
    )
    vessel_positive = vesselness[vesselness > 0.0]
    if vessel_positive.size == 0:
        return []

    vessel_threshold = float(
        np.percentile(vessel_positive, SEISMIC_EVENT_VESSELNESS_PERCENTILE)
    )
    amplitude_threshold = float(
        np.percentile(positive, SEISMIC_EVENT_AMPLITUDE_PERCENTILE)
    )
    mask = (vesselness >= vessel_threshold) & (normalized >= amplitude_threshold)
    mask = morphology.remove_small_objects(mask, max_size=max(8, SEISMIC_EVENT_MIN_LENGTH // 2))
    skeleton = morphology.skeletonize(mask)
    return _tracks_from_skeleton(
        distances,
        depths,
        skeleton,
        vesselness,
        min_length=SEISMIC_EVENT_MIN_LENGTH,
        max_slope=SEISMIC_EVENT_MAX_SLOPE,
    )


def _add_seismic_event_candidates(
    fig: go.Figure,
    distances: np.ndarray,
    depths: np.ndarray,
    section: np.ndarray,
) -> None:
    finite = section[np.isfinite(section)]
    if finite.size == 0:
        return

    polarities = [
        ("ridge", section, "#d62728", "positive Sato ridge"),
        ("valley", -section, "#1f77b4", "negative Sato valley"),
    ]
    for polarity, values, color, label in polarities:
        tracks = _sato_centerline_tracks(distances, depths, values)
        tracks = sorted(tracks, key=lambda item: float(item["score"]), reverse=True)
        for idx, track in enumerate(tracks[:SEISMIC_EVENT_MAX_LINES_PER_POLARITY]):
            fig.add_trace(
                go.Scatter(
                    x=track["x"],
                    y=track["y"],
                    mode="lines",
                    line={"color": color, "width": 2, "dash": "dot"},
                    name=label,
                    legendgroup=f"seismic {polarity}",
                    showlegend=idx == 0,
                    visible=False,
                    hovertemplate=(
                        f"{label}<br>distance=%{{x:.1f}}<br>depth=%{{y:.1f}} m<extra></extra>"
                    ),
                    meta={"seismic_event_candidate": True, "polarity": polarity},
                )
            )


def _selected_layer_paths(layer_dir: Path, layer_names: Iterable[str] | None) -> list[Path]:
    if layer_names is None:
        manifest_path = layer_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            preferred = ["tg", "t7zll", "xt7", "csqc2", "es4qc2"]
            existing = {item["name"] for item in manifest.get("files", [])}
            names = [name for name in preferred if name in existing]
            return [layer_dir / f"{name}.npz" for name in names]
        return sorted(layer_dir.glob("*.npz"))[:5]
    return [layer_dir / f"{name}.npz" for name in layer_names if (layer_dir / f"{name}.npz").exists()]


def _selected_fault_paths(fault_dir: Path, fault_names: Iterable[str] | None = None) -> list[Path]:
    if fault_names is None:
        return sorted(fault_dir.glob("*_layer_compatible.npz"))
    return [
        fault_dir / f"{name}_layer_compatible.npz"
        for name in fault_names
        if (fault_dir / f"{name}_layer_compatible.npz").exists()
    ]


def _add_lith_traces(fig: go.Figure, section_wells: list[SectionWell]) -> None:
    for well in section_wells:
        depths = _samples_to_depth(well.log[:, 0])
        for lith_value in sorted({int(v) for v in well.log[:, 1].tolist()}):
            mask = well.log[:, 1].astype(int) == lith_value
            fig.add_trace(
                go.Scatter(
                    x=np.full(int(mask.sum()), well.distance),
                    y=depths[mask],
                    mode="markers",
                    marker={
                        "symbol": "square",
                        "size": 7,
                        "color": LITH_COLORS.get(lith_value, "#d62728"),
                    },
                    name=f"lith {lith_value}",
                    legendgroup=f"lith {lith_value}",
                    showlegend=well is section_wells[0],
                    hovertemplate=f"{well.info.name}<br>depth=%{{y:.1f}} m<br>lith={lith_value}<extra></extra>",
                    meta={
                        "lith_anchor_x": float(well.distance),
                        "lith_depths": depths[mask].astype(float).tolist(),
                        "lith_values": np.full(int(mask.sum()), lith_value, dtype=np.int32).astype(float).tolist(),
                        "lith_well_name": well.info.name,
                        "lith_class": int(lith_value),
                    },
                )
            )


def _add_por_bar_traces(fig: go.Figure, section_wells: list[SectionWell]) -> None:
    values = np.concatenate([well.log[:, 1] for well in section_wells])
    if values.size == 0:
        return
    vmin = float(np.nanpercentile(values, 2.0))
    vmax = float(np.nanpercentile(values, 98.0))
    total_distance = max(max(well.distance for well in section_wells), 1.0)
    bar_offset = max(0.35, total_distance * 0.004)
    bar_width = max(0.20, total_distance * 0.002)
    for well in section_wells:
        depths = _samples_to_depth(well.log[:, 0])
        x0 = well.distance + bar_offset - bar_width / 2.0
        x1 = well.distance + bar_offset + bar_width / 2.0
        z = np.column_stack([well.log[:, 1], well.log[:, 1]])
        fig.add_trace(
            go.Heatmap(
                x=[x0, x1],
                y=depths,
                z=z,
                colorscale="Viridis",
                zmin=vmin,
                zmax=vmax,
                showscale=well is section_wells[0],
                colorbar={"title": "POR", "x": 1.08, "y": 0.76, "len": 0.34},
                name=f"{well.info.name} POR bar",
                showlegend=False,
                hovertemplate=f"{well.info.name}<br>depth=%{{y:.1f}} m<br>POR=%{{z:.3f}}<extra></extra>",
                meta={
                    "por_anchor_x": float(well.distance),
                    "por_samples": well.log[:, 0].astype(float).tolist(),
                    "por_depths": depths.astype(float).tolist(),
                    "por_values": well.log[:, 1].astype(float).tolist(),
                    "por_well_name": well.info.name,
                },
            )
        )


def _add_por_point_traces(fig: go.Figure, section_wells: list[SectionWell]) -> None:
    for well in section_wells:
        depths = _samples_to_depth(well.log[:, 0])
        fig.add_trace(
            go.Scatter(
                x=np.full(well.log.shape[0], well.distance),
                y=depths,
                mode="markers",
                marker={
                    "size": 4,
                    "color": well.log[:, 1],
                    "colorscale": "Viridis",
                    "showscale": well is section_wells[0],
                    "colorbar": {"title": "POR", "x": 1.08, "y": 0.76, "len": 0.34},
                },
                name=f"{well.info.name} POR points",
                showlegend=False,
                hovertemplate=f"{well.info.name}<br>depth=%{{y:.1f}} m<br>POR=%{{marker.color:.3f}}<extra></extra>",
                meta={
                    "por_anchor_x": float(well.distance),
                    "por_samples": well.log[:, 0].astype(float).tolist(),
                    "por_depths": depths.astype(float).tolist(),
                    "por_values": well.log[:, 1].astype(float).tolist(),
                    "por_well_name": well.info.name,
                },
            )
        )


def _add_por_curve_traces(fig: go.Figure, section_wells: list[SectionWell]) -> None:
    values = np.concatenate([well.log[:, 1] for well in section_wells])
    vmax = float(np.nanpercentile(values, 98.0)) if values.size else 1.0
    scale = max(4.0, max(well.distance for well in section_wells) * 0.035)
    for well in section_wells:
        depths = _samples_to_depth(well.log[:, 0])
        x = well.distance + (well.log[:, 1] / max(vmax, 1e-6)) * scale
        fig.add_trace(
            go.Scatter(
                x=x,
                y=depths,
                mode="lines",
                line={"width": 1.5, "color": "#1f77b4"},
                name=f"{well.info.name} POR curve",
                showlegend=False,
                hovertemplate=f"{well.info.name}<br>depth=%{{y:.1f}} m<br>POR=%{{customdata:.3f}}<extra></extra>",
                customdata=well.log[:, 1],
                meta={
                    "por_anchor_x": float(well.distance),
                    "por_samples": well.log[:, 0].astype(float).tolist(),
                    "por_depths": depths.astype(float).tolist(),
                    "por_values": well.log[:, 1].astype(float).tolist(),
                    "por_well_name": well.info.name,
                },
            )
        )


def _add_por_traces(fig: go.Figure, section_wells: list[SectionWell], por_style: str) -> None:
    if por_style == "curve":
        _add_por_curve_traces(fig, section_wells)
    elif por_style == "points":
        _add_por_point_traces(fig, section_wells)
    else:
        _add_por_bar_traces(fig, section_wells)


def _add_perm_traces(fig: go.Figure, section_wells: list[SectionWell]) -> None:
    values = np.concatenate([well.log[:, 1] for well in section_wells])
    if values.size == 0:
        return
    vmin = float(np.nanpercentile(values, 2.0))
    vmax = float(np.nanpercentile(values, 98.0))
    total_distance = max(max(well.distance for well in section_wells), 1.0)
    bar_offset = max(0.35, total_distance * 0.004)
    bar_width = max(0.20, total_distance * 0.002)
    bar_gap = max(0.04, bar_width * 0.18)
    for well in section_wells:
        depths = _samples_to_depth(well.log[:, 0])
        perm_center = well.distance + bar_offset + bar_width + bar_gap
        x0 = perm_center - bar_width / 2.0
        x1 = perm_center + bar_width / 2.0
        z = np.column_stack([well.log[:, 1], well.log[:, 1]])
        fig.add_trace(
            go.Heatmap(
                x=[x0, x1],
                y=depths,
                z=z,
                colorscale="Plasma",
                zmin=vmin,
                zmax=vmax,
                showscale=well is section_wells[0],
                colorbar={"title": "PERM", "x": 1.15, "y": 0.28, "len": 0.34},
                name=f"{well.info.name} PERM bar",
                showlegend=False,
                hovertemplate=f"{well.info.name}<br>depth=%{{y:.1f}} m<br>PERM=%{{z:.3f}}<extra></extra>",
                meta={
                    "perm_anchor_x": float(well.distance),
                    "perm_samples": well.log[:, 0].astype(float).tolist(),
                    "perm_depths": depths.astype(float).tolist(),
                    "perm_values": well.log[:, 1].astype(float).tolist(),
                    "perm_well_name": well.info.name,
                },
            )
        )


def _build_overview_section_figure(
    overview: go.Figure,
    section_wells: list[SectionWell],
    title: str,
    z_count: int,
) -> go.Figure:
    fig = go.Figure(overview)

    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        line={"color": "red", "width": 3},
        fillcolor="rgba(255,0,0,0)",
        visible=False,
    )

    for well in section_wells:
        fig.add_vline(x=well.distance, line_width=1, line_dash="dot", line_color="black")
        fig.add_annotation(
            x=well.distance,
            y=_sample_to_depth(-12.0),
            text=well.info.name,
            showarrow=False,
            textangle=-35,
            yanchor="bottom",
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        dragmode="drawrect",
        newshape={"line": {"color": "red", "width": 3}, "fillcolor": "rgba(255,0,0,0)"},
        height=850,
        margin={"l": 70, "r": 170, "t": 120, "b": 70},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    fig.update_xaxes(title_text="Distance along selected wells (inline/crossline index units)")
    fig.update_yaxes(
        title_text="Depth (m)",
        range=[_sample_to_depth(z_count - 1), _sample_to_depth(-20.0)],
    )
    seismic_trace_indices = [
        idx for idx, trace in enumerate(fig.data)
        if bool((getattr(trace, "meta", None) or {}).get("section_seismic_color"))
    ]
    wiggle_trace_indices = [
        idx for idx, trace in enumerate(fig.data)
        if bool((getattr(trace, "meta", None) or {}).get("seismic_wiggle"))
    ]
    updatemenus = []
    if seismic_trace_indices:
        updatemenus.append(
            {
                "buttons": [
                    {
                        "label": name,
                        "method": "restyle",
                        "args": [
                            {
                                "colorscale": colorscale,
                                "reversescale": reversescale,
                            },
                            seismic_trace_indices,
                        ],
                    }
                    for name, (colorscale, reversescale) in SEISMIC_COLOR_OPTIONS.items()
                ],
                "direction": "down",
                "showactive": True,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.12,
                "yanchor": "top",
            }
        )
    if seismic_trace_indices or wiggle_trace_indices:
        display_buttons = []
        for display in SEISMIC_DISPLAY_OPTIONS:
            visible = []
            for idx, trace in enumerate(fig.data):
                if idx in seismic_trace_indices:
                    visible.append(_seismic_display_has_color(display))
                elif idx in wiggle_trace_indices:
                    visible.append(_seismic_display_has_wiggle(display))
                else:
                    trace_visible = getattr(trace, "visible", None)
                    visible.append(False if trace_visible is False else True)
            display_buttons.append(
                {
                    "label": display,
                    "method": "restyle",
                    "args": [{"visible": visible}],
                }
            )
        updatemenus.append(
            {
                "buttons": display_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.78,
                "xanchor": "right",
                "y": 1.12,
                "yanchor": "top",
            }
        )
    if updatemenus:
        fig.update_layout(updatemenus=updatemenus)
    return fig


def _zoom_post_script(
    section_payload_json: str = "null",
    dl_backend_url_json: str = "null",
) -> str:
    return """
const plot = document.getElementById('{plot_id}');
const sectionPayload = __SECTION_PAYLOAD__;
const dlBackendUrl = __DL_BACKEND_URL__;
let applyingSectionZoom = false;
const fullX = plot._fullLayout.xaxis.range.slice();
const fullY = plot._fullLayout.yaxis.range.slice();
const persistentShapeCount = (plot.layout.shapes || []).length;
const zoomState = {rect: null, zoom: null};
const lithColors = {
  0: '#8f8f8f',
  1: '#f2c84b',
  3: '#2ca02c',
  4: '#9467bd',
  5: '#17becf',
};

const zoomControls = document.createElement('div');
zoomControls.style.cssText = 'display:flex;align-items:center;gap:10px;margin:8px 0 12px 0;font-family:sans-serif;';
zoomControls.innerHTML = `
  <button type="button" disabled style="padding:6px 12px;cursor:not-allowed;">Open zoom window</button>
  <span style="color:#666;font-size:13px;">Draw a rectangle on the section, then the zoom will open in a new window.</span>
`;
plot.parentNode.insertBefore(zoomControls, plot);
const zoomButton = zoomControls.querySelector('button');
const zoomStatus = zoomControls.querySelector('span');
let pendingRectTimer = null;

zoomButton.addEventListener('click', function() {
  if (!zoomState.rect || !zoomState.zoom) return;
  openZoomWindow(zoomState.rect, zoomState.zoom, false);
});

function hasRange(eventData, axisName) {
  return Object.prototype.hasOwnProperty.call(eventData, `${axisName}.range[0]`)
    && Object.prototype.hasOwnProperty.call(eventData, `${axisName}.range[1]`);
}

function rectFromRelayout(eventData) {
  const x0Key = Object.keys(eventData).find((key) => /^shapes\\[\\d+\\]\\.x0$/.test(key));
  if (!x0Key) {
    if (Array.isArray(eventData.shapes)) {
      for (let idx = eventData.shapes.length - 1; idx >= persistentShapeCount; idx -= 1) {
        const shape = eventData.shapes[idx] || {};
        const x0 = Number(shape.x0);
        const x1 = Number(shape.x1);
        const y0 = Number(shape.y0);
        const y1 = Number(shape.y1);
        if ([x0, x1, y0, y1].every(Number.isFinite)) return {x0, x1, y0, y1};
      }
    }
    return null;
  }
  const match = x0Key.match(/^shapes\\[(\\d+)\\]\\.x0$/);
  const idx = Number(match[1]);
  const shape = (plot.layout.shapes || [])[idx] || {};
  const read = (field) => {
    const key = `shapes[${idx}].${field}`;
    return Object.prototype.hasOwnProperty.call(eventData, key) ? eventData[key] : shape[field];
  };
  const x0 = Number(read('x0'));
  const x1 = Number(read('x1'));
  const y0 = Number(read('y0'));
  const y1 = Number(read('y1'));
  if (![x0, x1, y0, y1].every(Number.isFinite)) return null;
  return {x0, x1, y0, y1};
}

function latestTransientRect() {
  const shapes = plot.layout.shapes || [];
  for (let idx = shapes.length - 1; idx >= persistentShapeCount; idx -= 1) {
    const shape = shapes[idx] || {};
    const x0 = Number(shape.x0);
    const x1 = Number(shape.x1);
    const y0 = Number(shape.y0);
    const y1 = Number(shape.y1);
    if ([x0, x1, y0, y1].every(Number.isFinite)) {
      return {x0, x1, y0, y1};
    }
  }
  return null;
}

function handleCompletedRect(initialRect) {
  const rect = latestTransientRect() || initialRect;
  if (!rect) return;
  const zoom = zoomFromRect(rect.x0, rect.x1, rect.y0, rect.y1);
  applyingSectionZoom = true;
  rememberZoomSelection(rect, zoom);
  openZoomWindow(rect, zoom, true);
  updateMainRectangle(rect).then(function() {
    applyingSectionZoom = false;
  });
}

function zoomFromRect(x0, x1, y0, y1) {
  const topXUnitsPerPx = Math.abs(fullX[1] - fullX[0]) / Math.max(plot._fullLayout.xaxis._length, 1);
  const topYUnitsPerPx = Math.abs(fullY[1] - fullY[0]) / Math.max(plot._fullLayout.yaxis._length, 1);
  const selectedWidthPx = Math.max(Math.abs(x1 - x0) / Math.max(topXUnitsPerPx, 1e-12), 20);
  const selectedHeightPx = Math.max(Math.abs(y1 - y0) / Math.max(topYUnitsPerPx, 1e-12), 20);
  const maxPlotWidth = Math.max(700, Math.min((window.screen && window.screen.availWidth ? window.screen.availWidth - 260 : 1200), 1400));
  const maxPlotHeight = Math.max(360, Math.min((window.screen && window.screen.availHeight ? window.screen.availHeight - 260 : 820), 900));
  let scale = Math.max(1, 520 / selectedWidthPx, 320 / selectedHeightPx);
  scale = Math.min(scale, maxPlotWidth / selectedWidthPx, maxPlotHeight / selectedHeightPx, 8);
  scale = Math.max(scale, 1);
  const plotWidth = Math.round(selectedWidthPx * scale);
  const plotHeight = Math.round(selectedHeightPx * scale);
  const yRange = fullY[0] > fullY[1]
    ? [Math.max(y0, y1), Math.min(y0, y1)]
    : [Math.min(y0, y1), Math.max(y0, y1)];
  return {
    xRange: [Math.min(x0, x1), Math.max(x0, x1)],
    yRange: yRange,
    plotWidth: plotWidth,
    plotHeight: plotHeight,
  };
}

function cloneForZoom(trace, xRange) {
  const cloned = JSON.parse(JSON.stringify(trace));
  delete cloned.xaxis;
  delete cloned.yaxis;

  if (String(cloned.name || '').includes('POR bar')) {
    const meta = cloned.meta || {};
    const y = Array.isArray(meta.por_depths)
      ? meta.por_depths.map(Number)
      : (Array.isArray(meta.por_samples) ? meta.por_samples.map(Number) : Array.from(cloned.y || []));
    const colors = Array.isArray(meta.por_values) ? meta.por_values.map(Number) : [];
    const anchor = Number.isFinite(Number(meta.por_anchor_x))
      ? Number(meta.por_anchor_x)
      : (Array.from(cloned.x || []).reduce((a, b) => a + Number(b), 0) / Math.max((cloned.x || []).length, 1));
    const dx = Math.max(Math.abs(xRange[1] - xRange[0]), 1e-6);
    return {
      type: 'scatter',
      mode: 'markers',
      x: Array.from({length: y.length}, () => anchor + dx * 0.018),
      y: y,
      marker: {
        symbol: 'square',
        size: 9,
        color: colors,
        colorscale: 'Viridis',
        cmin: cloned.zmin,
        cmax: cloned.zmax,
        showscale: cloned.showscale,
        colorbar: cloned.colorbar || {title: 'POR'},
      },
      name: cloned.name,
      showlegend: false,
      customdata: colors,
      hovertemplate: `${meta.por_well_name || cloned.name}<br>depth=%{y:.1f} m<br>POR=%{customdata:.3f}<extra></extra>`,
    };
  }

  if (String(cloned.name || '').includes('PERM bar')) {
    const meta = cloned.meta || {};
    const y = Array.isArray(meta.perm_depths)
      ? meta.perm_depths.map(Number)
      : (Array.isArray(meta.perm_samples) ? meta.perm_samples.map(Number) : Array.from(cloned.y || []));
    const colors = Array.isArray(meta.perm_values) ? meta.perm_values.map(Number) : [];
    const anchor = Number.isFinite(Number(meta.perm_anchor_x))
      ? Number(meta.perm_anchor_x)
      : (Array.from(cloned.x || []).reduce((a, b) => a + Number(b), 0) / Math.max((cloned.x || []).length, 1));
    const dx = Math.max(Math.abs(xRange[1] - xRange[0]), 1e-6);
    return {
      type: 'scatter',
      mode: 'markers',
      x: Array.from({length: y.length}, () => anchor + dx * 0.032),
      y: y,
      marker: {
        symbol: 'square',
        size: 9,
        color: colors,
        colorscale: 'Plasma',
        cmin: cloned.zmin,
        cmax: cloned.zmax,
        showscale: cloned.showscale,
        colorbar: cloned.colorbar || {title: 'PERM'},
      },
      name: cloned.name,
      showlegend: false,
      customdata: colors,
      hovertemplate: `${meta.perm_well_name || cloned.name}<br>depth=%{y:.1f} m<br>PERM=%{customdata:.3f}<extra></extra>`,
    };
  }

  if (cloned.meta && cloned.meta.seismic_event_candidate) {
    cloned.visible = true;
    if (cloned.line) {
      cloned.line.width = 2.5;
    }
    return cloned;
  }

  if (cloned.meta && cloned.meta.seismic_wiggle) {
    if (cloned.meta.wiggle_line && cloned.line) {
      cloned.line.width = 0.9;
    }
    return cloned;
  }

  if (cloned.marker && cloned.marker.showscale) {
    cloned.marker.showscale = false;
  }
  if (Object.prototype.hasOwnProperty.call(cloned, 'showscale')) {
    cloned.showscale = false;
  }
  return cloned;
}

function buildZoomTraces(xRange) {
  return plot.data.map((trace) => cloneForZoom(trace, xRange));
}

function toNumericArray(values) {
  if (values == null) return [];
  if (Array.isArray(values)) return values.map(Number);
  try {
    return Array.from(values).map(Number);
  } catch (error) {
    return [];
  }
}

function hasSectionPayload() {
  return !!(
    sectionPayload
    && Array.isArray(sectionPayload.distances)
    && Array.isArray(sectionPayload.depths)
    && Array.isArray(sectionPayload.section)
  );
}

function payloadRangesOverlap(rect, xs, ys) {
  if (!xs.length || !ys.length) return false;
  const xLow = Math.min(rect.x0, rect.x1);
  const xHigh = Math.max(rect.x0, rect.x1);
  const yLow = Math.min(rect.y0, rect.y1);
  const yHigh = Math.max(rect.y0, rect.y1);
  const payloadXLow = Math.min(xs[0], xs[xs.length - 1]);
  const payloadXHigh = Math.max(xs[0], xs[xs.length - 1]);
  const payloadYLow = Math.min(ys[0], ys[ys.length - 1]);
  const payloadYHigh = Math.max(ys[0], ys[ys.length - 1]);
  return Math.max(xLow, payloadXLow) <= Math.min(xHigh, payloadXHigh)
    && Math.max(yLow, payloadYLow) <= Math.min(yHigh, payloadYHigh);
}

function nearestIndex(values, target) {
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let idx = 0; idx < values.length; idx += 1) {
    const dist = Math.abs(Number(values[idx]) - Number(target));
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = idx;
    }
  }
  return bestIdx;
}

function inRange(value, low, high) {
  const numeric = Number(value);
  return numeric >= low && numeric <= high;
}

function mergeSeed(seedMap, kind, wellName, anchorX, depths, values) {
  const key = `${kind}:${wellName}:${Number(anchorX).toFixed(3)}`;
  if (!seedMap[key]) {
    seedMap[key] = {
      kind: kind,
      wellName: wellName,
      anchorX: Number(anchorX),
      depths: [],
      values: [],
    };
  }
  for (let idx = 0; idx < depths.length; idx += 1) {
    seedMap[key].depths.push(Number(depths[idx]));
    seedMap[key].values.push(Number(values[idx]));
  }
}

function finalizeSeeds(seedMap) {
  return Object.values(seedMap).map((seed) => {
    const pairs = seed.depths.map((depth, idx) => ({depth, value: seed.values[idx]}))
      .filter((item) => Number.isFinite(item.depth) && Number.isFinite(item.value))
      .sort((left, right) => left.depth - right.depth);
    return {
      kind: seed.kind,
      wellName: seed.wellName,
      anchorX: seed.anchorX,
      depths: pairs.map((item) => item.depth),
      values: pairs.map((item) => item.value),
    };
  }).filter((seed) => seed.depths.length > 0);
}

function collectSeedsForRect(rect) {
  const xLow = Math.min(rect.x0, rect.x1);
  const xHigh = Math.max(rect.x0, rect.x1);
  const yLow = Math.min(rect.y0, rect.y1);
  const yHigh = Math.max(rect.y0, rect.y1);
  const porMap = {};
  const permMap = {};
  const lithMap = {};

  plot.data.forEach((trace) => {
    const meta = trace.meta || {};
    if (Array.isArray(meta.por_depths) && Array.isArray(meta.por_values)) {
      const anchor = Number(meta.por_anchor_x);
      if (Number.isFinite(anchor) && inRange(anchor, xLow, xHigh)) {
        const depths = [];
        const values = [];
        for (let idx = 0; idx < meta.por_depths.length; idx += 1) {
          const depth = Number(meta.por_depths[idx]);
          const value = Number(meta.por_values[idx]);
          if (Number.isFinite(depth) && Number.isFinite(value) && inRange(depth, yLow, yHigh)) {
            depths.push(depth);
            values.push(value);
          }
        }
        if (depths.length > 0) {
          mergeSeed(porMap, 'por', meta.por_well_name || trace.name || 'POR', anchor, depths, values);
        }
      }
    }

    if (Array.isArray(meta.perm_depths) && Array.isArray(meta.perm_values)) {
      const anchor = Number(meta.perm_anchor_x);
      if (Number.isFinite(anchor) && inRange(anchor, xLow, xHigh)) {
        const depths = [];
        const values = [];
        for (let idx = 0; idx < meta.perm_depths.length; idx += 1) {
          const depth = Number(meta.perm_depths[idx]);
          const value = Number(meta.perm_values[idx]);
          if (Number.isFinite(depth) && Number.isFinite(value) && inRange(depth, yLow, yHigh)) {
            depths.push(depth);
            values.push(value);
          }
        }
        if (depths.length > 0) {
          mergeSeed(permMap, 'perm', meta.perm_well_name || trace.name || 'PERM', anchor, depths, values);
        }
      }
    }

    if (Array.isArray(meta.lith_depths) && Array.isArray(meta.lith_values)) {
      const anchor = Number(meta.lith_anchor_x);
      if (Number.isFinite(anchor) && inRange(anchor, xLow, xHigh)) {
        const depths = [];
        const values = [];
        for (let idx = 0; idx < meta.lith_depths.length; idx += 1) {
          const depth = Number(meta.lith_depths[idx]);
          const value = Number(meta.lith_values[idx]);
          if (Number.isFinite(depth) && Number.isFinite(value) && inRange(depth, yLow, yHigh)) {
            depths.push(depth);
            values.push(value);
          }
        }
        if (depths.length > 0) {
          mergeSeed(lithMap, 'lith', meta.lith_well_name || trace.name || 'LITH', anchor, depths, values);
        }
      }
    }
  });

  return {
    por: finalizeSeeds(porMap),
    perm: finalizeSeeds(permMap),
    lith: finalizeSeeds(lithMap),
  };
}

function indexRange(values, low, high) {
  const indices = [];
  for (let idx = 0; idx < values.length; idx += 1) {
    const value = Number(values[idx]);
    if (value >= low && value <= high) indices.push(idx);
  }
  if (indices.length > 0) return indices;
  if (values.length === 0) return [];
  return [nearestIndex(values, low), nearestIndex(values, high)].sort((left, right) => left - right);
}

function subsetSectionPayload(rect) {
  if (!hasSectionPayload()) {
    return {subX: [], subY: [], subZ: []};
  }
  const xs = toNumericArray(sectionPayload.distances);
  const ys = toNumericArray(sectionPayload.depths);
  if (!payloadRangesOverlap(rect, xs, ys)) {
    return {subX: [], subY: [], subZ: []};
  }
  const xIndices = indexRange(xs, Math.min(rect.x0, rect.x1), Math.max(rect.x0, rect.x1));
  const yIndices = indexRange(ys, Math.min(rect.y0, rect.y1), Math.max(rect.y0, rect.y1));
  const z = Array.isArray(sectionPayload.section) ? sectionPayload.section : [];
  const subX = xIndices.map((idx) => xs[idx]);
  const subY = yIndices.map((idx) => ys[idx]);
  const subZ = yIndices.map((rowIdx) => {
    const row = Array.isArray(z[rowIdx]) ? z[rowIdx] : Array.from(z[rowIdx] || []);
    return xIndices.map((colIdx) => Number(row[colIdx]));
  });
  return {subX, subY, subZ};
}

function normalizeTrace(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return values.map(() => 0);
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const variance = finite.reduce((sum, value) => sum + (value - mean) * (value - mean), 0) / finite.length;
  const std = Math.max(Math.sqrt(variance), 1e-6);
  return values.map((value) => Number.isFinite(value) ? (value - mean) / std : 0);
}

function movingAverage(values, radius) {
  const out = new Array(values.length).fill(0);
  for (let idx = 0; idx < values.length; idx += 1) {
    let sum = 0;
    let count = 0;
    for (let offset = -radius; offset <= radius; offset += 1) {
      const j = idx + offset;
      if (j < 0 || j >= values.length) continue;
      sum += Number(values[j]);
      count += 1;
    }
    out[idx] = count > 0 ? sum / count : Number(values[idx] || 0);
  }
  return out;
}

function correlationAtLag(seedTrace, targetTrace, center, lag, halfWindow) {
  const seedValues = [];
  const targetValues = [];
  for (let offset = -halfWindow; offset <= halfWindow; offset += 1) {
    const seedIdx = center + lag + offset;
    const targetIdx = center + offset;
    if (seedIdx < 0 || seedIdx >= seedTrace.length || targetIdx < 0 || targetIdx >= targetTrace.length) continue;
    const left = Number(seedTrace[seedIdx]);
    const right = Number(targetTrace[targetIdx]);
    if (!Number.isFinite(left) || !Number.isFinite(right)) continue;
    seedValues.push(left);
    targetValues.push(right);
  }
  if (seedValues.length < 5) return -2;
  let sumX = 0;
  let sumY = 0;
  for (let idx = 0; idx < seedValues.length; idx += 1) {
    sumX += seedValues[idx];
    sumY += targetValues[idx];
  }
  const meanX = sumX / seedValues.length;
  const meanY = sumY / seedValues.length;
  let numerator = 0;
  let denomX = 0;
  let denomY = 0;
  for (let idx = 0; idx < seedValues.length; idx += 1) {
    const dx = seedValues[idx] - meanX;
    const dy = targetValues[idx] - meanY;
    numerator += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }
  if (denomX <= 1e-12 || denomY <= 1e-12) return -2;
  return numerator / Math.sqrt(denomX * denomY);
}

function estimateShiftModel(seedTrace, targetTrace) {
  const depthCount = targetTrace.length;
  const maxLag = Math.max(2, Math.min(18, Math.floor(depthCount * 0.08)));
  const halfWindow = Math.max(3, Math.min(8, Math.floor(depthCount * 0.04)));
  const shifts = new Array(depthCount).fill(0);
  const scores = new Array(depthCount).fill(0);

  for (let depthIdx = 0; depthIdx < depthCount; depthIdx += 1) {
    let bestLag = 0;
    let bestScore = -2;
    for (let lag = -maxLag; lag <= maxLag; lag += 1) {
      const score = correlationAtLag(seedTrace, targetTrace, depthIdx, lag, halfWindow);
      if (score > bestScore) {
        bestScore = score;
        bestLag = lag;
      }
    }
    shifts[depthIdx] = bestLag;
    scores[depthIdx] = Math.max(0, bestScore);
  }

  return {
    shifts: movingAverage(shifts, 3),
    scores: movingAverage(scores, 2).map((value) => Math.max(0.02, Math.min(1.0, value))),
  };
}

function interpolateContinuous(depths, values, queryDepth) {
  if (!depths.length || queryDepth < depths[0] || queryDepth > depths[depths.length - 1]) return NaN;
  if (queryDepth === depths[0]) return Number(values[0]);
  for (let idx = 1; idx < depths.length; idx += 1) {
    const leftDepth = Number(depths[idx - 1]);
    const rightDepth = Number(depths[idx]);
    if (queryDepth > rightDepth) continue;
    const leftValue = Number(values[idx - 1]);
    const rightValue = Number(values[idx]);
    const span = Math.max(rightDepth - leftDepth, 1e-6);
    const t = (queryDepth - leftDepth) / span;
    return leftValue * (1 - t) + rightValue * t;
  }
  return Number(values[values.length - 1]);
}

function sampleLith(depths, values, queryDepth) {
  if (!depths.length || queryDepth < depths[0] || queryDepth > depths[depths.length - 1]) return NaN;
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let idx = 0; idx < depths.length; idx += 1) {
    const dist = Math.abs(Number(depths[idx]) - queryDepth);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = idx;
    }
  }
  return Number(values[bestIdx]);
}

function prepareNormalizedColumns(subZ) {
  if (!subZ.length || !subZ[0].length) return [];
  const columns = [];
  for (let colIdx = 0; colIdx < subZ[0].length; colIdx += 1) {
    const column = subZ.map((row) => Number(row[colIdx]));
    columns.push(normalizeTrace(column));
  }
  return columns;
}

function propagateContinuous(kind, seeds, subset) {
  const {subX, subY, subZ} = subset;
  const columns = prepareNormalizedColumns(subZ);
  if (!columns.length) return null;
  const depthStep = subY.length > 1 ? Math.abs(Number(subY[1]) - Number(subY[0])) : 10.0;
  const sigmaX = Math.max(Math.abs(subX[subX.length - 1] - subX[0]) * 0.25, 1.0);
  const sum = subY.map(() => subX.map(() => 0));
  const weight = subY.map(() => subX.map(() => 0));

  seeds.forEach((seed) => {
    const seedIdx = nearestIndex(subX, seed.anchorX);
    const seedTrace = columns[seedIdx];
    for (let colIdx = 0; colIdx < subX.length; colIdx += 1) {
      const targetTrace = columns[colIdx];
      const shiftModel = estimateShiftModel(seedTrace, targetTrace);
      const lateralWeight = Math.exp(-Math.abs(Number(subX[colIdx]) - Number(seed.anchorX)) / sigmaX);
      for (let rowIdx = 0; rowIdx < subY.length; rowIdx += 1) {
        const sourceDepth = Number(subY[rowIdx]) + Number(shiftModel.shifts[rowIdx]) * depthStep;
        const predicted = interpolateContinuous(seed.depths, seed.values, sourceDepth);
        if (!Number.isFinite(predicted)) continue;
        const localWeight = lateralWeight * Math.pow(Math.max(shiftModel.scores[rowIdx], 0.05), 2.0);
        sum[rowIdx][colIdx] += predicted * localWeight;
        weight[rowIdx][colIdx] += localWeight;
      }
    }
  });

  const prediction = sum.map((row, rowIdx) => row.map((value, colIdx) => {
    const w = weight[rowIdx][colIdx];
    return w > 1e-6 ? value / w : NaN;
  }));
  return {
    kind: kind,
    x: subX,
    y: subY,
    z: prediction,
    seeds: seeds,
  };
}

function propagateLith(seeds, subset) {
  const {subX, subY, subZ} = subset;
  const columns = prepareNormalizedColumns(subZ);
  if (!columns.length) return null;
  const depthStep = subY.length > 1 ? Math.abs(Number(subY[1]) - Number(subY[0])) : 10.0;
  const sigmaX = Math.max(Math.abs(subX[subX.length - 1] - subX[0]) * 0.25, 1.0);
  const votes = subY.map(() => subX.map(() => ({})));

  seeds.forEach((seed) => {
    const seedIdx = nearestIndex(subX, seed.anchorX);
    const seedTrace = columns[seedIdx];
    for (let colIdx = 0; colIdx < subX.length; colIdx += 1) {
      const targetTrace = columns[colIdx];
      const shiftModel = estimateShiftModel(seedTrace, targetTrace);
      const lateralWeight = Math.exp(-Math.abs(Number(subX[colIdx]) - Number(seed.anchorX)) / sigmaX);
      for (let rowIdx = 0; rowIdx < subY.length; rowIdx += 1) {
        const sourceDepth = Number(subY[rowIdx]) + Number(shiftModel.shifts[rowIdx]) * depthStep;
        const predicted = sampleLith(seed.depths, seed.values, sourceDepth);
        if (!Number.isFinite(predicted)) continue;
        const localWeight = lateralWeight * Math.pow(Math.max(shiftModel.scores[rowIdx], 0.05), 2.0);
        const bucket = votes[rowIdx][colIdx];
        const key = String(Math.round(predicted));
        bucket[key] = (bucket[key] || 0) + localWeight;
      }
    }
  });

  const prediction = votes.map((row) => row.map((bucket) => {
    let bestKey = null;
    let bestWeight = -1;
    Object.keys(bucket).forEach((key) => {
      if (bucket[key] > bestWeight) {
        bestWeight = bucket[key];
        bestKey = key;
      }
    });
    return bestKey === null ? NaN : Number(bestKey);
  }));
  return {
    kind: 'lith',
    x: subX,
    y: subY,
    z: prediction,
    seeds: seeds,
  };
}

function discreteLithColorscale(classValues) {
  if (!classValues.length) return [[0, '#8f8f8f'], [1, '#f2c84b']];
  const sortedClasses = Array.from(new Set(classValues.map((value) => Math.round(Number(value))))).sort((left, right) => left - right);
  const minClass = sortedClasses[0];
  const maxClass = sortedClasses[sortedClasses.length - 1];
  if (minClass === maxClass) return [[0, lithColors[minClass] || '#d62728'], [1, lithColors[minClass] || '#d62728']];
  const scale = [];
  const denom = Math.max((maxClass + 0.5) - (minClass - 0.5), 1);
  sortedClasses.forEach((classValue) => {
    const left = ((classValue - 0.5) - (minClass - 0.5)) / denom;
    const right = ((classValue + 0.5) - (minClass - 0.5)) / denom;
    const color = lithColors[classValue] || '#d62728';
    scale.push([Math.max(0, left), color]);
    scale.push([Math.min(1, right), color]);
  });
  return scale;
}

function percentile(values, p) {
  if (!values.length) return NaN;
  const sorted = values.slice().sort((left, right) => left - right);
  const rank = (Math.max(0, Math.min(100, p)) / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return sorted[lower];
  const t = rank - lower;
  return sorted[lower] * (1 - t) + sorted[upper] * t;
}

function seismicUnderlayTrace(data) {
  if (!data.seismic || !data.seismic.subX || !data.seismic.subY || !data.seismic.subZ) return null;
  const seismicValues = data.seismic.subZ.flat().filter((value) => Number.isFinite(value));
  if (!seismicValues.length) return null;
  const lo = percentile(seismicValues, 2);
  const hi = percentile(seismicValues, 98);
  const limit = Math.max(Math.abs(lo), Math.abs(hi), 1e-6);
  return {
    type: 'heatmap',
    x: data.seismic.subX,
    y: data.seismic.subY,
    z: data.seismic.subZ,
    colorscale: 'RdBu',
    reversescale: true,
    zmin: -limit,
    zmax: limit,
    opacity: 0.34,
    showscale: false,
    hoverinfo: 'skip',
  };
}

function faultBarrierTrace(data) {
  const barrier = data.seismic && data.seismic.faultBarrier;
  if (!barrier || !barrier.x || !barrier.y || !barrier.z || !barrier.x.length || !barrier.y.length) return null;
  return {
    type: 'heatmap',
    x: barrier.x,
    y: barrier.y,
    z: barrier.z,
    colorscale: [
      [0.00, 'rgba(0,0,0,0)'],
      [0.58, 'rgba(0,0,0,0)'],
      [0.80, 'rgba(239,68,68,0.22)'],
      [1.00, 'rgba(185,28,28,0.48)'],
    ],
    zmin: 0,
    zmax: 1,
    showscale: false,
    hovertemplate: 'possible discontinuity<br>distance=%{x:.1f}<br>depth=%{y:.1f} m<br>barrier=%{z:.2f}<extra></extra>',
    name: 'fault-aware barrier',
  };
}

function labelConflictTrace(data) {
  const conflict = data.seismic && data.seismic.labelConflict;
  if (!conflict || !conflict.x || !conflict.y || !conflict.z || !conflict.x.length || !conflict.y.length) return null;
  return {
    type: 'heatmap',
    x: conflict.x,
    y: conflict.y,
    z: conflict.z,
    colorscale: [
      [0.00, 'rgba(0,0,0,0)'],
      [0.35, 'rgba(0,0,0,0)'],
      [0.70, 'rgba(245,158,11,0.28)'],
      [1.00, 'rgba(217,119,6,0.62)'],
    ],
    zmin: 0,
    zmax: 1,
    showscale: false,
    hovertemplate: 'label conflict<br>distance=%{x:.1f}<br>depth=%{y:.1f} m<br>conflict=%{z:.2f}<extra></extra>',
    name: 'label conflict',
  };
}

function medianSpacing(values) {
  if (!values || values.length < 2) return 1.0;
  const diffs = [];
  for (let idx = 1; idx < values.length; idx += 1) {
    const diff = Math.abs(Number(values[idx]) - Number(values[idx - 1]));
    if (Number.isFinite(diff) && diff > 0) diffs.push(diff);
  }
  if (!diffs.length) return 1.0;
  diffs.sort((left, right) => left - right);
  return Number(diffs[Math.floor(diffs.length / 2)]) || 1.0;
}

function seedColumnHeatmapTrace(seed, data, kind, style) {
  const xSpacing = medianSpacing(data.x);
  const halfWidth = Math.max(xSpacing * 0.08, 0.24);
  const xLeft = Number(seed.anchorX) - halfWidth;
  const xRight = Number(seed.anchorX) + halfWidth;
  const yValues = seed.depths.map(Number);
  const columnValues = seed.values.map(Number);
  const z = columnValues.map((value) => [value, value]);
  const trace = {
    type: 'heatmap',
    x: [xLeft, xRight],
    y: yValues,
    z: z,
    showscale: false,
    hovertemplate: `${seed.wellName}<br>distance=${Number(seed.anchorX).toFixed(1)}<br>depth=%{y:.1f} m<br>${kind.toUpperCase()}=%{z:.3f}<extra></extra>`,
  };
  if (kind === 'lith') {
    trace.coloraxis = 'coloraxis';
    trace.hovertemplate = `${seed.wellName}<br>distance=${Number(seed.anchorX).toFixed(1)}<br>depth=%{y:.1f} m<br>LITH=%{z:.0f}<extra></extra>`;
    return trace;
  }
  trace.colorscale = style.colorscale;
  trace.zmin = style.cmin;
  trace.zmax = style.cmax;
  return trace;
}

function predictionLayout(title, data, zoom, isLith) {
  const margin = {l: 70, r: 90, t: 70, b: 60};
  const layout = {
    title: {text: title},
    template: 'plotly_white',
    margin: margin,
    autosize: false,
    width: Math.max(560, Math.round(zoom.plotWidth + margin.l + margin.r)),
    height: Math.max(320, Math.round(zoom.plotHeight + margin.t + margin.b)),
    xaxis: {
      title: {text: 'Distance along selected wells (inline/crossline index units)'},
      range: [Math.min(...data.x), Math.max(...data.x)],
    },
    yaxis: {
      title: {text: 'Depth (m)'},
      range: [Math.max(...data.y), Math.min(...data.y)],
    },
  };
  if (isLith) {
    const lithClasses = Array.from(
      new Set(
        data.seeds.flatMap((seed) => seed.values.map((value) => Math.round(Number(value))))
          .concat(data.z.flat().filter((value) => Number.isFinite(value)).map((value) => Math.round(Number(value))))
      )
    ).sort((left, right) => left - right);
    const minClass = lithClasses.length ? lithClasses[0] : 0;
    const maxClass = lithClasses.length ? lithClasses[lithClasses.length - 1] : 1;
    layout.coloraxis = {
      cmin: minClass - 0.5,
      cmax: maxClass + 0.5,
      colorscale: discreteLithColorscale(lithClasses),
      colorbar: {title: '岩性'},
    };
  }
  return layout;
}

function predictionTraces(data) {
  const flatValues = data.z.flat().filter((value) => Number.isFinite(value));
  const seedValues = data.seeds.flatMap((seed) => seed.values.map((value) => Number(value))).filter((value) => Number.isFinite(value));
  const seismicTrace = seismicUnderlayTrace(data);
  const conflictTrace = labelConflictTrace(data);
  if (data.kind === 'lith') {
    return [
      ...(seismicTrace ? [seismicTrace] : []),
      {
        type: 'heatmap',
        x: data.x,
        y: data.y,
        z: data.z,
        coloraxis: 'coloraxis',
        opacity: 0.82,
        hovertemplate: '距离=%{x:.1f}<br>深度=%{y:.1f} 米<br>岩性=%{z:.0f}<extra></extra>',
      },
      ...(conflictTrace ? [conflictTrace] : []),
      ...data.seeds.map((seed) => seedColumnHeatmapTrace(seed, data, 'lith', null)),
    ];
  }

  const colorscale = data.kind === 'perm' ? 'Plasma' : 'Viridis';
  const colorValues = [...flatValues, ...seedValues];
  const cmin = colorValues.length ? percentile(colorValues, 2) : 0;
  const cmax = colorValues.length ? percentile(colorValues, 98) : 1;
  const style = {colorscale, cmin, cmax};
  return [
    ...(seismicTrace ? [seismicTrace] : []),
    {
      type: 'heatmap',
      x: data.x,
      y: data.y,
      z: data.z,
      colorscale: colorscale,
      zmin: cmin,
      zmax: cmax,
      opacity: 0.82,
      colorbar: {title: data.kind === 'perm' ? '渗透率' : '孔隙度'},
      hovertemplate: `距离=%{x:.1f}<br>深度=%{y:.1f} 米<br>${data.kind === 'perm' ? '渗透率' : '孔隙度'}=%{z:.3f}<extra></extra>`,
    },
    ...(conflictTrace ? [conflictTrace] : []),
    ...data.seeds.map((seed) => seedColumnHeatmapTrace(seed, data, data.kind, style)),
  ];
}

function propagationContext(rect) {
  if (!hasSectionPayload()) {
    return {ok: false, message: '无法进行传统传播：当前剖面缺少地震体数据。'};
  }
  const subset = subsetSectionPayload(rect);
  if (!subset.subX.length || !subset.subY.length) {
    return {ok: false, message: '无法进行传统传播：所选框与地震剖面没有重叠区域。'};
  }
  const seeds = collectSeedsForRect(rect);
  const totalSeedWells = seeds.por.length + seeds.perm.length + seeds.lith.length;
  if (totalSeedWells === 0) {
    return {ok: false, message: '所选框内没有井标签。请在框选区域包含岩性、孔隙度或渗透率标签后再点击按钮。'};
  }
  return {
    ok: true,
    subset: subset,
    seeds: seeds,
    counts: {
      por: seeds.por.length,
      perm: seeds.perm.length,
      lith: seeds.lith.length,
    },
  };
}

function buildPropagationResult(kind, context) {
  if (!context.ok) return null;
  if (kind === 'lith') {
    const result = context.seeds.lith.length ? propagateLith(context.seeds.lith, context.subset) : null;
    if (result) result.seismic = context.subset;
    return result;
  }
  if (kind === 'perm') {
    const result = context.seeds.perm.length ? propagateContinuous('perm', context.seeds.perm, context.subset) : null;
    if (result) result.seismic = context.subset;
    return result;
  }
  const result = context.seeds.por.length ? propagateContinuous('por', context.seeds.por, context.subset) : null;
  if (result) result.seismic = context.subset;
  return result;
}

function openPredictionResultWindow(result, label, zoomWindow, zoom, context, titlePrefix, existingWindow) {
  const traces = predictionTraces(result);
  const layout = predictionLayout(titlePrefix, result, zoom, result.kind === 'lith');
  const outerWidth = Math.round(layout.width + 36);
  const outerHeight = Math.round(layout.height + 96);
  const resultWindow = existingWindow && !existingWindow.closed
    ? existingWindow
    : zoomWindow.open('', '_blank', `width=${Math.round(Math.min(outerWidth, 1500))},height=${Math.round(Math.min(Math.max(outerHeight, 420), 1000))},scrollbars=yes,resizable=yes`);
  if (!resultWindow) {
    const message = `The browser blocked the ${label} result window. Please allow pop-ups for this page.`;
    const statusNode = zoomWindow.document.getElementById('modelStatus');
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  const elapsedText = result.meta && Number.isFinite(Number(result.meta.elapsed_seconds))
    ? ` Runtime: ${Number(result.meta.elapsed_seconds).toFixed(2)} s.`
    : '';
  const deviceText = result.meta && result.meta.device
    ? ` Device: ${result.meta.device}.`
    : '';
  const resolutionText = result.meta && result.meta.resolution_mode
    ? ` Resolution: ${result.meta.resolution_mode}, grid ${Array.isArray(result.meta.grid_shape) ? result.meta.grid_shape.join('x') : 'unknown'}, dz=${Number(result.meta.depth_step_m || 0).toFixed(2)} m.`
    : '';

  resultWindow.document.open();
  resultWindow.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>${label} result</title>
  <style>
    html, body { margin: 0; font-family: sans-serif; background: #f8fafc; }
    #resultRoot { padding: 12px; }
    #resultMeta { margin: 0 0 10px 2px; font-size: 14px; color: #334155; }
    #resultPlot { width: ${layout.width}px; height: ${layout.height}px; }
  </style>
</head>
<body>
  <div id="resultRoot">
    <div id="resultMeta">${label} result for x ${Math.min(...result.x).toFixed(1)}-${Math.max(...result.x).toFixed(1)}, depth ${Math.min(...result.y).toFixed(1)}-${Math.max(...result.y).toFixed(1)} m. Seed wells: ${context.counts[result.kind] || context.counts.por}.${elapsedText}${deviceText}${resolutionText}</div>
    <div id="resultPlot"></div>
  </div>
</body>
</html>`);
  resultWindow.document.close();
  resultWindow.Plotly = window.Plotly;
  resultWindow.Plotly.newPlot(
    resultWindow.document.getElementById('resultPlot'),
    traces,
    layout,
    {responsive: false, displaylogo: false, scrollZoom: true}
  );
  return true;
}

function openPropagationWindow(kind, context, zoomWindow, zoom) {
  const label = kind === 'lith' ? 'LITH' : kind.toUpperCase();
  if (!context.ok) {
    const message = context.message || 'Traditional propagation is unavailable for the current box.';
    const statusNode = zoomWindow.document.getElementById('modelStatus');
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  if (!context.counts[kind]) {
    const message = `The selected box does not contain ${label} labels, so ${label} propagation cannot be generated.`;
    const statusNode = zoomWindow.document.getElementById('modelStatus');
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  const result = buildPropagationResult(kind, context);
  if (!result) {
    const message = `${label} labels were found, but the local propagation result could not be built for this box.`;
    const statusNode = zoomWindow.document.getElementById('modelStatus');
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }

  const title = result.kind === 'lith'
    ? 'Traditional propagation: seismic-guided LITH extrapolation'
    : `Traditional propagation: seismic-guided ${result.kind.toUpperCase()} extrapolation`;
  if (!openPredictionResultWindow(result, `${label} propagation`, zoomWindow, zoom, context, title)) {
    return false;
  }

  const statusNode = zoomWindow.document.getElementById('modelStatus');
  if (statusNode) {
    statusNode.textContent = `${label} propagation window opened. Seed wells inside box: ${context.counts[kind]}.`;
    statusNode.style.color = '#166534';
  }
  return true;
}

async function openDlWindow(kind, context, zoomWindow, zoom) {
  const label = kind === 'lith' ? 'LITH' : kind.toUpperCase();
  const statusNode = zoomWindow.document.getElementById('modelStatus');
  if (!context.ok) {
    const message = context.message || `${label} DL modeling is unavailable for the current box.`;
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  if (!dlBackendUrl) {
    const message = `${label} DL modeling is unavailable because the local backend URL is missing.`;
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  if (!context.counts[kind]) {
    const message = `The selected box does not contain ${label} labels, so ${label} DL modeling cannot be generated.`;
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  if (statusNode) {
    statusNode.textContent = `Running local ${label} BiGRU modeling for the selected box...`;
    statusNode.style.color = '#1d4ed8';
  }
  const jobId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const statusUrl = dlBackendUrl.replace(/\\/local-dl-model$/, '/local-dl-status');
  const pendingWindow = zoomWindow.open('', '_blank', 'width=980,height=720,scrollbars=yes,resizable=yes');
  if (!pendingWindow) {
    const message = `The browser blocked the ${label} DL modeling result window. Please allow pop-ups for this page.`;
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
  pendingWindow.document.open();
  pendingWindow.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>${label} DL modeling</title>
  <style>
    html, body { margin: 0; font-family: sans-serif; background: #f8fafc; color: #334155; }
    #pendingRoot { padding: 18px; }
    #pendingTitle { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
    #pendingText { font-size: 14px; line-height: 1.5; margin-bottom: 14px; }
    #progressShell { width: min(720px, calc(100vw - 36px)); height: 16px; border: 1px solid #bfdbfe; border-radius: 999px; background: #eff6ff; overflow: hidden; }
    #progressBar { width: 0%; height: 100%; background: linear-gradient(90deg, #2563eb, #38bdf8); transition: width 220ms ease; }
    #progressMeta { margin-top: 8px; font-size: 13px; color: #475569; }
  </style>
</head>
<body>
  <div id="pendingRoot">
    <div id="pendingTitle">Running ${label} DL modeling...</div>
    <div id="pendingText">The result will appear here when local BiGRU training finishes. Progress is reported from actual backend epochs.</div>
    <div id="progressShell"><div id="progressBar"></div></div>
    <div id="progressMeta">Preparing local training... 0%</div>
  </div>
</body>
</html>`);
  pendingWindow.document.close();
  const updateRealProgress = async () => {
    if (!pendingWindow || pendingWindow.closed) {
      zoomWindow.clearInterval(progressTimer);
      return;
    }
    try {
      const response = await fetch(`${statusUrl}?job_id=${encodeURIComponent(jobId)}`, {cache: 'no-store'});
      if (!response.ok) return;
      const payload = await response.json();
      if (!payload.ok || !payload.status) return;
      const jobStatus = payload.status;
      const progress = Math.max(0, Math.min(100, Number(jobStatus.progress || 0)));
      const currentEpoch = Number(jobStatus.current_epoch || 0);
      const totalEpochs = Number(jobStatus.total_epochs || 0);
      const phase = jobStatus.phase || 'preparing';
      const message = jobStatus.message || `Running local ${label} BiGRU.`;
      const epochText = totalEpochs > 0 ? ` epoch ${currentEpoch}/${totalEpochs}` : '';
      const displayText = `${message}${epochText ? ` (${epochText.trim()})` : ''} ${progress.toFixed(1)}%`;
      const bar = pendingWindow.document.getElementById('progressBar');
      const meta = pendingWindow.document.getElementById('progressMeta');
      if (bar) bar.style.width = `${progress.toFixed(1)}%`;
      if (meta) meta.textContent = displayText;
      if (statusNode) {
        statusNode.textContent = `${label} DL ${phase}: ${progress.toFixed(1)}%${epochText ? `, ${epochText.trim()}` : ''}`;
        statusNode.style.color = jobStatus.state === 'failed' ? '#b45309' : '#1d4ed8';
      }
    } catch (_error) {
      return;
    }
  };
  const progressTimer = zoomWindow.setInterval(updateRealProgress, 500);
  updateRealProgress();
  try {
    const response = await fetch(dlBackendUrl, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        kind: kind,
        job_id: jobId,
        subset: context.subset,
        seeds: context.seeds[kind],
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok || !payload.result) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    const result = payload.result;
    zoomWindow.clearInterval(progressTimer);
    if (pendingWindow && !pendingWindow.closed) {
      const bar = pendingWindow.document.getElementById('progressBar');
      const meta = pendingWindow.document.getElementById('progressMeta');
      if (bar) bar.style.width = '100%';
      if (meta) meta.textContent = 'Local DL modeling complete. 100%';
    }
    if (!openPredictionResultWindow(
      result,
      `${label} DL modeling`,
      zoomWindow,
      zoom,
      context,
      `DL local modeling: BiGRU ${label} prediction`,
      pendingWindow
    )) {
      return false;
    }
    if (statusNode) {
      const elapsedText = result.meta && Number.isFinite(Number(result.meta.elapsed_seconds))
        ? ` Runtime: ${Number(result.meta.elapsed_seconds).toFixed(2)} s.`
        : '';
      statusNode.textContent = `${label} DL modeling window opened. Seed wells inside box: ${context.counts[kind]}.${elapsedText}`;
      statusNode.style.color = '#166534';
    }
    return true;
  } catch (error) {
    zoomWindow.clearInterval(progressTimer);
    const message = `${label} DL modeling failed: ${error && error.message ? error.message : error}`;
    if (pendingWindow && !pendingWindow.closed) {
      pendingWindow.document.open();
      pendingWindow.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>${label} DL modeling failed</title>
  <style>
    html, body { margin: 0; font-family: sans-serif; background: #fff7ed; color: #9a3412; }
    #errorRoot { padding: 18px; }
    #errorTitle { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
    #errorText { font-size: 14px; line-height: 1.5; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div id="errorRoot">
    <div id="errorTitle">${label} DL modeling failed</div>
    <div id="errorText">${message}</div>
  </div>
</body>
</html>`);
      pendingWindow.document.close();
    }
    if (statusNode) {
      statusNode.textContent = message;
      statusNode.style.color = '#b45309';
    }
    zoomWindow.alert(message);
    return false;
  }
}

function setupPropagationControls(zoomWindow, rect, zoom) {
  const statusNode = zoomWindow.document.getElementById('modelStatus');
  const buttonBar = zoomWindow.document.getElementById('modelButtons');
  if (!statusNode || !buttonBar) return;
  const context = propagationContext(rect);
  zoomWindow.__propagationContext = context;

  if (!context.ok) {
    statusNode.textContent = context.message;
    statusNode.style.color = '#b45309';
  } else {
    statusNode.textContent = `Labels detected in this box: LITH ${context.counts.lith}, POR ${context.counts.por}, PERM ${context.counts.perm}. Click a button below to open the corresponding propagation result in a new window.`;
    statusNode.style.color = '#334155';
  }

  [
    {kind: 'lith', label: 'LITH'},
    {kind: 'por', label: 'POR'},
    {kind: 'perm', label: 'PERM'},
  ].forEach((item) => {
    const button = zoomWindow.document.createElement('button');
    const count = context.ok ? context.counts[item.kind] : 0;
    button.textContent = `${item.label} propagation (${count})`;
    button.type = 'button';
    button.style.cssText = 'padding:8px 14px;border:1px solid #cbd5e1;border-radius:8px;background:#ffffff;cursor:pointer;font-size:14px;';
    button.disabled = !context.ok || count <= 0;
    if (button.disabled) {
      button.style.cursor = 'not-allowed';
      button.style.opacity = '0.55';
    }
    button.addEventListener('click', function() {
      openPropagationWindow(item.kind, zoomWindow.__propagationContext || context, zoomWindow, zoom);
    });
    buttonBar.appendChild(button);
  });

  [
    {kind: 'lith', label: 'LITH'},
    {kind: 'por', label: 'POR'},
    {kind: 'perm', label: 'PERM'},
  ].forEach((item) => {
    const dlButton = zoomWindow.document.createElement('button');
    const dlCount = context.ok ? context.counts[item.kind] : 0;
    dlButton.textContent = `${item.label} DL modeling (${dlCount})`;
    dlButton.type = 'button';
    dlButton.style.cssText = 'padding:8px 14px;border:1px solid #93c5fd;border-radius:8px;background:#eff6ff;cursor:pointer;font-size:14px;color:#1d4ed8;font-weight:600;';
    dlButton.disabled = !context.ok || dlCount <= 0 || !dlBackendUrl;
    if (dlButton.disabled) {
      dlButton.style.cursor = 'not-allowed';
      dlButton.style.opacity = '0.55';
    }
    dlButton.addEventListener('click', function() {
      openDlWindow(item.kind, zoomWindow.__propagationContext || context, zoomWindow, zoom);
    });
    buttonBar.appendChild(dlButton);
  });
}

function zoomLayout(rect, zoom) {
  const baseLayout = JSON.parse(JSON.stringify(plot.layout));
  delete baseLayout.shapes;
  delete baseLayout.annotations;
  delete baseLayout.updatemenus;
  const margin = {l: 70, r: 170, t: 24, b: 70};
  baseLayout.title = {text: ''};
  baseLayout.showlegend = false;
  baseLayout.dragmode = 'pan';
  baseLayout.autosize = false;
  baseLayout.width = zoom.plotWidth + margin.l + margin.r;
  baseLayout.height = zoom.plotHeight + margin.t + margin.b;
  baseLayout.margin = margin;
  baseLayout.xaxis = {
    ...(baseLayout.xaxis || {}),
    range: zoom.xRange,
    title: {text: '沿所选井方向的距离（inline/crossline 索引单位）'},
  };
  baseLayout.yaxis = {
    ...(baseLayout.yaxis || {}),
    range: zoom.yRange,
    title: {text: '深度（米）'},
  };
  return baseLayout;
}

function openZoomWindow(rect, zoom, quiet) {
  const windowWidth = Math.min(zoom.plotWidth + 280, window.screen && window.screen.availWidth ? window.screen.availWidth - 80 : zoom.plotWidth + 280);
  const windowHeight = Math.min(zoom.plotHeight + 220, window.screen && window.screen.availHeight ? window.screen.availHeight - 80 : zoom.plotHeight + 220);
  const zoomWindow = window.open('', '_blank', `width=${Math.round(windowWidth)},height=${Math.round(windowHeight)},scrollbars=yes,resizable=yes`);
  if (!zoomWindow) {
    zoomStatus.textContent = '浏览器阻止了自动打开放大窗口。请点击“打开放大窗口”，或允许此页面弹出窗口。';
    if (!quiet) {
      alert('浏览器阻止了放大窗口，请允许此页面弹出窗口。');
    }
    return false;
  }
  const traces = buildZoomTraces(zoom.xRange);
  const layout = zoomLayout(rect, zoom);
  const titleText = `放大井剖面：x ${Math.min(rect.x0, rect.x1).toFixed(1)}-${Math.max(rect.x0, rect.x1).toFixed(1)}，深度 ${Math.min(rect.y0, rect.y1).toFixed(1)}-${Math.max(rect.y0, rect.y1).toFixed(1)} 米`;
  const config = {responsive: false, scrollZoom: true};
  zoomWindow.document.open();
  zoomWindow.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>放大井剖面</title>
  <style>
    html, body { margin: 0; min-width: ${layout.width}px; min-height: ${layout.height}px; font-family: sans-serif; background:#f8fafc; }
    #zoomRoot { padding: 12px; }
    #zoomTitle { margin: 0 0 10px 2px; font-size: 24px; line-height: 1.25; color: #1e3a8a; font-weight: 600; }
    #zoomControlsPanel { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin: 0 0 10px 0; }
    #zoomPlot { width: ${layout.width}px; height: ${layout.height}px; }
    #modelStatus { margin: 0 2px 0 2px; font-size: 14px; color: #475569; flex: 1 1 320px; min-width: 280px; }
    #modelButtons { display:flex; flex-wrap:wrap; gap:10px; margin: 0; flex: 2 1 420px; }
  </style>
</head>
<body>
  <div id="zoomRoot">
    <div id="zoomTitle">${titleText}</div>
    <div id="zoomControlsPanel">
      <div id="modelButtons"></div>
      <div id="modelStatus">Checking whether the selected box contains usable well labels...</div>
    </div>
    <div id="zoomPlot"></div>
  </div>
</body>
</html>`);
  zoomWindow.document.close();
  zoomWindow.Plotly = window.Plotly;
  zoomWindow.Plotly
    .newPlot(zoomWindow.document.getElementById('zoomPlot'), traces, layout, config)
    .then(function() {
      setupPropagationControls(zoomWindow, rect, zoom);
    });
  zoomStatus.textContent = 'Zoom window opened. Use the LITH / POR / PERM buttons in that page to open propagation results for the selected box.';
  return true;
}

function rememberZoomSelection(rect, zoom) {
  zoomState.rect = rect;
  zoomState.zoom = zoom;
  zoomButton.disabled = false;
  zoomButton.style.cursor = 'pointer';
  zoomStatus.textContent = 'Selection ready. If a zoom window did not open automatically, click "Open zoom window".';
}

function updateMainRectangle(rect) {
  const nextShapes = (plot.layout.shapes || [])
    .slice(0, persistentShapeCount)
    .map((shape) => ({...shape}));
  nextShapes[0] = {
    ...nextShapes[0],
    visible: true,
    x0: rect.x0,
    x1: rect.x1,
    y0: rect.y0,
    y1: rect.y1,
  };
  return Plotly.relayout(plot, {
    'xaxis.range': fullX,
    'yaxis.range': fullY,
    'xaxis.autorange': false,
    'yaxis.autorange': false,
    'shapes': nextShapes,
  });
}

function keepOverviewFixed() {
  return Plotly.relayout(plot, {
    'xaxis.range': fullX,
    'yaxis.range': fullY,
    'xaxis.autorange': false,
    'yaxis.autorange': false,
  });
}

plot.on('plotly_relayout', function(eventData) {
  if (applyingSectionZoom) return;
  const rect = rectFromRelayout(eventData);
  if (rect) {
    if (pendingRectTimer !== null) {
      window.clearTimeout(pendingRectTimer);
    }
    pendingRectTimer = window.setTimeout(function() {
      pendingRectTimer = null;
      handleCompletedRect(rect);
    }, 0);
    return;
  }

  if (!hasRange(eventData, 'xaxis') && !eventData['xaxis.autorange']) return;
  applyingSectionZoom = true;
  keepOverviewFixed().then(function() {
    applyingSectionZoom = false;
  });
});
""".replace("__SECTION_PAYLOAD__", section_payload_json).replace("__DL_BACKEND_URL__", dl_backend_url_json)


def build_well_section_html(
    selected_wells: list[str],
    *,
    mode: str,
    coords_csv: Path,
    log_dir: Path,
    layer_dir: Path,
    output_dir: Path,
    seismic_path: Path | None = None,
    z_count: int = 504,
    layer_names: Iterable[str] | None = None,
    fault_dir: Path | None = None,
    por_dir: Path | None = None,
    perm_dir: Path | None = None,
    por_style: str = "条形",
    seismic_colorscale: str = "RdBu",
    seismic_display: str = "彩色",
) -> Path:
    legacy_por_style_map = {"bar": "条形", "points": "点状", "curve": "曲线"}
    por_style = por_style if por_style in POR_STYLES else legacy_por_style_map.get(por_style, "条形")
    seismic_display = _seismic_display_setting(seismic_display)
    value_column = "lith" if mode == "lith" else "por"
    section_wells = _build_section_wells(selected_wells, coords_csv, log_dir, value_column, z_count)
    if len(section_wells) < 2:
        raise ValueError("Select at least two wells with valid logs in the current window.")

    overview = go.Figure()
    section_payload_json = "null"
    dl_backend_url_json = "null"
    tg_distances: np.ndarray | None = None
    tg_depths: np.ndarray | None = None
    tg_path = _tg_layer_path(layer_dir)
    if tg_path is not None:
        tg_distances, tg_inlines, tg_crosslines = _section_sample_points(section_wells)
        tg_samples = _sample_layer_at_points(tg_path, tg_inlines, tg_crosslines)
        tg_depths = _samples_to_depth(tg_samples)

    if seismic_path is not None and seismic_path.exists():
        distances, depths, section, inlines, crosslines = _sample_seismic_section_with_geometry(
            seismic_path,
            section_wells,
        )
        if tg_path is not None:
            tg_distances = distances
            tg_depths = _samples_to_depth(_sample_layer_at_points(tg_path, inlines, crosslines))
            depths, section = _apply_tg_boundary_to_seismic(depths, section, tg_depths)
        _add_sampled_seismic_background(
            overview,
            distances,
            depths,
            section,
            seismic_colorscale,
            seismic_display,
        )
        section_payload_json = _section_payload_json(distances, depths, section)
        dl_backend_url_json = json.dumps(_ensure_local_dl_backend(), ensure_ascii=False)

    display_section_wells = _crop_section_wells_to_tg(section_wells, tg_distances, tg_depths)
    if mode == "lith":
        if display_section_wells:
            _add_lith_traces(overview, display_section_wells)
        if por_dir is not None and por_dir.exists():
            por_section_wells = _build_companion_wells(section_wells, por_dir, "por", z_count)
            por_section_wells = _crop_section_wells_to_tg(por_section_wells, tg_distances, tg_depths)
            if por_section_wells:
                _add_por_traces(overview, por_section_wells, por_style)
        if perm_dir is not None and perm_dir.exists():
            perm_section_wells = _build_companion_wells(section_wells, perm_dir, "perm", z_count)
            perm_section_wells = _crop_section_wells_to_tg(perm_section_wells, tg_distances, tg_depths)
            if perm_section_wells:
                _add_perm_traces(overview, perm_section_wells)
    else:
        if display_section_wells:
            _add_por_traces(overview, display_section_wells, por_style)

    for layer_path in _selected_layer_paths(layer_dir, layer_names):
        xs, zs = _sample_layer_along_section(layer_path, section_wells)
        depth_zs = _samples_to_depth(zs)
        depth_zs = _mask_depths_below_boundary(xs, depth_zs, tg_distances, tg_depths)
        overview.add_trace(
            go.Scatter(
                x=xs,
                y=depth_zs,
                mode="lines",
                line={"width": 4},
                name=layer_path.stem,
                hovertemplate=f"{layer_path.stem}<br>distance=%{{x:.1f}}<br>depth=%{{y:.1f}} m<extra></extra>",
            )
        )
    if fault_dir is not None and fault_dir.exists():
        for fault_path in _selected_fault_paths(fault_dir):
            xs, zs = _sample_layer_along_section(fault_path, section_wells)
            depth_zs = _samples_to_depth(zs)
            depth_zs = _mask_depths_below_boundary(xs, depth_zs, tg_distances, tg_depths)
            finite = np.isfinite(depth_zs)
            if int(np.count_nonzero(finite)) < 3:
                continue
            overview.add_trace(
                go.Scatter(
                    x=xs,
                    y=depth_zs,
                    mode="lines",
                    line={"width": 3, "color": "#ff7f0e", "dash": "dash"},
                    name=fault_path.stem.replace('_layer_compatible', ''),
                    hovertemplate=(
                        f"{fault_path.stem.replace('_layer_compatible', '')}"
                        "<br>distance=%{x:.1f}<br>depth=%{y:.1f} m<extra></extra>"
                    ),
                )
            )

    title = "岩性井剖面" if mode == "lith" else "孔隙度井剖面"
    fig = _build_overview_section_figure(
        overview,
        section_wells,
        title=f"{title}: {' -> '.join(well.info.name for well in section_wells)}",
        z_count=z_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "_".join(_safe_name(well.info.name) for well in section_wells[:5])
    output_path = output_dir / f"well_section_{mode}_{stamp}_{stem}.html"
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        post_script=_zoom_post_script(section_payload_json, dl_backend_url_json),
        config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"], "displaylogo": False},
    )
    return output_path


def attach_well_section_gui(
    *,
    server,
    mode: str,
    available_wells: list[str],
    coords_csv: Path,
    log_dir: Path,
    layer_dir: Path,
    output_dir: Path,
    seismic_path: Path | None = None,
    z_count: int = 504,
    folder_name: str = "井剖面",
    scene_prefix: str = "well-section",
    fault_dir: Path | None = None,
    por_dir: Path | None = None,
    perm_dir: Path | None = None,
    por_style: str = "条形",
    seismic_colorscale: str = "RdBu",
    seismic_display: str = "彩色",
) -> None:
    if len(available_wells) < 2:
        return

    selected: list[str] = []
    picking_enabled = {"value": False}
    target_handles = []
    selected_handles = []
    well_infos = load_well_infos(coords_csv, available_wells)
    value_column = "lith" if mode == "lith" else "por"

    def current_scale() -> tuple[float, float, float]:
        if hasattr(server, "_gui_scale"):
            return tuple(float(x) for x in server._gui_scale.value)
        return (1.0, 1.0, 1.0)

    def init_scale() -> tuple[float, float, float]:
        return tuple(float(x) for x in getattr(server, "init_scale", [1.0, 1.0, 1.0]))

    def scene_position(info: WellInfo, sample: float) -> tuple[float, float, float]:
        base = init_scale()
        scale = current_scale()
        return (
            float(info.inline) * base[0] * scale[0],
            float(info.crossline) * base[1] * scale[1],
            float(sample) * base[2] * scale[2],
        )

    def well_head_sample(info: WellInfo) -> float:
        log_path = log_dir / f"{info.csv_name}.csv"
        if not log_path.exists():
            return 0.0
        log = _load_log(log_path, value_column, z_count)
        if log.size == 0:
            return 0.0
        return float(np.nanmin(log[:, 0]))

    head_samples = {
        name: well_head_sample(info)
        for name, info in well_infos.items()
        if name in available_wells
    }

    def selected_text() -> str:
        return " -> ".join(selected) if selected else "（无）"

    safe_scene_prefix = _safe_name(scene_prefix) or "well-section"
    if por_dir is None and mode == "lith":
        candidate = log_dir.parent.parent / "por"
        if candidate.exists():
            por_dir = candidate
    if fault_dir is None:
        candidate = log_dir.parent.parent / "断层"
        if candidate.exists():
            fault_dir = candidate
    if perm_dir is None and mode == "lith":
        candidate = log_dir.parent.parent / "perm"
        if candidate.exists():
            perm_dir = candidate

    with server.gui.add_folder(folder_name):
        start_button = server.gui.add_button("开始选井")
        stop_button = server.gui.add_button("停止选井")
        selected_handle = server.gui.add_text(
            "已选井",
            initial_value=selected_text(),
            disabled=True,
        )
        status_handle = server.gui.add_text(
            "状态",
            initial_value="请先开始选井，然后点击黄色井口标记。",
            disabled=True,
        )
        por_style_dropdown = server.gui.add_dropdown(
            "孔隙度显示方式",
            options=list(POR_STYLES),
            initial_value=por_style if por_style in POR_STYLES else "条形",
        )
        seismic_cmap_dropdown = server.gui.add_dropdown(
            "地震体配色",
            options=list(SEISMIC_COLOR_OPTIONS.keys()),
            initial_value=seismic_colorscale
            if seismic_colorscale in SEISMIC_COLOR_OPTIONS
            else "RdBu",
        )
        seismic_display_dropdown = server.gui.add_dropdown(
            "地震体显示方式",
            options=list(SEISMIC_DISPLAY_OPTIONS),
            initial_value=_seismic_display_setting(seismic_display),
        )
        remove_button = server.gui.add_button("移除最后一个")
        clear_button = server.gui.add_button("清空")
        generate_button = server.gui.add_button("生成剖面 HTML")
        link_handle = server.gui.add_markdown("")

    def refresh_status(message: str) -> None:
        selected_handle.value = selected_text()
        status_handle.value = message

    def set_target_visibility(visible: bool) -> None:
        for _name, handle in target_handles:
            handle.visible = visible

    def add_well(well: str) -> None:
        if well in selected:
            refresh_status(f"{well} 已经选中过了。")
            return
        selected.append(well)
        refresh_status(f"已添加 {well}。")
        update_selected_markers()

    def remove_last(_event) -> None:
        if selected:
            removed = selected.pop()
            refresh_status(f"已移除 {removed}。")
            update_selected_markers()
        else:
            refresh_status("当前没有可移除的已选井。")

    def clear(_event) -> None:
        selected.clear()
        refresh_status("已清空选择。")
        update_selected_markers()

    def start_picking(_event) -> None:
        picking_enabled["value"] = True
        set_target_visibility(True)
        refresh_status("已开启选井。请按剖面顺序点击黄色井口标记。")

    def stop_picking(_event) -> None:
        picking_enabled["value"] = False
        set_target_visibility(False)
        refresh_status("已停止选井。")

    def generate(_event) -> None:
        if len(selected) < 2:
            refresh_status("请至少先选择两口井。")
            return
        try:
            path = build_well_section_html(
                selected,
                mode=mode,
                coords_csv=coords_csv,
                log_dir=log_dir,
                layer_dir=layer_dir,
                output_dir=output_dir,
                seismic_path=seismic_path,
                z_count=z_count,
                fault_dir=fault_dir,
                por_dir=por_dir,
                perm_dir=perm_dir,
                por_style=str(por_style_dropdown.value),
                seismic_colorscale=str(seismic_cmap_dropdown.value),
                seismic_display=str(seismic_display_dropdown.value),
            )
        except Exception as exc:  # GUI callback should report, not crash the server.
            refresh_status(f"生成失败：{exc}")
            print(f"[井剖面] 生成失败：{exc}")
            return
        uri = path.resolve().as_uri()
        try:
            webbrowser.open(uri, new=2)
            refresh_status(f"已打开：{path.name}")
        except Exception as exc:
            refresh_status(f"已保存，但浏览器打开失败：{exc}")
        link_handle.content = f"[打开已生成剖面]({uri})"
        print(f"[井剖面] 已保存：{path}")

    def update_target_positions() -> None:
        for name, handle in target_handles:
            info = well_infos.get(name)
            if info is None:
                continue
            handle.position = scene_position(info, head_samples.get(name, 0.0))

    def update_selected_markers() -> None:
        for handle in selected_handles:
            handle.remove()
        selected_handles.clear()
        for idx, name in enumerate(selected, start=1):
            info = well_infos.get(name)
            if info is None:
                continue
            pos = scene_position(info, head_samples.get(name, 0.0))
            safe = _safe_name(name)
            sphere = server.scene.add_icosphere(
                f"/{safe_scene_prefix}-selected/{idx}-{safe}",
                radius=0.022,
                color=(255, 60, 40),
                opacity=0.85,
                position=pos,
            )
            label = server.scene.add_label(
                f"/{safe_scene_prefix}-selected-label/{idx}-{safe}",
                f"{idx}",
                position=(pos[0], pos[1], pos[2] - 0.035),
                font_size_mode="screen",
                font_screen_scale=1.4,
                depth_test=False,
                anchor="bottom-center",
            )
            selected_handles.extend([sphere, label])

    for name in available_wells:
        info = well_infos.get(name)
        if info is None:
            continue
        handle = server.scene.add_icosphere(
            f"/{safe_scene_prefix}-pick-targets/{_safe_name(name)}",
            radius=0.018,
            color=(255, 230, 0),
            opacity=0.45,
            position=scene_position(info, head_samples.get(name, 0.0)),
            visible=False,
        )
        handle.on_click(lambda _event, well_name=name: add_well(well_name) if picking_enabled["value"] else None)
        target_handles.append((name, handle))

    if hasattr(server, "_gui_scale"):
        def _on_scale_change(_event) -> None:
            update_target_positions()
            update_selected_markers()

        server._gui_scale.on_update(_on_scale_change)

    start_button.on_click(start_picking)
    stop_button.on_click(stop_picking)
    remove_button.on_click(remove_last)
    clear_button.on_click(clear)
    generate_button.on_click(generate)
