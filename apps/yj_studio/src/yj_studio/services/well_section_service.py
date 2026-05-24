from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yj_studio.data.volume_store import VolumeStore
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import VolumeLayer, WellLayer, WellLogLayer


@dataclass(frozen=True, slots=True)
class WellSectionWell:
    name: str
    layer_id: str
    distance: float
    inline: float
    xline: float
    logs: tuple[WellLogLayer, ...]


@dataclass(frozen=True, slots=True)
class WellSectionData:
    names: tuple[str, ...]
    mode: str
    distances: np.ndarray
    depths_m: np.ndarray
    seismic: np.ndarray
    wells: tuple[WellSectionWell, ...]


def build_well_section_data(
    layer_store: LayerStore,
    volume_store: VolumeStore,
    volume_layer: VolumeLayer,
    selected_wells: list[str],
    *,
    mode: str,
    max_trace_count: int = 900,
) -> WellSectionData:
    """Build an in-application connected-well section from scene layers."""

    if volume_layer.shape is None:
        raise ValueError("Load a volume before opening a well section.")
    well_layers = _selected_well_layers(layer_store, selected_wells)
    if len(well_layers) < 2:
        raise ValueError("Select at least two wells.")

    section_wells = _section_wells(layer_store, well_layers, mode)
    distances = np.asarray([well.distance for well in section_wells], dtype=np.float32)
    trace_distances, inlines, xlines = _sample_section_polyline(section_wells, max_trace_count)
    volume = volume_store.get_volume(volume_layer.volume_id)
    nx, ny, nz = volume_layer.shape
    ii = np.clip(np.rint(inlines).astype(np.int64), 0, nx - 1)
    jj = np.clip(np.rint(xlines).astype(np.int64), 0, ny - 1)
    seismic = np.asarray(volume[ii, jj, :], dtype=np.float32).T
    depths_m = np.arange(nz, dtype=np.float32) * 10.0
    return WellSectionData(
        names=tuple(well.name for well in section_wells),
        mode=mode,
        distances=trace_distances if trace_distances.size else distances,
        depths_m=depths_m,
        seismic=seismic,
        wells=tuple(section_wells),
    )


def _selected_well_layers(layer_store: LayerStore, selected_wells: list[str]) -> list[WellLayer]:
    by_name = {
        (layer.well_name or layer.name): layer
        for layer in layer_store.iter_by_type(WellLayer)
        if layer.head_position is not None
    }
    return [by_name[name] for name in selected_wells if name in by_name]


def _section_wells(
    layer_store: LayerStore,
    well_layers: list[WellLayer],
    mode: str,
) -> list[WellSectionWell]:
    section_wells: list[WellSectionWell] = []
    distance = 0.0
    previous: WellLayer | None = None
    for layer in well_layers:
        if layer.head_position is None:
            continue
        if previous is not None and previous.head_position is not None:
            distance += float(
                np.hypot(
                    float(layer.head_position[0]) - float(previous.head_position[0]),
                    float(layer.head_position[1]) - float(previous.head_position[1]),
                )
            )
        name = layer.well_name or layer.name
        section_wells.append(
            WellSectionWell(
                name=name,
                layer_id=layer.id,
                distance=distance,
                inline=float(layer.head_position[0]),
                xline=float(layer.head_position[1]),
                logs=tuple(_logs_for_well(layer_store, name, mode)),
            )
        )
        previous = layer
    return section_wells


def _logs_for_well(layer_store: LayerStore, well_name: str, mode: str) -> list[WellLogLayer]:
    logs: list[WellLogLayer] = []
    for layer in layer_store.iter_by_type(WellLogLayer):
        if layer.well_name == well_name and layer.mode == mode and layer.samples is not None:
            logs.append(layer)
    return logs


def _sample_section_polyline(
    wells: list[WellSectionWell],
    max_trace_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(wells) < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    total = max(float(wells[-1].distance), 1.0)
    distances: list[float] = []
    inlines: list[float] = []
    xlines: list[float] = []
    for left, right in zip(wells[:-1], wells[1:]):
        span = max(float(right.distance - left.distance), 1.0)
        count = max(2, int(round(float(max_trace_count) * span / total)))
        for idx in range(count):
            frac = float(idx) / float(max(1, count - 1))
            distances.append(float(left.distance) + span * frac)
            inlines.append(float(left.inline) + (float(right.inline) - float(left.inline)) * frac)
            xlines.append(float(left.xline) + (float(right.xline) - float(left.xline)) * frac)
    return (
        np.asarray(distances, dtype=np.float32),
        np.asarray(inlines, dtype=np.float32),
        np.asarray(xlines, dtype=np.float32),
    )
