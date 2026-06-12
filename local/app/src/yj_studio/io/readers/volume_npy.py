from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yj_studio.config.styles import MODEL_VOLUME_DISPLAY_STYLE, VOLUME_DISPLAY_STYLE


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    key: str
    path: Path
    label: str
    cmap: str
    filename: str


def load_available_volume_specs(
    seismic_path: Path,
    attribute_dir: Path,
    model_dir: Path | None = None,
) -> tuple[dict[str, VolumeSpec], list[str]]:
    """Discover available seismic, attribute, and model volumes."""

    specs: dict[str, VolumeSpec] = {}
    notes: list[str] = []

    attribute_dir_exists = attribute_dir.exists()
    for key, style in VOLUME_DISPLAY_STYLE.items():
        path = seismic_path if key == "seismic" else attribute_dir / str(style["filename"])
        if not path.exists():
            if key != "seismic" and attribute_dir_exists:
                notes.append(f"{key}: missing {path.name}, skipped")
            continue
        specs[key] = VolumeSpec(
            key=key,
            path=path,
            label=str(style["label"]),
            cmap=str(style["cmap"]),
            filename=str(style["filename"]),
        )

    if model_dir is not None:
        for key, style in MODEL_VOLUME_DISPLAY_STYLE.items():
            path = model_dir / str(style["filename"])
            if not path.exists():
                notes.append(f"{key}: missing {path.name}, skipped")
                continue
            specs[key] = VolumeSpec(
                key=key,
                path=path,
                label=str(style["label"]),
                cmap=str(style["cmap"]),
                filename=str(style["filename"]),
            )
    return specs, notes


def load_volume_by_key(volume_key: str, volume_specs: dict[str, VolumeSpec]) -> np.memmap:
    spec = volume_specs.get(volume_key)
    if spec is None:
        raise KeyError(volume_key)
    return np.load(spec.path, mmap_mode="r")
