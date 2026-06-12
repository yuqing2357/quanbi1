from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LayerGrid:
    name: str
    sample: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any]
    path: Path


@dataclass(frozen=True, slots=True)
class LayerGridSummary:
    name: str
    metadata: dict[str, Any]
    path: Path


def load_layers(layer_dir: Path, target: str = "all") -> list[LayerGrid]:
    npz_paths = sorted(layer_dir.glob("*.npz")) if target == "all" else [layer_dir / f"{target}.npz"]
    if target == "all":
        npz_paths = [path for path in npz_paths if not path.stem.endswith("_fault")]
    if not npz_paths:
        raise FileNotFoundError(f"No layer files found in {layer_dir}")

    layers: list[LayerGrid] = []
    for path in npz_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        metadata: dict[str, Any] = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"]))
        layers.append(
            LayerGrid(
                name=path.stem,
                sample=data["sample"].astype(np.float32),
                mask=data["mask"].astype(bool),
                metadata=metadata,
                path=path,
            )
        )
    return layers


def load_layer_grid(path: Path) -> LayerGrid:
    """Load one horizon/layer grid from an npz file."""

    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))
    return LayerGrid(
        name=path.stem,
        sample=data["sample"].astype(np.float32),
        mask=data["mask"].astype(bool),
        metadata=metadata,
        path=path,
    )


def discover_layer_summaries(layer_dir: Path, target: str = "all") -> list[LayerGridSummary]:
    """Discover layer files without loading large sample/mask arrays."""

    npz_paths = sorted(layer_dir.glob("*.npz")) if target == "all" else [layer_dir / f"{target}.npz"]
    if target == "all":
        npz_paths = [path for path in npz_paths if not path.stem.endswith("_fault")]
    if not npz_paths:
        raise FileNotFoundError(f"No layer files found in {layer_dir}")

    summaries: list[LayerGridSummary] = []
    for path in npz_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        metadata: dict[str, Any] = {}
        with np.load(path) as data:
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"]))
        summaries.append(LayerGridSummary(name=path.stem, metadata=metadata, path=path))
    return summaries
