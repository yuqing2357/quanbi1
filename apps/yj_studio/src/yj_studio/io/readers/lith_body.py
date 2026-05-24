from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yj_studio.config.styles import LITH_BODY_STYLE


@dataclass(frozen=True, slots=True)
class LithBodyMesh:
    class_value: int
    class_name: str
    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any]
    path: Path


@dataclass(frozen=True, slots=True)
class LithBodyMeshSummary:
    class_value: int
    class_name: str
    metadata: dict[str, Any]
    path: Path


def discover_lith_body_mesh_summaries(model_dir: Path) -> list[LithBodyMeshSummary]:
    """Discover lithology body meshes without loading their full geometry."""

    summaries: list[LithBodyMeshSummary] = []
    for class_value, style in LITH_BODY_STYLE.items():
        path = model_dir / f"lithology_body_class_{class_value}_{style['slug']}_mesh.npz"
        if not path.exists():
            continue
        metadata: dict[str, Any] = {}
        with np.load(path) as data:
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"]))
        summaries.append(
            LithBodyMeshSummary(
                class_value=int(class_value),
                class_name=str(style["name"]),
                metadata=metadata,
                path=path,
            )
        )
    if not summaries:
        raise FileNotFoundError(f"No lithology body mesh files found in {model_dir}")
    return summaries


def load_lith_body_mesh(
    path: Path,
    *,
    class_value: int | None = None,
    class_name: str | None = None,
) -> LithBodyMesh:
    """Load one lithology body mesh generated from the YJ lithology model."""

    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))
    value = int(class_value if class_value is not None else metadata.get("class_value", 0))
    name = class_name or str(metadata.get("class_name") or LITH_BODY_STYLE.get(value, {}).get("name", ""))
    return LithBodyMesh(
        class_value=value,
        class_name=name,
        vertices=np.asarray(data["vertices"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int32),
        metadata=metadata,
        path=path,
    )
