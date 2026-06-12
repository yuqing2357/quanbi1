from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FaultMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any]
    path: Path


@dataclass(frozen=True, slots=True)
class FaultMeshSummary:
    name: str
    metadata: dict[str, Any]
    path: Path


def discover_fault_mesh_summaries(fault_dir: Path, target: str = "all") -> list[FaultMeshSummary]:
    """Discover fault mesh files without loading full mesh arrays."""

    npz_paths = (
        sorted(fault_dir.glob("*_mesh.npz")) if target == "all" else [fault_dir / f"{target}_mesh.npz"]
    )
    if not npz_paths:
        raise FileNotFoundError(f"No fault mesh files found in {fault_dir}")

    summaries: list[FaultMeshSummary] = []
    for path in npz_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        metadata: dict[str, Any] = {}
        with np.load(path) as data:
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"]))
        summaries.append(
            FaultMeshSummary(
                name=path.stem.removesuffix("_mesh"),
                metadata=metadata,
                path=path,
            )
        )
    return summaries


def load_fault_mesh(path: Path) -> FaultMesh:
    """Load one fault mesh from an npz file."""

    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))
    return FaultMesh(
        name=path.stem.removesuffix("_mesh"),
        vertices=np.asarray(data["vertices_ijk"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int32),
        metadata=metadata,
        path=path,
    )

