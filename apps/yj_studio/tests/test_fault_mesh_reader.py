from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yj_studio.io.readers.fault_mesh import discover_fault_mesh_summaries, load_fault_mesh


class FakeNpz:
    files = ["vertices_ijk", "faces", "metadata_json"]

    def __init__(self) -> None:
        self.values = {
            "vertices_ijk": np.asarray([[0, 0, 1], [1, 0, 2], [0, 1, 3]], dtype=np.float32),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            "metadata_json": json.dumps({"raw_point_count": 3}),
        }

    def __getitem__(self, key: str):
        return self.values[key]

    def __enter__(self) -> "FakeNpz":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_fault_mesh_reader_round_trip(monkeypatch) -> None:
    path = Path("F1-YJ_mesh.npz")

    monkeypatch.setattr(Path, "glob", lambda _self, pattern: [path])
    monkeypatch.setattr(Path, "exists", lambda _self: True)
    monkeypatch.setattr(np, "load", lambda _path: FakeNpz())

    summaries = discover_fault_mesh_summaries(Path("faults"))
    mesh = load_fault_mesh(path)

    assert summaries[0].name == "F1-YJ"
    assert summaries[0].metadata["raw_point_count"] == 3
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
