from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from yj_studio.io.readers.lith_body import (
    discover_lith_body_mesh_summaries,
    load_lith_body_mesh,
)


def test_discover_and_load_lith_body_mesh() -> None:
    scratch = Path(__file__).resolve().parent / "_scratch" / uuid4().hex
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "lithology_body_class_1_sandstone_mesh.npz"
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    metadata = {"class_value": 1, "class_name": "砂岩", "stride": [6, 6, 4]}
    np.savez(path, vertices=vertices, faces=faces, metadata_json=json.dumps(metadata))

    summaries = discover_lith_body_mesh_summaries(scratch)
    mesh = load_lith_body_mesh(summaries[0].path)

    assert summaries[0].class_value == 1
    assert summaries[0].class_name == "砂岩"
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    assert mesh.metadata["stride"] == [6, 6, 4]
