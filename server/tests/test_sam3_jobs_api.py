from __future__ import annotations

import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
APP_SRC = ROOT / "apps" / "yj_studio" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from yj_studio_server.app import create_app  # noqa: E402
from yj_studio_server.config import ServerConfig  # noqa: E402


def _wait_done(client: TestClient, job_id: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/sam3/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last.get("state") in {"done", "error", "cancelled"}:
            return last
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for job {job_id}: {last}")


class _FakeSAM3Engine:
    def segment(
        self,
        rgb: np.ndarray,
        *,
        text: str = "",
        boxes: list[list[float]] | None = None,
        points: list[list[float]] | None = None,
        point_box_radius_px: float = 8.0,
        confidence: float = 0.4,
    ) -> list[dict[str, Any]]:
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[1:height - 1, 1:width - 1] = True
        return [{"mask": mask, "score": 0.8, "box": [1.0, 1.0, float(width - 1), float(height - 1)]}]


def test_sam3_segment_job_contract(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    runtime_root = tmp_path / "runtime" / "server"
    results_root = data_root / "results"
    volume_dir = data_root / "seismic"
    volume_dir.mkdir(parents=True)
    volume_path = volume_dir / "tiny.npy"
    np.save(volume_path, np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5), allow_pickle=False)
    metadata_path = volume_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seismic_index_origin": {"axis0": 204, "axis1": 0, "sample": 88},
                "scale_axis0_axis1_sample": [2, 2, 5],
                "index_mapping": "seismic_axis = origin + output_index / scale",
            }
        ),
        encoding="utf-8",
    )

    cfg = ServerConfig(
        project_root=tmp_path,
        data_root=data_root,
        runtime_root=runtime_root,
        results_root=results_root,
        project_id="default",
        volumes={
            "tiny": {
                "label": "Tiny",
                "path": "seismic/tiny.npy",
                "metadata_path": "seismic/metadata.json",
                "clim": None,
                "voxel_spacing_m": {"axis0": 12.5 / 3.0, "axis1": 12.5 / 3.0, "sample": 10.0 / 3.0},
            }
        },
        sam3={"checkpoint": "weights/sam3.pt", "results_subdir": "sam3"},
    )
    app = create_app(cfg)
    app.state.sam3 = _FakeSAM3Engine()

    with TestClient(app) as client:
        volumes = client.get("/volumes")
        assert volumes.status_code == 200
        grid_reference = volumes.json()[0]["grid_reference"]
        assert grid_reference["seismic_index_origin"] == {
            "axis0": 204.0,
            "axis1": 0.0,
            "sample": 88.0,
        }
        assert grid_reference["scale_axis0_axis1_sample"] == [2.0, 2.0, 5.0]

        submitted = client.post(
            "/sam3/jobs",
            json={
                "kind": "segment",
                "volume_id": "tiny",
                "axis": "inline",
                "index": 1,
                "prompts": {
                    "text": "sandbody",
                    "boxes": [[1, 1, 3, 4]],
                    "points": [[2, 2]],
                },
                "confidence": 0.4,
                "keep_top_k": 1,
                "target_type": "sandbody",
            },
        )
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]

        status = _wait_done(client, job_id)
        assert status["state"] == "done"

        result_response = client.get(f"/sam3/jobs/{job_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        assert result["volume_id"] == "tiny"
        assert result["project"] == "default"
        assert result["axis"] == "inline"
        assert result["index"] == 1
        assert result["volume_shape"] == [3, 4, 5]
        assert result["grid_reference"] == grid_reference
        assert len(result["candidates"]) == 1
        assert len(result["targets"]) == 1
        assert result["targets"][0]["id"] == "T1"
        assert result["targets"][0]["type"] == "sandbody"
        assert result["targets"][0]["metadata"]["volume_grid"] == grid_reference
        assert result["candidates"][0]["shape"] == [5, 4]
        assert result["candidates"][0]["target_id"] == "T1"
        assert result["candidates"][0]["mask_url"] == f"/sam3/jobs/{job_id}/mask/0"

        mask_response = client.get(result["candidates"][0]["mask_url"])
        assert mask_response.status_code == 200
        mask = np.load(BytesIO(mask_response.content), allow_pickle=False)
        assert mask.shape == (5, 4)
        assert mask.dtype == np.uint8
        assert mask.any()

        targets_response = client.get("/sam3/targets?project=default&volume_id=tiny")
        assert targets_response.status_code == 200
        target_set = targets_response.json()
        assert list(target_set["targets"]) == ["T1"]
        assert target_set["summaries"][0]["id"] == "T1"
        assert target_set["metadata"]["volume_grid"] == grid_reference

        target_mask_response = client.get("/sam3/targets/T1/mask/inline/1?project=default&volume_id=tiny")
        assert target_mask_response.status_code == 200
        target_mask = np.load(BytesIO(target_mask_response.content), allow_pickle=False)
        assert target_mask.shape == (5, 4)

        mask3d_response = client.get("/sam3/targets/T1/mask3d?project=default&volume_id=tiny")
        assert mask3d_response.status_code == 200
        mask3d = np.load(BytesIO(mask3d_response.content), allow_pickle=False)
        assert mask3d.shape == (1, 5, 4)
        assert mask3d_response.headers["X-Mask3D-Index-Lo"] == "1"
        assert mask3d_response.headers["X-Mask3D-Index-Hi"] == "1"
        assert mask3d_response.headers["X-Mask3D-Shape"] == "1,5,4"
        assert mask3d_response.headers["X-Mask3D-Voxel-Count"] == str(int(mask3d.sum()))
        assert mask3d_response.headers["X-Mask3D-Volume-M3"] == f"{float(mask3d.sum()) * (12.5 / 3.0) * (12.5 / 3.0) * (10.0 / 3.0):.12g}"
        assert mask3d_response.headers["X-Mask3D-Voxel-Spacing"] == "4.16666666667,4.16666666667,3.33333333333"
        assert mask3d_response.headers["X-Mask3D-Voxel-Spacing-Source"] == "config"
