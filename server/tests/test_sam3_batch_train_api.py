from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
APP_SRC = ROOT / "apps" / "yj_studio" / "src"
for path in (SERVER_SRC, APP_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
        return [{"mask": mask, "score": 0.75, "box": [1.0, 1.0, float(width - 1), float(height - 1)]}]


def _make_client(tmp_path: Path) -> TestClient:
    data_root = tmp_path / "data"
    runtime_root = tmp_path / "runtime" / "server"
    results_root = data_root / "results"
    volume_dir = data_root / "seismic"
    volume_dir.mkdir(parents=True)
    np.save(volume_dir / "tiny.npy", np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6), allow_pickle=False)
    cfg = ServerConfig(
        project_root=tmp_path,
        project_id="default",
        data_root=data_root,
        runtime_root=runtime_root,
        results_root=results_root,
        volumes={"tiny": {"label": "Tiny", "path": "seismic/tiny.npy", "clim": None}},
        sam3={
            "checkpoint": "weights/sam3.pt",
            "results_subdir": "sam3",
            "gpu_ids": [0, 1, 2, 3],
            "worker_count": 4,
        },
        training={"dataset_subdir": "sam3/datasets", "models_subdir": "sam3/models"},
    )
    app = create_app(cfg)
    app.state.sam3 = _FakeSAM3Engine()
    return TestClient(app)


def test_batch_targets_train_and_model_registry(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        gpus = client.get("/sam3/gpus")
        assert gpus.status_code == 200
        assert gpus.json()["worker_count"] == 4

        submitted = client.post(
            "/sam3/jobs/batch",
            json={
                "project": "default",
                "volume_id": "tiny",
                "axis": "inline",
                "indices": [0, 1],
                "target_type": "fault",
                "keep_top_k": 1,
            },
        )
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]
        status = _wait_done(client, job_id)
        assert status["state"] == "done"

        result = client.get(f"/sam3/jobs/{job_id}/result").json()
        assert result["kind"] == "batch"
        assert result["target_ids"] == ["T1", "T2"]

        infer = client.post(
            "/sam3/jobs",
            json={
                "kind": "infer_volume",
                "project": "default",
                "volume_id": "tiny",
                "axis": "inline",
                "indices": [2],
                "target_type": "sandbody",
                "keep_top_k": 1,
            },
        )
        assert infer.status_code == 200
        infer_id = infer.json()["job_id"]
        infer_status = _wait_done(client, infer_id)
        assert infer_status["state"] == "done"
        infer_result = client.get(f"/sam3/jobs/{infer_id}/result").json()
        assert infer_result["kind"] == "infer_volume"
        assert infer_result["target_ids"] == ["T3"]
        targets = client.get("/sam3/targets?project=default&volume_id=tiny").json()
        assert targets["targets"]["T3"]["status"] == "to_review"

        confirmed = client.patch("/sam3/targets/T1?project=default&volume_id=tiny", json={"status": "confirmed"})
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == "confirmed"

        train = client.post("/sam3/train/jobs", json={"project": "default", "volume_id": "tiny"})
        assert train.status_code == 200
        train_id = train.json()["job_id"]
        train_status = _wait_done(client, train_id)
        assert train_status["state"] == "done"
        train_response = client.get(f"/sam3/train/jobs/{train_id}")
        assert train_response.status_code == 200
        dataset_path = Path(train_response.json()["result"]["dataset_path"])
        assert (dataset_path / "annotations.json").exists()

        models = client.get("/sam3/models")
        assert models.status_code == 200
        model_id = models.json()["models"][0]["id"]
        activated = client.post(f"/sam3/models/{model_id}/activate")
        assert activated.status_code == 200
        assert activated.json()["active_model"] == model_id
