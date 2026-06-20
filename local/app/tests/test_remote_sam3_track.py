from __future__ import annotations

import json
from typing import Any

from yj_studio.ai.remote_client import RemoteSAM3Client
from yj_studio.algorithms.runner import RemoteSAM3TrackTask
from yj_studio.ai.state import AIServiceState


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_remote_sam3_client_submit_track_posts_contract(monkeypatch, qapp) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"job_id": "track-1", "state": "queued"})

    monkeypatch.setattr("yj_studio.ai.remote_client.urlopen", fake_urlopen)
    client = RemoteSAM3Client("http://server:8765", project_id="demo", timeout_s=12.0)

    job_id = client.submit_track(
        volume_id="seismic",
        axis="inline",
        seed=20,
        back=3,
        fwd=4,
        boxes=[(1.0, 2.0, 10.0, 20.0)],
        text="sandbody",
        confidence=0.55,
        keep_top_k=2,
        target_type="sandbody",
    )

    assert job_id == "track-1"
    assert captured["url"] == "http://server:8765/sam3/jobs"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 12.0
    assert captured["body"] == {
        "kind": "track",
        "project": "demo",
        "volume_id": "seismic",
        "axis": "inline",
        "index": {"seed": 20, "back": 3, "fwd": 4},
        "target_type": "sandbody",
        "prompts": {"text": "sandbody", "boxes": [[1.0, 2.0, 10.0, 20.0]]},
        "confidence": 0.55,
        "keep_top_k": 2,
    }


class _FakeTrackClient:
    def __init__(self) -> None:
        self.busy_messages: list[str] = []
        self.ready_count = 0
        self.cancelled: list[str] = []
        self.polls = [
            {"state": "running", "progress": 0.5, "message": "tracking"},
            {"state": "done", "progress": 1.0, "message": "done"},
        ]

    @property
    def state(self):
        return AIServiceState.READY

    def is_ready(self) -> bool:
        return True

    def mark_busy(self, message: str = "运行中") -> None:
        self.busy_messages.append(message)

    def mark_ready(self, message: str = "远程 SAM3 已就绪") -> None:
        self.ready_count += 1

    def submit_track(self, **params) -> str:  # noqa: ANN003
        self.params = params
        return "job-1"

    def poll(self, job_id: str) -> dict[str, Any]:
        assert job_id == "job-1"
        return self.polls.pop(0)

    def result(self, job_id: str) -> dict[str, Any]:
        assert job_id == "job-1"
        return {
            "job_id": job_id,
            "target_ids": ["T1", "T2"],
            "tracking_diagnostics": {
                "requested_frame_count": 11,
                "persisted_target_frames": {"T1": 9, "T2": 7},
            },
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        self.cancelled.append(job_id)
        return {"state": "cancelled"}


def test_remote_sam3_track_task_finishes_with_result(qapp) -> None:
    client = _FakeTrackClient()
    task = RemoteSAM3TrackTask(client, {"volume_id": "seismic", "axis": "inline", "seed": 1})
    progress: list[tuple[float, str]] = []
    finished: list[tuple[dict[str, Any], str]] = []
    task.progress.connect(lambda fraction, message: progress.append((fraction, message)))
    task.finished.connect(lambda result, summary: finished.append((result, summary)))

    task.start()
    task._poll()
    task._poll()

    assert client.busy_messages == ["远程 SAM3 追踪中"]
    assert client.ready_count == 1
    assert progress[0][1] == "已提交远程 SAM3 追踪任务"
    assert finished == [
        (
            {
                "job_id": "job-1",
                "target_ids": ["T1", "T2"],
                "tracking_diagnostics": {
                    "requested_frame_count": 11,
                    "persisted_target_frames": {"T1": 9, "T2": 7},
                },
            },
            "追踪完成：2 个目标；有效帧数：9，7 / 请求 11",
        )
    ]
