from __future__ import annotations

import json
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QPushButton

from yj_studio.ai.remote_client import RemoteSAM3Client
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


def test_sam3_backend_is_remote_only_without_local_fallback(monkeypatch, qapp) -> None:
    from yj_studio.ui.main_window import _make_sam3_backend

    monkeypatch.delenv("YJ_STUDIO_SAM3_BACKEND", raising=False)
    monkeypatch.delenv("YJ_STUDIO_VOLUME_BACKEND", raising=False)
    monkeypatch.delenv("YJ_STUDIO_SERVER_URL", raising=False)

    client = _make_sam3_backend()
    assert isinstance(client, RemoteSAM3Client)
    assert client.server_url == ""

    client.start()

    assert client.state == AIServiceState.ERROR
    assert "YJ_STUDIO_SERVER_URL" in client.message


def test_remote_sam3_client_submit_segment_posts_jobs_contract(monkeypatch, qapp) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"job_id": "segment-1", "state": "queued"})

    monkeypatch.setattr("yj_studio.ai.remote_client.urlopen", fake_urlopen)
    client = RemoteSAM3Client("http://server:8765", project_id="demo", timeout_s=12.0)

    job_id = client.submit_segment(
        volume_id="model_lithology",
        axis="inline",
        index=2226,
        text="sandbody",
        boxes=[(1.0, 2.0, 10.0, 20.0)],
        points=[(5.0, 6.0)],
        point_box_radius_px=9.0,
        confidence=0.55,
        keep_top_k=2,
        target_type="sandbody",
    )

    assert job_id == "segment-1"
    assert captured["url"] == "http://server:8765/sam3/jobs"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 12.0
    assert captured["body"] == {
        "kind": "segment",
        "project": "demo",
        "volume_id": "model_lithology",
        "axis": "inline",
        "index": 2226,
        "target_type": "sandbody",
        "prompts": {
            "text": "sandbody",
            "boxes": [[1.0, 2.0, 10.0, 20.0]],
            "points": [[5.0, 6.0]],
        },
        "point_box_radius_px": 9.0,
        "confidence": 0.55,
        "keep_top_k": 2,
    }


def test_algorithm_registry_does_not_publish_sam3_algorithms(qapp) -> None:
    from yj_studio.algorithms import builtin as _builtin  # noqa: F401
    from yj_studio.algorithms.registry import registry

    ids = {algorithm.id for algorithm in registry.iter_algorithms()}

    assert not {item for item in ids if item.startswith("ai.sam3.")}


def test_algorithm_dock_hides_sam3_algorithms_if_present(qapp) -> None:
    from yj_studio.algorithms import AlgorithmRunner
    from yj_studio.algorithms.remote_sam3 import RemoteSAM3SegmentAlgorithm
    from yj_studio.algorithms.registry import AlgorithmRegistry
    from yj_studio.scene import LayerStore
    from yj_studio.ui.docks.algorithm_dock import AlgorithmDock

    registry = AlgorithmRegistry()
    registry.register(RemoteSAM3SegmentAlgorithm)
    dock = AlgorithmDock(LayerStore(), registry, AlgorithmRunner())

    assert dock._tree.topLevelItemCount() == 0

    dock.close()


def test_target_dock_has_no_batch_extract_entry(qapp) -> None:
    from yj_studio.scene import LayerStore
    from yj_studio.ui.docks.target_dock import TargetDock

    dock = TargetDock(LayerStore(), None)
    button_texts = {button.text() for button in dock.findChildren(QPushButton)}

    assert "提取" not in button_texts
    assert not hasattr(dock, "_extract_button")

    dock.close()


def test_ai_dock_segment_completion_emits_target_refresh_payload(qapp) -> None:
    from yj_studio.algorithms import AlgorithmRunner
    from yj_studio.scene import LayerStore
    from yj_studio.scene.layers import MaskLayer
    from yj_studio.tools import ToolManager
    from yj_studio.ui.docks.ai_dock import AIDock

    store = LayerStore()
    client = RemoteSAM3Client("http://server:8765")
    client._state = AIServiceState.READY
    dock = AIDock(store, client, AlgorithmRunner(layer_store=store), ToolManager())
    emitted: list[dict[str, object]] = []
    dock.job_finished.connect(lambda payload: emitted.append(dict(payload)))
    layer = MaskLayer(
        name="SAM3 T1",
        mask=np.ones((2, 2), dtype=np.uint8),
        axis="inline",
        slice_index=1,
        metadata={"target_id": "T1"},
    )

    dock._on_finished([layer], "done")

    assert emitted == [{"kind": "segment", "target_ids": ["T1"]}]
    assert next(store.iter_by_type(MaskLayer)) is layer

    dock.close()
