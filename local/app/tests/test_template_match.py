"""Desktop side of template-guided 2D search: client contract, task, whiteboard."""

from __future__ import annotations

import json
import os
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import numpy as np

from yj_studio.ai.remote_client import RemoteSAM3Client
from yj_studio.ai.state import AIServiceState
from yj_studio.algorithms.runner import RemoteSAM3TemplateMatchTask
from yj_studio.scene.layers import VolumeLayer
from yj_studio.ui.dialogs.template_canvas_dialog import TemplateCanvasDialog


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_submit_template_match_posts_contract(monkeypatch, qapp) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"job_id": "tpl-1", "state": "queued"})

    monkeypatch.setattr("yj_studio.ai.remote_client.urlopen", fake_urlopen)
    client = RemoteSAM3Client("http://server:8765", project_id="demo")

    job_id = client.submit_template_match(
        volume_id="model_lithology",
        axis="inline",
        index=42,
        template=[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        confidence=0.5,
        keep_top_k=6,
        grid=24,
        target_type="trap",
    )

    assert job_id == "tpl-1"
    assert captured["url"] == "http://server:8765/sam3/jobs"
    assert captured["body"] == {
        "kind": "template_match",
        "project": "demo",
        "volume_id": "model_lithology",
        "axis": "inline",
        "index": 42,
        "template": [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "confidence": 0.5,
        "keep_top_k": 6,
        "target_type": "trap",
        "grid": 24,
    }


class _FakeTemplateClient:
    def __init__(self) -> None:
        self.busy_messages: list[str] = []
        self.ready_count = 0
        self.polls = [
            {"state": "running", "progress": 0.4, "message": "ranking"},
            {"state": "done", "progress": 1.0, "message": "done"},
        ]

    @property
    def state(self):
        return AIServiceState.READY

    def is_ready(self) -> bool:
        return True

    def mark_busy(self, message: str = "运行中") -> None:
        self.busy_messages.append(message)

    def mark_ready(self, message: str = "ready") -> None:
        self.ready_count += 1

    def submit_template_match(self, **params) -> str:  # noqa: ANN003
        self.params = params
        return "tpl-job"

    def poll(self, job_id: str) -> dict[str, Any]:
        return self.polls.pop(0)

    def result(self, job_id: str) -> dict[str, Any]:
        return {
            "kind": "template_match",
            "axis": "inline",
            "index": 7,
            "volume_id": "model_lithology",
            "candidates": [
                {"index": 0, "target_id": "", "score": 0.9, "box": [1, 1, 5, 5]},
                {"index": 1, "target_id": "", "score": 0.6, "box": [2, 2, 6, 6]},
            ],
        }

    def fetch_mask(self, job_id: str, candidate_index: int) -> np.ndarray:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[candidate_index : candidate_index + 3, 1:4] = 1
        return mask


def test_template_match_task_builds_visualisation_layers(qapp) -> None:
    client = _FakeTemplateClient()
    volume = VolumeLayer(name="岩性模型", volume_id="model_lithology", shape=(10, 10, 10))
    task = RemoteSAM3TemplateMatchTask(
        client,
        {
            "axis": "inline",
            "slice_index": 7,
            "template": [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "keep_top_k": 8,
            "name_prefix": "模板",
        },
        {"volume": volume},
    )
    finished: list[tuple[list, str]] = []
    task.finished.connect(lambda layers, summary: finished.append((layers, summary)))

    task.start()
    task._poll()
    task._poll()

    assert client.busy_messages == ["远程形态模板搜索中"]
    assert client.params["volume_id"] == "model_lithology"
    assert len(finished) == 1
    layers, _summary = finished[0]
    assert len(layers) == 2
    # Visualisation-only: candidates carry no target id.
    assert all(layer.metadata.get("target_id", "") == "" for layer in layers)


def test_template_canvas_dialog_returns_normalized_polygon(qapp) -> None:
    dialog = TemplateCanvasDialog()
    ax = dialog._axes

    class _Evt:
        def __init__(self, x, y, button=1):  # noqa: ANN001
            self.xdata = x
            self.ydata = y
            self.button = button
            self.inaxes = ax

    # Free-hand a small wedge on the blank board (data coords already [0, 1]).
    dialog._on_press(_Evt(0.1, 0.1))
    dialog._on_motion(_Evt(0.1, 0.9))
    dialog._on_motion(_Evt(0.9, 0.9))
    dialog._on_release(_Evt(0.5, 0.9))

    points = dialog.template_points()
    assert len(points) >= 3
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
    # Save button only enables once a valid (>=3 point) shape exists.
    assert dialog._save_button.isEnabled()
    # First vertex is where drawing started.
    assert abs(points[0][0] - 0.1) < 1e-6 and abs(points[0][1] - 0.1) < 1e-6


def test_ai_dock_template_targets_reservoir_and_gates_button(qapp) -> None:
    from yj_studio.algorithms import AlgorithmRunner
    from yj_studio.scene import LayerStore
    from yj_studio.tools import ToolManager
    from yj_studio.ui.docks.ai_dock import AIDock

    store = LayerStore()
    store.add(VolumeLayer(name="地震体数据", volume_id="seismic", shape=(10, 10, 10)))
    store.add(VolumeLayer(name="岩性模型", volume_id="model_lithology", shape=(8, 8, 8)))
    store.add(VolumeLayer(name="孔隙度模型", volume_id="model_porosity", shape=(8, 8, 8)))

    client = RemoteSAM3Client("http://server:8765")
    client._state = AIServiceState.READY
    dock = AIDock(store, client, AlgorithmRunner(layer_store=store), ToolManager())

    # The reservoir combo lists only model volumes — seismic is excluded.
    reservoir_ids = {layer.volume_id for layer in dock._reservoir_volume_layers()}
    assert reservoir_ids == {"model_lithology", "model_porosity"}
    combo_ids = {
        dock._template_volume_combo.itemData(i)
        for i in range(dock._template_volume_combo.count())
    }
    assert combo_ids == {"model_lithology", "model_porosity"}

    # Without a saved template the search button stays disabled even when ready.
    assert dock._template is None
    assert not dock._template_search_button.isEnabled()

    # Saving a template (as the whiteboard would) enables the search button.
    dock._template = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    dock._update_template_search_enabled()
    assert dock._template_search_button.isEnabled()
    dock.close()


def test_template_canvas_clear_disables_save(qapp) -> None:
    dialog = TemplateCanvasDialog()
    ax = dialog._axes

    class _Evt:
        def __init__(self, x, y):  # noqa: ANN001
            self.xdata = x
            self.ydata = y
            self.button = 1
            self.inaxes = ax

    dialog._on_press(_Evt(0.2, 0.2))
    dialog._on_motion(_Evt(0.2, 0.8))
    dialog._on_release(_Evt(0.8, 0.8))
    assert dialog._save_button.isEnabled()

    dialog._clear()
    assert not dialog._save_button.isEnabled()
    assert dialog.template_points() == []
