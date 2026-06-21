from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from yj_studio_core.masks import decode_sparse_mask, is_sparse_mask_payload

from .state import AIServiceState


def _decode_mask_bytes(data: bytes, content_type: str = "") -> np.ndarray:
    """Decode a mask response that may be sparse JSON or a dense ``.npy``."""

    if "json" in (content_type or "").lower():
        return decode_sparse_mask(json.loads(data.decode("utf-8")))
    # Probe for a sparse JSON body even when the content-type is unhelpful.
    if data[:1] in (b"{", b"["):
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        if is_sparse_mask_payload(payload):
            return decode_sparse_mask(payload)
    return np.load(BytesIO(data), allow_pickle=False)


@dataclass(frozen=True, slots=True)
class RemoteSAM3Config:
    server_url: str

    @property
    def checkpoint_path(self) -> str:
        return f"{self.server_url.rstrip('/')}/sam3"

    def checkpoint_exists(self) -> bool:
        return True


class RemoteSAM3Client(QObject):
    """Qt-shaped SAM3 service that submits inference jobs to YJ Studio Server."""

    state_changed = pyqtSignal(AIServiceState, str)
    box_prompt_added = pyqtSignal(str, int, float, float, float, float)
    point_prompt_added = pyqtSignal(str, int, float, float)

    def __init__(
        self,
        server_url: str,
        *,
        project_id: str = "default",
        timeout_s: float = 180.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.server_url = server_url.rstrip("/")
        self.project_id = project_id or "default"
        self.timeout_s = float(timeout_s)
        self._state = AIServiceState.IDLE
        self._message = "远程 SAM3 未连接"
        self._config = RemoteSAM3Config(self.server_url)

    @property
    def state(self) -> AIServiceState:
        return self._state

    @property
    def message(self) -> str:
        return self._message

    @property
    def config(self) -> RemoteSAM3Config:
        return self._config

    def is_ready(self) -> bool:
        return self._state == AIServiceState.READY

    def start(self) -> None:
        if not self.server_url:
            self._set_state(AIServiceState.ERROR, "未配置远程 SAM3 服务器（YJ_STUDIO_SERVER_URL）")
            return
        self._set_state(AIServiceState.LOADING, "正在连接远程 SAM3 服务")
        try:
            health = self._get_json("/health", timeout_s=min(self.timeout_s, 10.0))
        except (OSError, RuntimeError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self._set_state(AIServiceState.ERROR, f"远程服务器不可用：{exc}")
            return
        status = health.get("status", "unknown") if isinstance(health, dict) else "unknown"
        self._set_state(AIServiceState.READY, f"远程 SAM3 已就绪（server={status}）")

    def shutdown(self) -> None:
        self._set_state(AIServiceState.IDLE, "远程 SAM3 已断开")

    def mark_busy(self, message: str = "运行中") -> None:
        if self._state == AIServiceState.READY:
            self._set_state(AIServiceState.BUSY, message)

    def mark_ready(self, message: str = "远程 SAM3 已就绪") -> None:
        if self._state == AIServiceState.BUSY:
            self._set_state(AIServiceState.READY, message)

    def emit_box_prompt(
        self,
        axis: str,
        slice_index: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        self.box_prompt_added.emit(
            axis, int(slice_index), float(x_min), float(y_min), float(x_max), float(y_max)
        )

    def emit_point_prompt(self, axis: str, slice_index: int, x: float, y: float) -> None:
        self.point_prompt_added.emit(axis, int(slice_index), float(x), float(y))

    def submit_segment(
        self,
        *,
        volume_id: str,
        axis: str,
        index: int,
        text: str = "",
        boxes: list[tuple[float, float, float, float]] | None = None,
        points: list[tuple[float, float]] | None = None,
        point_box_radius_px: float = 8.0,
        confidence: float = 0.4,
        keep_top_k: int = 3,
        target_type: str = "unknown",
        box_strict: bool = False,
    ) -> str:
        body = {
            "kind": "segment",
            "project": self.project_id,
            "volume_id": volume_id,
            "axis": axis,
            "index": int(index),
            "target_type": target_type or "unknown",
            "prompts": {
                "text": text,
                "boxes": [list(box) for box in (boxes or [])],
                "points": [list(point) for point in (points or [])],
            },
            "point_box_radius_px": float(point_box_radius_px),
            "confidence": float(confidence),
            "keep_top_k": int(keep_top_k),
            "box_strict": bool(box_strict),
        }
        payload = self._post_json("/sam3/jobs", body)
        job_id = str(payload.get("job_id", ""))
        if not job_id:
            raise RuntimeError("Remote SAM3 server did not return a job_id")
        return job_id

    def submit_track(
        self,
        *,
        volume_id: str,
        axis: str,
        seed: int,
        back: int,
        fwd: int,
        boxes: list[tuple[float, float, float, float]] | None = None,
        text: str = "",
        confidence: float = 0.4,
        keep_top_k: int = 3,
        target_type: str = "unknown",
        box_strict: bool = False,
        auto: bool = False,
    ) -> str:
        body = {
            "kind": "track",
            "project": self.project_id,
            "volume_id": volume_id,
            "axis": axis,
            "index": {
                "seed": int(seed),
                "back": int(back),
                "fwd": int(fwd),
            },
            "target_type": target_type or "unknown",
            "prompts": {
                "text": text,
                "boxes": [list(box) for box in (boxes or [])],
            },
            "confidence": float(confidence),
            "keep_top_k": int(keep_top_k),
            "box_strict": bool(box_strict),
            "auto": bool(auto),
        }
        payload = self._post_json("/sam3/jobs", body)
        job_id = str(payload.get("job_id", ""))
        if not job_id:
            raise RuntimeError("Remote SAM3 server did not return a job_id for track")
        return job_id

    def poll(self, job_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/sam3/jobs/{job_id}", timeout_s=self.timeout_s)
        if not isinstance(payload, dict):
            raise ValueError("SAM3 job status must be a JSON object")
        return payload

    def result(self, job_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/sam3/jobs/{job_id}/result", timeout_s=self.timeout_s)
        if not isinstance(payload, dict):
            raise ValueError("SAM3 job result must be a JSON object")
        return payload

    def fetch_mask(self, job_id: str, candidate_index: int) -> np.ndarray:
        # Ask for the bbox/bit-packed sparse payload (tiny vs. the dense
        # full-slice .npy); transparently fall back to dense if an older server
        # ignores the query and streams raw bytes.
        with urlopen(
            self._url(f"/sam3/jobs/{job_id}/mask/{int(candidate_index)}?format=sparse"),
            timeout=self.timeout_s,
        ) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
        return _decode_mask_bytes(data, content_type)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._post_json(f"/sam3/jobs/{job_id}/cancel", {})

    def _set_state(self, state: AIServiceState, message: str) -> None:
        self._state = state
        self._message = message
        self.state_changed.emit(state, message)

    def _url(self, path: str) -> str:
        if not self.server_url:
            raise RuntimeError("未配置远程 SAM3 服务器（YJ_STUDIO_SERVER_URL）")
        return f"{self.server_url}{path}"

    def _get_json(self, path: str, *, timeout_s: float) -> Any:
        with urlopen(self._url(path), timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON response from {path}")
        return data
