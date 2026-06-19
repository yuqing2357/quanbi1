from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np

from yj_studio_core.targets import GeoTarget, TargetSet


@dataclass(frozen=True, slots=True)
class Mask3DResult:
    mask: np.ndarray
    index_lo: int | None = None
    index_hi: int | None = None
    voxel_count: int | None = None
    volume_m3: float | None = None
    voxel_spacing: tuple[float, float, float] | None = None
    voxel_spacing_source: str | None = None


class RemoteTargetStore:
    """Client for target/mask/model APIs served by YJ Studio Server."""

    def __init__(
        self,
        server_url: str,
        *,
        project_id: str = "default",
        timeout_s: float = 180.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.project_id = project_id or "default"
        self.timeout_s = float(timeout_s)

    def load_targets(self, *, volume_id: str | None = None, include_deleted: bool = False) -> TargetSet:
        payload = self._get_json(
            "/sam3/targets",
            query={"project": self.project_id, "volume_id": volume_id, "include_deleted": include_deleted},
        )
        if not isinstance(payload, dict):
            raise ValueError("Target list response must be a JSON object")
        return TargetSet.model_validate(payload)

    def summaries(self, *, volume_id: str | None = None, include_deleted: bool = False) -> list[dict[str, Any]]:
        payload = self._get_json(
            "/sam3/targets",
            query={"project": self.project_id, "volume_id": volume_id, "include_deleted": include_deleted},
        )
        if not isinstance(payload, dict):
            return []
        summaries = payload.get("summaries", [])
        return summaries if isinstance(summaries, list) else []

    def fetch_target(self, target_id: str, *, volume_id: str | None = None) -> GeoTarget:
        payload = self._get_json(
            f"/sam3/targets/{quote(target_id)}",
            query={"project": self.project_id, "volume_id": volume_id},
        )
        if not isinstance(payload, dict):
            raise ValueError("Target response must be a JSON object")
        return GeoTarget.model_validate(payload)

    def fetch_mask(
        self,
        target_id: str,
        axis: str,
        index: int,
        *,
        volume_id: str | None = None,
    ) -> np.ndarray:
        return self._get_npy(
            f"/sam3/targets/{quote(target_id)}/mask/{quote(axis)}/{int(index)}",
            query={"project": self.project_id, "volume_id": volume_id},
        )

    def fetch_cells(self, target_id: str, *, volume_id: str | None = None) -> np.ndarray:
        return self._get_npy(
            f"/sam3/targets/{quote(target_id)}/cells",
            query={"project": self.project_id, "volume_id": volume_id},
        )

    def fetch_mask3d(self, target_id: str, *, volume_id: str | None = None) -> np.ndarray:
        return self.fetch_mask3d_with_metadata(target_id, volume_id=volume_id).mask

    def fetch_mask3d_with_metadata(self, target_id: str, *, volume_id: str | None = None) -> Mask3DResult:
        return self._get_npy_with_headers(
            f"/sam3/targets/{quote(target_id)}/mask3d",
            query={"project": self.project_id, "volume_id": volume_id},
        )

    def put_mask(
        self,
        target_id: str,
        axis: str,
        index: int,
        mask: np.ndarray,
        *,
        volume_id: str | None = None,
    ) -> GeoTarget:
        buffer = BytesIO()
        np.save(buffer, np.asarray(mask), allow_pickle=False)
        payload = self._request_json(
            f"/sam3/targets/{quote(target_id)}/mask/{quote(axis)}/{int(index)}",
            method="PUT",
            payload_bytes=buffer.getvalue(),
            query={"project": self.project_id, "volume_id": volume_id},
            content_type="application/x-npy",
        )
        return GeoTarget.model_validate(payload)

    def create_cell_target(
        self,
        cells: np.ndarray,
        *,
        axis: str,
        index: int,
        index_hi: int | None = None,
        volume_id: str | None = None,
        target_type: str = "sandbody",
        name: str | None = None,
        source: str = "sam3_reservoir",
        grid_id: str | None = None,
        grid_layer_id: str | None = None,
    ) -> GeoTarget:
        arr = np.asarray(cells, dtype=np.int32)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"cells must have shape (N, 3), got {arr.shape}")
        buffer = BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        payload = self._request_json(
            "/sam3/targets/cells",
            method="POST",
            payload_bytes=buffer.getvalue(),
            query={
                "project": self.project_id,
                "volume_id": volume_id,
                "axis": axis,
                "index": int(index),
                "index_hi": int(index_hi) if index_hi is not None else None,
                "target_type": target_type,
                "name": name,
                "source": source,
                "grid_id": grid_id,
                "grid_layer_id": grid_layer_id,
            },
            content_type="application/x-npy",
        )
        return GeoTarget.model_validate(payload)

    def patch_target(self, target_id: str, updates: dict[str, Any]) -> GeoTarget:
        payload = self._request_json(
            f"/sam3/targets/{quote(target_id)}",
            method="PATCH",
            payload=updates,
            query={"project": self.project_id},
        )
        return GeoTarget.model_validate(payload)

    def delete_target(self, target_id: str) -> GeoTarget:
        payload = self._request_json(
            f"/sam3/targets/{quote(target_id)}",
            method="DELETE",
            payload=None,
            query={"project": self.project_id},
        )
        return GeoTarget.model_validate(payload)

    def merge_targets(self, target_ids: list[str]) -> GeoTarget:
        payload = self._request_json(
            "/sam3/targets/merge",
            method="POST",
            payload={"target_ids": target_ids},
            query={"project": self.project_id},
        )
        return GeoTarget.model_validate(payload)

    def split_target(self, target_id: str, groups: list[list[str]] | None = None) -> dict[str, Any]:
        return self._request_json(
            f"/sam3/targets/{quote(target_id)}/split",
            method="POST",
            payload={"groups": groups or []},
            query={"project": self.project_id},
        )

    def gpus(self) -> dict[str, Any]:
        payload = self._get_json("/sam3/gpus")
        return payload if isinstance(payload, dict) else {}

    def submit_train_job(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = dict(payload or {})
        body.setdefault("project", self.project_id)
        return self._request_json("/sam3/train/jobs", method="POST", payload=body)

    def train_status(self, job_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/sam3/train/jobs/{quote(job_id)}")
        return payload if isinstance(payload, dict) else {}

    def models(self) -> dict[str, Any]:
        payload = self._get_json("/sam3/models")
        return payload if isinstance(payload, dict) else {}

    def activate_model(self, model_id: str) -> dict[str, Any]:
        return self._request_json(f"/sam3/models/{quote(model_id)}/activate", method="POST", payload={})

    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        clean = {key: value for key, value in (query or {}).items() if value is not None}
        suffix = f"?{urlencode(clean)}" if clean else ""
        return f"{self.server_url}{path}{suffix}"

    def _get_json(self, path: str, query: dict[str, Any] | None = None) -> Any:
        with urlopen(self._url(path, query), timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_npy(self, path: str, query: dict[str, Any] | None = None) -> np.ndarray:
        with urlopen(self._url(path, query), timeout=self.timeout_s) as response:
            data = response.read()
        return np.load(BytesIO(data), allow_pickle=False)

    def _get_npy_with_headers(self, path: str, query: dict[str, Any] | None = None) -> Mask3DResult:
        with urlopen(self._url(path, query), timeout=self.timeout_s) as response:
            data = response.read()
            headers = response.headers
        mask = np.load(BytesIO(data), allow_pickle=False)
        return Mask3DResult(
            mask=mask,
            index_lo=_header_int(headers, "X-Mask3D-Index-Lo"),
            index_hi=_header_int(headers, "X-Mask3D-Index-Hi"),
            voxel_count=_header_int(headers, "X-Mask3D-Voxel-Count"),
            volume_m3=_header_float(headers, "X-Mask3D-Volume-M3"),
            voxel_spacing=_header_spacing(headers, "X-Mask3D-Voxel-Spacing"),
            voxel_spacing_source=_header_text(headers, "X-Mask3D-Voxel-Spacing-Source"),
        )

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        payload_bytes: bytes | None = None,
        query: dict[str, Any] | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        if payload_bytes is not None:
            data = payload_bytes
        elif payload is None:
            data = None
        else:
            data = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path, query),
            data=data,
            headers={"Content-Type": content_type},
            method=method,
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError(f"Unexpected JSON response from {path}")
        return decoded


def _header_text(headers: Any, key: str) -> str | None:
    value = headers.get(key) if hasattr(headers, "get") else None
    text = str(value).strip() if value is not None else ""
    return text or None


def _header_int(headers: Any, key: str) -> int | None:
    text = _header_text(headers, key)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _header_float(headers: Any, key: str) -> float | None:
    text = _header_text(headers, key)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _header_spacing(headers: Any, key: str) -> tuple[float, float, float] | None:
    text = _header_text(headers, key)
    if text is None:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    return values if all(value > 0.0 for value in values) else None
