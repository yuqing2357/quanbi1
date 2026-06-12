from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "models.json"

    def load(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return {"active_model": None, "models": [], "updated_at": utc_now_iso()}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = utc_now_iso()
        text = json.dumps(payload, indent=2)
        if os.name == "nt":
            self.path.write_text(text, encoding="utf-8")
            return
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def add_model(
        self,
        *,
        checkpoint: str | None,
        dataset_version: str,
        metrics: dict[str, Any] | None = None,
        status: str = "ready",
        parent_model_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        model_id = f"M{uuid.uuid4().hex[:8]}"
        parent = parent_model_id if parent_model_id is not None else payload.get("active_model")
        row = {
            "id": model_id,
            "checkpoint": checkpoint,
            "dataset_version": dataset_version,
            "metrics": metrics or {},
            "parent_model_id": parent,
            "status": status,
            "created_at": utc_now_iso(),
        }
        payload.setdefault("models", []).append(row)
        if payload.get("active_model") is None and status == "ready":
            payload["active_model"] = model_id
        self.save(payload)
        return row

    def activate(self, model_id: str) -> dict[str, Any]:
        payload = self.load()
        if not any(model.get("id") == model_id for model in payload.get("models", [])):
            raise KeyError(model_id)
        payload["active_model"] = model_id
        self.save(payload)
        return payload
