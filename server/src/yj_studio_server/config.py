from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    log_level: str = "info"
    project_id: str = "default"
    project_root: Path = Path("/root/quanbi")
    data_root: Path = Path("/root/quanbi/data")
    runtime_root: Path = Path("/root/quanbi/runtime/server")
    results_root: Path = Path("/root/quanbi/data/results")
    slice_cache_max_gb: float = 100.0
    volumes: dict[str, dict[str, Any]] = field(default_factory=dict)
    sam3: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    preload: dict[str, Any] = field(default_factory=dict)


def default_config_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(os.environ.get("YJ_STUDIO_ROOT", "/root/quanbi"))
    explicit = os.environ.get("YJ_STUDIO_SERVER_CONFIG")
    if explicit:
        return Path(explicit)
    # Unified config/ is preferred; server/config/ kept as legacy fallback so a
    # not-yet-redeployed server keeps booting from its old location.
    candidates = [
        root / "config" / "server.yaml",
        root / "config" / "server.example.yaml",
        root / "server" / "config" / "server.yaml",
        root / "server" / "config" / "server.example.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_config(path: Path | None = None) -> ServerConfig:
    config_path = path or default_config_path()
    payload = _load_yaml(config_path)
    project_root = Path(payload.get("project_root", "/root/quanbi")).expanduser()
    data_root = Path(payload.get("data_root", project_root / "data")).expanduser()
    runtime_root = Path(payload.get("runtime_root", project_root / "runtime" / "server")).expanduser()
    results_root = Path(payload.get("results_root", project_root / "data" / "results")).expanduser()
    return ServerConfig(
        host=str(payload.get("host", "0.0.0.0")),
        port=int(payload.get("port", 8765)),
        log_level=str(payload.get("log_level", "info")),
        project_id=str(payload.get("project_id", "default")),
        project_root=project_root,
        data_root=data_root,
        runtime_root=runtime_root,
        results_root=results_root,
        slice_cache_max_gb=float(payload.get("slice_cache_max_gb", 100.0)),
        volumes=dict(payload.get("volumes", {})),
        sam3=dict(payload.get("sam3", {})),
        training=dict(payload.get("training", {})),
        auth=dict(payload.get("auth", {})),
        preload=dict(payload.get("preload", {})),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Server config must be a mapping: {path}")
    return data
