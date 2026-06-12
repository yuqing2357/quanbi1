from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
# Config now lives in the unified top-level config/ folder. The local/config/
# paths are kept as legacy fallbacks so older checkouts keep working.
DEFAULT_CONFIG = ROOT / "config" / "local.yaml"
FALLBACK_CONFIG = ROOT / "config" / "local.example.yaml"
if not DEFAULT_CONFIG.exists() and not FALLBACK_CONFIG.exists():
    DEFAULT_CONFIG = ROOT / "local" / "config" / "local.yaml"
    FALLBACK_CONFIG = ROOT / "local" / "config" / "local.example.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local YJ Studio viewer.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else FALLBACK_CONFIG),
        help="Local viewer config file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load config and probe the remote server without opening the GUI.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Open the GUI without probing the configured remote server first.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    apply_runtime_env(config_path, cfg)

    server_url = str(cfg.get("server_url", "")).rstrip("/")
    if not args.skip_probe and _as_bool(cfg.get("probe_server_on_start", True)) and server_url:
        ok = probe_server(server_url, timeout_s=float(cfg.get("request_timeout_s", 30)))
        if not ok and _as_bool(cfg.get("require_server", False)):
            return 2

    if args.check_only:
        print("[OK] local viewer config loaded")
        print(f"config: {config_path}")
        print(f"server_url: {server_url or '(not set)'}")
        return 0

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import run_yj_studio

    return run_yj_studio.main()


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Local config not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        return _load_simple_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Local config must be a mapping: {path}")
    return data


def apply_runtime_env(config_path: Path, cfg: dict[str, object]) -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ["YJ_STUDIO_LOCAL_CONFIG"] = str(config_path)
    if "mode" in cfg:
        os.environ["YJ_STUDIO_MODE"] = str(cfg["mode"])
    if "server_url" in cfg:
        os.environ["YJ_STUDIO_SERVER_URL"] = str(cfg["server_url"]).rstrip("/")
    if "project_id" in cfg:
        os.environ["YJ_STUDIO_PROJECT_ID"] = str(cfg["project_id"])
    if "volume_backend" in cfg:
        os.environ["YJ_STUDIO_VOLUME_BACKEND"] = str(cfg["volume_backend"])
    if "target_backend" in cfg:
        os.environ["YJ_STUDIO_TARGET_BACKEND"] = str(cfg["target_backend"])
    elif str(cfg.get("mode", "")).strip().lower() == "remote":
        os.environ["YJ_STUDIO_TARGET_BACKEND"] = "remote"
    if "sam3_backend" in cfg:
        os.environ["YJ_STUDIO_SAM3_BACKEND"] = str(cfg["sam3_backend"])
    elif str(cfg.get("mode", "")).strip().lower() == "remote":
        os.environ["YJ_STUDIO_SAM3_BACKEND"] = "remote"
    if "request_timeout_s" in cfg:
        os.environ["YJ_STUDIO_REQUEST_TIMEOUT_S"] = str(cfg["request_timeout_s"])
    if _as_bool(cfg.get("disable_3d", False)):
        os.environ["YJ_STUDIO_DISABLE_3D"] = "1"


def probe_server(server_url: str, *, timeout_s: float) -> bool:
    print(f"[YJ Studio] probing remote server: {server_url}")
    try:
        health = _get_json(f"{server_url}/health", timeout_s=timeout_s)
        volumes = _get_json(f"{server_url}/volumes", timeout_s=timeout_s)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] remote server probe failed: {exc}")
        return False

    print(f"[OK] server health: {health.get('status', 'unknown')}")
    if isinstance(volumes, list):
        print(f"[OK] remote volumes: {len(volumes)}")
        for volume in volumes:
            label = volume.get("label", volume.get("id", "?"))
            shape = volume.get("shape", "?")
            dtype = volume.get("dtype", "?")
            exists = volume.get("exists", False)
            print(f"  - {label}: shape={shape}, dtype={dtype}, exists={exists}")
    return True


def _get_json(url: str, *, timeout_s: float) -> dict[str, object] | list[dict[str, object]]:
    with urlopen(url, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Unexpected JSON payload from {url}")
    return data


def _load_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    list_key: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- ") and list_key:
                current = data.setdefault(list_key, [])
                if not isinstance(current, list):
                    raise ValueError(f"Cannot append list item to non-list key: {list_key}")
                current.append(_parse_scalar(stripped[2:].strip()))
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = _parse_scalar(value)
                list_key = None
            else:
                data[key] = []
                list_key = key
    return data


def _parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
