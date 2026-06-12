from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = ROOT / "server" / "src"
DEFAULT_CONFIG = ROOT / "server" / "config" / "server.yaml"
FALLBACK_CONFIG = ROOT / "server" / "config" / "server.example.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the YJ Studio remote API server.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else FALLBACK_CONFIG),
        help="Server config file.",
    )
    parser.add_argument("--host", help="Override host from config.")
    parser.add_argument("--port", type=int, help="Override port from config.")
    parser.add_argument("--log-level", help="Override uvicorn log level from config.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load config and print the resolved server address without starting uvicorn.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    os.environ["YJ_STUDIO_ROOT"] = str(ROOT)
    os.environ["YJ_STUDIO_SERVER_CONFIG"] = str(config_path)
    sys.path.insert(0, str(SERVER_SRC))

    from yj_studio_server.config import load_config

    cfg = load_config(config_path)
    host = args.host or cfg.host
    port = args.port or cfg.port
    log_level = args.log_level or cfg.log_level

    if args.check_only:
        print("[OK] server config loaded")
        print(f"config: {config_path}")
        print(f"listen: http://{host}:{port}")
        print(f"data_root: {cfg.data_root} exists={cfg.data_root.exists()}")
        print(f"volumes: {len(cfg.volumes)}")
        return 0

    import uvicorn

    uvicorn.run(
        "yj_studio_server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
