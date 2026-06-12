#!/usr/bin/env bash
set -euo pipefail

HOST="${YJ_STUDIO_SERVER_HOST:-127.0.0.1}"
PORT="${YJ_STUDIO_SERVER_PORT:-8765}"
PYTHON="${YJ_STUDIO_PYTHON:-python}"

"$PYTHON" - "$HOST" "$PORT" <<'PY'
from __future__ import annotations

import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

host, port = sys.argv[1], sys.argv[2]
try:
    with urlopen(f"http://{host}:{port}/health", timeout=10) as response:
        payload = json.load(response)
except URLError as exc:
    print(f"healthcheck failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
