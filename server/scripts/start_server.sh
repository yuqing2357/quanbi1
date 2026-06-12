#!/usr/bin/env bash
set -euo pipefail

ROOT="${YJ_STUDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${YJ_STUDIO_CONDA_ENV:-yjstudio-server}"
CONDA_SH="${YJ_STUDIO_CONDA_SH:-/root/anaconda3/etc/profile.d/conda.sh}"
CONFIG="${YJ_STUDIO_SERVER_CONFIG:-}"
if [[ -z "$CONFIG" ]]; then
  # Unified config/ preferred; server/config/ kept as legacy fallback.
  for c in "$ROOT/config/server.yaml" "$ROOT/config/server.example.yaml" \
           "$ROOT/server/config/server.yaml" "$ROOT/server/config/server.example.yaml"; do
    if [[ -f "$c" ]]; then CONFIG="$c"; break; fi
  done
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
else
  eval "$(conda shell.bash hook)"
fi

conda activate "$ENV_NAME"

mkdir -p "$ROOT/runtime/server/logs" "$ROOT/runtime/server/cache" "$ROOT/runtime/server/jobs"

export YJ_STUDIO_ROOT="$ROOT"
export YJ_STUDIO_SERVER_CONFIG="$CONFIG"
export PYTHONPATH="$ROOT/server/src:$ROOT/shared/src:$ROOT/local/app/src:$ROOT/libs:${PYTHONPATH:-}"

HOST="${YJ_STUDIO_SERVER_HOST:-0.0.0.0}"
PORT="${YJ_STUDIO_SERVER_PORT:-8765}"
LOG_LEVEL="${YJ_STUDIO_SERVER_LOG_LEVEL:-info}"

exec python -m uvicorn yj_studio_server.app:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
