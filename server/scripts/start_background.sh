#!/usr/bin/env bash
set -euo pipefail

ROOT="${YJ_STUDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PID_FILE="$ROOT/runtime/server/server.pid"
LOG_DIR="$ROOT/runtime/server/logs"

mkdir -p "$LOG_DIR" "$(dirname "$PID_FILE")"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "YJ Studio server already running: pid=$old_pid"
    exit 0
  fi
fi

nohup bash "$ROOT/server/scripts/start_server.sh" \
  > "$LOG_DIR/server.out.log" \
  2> "$LOG_DIR/server.err.log" &
pid="$!"
echo "$pid" > "$PID_FILE"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "YJ Studio server failed to start; see $LOG_DIR/server.err.log" >&2
  exit 1
fi

echo "YJ Studio server started: pid=$pid"
echo "Logs:"
echo "  $LOG_DIR/server.out.log"
echo "  $LOG_DIR/server.err.log"
