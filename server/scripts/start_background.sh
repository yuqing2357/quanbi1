#!/usr/bin/env bash
set -euo pipefail

ROOT="${YJ_STUDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "start_background.sh is kept only for compatibility."
echo "The server now runs in the current terminal so status, errors, and logs stay visible."
echo "Press Ctrl+C in this terminal to stop it."
exec bash "$ROOT/server/scripts/start_server.sh"
