#!/usr/bin/env bash
set -euo pipefail

ROOT="${YJ_STUDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${YJ_STUDIO_CONDA_ENV:-yjstudio-server}"
CONDA_SH="${YJ_STUDIO_CONDA_SH:-/root/anaconda3/etc/profile.d/conda.sh}"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
else
  eval "$(conda shell.bash hook)"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.12 pip setuptools wheel -y
fi

conda activate "$ENV_NAME"
python -m pip install -r "$ROOT/config/env/requirements-server.txt"
python -m pip install -e "$ROOT/shared"
python -m pip install -e "$ROOT/local/app"

echo "Server environment ready: $ENV_NAME"
