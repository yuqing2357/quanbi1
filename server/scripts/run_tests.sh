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

conda activate "$ENV_NAME"
cd "$ROOT"

python -m pip check
python tools/check_sam3_setup.py
QT_QPA_PLATFORM=offscreen python -m pytest \
  server/tests/test_sam3_jobs_api.py \
  apps/yj_studio/tests/test_layer_store.py \
  apps/yj_studio/tests/test_algorithms_serialization.py \
  apps/yj_studio/tests/test_volume_slice_renderer.py
