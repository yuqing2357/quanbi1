from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.sam3.gpu_pool import GpuWorkerPool  # noqa: E402


def test_pool_spawns_workers_and_binds_gpu_env() -> None:
    """Plumbing only (engine_cfg=None): spawn workers, claim a GPU each, return.

    Validates the hardest mechanical part — spawn + per-worker init binding
    CUDA_VISIBLE_DEVICES — without needing CUDA or the SAM3 library. The real
    CUDA/model path is validated on the GPU server.
    """
    gpu_ids = [10, 11, 12, 13]
    try:
        pool = GpuWorkerPool(gpu_ids, engine_cfg=None)
    except Exception as exc:  # pragma: no cover - platform without spawn support
        pytest.skip(f"cannot start process pool: {exc}")

    try:
        infos = pool.warmup()
    finally:
        pool.shutdown()

    assert len(infos) == len(gpu_ids)
    for info in infos:
        # No engine was built (engine_cfg=None).
        assert info["engine_loaded"] is False
        # Each worker bound itself to one of the configured GPUs.
        assert int(info["cuda_visible_devices"]) in gpu_ids
