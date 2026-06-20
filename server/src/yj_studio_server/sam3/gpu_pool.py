"""Multi-GPU SAM3 worker pool (true per-card parallelism).

Each worker is a *spawned* process pinned to one GPU via ``CUDA_VISIBLE_DEVICES``
that builds and warms its own :class:`SAM3Engine`.  Spawn (not fork) is required
because a CUDA context cannot survive ``fork``.

Division of labour (docs/project_review_and_remediation.md §4): **workers only
run inference and return picklable masks; the main process persists targets
under its per-project lock.**  This sidesteps the fact that the in-process
target lock does not span processes.

The pool is engine-agnostic at the plumbing level: ``engine_cfg=None`` skips
engine construction so the spawn / per-worker GPU-claim machinery can be unit
tested without CUDA or the SAM3 library.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-worker-process globals, set by _worker_init in each spawned process.
_ENGINE: Any | None = None
_GPU: int | None = None
_RELOAD_SIGNAL_PATH: str | None = None
_LOADED_SIG_VERSION: Any | None = None


def _read_reload_signal() -> dict[str, Any] | None:
    if not _RELOAD_SIGNAL_PATH:
        return None
    try:
        path = Path(_RELOAD_SIGNAL_PATH)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - signal is best-effort
        return None


def _maybe_reload() -> None:
    """Hot-swap this worker's checkpoint if a newer activation was signalled."""
    global _LOADED_SIG_VERSION
    if _ENGINE is None:
        return
    signal = _read_reload_signal()
    if not signal:
        return
    version = signal.get("version")
    checkpoint = signal.get("checkpoint")
    if version is None or checkpoint is None or version == _LOADED_SIG_VERSION:
        return
    try:
        _ENGINE.reload_checkpoint(checkpoint)
        _ENGINE.warmup()
        _LOADED_SIG_VERSION = version
        logger.info("gpu worker reloaded checkpoint (CUDA_VISIBLE_DEVICES=%s): %s", _GPU, checkpoint)
    except Exception:  # noqa: BLE001 - keep serving the old model on failure
        logger.exception("gpu worker checkpoint reload failed: %s", checkpoint)


def _worker_init(counter, lock, gpu_ids, engine_cfg) -> None:
    """Claim a unique GPU (atomic counter) and build this worker's engine."""
    global _ENGINE, _GPU, _RELOAD_SIGNAL_PATH
    with lock:
        idx = counter.value
        counter.value = idx + 1
    gpu = gpu_ids[idx % len(gpu_ids)] if gpu_ids else 0
    _GPU = int(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if engine_cfg is None:
        # Plumbing-only mode (tests): no CUDA, no SAM3 library.
        return
    _RELOAD_SIGNAL_PATH = engine_cfg.get("reload_signal_path")
    try:
        from .engine import SAM3Engine

        engine = SAM3Engine(
            engine_cfg["checkpoint"],
            device=str(engine_cfg.get("device", "cuda")),
            resolution=int(engine_cfg.get("resolution", 1008)),
            source_root=engine_cfg.get("source_root"),
            load_video=bool(engine_cfg.get("load_video", True)),
            video_temporal_disambiguation=bool(
                engine_cfg.get("video_temporal_disambiguation", False)
            ),
        )
        engine.warmup()
        _ENGINE = engine
        logger.info("gpu worker ready: CUDA_VISIBLE_DEVICES=%s", gpu)
        # A checkpoint may have been activated before this worker spawned.
        _maybe_reload()
    except Exception:  # noqa: BLE001 - surface in worker logs; tasks will error clearly
        logger.exception("gpu worker init/warmup failed (CUDA_VISIBLE_DEVICES=%s)", gpu)


def _worker_segment(rgb, kwargs) -> list[dict[str, Any]]:
    if _ENGINE is None:
        raise RuntimeError("GPU worker has no SAM3 engine loaded")
    _maybe_reload()
    return _ENGINE.segment(rgb, **kwargs)


def _worker_track(frames_dir, seeds, seed_local, fwd_budget, back_budget, indices, signal_dir=None):
    if _ENGINE is None:
        raise RuntimeError("GPU worker has no SAM3 engine loaded")
    _maybe_reload()
    from .tracking import collect_object_frames

    cancelled = None
    progress = None
    if signal_dir:
        sig = Path(signal_dir)
        cancel_file = sig / "cancel"
        progress_file = sig / "progress"

        def cancelled() -> bool:  # noqa: F811 - intentional local rebind
            return cancel_file.exists()

        def progress(done: int) -> None:  # noqa: F811
            try:
                progress_file.write_text(str(int(done)), encoding="utf-8")
            except Exception:  # noqa: BLE001 - progress is best-effort
                pass

    return collect_object_frames(
        _ENGINE,
        frames_dir,
        seeds=seeds,
        seed_local=seed_local,
        fwd_budget=fwd_budget,
        back_budget=back_budget,
        indices=indices,
        cancelled=cancelled,
        progress=progress,
    )


def _worker_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engine_loaded": bool(_ENGINE is not None and getattr(_ENGINE, "is_loaded", False)),
        "pid": os.getpid(),
    }
    try:
        import torch  # type: ignore

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["name"] = torch.cuda.get_device_name(0)
            info["mem_free_bytes"] = int(free)
            info["mem_total_bytes"] = int(total)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


class GpuWorkerPool:
    """A spawn-based ProcessPoolExecutor with one SAM3 engine per GPU."""

    def __init__(
        self,
        gpu_ids: list[int],
        engine_cfg: dict[str, Any] | None,
        *,
        mp_context: str = "spawn",
    ) -> None:
        self.gpu_ids = [int(g) for g in (gpu_ids or [0])]
        self._reload_signal_path = (engine_cfg or {}).get("reload_signal_path")
        self._ctx = get_context(mp_context)
        counter = self._ctx.Value("i", 0)
        lock = self._ctx.Lock()
        self._executor = ProcessPoolExecutor(
            max_workers=len(self.gpu_ids),
            mp_context=self._ctx,
            initializer=_worker_init,
            initargs=(counter, lock, self.gpu_ids, engine_cfg),
        )
        self._worker_infos: list[dict[str, Any]] = []

    def warmup(self) -> list[dict[str, Any]]:
        """Force every worker to spawn + run its initializer (eager model load).

        Submitting one task per GPU makes the executor spawn all workers; the
        long model load inside each initializer keeps them busy so the tasks map
        one-to-one onto distinct workers.
        """
        futures = [self._executor.submit(_worker_gpu_info) for _ in self.gpu_ids]
        self._worker_infos = [f.result() for f in futures]
        return self._worker_infos

    def segment(self, rgb, **kwargs) -> list[dict[str, Any]]:
        return self._executor.submit(_worker_segment, rgb, kwargs).result()

    def submit_segment(self, rgb, **kwargs) -> Future:
        return self._executor.submit(_worker_segment, rgb, kwargs)

    def track(
        self,
        frames_dir,
        *,
        seeds,
        seed_local,
        fwd_budget,
        back_budget,
        indices,
    ):
        return self.submit_track(
            frames_dir,
            seeds=seeds,
            seed_local=seed_local,
            fwd_budget=fwd_budget,
            back_budget=back_budget,
            indices=indices,
        ).result()

    def submit_track(
        self,
        frames_dir,
        *,
        seeds,
        seed_local,
        fwd_budget,
        back_budget,
        indices,
        signal_dir=None,
    ) -> Future:
        """Non-blocking track on one worker. ``signal_dir`` (if given) lets the
        caller poll ``signal_dir/progress`` and request cancel by creating
        ``signal_dir/cancel``."""
        return self._executor.submit(
            _worker_track,
            str(frames_dir),
            seeds,
            int(seed_local),
            int(fwd_budget),
            int(back_budget),
            list(indices),
            str(signal_dir) if signal_dir is not None else None,
        )

    def reload_checkpoint(self, checkpoint_path: str) -> None:
        """Signal every worker to hot-swap to ``checkpoint_path`` before its
        next inference task (no restart needed)."""
        if not self._reload_signal_path:
            logger.warning("reload_checkpoint ignored: no reload_signal_path configured")
            return
        import time

        path = Path(self._reload_signal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": time.time_ns(), "checkpoint": str(checkpoint_path)}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        logger.info("signalled GPU workers to reload checkpoint: %s", checkpoint_path)

    def gpu_info(self) -> list[dict[str, Any]]:
        if self._worker_infos:
            return self._worker_infos
        return self.warmup()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
