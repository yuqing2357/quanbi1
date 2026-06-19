from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"
    cancelled = "cancelled"


TERMINAL_STATES = {JobState.done, JobState.error, JobState.cancelled}


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    state: JobState = JobState.queued
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    mask_paths: list[str] = field(default_factory=list)

    def status_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "state": self.state.value,
            "progress": float(self.progress),
            "message": self.message,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "state": self.state.value,
            "progress": float(self.progress),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "mask_paths": list(self.mask_paths),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Job":
        return cls(
            id=str(payload["id"]),
            kind=str(payload.get("kind", "")),
            params=dict(payload.get("params") or {}),
            state=JobState(str(payload.get("state", JobState.queued.value))),
            progress=float(payload.get("progress", 0.0)),
            message=str(payload.get("message", "")),
            result=payload.get("result") if isinstance(payload.get("result"), dict) else None,
            error=str(payload["error"]) if payload.get("error") is not None else None,
            mask_paths=[str(path) for path in payload.get("mask_paths", [])],
        )


class JobStore:
    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._persist_dir = Path(persist_dir) if persist_dir is not None else None
        self._load_persisted_jobs()

    def create(self, kind: str, params: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, params=dict(params))
        with self._lock:
            self._jobs[job.id] = job
        logger.info("job queued: id=%s kind=%s", job.id, job.kind)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            job = self._load_job_file(job_id)
            if job is not None:
                self._jobs[job.id] = job
            return job

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            previous_state = job.state
            for key, value in fields.items():
                if key == "state" and not isinstance(value, JobState):
                    value = JobState(str(value))
                setattr(job, key, value)
            self._persist_if_terminal(job)
            _log_job_update(job, fields, previous_state)
            return job

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state in {JobState.done, JobState.error}:
                return job
            job.state = JobState.cancelled
            job.progress = min(float(job.progress), 1.0)
            job.message = "cancelled"
            self._persist_if_terminal(job)
            logger.info("job cancelled: id=%s kind=%s", job.id, job.kind)
            return job

    def _load_persisted_jobs(self) -> None:
        if self._persist_dir is None or not self._persist_dir.exists():
            return
        for path in self._persist_dir.glob("*.json"):
            job = self._load_job_path(path)
            if job is not None:
                self._jobs[job.id] = job

    def _load_job_file(self, job_id: str) -> Job | None:
        if self._persist_dir is None:
            return None
        safe_id = "".join(ch for ch in str(job_id) if ch.isalnum() or ch in "._-")
        if safe_id != job_id:
            return None
        return self._load_job_path(self._persist_dir / f"{safe_id}.json")

    def _load_job_path(self, path: Path) -> Job | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            return Job.from_dict(payload)
        except Exception:
            return None

    def _persist_if_terminal(self, job: Job) -> None:
        if self._persist_dir is None or job.state not in TERMINAL_STATES:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / f"{job.id}.json"
        text = json.dumps(job.to_dict(), ensure_ascii=False, indent=2, default=str)
        if os.name == "nt":
            path.write_text(text, encoding="utf-8")
            return
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)


class JobQueue:
    """Small in-process queue used by the API layer.

    This keeps route handlers from running work directly.  The next step for
    true multi-GPU scheduling is to replace the thread executor with worker
    processes that each own a SAM3 engine and CUDA device binding.
    """

    def __init__(self, worker_count: int = 1) -> None:
        self.worker_count = max(1, int(worker_count))
        self._executor = ThreadPoolExecutor(max_workers=self.worker_count, thread_name_prefix="sam3-worker")
        self._futures: set[Future] = set()
        self._lock = threading.Lock()

    def submit(self, func, *args, **kwargs) -> Future:
        future = self._executor.submit(func, *args, **kwargs)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._discard)
        return future

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _discard(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)


def _log_job_update(job: Job, fields: dict[str, Any], previous_state: JobState) -> None:
    if "state" not in fields and job.state not in TERMINAL_STATES:
        return
    if job.state == previous_state and job.state not in TERMINAL_STATES:
        return
    if job.state == JobState.error:
        logger.error(
            "job state: id=%s kind=%s state=%s progress=%.2f message=%s error=%s",
            job.id,
            job.kind,
            job.state.value,
            float(job.progress),
            job.message,
            job.error,
        )
    else:
        logger.info(
            "job state: id=%s kind=%s state=%s progress=%.2f message=%s",
            job.id,
            job.kind,
            job.state.value,
            float(job.progress),
            job.message,
        )
