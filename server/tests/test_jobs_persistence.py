from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SRC = _REPO_ROOT / "server" / "src"
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))

from yj_studio_server.sam3.jobs import JobState, JobStore  # noqa: E402


def test_terminal_job_is_persisted_and_reloaded(tmp_path: Path) -> None:
    store = JobStore(persist_dir=tmp_path / "jobs")
    job = store.create("segment", {"volume_id": "tiny"})

    store.update(
        job.id,
        state=JobState.done,
        progress=1.0,
        message="done",
        result={"target_ids": ["T1"]},
        mask_paths=["/tmp/T1.npy"],
    )

    reloaded = JobStore(persist_dir=tmp_path / "jobs").get(job.id)

    assert reloaded is not None
    assert reloaded.id == job.id
    assert reloaded.state == JobState.done
    assert reloaded.result == {"target_ids": ["T1"]}
    assert reloaded.mask_paths == ["/tmp/T1.npy"]


def test_cancelled_job_is_persisted(tmp_path: Path) -> None:
    store = JobStore(persist_dir=tmp_path / "jobs")
    job = store.create("track", {"axis": "inline"})

    cancelled = store.cancel(job.id)
    reloaded = JobStore(persist_dir=tmp_path / "jobs").get(job.id)

    assert cancelled is not None
    assert reloaded is not None
    assert reloaded.state == JobState.cancelled
    assert reloaded.message == "cancelled"
