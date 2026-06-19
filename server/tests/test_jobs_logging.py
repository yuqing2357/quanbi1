from __future__ import annotations

import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.sam3.jobs import JobState, JobStore  # noqa: E402


def test_job_store_logs_state_changes(caplog) -> None:
    store = JobStore()

    with caplog.at_level(logging.INFO, logger="yj_studio_server.sam3.jobs"):
        job = store.create("segment", {"volume_id": "tiny"})
        store.update(job.id, state=JobState.running, progress=0.25, message="running SAM3")
        store.update(job.id, state=JobState.done, progress=1.0, message="done")

    messages = [record.getMessage() for record in caplog.records]
    assert any("job queued" in message and "kind=segment" in message for message in messages)
    assert any("state=running" in message and "running SAM3" in message for message in messages)
    assert any("state=done" in message and "message=done" in message for message in messages)


def test_job_store_logs_errors(caplog) -> None:
    store = JobStore()

    with caplog.at_level(logging.INFO, logger="yj_studio_server.sam3.jobs"):
        job = store.create("track", {})
        store.update(
            job.id,
            state=JobState.error,
            progress=1.0,
            message="failed",
            error="RuntimeError: video predictor not loaded",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("state=error" in message for message in messages)
    assert any("video predictor not loaded" in message for message in messages)
