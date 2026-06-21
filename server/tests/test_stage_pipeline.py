"""HTTP-level tests for the four-stage target pipeline.

Covers the stage query plumbing plus the promote/clear/renumber endpoints and
the temporary-stage admission gate. Skipped where fastapi is unavailable (the
desktop-only dev env); runs in the server env where fastapi is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for sub in ("server/src", "shared/src"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

from yj_studio_server.app import _eligible_track_objects, _target_store, create_app  # noqa: E402
from yj_studio_server.config import ServerConfig  # noqa: E402
from yj_studio_server.targets import GeoTarget, TargetStage  # noqa: E402


def _mask() -> np.ndarray:
    arr = np.zeros((4, 4), dtype=bool)
    arr[1:3, 1:3] = True
    return arr


def _seed_temp(cfg: ServerConfig, count: int = 2) -> None:
    store = _target_store(cfg, stage=TargetStage.TEMPORARY)
    with store.mutate() as ts:
        for i in range(count):
            tid = ts.new_id()
            target = GeoTarget(id=tid, type="trap")
            for index in (10 + i, 11 + i):
                frame = store.frame_from_mask(target_id=tid, axis="inline", index=index, mask=_mask())
                target.add_frame(frame)
            ts.add_target(target)


def _make_cfg(tmp_path: Path) -> ServerConfig:
    return ServerConfig(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        results_root=tmp_path / "results",
        project_id="default",
        volumes={},
        sam3={"checkpoint": "weights/sam3.pt", "results_subdir": "sam3"},
    )


def test_stage_list_promote_clear_renumber(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)
    _seed_temp(cfg, count=2)

    with TestClient(app) as client:
        # Temp stage lists the seeded TMP ids; saved/training empty.
        temp = client.get("/sam3/targets", params={"stage": "temp"}).json()
        assert temp["stage"] == "temporary"
        assert sorted(t["id"] for t in temp["summaries"]) == ["TMP1", "TMP2"]
        assert client.get("/sam3/targets", params={"stage": "saved"}).json()["summaries"] == []

        # Promote TMP1 -> saved (move). Temp loses it, saved gains SAV1.
        promoted = client.post(
            "/sam3/targets/promote",
            json={"target_ids": ["TMP1"], "from_stage": "temporary"},
        )
        assert promoted.status_code == 200
        assert promoted.json()["to_stage"] == "saved"
        saved = client.get("/sam3/targets", params={"stage": "saved"}).json()
        assert [t["id"] for t in saved["summaries"]] == ["SAV1"]
        temp_after = client.get("/sam3/targets", params={"stage": "temp"}).json()
        assert [t["id"] for t in temp_after["summaries"]] == ["TMP2"]

        # Promote SAV1 -> training requires a category (copy semantics).
        no_cat = client.post(
            "/sam3/targets/promote",
            json={"target_ids": ["SAV1"], "from_stage": "saved"},
        )
        assert no_cat.status_code == 400
        with_cat = client.post(
            "/sam3/targets/promote",
            json={"target_ids": ["SAV1"], "from_stage": "saved", "category": "turbidite"},
        )
        assert with_cat.status_code == 200
        training = client.get("/sam3/targets", params={"stage": "training"}).json()
        assert [t["id"] for t in training["summaries"]] == ["TRN1"]
        # Saved keeps the source (long-term pool).
        assert [t["id"] for t in client.get("/sam3/targets", params={"stage": "saved"}).json()["summaries"]] == ["SAV1"]

        # Clear temp empties it.
        assert client.post("/sam3/targets/clear", params={"stage": "temp"}).status_code == 200
        assert client.get("/sam3/targets", params={"stage": "temp"}).json()["summaries"] == []

        # Renumber saved (no-op single target) stays gapless.
        renum = client.post("/sam3/targets/renumber", params={"stage": "saved"}).json()
        assert [t["id"] for t in renum["summaries"]] == ["SAV1"]


def test_delete_single_frame_keeps_target(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)
    _seed_temp(cfg, count=1)  # TMP1 with frames inline:10, inline:11

    with TestClient(app) as client:
        before = client.get("/sam3/targets/TMP1", params={"stage": "temp"}).json()
        assert sorted(before["frames"].keys()) == ["inline:10", "inline:11"]

        deleted = client.delete("/sam3/targets/TMP1/mask/inline/10", params={"stage": "temp"})
        assert deleted.status_code == 200
        after = deleted.json()
        # Frame dropped, target itself survives.
        assert list(after["frames"].keys()) == ["inline:11"]
        assert "inline:10" not in after["frames"]

        # Deleting a non-existent frame is a 404.
        missing = client.delete("/sam3/targets/TMP1/mask/inline/99", params={"stage": "temp"})
        assert missing.status_code == 404


def test_eligible_track_objects_gate() -> None:
    masks = {i: _mask() for i in range(3)}
    collected = {
        1: {0: _mask()},                       # single frame -> dropped
        2: dict(masks),                        # 3 contiguous -> kept
        3: {0: _mask(), 9: _mask()},           # gap of 8 > limit -> dropped
        4: {0: _mask(), 3: _mask()},           # gap of 2 <= limit -> kept
    }
    eligible, dropped = _eligible_track_objects(collected, gap_limit=5, min_frames=2)
    assert sorted(eligible) == [2, 4]
    assert sorted(dropped) == [1, 3]
