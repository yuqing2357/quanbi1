"""Template-guided 2D search on a reservoir model: match foreground, not background.

Verifies the ``template_match`` job extracts the reservoir foreground, finds the
local structure whose silhouette matches the drawn template (NOT the empty black
background), serves candidate masks, and writes NOTHING to any target store.
Skipped where fastapi is unavailable (desktop-only dev env).
"""

from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for sub in ("server/src", "shared/src"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

from yj_studio_core.shapes import rasterize_polygon  # noqa: E402
from yj_studio_server.app import create_app  # noqa: E402
from yj_studio_server.config import ServerConfig  # noqa: E402

# Left-high / right-low wedge, full extent of whatever box it is drawn into.
_WEDGE = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def _wait_done(client: TestClient, job_id: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = client.get(f"/sam3/jobs/{job_id}").json()
        if last.get("state") in {"done", "error", "cancelled"}:
            return last
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for job {job_id}: {last}")


def _make_cfg(tmp_path: Path) -> ServerConfig:
    """A reservoir-style binary volume: black background with one wedge body.

    ``inline`` index 0 of a (1, H, W) volume gives an (H, W) slice that, after the
    job's ``.T``, is (W, H). We build the slice so the body lands in a known
    sub-region and everything else is background (0).
    """
    data_root = tmp_path / "data"
    volume_dir = data_root / "reservoir"
    volume_dir.mkdir(parents=True)
    # Volume axis0=1 (single inline), axis1=W=120, sample=H=100.
    w, h = 120, 100
    body = rasterize_polygon(_WEDGE, 40, 40, normalized=True)  # 40x40 wedge
    slice_hw = np.zeros((h, w), dtype=np.float32)  # rows=sample(H), cols=axis1(W)
    # Place the body in the lower-right of the (H, W) display slice.
    slice_hw[60:100, 80:120] = body.astype(np.float32)
    # Stored volume is indexed [axis0, axis1, sample]; the job reads slice as
    # data=(axis1, sample) then transposes to (sample, axis1)=(H, W)=slice_hw.
    volume = slice_hw.T.reshape(1, w, h)
    np.save(volume_dir / "litho.npy", volume, allow_pickle=False)
    return ServerConfig(
        project_root=tmp_path,
        data_root=data_root,
        runtime_root=tmp_path / "runtime" / "server",
        results_root=data_root / "results",
        project_id="default",
        volumes={"model_lithology": {"label": "岩性模型", "path": "reservoir/litho.npy", "clim": None}},
        sam3={"checkpoint": "weights/sam3.pt", "results_subdir": "sam3"},
    )


def test_template_match_finds_body_ignores_background(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)

    with TestClient(app) as client:
        submitted = client.post(
            "/sam3/jobs",
            json={
                "kind": "template_match",
                "volume_id": "model_lithology",
                "axis": "inline",
                "index": 0,
                "template": _WEDGE,
                "keep_top_k": 5,
                "min_score": 0.3,
                "target_type": "trap",
            },
        )
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]

        status = _wait_done(client, job_id)
        assert status["state"] == "done", status

        result = client.get(f"/sam3/jobs/{job_id}/result").json()
        assert result["kind"] == "template_match"
        assert "targets" not in result  # nothing persisted
        candidates = result["candidates"]
        assert candidates, "expected a match on the reservoir body"

        top = candidates[0]
        assert top["target_id"] == ""  # 2D-only, no target id
        # The match must sit over the body's lower-right region, never the empty
        # background in the top-left.
        x0, y0, x1, y1 = top["box"]
        assert x1 >= 70 and y1 >= 50

        mask_response = client.get(top["mask_url"])
        assert mask_response.status_code == 200
        mask = np.load(BytesIO(mask_response.content), allow_pickle=False)
        assert mask.dtype == np.uint8 and mask.any()
        # No foreground pixels in the top-left background quadrant.
        assert mask[:50, :60].sum() == 0

        # No target store touched in ANY stage.
        for stage in ("temp", "saved", "training"):
            listed = client.get("/sam3/targets", params={"stage": stage}).json()
            assert listed["summaries"] == [], stage


def test_template_match_rejects_degenerate_template(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)

    with TestClient(app) as client:
        bad = client.post(
            "/sam3/jobs",
            json={
                "kind": "template_match",
                "volume_id": "model_lithology",
                "axis": "inline",
                "index": 0,
                "template": [[0.1, 0.1], [0.2, 0.2]],
            },
        )
        assert bad.status_code == 400
