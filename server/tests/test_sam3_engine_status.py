from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.sam3.engine import SAM3Engine  # noqa: E402


def test_track_video_reports_disabled_config() -> None:
    engine = SAM3Engine(Path("__missing_sam3.pt__"), load_video=False)
    engine._processor = object()

    with pytest.raises(RuntimeError, match="sam3.load_video=false"):
        list(
            engine.track_video(
                Path("."),
                seeds=[],
                seed_local=0,
                fwd_budget=1,
                back_budget=0,
            )
        )


def test_track_video_reports_video_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SAM3Engine(Path("__missing_sam3.pt__"), load_video=True)
    engine._processor = object()
    sam3_pkg = types.ModuleType("sam3")
    sam3_pkg.__path__ = []
    model_builder = types.ModuleType("sam3.model_builder")
    model_builder.build_sam3_video_model = object()
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)

    def fail_load(_builder) -> None:
        raise ImportError("No module named 'triton'")

    monkeypatch.setattr(engine, "_load_video_predictor", fail_load)

    with pytest.raises(RuntimeError, match="triton"):
        list(
            engine.track_video(
                Path("."),
                seeds=[],
                seed_local=0,
                fwd_budget=1,
                back_budget=0,
            )
        )

    assert engine.status_payload()["video_enabled"] is True
    assert engine.status_payload()["video_loaded"] is False
    assert "triton" in str(engine.status_payload()["video_error"])
