"""State-machine checks for AIService that don't touch SAM3."""

from __future__ import annotations

from pathlib import Path

from yj_studio.ai import AIService, AIServiceState, SAM3Config


def test_initial_state_is_idle(qapp) -> None:
    service = AIService(SAM3Config(checkpoint_path=Path("/does/not/exist")))
    assert service.state == AIServiceState.IDLE
    assert not service.is_ready()
    assert service.image_processor is None


def test_mark_busy_only_in_ready(qapp) -> None:
    service = AIService(SAM3Config(checkpoint_path=Path("/does/not/exist")))
    states: list[AIServiceState] = []
    service.state_changed.connect(lambda state, _m: states.append(state))
    service.mark_busy("test")
    # Not READY yet, so mark_busy is a no-op.
    assert states == []


def test_emit_prompt_signals(qapp) -> None:
    service = AIService(SAM3Config(checkpoint_path=Path("/does/not/exist")))
    boxes: list[tuple] = []
    points: list[tuple] = []
    service.box_prompt_added.connect(
        lambda axis, idx, x0, y0, x1, y1: boxes.append((axis, idx, x0, y0, x1, y1))
    )
    service.point_prompt_added.connect(
        lambda axis, idx, x, y: points.append((axis, idx, x, y))
    )
    service.emit_box_prompt("inline", 320, 10, 20, 30, 40)
    service.emit_point_prompt("xline", 200, 5.5, 6.5)
    assert boxes == [("inline", 320, 10.0, 20.0, 30.0, 40.0)]
    assert points == [("xline", 200, 5.5, 6.5)]
