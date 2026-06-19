from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_background_entrypoint_stays_in_foreground() -> None:
    script = (ROOT / "server" / "scripts" / "start_background.sh").read_text(encoding="utf-8")

    assert "nohup" not in script
    assert "server.out.log" not in script
    assert "server.err.log" not in script
    assert 'exec bash "$ROOT/server/scripts/start_server.sh"' in script
