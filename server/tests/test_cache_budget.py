from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SRC = _REPO_ROOT / "server" / "src"
if str(_SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVER_SRC))

from yj_studio_server.cache import plan_slice_cache_removals  # noqa: E402


def test_plan_slice_cache_removals_selects_oldest_files() -> None:
    old = Path("old.npy")
    middle = Path("middle.npy")
    new = Path("new.npy")

    planned, remaining = plan_slice_cache_removals(
        [
            (20.0, middle, 10),
            (30.0, new, 10),
            (10.0, old, 10),
        ],
        budget_bytes=15,
    )

    assert [path for _mtime, path, _size in planned] == [old, middle]
    assert remaining == 10
