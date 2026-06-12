"""Server-side access to shared target models."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_desktop_src_on_path() -> None:
    try:
        import yj_studio.targets  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    desktop_src = repo_root / "apps" / "yj_studio" / "src"
    if desktop_src.exists() and str(desktop_src) not in sys.path:
        sys.path.insert(0, str(desktop_src))


_ensure_desktop_src_on_path()

from yj_studio.targets import (  # noqa: E402
    GeoTarget,
    TargetFrame,
    TargetSet,
    TargetStatus,
    TargetStore,
    export_confirmed_to_coco,
    frame_key,
    normalise_target_type,
)

__all__ = [
    "GeoTarget",
    "TargetFrame",
    "TargetSet",
    "TargetStatus",
    "TargetStore",
    "export_confirmed_to_coco",
    "frame_key",
    "normalise_target_type",
]
