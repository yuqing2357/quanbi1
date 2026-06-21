"""Server-side access to shared target models.

Target data models live in the shared core package (`yj_studio_core.targets`)
used by both the desktop app and the server. This module ensures that package
is importable (adding `shared/src` to sys.path if a launcher didn't already) and
re-exports the public surface the server uses.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_shared_src_on_path() -> None:
    try:
        import yj_studio_core.targets  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    shared_src = repo_root / "shared" / "src"
    if shared_src.exists() and str(shared_src) not in sys.path:
        sys.path.insert(0, str(shared_src))


_ensure_shared_src_on_path()

from yj_studio_core.targets import (  # noqa: E402
    STAGE_PREFIX,
    STAGE_SUBDIR,
    GeoTarget,
    TargetFrame,
    TargetSet,
    TargetStage,
    TargetStatus,
    TargetStore,
    coerce_stage,
    export_confirmed_to_coco,
    export_stage_to_coco,
    frame_key,
    mask_volume_stats,
    normalise_target_type,
    relocate_target,
    resolve_voxel_spacing,
)

__all__ = [
    "STAGE_PREFIX",
    "STAGE_SUBDIR",
    "GeoTarget",
    "TargetFrame",
    "TargetSet",
    "TargetStage",
    "TargetStatus",
    "TargetStore",
    "coerce_stage",
    "export_confirmed_to_coco",
    "export_stage_to_coco",
    "frame_key",
    "mask_volume_stats",
    "normalise_target_type",
    "relocate_target",
    "resolve_voxel_spacing",
]
