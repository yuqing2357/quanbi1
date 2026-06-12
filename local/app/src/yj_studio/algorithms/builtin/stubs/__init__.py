"""Phase-2 stub algorithms.

These classes carry full pydantic schemas so the AlgorithmDock can render
their parameter forms, but their ``run`` method raises ``NotImplementedError``
to signal the feature is queued for Phase 2 of the project (cf.
``docs/implementation_plan.md`` §8). Importing this module is enough to
register all stubs with the global ``AlgorithmRegistry``.
"""

from __future__ import annotations

from .horizon_autotrack import HorizonAutotrackAlgorithm
from .auto_track_horizon_3d import AutoTrackHorizon3DAlgorithm
from .fault_autopick import FaultAutopickAlgorithm
from .region_grow import RegionGrowAlgorithm

__all__ = [
    "AutoTrackHorizon3DAlgorithm",
    "FaultAutopickAlgorithm",
    "HorizonAutotrackAlgorithm",
    "RegionGrowAlgorithm",
]
