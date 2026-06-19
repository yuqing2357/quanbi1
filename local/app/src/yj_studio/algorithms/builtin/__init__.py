"""Built-in algorithms shipped with YJ Studio.

Importing this package registers every algorithm with the global
``AlgorithmRegistry`` via the ``@register_algorithm`` decorator side effect.
Phase-2 stubs live under :mod:`yj_studio.algorithms.builtin.stubs`.

SAM3 is intentionally not registered here: the only user-facing SAM3 entry is
the AI dock, which uses a remote-only descriptor from
``yj_studio.algorithms.remote_sam3`` and submits ``/sam3/jobs`` through
``RemoteSAM3Client``.
"""

from __future__ import annotations

from .closure_contour import ClosureContourAlgorithm
from .connectivity import ConnectivityAlgorithm
from .sandbody_extract import SandbodyExtractAlgorithm
from .thickness import ThicknessAlgorithm
from .trap_detect import TrapDetectAlgorithm
from .trap_evaluate import TrapEvaluateAlgorithm
from . import stubs  # noqa: F401 — imported for the registration side effect

__all__ = [
    "ClosureContourAlgorithm",
    "ConnectivityAlgorithm",
    "SandbodyExtractAlgorithm",
    "ThicknessAlgorithm",
    "TrapDetectAlgorithm",
    "TrapEvaluateAlgorithm",
]
