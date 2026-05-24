"""Built-in algorithms shipped with YJ Studio.

Importing this package registers every algorithm with the global
``AlgorithmRegistry`` via the ``@register_algorithm`` decorator side effect.
Phase-2 stubs live under :mod:`yj_studio.algorithms.builtin.stubs`.
"""

from __future__ import annotations

from .measure import MeasureAreaAlgorithm, MeasureDistanceAlgorithm
from .thickness import ThicknessAlgorithm
from . import ai as _ai  # noqa: F401 — registers SAM3 algorithms
from . import stubs  # noqa: F401 — imported for the registration side effect

__all__ = [
    "MeasureAreaAlgorithm",
    "MeasureDistanceAlgorithm",
    "ThicknessAlgorithm",
]
