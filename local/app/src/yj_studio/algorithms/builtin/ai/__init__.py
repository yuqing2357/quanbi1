"""AI-powered algorithms (SAM3 etc.).

Importing this module registers each AI algorithm with the global
``AlgorithmRegistry`` via the ``@register_algorithm`` decorator. All AI
algorithms run in-process (``runs_in_subprocess = False``) so they can reach
the loaded SAM3 model via ``ctx.services['ai_service']``.
"""

from __future__ import annotations

from .sam3_propagate import SAM3PropagateAlgorithm
from .sam3_refine import SAM3RefineAlgorithm
from .sam3_segment import SAM3SegmentAlgorithm

__all__ = [
    "SAM3PropagateAlgorithm",
    "SAM3RefineAlgorithm",
    "SAM3SegmentAlgorithm",
]
