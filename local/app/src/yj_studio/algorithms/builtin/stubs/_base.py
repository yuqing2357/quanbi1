"""Shared scaffolding for Phase-2 stub algorithms."""

from __future__ import annotations

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.result import AlgorithmResult


class PhaseTwoStub(Algorithm):
    """Base class that fails fast with a friendly message on ``run``."""

    runs_in_subprocess = False  # stubs do nothing useful; cheap to run in-proc

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        return AlgorithmResult.failure(
            f"'{self.label}' is a Phase-2 stub. Schema is defined but the"
            " algorithm has not been implemented yet."
        )
