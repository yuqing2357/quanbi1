from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class TrapDetectParams(BaseModel):
    structural_only: bool = Field(default=True)
    score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


@register_algorithm
class TrapDetectAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "trap.detect_structural"
    category: ClassVar[str] = "trap"
    label: ClassVar[str] = "Trap Detection"
    description: ClassVar[str] = (
        "Combine horizon closures with fault polygons to find candidate"
        " structural traps; emit TrapLayer with confidence score. Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = TrapDetectParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}
