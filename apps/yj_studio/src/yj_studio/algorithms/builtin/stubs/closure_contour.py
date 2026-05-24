from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class ClosureContourParams(BaseModel):
    contour_interval_m: float = Field(default=20.0, gt=0.0)
    min_closure_area: float = Field(default=1000.0, ge=0.0)


@register_algorithm
class ClosureContourAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "trap.closure_contour"
    category: ClassVar[str] = "trap"
    label: ClassVar[str] = "Closure Contour Extraction"
    description: ClassVar[str] = (
        "Extract closed contour rings from a horizon depth grid; closed rings"
        " are candidate structural closures. Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = ClosureContourParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}
