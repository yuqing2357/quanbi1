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
    label: ClassVar[str] = "闭合等值线提取"
    description: ClassVar[str] = (
        "从层位深度网格中提取闭合等值线环；闭合环是候选构造闭合。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = ClosureContourParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}
