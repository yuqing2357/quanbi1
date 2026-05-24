from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class TrapEvaluateParams(BaseModel):
    use_well_data: bool = Field(default=True)
    risk_weight: float = Field(default=0.5, ge=0.0, le=1.0)


@register_algorithm
class TrapEvaluateAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "trap.evaluate"
    category: ClassVar[str] = "trap"
    label: ClassVar[str] = "圈闭评估"
    description: ClassVar[str] = (
        "结合层位、断层和井证据评估候选 TrapLayer，并生成风险/质量表"
        "（AnnotationLayer）。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = TrapEvaluateParams
    layer_inputs: ClassVar[dict[str, str]] = {"trap": "trap"}
