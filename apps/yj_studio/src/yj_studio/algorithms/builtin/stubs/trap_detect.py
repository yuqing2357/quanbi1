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
    label: ClassVar[str] = "圈闭检测"
    description: ClassVar[str] = (
        "结合层位闭合与断层多边形寻找候选构造圈闭，并输出带置信度的"
        " TrapLayer。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = TrapDetectParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}
