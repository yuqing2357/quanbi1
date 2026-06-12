from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class FaultAutopickParams(BaseModel):
    attribute_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_surface_size: int = Field(default=200, ge=1)


@register_algorithm
class FaultAutopickAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "fault.autopick"
    category: ClassVar[str] = "fault"
    label: ClassVar[str] = "断层自动拾取"
    description: ClassVar[str] = (
        "基于断层增强属性体识别断层面（如相干体、ant-tracking）。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = FaultAutopickParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume"}
