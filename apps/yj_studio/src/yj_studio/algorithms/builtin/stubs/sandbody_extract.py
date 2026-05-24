from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class SandbodyExtractParams(BaseModel):
    min_thickness_m: float = Field(default=2.0, ge=0.0)
    amplitude_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


@register_algorithm
class SandbodyExtractAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "reservoir.sandbody_extract"
    category: ClassVar[str] = "reservoir"
    label: ClassVar[str] = "砂体提取"
    description: ClassVar[str] = (
        "基于地震属性阈值识别上下层位之间的砂体，输出 MaskLayer 及"
        "厚度统计。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = SandbodyExtractParams
    layer_inputs: ClassVar[dict[str, str]] = {
        "volume": "volume",
        "top": "horizon",
        "bottom": "horizon",
    }
