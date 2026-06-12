from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class HorizonAutotrackParams(BaseModel):
    seed_layer_role: str = Field(default="seed", description="种子层位杆对应的角色键。")
    similarity_window: int = Field(default=5, ge=1, le=51, description="相似性搜索窗口大小。")
    max_iterations: int = Field(default=200, ge=1, description="最大迭代次数。")


@register_algorithm
class HorizonAutotrackAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "horizon.autotrack"
    category: ClassVar[str] = "horizon"
    label: ClassVar[str] = "层位自动追踪（2D）"
    description: ClassVar[str] = (
        "沿 Inline/Xline 相似性在体内传播手工拾取的层位杆。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = HorizonAutotrackParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume", "seed": "horizon_stick"}
