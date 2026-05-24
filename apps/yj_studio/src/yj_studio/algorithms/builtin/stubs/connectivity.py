from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class ConnectivityParams(BaseModel):
    neighbourhood: Literal["6", "18", "26"] = Field(default="26")
    min_component_size: int = Field(default=50, ge=1)


@register_algorithm
class ConnectivityAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "reservoir.connectivity"
    category: ClassVar[str] = "reservoir"
    label: ClassVar[str] = "连通性分析"
    description: ClassVar[str] = (
        "将三维掩膜分解为连通分量，并报告每个分量接触到哪些井。第二阶段提供。"
    )
    input_schema: ClassVar[type[BaseModel]] = ConnectivityParams
    layer_inputs: ClassVar[dict[str, str]] = {"mask": "mask"}
