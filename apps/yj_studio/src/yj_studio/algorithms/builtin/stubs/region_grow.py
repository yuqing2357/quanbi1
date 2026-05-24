from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class RegionGrowParams(BaseModel):
    similarity_tolerance: float = Field(default=0.1, ge=0.0, le=1.0)
    max_cells: int = Field(default=500_000, ge=1)


@register_algorithm
class RegionGrowAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "mask.region_grow"
    category: ClassVar[str] = "reservoir"
    label: ClassVar[str] = "Region Grow"
    description: ClassVar[str] = (
        "Grow a seed mask outward while neighbours stay within tolerance of"
        " the seed mean amplitude. Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = RegionGrowParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume", "seed": "mask"}
