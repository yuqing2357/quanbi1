from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class HorizonAutotrackParams(BaseModel):
    seed_layer_role: str = Field(default="seed", description="Role key for the seed HorizonStickLayer.")
    similarity_window: int = Field(default=5, ge=1, le=51)
    max_iterations: int = Field(default=200, ge=1)


@register_algorithm
class HorizonAutotrackAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "horizon.autotrack"
    category: ClassVar[str] = "horizon"
    label: ClassVar[str] = "Horizon Autotrack (2D)"
    description: ClassVar[str] = (
        "Propagate a manually picked horizon stick across the volume by"
        " similarity-following along inline/xline. Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = HorizonAutotrackParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume", "seed": "horizon_stick"}
