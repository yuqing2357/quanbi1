from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from yj_studio.algorithms.registry import register_algorithm

from ._base import PhaseTwoStub


class AutoTrackHorizon3DParams(BaseModel):
    similarity_window: int = Field(default=7, ge=1, le=51)
    smoothness_weight: float = Field(default=0.5, ge=0.0, le=10.0)
    max_extent: int = Field(default=500, ge=1)


@register_algorithm
class AutoTrackHorizon3DAlgorithm(PhaseTwoStub):
    id: ClassVar[str] = "horizon.autotrack_3d"
    category: ClassVar[str] = "horizon"
    label: ClassVar[str] = "Horizon Autotrack (3D)"
    description: ClassVar[str] = (
        "3D region-grow style horizon propagation from a HorizonStick seed."
        " Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = AutoTrackHorizon3DParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume", "seed": "horizon_stick"}
