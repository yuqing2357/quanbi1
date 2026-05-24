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
    label: ClassVar[str] = "Fault Auto-Pick"
    description: ClassVar[str] = (
        "Detect fault surfaces from a fault-emphasising attribute volume"
        " (e.g. coherence, ant-tracking). Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = FaultAutopickParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume"}
