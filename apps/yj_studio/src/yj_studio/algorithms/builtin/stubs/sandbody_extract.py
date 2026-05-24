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
    label: ClassVar[str] = "Sandbody Extraction"
    description: ClassVar[str] = (
        "Identify sandbodies between top/bottom horizons using a"
        " seismic-attribute threshold. Outputs a MaskLayer plus thickness"
        " statistics. Phase 2."
    )
    input_schema: ClassVar[type[BaseModel]] = SandbodyExtractParams
    layer_inputs: ClassVar[dict[str, str]] = {
        "volume": "volume",
        "top": "horizon",
        "bottom": "horizon",
    }
