from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .algorithm import Algorithm
from .context import AlgorithmContext
from .result import AlgorithmResult


PointSpec = tuple[float, float]
BoxSpec = tuple[float, float, float, float]


class RemoteSAM3SegmentParams(BaseModel):
    axis: Literal["inline", "xline", "z"] = Field(default="inline")
    slice_index: int = Field(default=0, ge=0)
    text_prompt: str = ""
    boxes: list[BoxSpec] = Field(default_factory=list)
    points: list[PointSpec] = Field(default_factory=list)
    point_box_radius_px: float = Field(default=8.0, gt=0.0)
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    keep_top_k: int = Field(default=3, ge=1, le=50)
    target_type: str = "unknown"
    name_prefix: str = "SAM3"


class RemoteSAM3SegmentAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.segment"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 远程分割"
    description: ClassVar[str] = "AI 面板专用描述类；实际执行由 RemoteSAM3Task 提交到服务器 /sam3/jobs。"
    input_schema: ClassVar[type[BaseModel]] = RemoteSAM3SegmentParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume"}
    runs_in_subprocess: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = True

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        return AlgorithmResult.failure("SAM3 分割只允许通过远程 /sam3/jobs 执行。")
