"""Single-slice segmentation through SAM3 image model.

Pipeline:

1. Pull the requested slice from the active ``VolumeLayer`` via
   ``volume_store`` (registered as a service on the runner).
2. Stretch to 8-bit RGB with :mod:`yj_studio.ai.adapters.volume_to_image`.
3. Push image + prompts into ``Sam3Processor`` (from ``ai_service``).
4. Decode SAM3 state into one or more candidate ``MaskLayer`` instances.

This algorithm runs **in-process** because the SAM3 model lives on the
GPU; the runner schedules it on a ``QThread`` so the UI stays responsive.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.ai.adapters import build_mask_layer, sam3_mask_to_layer, slice_to_rgb_image
from yj_studio.ai.adapters.mask_to_layer import decode_sam3_masks
from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import MaskLayer, VolumeLayer


PointSpec = tuple[float, float]
"""(inline_or_xline_coord, sample_coord). Order matches the slice's local
(x, y) axes (i.e. the same coords the user clicks in the 2D view)."""

BoxSpec = tuple[float, float, float, float]
"""(x_min, y_min, x_max, y_max) in slice pixel coords (NOT normalised)."""


class SAM3SegmentParams(BaseModel):
    axis: Literal["inline", "xline", "z"] = Field(
        default="inline", description="沿哪个轴切取体数据。"
    )
    slice_index: int = Field(
        default=0, ge=0, description="所选轴上的剖面索引。"
    )
    text_prompt: str = Field(
        default="",
        description=(
            "可选文本提示（如“盐丘”“河道砂体”）。留空时仅使用几何提示。"
        ),
    )
    boxes: list[BoxSpec] = Field(
        default_factory=list,
        description=(
            "可选正样本框，像素坐标为 (x_min, y_min, x_max, y_max)。"
        ),
    )
    points: list[PointSpec] = Field(
        default_factory=list,
        description=(
            "可选正样本点提示。每个点会先转换成一个小框再送入 SAM3，"
            "因为 SAM3 没有原生点提示接口。"
        ),
    )
    point_box_radius_px: float = Field(
        default=8.0,
        gt=0.0,
        description="包裹每个点的伪框半宽。",
    )
    confidence_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="丢弃低于此阈值的掩膜。",
    )
    keep_top_k: int = Field(
        default=3, ge=1, le=50,
        description="最多保留的候选掩膜数。",
    )
    name_prefix: str = Field(
        default="SAM3 掩膜",
        description="结果 MaskLayer 的名称前缀。",
    )


class SAM3SegmentOutput(BaseModel):
    candidates: int


@register_algorithm
class SAM3SegmentAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.segment"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 单剖面分割"
    description: ClassVar[str] = (
        "在单张二维剖面上运行 SAM3 图像分割。支持自由文本、框提示和点提示"
        "（点会先转换成小框，因为 SAM3 没有原生点提示接口）。输出为每个"
        "保留候选对应的一个 MaskLayer。"
    )
    input_schema: ClassVar[type[BaseModel]] = SAM3SegmentParams
    output_schema: ClassVar[type[BaseModel]] = SAM3SegmentOutput
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume"}
    runs_in_subprocess: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = True

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        volume_layer = ctx.input_layers.get("volume")
        if not isinstance(volume_layer, VolumeLayer):
            return AlgorithmResult.failure("需要一个作为 'volume' 输入的 VolumeLayer")

        ai_service = ctx.services.get("ai_service")
        volume_store = ctx.services.get("volume_store")
        if ai_service is None or volume_store is None:
            return AlgorithmResult.failure(
                "所需服务缺失：'ai_service' 和 'volume_store'。"
                " 请先在 AI 面板中启动 AI。"
            )
        if not ai_service.is_ready():
            return AlgorithmResult.failure(
                f"AI 服务未就绪（状态={ai_service.state.value}）。"
            )

        ctx.report_progress(0.05, "读取剖面")
        slice_arr = volume_store.get_slice(
            volume_layer.volume_id, ctx.params.axis, ctx.params.slice_index
        )
        # Match the orientation the 2D view uses (see volume_slice_renderer:
        # both inline and xline slices are transposed to (Z, other)).
        if ctx.params.axis in {"inline", "xline"}:
            slice2d = np.asarray(slice_arr, dtype=np.float32).T
        else:
            slice2d = np.asarray(slice_arr, dtype=np.float32).T
        rgb = slice_to_rgb_image(slice2d, clim=volume_layer.clim)
        height, width = rgb.shape[:2]

        ctx.report_progress(0.2, "发送图像到 SAM3")
        ai_service.mark_busy("SAM3 分割中")
        try:
            processor = ai_service.image_processor
            processor.set_confidence_threshold(float(ctx.params.confidence_threshold))
            # SAM3 wants a PIL.Image or a tensor; PIL keeps the path simple.
            from PIL import Image  # local import: optional dep boundary

            pil = Image.fromarray(rgb)
            state = processor.set_image(pil)

            if ctx.params.text_prompt.strip():
                ctx.report_progress(0.4, "应用文本提示")
                state = processor.set_text_prompt(
                    prompt=ctx.params.text_prompt.strip(), state=state
                )

            ctx.check_cancel()

            for box in ctx.params.boxes:
                x0, y0, x1, y1 = box
                state = _apply_box_prompt(processor, state, x0, y0, x1, y1, width, height)
                ctx.check_cancel()

            radius = float(ctx.params.point_box_radius_px)
            for px, py in ctx.params.points:
                x0 = px - radius
                y0 = py - radius
                x1 = px + radius
                y1 = py + radius
                state = _apply_box_prompt(processor, state, x0, y0, x1, y1, width, height)
                ctx.check_cancel()

            ctx.report_progress(0.85, "解码掩膜")
            detections = decode_sam3_masks(state)
        finally:
            ai_service.mark_ready()

        detections.sort(key=lambda d: d["score"], reverse=True)
        detections = detections[: int(ctx.params.keep_top_k)]

        output_layers: list[MaskLayer] = []
        axis_name = {"inline": "Inline", "xline": "Xline", "z": "Z"}.get(ctx.params.axis, ctx.params.axis)
        for i, det in enumerate(detections, start=1):
            # Convert SAM3 image-order mask (rows=samples, cols=inline/xline)
            # to MaskLayer orientation via the single canonical helper — do not
            # hand-write another .T. See mask_to_layer.sam3_mask_to_layer.
            sam3_mask = sam3_mask_to_layer(det["mask"])
            output_layers.append(
                build_mask_layer(
                    sam3_mask,
                    name=f"{ctx.params.name_prefix} 第{i}个（{det['score']:.2f}）",
                    axis=ctx.params.axis,
                    slice_index=int(ctx.params.slice_index),
                    score=det["score"],
                    metadata={
                        "box": list(det["box"]),
                        "text_prompt": ctx.params.text_prompt,
                        "volume_id": volume_layer.volume_id,
                    },
                )
            )

        ctx.report_progress(1.0, "完成")
        summary = (
            f"SAM3：在 {axis_name}={ctx.params.slice_index} 上生成"
            f" {len(output_layers)} 个候选掩膜"
        )
        return AlgorithmResult.success(output_layers=output_layers, summary=summary)


def _apply_box_prompt(processor, state, x0: float, y0: float, x1: float, y1: float,
                      width: int, height: int):
    """Convert a pixel-space box to SAM3's normalised [cx, cy, w, h] format
    and append it via ``add_geometric_prompt``."""

    x0 = max(0.0, min(float(x0), width - 1.0))
    x1 = max(0.0, min(float(x1), width - 1.0))
    y0 = max(0.0, min(float(y0), height - 1.0))
    y1 = max(0.0, min(float(y1), height - 1.0))
    if x1 <= x0 or y1 <= y0:
        return state
    cx = (x0 + x1) / 2.0 / float(width)
    cy = (y0 + y1) / 2.0 / float(height)
    bw = (x1 - x0) / float(width)
    bh = (y1 - y0) / float(height)
    return processor.add_geometric_prompt(
        box=[cx, cy, bw, bh], label=True, state=state
    )
