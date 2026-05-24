"""Re-run SAM3 image segmentation using a previously edited mask as a hint.

SAM3 has no native mask-prompt API, so this algorithm extracts the
bounding box of the edited ``MaskLayer`` (after Brush/Eraser tweaks) and
feeds it as a box prompt alongside the original text prompt. The result
is a fresh set of candidate masks that respect the user's edit.

The companion ``MaskLayer.provenance`` carries ``text_prompt`` from the
original segmentation, so the refine step reuses it automatically.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.ai.adapters import build_mask_layer, slice_to_rgb_image
from yj_studio.ai.adapters.mask_to_layer import decode_sam3_masks
from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.builtin.ai.sam3_segment import _apply_box_prompt
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import MaskLayer, VolumeLayer


class SAM3RefineParams(BaseModel):
    text_prompt_override: str = Field(
        default="",
        description=(
            "如果不为空，则覆盖编辑后掩膜来源中存储的文本提示；留空则"
            "复用原始提示。"
        ),
    )
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    keep_top_k: int = Field(default=3, ge=1, le=50)
    pad_box_px: float = Field(
        default=4.0,
        ge=0.0,
        description="将掩膜边界框向四周外扩的像素数。",
    )


class SAM3RefineOutput(BaseModel):
    candidates: int


@register_algorithm
class SAM3RefineAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.refine"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 掩膜精修"
    description: ClassVar[str] = (
        "基于已编辑的 MaskLayer 重新分割同一剖面。算法会从编辑后的掩膜"
        " 中提取边界框，并将其作为正样本框与原始文本提示一起送入 SAM3。"
        " 原始文本提示（如果有）会从掩膜来源中恢复，除非被覆盖。"
    )
    input_schema: ClassVar[type[BaseModel]] = SAM3RefineParams
    output_schema: ClassVar[type[BaseModel]] = SAM3RefineOutput
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume", "edited_mask": "mask"}
    runs_in_subprocess: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = True

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        volume_layer = ctx.input_layers.get("volume")
        edited = ctx.input_layers.get("edited_mask")
        if not isinstance(volume_layer, VolumeLayer):
            return AlgorithmResult.failure("需要一个作为 'volume' 输入的 VolumeLayer")
        if not isinstance(edited, MaskLayer) or edited.mask is None:
            return AlgorithmResult.failure("需要一个已编辑的二维 MaskLayer 作为 'edited_mask'")
        if edited.axis not in {"inline", "xline", "z"}:
            return AlgorithmResult.failure(
                f"编辑后的掩膜轴必须是 Inline、Xline 或 Z，当前为 {edited.axis!r}"
            )
        if edited.slice_index is None:
            return AlgorithmResult.failure("编辑后的掩膜没有 slice_index")
        if edited.mask.ndim != 2:
            return AlgorithmResult.failure(
                "精修仅支持二维掩膜（3D 请使用传播算法）"
            )

        ai_service = ctx.services.get("ai_service")
        volume_store = ctx.services.get("volume_store")
        if ai_service is None or volume_store is None:
            return AlgorithmResult.failure(
                "所需服务缺失，请先在 AI 面板中启动 AI。"
            )
        if not ai_service.is_ready():
            return AlgorithmResult.failure(
                f"AI 服务未就绪（状态={ai_service.state.value}）。"
            )

        mask_arr = np.asarray(edited.mask, dtype=bool)
        if not mask_arr.any():
            return AlgorithmResult.failure("编辑后的掩膜为空")

        ctx.report_progress(0.05, "读取剖面")
        raw = volume_store.get_slice(volume_layer.volume_id, edited.axis, int(edited.slice_index))
        if edited.axis in {"inline", "xline"}:
            slice2d = np.asarray(raw, dtype=np.float32).T
        else:
            slice2d = np.asarray(raw, dtype=np.float32).T
        rgb = slice_to_rgb_image(slice2d, clim=volume_layer.clim)
        height, width = rgb.shape[:2]

        # Derive a padded box in the slice image's coord space.
        #
        # YJ Studio stores masks in axis1 × axis2 layout (see sam3_segment.py
        # comment), so coords[:, 0] is the in-slice horizontal index
        # (inline / xline) and coords[:, 1] is the vertical index (z / sample).
        # SAM3's image, however, has shape (H=z, W=inline_or_xline), so we
        # have to swap when remapping.
        src_w, src_h = mask_arr.shape  # axis1 → image width, axis2 → image height
        coords = np.argwhere(mask_arr)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        scale_x = float(width) / float(src_w)
        scale_y = float(height) / float(src_h)
        x0 = float(x_min) * scale_x - float(ctx.params.pad_box_px)
        x1 = float(x_max + 1) * scale_x + float(ctx.params.pad_box_px)
        y0 = float(y_min) * scale_y - float(ctx.params.pad_box_px)
        y1 = float(y_max + 1) * scale_y + float(ctx.params.pad_box_px)

        text_prompt = ctx.params.text_prompt_override.strip() or str(
            edited.metadata.get("text_prompt") or ""
        ).strip()

        ai_service.mark_busy("SAM3 精修中")
        try:
            from PIL import Image

            processor = ai_service.image_processor
            processor.set_confidence_threshold(float(ctx.params.confidence_threshold))
            pil = Image.fromarray(rgb)
            state = processor.set_image(pil)

            ctx.report_progress(0.4, "应用提示")
            if text_prompt:
                state = processor.set_text_prompt(prompt=text_prompt, state=state)
            ctx.check_cancel()
            state = _apply_box_prompt(processor, state, x0, y0, x1, y1, width, height)
            ctx.check_cancel()

            ctx.report_progress(0.8, "解码掩膜")
            detections = decode_sam3_masks(state)
        finally:
            ai_service.mark_ready()

        detections.sort(key=lambda d: d["score"], reverse=True)
        detections = detections[: int(ctx.params.keep_top_k)]
        output_layers: list[MaskLayer] = []
        axis_name = {"inline": "Inline", "xline": "Xline", "z": "Z"}.get(edited.axis, edited.axis)
        for i, det in enumerate(detections, start=1):
            # See sam3_segment.py for why the SAM3 mask is transposed at the
            # AI → scene seam: brush + MaskLayer renderer + view_2d_section
            # all assume axis-1 × axis-2 mask layout, but SAM3 returns
            # row-major image pixels (H × W).
            sam3_mask = np.ascontiguousarray(np.asarray(det["mask"]).T)
            output_layers.append(
                build_mask_layer(
                    sam3_mask,
                    name=f"{edited.name}（精修第{i}个，{det['score']:.2f}）",
                    axis=edited.axis,
                    slice_index=int(edited.slice_index),
                    score=det["score"],
                    metadata={
                        "box": list(det["box"]),
                        "text_prompt": text_prompt,
                        "volume_id": volume_layer.volume_id,
                        "refined_from": edited.id,
                    },
                    provenance={
                        "source": "ai.sam3.refine",
                        "refined_from_id": edited.id,
                    },
                )
            )

        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=output_layers,
            summary=(
                f"精修完成：在 {axis_name}={edited.slice_index} 上生成"
                f" {len(output_layers)} 个候选"
            ),
        )
