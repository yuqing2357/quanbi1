"""Cross-slice propagation using the SAM3 video predictor.

Given a seed ``MaskLayer`` on a single slice, this algorithm:

1. Exports a contiguous band of slices around the seed as JPEGs.
2. Opens a SAM3 video session on that JPEG directory.
3. Seeds the session with a bounding box derived from the seed mask.
4. Calls ``propagate_in_video`` forward / backward as requested.
5. Stitches the per-frame 2D masks back into a single 3D ``MaskLayer``.

Like ``SAM3SegmentAlgorithm`` this runs in-process (it needs the loaded
SAM3 video predictor on the GPU). The runner schedules it on a QThread so
the UI keeps spinning.
"""

from __future__ import annotations

import logging
import shutil
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.ai.adapters import (
    export_axis_range_to_jpegs,
)
from yj_studio.ai.adapters.mask_to_layer import _to_numpy
from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import MaskLayer, VolumeLayer

logger = logging.getLogger(__name__)


class SAM3PropagateParams(BaseModel):
    forward_steps: int = Field(
        default=10, ge=0, le=500,
        description="向前传播的剖面数（朝更大的索引方向）。",
    )
    backward_steps: int = Field(
        default=10, ge=0, le=500,
        description="向后传播的剖面数（朝更小的索引方向）。",
    )
    text_prompt: str = Field(
        default="",
        description=(
            "可选文本提示，用来补充种子掩膜；留空时回退到 visual"
            "（仅几何提示）。"
        ),
    )
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    name_prefix: str = Field(default="SAM3 体掩膜")
    drop_low_confidence_frames: bool = Field(
        default=True,
        description=(
            "为 True 时，一旦某一帧的置信度低于 ``confidence_threshold`` 就"
            "停止传播。"
        ),
    )


class SAM3PropagateOutput(BaseModel):
    frames_covered: int


@register_algorithm
class SAM3PropagateAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.propagate"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 跨剖面传播"
    description: ClassVar[str] = (
        "使用 SAM3 视频跟踪器，将单张二维 MaskLayer 种子传播到相邻剖面。"
        " 输出为覆盖 ``forward_steps + backward_steps + 1`` 个剖面的三维"
        " MaskLayer。"
    )
    input_schema: ClassVar[type[BaseModel]] = SAM3PropagateParams
    output_schema: ClassVar[type[BaseModel]] = SAM3PropagateOutput
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume", "seed_mask": "mask"}
    runs_in_subprocess: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = True

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        volume_layer = ctx.input_layers.get("volume")
        seed_layer = ctx.input_layers.get("seed_mask")
        if not isinstance(volume_layer, VolumeLayer) or volume_layer.shape is None:
            return AlgorithmResult.failure("需要一个作为 'volume' 输入的 VolumeLayer")
        if not isinstance(seed_layer, MaskLayer) or seed_layer.mask is None:
            return AlgorithmResult.failure("需要一个二维 MaskLayer 作为 'seed_mask' 输入")
        if seed_layer.axis not in {"inline", "xline", "z"}:
            return AlgorithmResult.failure(
                f"种子掩膜的轴未知：{seed_layer.axis!r}"
            )
        if seed_layer.slice_index is None:
            return AlgorithmResult.failure("种子掩膜没有 slice_index")

        ai_service = ctx.services.get("ai_service")
        volume_store = ctx.services.get("volume_store")
        if ai_service is None or volume_store is None:
            return AlgorithmResult.failure(
                "所需服务缺失，请先打开 AI 面板并启动 AI。"
            )
        if not ai_service.is_ready():
            return AlgorithmResult.failure(
                f"AI 服务未就绪（状态={ai_service.state.value}）。"
            )
        predictor = ai_service.video_predictor
        if predictor is None:
            return AlgorithmResult.failure(
                "SAM3 视频预测器未加载。请在 SAM3Config 中启用 load_video_model"
                " 并重启 AI 服务。"
            )

        axis = str(seed_layer.axis)
        seed_index = int(seed_layer.slice_index)
        nx, ny, nz = volume_layer.shape
        axis_size = {"inline": nx, "xline": ny, "z": nz}[axis]

        first = max(0, seed_index - int(ctx.params.backward_steps))
        last = min(axis_size - 1, seed_index + int(ctx.params.forward_steps))
        if last < first:
            return AlgorithmResult.failure("传播范围为空")
        indices = list(range(first, last + 1))
        seed_frame_idx = indices.index(seed_index)

        ctx.report_progress(0.05, f"导出 {len(indices)} 帧")
        export = export_axis_range_to_jpegs(
            volume_store,
            volume_layer.volume_id,
            axis,
            indices,
            clim=volume_layer.clim,
        )

        ai_service.mark_busy("SAM3 传播中")
        session_state = None
        try:
            ctx.check_cancel()
            box_norm = _seed_box_normalised(seed_layer.mask, export.width, export.height)
            if box_norm is None:
                return AlgorithmResult.failure(
                    "种子掩膜为空，无法推导边界框。"
                )

            ctx.report_progress(0.1, "初始化 SAM3 视频会话")
            session_state = predictor.init_state(
                resource_path=str(export.directory),
                async_loading_frames=False,
                video_loader_type="jpg",
            )
            text = ctx.params.text_prompt.strip() or "visual"

            ctx.report_progress(0.2, "添加种子提示")
            predictor.add_prompt(
                inference_state=session_state,
                frame_idx=seed_frame_idx,
                text_str=text,
                points=None,
                point_labels=None,
                boxes_xywh=[list(box_norm)],
                box_labels=[1],
                obj_id=1,
            )

            ctx.report_progress(0.25, "传播中")
            collected: dict[int, np.ndarray] = {}
            scores: dict[int, float] = {}
            total_frames = len(indices)

            def _consume(generator) -> None:
                for frame_idx, outputs in generator:
                    ctx.check_cancel()
                    mask, score = _extract_first_object(outputs)
                    if mask is not None:
                        collected[int(frame_idx)] = mask
                        if score is not None:
                            scores[int(frame_idx)] = float(score)
                    done = len(collected)
                    ctx.report_progress(
                        0.25 + 0.7 * (done / max(total_frames, 1)),
                        f"第 {int(frame_idx) - seed_frame_idx + seed_index} 帧",
                    )
                    if (
                        ctx.params.drop_low_confidence_frames
                        and score is not None
                        and float(score) < ctx.params.confidence_threshold
                    ):
                        break

            if ctx.params.forward_steps > 0:
                _consume(
                    predictor.model.propagate_in_video(
                        inference_state=session_state,
                        start_frame_idx=seed_frame_idx,
                        max_frame_num_to_track=ctx.params.forward_steps,
                        reverse=False,
                    )
                )
            if ctx.params.backward_steps > 0:
                _consume(
                    predictor.model.propagate_in_video(
                        inference_state=session_state,
                        start_frame_idx=seed_frame_idx,
                        max_frame_num_to_track=ctx.params.backward_steps,
                        reverse=True,
                    )
                )
        finally:
            ai_service.mark_ready()
            try:
                export.cleanup()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to clean up SAM3 frames")

        if not collected:
            return AlgorithmResult.failure("传播没有产生任何帧")

        ctx.report_progress(0.95, "叠加三维掩膜")
        ordered_frames = sorted(collected.items(), key=lambda kv: kv[0])
        height, width = ordered_frames[0][1].shape
        volume_mask = np.zeros((len(indices), height, width), dtype=bool)
        confidence_arr = np.zeros(len(indices), dtype=np.float32)
        for frame_idx, mask in ordered_frames:
            volume_mask[frame_idx] = mask
            confidence_arr[frame_idx] = scores.get(frame_idx, 0.0)

        axis_name = {"inline": "纵向", "xline": "横向", "z": "Z向"}.get(axis, axis)
        layer = MaskLayer(
            name=f"{ctx.params.name_prefix}（{axis_name} {first}-{last}）",
            mask=volume_mask,
            confidence=confidence_arr,
            axis=axis,
            slice_index=seed_index,
            color=(1.0, 0.4, 0.2, 0.45),
            opacity=0.45,
            visible=True,
            metadata={
                "axis": axis,
                "frame_indices": indices,
                "seed_slice_index": seed_index,
                "volume_id": volume_layer.volume_id,
                "text_prompt": ctx.params.text_prompt,
            },
            provenance={
                "source": "ai.sam3.propagate",
                "seed_mask_id": seed_layer.id,
            },
        )

        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=[layer],
            summary=(
                f"SAM3 传播：在 {axis_name}={seed_index} 周围覆盖"
                f" {len(collected)}/{len(indices)} 帧"
            ),
        )


def _seed_box_normalised(
    mask: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Return the seed mask's bounding box in SAM3's normalised xywh format."""

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 2 or mask_arr.size == 0 or not mask_arr.any():
        return None
    # YJ Studio MaskLayer convention: axis1 × axis2, i.e. coords[:, 0] is
    # the in-slice horizontal index and coords[:, 1] is the vertical index.
    # SAM3 video frames are exported as (H=z, W=inline_or_xline) JPEGs, so
    # we map axis1 → image x and axis2 → image y.
    src_w, src_h = mask_arr.shape
    coords = np.argwhere(mask_arr)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    scale_x = float(width) / float(src_w)
    scale_y = float(height) / float(src_h)
    x0 = float(x_min) * scale_x
    x1 = float(x_max + 1) * scale_x
    y0 = float(y_min) * scale_y
    y1 = float(y_max + 1) * scale_y
    cx = (x0 + x1) / 2.0 / float(width)
    cy = (y0 + y1) / 2.0 / float(height)
    bw = (x1 - x0) / float(width)
    bh = (y1 - y0) / float(height)
    return (cx, cy, bw, bh)


def _extract_first_object(outputs) -> tuple[np.ndarray | None, float | None]:
    """SAM3 propagation outputs are usually ``{obj_id: {"masks": tensor,
    "scores": tensor}}``. We track a single object (obj_id=1), so we extract
    the first one we find.
    """

    if not isinstance(outputs, dict) or not outputs:
        return None, None
    first_key = next(iter(outputs))
    payload = outputs[first_key]
    if isinstance(payload, dict):
        masks = payload.get("masks")
        scores = payload.get("scores")
    else:
        masks = payload
        scores = None
    if masks is None:
        return None, None
    masks_np = _to_numpy(masks)
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    if masks_np.ndim == 3:
        masks_np = masks_np[0]
    mask = masks_np > 0
    score: float | None = None
    if scores is not None:
        scores_np = _to_numpy(scores).reshape(-1)
        if scores_np.size > 0:
            score = float(scores_np[0])
    return mask, score
