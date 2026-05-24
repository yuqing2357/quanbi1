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
        description="Number of slices to propagate forward (toward higher indices).",
    )
    backward_steps: int = Field(
        default=10, ge=0, le=500,
        description="Number of slices to propagate backward (toward lower indices).",
    )
    text_prompt: str = Field(
        default="",
        description=(
            "Optional text prompt that supplements the seed mask. Empty"
            " falls back to 'visual' (geometric prompt only)."
        ),
    )
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    name_prefix: str = Field(default="SAM3 Volume Mask")
    drop_low_confidence_frames: bool = Field(
        default=True,
        description=(
            "When True, stops propagation as soon as a frame's confidence"
            " falls below ``confidence_threshold``."
        ),
    )


class SAM3PropagateOutput(BaseModel):
    frames_covered: int


@register_algorithm
class SAM3PropagateAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.propagate"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 Cross-Slice Propagate"
    description: ClassVar[str] = (
        "Take a 2D MaskLayer seed and propagate it through neighbouring"
        " slices using the SAM3 video tracker. Output is a 3D MaskLayer"
        " covering ``forward_steps + backward_steps + 1`` slices."
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
            return AlgorithmResult.failure("Need a VolumeLayer as 'volume' input")
        if not isinstance(seed_layer, MaskLayer) or seed_layer.mask is None:
            return AlgorithmResult.failure("Need a 2D MaskLayer as 'seed_mask' input")
        if seed_layer.axis not in {"inline", "xline", "z"}:
            return AlgorithmResult.failure(
                f"Seed mask has unknown axis {seed_layer.axis!r}"
            )
        if seed_layer.slice_index is None:
            return AlgorithmResult.failure("Seed mask has no slice_index")

        ai_service = ctx.services.get("ai_service")
        volume_store = ctx.services.get("volume_store")
        if ai_service is None or volume_store is None:
            return AlgorithmResult.failure(
                "Required services missing. Open the AI Dock and start AI."
            )
        if not ai_service.is_ready():
            return AlgorithmResult.failure(
                f"AI service not ready (state={ai_service.state.value})."
            )
        predictor = ai_service.video_predictor
        if predictor is None:
            return AlgorithmResult.failure(
                "SAM3 video predictor is not loaded. Enable load_video_model"
                " in SAM3Config and restart the AI service."
            )

        axis = str(seed_layer.axis)
        seed_index = int(seed_layer.slice_index)
        nx, ny, nz = volume_layer.shape
        axis_size = {"inline": nx, "xline": ny, "z": nz}[axis]

        first = max(0, seed_index - int(ctx.params.backward_steps))
        last = min(axis_size - 1, seed_index + int(ctx.params.forward_steps))
        if last < first:
            return AlgorithmResult.failure("Propagation range is empty")
        indices = list(range(first, last + 1))
        seed_frame_idx = indices.index(seed_index)

        ctx.report_progress(0.05, f"Exporting {len(indices)} frames")
        export = export_axis_range_to_jpegs(
            volume_store,
            volume_layer.volume_id,
            axis,
            indices,
            clim=volume_layer.clim,
        )

        ai_service.mark_busy("SAM3 propagating")
        session_state = None
        try:
            ctx.check_cancel()
            box_norm = _seed_box_normalised(seed_layer.mask, export.width, export.height)
            if box_norm is None:
                return AlgorithmResult.failure(
                    "Seed mask is empty — cannot derive a bounding box."
                )

            ctx.report_progress(0.1, "Initialising SAM3 video session")
            session_state = predictor.init_state(
                resource_path=str(export.directory),
                async_loading_frames=False,
                video_loader_type="jpg",
            )
            text = ctx.params.text_prompt.strip() or "visual"

            ctx.report_progress(0.2, "Adding seed prompt")
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

            ctx.report_progress(0.25, "Propagating")
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
                        f"Frame {int(frame_idx) - seed_frame_idx + seed_index}",
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
            return AlgorithmResult.failure("Propagation produced no frames")

        ctx.report_progress(0.95, "Stacking 3D mask")
        ordered_frames = sorted(collected.items(), key=lambda kv: kv[0])
        height, width = ordered_frames[0][1].shape
        volume_mask = np.zeros((len(indices), height, width), dtype=bool)
        confidence_arr = np.zeros(len(indices), dtype=np.float32)
        for frame_idx, mask in ordered_frames:
            volume_mask[frame_idx] = mask
            confidence_arr[frame_idx] = scores.get(frame_idx, 0.0)

        layer = MaskLayer(
            name=f"{ctx.params.name_prefix} ({axis} {first}-{last})",
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

        ctx.report_progress(1.0, "Done")
        return AlgorithmResult.success(
            output_layers=[layer],
            summary=(
                f"SAM3 propagation: covered {len(collected)} of {len(indices)}"
                f" frames around {axis}={seed_index}"
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
