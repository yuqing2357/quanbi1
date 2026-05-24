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

from yj_studio.ai.adapters import build_mask_layer, slice_to_rgb_image
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
        default="inline", description="Which axis to slice the volume along."
    )
    slice_index: int = Field(
        default=0, ge=0, description="Index of the slice on the chosen axis."
    )
    text_prompt: str = Field(
        default="",
        description=(
            "Free-form text prompt (e.g. 'salt body', 'channel sand'). When"
            " empty SAM3 still runs but only relies on geometric prompts."
        ),
    )
    boxes: list[BoxSpec] = Field(
        default_factory=list,
        description=(
            "Optional positive bounding boxes in pixel coords"
            " (x_min, y_min, x_max, y_max) of the slice image."
        ),
    )
    points: list[PointSpec] = Field(
        default_factory=list,
        description=(
            "Optional positive point hints. Each point is converted into a"
            " small box (point ± point_box_radius_px) before being fed to"
            " SAM3, which has no native point prompt API."
        ),
    )
    point_box_radius_px: float = Field(
        default=8.0,
        gt=0.0,
        description="Half-width of the pseudo-box wrapped around each point.",
    )
    confidence_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Drop masks whose SAM3 score is below this value.",
    )
    keep_top_k: int = Field(
        default=3, ge=1, le=50,
        description="Maximum number of candidate masks to surface.",
    )
    name_prefix: str = Field(
        default="SAM3",
        description="Prefix used to name the resulting MaskLayer(s).",
    )


class SAM3SegmentOutput(BaseModel):
    candidates: int


@register_algorithm
class SAM3SegmentAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.segment"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 Slice Segment"
    description: ClassVar[str] = (
        "Run SAM3 image segmentation on a single 2D slice of the active"
        " seismic volume. Supports free-form text + bounding boxes + points"
        " (points are wrapped into tiny boxes since SAM3 has no native"
        " point-prompt API). Outputs one MaskLayer per surviving candidate."
    )
    input_schema: ClassVar[type[BaseModel]] = SAM3SegmentParams
    output_schema: ClassVar[type[BaseModel]] = SAM3SegmentOutput
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume"}
    runs_in_subprocess: ClassVar[bool] = False
    supports_cancel: ClassVar[bool] = True

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        volume_layer = ctx.input_layers.get("volume")
        if not isinstance(volume_layer, VolumeLayer):
            return AlgorithmResult.failure("Need a VolumeLayer as 'volume' input")

        ai_service = ctx.services.get("ai_service")
        volume_store = ctx.services.get("volume_store")
        if ai_service is None or volume_store is None:
            return AlgorithmResult.failure(
                "Required services missing: 'ai_service' and 'volume_store'."
                " Click 'Start AI' in the AI Dock first."
            )
        if not ai_service.is_ready():
            return AlgorithmResult.failure(
                f"AI service not ready (state={ai_service.state.value})."
            )

        ctx.report_progress(0.05, "Reading slice")
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

        ctx.report_progress(0.2, "Pushing image to SAM3")
        ai_service.mark_busy("SAM3 segmenting")
        try:
            processor = ai_service.image_processor
            processor.set_confidence_threshold(float(ctx.params.confidence_threshold))
            # SAM3 wants a PIL.Image or a tensor; PIL keeps the path simple.
            from PIL import Image  # local import: optional dep boundary

            pil = Image.fromarray(rgb)
            state = processor.set_image(pil)

            if ctx.params.text_prompt.strip():
                ctx.report_progress(0.4, "Applying text prompt")
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

            ctx.report_progress(0.85, "Decoding masks")
            detections = decode_sam3_masks(state)
        finally:
            ai_service.mark_ready()

        detections.sort(key=lambda d: d["score"], reverse=True)
        detections = detections[: int(ctx.params.keep_top_k)]

        output_layers: list[MaskLayer] = []
        for i, det in enumerate(detections, start=1):
            # SAM3 returns a mask in image-pixel shape (rows=H, cols=W) where
            # H matches the slice's "z / sample" axis and W matches the
            # in-slice "inline / xline" axis. The rest of YJ Studio (brush
            # tool, MaskLayer renderer, view_2d_section._mask_rgba) stores
            # mask arrays in axis-1 × axis-2 order — i.e. transposed. So we
            # transpose SAM3 output once at the seam between the AI subsystem
            # and the scene/view layers, which keeps the SAM3 mask aligned
            # with the user's prompt box in every downstream view.
            sam3_mask = np.ascontiguousarray(np.asarray(det["mask"]).T)
            output_layers.append(
                build_mask_layer(
                    sam3_mask,
                    name=f"{ctx.params.name_prefix} #{i} ({det['score']:.2f})",
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

        ctx.report_progress(1.0, "Done")
        summary = (
            f"SAM3: {len(output_layers)} candidate mask(s) on"
            f" {ctx.params.axis}={ctx.params.slice_index}"
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
