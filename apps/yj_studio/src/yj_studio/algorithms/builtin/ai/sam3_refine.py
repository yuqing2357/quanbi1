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
            "If non-empty, overrides the text prompt stored on the edited"
            " mask's provenance. Empty re-uses the original prompt."
        ),
    )
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    keep_top_k: int = Field(default=3, ge=1, le=50)
    pad_box_px: float = Field(
        default=4.0,
        ge=0.0,
        description="Inflate the mask bbox by this many pixels on each side.",
    )


class SAM3RefineOutput(BaseModel):
    candidates: int


@register_algorithm
class SAM3RefineAlgorithm(Algorithm):
    id: ClassVar[str] = "ai.sam3.refine"
    category: ClassVar[str] = "ai"
    label: ClassVar[str] = "SAM3 Refine From Mask"
    description: ClassVar[str] = (
        "Take an edited MaskLayer, derive its bounding box, and re-segment"
        " the same slice with SAM3 using that box as a positive prompt. The"
        " original text prompt (if any) is recovered from the mask's"
        " provenance and reused unless overridden."
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
            return AlgorithmResult.failure("Need a VolumeLayer as 'volume' input")
        if not isinstance(edited, MaskLayer) or edited.mask is None:
            return AlgorithmResult.failure("Need an edited 2D MaskLayer as 'edited_mask'")
        if edited.axis not in {"inline", "xline", "z"}:
            return AlgorithmResult.failure(
                f"Edited mask axis must be inline/xline/z, got {edited.axis!r}"
            )
        if edited.slice_index is None:
            return AlgorithmResult.failure("Edited mask has no slice_index")
        if edited.mask.ndim != 2:
            return AlgorithmResult.failure(
                "Refine only supports 2D masks (use the propagate algorithm for 3D)"
            )

        ai_service = ctx.services.get("ai_service")
        volume_store = ctx.services.get("volume_store")
        if ai_service is None or volume_store is None:
            return AlgorithmResult.failure(
                "Required services missing. Start AI in the AI Dock first."
            )
        if not ai_service.is_ready():
            return AlgorithmResult.failure(
                f"AI service not ready (state={ai_service.state.value})."
            )

        mask_arr = np.asarray(edited.mask, dtype=bool)
        if not mask_arr.any():
            return AlgorithmResult.failure("Edited mask is empty")

        ctx.report_progress(0.05, "Reading slice")
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

        ai_service.mark_busy("SAM3 refining")
        try:
            from PIL import Image

            processor = ai_service.image_processor
            processor.set_confidence_threshold(float(ctx.params.confidence_threshold))
            pil = Image.fromarray(rgb)
            state = processor.set_image(pil)

            ctx.report_progress(0.4, "Applying prompts")
            if text_prompt:
                state = processor.set_text_prompt(prompt=text_prompt, state=state)
            ctx.check_cancel()
            state = _apply_box_prompt(processor, state, x0, y0, x1, y1, width, height)
            ctx.check_cancel()

            ctx.report_progress(0.8, "Decoding masks")
            detections = decode_sam3_masks(state)
        finally:
            ai_service.mark_ready()

        detections.sort(key=lambda d: d["score"], reverse=True)
        detections = detections[: int(ctx.params.keep_top_k)]
        output_layers: list[MaskLayer] = []
        for i, det in enumerate(detections, start=1):
            # See sam3_segment.py for why the SAM3 mask is transposed at the
            # AI → scene seam: brush + MaskLayer renderer + view_2d_section
            # all assume axis-1 × axis-2 mask layout, but SAM3 returns
            # row-major image pixels (H × W).
            sam3_mask = np.ascontiguousarray(np.asarray(det["mask"]).T)
            output_layers.append(
                build_mask_layer(
                    sam3_mask,
                    name=f"{edited.name} (refined #{i}, {det['score']:.2f})",
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

        ctx.report_progress(1.0, "Done")
        return AlgorithmResult.success(
            output_layers=output_layers,
            summary=f"Refined → {len(output_layers)} candidate(s) on {edited.axis}={edited.slice_index}",
        )
