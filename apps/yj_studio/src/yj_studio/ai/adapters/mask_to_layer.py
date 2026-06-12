"""Convert SAM3 inference output (boxes / masks / scores tensors) into one
or more ``MaskLayer`` instances we can hand back to ``LayerStore``.

The SAM3 image processor stores results inside its ``state`` dict (see
``Sam3Processor._forward_grounding``):

    state["masks"]   : (N, 1, H, W) bool tensor
    state["scores"]  : (N,)        float tensor
    state["boxes"]   : (N, 4)      xyxy in original image pixels

We accept anything that quacks like that - tensors are pulled to CPU numpy
in :func:`decode_sam3_masks` so the rest of the pipeline stays Qt/numpy.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from yj_studio.scene.layers import MaskLayer
from yj_studio.targets import mask_summary, target_type_color


def decode_sam3_masks(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a SAM3 inference state into a CPU-numpy list of detections.

    Each entry has ``mask`` (H, W bool), ``score`` (float), ``box``
    (x0, y0, x1, y1 float). Empty list when SAM3 returns no detections
    above the confidence threshold.
    """

    masks = state.get("masks")
    scores = state.get("scores")
    boxes = state.get("boxes")
    if masks is None or scores is None or boxes is None:
        return []

    masks_np = _to_numpy(masks)
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    scores_np = _to_numpy(scores).reshape(-1)
    boxes_np = _to_numpy(boxes).reshape(-1, 4)

    detections: list[dict[str, Any]] = []
    count = min(masks_np.shape[0], scores_np.shape[0], boxes_np.shape[0])
    for i in range(count):
        detections.append(
            {
                "mask": np.asarray(masks_np[i], dtype=bool),
                "score": float(scores_np[i]),
                "box": tuple(float(v) for v in boxes_np[i]),
            }
        )
    return detections


def sam3_mask_to_layer(mask: np.ndarray) -> np.ndarray:
    """Convert a SAM3/server mask into desktop ``MaskLayer`` orientation.

    SAM3 (and the server's stored masks) are in **image order**: rows are
    samples/depth, columns are trace (inline/xline). The desktop scene/view
    stack — ``MaskLayer``, the brush tool, ``view_2d_section._mask_rgba`` —
    stores masks transposed (axis1 × axis2, i.e. trace × samples). This is the
    **single, canonical place** desktop code flips that orientation; every
    consumer (single-slice algorithm, remote task, target dock) calls this
    instead of hand-writing ``.T``. Do NOT add another transpose downstream.

    See docs/project_review_and_remediation.md §1.2 for the convention and the
    golden round-trip test that guards it.
    """
    return np.ascontiguousarray(np.asarray(mask, dtype=bool).T)


def build_mask_layer(
    mask: np.ndarray,
    *,
    name: str,
    axis: str,
    slice_index: int,
    score: float | None = None,
    color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.55),
    metadata: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> MaskLayer:
    """Wrap a 2D boolean mask as a ``MaskLayer`` with provenance attached."""

    mask_arr = np.asarray(mask)
    if mask_arr.dtype != bool:
        mask_arr = mask_arr.astype(bool)
    meta = dict(metadata or {})
    meta.setdefault("axis", axis)
    meta.setdefault("slice_index", int(slice_index))
    summary = mask_summary(mask_arr)
    for key, value in summary.items():
        if value is not None:
            meta.setdefault(key, value)
    if score is not None:
        score_value = round(float(score), 6)
        meta.setdefault("score", score_value)
    if meta.get("target_type") and color == (1.0, 0.2, 0.2, 0.55):
        color = target_type_color(str(meta.get("target_type")), alpha=color[3])
    prov = dict(provenance or {})
    prov.setdefault("source", "ai.sam3")
    return MaskLayer(
        name=name,
        mask=mask_arr,
        axis=axis,
        slice_index=int(slice_index),
        confidence=score_value if score is not None else None,
        color=color,
        opacity=color[3],
        metadata=meta,
        provenance=prov,
    )


def build_layers_from_state(
    state: dict[str, Any],
    *,
    name_prefix: str,
    axis: str,
    slice_index: int,
    keep_top_k: int | None = None,
) -> list[MaskLayer]:
    """Build one MaskLayer per SAM3 detection in ``state``.

    Detections are sorted by descending confidence; ``keep_top_k`` caps how
    many candidates surface (None = keep all).
    """

    detections = sorted(
        decode_sam3_masks(state), key=lambda d: d["score"], reverse=True
    )
    if keep_top_k is not None:
        detections = detections[: int(keep_top_k)]
    layers: list[MaskLayer] = []
    for i, det in enumerate(detections, start=1):
        layers.append(
            build_mask_layer(
                det["mask"],
                name=f"{name_prefix} #{i} ({det['score']:.2f})",
                axis=axis,
                slice_index=slice_index,
                score=det["score"],
                metadata={"box": list(det["box"])},
            )
        )
    return layers


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or numpy array to CPU numpy without importing torch."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def iter_layers(payload: Iterable[MaskLayer]) -> Iterable[MaskLayer]:
    """Tiny pass-through so callers can ``yield from`` results cleanly."""

    yield from payload
