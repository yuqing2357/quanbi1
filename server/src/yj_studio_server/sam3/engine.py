from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


class SAM3Engine:
    """Lazy SAM3 image-model holder for the remote server process."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        resolution: int = 1008,
        source_root: str | Path | None = None,
        load_video: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.resolution = int(resolution)
        self.source_root = Path(source_root) if source_root is not None else None
        self.load_video = bool(load_video)
        self._processor: Any | None = None
        self._video_predictor: Any | None = None
        self._track_state: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None

    def reload_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._processor = None
        self._video_predictor = None
        self._track_state = None

    def segment(
        self,
        rgb: np.ndarray,
        *,
        text: str = "",
        boxes: list[list[float]] | None = None,
        points: list[list[float]] | None = None,
        point_box_radius_px: float = 8.0,
        confidence: float = 0.4,
    ) -> list[dict[str, Any]]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"SAM3 image must be HxWx3 RGB, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)
        height, width = image.shape[:2]

        processor = self._ensure_processor()
        processor.set_confidence_threshold(float(confidence))

        from PIL import Image

        state = processor.set_image(Image.fromarray(image))
        prompt_text = text.strip()
        if prompt_text:
            state = processor.set_text_prompt(prompt=prompt_text, state=state)

        for box in boxes or []:
            state = _apply_box_prompt(processor, state, box, width, height)

        radius = float(point_box_radius_px)
        for point in points or []:
            if len(point) < 2:
                continue
            px, py = float(point[0]), float(point[1])
            state = _apply_box_prompt(
                processor,
                state,
                [px - radius, py - radius, px + radius, py + radius],
                width,
                height,
            )

        return decode_sam3_masks(state)

    def track_video(
        self,
        frames_dir: str | Path,
        *,
        seeds: list[dict[str, Any]],
        seed_local: int,
        fwd_budget: int,
        back_budget: int,
    ):
        """Multi-object propagation over a pre-rendered JPEG frame sequence.

        ``frames_dir`` must contain ``00000.jpg`` ... in order (the caller
        renders axis slices to JPEG — no Qt/matplotlib needed for seismic
        axial slices). ``seeds`` is a list of
        ``{"obj_id": int, "box_xywh": [cx,cy,w,h], "text": str}`` — one entry
        per geological target, seeded on the ``seed_local`` frame. The obj_id ↔
        target_id mapping the caller assigns is what keeps numbering consistent
        across frames.

        Yields ``(frame_idx_local: int, {obj_id: mask (H,W) bool})`` for every
        propagated frame, forward then backward from the seed.
        """
        self._ensure_processor()
        predictor = self._video_predictor
        if predictor is None:
            raise RuntimeError(
                "SAM3 video predictor not loaded (load_video=False, or triton/CUDA unavailable);"
                " cross-frame tracking is disabled."
            )
        with self._autocast_ctx(), self._inference_ctx():
            for seed in seeds:
                predictor.add_prompt(
                    inference_state=self._track_state,
                    frame_idx=int(seed_local),
                    text_str=str(seed.get("text") or "visual"),
                    points=None,
                    point_labels=None,
                    boxes_xywh=[list(seed["box_xywh"])],
                    box_labels=[1],
                    obj_id=int(seed["obj_id"]),
                )
            for reverse, budget in ((False, int(fwd_budget)), (True, int(back_budget))):
                if budget <= 0:
                    continue
                for frame_idx_local, outputs in predictor.propagate_in_video(
                    inference_state=self._track_state,
                    start_frame_idx=int(seed_local),
                    max_frame_num_to_track=budget,
                    reverse=reverse,
                ):
                    yield int(frame_idx_local), _extract_objects(outputs)

    def init_track_state(self, frames_dir: str | Path):
        """Open a video session on a JPEG frame directory. Call before track_video."""
        self._ensure_processor()
        predictor = self._video_predictor
        if predictor is None:
            raise RuntimeError(
                "SAM3 video predictor not loaded; cross-frame tracking is disabled."
            )
        self._track_state = predictor.init_state(
            resource_path=str(frames_dir),
            async_loading_frames=False,
            video_loader_type="jpg",
        )
        return self._track_state

    def _autocast_ctx(self):
        # SAM3's add_prompt / propagate_in_video are NOT internally autocast
        # decorated (unlike its image entry points), so callers must provide
        # the bf16 context on CUDA — matching the official demo + the desktop
        # workbench. On CPU / when torch is unavailable (tests) use a no-op.
        from contextlib import nullcontext

        if "cuda" not in self.device:
            return nullcontext()
        try:
            import torch

            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        except Exception:  # noqa: BLE001 - diagnostics-only fallback
            return nullcontext()

    def _inference_ctx(self):
        from contextlib import nullcontext

        try:
            import torch

            return torch.inference_mode()
        except Exception:  # noqa: BLE001
            return nullcontext()

    def _ensure_processor(self):
        if self._processor is not None:
            return self._processor
        if self.source_root is not None:
            source_text = str(self.source_root)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint_path}")

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_model

        image_model = build_sam3_image_model(
            device=self.device,
            checkpoint_path=str(self.checkpoint_path),
        )
        self._processor = Sam3Processor(
            image_model,
            resolution=self.resolution,
            device=self.device,
        )
        if self.load_video:
            self._video_predictor = build_sam3_video_model(
                checkpoint_path=str(self.checkpoint_path),
                device=self.device,
                strict_state_dict_loading=False,
            )
        return self._processor


def decode_sam3_masks(state: dict[str, Any]) -> list[dict[str, Any]]:
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
                "box": [float(v) for v in boxes_np[i]],
            }
        )
    return detections


def _extract_objects(outputs: Any) -> dict[int, np.ndarray]:
    """Pull every object's binary mask out of one SAM3 video frame output.

    SAM3 yields ``{"out_obj_ids": [...], "out_binary_masks": (n_obj, H, W) bool, ...}``.
    Returns ``{obj_id: mask (H,W) bool}`` for every object present in the frame.
    """
    if not isinstance(outputs, dict):
        return {}
    masks = outputs.get("out_binary_masks")
    obj_ids = outputs.get("out_obj_ids")
    if masks is None or obj_ids is None or len(masks) == 0:
        return {}
    ids_list = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
    result: dict[int, np.ndarray] = {}
    for i, oid in enumerate(ids_list):
        mask = masks[i]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 3:
            mask = mask[0]
        result[int(oid)] = mask
    return result


def _apply_box_prompt(processor, state, box: list[float], width: int, height: int):
    if len(box) < 4:
        return state
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    x0 = max(0.0, min(x0, width - 1.0))
    x1 = max(0.0, min(x1, width - 1.0))
    y0 = max(0.0, min(y0, height - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    if x1 <= x0 or y1 <= y0:
        return state
    cx = (x0 + x1) / 2.0 / float(width)
    cy = (y0 + y1) / 2.0 / float(height)
    bw = (x1 - x0) / float(width)
    bh = (y1 - y0) / float(height)
    return processor.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)
