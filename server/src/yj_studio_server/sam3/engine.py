from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
        video_temporal_disambiguation: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.resolution = int(resolution)
        self.source_root = Path(source_root) if source_root is not None else None
        self.load_video = bool(load_video)
        self.video_temporal_disambiguation = bool(video_temporal_disambiguation)
        self._processor: Any | None = None
        self._video_predictor: Any | None = None
        self._video_load_error: str | None = None
        self._track_state: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None

    @property
    def video_loaded(self) -> bool:
        return self._video_predictor is not None

    @property
    def video_load_error(self) -> str | None:
        return self._video_load_error

    def status_payload(self) -> dict[str, Any]:
        return {
            "image_loaded": self.is_loaded,
            "video_enabled": self.load_video,
            "video_loaded": self.video_loaded,
            "video_error": self._video_load_error,
            "video_temporal_disambiguation": self.video_temporal_disambiguation,
            "device": self.device,
            "resolution": self.resolution,
        }

    def warmup(self) -> dict[str, Any]:
        """Eagerly load the image (+ video) model onto the device.

        Called once at server startup so the first inference request does not
        pay the model-load + device-binding cost. Safe to call repeatedly;
        ``_ensure_processor`` is idempotent.
        """
        self._ensure_processor()
        return self.status_payload()

    def reload_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._processor = None
        self._video_predictor = None
        self._video_load_error = None
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
        auto_stop: bool = False,
        disappear_patience: int = 3,
    ):
        """Multi-object propagation over a pre-rendered JPEG frame sequence.

        ``frames_dir`` must contain ``00000.jpg`` ... in order (the caller
        renders axis slices to JPEG — no Qt/matplotlib needed for seismic
        axial slices). ``seeds`` is a list of one entry per geological target,
        seeded on the ``seed_local`` frame::

            {"obj_id": int,
             "box_xywh": [xmin, ymin, w, h],   # normalised, top-left
             "points": [[x, y], ...],          # optional, normalised [0,1]
             "point_labels": [1, ...],         # optional, 1=fg / 0=bg
             "text": str,                      # only used by the detect path
             "mode": "memory" | "detect"}

        Seeding uses SAM3's **visual-grounding (detector) path**: every seed box
        is added as ONE multi-box prompt on the seed frame, then
        ``propagate_in_video`` runs full propagation forward/backward. SAM3's
        video model is detector-first — its Tracker is only a *refinement* layer
        on top of an existing propagation cache (``cached_frame_outputs``), so a
        bare point/mask seed on a fresh state is rejected with "No cached
        outputs found". Full VG propagation instead carries each prompted object
        across slices through the Tracker's internal memory + keep-alive, which
        is the supported mechanism. ``apply_temporal_disambiguation=False`` (the
        server default) disables the hot-start heuristic that would otherwise
        suppress a prompted object that is not re-detected every frame.

        A single multi-box prompt (not one ``add_prompt`` per box) is required
        because the detector path resets the inference state on every call, so
        sequential prompts would wipe each other — the cause of multi-object
        seeds collapsing to one. The obj_id ↔ target_id mapping the caller
        assigns keeps numbering consistent across frames.

        Yields ``(frame_idx_local: int, {obj_id: mask (H,W) bool})`` for every
        propagated frame, forward then backward from the seed.

        When ``auto_stop`` is set the caller does not know how far the target
        persists, so each direction is propagated over a large budget and cut
        off as soon as *every* tracked object has produced an empty mask for
        ``disappear_patience`` consecutive frames (the target has left the
        volume). The seed frame itself never counts toward the patience.
        """
        predictor = self._ensure_video_predictor()
        with self._autocast_ctx(), self._inference_ctx():
            seed_frame_objects, model_to_seed_id = self._seed_frame_via_detector(
                predictor, seeds, int(seed_local)
            )
            logger.info(
                "track seed frame: %d/%d seed object(s) detected "
                "(seed_ids=%s, model_to_seed=%s, seed_local=%d)",
                len(seed_frame_objects),
                len(seeds),
                sorted(seed_frame_objects),
                model_to_seed_id,
                int(seed_local),
            )
            # Emit the directly-prompted seed frame first so a good seed always
            # yields at least one tracked frame, regardless of propagation.
            if seed_frame_objects:
                yield int(seed_local), dict(seed_frame_objects)
            patience = max(1, int(disappear_patience))
            for reverse, budget in ((False, int(fwd_budget)), (True, int(back_budget))):
                if budget <= 0:
                    continue
                consecutive_empty = 0
                for frame_idx_local, outputs in predictor.propagate_in_video(
                    inference_state=self._track_state,
                    start_frame_idx=int(seed_local),
                    max_frame_num_to_track=budget,
                    reverse=reverse,
                ):
                    model_objects = _extract_objects(outputs)
                    # Only keep objects we can attribute to a requested seed;
                    # the detector path assigns its own ids, mapped on the seed
                    # frame. ``model_to_seed_id`` is stable across propagation
                    # because the Tracker keeps each masklet's id.
                    mapped = {
                        model_to_seed_id[model_id]: mask
                        for model_id, mask in model_objects.items()
                        if model_id in model_to_seed_id
                    }
                    yield int(frame_idx_local), mapped
                    if not auto_stop:
                        continue
                    # Auto range: stop this direction once the target has gone
                    # (all masks empty) for ``patience`` consecutive frames.
                    alive = any(
                        np.asarray(mask, dtype=bool).any() for mask in mapped.values()
                    )
                    consecutive_empty = 0 if alive else consecutive_empty + 1
                    if consecutive_empty >= patience:
                        logger.info(
                            "track auto-stop %s direction at frame_local=%d "
                            "(%d consecutive empty frames, patience=%d)",
                            "reverse" if reverse else "forward",
                            int(frame_idx_local),
                            consecutive_empty,
                            patience,
                        )
                        break

    def _seed_frame_via_detector(
        self,
        predictor: Any,
        seeds: list[dict[str, Any]],
        seed_local: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        """Add every seed box in one VG prompt; map detections back to seed ids.

        Returns ``(seed_frame_objects, model_to_seed_id)`` where
        ``seed_frame_objects`` is ``{seed_id: mask}`` on the seed frame and
        ``model_to_seed_id`` maps the detector's internal object ids to the
        caller's seed ids so propagated frames can be attributed.
        """
        boxes: list[list[float]] = []
        box_seed_ids: list[int] = []
        for seed in seeds:
            box = seed.get("box_xywh")
            if box and len(box) >= 4:
                boxes.append([float(v) for v in box[:4]])
                box_seed_ids.append(int(seed["obj_id"]))
        text = ""
        for seed in seeds:
            if seed.get("text"):
                text = str(seed["text"])
                break

        seeded = predictor.add_prompt(
            inference_state=self._track_state,
            frame_idx=int(seed_local),
            text_str=text or "visual",
            points=None,
            point_labels=None,
            boxes_xywh=boxes or None,
            box_labels=[1] * len(boxes) if boxes else None,
            obj_id=None,
        )
        objects = _extract_objects(_seed_output_dict(seeded))
        model_to_seed_id = _match_models_to_seeds(objects, boxes, box_seed_ids)
        seed_frame_objects = {
            seed_id: objects[model_id]
            for model_id, seed_id in model_to_seed_id.items()
            if model_id in objects
        }
        return seed_frame_objects, model_to_seed_id

    def init_track_state(self, frames_dir: str | Path, *, async_loading: bool = True):
        """Open a video session on a JPEG frame directory. Call before track_video.

        ``async_loading=True`` (the default) loads frames lazily as propagation
        reaches them instead of materialising the whole window up front. Combined
        with the auto-stop break in :meth:`track_video`, frames past the point
        where the target disappears are never decoded into memory at all — this
        is what keeps a local-target track from OOM-ing on a large auto window.
        Any previously open state is released first so video sessions never stack.
        """
        predictor = self._ensure_video_predictor()
        self.reset_track_state()
        try:
            self._track_state = predictor.init_state(
                resource_path=str(frames_dir),
                async_loading_frames=bool(async_loading),
                offload_video_to_cpu=True,
                video_loader_type="jpg",
            )
        except TypeError:
            # Older predictor builds without these kwargs: fall back to eager.
            self._track_state = predictor.init_state(
                resource_path=str(frames_dir),
                video_loader_type="jpg",
            )
        return self._track_state

    def reset_track_state(self) -> None:
        """Drop the current video session and free its GPU/CPU buffers.

        The SAM3 video state holds per-frame image features and the Tracker's
        memory bank; without an explicit release it stays resident until the next
        ``init_state`` and accumulates across repeated track jobs (a slow-burn
        OOM). Call after every track job — see ``collect_object_frames``.
        """
        state = self._track_state
        self._track_state = None
        if state is not None:
            reset = getattr(self._video_predictor, "reset_state", None)
            if callable(reset):
                try:
                    reset(state)
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
        del state
        self._empty_cuda_cache()

    def empty_cache(self) -> None:
        """Public hook so the GPU worker can release cached blocks after a task."""
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        if "cuda" not in self.device:
            return
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache release is best-effort
            pass

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
            try:
                self._load_video_predictor(build_sam3_video_model)
            except Exception as exc:  # noqa: BLE001 - keep image segmentation usable
                self._video_load_error = f"{type(exc).__name__}: {exc}"
        return self._processor

    def _ensure_video_predictor(self):
        self._ensure_processor()
        if self._video_predictor is not None:
            # Idempotent: guarantees the connected-components fallback is active
            # even on a worker warmed before this code was deployed.
            _install_cpu_connected_components_fallback()
            return self._video_predictor
        if not self.load_video:
            raise RuntimeError(
                "SAM3 video predictor is disabled by server config: sam3.load_video=false. "
                "Set sam3.load_video=true and restart the server to enable cross-frame tracking."
            )
        try:
            from sam3.model_builder import build_sam3_video_model

            self._load_video_predictor(build_sam3_video_model)
        except Exception as exc:  # noqa: BLE001 - surface dependency/CUDA detail to the job error
            self._video_load_error = f"{type(exc).__name__}: {exc}"
        if self._video_predictor is None:
            detail = f" Last video load error: {self._video_load_error}" if self._video_load_error else ""
            raise RuntimeError(
                "SAM3 video predictor is not loaded; cross-frame tracking is disabled."
                f"{detail}"
            )
        return self._video_predictor

    def _load_video_predictor(self, builder) -> None:
        self._video_predictor = builder(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            strict_state_dict_loading=False,
            # Natural-video temporal disambiguation uses a 15-frame hot-start
            # window and may suppress visual-prompted objects that are not
            # repeatedly re-detected. Geological tracking commonly uses only
            # 5–11 adjacent slices, so the interactive prompt must be allowed
            # to propagate directly instead of being removed by that heuristic.
            apply_temporal_disambiguation=self.video_temporal_disambiguation,
        )
        _install_cpu_connected_components_fallback()
        self._video_load_error = None


def _install_cpu_connected_components_fallback() -> None:
    """Route SAM3's connected-components through the CPU/skimage backend.

    The Tracker's ``fill_holes_in_mask_scores`` calls
    ``sam3.perflib.connected_components.connected_components``. Without the
    optional ``cc_torch`` extension, that dispatches to a Triton CUDA kernel
    which crashes with ``Triton Error [CUDA]: invalid argument`` on several
    GPU/Triton/driver combinations — surfacing only on the Tracker (memory)
    seeding path. ``cc_torch`` present → the fast native kernel is reliable, so
    we leave it untouched. Otherwise we force the (slightly slower but correct)
    CPU implementation instead of the broken Triton path. ``fill_holes`` runs
    on small object masks, so the CPU cost is negligible.

    Idempotent: ``_get_connected_components_with_padding`` re-imports the symbol
    on every call, so replacing the module attribute is enough.
    """
    try:
        import sam3.perflib.connected_components as cc_mod
    except Exception:  # noqa: BLE001 - perflib is optional; nothing to patch
        return
    if getattr(cc_mod, "_yj_forced_cpu_cc", False):
        return
    if getattr(cc_mod, "HAS_CC_TORCH", False):
        return  # native cc_torch kernel is reliable; keep the fast path

    skimage_impl = getattr(cc_mod, "connected_components_cpu", None)

    def _cpu_connected_components(input_tensor):  # type: ignore[no-untyped-def]
        tensor = input_tensor
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        cv2_result = _connected_components_cv2(tensor)
        if cv2_result is not None:
            return cv2_result
        if skimage_impl is not None:
            return skimage_impl(tensor)
        raise RuntimeError(
            "SAM3 connected-components fallback failed: neither cv2 nor "
            "scikit-image is available, and the Triton kernel is broken on this GPU."
        )

    cc_mod.connected_components = _cpu_connected_components
    cc_mod._yj_forced_cpu_cc = True
    logger.warning(
        "SAM3: cc_torch not installed; forcing CPU connected-components to avoid "
        "the Triton 'invalid argument' kernel crash on the Tracker seeding path."
    )


def _connected_components_cv2(tensor):
    """OpenCV connected-components matching SAM3's ``(labels, counts)`` contract.

    Input is a ``(B,1,H,W)`` mask tensor; returns ``(labels, counts)`` of the
    same shape on the same device, where ``counts`` holds each pixel's component
    size (background == 0). Returns ``None`` if cv2/torch are unavailable so the
    caller can try the skimage backend instead.
    """
    try:
        import cv2
        import numpy as _np
        import torch
    except Exception:  # noqa: BLE001 - let the caller fall through to skimage
        return None
    binary = (tensor != 0).to("cpu", torch.uint8).numpy()  # (B,1,H,W)
    batch, _ch, height, width = binary.shape
    labels_out = _np.zeros((batch, height, width), dtype=_np.int32)
    counts_out = _np.zeros((batch, height, width), dtype=_np.int32)
    for b in range(batch):
        num, labelled, stats, _centroids = cv2.connectedComponentsWithStats(
            binary[b, 0], connectivity=8
        )
        labels_out[b] = labelled
        if num > 0:
            areas = stats[:, cv2.CC_STAT_AREA].astype(_np.int32)
            # Background is label 0; SAM3 expects its per-pixel count to be 0.
            areas[0] = 0
            counts_out[b] = areas[labelled]
    out_shape = tensor.shape
    labels_t = torch.from_numpy(labels_out).to(tensor.device).view(out_shape)
    counts_t = torch.from_numpy(counts_out).to(tensor.device).view(out_shape)
    return labels_t, counts_t


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


def _seed_output_dict(seeded: Any) -> dict[str, Any]:
    """Pull the output dict out of a SAM3 ``add_prompt`` return value.

    ``add_prompt`` returns ``(frame_idx, output_dict)`` on the detector path
    (and may return longer tuples on other paths); normalise to the dict.
    """
    if isinstance(seeded, dict):
        return seeded
    if isinstance(seeded, tuple):
        found = next((item for item in seeded if isinstance(item, dict)), None)
        if found is not None:
            return found
    return {}


def _match_models_to_seeds(
    objects: dict[int, np.ndarray],
    boxes: list[list[float]],
    seed_ids: list[int],
) -> dict[int, int]:
    """Attribute detector object ids to the caller's seed ids.

    ``objects`` is ``{model_id: mask}`` from the seed frame. ``boxes`` are the
    normalised top-left ``[x,y,w,h]`` seed boxes, parallel to ``seed_ids``. Each
    seed is greedily matched to the unused detection whose mask bounding box has
    the highest IoU with the seed box; seeds/detections that cannot be matched
    by overlap fall back to descending-area order. Returns ``{model_id: seed_id}``.
    """
    if not objects or not seed_ids:
        return {}
    model_items = sorted(
        objects.items(),
        key=lambda item: int(np.asarray(item[1], dtype=bool).sum()),
        reverse=True,
    )
    model_ids = [mid for mid, _ in model_items]
    model_boxes = {mid: _mask_norm_box(mask) for mid, mask in objects.items()}

    mapping: dict[int, int] = {}
    used_models: set[int] = set()

    # Pass 1: best-IoU greedy matching when seed boxes are available.
    for seed_box, seed_id in zip(boxes, seed_ids):
        best_mid: int | None = None
        best_iou = 0.0
        for mid in model_ids:
            if mid in used_models:
                continue
            iou = _iou_xywh(seed_box, model_boxes[mid])
            if iou > best_iou:
                best_iou, best_mid = iou, mid
        if best_mid is not None and best_iou > 0.0:
            mapping[best_mid] = seed_id
            used_models.add(best_mid)

    # Pass 2: assign any still-unmatched seeds to the largest free detection
    # (covers text-only seeds with no box, or detections that don't overlap).
    matched_seed_ids = set(mapping.values())
    free_models = [mid for mid in model_ids if mid not in used_models]
    for seed_id in seed_ids:
        if seed_id in matched_seed_ids:
            continue
        if not free_models:
            break
        mid = free_models.pop(0)
        mapping[mid] = seed_id
        used_models.add(mid)
        matched_seed_ids.add(seed_id)

    return mapping


def _mask_norm_box(mask: np.ndarray) -> list[float]:
    """Normalised top-left ``[x,y,w,h]`` bounding box of a boolean mask."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return [0.0, 0.0, 0.0, 0.0]
    height, width = arr.shape
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [
        float(x0) / width,
        float(y0) / height,
        float(x1 - x0 + 1) / width,
        float(y1 - y0 + 1) / height,
    ]


def _iou_xywh(a: list[float], b: list[float]) -> float:
    """IoU of two normalised top-left ``[x,y,w,h]`` boxes."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax0, ay0, aw, ah = (float(v) for v in a[:4])
    bx0, by0, bw, bh = (float(v) for v in b[:4])
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


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
        try:
            return np.asarray(value.numpy())
        except TypeError:
            # numpy has no equivalent for some torch dtypes (e.g. bfloat16,
            # float8). Upcast to float32 before converting.
            if hasattr(value, "float"):
                return np.asarray(value.float().numpy())
            raise
    return np.asarray(value)
