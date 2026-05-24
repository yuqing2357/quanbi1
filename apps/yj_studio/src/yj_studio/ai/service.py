"""Qt facade that owns the loaded SAM3 models.

Loading SAM3 takes ~30s and ~1.5 GB GPU. We do it lazily on user request
(the AI Dock's "Start AI" button) and run the actual ``torch.load`` /
``build_sam3_image_model`` on a worker thread so the UI stays responsive.

The service exposes a tiny state machine via ``state_changed(AIServiceState,
message)``. Callers (algorithms, dock) check ``service.is_ready()`` before
talking to the model, and ``service.image_processor`` /
``service.video_predictor`` give them the loaded SAM3 objects.

We deliberately keep the SAM3 import inside the worker thread (``_load``):
this way, ``import yj_studio.ai`` stays cheap and does NOT require torch /
sam3 / a GPU. Tests can build an ``AIService`` and inspect state without
ever touching the model stack.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .config import SAM3Config

logger = logging.getLogger(__name__)


class AIServiceState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass(slots=True)
class _LoadedModels:
    image_model: Any
    image_processor: Any
    video_predictor: Any | None = None


class _LoaderWorker(QObject):
    """QObject that calls into SAM3 builders on a worker thread."""

    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)  # _LoadedModels
    failed = pyqtSignal(str)

    def __init__(self, config: SAM3Config) -> None:
        super().__init__()
        self._config = config

    def run(self) -> None:
        try:
            self.progress.emit("Importing SAM3 module")
            if self._config.sam3_source_root is not None:
                root = str(self._config.sam3_source_root)
                if root not in sys.path:
                    sys.path.insert(0, root)
            t0 = time.time()
            from sam3.model_builder import (
                build_sam3_image_model,
                build_sam3_video_model,
            )
            from sam3.model.sam3_image_processor import Sam3Processor
            logger.info("SAM3 import took %.2fs", time.time() - t0)

            if not self._config.checkpoint_exists():
                self.failed.emit(
                    f"SAM3 checkpoint not found: {self._config.checkpoint_path}"
                )
                return

            self.progress.emit("Loading SAM3 image model (~30s)")
            t0 = time.time()
            image_model = build_sam3_image_model(
                device=self._config.device,
                checkpoint_path=str(self._config.checkpoint_path),
            )
            logger.info("SAM3 image model loaded in %.2fs", time.time() - t0)
            processor = Sam3Processor(
                image_model,
                resolution=self._config.resolution,
                device=self._config.device,
                confidence_threshold=self._config.confidence_threshold,
            )

            video_predictor = None
            if self._config.load_video_model:
                self.progress.emit("Loading SAM3 video model (~30s)")
                t0 = time.time()
                try:
                    video_predictor = build_sam3_video_model(
                        checkpoint_path=str(self._config.checkpoint_path),
                        device=self._config.device,
                        strict_state_dict_loading=False,
                    )
                    logger.info(
                        "SAM3 video model loaded in %.2fs", time.time() - t0
                    )
                except Exception as exc:  # noqa: BLE001 — keep image model usable
                    # The video tracker depends on triton, which has no
                    # official Windows wheel. Don't let that drag the whole
                    # AI service down — single-slice segmentation still
                    # works without the video predictor.
                    logger.warning(
                        "SAM3 video model unavailable (%s); cross-slice"
                        " propagation will be disabled.",
                        exc,
                    )
                    self.progress.emit(
                        "SAM3 video model unavailable (image still works)"
                    )

            self.succeeded.emit(
                _LoadedModels(
                    image_model=image_model,
                    image_processor=processor,
                    video_predictor=video_predictor,
                )
            )
        except Exception as exc:  # noqa: BLE001 — worker boundary
            logger.exception("SAM3 loading failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class AIService(QObject):
    """Lazy holder for the SAM3 image processor + video predictor.

    Usage:

        service = AIService(SAM3Config())
        service.state_changed.connect(...)
        service.start()  # kicks off background load
        # later:
        if service.is_ready():
            state = service.image_processor.set_image(img)
            ...
    """

    state_changed = pyqtSignal(AIServiceState, str)
    # Emitted by AI prompt tools (AIBoxPromptTool / AIPointPromptTool); the AI
    # Dock listens and appends each entry to its current prompt collection.
    # Tuples are (axis, slice_index, x_min, y_min, x_max, y_max) for boxes
    # and (axis, slice_index, x, y) for points.
    box_prompt_added = pyqtSignal(str, int, float, float, float, float)
    point_prompt_added = pyqtSignal(str, int, float, float)

    def __init__(self, config: SAM3Config, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._state = AIServiceState.IDLE
        self._models: _LoadedModels | None = None
        self._thread: QThread | None = None
        self._worker: _LoaderWorker | None = None
        self._last_message = ""

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> AIServiceState:
        return self._state

    @property
    def message(self) -> str:
        return self._last_message

    def is_ready(self) -> bool:
        return self._state == AIServiceState.READY and self._models is not None

    @property
    def image_processor(self) -> Any | None:
        return self._models.image_processor if self._models else None

    @property
    def video_predictor(self) -> Any | None:
        return self._models.video_predictor if self._models else None

    @property
    def config(self) -> SAM3Config:
        return self._config

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Begin loading the model on a worker thread. No-op if already loading
        or ready."""

        if self._state in {AIServiceState.LOADING, AIServiceState.READY}:
            return
        self._set_state(AIServiceState.LOADING, "Starting SAM3 loader")
        self._thread = QThread(self)
        self._worker = _LoaderWorker(self._config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def shutdown(self) -> None:
        """Drop references to the model so torch can free GPU memory."""

        self._models = None
        self._set_state(AIServiceState.IDLE, "SAM3 unloaded")

    # ------------------------------------------------------------------ slots

    def _on_progress(self, message: str) -> None:
        self._set_state(AIServiceState.LOADING, message)

    def _on_succeeded(self, models: _LoadedModels) -> None:
        self._models = models
        self._cleanup_thread()
        self._set_state(AIServiceState.READY, "SAM3 ready")

    def _on_failed(self, message: str) -> None:
        self._cleanup_thread()
        self._models = None
        self._set_state(AIServiceState.ERROR, message)

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._worker = None

    def _set_state(self, new_state: AIServiceState, message: str) -> None:
        self._state = new_state
        self._last_message = message
        self.state_changed.emit(new_state, message)

    # ------------------------------------------------------------------ busy helpers

    def mark_busy(self, message: str = "Running") -> None:
        if self._state == AIServiceState.READY:
            self._set_state(AIServiceState.BUSY, message)

    def mark_ready(self, message: str = "SAM3 ready") -> None:
        if self._state == AIServiceState.BUSY:
            self._set_state(AIServiceState.READY, message)

    # ------------------------------------------------------------------ prompt forwarding

    def emit_box_prompt(
        self,
        axis: str,
        slice_index: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        self.box_prompt_added.emit(
            axis, int(slice_index), float(x_min), float(y_min), float(x_max), float(y_max)
        )

    def emit_point_prompt(self, axis: str, slice_index: int, x: float, y: float) -> None:
        self.point_prompt_added.emit(axis, int(slice_index), float(x), float(y))
