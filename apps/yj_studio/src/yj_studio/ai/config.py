r"""Paths and runtime tunables for SAM3.

A small dataclass that the UI fills in from settings.json (or from the
default constants below) and hands to ``AIService``. Kept separate so tests
and the worker subprocess can both construct one without touching Qt.

The defaults resolve to the vendored copies inside this project so a fresh
clone of ``f:\圈闭软件\`` is self-contained once the user has populated
``weights/sam3.pt`` and ``libs/sam3/`` per their READMEs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Project root is five levels up from this file. Counting from this file:
#   parents[0] = ai/
#   parents[1] = yj_studio/         (inner package dir)
#   parents[2] = src/
#   parents[3] = yj_studio/         (outer app dir)
#   parents[4] = apps/
#   parents[5] = 圈闭软件/           ← project root
_PROJECT_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_SAM3_CHECKPOINT = _PROJECT_ROOT / "weights" / "sam3.pt"
"""Default SAM3 unified checkpoint. The same file contains the image detector
and the video tracker weights (see model_builder._load_checkpoint)."""

DEFAULT_SAM3_SOURCE_ROOT = _PROJECT_ROOT / "libs"
"""Directory that *contains* the ``sam3`` Python package. Prepended to
``sys.path`` by ``AIService._LoaderWorker`` before importing sam3."""


@dataclass(slots=True)
class SAM3Config:
    """Frozen-ish runtime knobs for the SAM3 stack.

    Defaults match what ``build_sam3_image_model`` ships out of the box. The
    constructor lets the AI Dock override the most common knobs (weights
    path, device, confidence threshold) from settings.json.
    """

    checkpoint_path: Path = DEFAULT_SAM3_CHECKPOINT
    device: str = "cuda"  # falls back to "cpu" inside the loader if no GPU
    resolution: int = 1008
    confidence_threshold: float = 0.5
    # Whether to also build the video tracker (used for cross-slice
    # propagation). Skipping it roughly halves GPU memory.
    #
    # Default OFF on Windows: the video tracker pulls in ``triton`` for one
    # of its mask ops (sam3.model.edt) and ``triton`` does not have an
    # official Windows wheel. Single-slice segmentation (SAM3SegmentAlgorithm
    # + SAM3RefineAlgorithm) does not need triton, so we keep it usable by
    # default and let users opt back in once triton is sorted out.
    load_video_model: bool = False
    # SAM3 source tree must be importable. When the source lives outside
    # site-packages we prepend this path to sys.path before the import.
    sam3_source_root: Path | None = DEFAULT_SAM3_SOURCE_ROOT

    def checkpoint_exists(self) -> bool:
        return self.checkpoint_path.exists()
