"""Export a contiguous range of slices from a VolumeStore to a temp JPEG dir.

SAM3 video predictor expects either an MP4 file or a directory of JPEG
frames (see ``sam3.model.sam3_video_inference.init_state``). For seismic
volumes we always go the JPEG-directory route because:

1. mp4 encoding would re-quantise the slice intensities lossily;
2. JPEG mtime + filename order is what SAM3 uses, so we have direct control
   over which "frame" is which slice.

The exporter writes ``0000.jpg``, ``0001.jpg``, ... so the i-th file maps
to volume axis index ``start + i``. Callers should always pass the returned
``frame_index_map`` along when interpreting SAM3 output.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .volume_to_image import slice_to_rgb_image


@dataclass(slots=True)
class FrameExport:
    """Result of an export run."""

    directory: Path
    frame_index_map: list[int]  # frame_index_map[i] -> source slice index
    width: int
    height: int

    def cleanup(self) -> None:
        if self.directory.exists():
            shutil.rmtree(self.directory, ignore_errors=True)


def export_axis_range_to_jpegs(
    volume_store: Any,
    volume_id: str,
    axis: str,
    indices: list[int],
    *,
    clim: tuple[float, float] | None = None,
    output_dir: Path | None = None,
) -> FrameExport:
    """Materialise ``indices`` along ``axis`` as 0000.jpg, 0001.jpg, ...

    ``volume_store`` only needs a ``get_slice(volume_id, axis, index)``
    method, so tests can drop in a fake. ``output_dir`` defaults to a fresh
    temp directory; callers are responsible for ``FrameExport.cleanup()``.
    """

    from PIL import Image  # local import: optional dep boundary

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="yj_studio_sam3_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    width = height = 0
    for i, source_idx in enumerate(indices):
        raw = volume_store.get_slice(volume_id, axis, int(source_idx))
        if axis in {"inline", "xline"}:
            slice2d = np.asarray(raw, dtype=np.float32).T
        else:
            slice2d = np.asarray(raw, dtype=np.float32).T
        rgb = slice_to_rgb_image(slice2d, clim=clim)
        if i == 0:
            height, width = rgb.shape[:2]
        Image.fromarray(rgb).save(output_dir / f"{i:04d}.jpg", quality=92)
    return FrameExport(
        directory=output_dir,
        frame_index_map=list(int(v) for v in indices),
        width=int(width),
        height=int(height),
    )
