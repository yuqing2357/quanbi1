from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from yj_studio.ai.adapters import stretch_to_uint8
from yj_studio_core.targets import GeoTarget, TargetFrame


@dataclass(frozen=True, slots=True)
class TrackingFrameView:
    frame: TargetFrame
    image: np.ndarray
    mask: np.ndarray


def ordered_target_frames(target: GeoTarget) -> list[TargetFrame]:
    """Return frames from the target's dominant tracking axis by slice index."""

    dominant = _dominant_axis(target)
    return sorted(
        (frame for frame in target.frames.values() if frame.axis == dominant),
        key=lambda item: item.index,
    )


def load_tracking_frames(
    target: GeoTarget,
    target_store: Any,
    volume_store: Any | None,
) -> list[TrackingFrameView]:
    """Fetch source slices and masks using the same image orientation as SAM3."""

    present_frames = ordered_target_frames(target)
    if not present_frames:
        return []
    masks: dict[int, np.ndarray] = {}
    frame_by_index = {int(frame.index): frame for frame in present_frames}
    for frame in present_frames:
        fetched = np.asarray(
            target_store.fetch_mask(
                target.id,
                frame.axis,
                frame.index,
                volume_id=target.volume_id,
            ),
            dtype=bool,
        )
        if fetched.ndim != 2:
            continue
        masks[int(frame.index)] = fetched
    if not masks:
        return []

    dominant = present_frames[0].axis
    indices = set(masks)
    tracking = target.metadata.get("tracking", {})
    if isinstance(tracking, dict):
        last_gap = tracking.get("last_gap", {})
        if isinstance(last_gap, dict):
            missing = last_gap.get("missing", [])
            if isinstance(missing, list):
                indices.update(int(value) for value in missing)

    rows: list[TrackingFrameView] = []
    shape = next(iter(masks.values())).shape
    for index in sorted(indices):
        frame = frame_by_index.get(index) or TargetFrame(
            axis=dominant,
            index=index,
            area_px=0,
            origin="missing",
        )
        mask = masks.get(index)
        if mask is None:
            mask = np.zeros(shape, dtype=bool)
        image = _source_frame_image(
            volume_store,
            target.volume_id,
            _desktop_axis(frame.axis),
            index,
            mask.shape,
        )
        rows.append(TrackingFrameView(frame=frame, image=image, mask=mask))
    return rows


class TargetTrack2DDialog(QDialog):
    """Frame player for inspecting one tracked target through adjacent slices."""

    def __init__(
        self,
        target: GeoTarget,
        frames: list[TrackingFrameView],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{target.id} · 2D 追踪回放")
        self.resize(980, 760)
        self._target = target
        self._frames = frames
        self._crop = _union_crop([row.mask for row in frames])

        layout = QVBoxLayout(self)
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        layout.addWidget(self._canvas, 1)

        controls = QHBoxLayout()
        self._play_button = QPushButton("播放", self)
        self._frame_label = QLabel("", self)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, max(0, len(frames) - 1))
        self._slider.setSingleStep(1)
        controls.addWidget(self._play_button)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._frame_label)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(450)
        self._timer.timeout.connect(self._advance)
        self._play_button.clicked.connect(self._toggle_playback)
        self._slider.valueChanged.connect(self._draw_frame)
        self.finished.connect(lambda _result: self._timer.stop())
        self._draw_frame(0)

    def _toggle_playback(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._play_button.setText("播放")
            return
        if len(self._frames) <= 1:
            return
        self._timer.start()
        self._play_button.setText("暂停")

    def _advance(self) -> None:
        if not self._frames:
            return
        self._slider.setValue((self._slider.value() + 1) % len(self._frames))

    def _draw_frame(self, position: int) -> None:
        self._axes.clear()
        if not self._frames:
            self._axes.text(0.5, 0.5, "没有可显示的追踪帧", ha="center", va="center")
            self._axes.set_axis_off()
            self._canvas.draw_idle()
            return

        row = self._frames[int(np.clip(position, 0, len(self._frames) - 1))]
        self._axes.imshow(row.image, origin="upper", interpolation="nearest")
        overlay = np.zeros((*row.mask.shape, 4), dtype=np.float32)
        overlay[..., :3] = (0.1, 1.0, 0.45)
        overlay[..., 3] = row.mask.astype(np.float32) * 0.48
        self._axes.imshow(overlay, origin="upper", interpolation="nearest")
        if min(row.mask.shape) >= 2 and np.any(row.mask):
            self._axes.contour(
                row.mask.astype(np.uint8),
                levels=[0.5],
                colors=["#00ff72"],
                linewidths=1.5,
            )
        if self._crop is not None:
            x0, x1, y0, y1 = self._crop
            self._axes.set_xlim(x0, x1)
            self._axes.set_ylim(y1, y0)
        axis_label = _desktop_axis(row.frame.axis)
        state_text = "未识别" if row.frame.origin == "missing" else f"area={int(row.mask.sum())} px"
        self._axes.set_title(
            f"{self._target.id}  {axis_label}={row.frame.index}"
            f"  {state_text}"
        )
        self._axes.set_xlabel(_horizontal_label(axis_label))
        self._axes.set_ylabel(_vertical_label(axis_label))
        self._frame_label.setText(f"{position + 1} / {len(self._frames)}")
        self._canvas.draw_idle()


class TargetVolume3DDialog(QDialog):
    """Independent PyVista page for a tracked target reconstructed as a body."""

    def __init__(
        self,
        target: GeoTarget,
        mask: np.ndarray,
        *,
        axis: str,
        index_lo: int,
        voxel_spacing: tuple[float, float, float],
        color: tuple[float, float, float, float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{target.id} · 3D 目标体")
        self.resize(1000, 780)

        from pyvistaqt import QtInteractor

        layout = QVBoxLayout(self)
        summary = QLabel(
            f"{target.id}  类型：{target.type}  "
            f"帧范围：{target.frame_range}  体素：{int(np.count_nonzero(mask)):,}",
            self,
        )
        layout.addWidget(summary)
        self._plotter = QtInteractor(parent=self)
        layout.addWidget(self._plotter, 1)

        vertices, faces = target_surface_geometry(
            mask,
            axis=axis,
            index_lo=index_lo,
            voxel_spacing=voxel_spacing,
        )
        import pyvista as pv

        face_prefix = np.full((faces.shape[0], 1), 3, dtype=np.int64)
        vtk_faces = np.hstack([face_prefix, faces.astype(np.int64, copy=False)]).ravel()
        mesh = pv.PolyData(vertices, vtk_faces)
        self._plotter.set_background("#202124")
        self._plotter.add_mesh(
            mesh,
            color=color[:3],
            opacity=max(0.75, float(color[3])),
            smooth_shading=True,
            show_edges=False,
        )
        self._plotter.add_axes(
            xlabel="Inline",
            ylabel="Xline",
            zlabel="Depth",
        )
        self._plotter.show_grid(
            xtitle="Inline distance (m)",
            ytitle="Xline distance (m)",
            ztitle="Depth (m)",
            color="#bfc5cc",
        )
        self._plotter.view_isometric()
        self._plotter.reset_camera()
        self._plotter.render()
        self.finished.connect(lambda _result: self._plotter.close())


def target_surface_geometry(
    mask: np.ndarray,
    *,
    axis: str,
    index_lo: int,
    voxel_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create a cropped physical-coordinate surface from image-order mask3d."""

    raw = np.asarray(mask, dtype=bool)
    if raw.ndim != 3 or not np.any(raw):
        raise ValueError("3D mask 为空，无法重建目标体。")

    nonzero = np.argwhere(raw)
    lo = np.maximum(nonzero.min(axis=0) - 1, 0)
    hi = np.minimum(nonzero.max(axis=0) + 2, raw.shape)
    cropped = raw[
        lo[0] : hi[0],
        lo[1] : hi[1],
        lo[2] : hi[2],
    ]
    world = mask3d_to_world(cropped, axis)
    origin = mask3d_world_origin(axis, index_lo, tuple(int(v) for v in lo))
    spacing = tuple(float(v) for v in voxel_spacing)

    from skimage.measure import marching_cubes

    padded = np.pad(world.astype(np.float32, copy=False), 1, mode="constant")
    vertices, faces, _normals, _values = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
    )
    vertices -= np.asarray(spacing, dtype=np.float32)
    vertices += np.asarray(origin, dtype=np.float32) * np.asarray(spacing, dtype=np.float32)
    # Geological sample/depth grows downward; use negative display Z so deeper
    # samples appear below shallower samples in the standalone 3D page.
    vertices[:, 2] *= -1.0
    return vertices.astype(np.float32, copy=False), faces.astype(np.int64, copy=False)


def mask3d_to_world(mask: np.ndarray, axis: str) -> np.ndarray:
    """Map (frame, image-row, image-col) into (inline, xline, sample)."""

    arr = np.asarray(mask, dtype=bool)
    axis_key = _desktop_axis(axis)
    if axis_key == "inline":
        return np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))
    if axis_key == "xline":
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
    return np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))


def mask3d_world_origin(
    axis: str,
    index_lo: int,
    raw_crop_origin: tuple[int, int, int],
) -> tuple[int, int, int]:
    frame0, row0, col0 = raw_crop_origin
    axis_key = _desktop_axis(axis)
    if axis_key == "inline":
        return index_lo + frame0, col0, row0
    if axis_key == "xline":
        return col0, index_lo + frame0, row0
    return col0, row0, index_lo + frame0


def _source_frame_image(
    volume_store: Any | None,
    volume_id: str | None,
    axis: str,
    index: int,
    shape: tuple[int, int],
) -> np.ndarray:
    if volume_store is None or not volume_id:
        return np.zeros((*shape, 3), dtype=np.uint8)
    try:
        raw = np.asarray(volume_store.get_slice(volume_id, axis, int(index)), dtype=np.float32).T
    except Exception:  # noqa: BLE001 - mask-only playback is still useful
        return np.zeros((*shape, 3), dtype=np.uint8)
    if raw.shape != shape:
        return np.zeros((*shape, 3), dtype=np.uint8)
    if "lithology" in volume_id.lower():
        rgb = np.zeros((*raw.shape, 3), dtype=np.uint8)
        finite = np.isfinite(raw)
        classes = np.zeros(raw.shape, dtype=np.int16)
        classes[finite] = np.rint(raw[finite]).astype(np.int16, copy=False)
        rgb[finite & (classes == 0)] = (47, 47, 47)
        rgb[finite & (classes >= 1)] = (255, 221, 0)
        return rgb
    gray, _finite = stretch_to_uint8(raw)
    return np.repeat(gray[..., None], 3, axis=2)


def _union_crop(masks: list[np.ndarray]) -> tuple[float, float, float, float] | None:
    points = [np.argwhere(mask) for mask in masks if np.any(mask)]
    if not points:
        return None
    merged = np.vstack(points)
    y0, x0 = merged.min(axis=0)
    y1, x1 = merged.max(axis=0)
    height, width = masks[0].shape
    pad_x = max(12, int((x1 - x0 + 1) * 0.25))
    pad_y = max(12, int((y1 - y0 + 1) * 0.25))
    return (
        float(max(0, x0 - pad_x)),
        float(min(width - 1, x1 + pad_x)),
        float(max(0, y0 - pad_y)),
        float(min(height - 1, y1 + pad_y)),
    )


def _dominant_axis(target: GeoTarget) -> str:
    counts: dict[str, int] = {}
    for frame in target.frames.values():
        counts[frame.axis] = counts.get(frame.axis, 0) + 1
    return max(counts, key=counts.get) if counts else "timeslice"


def _desktop_axis(axis: str) -> str:
    return {"crossline": "xline", "timeslice": "z"}.get(str(axis), str(axis))


def _horizontal_label(axis: str) -> str:
    return {"inline": "Xline", "xline": "Inline", "z": "Inline"}.get(axis, "Trace")


def _vertical_label(axis: str) -> str:
    return "Xline" if axis == "z" else "Sample"
