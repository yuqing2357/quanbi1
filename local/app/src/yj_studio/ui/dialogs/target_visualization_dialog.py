from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yj_studio.ai.adapters import stretch_to_uint8
from yj_studio.view.rgt_compose import compose_rgt_rgb_from_meta, is_rgt_composite_meta
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
    *,
    stage: str | None = None,
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
                stage=stage,
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


def _stamp_brush(mask: np.ndarray, cx: int, cy: int, radius: int, value: bool) -> None:
    """Paint a filled disk of ``value`` into ``mask`` centred at ``(cx, cy)``."""
    height, width = mask.shape
    r = max(1, int(radius))
    y0, y1 = max(0, cy - r), min(height, cy + r + 1)
    x0, x1 = max(0, cx - r), min(width, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    mask[y0:y1, x0:x1][disk] = value


class TargetTrack2DDialog(QDialog):
    """Frame player + per-frame manual correction for one tracked target.

    When constructed with a ``target_store`` it gains an edit mode: a brush to
    add/erase mask pixels on the current slice (saved back via ``put_mask``) and
    a "删除此帧" action to drop a bad frame (``delete_frame``). Without a store it
    is a read-only player (back-compat).
    """

    def __init__(
        self,
        target: GeoTarget,
        frames: list[TrackingFrameView],
        parent: QWidget | None = None,
        *,
        target_store: Any | None = None,
        stage: str | None = None,
        on_changed: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{target.id} · 2D 追踪回放")
        self.resize(980, 760)
        self._target = target
        self._frames = frames
        self._crop = _union_crop([row.mask for row in frames])
        # Inset (zoom) axes, rebuilt each draw. The main axes shows the full
        # slice for global context; the inset magnifies the mask neighbourhood.
        self._inset: Any | None = None
        self._target_store = target_store
        self._stage = stage
        self._on_changed = on_changed
        self._editable = target_store is not None
        self._edit_mode = False
        self._brush_add = True
        self._painting = False
        # Editable mask overrides keyed by frame position (lazily copied so the
        # original fetched arrays are never mutated until saved).
        self._edited: dict[int, np.ndarray] = {}

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

        if self._editable:
            edit_row = QHBoxLayout()
            self._edit_check = QCheckBox("编辑掩膜", self)
            self._edit_check.setToolTip("勾选后用鼠标在切片上拖动来修改 mask。")
            # Explicit, mutually-exclusive add/erase mode so erasing is obvious.
            self._add_radio = QRadioButton("添加", self)
            self._erase_radio = QRadioButton("擦除", self)
            self._add_radio.setToolTip("在掩膜上补充漏掉的区域。")
            self._erase_radio.setToolTip("从掩膜上擦除多分割的区域。")
            self._add_radio.setChecked(True)
            self._brush_group = QButtonGroup(self)
            self._brush_group.addButton(self._add_radio)
            self._brush_group.addButton(self._erase_radio)
            self._brush_spin = QSpinBox(self)
            self._brush_spin.setRange(1, 80)
            self._brush_spin.setValue(8)
            self._brush_spin.setPrefix("笔刷 ")
            self._brush_spin.setSuffix(" px")
            self._save_button = QPushButton("保存修正", self)
            self._revert_button = QPushButton("撤销修正", self)
            self._delete_frame_button = QPushButton("删除此帧", self)
            self._edit_status = QLabel("", self)
            self._edit_check.toggled.connect(self._toggle_edit_mode)
            self._add_radio.toggled.connect(lambda checked: setattr(self, "_brush_add", bool(checked)))
            self._save_button.clicked.connect(self._save_current_frame)
            self._revert_button.clicked.connect(self._revert_current_frame)
            self._delete_frame_button.clicked.connect(self._delete_current_frame)
            for widget in (
                self._edit_check,
                self._add_radio,
                self._erase_radio,
                self._brush_spin,
                self._save_button,
                self._revert_button,
                self._delete_frame_button,
            ):
                edit_row.addWidget(widget)
            edit_row.addWidget(self._edit_status, 1)
            layout.addLayout(edit_row)
            self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
            self._update_edit_controls()

        self._timer = QTimer(self)
        self._timer.setInterval(450)
        self._timer.timeout.connect(self._advance)
        self._play_button.clicked.connect(self._toggle_playback)
        self._slider.valueChanged.connect(self._draw_frame)
        self.finished.connect(lambda _result: self._timer.stop())
        self._draw_frame(0)

    def _position(self) -> int:
        return int(np.clip(self._slider.value(), 0, max(0, len(self._frames) - 1)))

    def _effective_mask(self, position: int) -> np.ndarray:
        if position in self._edited:
            return self._edited[position]
        return self._frames[position].mask

    def _editable_mask(self, position: int) -> np.ndarray:
        """Return the position's mask as a writable buffer (copy-on-write)."""
        if position not in self._edited:
            self._edited[position] = np.array(self._frames[position].mask, dtype=bool, copy=True)
        return self._edited[position]

    def _toggle_edit_mode(self, checked: bool) -> None:
        self._edit_mode = bool(checked)
        if checked and self._timer.isActive():
            self._toggle_playback()
        self._update_edit_controls()
        # Switch the canvas layout: edit -> local-only zoom; view -> global+inset.
        self._draw_frame(self._position())

    def _update_edit_controls(self) -> None:
        if not self._editable:
            return
        editing = self._edit_mode
        self._add_radio.setEnabled(editing)
        self._erase_radio.setEnabled(editing)
        self._brush_spin.setEnabled(editing)
        has_edit = self._position() in self._edited
        self._save_button.setEnabled(editing and has_edit)
        self._revert_button.setEnabled(has_edit)
        has_frames = bool(self._frames)
        self._delete_frame_button.setEnabled(has_frames)
        self._play_button.setEnabled(not editing and len(self._frames) > 1)

    def _is_paint_axes(self, event: Any) -> bool:
        """Painting is allowed on either the global axes or the zoom inset.

        Both render the slice in the same full-image pixel coordinates, so the
        brush math is identical regardless of which one the cursor is over; the
        inset just exposes it at higher magnification.
        """
        return event.inaxes is self._axes or (
            self._inset is not None and event.inaxes is self._inset
        )

    def _on_canvas_press(self, event: Any) -> None:
        if not self._edit_mode or not self._is_paint_axes(event):
            return
        self._painting = True
        self._paint_at(event)

    def _on_canvas_motion(self, event: Any) -> None:
        if self._painting:
            self._paint_at(event)

    def _on_canvas_release(self, _event: Any) -> None:
        if self._painting:
            self._painting = False
            self._update_edit_controls()

    def _paint_at(self, event: Any) -> None:
        if not self._is_paint_axes(event) or event.xdata is None or event.ydata is None:
            return
        if not self._frames:
            return
        position = self._position()
        mask = self._editable_mask(position)
        cx, cy = int(round(event.xdata)), int(round(event.ydata))
        radius = int(self._brush_spin.value())
        _stamp_brush(mask, cx, cy, radius, self._brush_add)
        self._draw_frame(position)

    def _save_current_frame(self) -> None:
        if self._target_store is None or not self._frames:
            return
        position = self._position()
        if position not in self._edited:
            return
        row = self._frames[position]
        mask = self._edited[position]
        try:
            self._target_store.put_mask(
                self._target.id,
                row.frame.axis,
                int(row.frame.index),
                mask.astype(np.uint8),
                volume_id=self._target.volume_id,
                stage=self._stage,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存修正", str(exc))
            return
        # Bake the saved edit into the frame view so playback shows the result.
        self._frames[position] = TrackingFrameView(
            frame=row.frame, image=row.image, mask=np.array(mask, dtype=bool, copy=True)
        )
        self._edited.pop(position, None)
        self._edit_status.setText(f"已保存 {_desktop_axis(row.frame.axis)}={row.frame.index} 修正")
        # Saving exits edit mode and returns to the global + inset view so the
        # user immediately sees the corrected result in its full context.
        if self._edit_check.isChecked():
            self._edit_check.setChecked(False)  # triggers _toggle_edit_mode -> redraw
        else:
            self._draw_frame(position)
        if callable(self._on_changed):
            self._on_changed()

    def _revert_current_frame(self) -> None:
        position = self._position()
        if self._edited.pop(position, None) is not None:
            self._edit_status.setText("已撤销未保存修正")
            self._draw_frame(position)

    def _delete_current_frame(self) -> None:
        if self._target_store is None or not self._frames:
            return
        position = self._position()
        row = self._frames[position]
        if row.frame.origin == "missing":
            QMessageBox.information(self, "删除此帧", "该位置本就没有有效帧。")
            return
        if QMessageBox.question(
            self,
            "删除此帧",
            f"确定删除 {_desktop_axis(row.frame.axis)}={row.frame.index} 这一帧吗？",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._target_store.delete_frame(
                self._target.id,
                row.frame.axis,
                int(row.frame.index),
                volume_id=self._target.volume_id,
                stage=self._stage,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "删除此帧", str(exc))
            return
        del self._frames[position]
        # Edited overrides are keyed by position; rebuild past the removed index.
        self._edited = {
            (pos if pos < position else pos - 1): buf
            for pos, buf in self._edited.items()
            if pos != position
        }
        self._slider.setRange(0, max(0, len(self._frames) - 1))
        self._edit_status.setText(f"已删除 {_desktop_axis(row.frame.axis)}={row.frame.index}")
        self._draw_frame(self._position())
        if callable(self._on_changed):
            self._on_changed()

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
        # Full rebuild each draw so the inset (zoom) axes never accumulates.
        self._figure.clear()
        self._inset = None
        self._axes = self._figure.add_subplot(111)
        if not self._frames:
            self._axes.text(0.5, 0.5, "没有可显示的追踪帧", ha="center", va="center")
            self._axes.set_axis_off()
            self._canvas.draw_idle()
            return

        position = int(np.clip(position, 0, len(self._frames) - 1))
        row = self._frames[position]
        mask = self._effective_mask(position)
        edited = position in self._edited
        fill_rgb = (1.0, 0.6, 0.1) if edited else (0.1, 1.0, 0.45)
        contour_color = "#ffaa22" if edited else "#00ff72"

        def _render(ax: Any, *, draw_contour: bool) -> None:
            ax.imshow(row.image, origin="upper", interpolation="nearest")
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[..., :3] = fill_rgb
            overlay[..., 3] = mask.astype(np.float32) * 0.48
            ax.imshow(overlay, origin="upper", interpolation="nearest")
            if draw_contour and min(mask.shape) >= 2 and np.any(mask):
                ax.contour(
                    mask.astype(np.uint8),
                    levels=[0.5],
                    colors=[contour_color],
                    linewidths=1.5,
                )

        height, width = mask.shape
        editing = self._editable and self._edit_mode

        if editing:
            # Edit mode: ONLY the local zoom region on a single, large axes so the
            # brush has a clear, simple, precise canvas — no global view to fight
            # with. We zoom to the (padded) mask neighbourhood.
            _render(self._axes, draw_contour=True)
            if self._crop is not None:
                x0, x1, y0, y1 = self._crop
                self._axes.set_xlim(x0, x1)
                self._axes.set_ylim(y1, y0)
            else:
                self._axes.set_xlim(0, width - 1)
                self._axes.set_ylim(height - 1, 0)
            mode_tag = "编辑模式 · 仅局部放大"
        else:
            # View mode: the full slice (global position is never lost) plus an
            # inset that marks + magnifies the mask neighbourhood (paper-style
            # "inset zoom"), tied together by the indicator rectangle + lines.
            _render(self._axes, draw_contour=self._crop is None)
            self._axes.set_xlim(0, width - 1)
            self._axes.set_ylim(height - 1, 0)
            if self._crop is not None:
                x0, x1, y0, y1 = self._crop
                inset = self._axes.inset_axes([0.60, 0.58, 0.38, 0.38])
                _render(inset, draw_contour=True)
                inset.set_xlim(x0, x1)
                inset.set_ylim(y1, y0)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_edgecolor(contour_color)
                    spine.set_linewidth(1.4)
                self._axes.indicate_inset_zoom(inset, edgecolor=contour_color, alpha=0.9)
                inset.set_title("局部放大", fontsize=8, color=contour_color, pad=2)
                self._inset = inset
            mode_tag = "全局图 + 局部放大"

        axis_label = _desktop_axis(row.frame.axis)
        state_text = "未识别" if row.frame.origin == "missing" else f"area={int(mask.sum())} px"
        if edited:
            state_text += " · 已修改(未保存)"
        self._axes.set_title(
            f"{self._target.id}  {axis_label}={row.frame.index}"
            f"  {state_text}  ·  {mode_tag}"
        )
        self._axes.set_xlabel(_horizontal_label(axis_label))
        self._axes.set_ylabel(_vertical_label(axis_label))
        self._frame_label.setText(f"{position + 1} / {len(self._frames)}")
        self._canvas.draw_idle()
        if self._editable:
            self._update_edit_controls()


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
    # rgt_overlay composite: render the same image SAM3 saw, via the shared
    # renderer + catalogue params/span (so review matches training input).
    info_fn = getattr(volume_store, "info", None)
    if callable(info_fn):
        try:
            info = info_fn(volume_id) or {}
        except Exception:  # noqa: BLE001 - fall back to the raw path
            info = {}
        if is_rgt_composite_meta(info):
            try:
                rgb = compose_rgt_rgb_from_meta(volume_store, info, axis, int(index))
            except Exception:  # noqa: BLE001 - mask-only playback is still useful
                return np.zeros((*shape, 3), dtype=np.uint8)
            return rgb if rgb.shape[:2] == shape else np.zeros((*shape, 3), dtype=np.uint8)
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
