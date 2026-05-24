"""SAM3 workbench: ROI-bound view for prompt + segment + propagate.

Opened when the user draws an ROI on a regular reservoir section. The
workbench owns a fixed ROI for its entire lifetime; cells outside the
ROI are never drawn. Inside the ROI the user can:

- (this step) Step through frames along the ROI's propagation axis
  (i or j) and see the corner-point section coloured by lithology.
- (Step 3) Drop SAM3 prompts (text / box / point) onto the current
  frame and run image-mode segmentation.
- (Step 4) Reverse-lookup the resulting mask → cell IJK set →
  ReservoirSelectionLayer in the scene.
- (Step 5) Sweep the propagation axis to drive the SAM3 video
  predictor, yielding a 3D ReservoirBodyLayer.

The image shown here is rendered through ``render_roi_section`` so
it's pixel-stable across frames — that's what SAM3 video propagation
needs. The user-facing reservoir section view (``view_reservoir_section``)
remains free to zoom/pan; this workbench does not.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import logging

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yj_studio.algorithms.builtin.ai.sam3_segment import _apply_box_prompt
from yj_studio.ai.adapters.mask_to_layer import decode_sam3_masks
from yj_studio.ai.service import AIService
from yj_studio.reservoir import ReservoirGrid, SeismicIndexTransform
from yj_studio.reservoir.roi import ROI
from yj_studio.reservoir.sam3_render import SAM3Frame, render_roi_section

logger = logging.getLogger(__name__)


_AXIS_LABELS = {
    "i": "I 剖面 (inline)",
    "j": "J 剖面 (xline)",
}


class SAM3Workbench(QWidget):
    """A ROI-bound section view that will host SAM3 prompts in step 3."""

    # Emitted whenever a new frame finishes rendering, so that future
    # SAM3-related panels (mask preview, status bar, etc.) can react.
    frame_changed = pyqtSignal(str, str, int)    # section_id, axis, index
    # Emitted when the user clicks "save as selection" — main_window
    # creates and registers the ReservoirSelectionLayer.
    selection_committed = pyqtSignal(object)     # ReservoirSelectionLayer instance

    def __init__(
        self,
        grid: ReservoirGrid,
        roi: ROI,
        *,
        axis: str = "i",
        index: int | None = None,
        transform: SeismicIndexTransform | None = None,
        ai_service: AIService | None = None,
        grid_layer_id: str = "",
        grid_id: str = "",
        section_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if axis not in {"i", "j"}:
            raise ValueError(f"axis must be 'i' or 'j', got {axis!r}")

        self.section_id = section_id or str(uuid4())
        self._grid = grid
        self._roi = roi
        self._axis = axis
        self._transform = transform or SeismicIndexTransform()
        self._ai_service = ai_service
        self._grid_layer_id = grid_layer_id
        self._grid_id = grid_id
        self._frame: SAM3Frame | None = None
        self._last_mask: np.ndarray | None = None    # SAM3 result, for reverse-lookup
        # Frames produced by the last "沿轴追踪" run, keyed by index
        # along the propagation axis. Held in-memory so the user can
        # drag the spinbox back to any tracked frame and see exactly
        # what SAM3 chose — critical for spotting frames where the
        # tracker lost the body or jumped to the wrong neighbour.
        self._tracked_masks: dict[int, np.ndarray] = {}
        self._track_range: tuple[int, int] | None = None    # (lo, hi) inclusive
        # Playback timer for the ▶ button. None when not playing.
        self._playback_timer = None

        # Prompt state — pixel-space coordinates collected against
        # the current frame. Reset whenever the frame changes (i.e.
        # the underlying image changes), because SAM3 needs each
        # ``set_image`` call to be followed by fresh prompts.
        self._prompt_boxes: list[tuple[float, float, float, float]] = []    # (x0,y0,x1,y1) in pixels
        self._prompt_points: list[tuple[float, float]] = []                  # (x, y) in pixels
        # Visual artists drawn on top of the image; we clear them on
        # rerender or "clear prompts".
        self._prompt_artists: list = []
        self._mask_artist = None    # the imshow handle for the SAM3 mask overlay

        # Default the index to the ROI's mid-frame along the
        # propagation axis. The propagation axis on an I-section
        # workbench is i; on a J-section it's j. We still honour an
        # explicit caller-supplied index when given.
        lo, hi = self._propagation_range()
        if index is None:
            index = (lo + hi - 1) // 2
        self._index = max(lo, min(int(index), hi - 1))

        # Currently active prompt-input mode: "box", "point", or None.
        self._prompt_mode: str | None = None

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------ public

    @property
    def axis(self) -> str:
        return self._axis

    @property
    def index(self) -> int:
        return self._index

    @property
    def roi(self) -> ROI:
        return self._roi

    @property
    def title(self) -> str:
        il, ih, jl, jh, kl, kh = self._roi
        roi_str = f"i[{il}:{ih}] j[{jl}:{jh}] k[{kl}:{kh}]"
        return f"SAM3 · {_AXIS_LABELS[self._axis]} · {roi_str}"

    def current_frame(self) -> SAM3Frame | None:
        """The most recently rendered frame, or None before first render."""
        return self._frame

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        controls = QHBoxLayout()
        controls.setContentsMargins(6, 6, 6, 0)
        controls.addWidget(QLabel("剖面:"))
        self._axis_combo = QComboBox()
        for key, label in _AXIS_LABELS.items():
            self._axis_combo.addItem(label, key)
        self._axis_combo.setCurrentIndex(list(_AXIS_LABELS).index(self._axis))
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        controls.addWidget(self._axis_combo)

        controls.addWidget(QLabel("索引:"))
        self._index_spin = QSpinBox()
        lo, hi = self._propagation_range()
        self._index_spin.setRange(lo, hi - 1)
        self._index_spin.setValue(self._index)
        self._index_spin.valueChanged.connect(self._on_index_changed)
        controls.addWidget(self._index_spin)

        controls.addSpacing(12)
        controls.addWidget(QLabel("提示:"))
        self._box_button = QPushButton("框")
        self._box_button.setCheckable(True)
        self._box_button.setToolTip("拖矩形圈出感兴趣区域")
        self._box_button.toggled.connect(
            lambda on: self._set_prompt_mode("box" if on else None)
        )
        controls.addWidget(self._box_button)

        self._point_button = QPushButton("点")
        self._point_button.setCheckable(True)
        self._point_button.setToolTip("单击放置正向点提示")
        self._point_button.toggled.connect(
            lambda on: self._set_prompt_mode("point" if on else None)
        )
        controls.addWidget(self._point_button)

        controls.addWidget(QLabel("文字:"))
        self._text_edit = QLineEdit()
        self._text_edit.setPlaceholderText("可选,例如 砂体")
        self._text_edit.setMaximumWidth(160)
        controls.addWidget(self._text_edit)

        controls.addSpacing(6)
        self._clear_button = QPushButton("清除")
        self._clear_button.clicked.connect(self._clear_prompts)
        controls.addWidget(self._clear_button)

        self._run_button = QPushButton("运行 SAM3")
        self._run_button.setStyleSheet("font-weight: bold;")
        self._run_button.clicked.connect(self._run_sam3)
        controls.addWidget(self._run_button)

        self._save_button = QPushButton("保存为选择")
        self._save_button.setEnabled(False)
        self._save_button.setToolTip(
            "把当前 SAM3 mask 反查成储层 cell 集合,"
            "添加为新的 ReservoirSelectionLayer。"
        )
        self._save_button.clicked.connect(self._save_selection)
        controls.addWidget(self._save_button)

        self._play_button = QPushButton("▶")
        self._play_button.setCheckable(True)
        self._play_button.setEnabled(False)
        self._play_button.setMaximumWidth(36)
        self._play_button.setToolTip(
            "回放上一轮沿轴追踪的每一帧 mask,逐帧检查 SAM3 的追踪质量。"
        )
        self._play_button.toggled.connect(self._on_play_toggled)
        controls.addWidget(self._play_button)

        self._propagate_button = QPushButton("沿轴追踪…")
        self._propagate_button.setEnabled(False)
        self._propagate_button.setToolTip(
            "把当前 mask 沿剖面轴向前后追踪 N 帧,"
            "汇聚成 3D 储层选择体。"
        )
        self._propagate_button.clicked.connect(self._propagate_along_axis)
        controls.addWidget(self._propagate_button)

        controls.addStretch(1)
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #666;")
        controls.addWidget(self._info_label)

        layout.addLayout(controls)

        # The display canvas. We deliberately reuse the same image
        # shape SAM3 will see, so what the user clicks at maps 1:1 to
        # SAM3 pixels. Aspect ratio adapts to the ROI; the canvas
        # itself stretches with the dock — but the data we feed SAM3
        # is always the deterministic image inside ``SAM3Frame``, not
        # whatever this on-screen widget happens to display.
        # Canvas size matches the SAM3 image dimensions 1:1 so what
        # the user sees and what SAM3 ingests are the same pixels.
        # Picked once at construction; the figure is recreated on
        # axis switches if the ROI's aspect changes enough to need a
        # different shape (see ``_render``).
        self._figure = Figure(figsize=(26, 8), dpi=100, tight_layout=False)
        self._axes = self._figure.add_axes((0.0, 0.0, 1.0, 1.0))
        self._axes.set_axis_off()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        # Lazily created box-prompt selector; only active in "box" mode.
        self._selector: RectangleSelector | None = None

        # Wrap in a QScrollArea so Qt never resamples the canvas pixels.
        # If the canvas is larger than the dock, the user scrolls; if
        # smaller, it sits centred. Either way SAM3 sees the same
        # pixels the user clicks on.
        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------ slots

    def _on_axis_changed(self, _idx: int) -> None:
        new_axis = self._axis_combo.currentData()
        if new_axis == self._axis:
            return
        self._axis = new_axis
        lo, hi = self._propagation_range()
        self._index_spin.blockSignals(True)
        self._index_spin.setRange(lo, hi - 1)
        # Re-centre when switching axis; the previous index has no
        # meaningful counterpart on the new axis.
        self._index = (lo + hi - 1) // 2
        self._index_spin.setValue(self._index)
        self._index_spin.blockSignals(False)
        self._render()

    def _on_index_changed(self, value: int) -> None:
        self._index = int(value)
        self._render()

    # ------------------------------------------------------------------ prompts

    def _set_prompt_mode(self, mode: str | None) -> None:
        """Switch between 'box', 'point', or no prompt-input mode.

        The two buttons are kept mutually exclusive by suppressing the
        other one's checked state. We create the RectangleSelector
        lazily on first 'box' entry — its constructor needs an axes
        that already has data on it.
        """
        self._prompt_mode = mode
        # Sync button check states without re-emitting.
        if mode != "box" and self._box_button.isChecked():
            self._box_button.blockSignals(True)
            self._box_button.setChecked(False)
            self._box_button.blockSignals(False)
        if mode != "point" and self._point_button.isChecked():
            self._point_button.blockSignals(True)
            self._point_button.setChecked(False)
            self._point_button.blockSignals(False)
        # Set up / tear down the box selector as needed.
        if mode == "box":
            self._ensure_selector()
            if self._selector is not None:
                self._selector.set_active(True)
        elif self._selector is not None:
            self._selector.set_active(False)

    def _ensure_selector(self) -> None:
        if self._selector is not None:
            return
        self._selector = RectangleSelector(
            self._axes,
            self._on_box_drawn,
            useblit=False,
            button=[1],
            minspanx=4, minspany=4,
            spancoords="pixels",
            interactive=False,
            props={"facecolor": (1.0, 0.6, 0.0, 0.18),
                   "edgecolor": (1.0, 0.4, 0.0, 0.9),
                   "linewidth": 1.2},
        )
        self._selector.set_active(False)

    def _on_box_drawn(self, click, release) -> None:
        """Add a box prompt and draw it on top of the image."""
        if click.xdata is None or release.xdata is None:
            return
        x0, x1 = sorted((float(click.xdata), float(release.xdata)))
        y0, y1 = sorted((float(click.ydata), float(release.ydata)))
        self._prompt_boxes.append((x0, y0, x1, y1))
        rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=(1.0, 0.4, 0.0, 0.9), linewidth=1.5,
        )
        self._axes.add_patch(rect)
        self._prompt_artists.append(rect)
        self._canvas.draw_idle()
        # One-shot — flip mode off so the user can pan / inspect.
        self._box_button.setChecked(False)

    def _on_canvas_click(self, event) -> None:
        """Add a positive point prompt if in 'point' mode."""
        if self._prompt_mode != "point":
            return
        if event.inaxes is not self._axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        # The RectangleSelector eats clicks while active; we only get
        # here when mode == 'point', so no need to filter further.
        x, y = float(event.xdata), float(event.ydata)
        self._prompt_points.append((x, y))
        marker = Circle((x, y), radius=6.0,
                        facecolor=(0.0, 0.85, 0.4, 0.95),
                        edgecolor="white", linewidth=1.0)
        self._axes.add_patch(marker)
        self._prompt_artists.append(marker)
        self._canvas.draw_idle()

    def _clear_prompts(self) -> None:
        self._prompt_boxes.clear()
        self._prompt_points.clear()
        for artist in self._prompt_artists:
            artist.remove()
        self._prompt_artists.clear()
        if self._mask_artist is not None:
            self._mask_artist.remove()
            self._mask_artist = None
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ SAM3 run

    def _run_sam3(self) -> None:
        """Push the current frame + prompts to SAM3 and draw the mask."""
        if self._frame is None:
            return
        if not (self._prompt_boxes or self._prompt_points
                or self._text_edit.text().strip()):
            QMessageBox.information(
                self, "SAM3",
                "请先放置至少一个框、点,或输入文字提示。",
            )
            return
        if self._ai_service is None or not self._ai_service.is_ready():
            QMessageBox.warning(
                self, "SAM3",
                "AI 服务未就绪 — 请在 AI 面板点击 \"启动 AI\"。",
            )
            return

        processor = self._ai_service.image_processor
        if processor is None:
            QMessageBox.warning(self, "SAM3", "SAM3 图像处理器不可用。")
            return

        # Build a PIL image from the offscreen-rendered RGB array.
        from PIL import Image
        pil = Image.fromarray(self._frame.image)
        height, width = self._frame.image.shape[:2]

        self._info_label.setText("SAM3 推理中…")
        self._canvas.setEnabled(False)
        QApplication = self._qapp()
        if QApplication is not None:
            QApplication.processEvents()
        try:
            self._ai_service.mark_busy("SAM3 储层剖面分割")
            state = processor.set_image(pil)
            text = self._text_edit.text().strip()
            if text:
                state = processor.set_text_prompt(prompt=text, state=state)
            for (bx0, by0, bx1, by1) in self._prompt_boxes:
                state = _apply_box_prompt(
                    processor, state, bx0, by0, bx1, by1, width, height,
                )
            # Points: wrap each in a tiny pseudo-box, matching how the
            # regular sam3_segment algorithm handles points (SAM3 has
            # no native point prompt API).
            radius = 8.0
            for (px, py) in self._prompt_points:
                state = _apply_box_prompt(
                    processor, state,
                    px - radius, py - radius, px + radius, py + radius,
                    width, height,
                )
            detections = decode_sam3_masks(state)
        except Exception as exc:    # noqa: BLE001 — surface to user
            logger.exception("SAM3 inference failed")
            QMessageBox.critical(self, "SAM3", f"推理失败:{exc}")
            self._info_label.setText("SAM3 推理失败")
            return
        finally:
            self._ai_service.mark_ready()
            self._canvas.setEnabled(True)

        if not detections:
            self._info_label.setText("SAM3: 无候选 mask")
            return

        detections.sort(key=lambda d: d["score"], reverse=True)
        best = detections[0]
        mask = np.asarray(best["mask"], dtype=bool)
        # SAM3 returns (H, W) row-major image-pixel layout, same as
        # the image we fed it — overlay directly.
        if mask.shape != (height, width):
            logger.warning(
                "SAM3 mask shape %s != frame shape (%d, %d); skipping overlay",
                mask.shape, height, width,
            )
            return
        self._draw_mask_overlay(mask)
        self._info_label.setText(
            f"SAM3: 候选 {len(detections)} 个,显示最优 (score {best['score']:.2f})"
        )

    def _draw_mask_overlay(self, mask: np.ndarray) -> None:
        """Paint the SAM3 mask as a translucent red layer."""
        if self._mask_artist is not None:
            self._mask_artist.remove()
            self._mask_artist = None
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask, 0] = 1.0    # red
        overlay[mask, 1] = 0.3
        overlay[mask, 2] = 0.2
        overlay[mask, 3] = 0.45    # alpha
        self._mask_artist = self._axes.imshow(
            overlay, interpolation="nearest", aspect="equal",
        )
        self._canvas.draw_idle()
        # Stash the mask + enable "save" and "propagate" — both need
        # the mask plus the current frame's cell-id grid.
        self._last_mask = mask
        self._save_button.setEnabled(True)
        self._propagate_button.setEnabled(True)

    @staticmethod
    def _qapp():
        try:
            from PyQt6.QtWidgets import QApplication
            return QApplication
        except Exception:
            return None

    # ------------------------------------------------------------------ playback

    def _on_play_toggled(self, on: bool) -> None:
        """Start / stop frame-by-frame playback of the tracked range."""
        from PyQt6.QtCore import QTimer
        if on:
            if self._track_range is None:
                self._play_button.setChecked(False)
                return
            lo, hi = self._track_range
            # If user paused in the middle, resume from current index;
            # otherwise start fresh from lo.
            if not (lo <= self._index <= hi):
                self._index_spin.setValue(lo)
            self._play_button.setText("⏸")
            timer = QTimer(self)
            timer.setInterval(220)    # ms per frame; ~4.5 FPS feels readable
            timer.timeout.connect(self._on_play_tick)
            timer.start()
            self._playback_timer = timer
        else:
            if self._playback_timer is not None:
                self._playback_timer.stop()
                self._playback_timer.deleteLater()
                self._playback_timer = None
            self._play_button.setText("▶")

    def _on_play_tick(self) -> None:
        if self._track_range is None:
            self._play_button.setChecked(False)
            return
        lo, hi = self._track_range
        next_idx = self._index + 1
        if next_idx > hi:
            # Reached the end — loop back to lo, but stop after one
            # full pass so the timer doesn't run forever.
            next_idx = lo
            self._play_button.setChecked(False)
            return
        self._index_spin.setValue(next_idx)

    # ------------------------------------------------------------------ propagation

    def _propagate_along_axis(self) -> None:
        """Sweep the propagation axis, re-segmenting each frame with
        the previous mask's bbox as a box prompt.

        This is "box tracking", not the SAM3 video predictor — it
        falls back to image-mode SAM3, which avoids the Triton /
        offscreen-JPEG requirements of the video pipeline but loses
        cross-frame appearance memory. Good enough for short sweeps
        where the body's bbox is stable frame to frame; switch to
        the video predictor in a follow-up step if needed.
        """
        from PyQt6.QtWidgets import QInputDialog
        from PIL import Image
        from yj_studio.scene.layers import ReservoirSelectionLayer

        if self._frame is None or self._last_mask is None:
            return
        if self._ai_service is None or not self._ai_service.is_ready():
            QMessageBox.warning(self, "SAM3", "AI 服务未就绪。")
            return
        processor = self._ai_service.image_processor
        if processor is None:
            return

        # Ask how far to sweep in each direction along the axis.
        lo, hi = self._propagation_range()
        seed_idx = self._frame.index
        max_back = seed_idx - lo
        max_fwd = (hi - 1) - seed_idx
        n_each, ok = QInputDialog.getInt(
            self, "沿轴追踪",
            f"种子帧 {self._axis}={seed_idx}\n"
            f"前后各追踪多少帧? (范围 [-{max_back}, +{max_fwd}])",
            value=min(10, max(max_back, max_fwd)),
            min=1, max=max(1, max(max_back, max_fwd)),
        )
        if not ok:
            return

        # Build the seed bbox from the current mask (in image pixel coords).
        seed_mask = self._last_mask
        ys, xs = np.where(seed_mask)
        if xs.size == 0:
            return
        seed_box_px = (
            float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max()),
        )
        # Pad a few pixels so tracking has room when the body shifts.
        pad = 6.0
        text = self._text_edit.text().strip()

        # Reset prior tracking results — a fresh sweep starts a new
        # range.
        self._tracked_masks = {seed_idx: seed_mask.copy()}

        # Run forward then backward from the seed, accumulating cells.
        accumulated_cells: list[np.ndarray] = []
        accumulated_cells.append(self._frame.cell_id_grid[
            seed_mask & (self._frame.cell_id_grid[..., 0] >= 0)
        ])
        index_lo = seed_idx
        index_hi = seed_idx
        score_threshold = 0.25    # stop the sweep when SAM3 gets unsure

        self._info_label.setText("追踪中…")
        self._canvas.setEnabled(False)
        QApp = self._qapp()
        if QApp is not None:
            QApp.processEvents()

        try:
            self._ai_service.mark_busy("SAM3 沿轴追踪")
            for direction, step, end in (
                ("→", 1, min(seed_idx + n_each, hi - 1)),
                ("←", -1, max(seed_idx - n_each, lo)),
            ):
                prev_box = seed_box_px
                idx = seed_idx
                while True:
                    idx_next = idx + step
                    if (step > 0 and idx_next > end) or (step < 0 and idx_next < end):
                        break
                    frame = render_roi_section(
                        self._grid, self._axis, idx_next,
                        self._roi, transform=self._transform,
                    )
                    pil = Image.fromarray(frame.image)
                    height, width = frame.image.shape[:2]
                    state = processor.set_image(pil)
                    if text:
                        state = processor.set_text_prompt(prompt=text, state=state)
                    x0, y0, x1, y1 = prev_box
                    state = _apply_box_prompt(
                        processor, state,
                        x0 - pad, y0 - pad, x1 + pad, y1 + pad,
                        width, height,
                    )
                    detections = decode_sam3_masks(state)
                    if not detections:
                        logger.info("propagate %s: no mask at %s=%d, stopping",
                                    direction, self._axis, idx_next)
                        break
                    detections.sort(key=lambda d: d["score"], reverse=True)
                    best = detections[0]
                    if best["score"] < score_threshold:
                        logger.info("propagate %s: low score %.2f at %s=%d, stopping",
                                    direction, best["score"], self._axis, idx_next)
                        break
                    new_mask = np.asarray(best["mask"], dtype=bool)
                    if new_mask.shape != (height, width):
                        logger.warning("propagate: mask shape mismatch, stopping")
                        break
                    ys2, xs2 = np.where(new_mask)
                    if xs2.size == 0:
                        break
                    prev_box = (
                        float(xs2.min()), float(ys2.min()),
                        float(xs2.max()), float(ys2.max()),
                    )
                    # Reverse-lookup this frame's cells.
                    valid_pixels = new_mask & (frame.cell_id_grid[..., 0] >= 0)
                    accumulated_cells.append(frame.cell_id_grid[valid_pixels])
                    # Stash mask for in-app playback / inspection. The
                    # mask is per-pixel and a few hundred KB / frame —
                    # cheap enough to keep all of them in memory for
                    # a typical sweep (~50 frames).
                    self._tracked_masks[idx_next] = new_mask
                    idx = idx_next
                    if step > 0:
                        index_hi = idx
                    else:
                        index_lo = idx
                    self._info_label.setText(
                        f"追踪 {direction} {self._axis}={idx} (score {best['score']:.2f})"
                    )
                    if QApp is not None:
                        QApp.processEvents()
        finally:
            self._ai_service.mark_ready()
            self._canvas.setEnabled(True)

        if not accumulated_cells:
            self._info_label.setText("追踪完成,但无任何 cell")
            return
        all_cells = np.concatenate(accumulated_cells, axis=0)
        unique_cells = np.unique(all_cells, axis=0)
        layer = ReservoirSelectionLayer(
            name=(
                f"SAM3 体 · {self._axis}=[{index_lo},{index_hi}] · "
                f"{len(unique_cells):,} 单元"
            ),
            grid_layer_id=self._grid_layer_id,
            grid_id=self._grid_id,
            cell_ids=unique_cells,
            source_axis=self._axis,
            source_index_lo=index_lo,
            source_index_hi=index_hi,
            color=(1.0, 0.3, 0.2, 1.0),
            opacity=1.0,
            visible=True,
        )
        self.selection_committed.emit(layer)
        self._track_range = (index_lo, index_hi)
        self._info_label.setText(
            f"追踪完成: {self._axis}=[{index_lo}, {index_hi}], "
            f"{len(unique_cells):,} 单元 · 拖索引或点 ▶ 回看"
        )
        # Restore the user back to the seed frame; the spinbox setter
        # triggers _render which will overlay the cached mask.
        if self._index != seed_idx:
            self._index_spin.setValue(seed_idx)
        else:
            self._render()    # force overlay refresh on seed frame
        if hasattr(self, "_play_button"):
            self._play_button.setEnabled(True)

    # ------------------------------------------------------------------ selection

    def _save_selection(self) -> None:
        """Reverse-lookup the current mask to cell IJK + emit as a layer.

        Looks up every mask pixel in the frame's cell_id_grid; pixels
        that don't sit on any cell (cell_id == -1) are skipped. The
        result is deduplicated, packed into a ReservoirSelectionLayer,
        and handed to main_window via ``selection_committed``.
        """
        from yj_studio.scene.layers import ReservoirSelectionLayer

        if self._frame is None or self._last_mask is None:
            return
        cell_grid = self._frame.cell_id_grid
        mask = self._last_mask
        if mask.shape != cell_grid.shape[:2]:
            logger.warning(
                "mask shape %s != cell_id_grid shape %s; aborting save",
                mask.shape, cell_grid.shape[:2],
            )
            return

        valid_pixels = mask & (cell_grid[..., 0] >= 0)
        if not valid_pixels.any():
            logger.info("SAM3 mask covers only non-cell pixels — nothing to save")
            return

        cell_triples = cell_grid[valid_pixels]    # (N, 3) int32
        unique_cells = np.unique(cell_triples, axis=0)

        # Index along the propagation axis is the section's slicing
        # axis, which is always the same as self._axis on a workbench.
        layer = ReservoirSelectionLayer(
            name=f"SAM3 选择 · {self._axis}={self._frame.index} · {len(unique_cells)} 单元",
            grid_layer_id=self._grid_layer_id,
            grid_id=self._grid_id,
            cell_ids=unique_cells,
            source_axis=self._axis,
            source_index_lo=self._frame.index,
            source_index_hi=self._frame.index,
            color=(1.0, 0.3, 0.2, 1.0),
            opacity=1.0,
            visible=True,
        )
        self.selection_committed.emit(layer)
        self._info_label.setText(
            f"已保存 {len(unique_cells):,} 单元到图层"
        )

    # ------------------------------------------------------------------ rendering

    def _render(self) -> None:
        frame = render_roi_section(
            self._grid,
            self._axis,
            self._index,
            self._roi,
            transform=self._transform,
        )
        self._frame = frame

        # Switching frames invalidates the prompts (they were drawn
        # against a different image) and the prior SAM3 mask. Reset
        # everything before redrawing.
        self._prompt_boxes.clear()
        self._prompt_points.clear()
        self._prompt_artists.clear()
        self._mask_artist = None
        self._last_mask = None
        # Save / propagate are invalid until a fresh mask is computed
        # for the new frame.
        if hasattr(self, "_save_button"):
            self._save_button.setEnabled(False)
        if hasattr(self, "_propagate_button"):
            self._propagate_button.setEnabled(False)
        # RectangleSelector holds an internal reference to the old
        # axes; drop it so a fresh one is created next time the user
        # enters box mode.
        self._selector = None

        self._axes.clear()
        self._axes.set_axis_off()
        # imshow with the default aspect="equal" gives matplotlib free
        # rein over the axes box; we want 1:1 pixel mapping so prompts
        # stay aligned with what SAM3 sees. Force the figure size to
        # match the frame and tell imshow to use raw pixel coords.
        h, w = frame.image.shape[:2]
        self._figure.set_size_inches(w / 100.0, h / 100.0, forward=True)
        self._canvas.setFixedSize(QSize(w, h))
        self._axes.imshow(
            frame.image,
            interpolation="nearest",
            aspect="equal",
        )

        # If this frame is part of a recent ▶ sweep, redisplay the
        # cached SAM3 mask. Useful for spotting frames where the
        # tracker drifted or lost the body — drag the spinbox back
        # and you see exactly what SAM3 produced there.
        cached = self._tracked_masks.get(self._index)
        if cached is not None and cached.shape == (h, w):
            self._draw_mask_overlay(cached)

        self._canvas.draw_idle()

        ny_active = int((frame.cell_id_grid[..., 0] >= 0).sum())
        suffix = ""
        if self._track_range is not None:
            lo, hi = self._track_range
            suffix = f" · 追踪 {self._axis}=[{lo}, {hi}]"
        self._info_label.setText(
            f"{w}×{h} px · {ny_active:,} cell-像素{suffix}"
        )
        self.frame_changed.emit(self.section_id, self._axis, self._index)

    # ------------------------------------------------------------------ helpers

    def _propagation_range(self) -> tuple[int, int]:
        """Half-open index range along the section's slicing axis."""
        il, ih, jl, jh, _kl, _kh = self._roi
        if self._axis == "i":
            return il, ih
        return jl, jh
