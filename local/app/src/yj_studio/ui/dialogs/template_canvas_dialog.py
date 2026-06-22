"""Blank whiteboard for sketching a morphology template.

The "形态模板识别" feature matches *shape*, not a region of the current image,
so the template is drawn on an independent blank canvas — not on the section.
The user free-hand draws an abstract outline (a left-high/right-low wedge, a
lens, a triangle…), then clicks 保存模板; the dialog returns the outline as a
normalized polygon (``[0, 1]`` in image coordinates, ``y`` increasing downward
so it matches how the server rasterises slices). Absolute size is irrelevant —
only the form is captured — so the canvas is unitless.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Free-hand drawing yields one point per mouse-move; decimate before returning so
# the polygon stays well under the server's point cap while keeping its shape.
_MAX_POINTS = 80
_MIN_POINTS = 3


class TemplateCanvasDialog(QDialog):
    """Modal whiteboard returning a hand-drawn shape as a normalized polygon."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("绘制形态模板")
        self.resize(460, 480)
        self._points: list[tuple[float, float]] = []
        self._drawing = False
        self._line = None
        self._patch = None

        layout = QVBoxLayout(self)
        hint = QLabel(
            "在下面的白板上按住左键手绘目标形态（只看形状，不看大小）。\n"
            "例如左高右低的楔形、三角形、透镜状轮廓等。",
            self,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._reset_axes()
        layout.addWidget(self._canvas, 1)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

        controls = QHBoxLayout()
        self._clear_button = QPushButton("清除", self)
        self._clear_button.clicked.connect(self._clear)
        controls.addWidget(self._clear_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        self._save_button = self._buttons.addButton(
            "保存模板", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._save_button.setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------ canvas

    def _reset_axes(self) -> None:
        self._axes.clear()
        self._axes.set_xlim(0.0, 1.0)
        # Image coordinates: y increases downward (top == 0) to match the server's
        # slice rasterisation, so "上/下" in the drawing maps correctly.
        self._axes.set_ylim(1.0, 0.0)
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._axes.set_facecolor("white")
        self._axes.set_aspect("equal", adjustable="box")
        self._line = None
        self._patch = None
        self._canvas.draw_idle()

    def _clamp(self, value: float | None) -> float | None:
        if value is None:
            return None
        return min(1.0, max(0.0, float(value)))

    def _on_press(self, event) -> None:  # noqa: ANN001 - matplotlib event
        if event.inaxes is not self._axes or event.button != 1:
            return
        x, y = self._clamp(event.xdata), self._clamp(event.ydata)
        if x is None or y is None:
            return
        self._drawing = True
        self._points = [(x, y)]
        self._redraw_stroke(closed=False)

    def _on_motion(self, event) -> None:  # noqa: ANN001
        if not self._drawing or event.inaxes is not self._axes:
            return
        x, y = self._clamp(event.xdata), self._clamp(event.ydata)
        if x is None or y is None:
            return
        self._points.append((x, y))
        self._redraw_stroke(closed=False)

    def _on_release(self, event) -> None:  # noqa: ANN001
        if not self._drawing:
            return
        self._drawing = False
        x, y = self._clamp(event.xdata), self._clamp(event.ydata)
        if x is not None and y is not None:
            self._points.append((x, y))
        self._redraw_stroke(closed=len(self._points) >= _MIN_POINTS)
        self._save_button.setEnabled(len(self._points) >= _MIN_POINTS)

    def _redraw_stroke(self, *, closed: bool) -> None:
        from matplotlib.patches import Polygon

        if self._line is not None:
            try:
                self._line.remove()
            except Exception:  # noqa: BLE001
                pass
            self._line = None
        if self._patch is not None:
            try:
                self._patch.remove()
            except Exception:  # noqa: BLE001
                pass
            self._patch = None
        if len(self._points) >= 2:
            xs = [p[0] for p in self._points]
            ys = [p[1] for p in self._points]
            self._line = self._axes.plot(
                xs, ys, color="#ff9800", linewidth=2.0, zorder=5
            )[0]
        if closed and len(self._points) >= _MIN_POINTS:
            self._patch = Polygon(
                self._points,
                closed=True,
                facecolor="#ff980033",
                edgecolor="#ff9800",
                linewidth=2.0,
                zorder=4,
            )
            self._axes.add_patch(self._patch)
        self._canvas.draw_idle()

    def _clear(self) -> None:
        self._points = []
        self._drawing = False
        self._save_button.setEnabled(False)
        self._reset_axes()

    # ------------------------------------------------------------------ result

    def template_points(self) -> list[list[float]]:
        """The drawn outline as a normalized ``[[x, y], ...]`` polygon ([0, 1])."""
        points = self._points
        if len(points) > _MAX_POINTS:
            step = len(points) / float(_MAX_POINTS)
            sampled = [points[int(i * step)] for i in range(_MAX_POINTS)]
            if sampled[-1] != points[-1]:
                sampled[-1] = points[-1]
            points = sampled
        return [[float(x), float(y)] for x, y in points]
