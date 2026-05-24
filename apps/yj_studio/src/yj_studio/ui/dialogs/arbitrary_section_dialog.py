from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class WellMapPoint:
    name: str
    inline: float
    xline: float


class ArbitrarySectionDialog(QDialog):
    """Draw an arbitrary vertical section on a top-down map."""

    def __init__(
        self,
        *,
        shape: tuple[int, int, int],
        topdown_image: np.ndarray | None = None,
        topdown_cmap: str = "Petrel",
        well_points: tuple[WellMapPoint, ...] = (),
        initial_polyline: np.ndarray | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建任意剖面")
        self._shape = shape
        self._topdown_image = _valid_topdown_image(topdown_image, shape)
        self._topdown_cmap = topdown_cmap
        self._well_points = tuple(well_points)
        self._points: list[tuple[float, float]] = _initial_points(initial_polyline)
        self._hover_point: tuple[float, float] | None = None

        layout = QVBoxLayout(self)
        self.status_label = QLabel("左键添加点，右键或撤销删除最后一个点。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._figure = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._canvas.setMinimumHeight(420)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self._canvas)

        form = QFormLayout()
        self.z_start = QSpinBox(self)
        self.z_start.setRange(0, max(0, shape[2] - 1))
        self.z_start.setValue(0)
        self.z_end = QSpinBox(self)
        self.z_end.setRange(0, max(0, shape[2] - 1))
        self.z_end.setValue(max(0, shape[2] - 1))
        self.max_traces = QSpinBox(self)
        self.max_traces.setRange(2, 5000)
        self.max_traces.setValue(min(900, max(2, shape[0] + shape[1])))
        self.snap_radius = QSpinBox(self)
        self.snap_radius.setRange(0, 100)
        self.snap_radius.setValue(8)
        form.addRow("起始 Z", self.z_start)
        form.addRow("结束 Z", self.z_end)
        form.addRow("最大道数", self.max_traces)
        form.addRow("吸附半径", self.snap_radius)
        layout.addLayout(form)

        undo_button = QPushButton("撤销", self)
        undo_button.clicked.connect(self._undo_point)
        clear_button = QPushButton("清空", self)
        clear_button.clicked.connect(self._clear_points)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.addButton(undo_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(clear_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._line_artist = None
        self._point_artist = None
        self._hover_artist = None
        self._canvas.mpl_connect("button_press_event", self._on_button_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._draw_base()

    def accept(self) -> None:
        if len(self._points) < 2:
            QMessageBox.warning(self, "新建任意剖面", "请至少在地图上绘制两个点。")
            return
        super().accept()

    def polyline(self) -> np.ndarray:
        if len(self._points) < 2:
            raise ValueError("请至少在地图上绘制两个点。")
        return np.asarray(self._points, dtype=np.float32)

    def _on_button_press(self, event) -> None:
        if event.inaxes is not self._axes:
            return
        button = _button_value(getattr(event, "button", None))
        if button == 3:
            self._undo_point()
            return
        if button != 1 or event.xdata is None or event.ydata is None:
            return
        point = _clipped_point((float(event.xdata), float(event.ydata)), self._shape)
        snapped, well = snap_point_to_well(point, self._well_points, float(self.snap_radius.value()))
        self._points.append((float(snapped[0]), float(snapped[1])))
        self._set_status(well)
        self._update_overlay()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is not self._axes or event.xdata is None or event.ydata is None:
            self._hover_point = None
            self._update_overlay()
            return
        point = _clipped_point((float(event.xdata), float(event.ydata)), self._shape)
        snapped, _well = snap_point_to_well(point, self._well_points, float(self.snap_radius.value()))
        self._hover_point = (float(snapped[0]), float(snapped[1]))
        self._update_overlay()

    def _undo_point(self) -> None:
        if self._points:
            self._points.pop()
            self._set_status(None)
            self._update_overlay()

    def _clear_points(self) -> None:
        self._points = []
        self._hover_point = None
        self._set_status(None)
        self._update_overlay()

    def _draw_base(self) -> None:
        self._axes.clear()
        nx, ny, _ = self._shape
        if self._topdown_image is not None:
            image = np.asarray(self._topdown_image, dtype=np.float32)
            finite = image[np.isfinite(image)]
            vmin, vmax = np.percentile(finite, [2.0, 98.0]) if finite.size else (-1.0, 1.0)
            self._axes.imshow(
                image.T,
                cmap=_matplotlib_cmap_name(self._topdown_cmap),
                vmin=float(vmin),
                vmax=float(vmax),
                origin="lower",
                aspect="auto",
                extent=(0.0, float(nx - 1), 0.0, float(ny - 1)),
            )
        else:
            self._axes.set_facecolor("#202124")
        self._axes.set_xlim(0.0, float(max(0, nx - 1)))
        self._axes.set_ylim(0.0, float(max(0, ny - 1)))
        self._axes.set_xlabel("Inline")
        self._axes.set_ylabel("Xline")
        self._axes.set_title("顶视图剖面路径")
        self._draw_wells()
        (self._line_artist,) = self._axes.plot([], [], color="#ffb000", linewidth=2.2, zorder=5)
        self._point_artist = self._axes.scatter([], [], c="#ffb000", s=36, edgecolors="black", zorder=6)
        self._hover_artist = self._axes.scatter([], [], c="none", edgecolors="#40c4ff", s=80, linewidths=1.4, zorder=7)
        self._update_overlay(draw=False)
        self._canvas.draw()

    def _draw_wells(self) -> None:
        if not self._well_points:
            return
        xy = np.asarray([(well.inline, well.xline) for well in self._well_points], dtype=np.float32)
        self._axes.scatter(
            xy[:, 0],
            xy[:, 1],
            c="#49d2ff",
            s=34,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        for well in self._well_points:
            self._axes.text(
                well.inline,
                well.xline,
                f" {well.name}",
                color="white",
                fontsize=7,
                ha="left",
                va="center",
                zorder=4,
            )

    def _update_overlay(self, *, draw: bool = True) -> None:
        points = np.asarray(self._points, dtype=np.float32)
        if points.size == 0:
            self._line_artist.set_data([], [])
            self._point_artist.set_offsets(np.empty((0, 2), dtype=np.float32))
        else:
            self._line_artist.set_data(points[:, 0], points[:, 1])
            self._point_artist.set_offsets(points[:, :2])
        if self._hover_point is None:
            self._hover_artist.set_offsets(np.empty((0, 2), dtype=np.float32))
        else:
            self._hover_artist.set_offsets(np.asarray([self._hover_point], dtype=np.float32))
        if draw:
            self._canvas.draw()

    def _set_status(self, well: WellMapPoint | None) -> None:
        prefix = f"{len(self._points)} 个点"
        if well is None:
            self.status_label.setText(f"{prefix}。左键添加点，右键或撤销删除最后一个点。")
            return
        self.status_label.setText(f"{prefix}。已吸附到井 {well.name}。")


def snap_point_to_well(
    point_xy: tuple[float, float],
    well_points: tuple[WellMapPoint, ...],
    snap_radius: float,
) -> tuple[tuple[float, float], WellMapPoint | None]:
    if snap_radius <= 0.0 or not well_points:
        return point_xy, None
    point = np.asarray(point_xy, dtype=np.float32)
    wells = np.asarray([(well.inline, well.xline) for well in well_points], dtype=np.float32)
    distances = np.sqrt(np.sum((wells - point) ** 2, axis=1))
    index = int(np.argmin(distances))
    if float(distances[index]) <= float(snap_radius):
        well = well_points[index]
        return (float(well.inline), float(well.xline)), well
    return point_xy, None


def parse_polyline_text(text: str) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.replace(";", ",").replace("\t", ",")
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"应为 Inline,Xline：{raw_line!r}")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 2:
        raise ValueError("请至少输入两个 Inline,Xline 点。")
    return np.asarray(points, dtype=np.float32)


def _initial_points(initial_polyline: np.ndarray | None) -> list[tuple[float, float]]:
    if initial_polyline is None:
        return []
    points = np.asarray(initial_polyline, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return []
    finite = np.all(np.isfinite(points[:, :2]), axis=1)
    return [(float(point[0]), float(point[1])) for point in points[finite, :2]]


def _valid_topdown_image(topdown_image: np.ndarray | None, shape: tuple[int, int, int]) -> np.ndarray | None:
    if topdown_image is None:
        return None
    image = np.asarray(topdown_image, dtype=np.float32)
    if image.ndim != 2 or image.shape != shape[:2]:
        return None
    return image


def _clipped_point(point_xy: tuple[float, float], shape: tuple[int, int, int]) -> tuple[float, float]:
    return (
        float(np.clip(point_xy[0], 0.0, max(0, shape[0] - 1))),
        float(np.clip(point_xy[1], 0.0, max(0, shape[1] - 1))),
    )


def _button_value(button) -> int | None:
    raw = getattr(button, "value", button)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _matplotlib_cmap_name(cmap: str) -> str:
    name = {"Petrel": "seismic", "petrel": "seismic"}.get(cmap, cmap)
    return name if name in colormaps else "seismic"
