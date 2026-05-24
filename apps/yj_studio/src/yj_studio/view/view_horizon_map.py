from __future__ import annotations

from uuid import uuid4

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from yj_studio.services.horizon_service import HorizonSampleMap


class ViewHorizonMap(QWidget):
    """Matplotlib-backed horizon map view inside the desktop application."""

    section_updated = pyqtSignal(str, str, int)

    def __init__(
        self,
        data: HorizonSampleMap,
        *,
        axis: str = "horizon",
        index: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.section_id = str(uuid4())
        self.data = data
        self.axis = axis
        self.index = int(index)
        self._figure = Figure(figsize=(6, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self.refresh()

    @property
    def title(self) -> str:
        return self.data.title

    def refresh(self) -> None:
        self._figure.clear()
        axes = self._figure.add_subplot(111)
        values = np.asarray(self.data.values, dtype=np.float32)
        finite = np.isfinite(values) & np.asarray(self.data.mask, dtype=bool)
        if not np.any(finite):
            axes.text(0.5, 0.5, "没有有效层位样点", ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            self._canvas.draw_idle()
            return
        masked = _display_values(values, finite)
        nx, ny = values.shape
        # Plan view: inline and xline are the same kind of index, so locking
        # aspect to "equal" gives an undistorted map regardless of window size.
        image = axes.imshow(
            masked,
            cmap=_matplotlib_cmap_name(self.data.cmap),
            origin="lower",
            aspect="equal",
            extent=(0, nx - 1, 0, ny - 1),
        )
        _draw_contours(axes, masked)
        if self.data.high_point is not None:
            point = self.data.high_point
            axes.scatter(
                [point.inline],
                [point.xline],
                marker="+",
                s=120,
                linewidths=2.0,
                color="yellow",
                zorder=5,
            )
            axes.text(
                point.inline,
                point.xline,
                f" 高点 {point.sample:.1f}",
                color="yellow",
                fontsize=9,
                va="bottom",
                ha="left",
                zorder=6,
            )
        axes.set_title(self.data.title)
        axes.set_xlabel("纵向线号")
        axes.set_ylabel("横向线号")
        self._figure.colorbar(image, ax=axes, label=self.data.colorbar_label)
        self._canvas.draw_idle()


def _display_values(values: np.ndarray, finite: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~finite.T, values.T)


def _draw_contours(axes, values: np.ma.MaskedArray) -> None:
    finite = np.asarray(values.compressed(), dtype=np.float32)
    if finite.size < 4:
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return
    ny, nx = values.shape
    x_grid, y_grid = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
    )
    level_count = min(12, max(4, int(np.sqrt(finite.size) // 12)))
    axes.contour(
        x_grid,
        y_grid,
        values,
        levels=level_count,
        colors="black",
        linewidths=0.45,
        alpha=0.45,
    )


def _matplotlib_cmap_name(cmap: str) -> str:
    return {"Petrel": "seismic", "petrel": "seismic"}.get(cmap, cmap)
