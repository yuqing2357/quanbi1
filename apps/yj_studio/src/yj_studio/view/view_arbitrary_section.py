from __future__ import annotations

from uuid import uuid4

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import ArbitrarySectionLayer


class ViewArbitrarySection(QWidget):
    """Matplotlib-backed arbitrary polyline section view."""

    section_updated = pyqtSignal(str, str, int)

    def __init__(
        self,
        layer_store: LayerStore,
        layer_id: str,
        *,
        section_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.section_id = section_id or str(uuid4())
        self.axis = "arbitrary"
        self.index = 0
        self._layer_store = layer_store
        self._layer_id = layer_id

        self._figure = Figure(figsize=(7, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        layer_store.layer_changed.connect(self._on_layer_changed)
        layer_store.layer_removed.connect(self._on_layer_removed)
        self.refresh()

    @property
    def title(self) -> str:
        layer = self._layer()
        return layer.name if layer is not None else "Arbitrary Section"

    def refresh(self) -> None:
        self._axes.clear()
        layer = self._layer()
        if layer is None:
            self._draw_message("Section layer removed")
            return
        if layer.image is None or layer.image.size == 0:
            self._draw_message("No section image")
            return
        values = np.asarray(layer.image, dtype=np.float32)
        finite = values[np.isfinite(values)]
        vmin, vmax = np.percentile(finite, [2.0, 98.0]) if finite.size else (-1.0, 1.0)
        distances = _axis_values(layer.distances, values.shape[1])
        depths = _axis_values(layer.depths, values.shape[0])
        extent = (
            float(distances[0]),
            float(distances[-1]),
            float(depths[-1]),
            float(depths[0]),
        )
        self._axes.imshow(
            values,
            cmap=_matplotlib_cmap_name(str(layer.metadata.get("cmap", "seismic"))),
            vmin=float(vmin),
            vmax=float(vmax),
            origin="upper",
            aspect="equal",
            extent=extent,
        )
        self._axes.set_title(layer.name)
        self._axes.set_xlabel(layer.axis_label)
        self._axes.set_ylabel("Sample")
        self._canvas.draw()

    def _draw_message(self, message: str) -> None:
        self._axes.clear()
        self._axes.text(0.5, 0.5, message, ha="center", va="center", transform=self._axes.transAxes)
        self._axes.set_axis_off()
        self._canvas.draw()

    def _layer(self) -> ArbitrarySectionLayer | None:
        try:
            layer = self._layer_store.get(self._layer_id)
        except KeyError:
            return None
        if isinstance(layer, ArbitrarySectionLayer):
            return layer
        return None

    def _on_layer_changed(self, layer_id: str, _field: str) -> None:
        if layer_id == self._layer_id:
            self.refresh()
            self.section_updated.emit(self.section_id, self.title, self.index)

    def _on_layer_removed(self, layer_id: str) -> None:
        if layer_id == self._layer_id:
            self.refresh()


def _axis_values(values: np.ndarray | None, count: int) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=np.float32)
    axis = np.asarray(values, dtype=np.float32)
    if axis.ndim != 1 or axis.size != count:
        return np.arange(count, dtype=np.float32)
    return axis


def _matplotlib_cmap_name(cmap: str) -> str:
    name = {"Petrel": "seismic", "petrel": "seismic"}.get(cmap, cmap)
    return name if name in colormaps else "seismic"
