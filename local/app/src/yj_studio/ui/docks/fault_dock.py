from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import FaultSurfaceLayer

from .layer_filter_dock import LayerFilterDock


class FaultDock(LayerFilterDock):
    """List fault surface layers."""

    def __init__(self, layer_store: LayerStore, parent: QWidget | None = None) -> None:
        super().__init__(
            "断层",
            layer_store,
            lambda layer: isinstance(layer, FaultSurfaceLayer),
            parent,
        )
