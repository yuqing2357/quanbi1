from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import TypeVar

from PyQt6.QtCore import QObject, pyqtSignal

from .layer import Layer

LayerT = TypeVar("LayerT", bound=Layer)


class LayerStore(QObject):
    layer_added = pyqtSignal(str)
    layer_removed = pyqtSignal(str)
    layer_changed = pyqtSignal(str, str)
    selection_changed = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self._layers: OrderedDict[str, Layer] = OrderedDict()
        self._selection: list[str] = []

    def add(self, layer: Layer) -> str:
        self._layers[layer.id] = layer
        self.layer_added.emit(layer.id)
        return layer.id

    def remove(self, layer_id: str) -> Layer:
        layer = self._layers.pop(layer_id)
        if layer_id in self._selection:
            self._selection = [item for item in self._selection if item != layer_id]
            self.selection_changed.emit(list(self._selection))
        self.layer_removed.emit(layer_id)
        return layer

    def get(self, layer_id: str) -> Layer:
        return self._layers[layer_id]

    def update(self, layer_id: str, **fields: object) -> None:
        layer = self.get(layer_id)
        for field, value in fields.items():
            if not hasattr(layer, field):
                raise AttributeError(f"{layer.__class__.__name__} has no field {field!r}")
            setattr(layer, field, value)
            self.layer_changed.emit(layer_id, field)

    def iter_layers(self) -> Iterator[Layer]:
        return iter(self._layers.values())

    def iter_by_type(self, layer_cls: type[LayerT]) -> Iterator[LayerT]:
        for layer in self._layers.values():
            if isinstance(layer, layer_cls):
                yield layer

    def select(self, layer_ids: list[str]) -> None:
        missing = [layer_id for layer_id in layer_ids if layer_id not in self._layers]
        if missing:
            raise KeyError(missing[0])
        self._selection = list(layer_ids)
        self.selection_changed.emit(list(self._selection))

    @property
    def selection(self) -> tuple[str, ...]:
        return tuple(self._selection)

    def __len__(self) -> int:
        return len(self._layers)

