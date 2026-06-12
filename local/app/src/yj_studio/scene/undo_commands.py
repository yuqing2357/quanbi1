from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtGui import QUndoCommand

from .layer import Layer
from .layer_store import LayerStore


@dataclass(slots=True)
class LayerFieldChange:
    """Lightweight, framework-agnostic record of a single field edit.

    Kept for tests / non-Qt callers; new UI work should prefer the QUndoCommand
    subclasses below so changes integrate with the application undo stack.
    """

    store: LayerStore
    layer_id: str
    field: str
    old_value: object
    new_value: object

    def redo(self) -> None:
        self.store.update(self.layer_id, **{self.field: self.new_value})

    def undo(self) -> None:
        self.store.update(self.layer_id, **{self.field: self.old_value})


class _BaseLayerCommand(QUndoCommand):
    def __init__(self, store: LayerStore, text: str) -> None:
        super().__init__(text)
        self._store = store


class SetLayerFieldCommand(_BaseLayerCommand):
    """Generic single-field setter: visibility, color, opacity, name, locked, etc."""

    def __init__(
        self,
        store: LayerStore,
        layer_id: str,
        field: str,
        new_value: Any,
        *,
        text: str | None = None,
    ) -> None:
        super().__init__(store, text or f"Edit {field}")
        layer = store.get(layer_id)
        if not hasattr(layer, field):
            raise AttributeError(f"{layer.__class__.__name__} has no field {field!r}")
        self._layer_id = layer_id
        self._field = field
        self._old_value = getattr(layer, field)
        self._new_value = new_value

    def redo(self) -> None:
        self._store.update(self._layer_id, **{self._field: self._new_value})

    def undo(self) -> None:
        self._store.update(self._layer_id, **{self._field: self._old_value})

    # Merge consecutive edits to the same field on the same layer so dragging a
    # slider produces a single undo step.
    def id(self) -> int:  # noqa: A003 - Qt API name
        return hash((self._layer_id, self._field)) & 0x7FFFFFFF

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802 - Qt API name
        if not isinstance(other, SetLayerFieldCommand):
            return False
        if other._layer_id != self._layer_id or other._field != self._field:
            return False
        self._new_value = other._new_value
        return True


class RenameLayerCommand(SetLayerFieldCommand):
    def __init__(self, store: LayerStore, layer_id: str, new_name: str) -> None:
        super().__init__(store, layer_id, "name", new_name, text=f"Rename to '{new_name}'")


class SetColorCommand(SetLayerFieldCommand):
    def __init__(self, store: LayerStore, layer_id: str, color: tuple[float, float, float, float]) -> None:
        super().__init__(store, layer_id, "color", color, text="Change color")


class SetOpacityCommand(SetLayerFieldCommand):
    def __init__(self, store: LayerStore, layer_id: str, opacity: float) -> None:
        super().__init__(store, layer_id, "opacity", float(opacity), text="Change opacity")


class SetVisibleCommand(SetLayerFieldCommand):
    def __init__(self, store: LayerStore, layer_id: str, visible: bool) -> None:
        super().__init__(store, layer_id, "visible", bool(visible), text="Toggle visibility")


class AddLayerCommand(_BaseLayerCommand):
    def __init__(self, store: LayerStore, layer: Layer) -> None:
        super().__init__(store, f"Add {layer.name}")
        self._layer = layer
        self._layer_id: str | None = None

    @property
    def layer_id(self) -> str | None:
        return self._layer_id

    def redo(self) -> None:
        self._layer_id = self._store.add(self._layer)

    def undo(self) -> None:
        if self._layer_id is None:
            return
        self._store.remove(self._layer_id)


class RemoveLayerCommand(_BaseLayerCommand):
    def __init__(self, store: LayerStore, layer_id: str) -> None:
        layer = store.get(layer_id)
        super().__init__(store, f"Remove {layer.name}")
        self._layer_id = layer_id
        self._layer = layer

    def redo(self) -> None:
        self._store.remove(self._layer_id)

    def undo(self) -> None:
        # Re-add with the same id so any references survive the round-trip.
        self._store.add(self._layer)


class MergeLayersCommand(_BaseLayerCommand):
    """Merge several same-kind layers into a new layer produced by ``merger``.

    ``merger`` receives the source layer instances and must return a fresh
    ``Layer`` of the same kind. The original layers are removed on redo and
    restored on undo. The merged layer keeps its id across redo cycles so other
    state (selection, highlight) stays consistent.
    """

    def __init__(
        self,
        store: LayerStore,
        layer_ids: list[str],
        merger,
        *,
        text: str | None = None,
    ) -> None:
        if len(layer_ids) < 2:
            raise ValueError("MergeLayersCommand requires at least two layers")
        sources = [store.get(lid) for lid in layer_ids]
        kinds = {layer.kind for layer in sources}
        if len(kinds) != 1:
            raise ValueError("Cannot merge layers of different kinds")
        super().__init__(store, text or "Merge layers")
        self._source_ids = list(layer_ids)
        self._sources = sources
        self._merged: Layer = merger(sources)
        self._merged_id: str | None = None

    @property
    def merged_layer_id(self) -> str | None:
        return self._merged_id

    def redo(self) -> None:
        for layer_id in self._source_ids:
            self._store.remove(layer_id)
        self._merged_id = self._store.add(self._merged)

    def undo(self) -> None:
        if self._merged_id is not None:
            self._store.remove(self._merged_id)
        for layer in self._sources:
            self._store.add(layer)


class SplitLayerCommand(_BaseLayerCommand):
    """Replace one layer with several produced by ``splitter``."""

    def __init__(
        self,
        store: LayerStore,
        layer_id: str,
        splitter,
        *,
        text: str | None = None,
    ) -> None:
        source = store.get(layer_id)
        super().__init__(store, text or f"Split {source.name}")
        self._source_id = layer_id
        self._source = source
        self._parts: list[Layer] = list(splitter(source))
        if len(self._parts) < 2:
            raise ValueError("SplitLayerCommand requires at least two output layers")
        self._part_ids: list[str] = []

    @property
    def split_layer_ids(self) -> tuple[str, ...]:
        return tuple(self._part_ids)

    def redo(self) -> None:
        self._store.remove(self._source_id)
        self._part_ids = [self._store.add(layer) for layer in self._parts]

    def undo(self) -> None:
        for part_id in self._part_ids:
            self._store.remove(part_id)
        self._part_ids.clear()
        self._store.add(self._source)
