from __future__ import annotations

from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import WellLayer, WellLogLayer
from yj_studio.scene.layer import Layer

HIGHLIGHT_COLOR = (1.0, 0.92, 0.15, 1.0)


def selected_well_names(layer_store: LayerStore) -> set[str]:
    names: set[str] = set()
    for layer_id in layer_store.selection:
        try:
            layer = layer_store.get(layer_id)
        except KeyError:
            continue
        if isinstance(layer, WellLayer):
            names.add(layer.well_name or layer.name)
        elif isinstance(layer, WellLogLayer):
            names.add(layer.well_name)
    return names


def is_layer_highlighted(
    layer: Layer,
    selected_ids: set[str],
    selected_wells: set[str],
) -> bool:
    if layer.id in selected_ids:
        return True
    if isinstance(layer, WellLayer):
        return (layer.well_name or layer.name) in selected_wells
    if isinstance(layer, WellLogLayer):
        return layer.well_name in selected_wells
    return False


def highlight_color(color: tuple[float, float, float, float], highlighted: bool) -> tuple[float, float, float]:
    if highlighted:
        return HIGHLIGHT_COLOR[:3]
    return tuple(float(c) for c in color[:3])


def highlight_opacity(opacity: float, highlighted: bool) -> float:
    if highlighted:
        return max(float(opacity), 0.9)
    return float(opacity)
