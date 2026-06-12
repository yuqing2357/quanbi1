from __future__ import annotations

from yj_studio.data.volume_store import VolumeStore
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import (
    ArbitrarySectionLayer,
    FaultSurfaceLayer,
    FaultStickLayer,
    HorizonLayer,
    HorizonStickLayer,
    LithBodyLayer,
    MaskLayer,
    MeasurementLayer,
    PolygonLayer,
    ReservoirGridLayer,
    ReservoirPropertyLayer,
    ReservoirSelectionLayer,
    TrapLayer,
    VolumeLayer,
    WellLayer,
    WellLogLayer,
)

from .renderers.fault_renderer import FaultRenderer
from .renderers.horizon_renderer import HorizonRenderer
from .renderers.mask_renderer import MaskRenderer
from .renderers.manual_geometry_renderer import ManualGeometryRenderer
from .renderers.lith_body_renderer import LithBodyRenderer
from .renderers.reservoir_grid_renderer import ReservoirGridRenderer
from .renderers.reservoir_selection_renderer import ReservoirSelectionRenderer
from .renderers.volume_slice_renderer import VolumeSliceRenderer
from .renderers.well_log_renderer import WellLogRenderer
from .renderers.well_renderer import WellRenderer
from .highlight import is_layer_highlighted, selected_well_names


class SceneController:
    """Translate scene Layer changes into viewport renderer updates."""

    def __init__(
        self,
        layer_store: LayerStore,
        volume_store: VolumeStore,
        view_3d,
        *,
        reservoir_registry=None,
    ) -> None:
        self._layer_store = layer_store
        self._volume_renderer = VolumeSliceRenderer(view_3d, volume_store)
        self._horizon_renderer = HorizonRenderer(view_3d)
        self._fault_renderer = FaultRenderer(view_3d)
        self._lith_body_renderer = LithBodyRenderer(view_3d)
        self._mask_renderer = MaskRenderer(view_3d)
        self._manual_geometry_renderer = ManualGeometryRenderer(view_3d)
        self._well_renderer = WellRenderer(view_3d)
        self._well_log_renderer = WellLogRenderer(view_3d)
        self._reservoir_grid_renderer = (
            ReservoirGridRenderer(view_3d, reservoir_registry)
            if reservoir_registry is not None
            else None
        )
        self._reservoir_selection_renderer = (
            ReservoirSelectionRenderer(
                view_3d,
                reservoir_registry,
                self._reservoir_grid_renderer,
            )
            if self._reservoir_grid_renderer is not None and reservoir_registry is not None
            else None
        )
        self._reservoir_property_parent_ids: dict[str, str] = {}
        self._active_volume_layer_id: str | None = None
        self._active_z_count: int | None = None
        self._selected_layer_ids: set[str] = set()
        self._selected_well_names: set[str] = set()

        layer_store.layer_added.connect(self._on_layer_added)
        layer_store.layer_changed.connect(self._on_layer_changed)
        layer_store.layer_removed.connect(self._on_layer_removed)
        layer_store.selection_changed.connect(self._on_selection_changed)

    def _on_layer_added(self, layer_id: str) -> None:
        layer = self._layer_store.get(layer_id)
        if isinstance(layer, VolumeLayer) and layer.visible:
            self._set_active_volume_layer(layer_id, layer)
        elif isinstance(layer, HorizonLayer) and layer.visible:
            self._horizon_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
        elif isinstance(layer, FaultSurfaceLayer) and layer.visible:
            self._fault_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
        elif isinstance(layer, LithBodyLayer) and layer.visible:
            self._lith_body_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
        elif isinstance(layer, MaskLayer) and layer.visible:
            self._mask_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
        elif isinstance(layer, (ArbitrarySectionLayer, PolygonLayer, HorizonStickLayer, FaultStickLayer, MeasurementLayer, TrapLayer)) and layer.visible:
            self._manual_geometry_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
        elif isinstance(layer, ReservoirGridLayer):
            self._render_reservoir_grid(layer)
        elif isinstance(layer, ReservoirPropertyLayer):
            self._reservoir_property_parent_ids[layer.id] = layer.grid_layer_id
            self._render_reservoir_parent_for_property(layer)
        elif isinstance(layer, ReservoirSelectionLayer):
            self._render_reservoir_selection(layer)
        elif isinstance(layer, WellLayer) and layer.visible:
            self._well_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
        elif isinstance(layer, WellLogLayer) and layer.visible:
            self._well_log_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)

    def _on_layer_changed(self, layer_id: str, field: str) -> None:
        layer = self._layer_store.get(layer_id)
        if isinstance(layer, VolumeLayer):
            if layer.visible:
                self._set_active_volume_layer(layer_id, layer)
            elif layer_id == self._active_volume_layer_id:
                self._active_volume_layer_id = None
                self._active_z_count = None
                self._volume_renderer.clear()
            return
        if isinstance(layer, HorizonLayer):
            self._horizon_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return
        if isinstance(layer, FaultSurfaceLayer):
            self._fault_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return
        if isinstance(layer, LithBodyLayer):
            self._lith_body_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return
        if isinstance(layer, MaskLayer):
            self._mask_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return
        if isinstance(layer, (ArbitrarySectionLayer, PolygonLayer, HorizonStickLayer, FaultStickLayer, MeasurementLayer, TrapLayer)):
            self._manual_geometry_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return
        if isinstance(layer, ReservoirGridLayer):
            self._render_reservoir_grid(layer)
            return
        if isinstance(layer, ReservoirPropertyLayer):
            self._reservoir_property_parent_ids[layer.id] = layer.grid_layer_id
            self._render_reservoir_parent_for_property(layer)
            return
        if isinstance(layer, ReservoirSelectionLayer):
            self._render_reservoir_selection(layer)
            return
        if isinstance(layer, WellLayer):
            self._well_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return
        if isinstance(layer, WellLogLayer):
            self._well_log_renderer.render(layer, highlighted=self._highlighted(layer), z_count=self._active_z_count)
            return

    def _on_layer_removed(self, layer_id: str) -> None:
        if layer_id == self._active_volume_layer_id:
            self._active_volume_layer_id = None
            self._active_z_count = None
            self._volume_renderer.clear()
        self._horizon_renderer.clear(layer_id)
        self._fault_renderer.clear(layer_id)
        self._lith_body_renderer.clear(layer_id)
        self._mask_renderer.clear(layer_id)
        self._manual_geometry_renderer.clear(layer_id)
        if self._reservoir_grid_renderer is not None:
            self._reservoir_grid_renderer.clear(layer_id)
        if self._reservoir_selection_renderer is not None:
            self._reservoir_selection_renderer.clear(layer_id)
        parent_id = self._reservoir_property_parent_ids.pop(layer_id, None)
        if parent_id:
            try:
                parent = self._layer_store.get(parent_id)
            except KeyError:
                parent = None
            if isinstance(parent, ReservoirGridLayer):
                self._render_reservoir_grid(parent)
        self._well_renderer.clear(layer_id)
        self._well_log_renderer.clear(layer_id)

    def _on_selection_changed(self, layer_ids: list[str]) -> None:
        self._selected_layer_ids = set(layer_ids)
        self._selected_well_names = selected_well_names(self._layer_store)
        for layer in self._layer_store.iter_layers():
            if isinstance(
                layer,
                (
                    HorizonLayer,
                    FaultSurfaceLayer,
                    LithBodyLayer,
                    MaskLayer,
                    ArbitrarySectionLayer,
                    PolygonLayer,
                    TrapLayer,
                    ReservoirGridLayer,
                    ReservoirSelectionLayer,
                    HorizonStickLayer,
                    FaultStickLayer,
                    MeasurementLayer,
                    WellLayer,
                    WellLogLayer,
                ),
            ):
                if layer.visible:
                    self._on_layer_changed(layer.id, "selection")

    def _highlighted(self, layer) -> bool:
        return is_layer_highlighted(layer, self._selected_layer_ids, self._selected_well_names)

    def _set_active_volume_layer(self, layer_id: str, layer: VolumeLayer) -> None:
        self._active_volume_layer_id = layer_id
        self._active_z_count = layer.shape[2] if layer.shape is not None else None
        self._volume_renderer.set_layer(layer)

    def _render_reservoir_grid(self, layer: ReservoirGridLayer) -> None:
        if self._reservoir_grid_renderer is None:
            return
        self._reservoir_grid_renderer.render(
            layer,
            list(self._layer_store.iter_by_type(ReservoirPropertyLayer)),
            highlighted=self._highlighted(layer),
            z_count=self._active_z_count,
        )

    def _render_reservoir_parent_for_property(self, layer: ReservoirPropertyLayer) -> None:
        if self._reservoir_grid_renderer is None:
            return
        parent = None
        if layer.grid_layer_id:
            try:
                candidate = self._layer_store.get(layer.grid_layer_id)
            except KeyError:
                candidate = None
            if isinstance(candidate, ReservoirGridLayer):
                parent = candidate
        if parent is None:
            for candidate in self._layer_store.iter_by_type(ReservoirGridLayer):
                if candidate.grid_id == layer.grid_id:
                    parent = candidate
                    break
        if parent is not None:
            self._render_reservoir_grid(parent)

    def _render_reservoir_selection(self, layer: ReservoirSelectionLayer) -> None:
        if self._reservoir_selection_renderer is None:
            return
        self._reservoir_selection_renderer.render(
            layer,
            highlighted=self._highlighted(layer),
            z_count=self._active_z_count,
        )
