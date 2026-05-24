from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np


def test_core_imports() -> None:
    from yj_studio.algorithms import AlgorithmContext, AlgorithmResult, AlgorithmRunner
    from yj_studio.data import CoordTransform, VolumeStore
    from yj_studio.scene import LayerStore
    from yj_studio.scene.layers import HorizonLayer, TrapLayer, VolumeLayer
    from yj_studio.services import ViewSyncService
    from yj_studio.tools import InteractionTool, ToolManager
    from yj_studio.view import PickResult

    assert CoordTransform
    assert VolumeStore
    assert LayerStore
    assert VolumeLayer
    assert HorizonLayer
    assert TrapLayer
    assert InteractionTool
    assert ToolManager
    assert AlgorithmContext
    assert AlgorithmResult
    assert AlgorithmRunner
    assert PickResult
    assert ViewSyncService


def test_main_window_smoke() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.ui.main_window import MainWindow

    app = create_application([])
    window = MainWindow(auto_load=False, enable_3d=False)
    assert app.applicationName() == "YJ Studio"
    assert "YJ Studio" in window.windowTitle()
    window.close()


def test_open_connected_well_section_uses_layer_store_and_parent(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.scene.layers import VolumeLayer, WellLayer
    from yj_studio.ui import main_window as main_window_module

    app = create_application([])
    window = main_window_module.MainWindow(auto_load=False, enable_3d=False)

    volume_layer = VolumeLayer(name="cube", volume_id="cube", shape=(8, 9, 10))
    window._active_volume_layer_id = window.layer_store.add(volume_layer)
    window.layer_store.add(
        WellLayer(name="W1", well_name="W1", head_position=(1.0, 2.0, 3.0), trajectory=np.zeros((2, 3)))
    )
    window.layer_store.add(
        WellLayer(name="W2", well_name="W2", head_position=(4.0, 5.0, 6.0), trajectory=np.zeros((2, 3)))
    )

    captured: dict[str, object] = {}

    class DummyView:
        def __init__(self, data, layer_store, parent=None) -> None:
            captured["args"] = (data, layer_store, parent)
            self.title = "Dummy Section"

    def fake_build(*args, **kwargs):
        captured["build_args"] = (args, kwargs)
        return SimpleNamespace(names=("W1", "W2"))

    def fake_add_internal_section(self, view, *, title, axis, index):
        captured["section"] = (view, title, axis, index)
        return "section-id"

    monkeypatch.setattr(main_window_module, "ViewWellSection", DummyView)
    monkeypatch.setattr(main_window_module, "build_well_section_data", fake_build)
    monkeypatch.setattr(main_window_module.ViewsArea, "add_internal_section", fake_add_internal_section)

    window._open_connected_well_section(["W1", "W2"], "por")

    assert captured["args"][1] is window.layer_store
    assert captured["args"][2] is window._views_area
    assert captured["section"][1] == "Dummy Section"
    assert captured["section"][2] == "well"
    window.close()
    app.quit()


def test_open_well_adjacent_section_opens_inline_and_xline(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.scene.layers import VolumeLayer, WellLayer
    from yj_studio.ui import main_window as main_window_module

    app = create_application([])
    window = main_window_module.MainWindow(auto_load=False, enable_3d=False)
    volume_layer = VolumeLayer(
        name="cube",
        volume_id="cube",
        shape=(8, 9, 10),
        slice_indices={"inline": 4, "xline": 4, "z": 5},
    )
    window._active_volume_layer_id = window.layer_store.add(volume_layer)
    well = WellLayer(
        name="W1",
        well_name="W1",
        head_position=(3.2, 7.6, 0.0),
        trajectory=np.zeros((2, 3)),
        visible=False,
    )
    window.layer_store.add(well)
    opened: list[tuple[str, int]] = []

    def fake_add_orthogonal_section(self, *, volume_layer_id, axis, index):
        opened.append((axis, index))
        return None

    monkeypatch.setattr(main_window_module.ViewsArea, "add_orthogonal_section", fake_add_orthogonal_section)

    window._open_well_adjacent_section(well)

    assert opened == [("inline", 3), ("xline", 8)]
    assert volume_layer.slice_indices["inline"] == 3
    assert volume_layer.slice_indices["xline"] == 8
    assert well.visible
    window.close()
    app.quit()


def test_well_layers_have_single_visible_entry() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.scene import LayerStore
    from yj_studio.scene.layers import WellLayer, WellLogLayer
    from yj_studio.ui.docks.layer_tree_dock import LayerTreeDock
    from yj_studio.ui.docks.wells_dock import WellsDock

    app = create_application([])
    store = LayerStore()
    wells_dock = WellsDock(store)
    layer_tree = LayerTreeDock(store)
    store.add(WellLayer(name="W1", well_name="W1", head_position=(1.0, 2.0, 3.0)))
    store.add(
        WellLogLayer(
            name="W1 POR",
            well_name="W1",
            mode="por",
            samples=np.zeros((2, 4), dtype=np.float32),
        )
    )

    assert wells_dock.tree.topLevelItemCount() == 1
    assert wells_dock.tree.topLevelItem(0).text(0) == "W1"
    assert layer_tree.tree.topLevelItemCount() == 1
    assert layer_tree.tree.topLevelItem(0).text(1) == "well"
    wells_dock.close()
    layer_tree.close()
    app.quit()


def test_well_display_mode_controls_log_visibility() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.scene.layers import WellLayer, WellLogLayer
    from yj_studio.ui.main_window import MainWindow

    app = create_application([])
    window = MainWindow(auto_load=False, enable_3d=False)
    well = WellLayer(name="W1", well_name="W1", head_position=(1.0, 2.0, 3.0), visible=False)
    por = WellLogLayer(
        name="W1 POR",
        well_name="W1",
        mode="por",
        samples=np.zeros((2, 4), dtype=np.float32),
        visible=False,
    )
    lith = WellLogLayer(
        name="W1 LITH",
        well_name="W1",
        mode="lith",
        samples=np.zeros((2, 4), dtype=np.float32),
        visible=False,
    )
    window.layer_store.add(well)
    window.layer_store.add(por)
    window.layer_store.add(lith)

    window._set_well_display_mode("por")
    window._show_well_layers("W1")
    assert well.visible
    assert por.visible
    assert not lith.visible

    window._set_well_display_mode("lith")
    assert well.visible
    assert not por.visible
    assert lith.visible
    window.close()
    app.quit()


def test_well_section_draws_selected_wells_even_when_hidden() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.scene import LayerStore
    from yj_studio.scene.layers import WellLayer
    from yj_studio.services.well_section_service import WellSectionData, WellSectionWell
    from yj_studio.view.view_well_section import ViewWellSection

    app = create_application([])
    store = LayerStore()
    w1 = WellLayer(name="W1", well_name="W1", visible=False)
    w2 = WellLayer(name="W2", well_name="W2", visible=False)
    store.add(w1)
    store.add(w2)
    data = WellSectionData(
        names=("W1", "W2"),
        mode="por",
        distances=np.asarray([0.0, 10.0], dtype=np.float32),
        depths_m=np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
        seismic=np.zeros((3, 2), dtype=np.float32),
        wells=(
            WellSectionWell("W1", w1.id, 0.0, 1.0, 1.0, ()),
            WellSectionWell("W2", w2.id, 10.0, 2.0, 2.0, ()),
        ),
    )

    view = ViewWellSection(data, store)

    assert len(view._axes.lines) == 2
    xmin, xmax = view._axes.get_xlim()
    assert xmin < 0.0
    assert xmax > 10.0
    view.close()
    app.quit()
