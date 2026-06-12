from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


class DummyWorkbench(QWidget):
    selection_committed = pyqtSignal(object)
    target_committed = pyqtSignal(object)

    instances: list["DummyWorkbench"] = []

    def __init__(
        self,
        grid,
        roi,
        *,
        axis,
        transform,
        ai_service,
        target_store,
        grid_layer_id,
        grid_id,
        parent=None,
        **_kwargs,
    ) -> None:
        super().__init__(parent)
        self.grid = grid
        self.roi = roi
        self.axis_arg = axis
        self.transform = transform
        self.ai_service = ai_service
        self.target_store = target_store
        self.grid_layer_id = grid_layer_id
        self.grid_id = grid_id
        self.index = 11
        DummyWorkbench.instances.append(self)

    @property
    def title(self) -> str:
        return "Dummy SAM3 Workbench"


class DummyReservoirSection(QWidget):
    roi_changed = pyqtSignal(object)
    roi_drawn = pyqtSignal(object, str)

    instances: list["DummyReservoirSection"] = []

    def __init__(
        self,
        grid,
        *,
        axis,
        property_name,
        transform,
        roi,
        parent=None,
        **_kwargs,
    ) -> None:
        super().__init__(parent)
        self.grid = grid
        self.axis = axis
        self.index = 7
        self.property_name = property_name
        self.transform = transform
        self.roi = roi
        DummyReservoirSection.instances.append(self)

    @property
    def title(self) -> str:
        return "Dummy Reservoir Section"

    def set_roi(self, roi) -> None:
        self.roi = roi


def _make_window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from yj_studio.app import create_application
    from yj_studio.ui import main_window as main_window_module

    app = create_application([])
    DummyWorkbench.instances.clear()
    DummyReservoirSection.instances.clear()
    monkeypatch.setattr(main_window_module, "SAM3Workbench", DummyWorkbench)
    monkeypatch.setattr(main_window_module, "ViewReservoirSection", DummyReservoirSection)
    window = main_window_module.MainWindow(auto_load=False, enable_3d=False)
    return app, window


def test_reservoir_section_roi_opens_sam3_workbench(monkeypatch) -> None:
    from yj_studio.scene.layers import ReservoirGridLayer, ReservoirPropertyLayer

    app, window = _make_window(monkeypatch)
    grid = SimpleNamespace(master_path=Path("dummy.GRDECL"))
    window.reservoir_registry._grids["grid-a"] = grid
    grid_layer = ReservoirGridLayer(
        name="Grid",
        id="grid-layer",
        grid_id="grid-a",
        roi=(0, 2, 0, 3, 0, 4),
    )
    prop_layer = ReservoirPropertyLayer(
        name="PORO",
        grid_layer_id="grid-layer",
        grid_id="grid-a",
        property_name="PORO",
        visible=True,
    )
    window.layer_store.add(grid_layer)
    window.layer_store.add(prop_layer)
    window.layer_store.select([prop_layer.id])

    window._open_selected_reservoir_section()
    section = DummyReservoirSection.instances[-1]
    assert section.grid is grid
    assert section.property_name == "PORO"
    assert section.roi == grid_layer.roi

    roi = (0, 2, 1, 3, 0, 4)
    section.roi_drawn.emit(roi, "j")
    workbench = DummyWorkbench.instances[-1]
    assert workbench.grid is grid
    assert workbench.roi == roi
    assert workbench.axis_arg == "j"
    assert workbench.grid_layer_id == "grid-layer"
    assert workbench.grid_id == "grid-a"

    window.close()
    app.quit()


def test_reservoir_workbench_selection_is_added_to_layer_store(monkeypatch) -> None:
    from yj_studio.scene.layers import ReservoirGridLayer, ReservoirSelectionLayer

    app, window = _make_window(monkeypatch)
    grid = SimpleNamespace(master_path=Path("dummy.GRDECL"))
    window.reservoir_registry._grids["grid-a"] = grid
    grid_layer = ReservoirGridLayer(
        name="Grid",
        id="grid-layer",
        grid_id="grid-a",
        roi=(0, 2, 0, 3, 0, 4),
    )
    window.layer_store.add(grid_layer)

    window._open_sam3_workbench_for_reservoir(
        "grid-layer",
        (0, 2, 0, 3, 0, 4),
        "i",
    )
    workbench = DummyWorkbench.instances[-1]
    selection = ReservoirSelectionLayer(
        name="Selection",
        grid_layer_id="grid-layer",
        grid_id="grid-a",
        cell_ids=np.asarray([[0, 1, 2], [0, 1, 2], [1, 1, 2]], dtype=np.int32),
    )

    workbench.selection_committed.emit(selection)

    layers = list(window.layer_store.iter_by_type(ReservoirSelectionLayer))
    assert len(layers) == 1
    assert layers[0].n_cells == 2
    assert window.layer_store.selection == (layers[0].id,)

    window.close()
    app.quit()
