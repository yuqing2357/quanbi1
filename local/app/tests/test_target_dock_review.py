from __future__ import annotations

import numpy as np

from yj_studio.data.remote_target_store import Mask3DResult
from yj_studio.scene.layer_store import LayerStore
from yj_studio.scene.layers import MaskLayer
from yj_studio.ui.docks.target_dock import TargetDock
from yj_studio_core.targets import GeoTarget, TargetFrame, TargetSet, TargetStatus
from yj_studio.ui.docks.target_dock import _ReviewQueueDialog, _review_rows


def _target(target_id: str, *, score: float, areas: list[int], status: TargetStatus = TargetStatus.ACTIVE) -> GeoTarget:
    target = GeoTarget(id=target_id, type="sandbody", score=score, status=status)
    for index, area in enumerate(areas):
        target.add_frame(TargetFrame(axis="inline", index=index, area_px=area, score=score))
    return target


def test_review_rows_formats_active_learning_queue() -> None:
    target_set = TargetSet(project="default")
    target_set.add_target(_target("T1", score=0.95, areas=[100, 100]))
    target_set.add_target(_target("T2", score=0.20, areas=[40, 180]))

    rows = _review_rows(target_set)

    assert [row["id"] for row in rows] == ["T2", "T1"]
    assert rows[0]["score"] == "0.200"
    assert rows[0]["uncertainty"] > rows[1]["uncertainty"]


def test_review_dialog_patches_selected_status_and_removes_rows(qapp) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.patches: list[tuple[str, dict[str, str]]] = []

        def patch_target(self, target_id: str, updates: dict[str, str]):
            self.patches.append((target_id, updates))
            return GeoTarget(id=target_id, status=updates["status"])

    store = FakeStore()
    dialog = _ReviewQueueDialog(
        store,  # type: ignore[arg-type]
        [
            {"id": "T1", "type": "trap", "status": "to_review", "frame_range": "inline:1", "area_px": 12, "score": "0.4", "uncertainty": "0.6"},
            {"id": "T2", "type": "fault", "status": "lost", "frame_range": "inline:2", "area_px": 8, "score": "0.3", "uncertainty": "0.7"},
        ],
    )
    dialog._table.selectRow(0)

    dialog._patch_selected(TargetStatus.CONFIRMED.value)

    assert store.patches == [("T1", {"status": "confirmed"})]
    assert dialog._table.rowCount() == 1
    assert dialog._table.item(0, 0).text() == "T2"
    dialog.close()


def test_target_dock_loads_mask3d_as_volume_layer_with_volume_stats() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    target = GeoTarget(id="T1", type="sandbody", volume_id="model_lithology", score=0.7)
    target.add_frame(TargetFrame(axis="inline", index=10, area_px=2, score=0.7))
    target_set = TargetSet(project="default", volume_id="model_lithology")
    target_set.add_target(target)

    class FakeStore:
        def load_targets(self, *, include_deleted: bool = False):  # noqa: ARG002
            return target_set

        def fetch_mask3d_with_metadata(self, target_id: str, *, volume_id: str | None = None):
            assert target_id == "T1"
            assert volume_id == "model_lithology"
            mask = np.zeros((2, 3, 4), dtype=np.uint8)
            mask[0, 0, 0] = 1
            mask[1, 2, 3] = 1
            return Mask3DResult(
                mask=mask,
                index_lo=10,
                index_hi=11,
                voxel_count=2,
                volume_m3=115.74074074074076,
                voxel_spacing=(12.5 / 3.0, 12.5 / 3.0, 10.0 / 3.0),
                voxel_spacing_source="config",
            )

    layer_store = LayerStore()
    dock = TargetDock(layer_store, FakeStore())  # type: ignore[arg-type]
    dock.refresh()

    dock._load_selected_mask3d(target)

    layers = list(layer_store.iter_layers())
    assert len(layers) == 1
    layer = layers[0]
    assert isinstance(layer, MaskLayer)
    assert layer.mask is not None and layer.mask.shape == (2, 3, 4)
    assert layer.metadata["mask3d"] is True
    assert layer.metadata["mask3d_index_lo"] == 10
    assert layer.metadata["voxel_count"] == 2
    assert layer.metadata["voxel_spacing"] == [12.5 / 3.0, 12.5 / 3.0, 10.0 / 3.0]
    assert layer.metadata["volume_m3"] == 115.74074074074076
    assert dock._table.item(0, 5).text() == "116 m3"
    dock.close()
    app.processEvents()
