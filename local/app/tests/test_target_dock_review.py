from __future__ import annotations

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
