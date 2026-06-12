from __future__ import annotations

from yj_studio.ui.docks.target_dock import _model_rows


def test_model_rows_marks_active_and_formats_metrics() -> None:
    rows = _model_rows(
        {
            "active_model": "M2",
            "models": [
                {
                    "id": "M1",
                    "dataset_version": "d0",
                    "status": "ready",
                    "metrics": {"dice": 0.8},
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "id": "M2",
                    "parent_model_id": "M1",
                    "dataset_version": "d1",
                    "status": "ready",
                    "metrics": {"mask_iou": 0.91, "precision": 0.82},
                    "checkpoint": "best.pt",
                    "created_at": "2026-01-02T00:00:00Z",
                },
            ],
        }
    )

    assert rows[0]["id"] == "M2"
    assert rows[0]["active"] is True
    assert rows[0]["parent_model_id"] == "M1"
    assert rows[0]["metrics"] == "mask_iou=0.910, precision=0.820"
    assert rows[1]["id"] == "M1"
