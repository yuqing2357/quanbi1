"""Per-frame manual correction in the 2D target dialog: brush edit + delete."""

from __future__ import annotations

import numpy as np

from yj_studio.ui.dialogs.target_visualization_dialog import (
    TargetTrack2DDialog,
    TrackingFrameView,
)
from yj_studio_core.targets import GeoTarget, TargetFrame


def _frames(n: int = 3) -> list[TrackingFrameView]:
    views = []
    for i in range(n):
        frame = TargetFrame(axis="inline", index=10 + i, area_px=4, origin="propagated")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:4, 2:4] = True
        views.append(TrackingFrameView(frame=frame, image=image, mask=mask))
    return views


class _FakeStore:
    def __init__(self) -> None:
        self.put_calls: list[tuple] = []
        self.deleted: list[tuple] = []

    def put_mask(self, target_id, axis, index, mask, *, volume_id=None, stage=None):  # noqa: ANN001
        self.put_calls.append((target_id, axis, index, np.asarray(mask).copy(), stage))
        return GeoTarget(id=target_id)

    def delete_frame(self, target_id, axis, index, *, volume_id=None, stage=None):  # noqa: ANN001
        self.deleted.append((target_id, axis, index, stage))
        return GeoTarget(id=target_id)


def test_brush_edit_saves_corrected_mask(qapp) -> None:
    store = _FakeStore()
    changed: list[int] = []
    target = GeoTarget(id="TMP1", type="trap", volume_id="vol")
    dialog = TargetTrack2DDialog(
        target,
        _frames(),
        target_store=store,
        stage="temporary",
        on_changed=lambda: changed.append(1),
    )
    dialog._slider.setValue(1)
    dialog._edit_check.setChecked(True)
    # Paint an add-brush stamp at (5, 6) with radius 2.
    dialog._brush_spin.setValue(2)

    class _Evt:
        inaxes = dialog._axes
        xdata = 5.0
        ydata = 6.0

    dialog._on_canvas_press(_Evt())
    dialog._on_canvas_release(_Evt())
    assert 1 in dialog._edited  # current position has an unsaved edit
    assert dialog._edited[1][6, 5]  # brush painted the target pixel

    dialog._save_current_frame()
    assert len(store.put_calls) == 1
    tid, axis, index, saved_mask, stage = store.put_calls[0]
    assert (tid, axis, index, stage) == ("TMP1", "inline", 11, "temporary")
    assert saved_mask[6, 5] == 1
    assert 1 not in dialog._edited  # edit baked in after save
    assert changed  # on_changed fired
    dialog.close()


def test_erase_brush_removes_mask_pixels(qapp) -> None:
    store = _FakeStore()
    target = GeoTarget(id="TMP1", type="trap", volume_id="vol")
    dialog = TargetTrack2DDialog(target, _frames(), target_store=store, stage="temporary")
    dialog._slider.setValue(0)
    dialog._edit_check.setChecked(True)
    # Switch to erase mode and wipe the seeded mask pixel at (2, 2).
    dialog._erase_radio.setChecked(True)
    assert dialog._brush_add is False
    dialog._brush_spin.setValue(2)

    class _Evt:
        inaxes = dialog._axes
        xdata = 2.0
        ydata = 2.0

    dialog._on_canvas_press(_Evt())
    dialog._on_canvas_release(_Evt())
    assert dialog._edited[0][2, 2] == False  # noqa: E712 - explicit erase
    dialog._save_current_frame()
    _, _, _, saved_mask, _ = store.put_calls[0]
    assert saved_mask[2, 2] == 0  # erased pixel persisted as 0
    dialog.close()


def test_delete_frame_drops_view_and_calls_store(qapp) -> None:
    store = _FakeStore()
    target = GeoTarget(id="TMP1", type="trap", volume_id="vol")
    dialog = TargetTrack2DDialog(target, _frames(3), target_store=store, stage="temporary")
    dialog._slider.setValue(1)

    # Bypass the confirmation dialog by deleting directly through the handler's core.
    row = dialog._frames[1]
    store.delete_frame(target.id, row.frame.axis, int(row.frame.index), stage=dialog._stage)
    del dialog._frames[1]
    dialog._slider.setRange(0, max(0, len(dialog._frames) - 1))

    assert store.deleted == [("TMP1", "inline", 11, "temporary")]
    assert len(dialog._frames) == 2
    assert [v.frame.index for v in dialog._frames] == [10, 12]
    dialog.close()


def test_read_only_dialog_without_store_has_no_edit_controls(qapp) -> None:
    dialog = TargetTrack2DDialog(GeoTarget(id="T1"), _frames(2))
    assert dialog._editable is False
    assert not hasattr(dialog, "_edit_check")
    dialog.close()


def _localized_frames() -> list[TrackingFrameView]:
    """A large slice with a small, off-centre target so the crop is a true
    sub-region (not the whole image)."""
    views = []
    for i in range(2):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=bool)
        mask[90:110, 140:170] = True
        frame = TargetFrame(axis="inline", index=50 + i, area_px=600, origin="propagated")
        views.append(TrackingFrameView(frame=frame, image=image, mask=mask))
    return views


def test_global_view_keeps_full_slice_with_zoom_inset(qapp) -> None:
    dialog = TargetTrack2DDialog(GeoTarget(id="SAV1", volume_id="vol"), _localized_frames())
    dialog._draw_frame(0)
    # Main axes spans the entire slice -> global position is preserved.
    x0, x1 = dialog._axes.get_xlim()
    assert (round(x0), round(x1)) == (0, 299)
    # A zoom inset exists and is bounded to the mask neighbourhood, not the slice.
    assert dialog._inset is not None
    ix0, ix1 = dialog._inset.get_xlim()
    assert 0 < ix0 < ix1 < 299
    iy_hi, iy_lo = dialog._inset.get_ylim()  # inverted y (origin upper)
    assert iy_lo < iy_hi
    dialog.close()


def test_edit_mode_hides_global_view_then_save_restores_it(qapp) -> None:
    store = _FakeStore()
    dialog = TargetTrack2DDialog(
        GeoTarget(id="TMP1", type="trap", volume_id="vol"),
        _localized_frames(),
        target_store=store,
        stage="temporary",
    )
    dialog._draw_frame(0)
    # View mode: global slice + inset zoom.
    assert dialog._inset is not None
    assert (round(dialog._axes.get_xlim()[0]), round(dialog._axes.get_xlim()[1])) == (0, 299)

    # Enter edit mode: only the local zoom region, no global view / inset.
    dialog._edit_check.setChecked(True)
    assert dialog._inset is None
    ex0, ex1 = dialog._axes.get_xlim()
    assert 0 < ex0 < ex1 < 299  # axes is zoomed to the mask neighbourhood

    # Make + save an edit; saving exits edit mode and restores global+inset.
    dialog._brush_spin.setValue(2)

    class _Evt:
        inaxes = dialog._axes
        xdata = 150.0
        ydata = 100.0

    dialog._on_canvas_press(_Evt())
    dialog._on_canvas_release(_Evt())
    dialog._save_current_frame()

    assert store.put_calls  # persisted
    assert dialog._edit_check.isChecked() is False  # auto-exited edit mode
    assert dialog._inset is not None  # back to global + inset view
    assert (round(dialog._axes.get_xlim()[0]), round(dialog._axes.get_xlim()[1])) == (0, 299)
    dialog.close()


def test_no_inset_when_target_has_no_mask(qapp) -> None:
    frame = TargetFrame(axis="inline", index=5, area_px=0, origin="missing")
    view = TrackingFrameView(
        frame=frame,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=bool),
    )
    dialog = TargetTrack2DDialog(GeoTarget(id="SAV2", volume_id="vol"), [view])
    dialog._draw_frame(0)
    assert dialog._inset is None  # nothing to zoom into; global view only
    dialog.close()
