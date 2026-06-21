"""Stage-aware target store: isolation, promotion, renumber, clear.

These exercise the shared ``yj_studio_core.targets`` store layer that backs the
four-stage pipeline (temporary -> saved -> training). They run anywhere (no Qt,
no server) since the store is pure filesystem + pydantic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yj_studio_core.targets import (
    STAGE_PREFIX,
    GeoTarget,
    TargetStage,
    TargetStore,
    relocate_target,
)


def _mask() -> np.ndarray:
    arr = np.zeros((4, 4), dtype=bool)
    arr[1:3, 1:3] = True
    return arr


def _add_target(store: TargetStore, *, axis: str = "inline", indices=(10, 11)) -> str:
    with store.mutate() as ts:
        tid = ts.new_id()
        target = GeoTarget(id=tid, type="trap")
        for index in indices:
            frame = store.frame_from_mask(target_id=tid, axis=axis, index=index, mask=_mask())
            target.add_frame(frame)
        ts.add_target(target)
    return tid


def test_stage_stores_are_isolated_with_distinct_prefixes(tmp_path: Path) -> None:
    temp = TargetStore(tmp_path, project="p", stage=TargetStage.TEMPORARY)
    saved = TargetStore(tmp_path, project="p", stage="saved")
    training = TargetStore(tmp_path, project="p", stage="training")

    assert temp.id_prefix == STAGE_PREFIX[TargetStage.TEMPORARY] == "TMP"
    assert saved.id_prefix == "SAV"
    assert training.id_prefix == "TRN"
    # Physically separate subdirectories.
    assert temp.project_root.name == "temp"
    assert saved.project_root.name == "saved"
    assert training.project_root.name == "training"

    t_id = _add_target(temp)
    s_id = _add_target(saved)
    assert t_id == "TMP1" and s_id == "SAV1"
    # Each stage numbers independently and does not see the other's targets.
    assert list(temp.load().targets) == ["TMP1"]
    assert list(saved.load().targets) == ["SAV1"]
    assert list(training.load().targets) == []


def test_promote_temp_to_saved_moves_files(tmp_path: Path) -> None:
    temp = TargetStore(tmp_path, project="p", stage="temp")
    saved = TargetStore(tmp_path, project="p", stage="saved")
    old_id = _add_target(temp)

    with saved.mutate() as saved_set, temp.mutate() as temp_set:
        new_id = saved_set.new_id()
        moved = relocate_target(temp, saved, temp_set.targets[old_id], new_id=new_id, move=True)
        saved_set.add_target(moved)
        temp_set.remove_target(old_id)

    assert new_id == "SAV1"
    # Moved into saved, with valid rewritten refs; gone from temp.
    saved_target = saved.load().targets["SAV1"]
    assert all(saved.resolve_ref(f.mask_ref).exists() for f in saved_target.frames.values())
    assert "TMP1" not in temp.load().targets
    assert not temp.target_mask_dir(old_id).exists()


def test_promote_saved_to_training_copies_files(tmp_path: Path) -> None:
    saved = TargetStore(tmp_path, project="p", stage="saved")
    training = TargetStore(tmp_path, project="p", stage="training")
    old_id = _add_target(saved)

    with training.mutate() as train_set, saved.mutate() as saved_set:
        new_id = train_set.new_id()
        copied = relocate_target(saved, training, saved_set.targets[old_id], new_id=new_id, move=False)
        train_set.add_target(copied)

    assert new_id == "TRN1"
    # Copied into training but still present in saved (long-term pool).
    assert "TRN1" in training.load().targets
    assert old_id in saved.load().targets
    assert saved.target_mask_dir(old_id).exists()
    train_target = training.load().targets["TRN1"]
    assert all(training.resolve_ref(f.mask_ref).exists() for f in train_target.frames.values())


def test_renumber_packs_ids_and_preserves_masks(tmp_path: Path) -> None:
    temp = TargetStore(tmp_path, project="p", stage="temp")
    for _ in range(3):
        _add_target(temp)
    # Drop the middle target to create a gap (TMP1, TMP3 remain).
    with temp.mutate() as ts:
        ts.remove_target("TMP2")
    import shutil

    shutil.rmtree(temp.target_mask_dir("TMP2"), ignore_errors=True)

    renumbered = temp.renumber()
    assert sorted(renumbered.targets) == ["TMP1", "TMP2"]
    assert renumbered.next_seq == 3
    for tid in renumbered.targets:
        target = renumbered.targets[tid]
        assert target.frames
        assert all(temp.resolve_ref(f.mask_ref).exists() for f in target.frames.values())


def test_clear_empties_stage(tmp_path: Path) -> None:
    temp = TargetStore(tmp_path, project="p", stage="temp")
    _add_target(temp)
    assert temp.masks_dir.exists() and any(temp.masks_dir.iterdir())

    temp.clear()
    assert list(temp.load().targets) == []
    assert not any(temp.masks_dir.iterdir())


def test_loading_legacy_prefix_advances_stage_sequence(tmp_path: Path) -> None:
    # A set written before id_prefix existed (default "T") opened as a saved
    # store should re-key to SAV and not collide with the old T ids.
    legacy = TargetStore(tmp_path, project="p")  # no stage -> flat/legacy
    _add_target(legacy)  # writes T1 at <root>/p/
    saved = TargetStore(tmp_path, project="p", stage="saved")
    new_id = saved.load().new_id()
    assert new_id == "SAV1"
