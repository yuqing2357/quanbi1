from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from yj_studio_core.targets import (
    GeoTarget,
    TargetSet,
    TargetStatus,
    TargetStore,
    export_confirmed_to_coco,
    mask_volume_stats,
    resolve_voxel_spacing,
    split_frames,
)


def test_target_store_keeps_arrays_out_of_targets_json(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="tiny")
    target_set = store.load()
    mask = np.zeros((5, 4), dtype=bool)
    mask[1:4, 1:3] = True

    target = store.add_single_frame_target(
        target_set,
        axis="inline",
        index=7,
        mask=mask,
        target_type="sandbody",
        score=0.9,
        volume_id="tiny",
    )
    store.save(target_set)

    loaded = store.load()
    assert loaded.next_seq == 2
    assert list(loaded.targets) == ["T1"]
    assert loaded.targets["T1"].type == "sandbody"
    frame = loaded.targets[target.id].frames["inline:7"]
    assert frame.mask_ref == "masks/T1/inline_7.npy"

    stored_mask = store.read_mask(frame.mask_ref)
    assert stored_mask.shape == (5, 4)
    assert stored_mask.dtype == np.uint8
    assert stored_mask.sum() == 6

    raw = json.loads(store.targets_path.read_text(encoding="utf-8"))
    assert raw["targets"]["T1"]["frames"]["inline:7"]["mask_ref"] == "masks/T1/inline_7.npy"
    assert "mask" not in raw["targets"]["T1"]["frames"]["inline:7"]
    assert store.metadata_is_lightweight()
    assert store.targets_path.stat().st_size < 10_000


def test_export_confirmed_targets_to_coco(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="tiny")
    target_set = store.load()
    target = store.add_single_frame_target(
        target_set,
        axis="inline",
        index=1,
        mask=np.ones((3, 2), dtype=np.uint8),
        target_type="trap",
        volume_id="tiny",
    )
    target.status = TargetStatus.CONFIRMED
    store.save(target_set)

    payload = export_confirmed_to_coco(store, target_set, tmp_path / "export")

    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1
    assert payload["info"]["schema_version"] == 1
    assert payload["images"][0]["split"] == "train"
    assert payload["annotations"][0]["split"] == "train"
    assert payload["annotations"][0]["target_id"] == "T1"
    assert (tmp_path / "export" / "annotations.json").exists()
    assert (tmp_path / "export" / "masks" / "T1_inline_1.png").exists()


def test_export_includes_edited_targets(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="tiny")
    target_set = store.load()
    target = store.add_single_frame_target(
        target_set,
        axis="inline",
        index=2,
        mask=np.ones((2, 2), dtype=np.uint8),
        target_type="fault",
        volume_id="tiny",
    )
    target.edits.append({"kind": "mask_put"})
    store.save(target_set)

    payload = export_confirmed_to_coco(store, target_set, tmp_path / "export")

    assert len(payload["images"]) == 1
    assert payload["annotations"][0]["target_id"] == "T1"


def test_split_frames_spatial_uses_contiguous_index_blocks() -> None:
    splits = split_frames([("inline", index) for index in range(10)], strategy="spatial")

    assert splits[:8] == ["train"] * 8
    assert splits[8] == "val"
    assert splits[9] == "test"


def test_export_confirmed_targets_uses_spatial_split() -> None:
    scratch = Path(__file__).parent / "_scratch" / "target_export_spatial"
    store = TargetStore(scratch / "sam3", project="default", volume_id="tiny")
    target_set = store.load()
    target = GeoTarget(id=target_set.new_id(), type="trap", volume_id="tiny", status=TargetStatus.CONFIRMED)
    for index in range(10):
        target.add_frame(
            store.frame_from_mask(
                target_id=target.id,
                axis="inline",
                index=index,
                mask=np.ones((3, 2), dtype=np.uint8),
            )
        )
    target_set.add_target(target)
    store.save(target_set)

    payload = export_confirmed_to_coco(store, target_set, scratch / "export")
    by_index = {int(image["index"]): image["split"] for image in payload["images"]}

    assert by_index[0] == "train"
    assert by_index[7] == "train"
    assert by_index[8] == "val"
    assert by_index[9] == "test"
    assert payload["info"]["split_strategy"] == "spatial"


def test_target_store_mutate_is_concurrency_safe(tmp_path: Path) -> None:
    """Concurrent writers must not lose targets or reuse ids (review §1.1)."""
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="tiny")
    store.ensure_dirs()

    n_workers = 24
    barrier = threading.Barrier(n_workers)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait()  # release all threads at once to maximise contention
            with store.mutate() as target_set:
                target_id = target_set.new_id()
                target_set.add_target(GeoTarget(id=target_id, type="trap", volume_id="tiny"))
        except Exception as exc:  # noqa: BLE001 - surfaced to the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    loaded = store.load()
    assert len(loaded.targets) == n_workers          # no lost updates
    assert len(set(loaded.targets)) == n_workers      # ids are unique
    assert loaded.next_seq == n_workers + 1


def test_target_store_mask_roundtrip_preserves_orientation(tmp_path: Path) -> None:
    """write_mask → read_mask must not flip/mirror the array (review §1.2)."""
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="tiny")
    mask = np.zeros((4, 6), dtype=bool)
    mask[0, 0] = True   # distinct corners pin down any flip/rotation
    mask[3, 5] = True

    ref = store.write_mask("T1", "inline", 5, mask)
    back = np.asarray(store.read_mask(ref))

    assert back.shape == (4, 6)
    assert bool(back[0, 0]) and bool(back[3, 5])
    assert int(back.sum()) == 2
    assert np.array_equal(back.astype(bool), mask)


def test_target_store_frame_from_cells_roundtrip(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default", volume_id="reservoir")
    cells = np.array([[1, 2, 3], [1, 2, 4], [5, 6, 7]], dtype=np.int64)

    frame = store.frame_from_cells(
        target_id="T1",
        axis="inline",
        index=12,
        cells=cells,
        origin="sam3_reservoir",
    )
    loaded = store.read_cells(frame.cell_ids_ref)

    assert frame.cell_ids_ref == "cells/T1/inline_12.npy"
    assert frame.area_px == 3
    assert loaded.dtype == np.int32
    assert np.array_equal(loaded, cells.astype(np.int32))


def test_target_store_mask3d_uses_real_frame_indices() -> None:
    scratch = Path("tests/_scratch") / f"mask3d_{uuid.uuid4().hex}"
    store = TargetStore(scratch / "sam3", project="default", volume_id="tiny")
    target_set = store.load()
    target = store.add_single_frame_target(
        target_set,
        axis="inline",
        index=5,
        mask=np.ones((2, 3), dtype=bool),
        target_type="trap",
        volume_id="tiny",
    )
    frame = store.frame_from_mask(
        target_id=target.id,
        axis="inline",
        index=7,
        mask=np.full((2, 3), True, dtype=bool),
        origin="propagated",
    )
    target.add_frame(frame)
    store.save(target_set)

    path, index_lo, index_hi = store.write_target_mask3d_cache(store.load().targets[target.id])
    volume = np.load(path, allow_pickle=False)

    assert (index_lo, index_hi) == (5, 7)
    assert volume.shape == (3, 2, 3)
    assert volume[0].sum() == 6
    assert volume[1].sum() == 0
    assert volume[2].sum() == 6
    shutil.rmtree(scratch, ignore_errors=True)


def test_mask_volume_stats_uses_configured_downsample_spacing() -> None:
    spacing, source = resolve_voxel_spacing(
        {"voxel_spacing": [1.0, 2.0, 4.0], "downsample_factor": [3.0, 3.0, 3.0]}
    )
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 2, 3] = 1

    stats = mask_volume_stats(mask, spacing)

    assert spacing == (3.0, 6.0, 12.0)
    assert source == "config+downsample"
    assert stats["voxel_count"] == 2
    assert stats["voxel_volume"] == 216.0
    assert stats["volume_m3"] == 432.0


def test_resolve_voxel_spacing_accepts_reservoir_metadata_axis_keys() -> None:
    spacing, source = resolve_voxel_spacing(
        {
            "voxel_spacing_m": {
                "axis0": 12.5 / 3.0,
                "axis1": 12.5 / 3.0,
                "sample": 10.0 / 3.0,
            }
        }
    )

    assert spacing == (12.5 / 3.0, 12.5 / 3.0, 10.0 / 3.0)
    assert source == "config"


def test_target_store_mutate_skips_save_on_error(tmp_path: Path) -> None:
    """A failing mutate body must not persist partial changes."""
    store = TargetStore(tmp_path / "sam3", project="default")

    with pytest.raises(RuntimeError):
        with store.mutate() as target_set:
            target_set.add_target(GeoTarget(id="T99", type="trap"))
            raise RuntimeError("boom")

    loaded = store.load()
    assert "T99" not in loaded.targets


def test_target_set_schema_is_backward_compatible() -> None:
    target_set = TargetSet.model_validate(
        {
            "project": "legacy",
            "version": 1,
            "unknown_future_field": "kept out of the model",
            "targets": {},
        }
    )

    assert target_set.project == "legacy"
    assert target_set.schema_version == 1
    assert not hasattr(target_set, "unknown_future_field")


def test_metadata_lightweight_detects_large_inline_arrays(tmp_path: Path) -> None:
    store = TargetStore(tmp_path / "sam3", project="default")
    store.ensure_dirs()
    store.targets_path.write_text(
        json.dumps({"targets": {"T1": {"frames": {"inline:1": {"mask": [1] * 2048}}}}}),
        encoding="utf-8",
    )

    assert not store.metadata_is_lightweight()
