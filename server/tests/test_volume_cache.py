from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERVER_SRC = ROOT / "server" / "src"
if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from yj_studio_server.volume_cache import VolumeCache  # noqa: E402


def _make_volume(tmp_path: Path) -> tuple[Path, dict[str, dict], np.ndarray]:
    data_root = tmp_path / "data"
    (data_root / "seismic").mkdir(parents=True)
    arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    np.save(data_root / "seismic" / "tiny.npy", arr, allow_pickle=False)
    volumes = {"tiny": {"label": "Tiny", "path": "seismic/tiny.npy"}}
    return data_root, volumes, arr


def test_preload_to_ram_marks_ready_and_serves_slices(tmp_path: Path) -> None:
    data_root, volumes, arr = _make_volume(tmp_path)
    cache = VolumeCache(data_root, volumes, preload_to_ram=True)
    cache.preload_all()

    status = cache.status()
    assert status["ready"] == 1
    assert status["ram_resident"] == 1
    assert status["volumes"][0]["state"] == "ready"
    assert status["volumes"][0]["mode"] == "ram"

    resident, mode = cache.get("tiny")
    assert mode == "ram"
    # In-RAM array, not a memmap.
    assert not isinstance(resident, np.memmap)
    np.testing.assert_array_equal(resident[1, :, :], arr[1, :, :])


def test_mmap_mode_holds_a_memmap_handle(tmp_path: Path) -> None:
    data_root, volumes, arr = _make_volume(tmp_path)
    cache = VolumeCache(data_root, volumes, preload_to_ram=False)
    cache.preload_all()

    resident, mode = cache.get("tiny")
    assert mode == "mmap"
    assert isinstance(resident, np.memmap)
    np.testing.assert_array_equal(np.asarray(resident), arr)


def test_get_loads_on_demand_before_preload(tmp_path: Path) -> None:
    data_root, volumes, arr = _make_volume(tmp_path)
    cache = VolumeCache(data_root, volumes, preload_to_ram=True)

    # No preload yet: get() must still return a (mmap) handle.
    resident, mode = cache.get("tiny")
    assert mode == "mmap"
    np.testing.assert_array_equal(np.asarray(resident), arr)

    # A later preload upgrades the same volume to a RAM-resident array.
    cache.preload_all()
    resident, mode = cache.get("tiny")
    assert mode == "ram"


def test_stage_dir_is_preferred_and_marked_shm(tmp_path: Path) -> None:
    data_root, volumes, arr = _make_volume(tmp_path)
    stage_dir = tmp_path / "shm"
    # Stage a copy under the mirrored relative path.
    staged = stage_dir / "seismic" / "tiny.npy"
    staged.parent.mkdir(parents=True)
    np.save(staged, arr, allow_pickle=False)

    cache = VolumeCache(data_root, volumes, preload_to_ram=False, stage_dir=stage_dir)
    cache.preload_all()

    resident, mode = cache.get("tiny")
    assert mode == "shm"
    status = cache.status()
    assert status["shm_resident"] == 1
    assert status["volumes"][0]["path"] == str(staged)
    np.testing.assert_array_equal(np.asarray(resident), arr)


def test_stage_dir_falls_back_to_data_root_when_not_staged(tmp_path: Path) -> None:
    data_root, volumes, arr = _make_volume(tmp_path)
    stage_dir = tmp_path / "shm"  # exists conceptually but nothing staged

    cache = VolumeCache(data_root, volumes, preload_to_ram=False, stage_dir=stage_dir)
    cache.preload_all()

    _resident, mode = cache.get("tiny")
    assert mode == "mmap"  # fell back to the data_root copy


def test_missing_volume_is_marked_not_raised(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    cache = VolumeCache(data_root, {"gone": {"path": "nope.npy"}}, preload_to_ram=True)
    cache.preload_all()  # must not raise

    status = cache.status()
    assert status["ready"] == 0
    assert status["volumes"][0]["state"] == "missing"
