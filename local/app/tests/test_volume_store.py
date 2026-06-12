from __future__ import annotations

from pathlib import Path

import numpy as np

from yj_studio.data import VolumeStore


def test_volume_store_mmap_and_slices(monkeypatch) -> None:
    path = Path("cube.npy")
    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    def fake_load(load_path, mmap_mode=None):
        assert Path(load_path) == path
        assert mmap_mode == "r"
        return cube

    monkeypatch.setattr(np, "load", fake_load)

    store = VolumeStore(cache_size=1)
    store.register_path("seismic", path, cmap="Petrel")

    assert store.shape("seismic") == (2, 3, 4)
    np.testing.assert_array_equal(store.get_slice("seismic", "inline", 1), cube[1, :, :])
    np.testing.assert_array_equal(store.get_slice("seismic", "xline", 2), cube[:, 2, :])
    np.testing.assert_array_equal(store.get_slice("seismic", "z", 3), cube[:, :, 3])


def test_volume_store_lru_keeps_cache_bounded(monkeypatch) -> None:
    store = VolumeStore(cache_size=1)
    volumes = {
        Path("cube_0.npy"): np.zeros((2, 2, 2), dtype=np.float32),
        Path("cube_1.npy"): np.ones((2, 2, 2), dtype=np.float32),
    }
    load_calls: list[Path] = []

    def fake_load(load_path, mmap_mode=None):
        assert mmap_mode == "r"
        path = Path(load_path)
        load_calls.append(path)
        return volumes[path]

    monkeypatch.setattr(np, "load", fake_load)

    for idx in range(2):
        path = Path(f"cube_{idx}.npy")
        store.register_path(f"v{idx}", path)

    assert store.get_slice("v0", "z", 0).shape == (2, 2)
    assert store.get_slice("v1", "z", 0).shape == (2, 2)
    assert store.shape("v1") == (2, 2, 2)
    assert load_calls == [Path("cube_0.npy"), Path("cube_1.npy")]
