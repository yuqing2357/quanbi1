from __future__ import annotations

from pathlib import Path

import numpy as np

from yj_studio.data.hybrid_volume_store import HybridVolumeStore, resolve_local_volume_path
from yj_studio.io.readers.volume_npy import VolumeSpec


class _FakeRemote:
    def __init__(self, specs: dict[str, VolumeSpec], info: dict[str, dict]):
        self._specs = specs
        self._info = info
        self.slice_calls: list[tuple] = []
        self.registered: list[str] = []

    def discover_specs(self):
        return dict(self._specs), []

    def register(self, spec: VolumeSpec) -> None:
        self.registered.append(spec.key)

    def info(self, volume_id: str) -> dict:
        return dict(self._info[volume_id])

    def get_slice(self, volume_id, axis, index):
        self.slice_calls.append((volume_id, axis, index))
        return np.full((2, 2), 7, dtype=np.float32)

    def get_volume(self, volume_id):
        return object()

    def shape(self, volume_id):
        return (3, 3, 3)

    def clear(self) -> None:
        pass


def _write_volume(path: Path, shape=(4, 4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape))


def test_resolve_maps_server_path_onto_local_root(tmp_path: Path) -> None:
    local_root = tmp_path / "data"
    target = local_root / "reservoir" / "npy_625x625x2_v3" / "porosity_float16.npy"
    _write_volume(target)
    server_path = "/root/quanbi/data/reservoir/npy_625x625x2_v3/porosity_float16.npy"
    resolved = resolve_local_volume_path(server_path, local_root)
    assert resolved == target


def test_resolve_returns_none_when_absent(tmp_path: Path) -> None:
    assert resolve_local_volume_path("/root/quanbi/data/seismic/huge.npy", tmp_path) is None


def test_hybrid_routes_present_volume_to_local_and_missing_to_remote(tmp_path: Path) -> None:
    local_root = tmp_path / "data"
    por_path = local_root / "reservoir" / "porosity_float16.npy"
    _write_volume(por_path, shape=(5, 6, 7))

    specs = {
        "model_porosity": VolumeSpec(
            key="model_porosity",
            path=Path("/root/quanbi/data/reservoir/porosity_float16.npy"),
            label="por",
            cmap="viridis",
            filename="porosity_float16.npy",
        ),
        "seismic": VolumeSpec(
            key="seismic",
            path=Path("/root/quanbi/data/seismic/YJ-ALL-SEISMIC.npy"),
            label="seis",
            cmap="gray",
            filename="YJ-ALL-SEISMIC.npy",
        ),
    }
    info = {"model_porosity": {"shape": [5, 6, 7]}, "seismic": {"shape": [9, 9, 9]}}
    remote = _FakeRemote(specs, info)

    store = HybridVolumeStore(remote, local_root)
    merged, notes = store.discover_specs()

    assert store._owner["model_porosity"] == "local"
    assert store._owner["seismic"] == "remote"
    # Local volume points at the on-disk copy, not the server path.
    assert merged["model_porosity"].path == por_path

    # Local-backed slice comes from the mmap, never touches the remote.
    local_slice = store.get_slice("model_porosity", "inline", 0)
    assert local_slice.shape == (6, 7)
    assert remote.slice_calls == []

    # Missing volume falls through to remote streaming.
    remote_slice = store.get_slice("seismic", "inline", 0)
    assert remote_slice.shape == (2, 2)
    assert remote.slice_calls == [("seismic", "inline", 0)]

    # Catalogue metadata always comes from the server.
    assert store.info("model_porosity") == {"shape": [5, 6, 7]}
    assert any("本地体数据" in note for note in notes)
