from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import numpy as np

from yj_studio.data.remote_target_store import RemoteTargetStore


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _FakeBytesResponse:
    def __init__(self, payload: bytes, headers: dict[str, str]):
        self._payload = payload
        self.headers = headers

    def __enter__(self) -> "_FakeBytesResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_remote_target_store_put_mask_uploads_npy(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["content_type"] = request.headers.get("Content-type") or request.headers.get("Content-Type")
        captured["timeout"] = timeout
        captured["mask"] = np.load(BytesIO(request.data), allow_pickle=False)
        return _FakeResponse({"id": "T1", "type": "trap", "frames": {}})

    monkeypatch.setattr("yj_studio.data.remote_target_store.urlopen", fake_urlopen)
    store = RemoteTargetStore("http://server:8765", project_id="demo", timeout_s=9)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    target = store.put_mask("T1", "inline", 7, mask, volume_id="vol")

    assert target.id == "T1"
    assert captured["method"] == "PUT"
    assert captured["content_type"] == "application/x-npy"
    assert captured["timeout"] == 9
    assert captured["url"] == "http://server:8765/sam3/targets/T1/mask/inline/7?project=demo&volume_id=vol"
    assert np.array_equal(captured["mask"], mask)


def test_remote_target_store_create_cell_target_uploads_npy(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["content_type"] = request.headers.get("Content-type") or request.headers.get("Content-Type")
        captured["timeout"] = timeout
        captured["cells"] = np.load(BytesIO(request.data), allow_pickle=False)
        return _FakeResponse({"id": "T2", "type": "sandbody", "frames": {}})

    monkeypatch.setattr("yj_studio.data.remote_target_store.urlopen", fake_urlopen)
    store = RemoteTargetStore("http://server:8765", project_id="demo", timeout_s=11)
    cells = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    target = store.create_cell_target(
        cells,
        axis="i",
        index=10,
        index_hi=14,
        volume_id="reservoir",
        target_type="sandbody",
        name="body",
        grid_id="grid-a",
        grid_layer_id="layer-a",
    )

    assert target.id == "T2"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/x-npy"
    assert captured["timeout"] == 11
    assert captured["url"] == (
        "http://server:8765/sam3/targets/cells?project=demo&volume_id=reservoir&axis=i&index=10"
        "&index_hi=14&target_type=sandbody&name=body&source=sam3_reservoir&grid_id=grid-a&grid_layer_id=layer-a"
    )
    assert captured["cells"].dtype == np.int32
    assert np.array_equal(captured["cells"], cells.astype(np.int32))


def test_remote_target_store_fetch_mask3d_reads_metadata_headers(monkeypatch) -> None:
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[0, 1, 1] = 1
    mask[1, 2, 3] = 1
    buffer = BytesIO()
    np.save(buffer, mask, allow_pickle=False)

    def fake_urlopen(url, timeout=None):  # noqa: ANN001
        assert url == "http://server:8765/sam3/targets/T1/mask3d?project=demo&volume_id=model_lithology"
        assert timeout == 13
        return _FakeBytesResponse(
            buffer.getvalue(),
            {
                "X-Mask3D-Index-Lo": "10",
                "X-Mask3D-Index-Hi": "11",
                "X-Mask3D-Voxel-Count": "2",
                "X-Mask3D-Volume-M3": "115.740740740741",
                "X-Mask3D-Voxel-Spacing": "4.166666666666667,4.166666666666667,3.3333333333333335",
                "X-Mask3D-Voxel-Spacing-Source": "config",
            },
        )

    monkeypatch.setattr("yj_studio.data.remote_target_store.urlopen", fake_urlopen)
    store = RemoteTargetStore("http://server:8765", project_id="demo", timeout_s=13)

    result = store.fetch_mask3d_with_metadata("T1", volume_id="model_lithology")

    assert np.array_equal(result.mask, mask)
    assert result.index_lo == 10
    assert result.index_hi == 11
    assert result.voxel_count == 2
    assert result.volume_m3 == 115.740740740741
    assert result.voxel_spacing == (12.5 / 3.0, 12.5 / 3.0, 10.0 / 3.0)
    assert result.voxel_spacing_source == "config"
