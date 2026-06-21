from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHARED_SRC = ROOT / "shared" / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from yj_studio_core.masks import (  # noqa: E402
    SPARSE_MASK_FORMAT,
    decode_sparse_mask,
    encode_sparse_mask,
    is_sparse_mask_payload,
)


def _roundtrip(mask: np.ndarray) -> np.ndarray:
    payload = encode_sparse_mask(mask)
    assert is_sparse_mask_payload(payload)
    assert payload["format"] == SPARSE_MASK_FORMAT
    assert payload["shape"] == [int(mask.shape[0]), int(mask.shape[1])]
    return decode_sparse_mask(payload)


def test_roundtrip_small_blob() -> None:
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[10:20, 30:45] = 1
    out = _roundtrip(mask)
    assert out.dtype == np.uint8
    assert np.array_equal(out, mask)


def test_roundtrip_all_zero() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    payload = encode_sparse_mask(mask)
    assert payload["bbox"] is None
    out = decode_sparse_mask(payload)
    assert out.shape == (32, 32)
    assert not out.any()


def test_roundtrip_full() -> None:
    mask = np.ones((17, 23), dtype=np.uint8)
    out = _roundtrip(mask)
    assert np.array_equal(out, mask)


def test_bbox_is_tight_and_payload_is_small() -> None:
    mask = np.zeros((2000, 2000), dtype=np.uint8)
    mask[100:130, 200:260] = 1  # tiny target in a huge slice
    payload = encode_sparse_mask(mask)
    assert payload["bbox"] == [100, 200, 130, 260]
    # bit-packed 30x60 crop is well under 1 KB, vs 4 MB dense.
    assert len(payload["packbits"]) < 1024
    assert np.array_equal(decode_sparse_mask(payload), mask)


def test_encode_thresholds_nonbinary() -> None:
    mask = np.array([[0, 2], [0, 0]], dtype=np.uint8)
    out = _roundtrip(mask)
    assert out.tolist() == [[0, 1], [0, 0]]
