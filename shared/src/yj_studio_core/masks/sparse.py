"""Sparse mask transport: bbox-cropped, bit-packed binary masks.

Geological target masks are binary and almost always occupy a small fraction of
the slice they live on, yet the legacy transport shipped the *whole* slice as a
dense ``uint8`` ``.npy`` (e.g. ~6 MB for a 2201x2826 reservoir slice, ~99% of it
zeros). This module encodes a mask as its bounding box plus the bit-packed crop
(``np.packbits``), which is lossless and typically 1-3 orders of magnitude
smaller. JSON-friendly (base64) so it rides inside a normal result payload.

The representation is symmetric: ``decode_sparse_mask(encode_sparse_mask(m))``
reproduces ``m`` (as ``uint8`` 0/1) for any 2D mask, including all-zero masks.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

SPARSE_MASK_FORMAT = "sparse-bbox-packbits-v1"


def is_sparse_mask_payload(payload: Any) -> bool:
    """True if *payload* looks like an encoded sparse mask (vs a dense array)."""

    return isinstance(payload, dict) and payload.get("format") == SPARSE_MASK_FORMAT


def encode_sparse_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2D binary mask as ``{format, shape, bbox, packbits}``.

    ``bbox`` is ``[r0, c0, r1, c1]`` with ``r1``/``c1`` exclusive, or ``None``
    when the mask has no set pixels. ``packbits`` is base64 of the bit-packed
    cropped region in C order; empty string for an all-zero mask.
    """

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Sparse mask must be 2D, got shape {arr.shape}")
    binary = arr > 0
    height, width = (int(binary.shape[0]), int(binary.shape[1]))

    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size == 0:
        return {
            "format": SPARSE_MASK_FORMAT,
            "shape": [height, width],
            "bbox": None,
            "packbits": "",
        }
    cols = np.flatnonzero(binary.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    crop = np.ascontiguousarray(binary[r0:r1, c0:c1])
    packed = np.packbits(crop)
    return {
        "format": SPARSE_MASK_FORMAT,
        "shape": [height, width],
        "bbox": [r0, c0, r1, c1],
        "packbits": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def decode_sparse_mask(payload: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_sparse_mask`; returns a ``uint8`` 0/1 mask."""

    if not is_sparse_mask_payload(payload):
        raise ValueError("payload is not an encoded sparse mask")
    shape = payload.get("shape")
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError(f"sparse mask shape invalid: {shape!r}")
    height, width = int(shape[0]), int(shape[1])
    out = np.zeros((height, width), dtype=np.uint8)
    bbox = payload.get("bbox")
    if bbox is None:
        return out
    r0, c0, r1, c1 = (int(v) for v in bbox)
    crop_h, crop_w = r1 - r0, c1 - c0
    count = crop_h * crop_w
    raw = base64.b64decode(payload.get("packbits", ""))
    packed = np.frombuffer(raw, dtype=np.uint8)
    crop = np.unpackbits(packed, count=count).reshape(crop_h, crop_w)
    out[r0:r1, c0:c1] = crop.astype(np.uint8, copy=False)
    return out
