"""Serialization helpers for shipping ``Layer`` instances across processes.

The existing ``Layer.to_dict()`` implementations replace large ``numpy`` arrays
with their shape only (so projects can write a small TOML and keep arrays in a
side directory). For inter-process algorithm calls we need the full data — so
this module produces a richer payload that pairs the dict output with a side
table of pickled arrays.

The payload is plain Python ``dict`` with bytes values, which is what
``multiprocessing.Queue`` ships natively (it uses ``pickle`` internally).
"""

from __future__ import annotations

import pickle
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from yj_studio.scene.layer import Layer
from yj_studio.scene.layers import (
    AnnotationLayer,
    ArbitrarySectionLayer,
    FaultStickLayer,
    FaultSurfaceLayer,
    HorizonLayer,
    HorizonStickLayer,
    LithBodyLayer,
    MaskLayer,
    MeasurementLayer,
    PolygonLayer,
    TrapLayer,
    VolumeLayer,
    WellLayer,
    WellLogLayer,
)

_LAYER_REGISTRY: dict[str, type[Layer]] = {
    AnnotationLayer.kind: AnnotationLayer,
    ArbitrarySectionLayer.kind: ArbitrarySectionLayer,
    FaultStickLayer.kind: FaultStickLayer,
    FaultSurfaceLayer.kind: FaultSurfaceLayer,
    HorizonLayer.kind: HorizonLayer,
    HorizonStickLayer.kind: HorizonStickLayer,
    LithBodyLayer.kind: LithBodyLayer,
    MaskLayer.kind: MaskLayer,
    MeasurementLayer.kind: MeasurementLayer,
    PolygonLayer.kind: PolygonLayer,
    TrapLayer.kind: TrapLayer,
    VolumeLayer.kind: VolumeLayer,
    WellLayer.kind: WellLayer,
    WellLogLayer.kind: WellLogLayer,
}


def layer_to_payload(layer: Layer) -> dict[str, Any]:
    """Return a process-portable representation of ``layer``.

    The returned dict has two top-level keys:

    - ``"meta"``: ``layer.to_dict()`` output (suitable for TOML/JSON).
    - ``"arrays"``: ``{attr_name: pickled_ndarray_bytes}`` for every dataclass
      field whose runtime value is a numpy array. ``meta`` already records the
      shape via ``update_with_shape`` so the receiver can sanity-check.
    """

    if not is_dataclass(layer):
        raise TypeError(f"Layer must be a dataclass instance, got {type(layer)!r}")
    meta = layer.to_dict()
    arrays: dict[str, bytes] = {}
    for f in fields(layer):
        value = getattr(layer, f.name)
        if isinstance(value, np.ndarray):
            arrays[f.name] = pickle.dumps(np.asarray(value), protocol=pickle.HIGHEST_PROTOCOL)
    return {"meta": meta, "arrays": arrays}


def payload_to_layer(payload: dict[str, Any]) -> Layer:
    """Reverse of :func:`layer_to_payload`."""

    meta = payload["meta"]
    kind = meta.get("kind")
    layer_cls = _LAYER_REGISTRY.get(kind)
    if layer_cls is None:
        raise ValueError(f"Unknown layer kind: {kind!r}")
    layer = layer_cls.from_dict(meta)
    for attr, raw in payload.get("arrays", {}).items():
        if hasattr(layer, attr):
            setattr(layer, attr, pickle.loads(raw))
    return layer


def layers_to_payloads(layers: dict[str, Layer]) -> dict[str, dict[str, Any]]:
    """Map a role-keyed dict of layers to a role-keyed dict of payloads."""

    return {role: layer_to_payload(layer) for role, layer in layers.items()}


def payloads_to_layers(payloads: dict[str, dict[str, Any]]) -> dict[str, Layer]:
    return {role: payload_to_layer(payload) for role, payload in payloads.items()}
