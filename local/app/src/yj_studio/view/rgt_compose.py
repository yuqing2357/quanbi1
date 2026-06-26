"""Desktop-side RGT stratigraphy compositing — same pixels SAM3 sees.

A ``model_rgt`` volume is not a single array; it is rendered from three source
volumes (lithology, porosity, rgt_field). The server renders it for SAM3 via
``shared/.../reservoir/rgt_overlay.render_rgt_section``; the desktop renders it
for display through the SAME function here, with the SAME params and fixed RGT
span (carried on the layer metadata from the server catalogue). So what the user
sees while drawing prompts is exactly what SAM3 ingests.

``compose_rgt_rgb`` returns RGB in (n_sample, n_long, 3) uint8 — the orientation
of ``raw_slice.T`` used by both the 3D slice renderer (build_slice_image) and the
2D section view (section.values), so callers can drop it straight into imshow /
the slice texture without another transpose.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from yj_studio_core.reservoir import RgtRenderParams, render_rgt_section

# RGT grid is the model's 2x lateral coarsening; an inline/xline model index maps
# to rgt index // 2 (the renderer upsamples within the slice).
_RGT_LATERAL_STRIDE = 2

# Composite RGB is deterministic for (sources, axis, index, span, params) and the
# gaussian-field render is the per-slice cost, so cache recent results. The 3D
# view renders three axes per refresh and the user scrubs slices — caching makes
# re-views instant. Each entry is a full-section RGB (~tens of MB); keep it small.
_CACHE_MAX = 8
_RENDER_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def clear_cache() -> None:
    _RENDER_CACHE.clear()


def is_rgt_composite(layer) -> bool:
    """True for a layer backed by the rgt_overlay composite render."""
    return str(getattr(layer, "metadata", {}).get("render", "")) == "rgt_overlay"


def is_rgt_composite_meta(metadata: dict | None) -> bool:
    """True for catalogue metadata describing an rgt_overlay composite."""
    return str((metadata or {}).get("render", "")) == "rgt_overlay"


def _source_ids(metadata: dict) -> tuple[str, str, str]:
    src = metadata.get("source_volumes") or {}
    return (
        str(src.get("lithology", "model_lithology")),
        str(src.get("porosity", "model_porosity")),
        str(src.get("rgt", "rgt_field")),
    )


def _span(metadata: dict) -> tuple[float, float] | None:
    span = metadata.get("rgt_span")
    if isinstance(span, (list, tuple)) and len(span) >= 2 and span[0] is not None:
        return (float(span[0]), float(span[1]))
    return None


def compose_rgt_rgb(volume_store, layer, axis: str, index: int) -> np.ndarray:
    """Render the composite section for a layer; see :func:`compose_rgt_rgb_from_meta`."""
    return compose_rgt_rgb_from_meta(
        volume_store, getattr(layer, "metadata", {}) or {}, axis, index
    )


def compose_rgt_rgb_from_meta(volume_store, metadata: dict, axis: str, index: int) -> np.ndarray:
    """Render the composite section to (n_sample, n_long, 3) uint8.

    Fetches the three source slices through ``volume_store`` (local mmap or remote
    /slice, whichever owns each source) and composites with the shared renderer
    using the pinned params/span from ``metadata`` (the server catalogue entry or
    the layer metadata, which carry identical fields).
    """
    metadata = dict(metadata or {})
    lith_id, poro_id, rgt_id = _source_ids(metadata)
    params = RgtRenderParams.from_mapping(metadata.get("render_params"))
    span = _span(metadata)

    key = (lith_id, poro_id, rgt_id, str(axis), int(index), span, params)
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        _RENDER_CACHE.move_to_end(key)
        return cached

    lith = volume_store.get_slice(lith_id, axis, int(index))
    poro = volume_store.get_slice(poro_id, axis, int(index))

    if axis in ("inline", "xline"):
        rgt_axis_len = volume_store.shape(rgt_id)[0 if axis == "inline" else 1]
        rgt_index = min(int(index) // _RGT_LATERAL_STRIDE, int(rgt_axis_len) - 1)
        rgt = volume_store.get_slice(rgt_id, axis, rgt_index)
        rgb = render_rgt_section(lith, poro, rgt, params=params, rgt_span=span)
    else:
        # z/timeslice: no RGT-stratigraphy axis; flat field keeps the 3-region base.
        rgb = render_rgt_section(
            lith, poro, np.zeros((1, 1), np.float32), params=params, rgt_span=span or (0.0, 1.0)
        )

    _RENDER_CACHE[key] = rgb
    _RENDER_CACHE.move_to_end(key)
    while len(_RENDER_CACHE) > _CACHE_MAX:
        _RENDER_CACHE.popitem(last=False)
    return rgb
