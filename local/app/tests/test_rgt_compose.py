"""Desktop RGT compositing equals the shared renderer SAM3 uses.

Guards that ``compose_rgt_rgb`` fetches the three source slices and produces the
exact image ``render_rgt_section`` produces server-side (same params + span), so
the on-screen section and the SAM3 input are pixel-identical.
"""

from __future__ import annotations

import numpy as np

from yj_studio.view.rgt_compose import compose_rgt_rgb, is_rgt_composite
from yj_studio_core.reservoir import extract_rgt_slice, render_rgt_section, RgtRenderParams, INLINE, XLINE


class _FakeStore:
    def __init__(self, lith, poro, rgt):
        self._v = {"model_lithology": lith, "model_porosity": poro, "rgt_field": rgt}

    def get_slice(self, volume_id, axis, index):
        a = self._v[volume_id]
        if axis == "inline":
            return np.asarray(a[index])
        if axis == "xline":
            return np.asarray(a[:, index, :])
        return np.asarray(a[:, :, index])

    def shape(self, volume_id):
        return tuple(int(v) for v in self._v[volume_id].shape)


class _Layer:
    def __init__(self, metadata):
        self.metadata = metadata
        self.volume_id = "model_rgt"


def _fixtures():
    rng = np.random.default_rng(0)
    n0, n1, ns = 4, 12, 20
    lith = (rng.random((n0, n1, ns)) > 0.6).astype(np.uint8)
    poro = rng.random((n0, n1, ns)).astype(np.float32)
    poro[lith == 0] = np.where(rng.random((poro[lith == 0].shape)) > 0.5, np.nan, 0.05)
    rgt = rng.random((n0 // 2, n1 // 2, ns // 5 or 1)).astype(np.float32)
    span = [0.1, 0.9]
    layer = _Layer(
        {
            "render": "rgt_overlay",
            "source_volumes": {
                "lithology": "model_lithology",
                "porosity": "model_porosity",
                "rgt": "rgt_field",
            },
            "render_params": {
                "alpha": 0.7, "sigma_lateral": 4.5, "sigma_depth": 0.9,
                "rgt_percentile": [2.0, 98.0],
                "sand_rgb": [255, 221, 0], "mud_rgb": [0, 0, 0],
                "nodata_rgb": [255, 255, 255], "smooth": True,
            },
            "rgt_span": span,
        }
    )
    return _FakeStore(lith, poro, rgt), layer, lith, poro, rgt, span


def test_is_rgt_composite_detects_metadata() -> None:
    _, layer, *_ = _fixtures()
    assert is_rgt_composite(layer)
    assert not is_rgt_composite(_Layer({}))


def test_compose_matches_shared_renderer_inline_and_xline() -> None:
    store, layer, lith, poro, rgt, span = _fixtures()
    params = RgtRenderParams.from_mapping(layer.metadata["render_params"])
    for axis, idx, rax in (("inline", 1, INLINE), ("xline", 3, XLINE)):
        desk = compose_rgt_rgb(store, layer, axis, idx)
        lith_s = store.get_slice("model_lithology", axis, idx)
        poro_s = store.get_slice("model_porosity", axis, idx).astype(np.float32)
        rgt_s = extract_rgt_slice(rgt, rax, idx)
        srv = render_rgt_section(lith_s, poro_s, rgt_s, params=params, rgt_span=tuple(span))
        np.testing.assert_array_equal(desk, srv)


def test_compose_z_axis_does_not_crash() -> None:
    store, layer, *_ = _fixtures()
    rgb = compose_rgt_rgb(store, layer, "z", 2)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3
