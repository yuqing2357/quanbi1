from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared/src"))

from yj_studio_core.multichannel import (  # noqa: E402
    extract_multichannel_slice,
    resample_nodes,
)


def test_resample_nodes_preserves_integer_nodes() -> None:
    source = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    result = resample_nodes(source, (5, 10))
    np.testing.assert_array_equal(result[::2, ::3], source)


def test_extract_multichannel_slice_all_axes(tmp_path: Path) -> None:
    model_shape = (5, 7, 11)
    scale = (2, 2, 5)
    seismic_shape = (3, 4, 3)
    data = tmp_path / "data"
    model_dir = data / "model"
    attrs_dir = data / "seismic/attrs"
    model_dir.mkdir(parents=True)
    attrs_dir.mkdir(parents=True)

    lithology = np.zeros(model_shape, dtype=np.uint8)
    lithology[::2, ::2, ::5] = 1
    porosity = np.linspace(0, 0.4, np.prod(model_shape), dtype=np.float32).reshape(
        model_shape
    )
    porosity[0, 0, 0] = np.nan
    cosphase = np.arange(np.prod(seismic_shape), dtype=np.float32).reshape(
        seismic_shape
    )
    coherence = cosphase / cosphase.max()
    np.save(model_dir / "lith.npy", lithology)
    np.save(model_dir / "poro.npy", porosity.astype(np.float16))
    np.save(attrs_dir / "cos.npy", cosphase.astype(np.float16))
    np.save(attrs_dir / "coh.npy", coherence.astype(np.float16))

    spec = {
        "grid_model_shape": list(model_shape),
        "scale": list(scale),
        "channels": [
            {
                "name": "lithology",
                "grid": "model",
                "path": "data/model/lith.npy",
                "norm": "as_is",
            },
            {
                "name": "porosity",
                "grid": "model",
                "path": "data/model/poro.npy",
                "norm": "clip01_porosity",
            },
            {
                "name": "cosphase",
                "grid": "seismic_crop",
                "path": "data/seismic/attrs/cos.npy",
                "norm": "as_is",
            },
            {
                "name": "coherence",
                "grid": "seismic_crop",
                "path": "data/seismic/attrs/coh.npy",
                "norm": "as_is",
            },
        ],
    }
    spec_path = attrs_dir / "channel_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    for axis, index, expected_hw in (
        ("inline", 2, (7, 11)),
        ("xline", 4, (5, 11)),
        ("depth", 5, (5, 7)),
    ):
        result = extract_multichannel_slice(spec_path, axis, index)
        assert result.shape == (4, *expected_hw)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert set(np.unique(result[0])).issubset({0.0, 1.0})

    inline = extract_multichannel_slice(spec_path, "inline", 2)
    np.testing.assert_array_equal(inline[2, ::2, ::5], cosphase[1])
