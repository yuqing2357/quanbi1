"""The RGT-overlay composite is the single image SAM3 sees.

Verifies that an ``rgt_overlay`` volume renders through the shared
reservoir/RGT renderer (three regions, model-grid shape, fixed-span caching),
that a plain volume is unchanged grayscale, and that the virtual composite does
not break VolumeCache preload/status. Skipped where fastapi is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[2]
for sub in ("server/src", "shared/src"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

from yj_studio_server.app import _render_section_rgb, create_app  # noqa: E402
from yj_studio_server.config import ServerConfig  # noqa: E402


def _make_cfg(tmp_path: Path) -> ServerConfig:
    data_root = tmp_path / "data"
    (data_root / "reservoir").mkdir(parents=True)
    (data_root / "seismic").mkdir(parents=True)

    # axis0=2 inlines, axis1=8 (long), sample=10. Column j (axis1) defines region:
    #   j<3 -> sand, 3<=j<6 -> mud, j>=6 -> no-data (NaN porosity).
    n0, n1, ns = 2, 8, 10
    lith = np.zeros((n0, n1, ns), np.uint8)
    poro = np.zeros((n0, n1, ns), np.float32)
    lith[:, 0:3, :] = 1
    poro[:, 0:3, :] = 0.2          # sand: finite poro
    poro[:, 3:6, :] = 0.05         # mud: finite poro, lith 0
    poro[:, 6:8, :] = np.nan       # no-data
    np.save(data_root / "reservoir/litho.npy", lith, allow_pickle=False)
    np.save(data_root / "reservoir/poro.npy", poro, allow_pickle=False)

    # RGT on the half-lateral grid; a depth gradient (sample axis kept full here).
    rgt = np.broadcast_to(
        np.linspace(0, 1, ns, dtype=np.float32), (1, n1 // 2, ns)
    ).astype(np.float32)
    np.save(data_root / "seismic/rgt.npy", rgt, allow_pickle=False)

    params = {
        "alpha": 0.7, "sigma_lateral": 4.5, "sigma_depth": 0.9,
        "rgt_percentile": [2.0, 98.0],
        "sand_rgb": [255, 221, 0], "mud_rgb": [0, 0, 0],
        "nodata_rgb": [255, 255, 255],
        "smooth": False,  # keep region columns crisp for assertions
    }
    return ServerConfig(
        project_root=tmp_path,
        data_root=data_root,
        runtime_root=tmp_path / "runtime" / "server",
        results_root=data_root / "results",
        project_id="default",
        volumes={
            "model_lithology": {"label": "lith", "path": "reservoir/litho.npy", "clim": None},
            "model_porosity": {"label": "poro", "path": "reservoir/poro.npy", "clim": None},
            "rgt_field": {"label": "rgt", "path": "seismic/rgt.npy"},
            "model_rgt": {
                "label": "RGT 地层色带",
                "render": "rgt_overlay",
                "lithology_volume": "model_lithology",
                "porosity_volume": "model_porosity",
                "rgt_volume": "rgt_field",
                "rgt_span": None,
                "params": params,
            },
        },
        sam3={"checkpoint": "weights/sam3.pt", "results_subdir": "sam3"},
    )


def test_composite_renders_three_regions_and_caches_span(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)

    rgb, shape = _render_section_rgb(app, cfg, "model_rgt", "inline", 0)
    # (n_sample, n_long, 3); shape is the model (lithology) grid.
    assert rgb.shape == (10, 8, 3)
    assert rgb.dtype == np.uint8
    assert shape == (2, 8, 10)

    # columns map to axis-1 of the output image.
    nodata_col = rgb[:, 7, :]
    mud_col = rgb[:, 4, :]
    sand_col = rgb[:, 1, :]
    assert np.all(nodata_col == 255)            # no-data -> white
    assert np.all(mud_col == 0)                 # mud -> black
    assert not np.all(sand_col == 0)            # sand -> RGT colour, not background
    assert not np.all(sand_col == 255)

    # fixed span computed once and cached on app.state.
    assert "rgt_field" in app.state.rgt_span_cache
    lo, hi = app.state.rgt_span_cache["rgt_field"]
    assert hi > lo


def test_plain_volume_is_unchanged_grayscale(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)
    rgb, shape = _render_section_rgb(app, cfg, "model_lithology", "inline", 0)
    assert rgb.shape == (10, 8, 3)
    assert shape == (2, 8, 10)
    # grayscale stretch => R == G == B everywhere.
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])


def test_virtual_volume_does_not_break_cache(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)
    app.state.volumes.preload_all()
    status = {v["volume_id"]: v for v in app.state.volumes.status()["volumes"]}
    assert status["model_rgt"]["state"] == "virtual"
    # real source volumes still load.
    arr, _mode = app.state.volumes.get("rgt_field")
    assert arr.ndim == 3


def test_volumes_catalogue_describes_composite(tmp_path: Path) -> None:
    """The desktop needs the composite to be self-describing (shape, sources,
    params, fixed span) or it cannot render the same image SAM3 sees."""
    from fastapi.testclient import TestClient

    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)
    with TestClient(app) as client:
        catalogue = {v["id"]: v for v in client.get("/volumes").json()}

    comp = catalogue["model_rgt"]
    assert comp["exists"] is True
    assert comp["render"] == "rgt_overlay"
    assert comp["shape"] == [2, 8, 10]  # borrowed from the lithology source grid
    assert comp["source_volumes"]["lithology"] == "model_lithology"
    assert comp["render_params"]["alpha"] == pytest.approx(0.7)
    span = comp["rgt_span"]
    assert isinstance(span, list) and len(span) == 2 and span[1] > span[0]
