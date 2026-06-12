from __future__ import annotations

import numpy as np

from yj_studio.algorithms.builtin.closure_contour import (
    ClosureContourAlgorithm,
    ClosureContourParams,
    _structural_highs,
    detect_closures,
)
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.scene.manual_geometry import is_manual_geometry_layer, manual_geometry_points
from yj_studio.scene.layers import HorizonLayer, TrapLayer
from yj_studio.view.renderers.manual_geometry_renderer import build_trap_surface


def _bowl(shape: tuple[int, int] = (41, 41), center: tuple[float, float] = (20.0, 20.0)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    return ((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / 20.0


def test_structural_highs_deduplicates_flat_plateau() -> None:
    z = np.ones((9, 9), dtype=np.float32) * 5.0
    z[3:6, 3:6] = 1.0
    valid = np.ones_like(z, dtype=bool)

    highs = _structural_highs(z, valid)

    assert len(highs) == 1
    assert 3 <= highs[0][0] <= 5
    assert 3 <= highs[0][1] <= 5


def test_detect_closures_single_bowl_returns_closed_boundary() -> None:
    z = _bowl()
    params = ClosureContourParams(level_step=1.0, min_relief_samples=3.0, min_area_cells=20)

    closures = detect_closures(z, np.ones_like(z, dtype=bool), params)

    assert len(closures) == 1
    closure = closures[0]
    assert closure.relief_samples > 3.0
    assert closure.area_cells >= 20
    assert closure.boundary_ij.shape[1] == 2
    assert np.linalg.norm(closure.boundary_ij[0] - closure.boundary_ij[-1]) < 1e-5


def test_detect_closures_two_separate_highs() -> None:
    rows, cols = np.indices((64, 64), dtype=np.float32)
    left = ((rows - 20.0) ** 2 + (cols - 20.0) ** 2) / 18.0
    right = ((rows - 44.0) ** 2 + (cols - 44.0) ** 2) / 18.0
    z = np.minimum(left, right)
    params = ClosureContourParams(level_step=1.0, min_relief_samples=3.0, min_area_cells=20, max_highs=10)

    closures = detect_closures(z, np.ones_like(z, dtype=bool), params)

    assert len(closures) == 2
    assert {tuple(c.high_ij) for c in closures} == {(20, 20), (44, 44)}


def test_detect_closures_monotonic_slope_returns_empty() -> None:
    z = np.tile(np.arange(30, dtype=np.float32)[:, None], (1, 30))

    closures = detect_closures(z, np.ones_like(z, dtype=bool), ClosureContourParams(level_step=1.0))

    assert closures == []


def test_detect_closures_thresholds_filter_small_closure() -> None:
    z = _bowl(shape=(25, 25), center=(12.0, 12.0))
    params = ClosureContourParams(level_step=1.0, min_relief_samples=1_000.0, min_area_cells=1)

    closures = detect_closures(z, np.ones_like(z, dtype=bool), params)

    assert closures == []


def test_detect_closures_nan_mask_is_ignored() -> None:
    z = _bowl()
    z[:5, :] = np.nan
    valid = np.isfinite(z)
    params = ClosureContourParams(level_step=1.0, min_relief_samples=3.0, min_area_cells=20)

    closures = detect_closures(z, valid, params)

    assert len(closures) == 1
    assert closures[0].high_ij == (20, 20)


def test_closure_contour_algorithm_outputs_trap_layer() -> None:
    horizon = HorizonLayer(name="Top", sample=_bowl(), mask=np.ones((41, 41), dtype=bool))
    ctx = AlgorithmContext(
        input_layers={"horizon": horizon},
        params=ClosureContourParams(level_step=1.0, min_relief_samples=3.0, min_area_cells=20, z_step_m=2.0),
    )

    result = ClosureContourAlgorithm().run(ctx)

    assert result.ok
    assert result.output_layers
    trap = result.output_layers[0]
    assert isinstance(trap, TrapLayer)
    assert trap.boundary is not None
    assert trap.boundary.shape[1] == 3
    assert trap.attributes["relief_m"] > 0.0
    assert trap.metadata["algorithm"] == "horizon.closure_contour"


def test_trap_layer_exposes_boundary_to_rendering_helpers() -> None:
    boundary = np.asarray(
        [
            [1.0, 1.0, 5.0],
            [3.0, 1.0, 5.0],
            [3.0, 4.0, 5.0],
            [1.0, 4.0, 5.0],
            [1.0, 1.0, 5.0],
        ],
        dtype=np.float32,
    )
    trap = TrapLayer(name="trap", boundary=boundary, score=0.7)

    assert is_manual_geometry_layer(trap)
    np.testing.assert_allclose(manual_geometry_points(trap), boundary)

    surface = build_trap_surface(trap)
    assert surface.n_points == 4
    assert surface.n_cells == 1
