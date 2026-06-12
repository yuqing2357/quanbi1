from __future__ import annotations

import numpy as np

from yj_studio.algorithms.builtin.closure_contour import ClosureResult
from yj_studio.algorithms.builtin.trap_detect import TrapDetectAlgorithm, TrapDetectParams, rank_closures
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.scene.layers import HorizonLayer, TrapLayer


def _bowl(shape: tuple[int, int] = (41, 41), center: tuple[float, float] = (20.0, 20.0)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    return ((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / 20.0


def _closure(relief: float, area: int, high: tuple[int, int]) -> ClosureResult:
    return ClosureResult(
        high_ij=high,
        spill_level=relief,
        relief_samples=relief,
        area_cells=area,
        boundary_ij=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_rank_closures_uses_relief_and_area_weight() -> None:
    small_high_relief = _closure(relief=10.0, area=10, high=(1, 1))
    large_low_relief = _closure(relief=5.0, area=100, high=(2, 2))

    relief_ranked = rank_closures([small_high_relief, large_low_relief], area_weight=0.0)
    area_ranked = rank_closures([small_high_relief, large_low_relief], area_weight=1.0)

    assert relief_ranked[0][0] is small_high_relief
    assert area_ranked[0][0] is large_low_relief
    assert [rank for _closure_item, _score, rank in relief_ranked] == [1, 2]


def test_trap_detect_outputs_ranked_trap_layers() -> None:
    horizon = HorizonLayer(name="Top", sample=_bowl(), mask=np.ones((41, 41), dtype=bool))
    ctx = AlgorithmContext(
        input_layers={"horizon": horizon},
        params=TrapDetectParams(
            score_threshold=0.1,
            min_relief_m=3.0,
            level_step=1.0,
            min_area_cells=20,
            z_step_m=2.0,
        ),
    )

    result = TrapDetectAlgorithm().run(ctx)

    assert result.ok
    assert len(result.output_layers) == 1
    trap = result.output_layers[0]
    assert isinstance(trap, TrapLayer)
    assert trap.name == "Trap-1"
    assert trap.score is not None and trap.score >= 0.1
    assert trap.attributes["rank"] == 1
    assert trap.attributes["relief_m"] >= 3.0
    assert trap.metadata["algorithm"] == "trap.detect_structural"


def test_trap_detect_filters_by_relief_threshold() -> None:
    horizon = HorizonLayer(name="Top", sample=_bowl(), mask=np.ones((41, 41), dtype=bool))
    ctx = AlgorithmContext(
        input_layers={"horizon": horizon},
        params=TrapDetectParams(score_threshold=0.1, min_relief_m=10_000.0),
    )

    result = TrapDetectAlgorithm().run(ctx)

    assert not result.ok
    assert result.error == "没有满足阈值的圈闭"


def test_trap_detect_fault_bounded_mode_is_v2_failure() -> None:
    horizon = HorizonLayer(name="Top", sample=_bowl(), mask=np.ones((41, 41), dtype=bool))
    ctx = AlgorithmContext(
        input_layers={"horizon": horizon},
        params=TrapDetectParams(structural_only=False),
    )

    result = TrapDetectAlgorithm().run(ctx)

    assert not result.ok
    assert "断层封堵圈闭为 v2 功能" in (result.error or "")
