"""Rank structural trap candidates from horizon closures."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.builtin.closure_contour import ClosureContourParams, ClosureResult, detect_closures
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import HorizonLayer, TrapLayer


class TrapDetectParams(BaseModel):
    structural_only: bool = Field(default=True, description="v1 仅做四面构造闭合。")
    score_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="综合分阈值。")
    z_step_m: float = Field(default=1.0, gt=0.0, description="每采样对应米数。")
    min_relief_m: float = Field(default=10.0, ge=0.0, description="最小闭合高度（米）。")
    area_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="综合分里面积权重。")
    level_step: float = Field(default=1.0, gt=0.0, description="水位抬升步长（采样单位）。")
    min_area_cells: int = Field(default=8, ge=1, description="最小闭合面积（网格点数）。")
    max_highs: int = Field(default=50, ge=1, description="候选构造高点数量上限。")
    shallower_is_smaller: bool = Field(default=True, description="True 表示采样值越小越浅。")


def rank_closures(
    closures: list[ClosureResult],
    *,
    area_weight: float = 0.3,
) -> list[tuple[ClosureResult, float, int]]:
    if not closures:
        return []
    reliefs = np.asarray([max(0.0, item.relief_samples) for item in closures], dtype=np.float32)
    areas = np.asarray([max(0, item.area_cells) for item in closures], dtype=np.float32)
    max_relief = float(reliefs.max()) if reliefs.size else 0.0
    max_area = float(areas.max()) if areas.size else 0.0
    area_w = float(np.clip(area_weight, 0.0, 1.0))
    rows: list[tuple[ClosureResult, float]] = []
    for closure, relief, area in zip(closures, reliefs, areas, strict=False):
        norm_relief = float(relief / max_relief) if max_relief > 0 else 0.0
        norm_area = float(area / max_area) if max_area > 0 else 0.0
        score = (1.0 - area_w) * norm_relief + area_w * norm_area
        rows.append((closure, float(np.clip(score, 0.0, 1.0))))
    rows.sort(key=lambda item: (item[1], item[0].relief_samples, item[0].area_cells), reverse=True)
    return [(closure, score, rank) for rank, (closure, score) in enumerate(rows, start=1)]


@register_algorithm
class TrapDetectAlgorithm(Algorithm):
    id: ClassVar[str] = "trap.detect_structural"
    category: ClassVar[str] = "trap"
    label: ClassVar[str] = "圈闭检测"
    description: ClassVar[str] = "复用闭合等值线结果，输出带排名和置信度的候选构造圈闭。"
    input_schema: ClassVar[type[BaseModel]] = TrapDetectParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}
    runs_in_subprocess: ClassVar[bool] = False

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        params = ctx.params if isinstance(ctx.params, TrapDetectParams) else TrapDetectParams.model_validate(ctx.params or {})
        if not params.structural_only:
            return AlgorithmResult.failure("断层封堵圈闭为 v2 功能，当前版本仅支持四面构造闭合。")

        layer = ctx.input_layers.get("horizon")
        if not isinstance(layer, HorizonLayer) or layer.sample is None:
            return AlgorithmResult.failure("缺少层位样点数据")
        z = np.asarray(layer.sample, dtype=np.float32)
        valid = np.isfinite(z)
        if layer.mask is not None:
            valid &= np.asarray(layer.mask, dtype=bool)
        if not valid.any():
            return AlgorithmResult.failure("层位没有有效样点")

        closure_params = ClosureContourParams(
            z_step_m=params.z_step_m,
            level_step=params.level_step,
            min_relief_samples=float(params.min_relief_m) / float(params.z_step_m),
            min_area_cells=params.min_area_cells,
            max_highs=params.max_highs,
            shallower_is_smaller=params.shallower_is_smaller,
        )
        ctx.report_progress(0.1, "检测构造闭合")
        closures = detect_closures(
            z,
            valid,
            closure_params,
            progress=lambda index, total: ctx.report_progress(
                0.1 + 0.5 * float(index + 1) / max(float(total), 1.0),
                "抬升水位",
            ),
            check_cancel=ctx.check_cancel,
        )
        ranked = [
            (closure, score, rank)
            for closure, score, rank in rank_closures(closures, area_weight=params.area_weight)
            if score >= float(params.score_threshold)
        ]
        if not ranked:
            return AlgorithmResult.failure("没有满足阈值的圈闭")

        output_layers: list[TrapLayer] = []
        for pos, (closure, score, rank) in enumerate(ranked, start=1):
            ctx.check_cancel()
            boundary = np.column_stack(
                [
                    closure.boundary_ij[:, 0],
                    closure.boundary_ij[:, 1],
                    np.full(closure.boundary_ij.shape[0], closure.spill_level, dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            high_i, high_j = closure.high_ij
            relief_m = closure.relief_samples * float(params.z_step_m)
            output_layers.append(
                TrapLayer(
                    name=f"Trap-{rank}",
                    boundary=boundary,
                    score=score,
                    attributes={
                        "rank": rank,
                        "candidate_score": score,
                        "high_inline": high_i,
                        "high_xline": high_j,
                        "spill_level": closure.spill_level,
                        "relief_samples": closure.relief_samples,
                        "relief_m": relief_m,
                        "area_cells": closure.area_cells,
                        "edge_limited": closure.edge_limited,
                        "source_horizon": layer.name,
                    },
                    color=_trap_score_color(score),
                    opacity=0.9,
                    metadata={"algorithm": self.id, "source_horizon_id": layer.id},
                    provenance={"source": f"algorithm.{self.id}", "horizon_id": layer.id},
                )
            )
            ctx.report_progress(0.6 + 0.35 * pos / len(ranked), "排序圈闭")

        best = max(float(layer.score or 0.0) for layer in output_layers)
        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=output_layers,
            summary=f"圈闭检测：{len(output_layers)} 个候选，最高置信度 {best:.2f}",
        )


def _trap_score_color(score: float) -> tuple[float, float, float, float]:
    s = float(np.clip(score, 0.0, 1.0))
    return (0.35 + 0.60 * s, 0.35 * (1.0 - s), 0.25 * (1.0 - s), 0.9)
