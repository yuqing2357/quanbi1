"""Interval thickness between two HorizonLayers (requirement 二-5)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.config.defaults import DEPTH_STEP_TO_SAMPLE
from yj_studio.scene.layers import HorizonLayer, MeasurementLayer


class ThicknessParams(BaseModel):
    depth_step_m: float = Field(
        default=DEPTH_STEP_TO_SAMPLE,
        description="每个采样点对应的米数，用于把采样差换算为米。",
        gt=0.0,
    )
    name: str = Field(
        default="层间厚度",
        description="结果测量图层的显示名称。",
    )


class ThicknessOutput(BaseModel):
    mean_m: float
    min_m: float
    max_m: float
    std_m: float
    coverage: float
    valid_cells: int


@register_algorithm
class ThicknessAlgorithm(Algorithm):
    id: ClassVar[str] = "horizon.thickness"
    category: ClassVar[str] = "horizon"
    label: ClassVar[str] = "层间厚度"
    description: ClassVar[str] = (
        "计算上下两层位之间的点厚度。层位坐标均以采样索引表示，输出一"
        "个 MeasurementLayer，包含厚度网格（geometry: (N, 4) = 纵向、"
        "横向、顶层 Z、底层 Z）以及以米为单位的统计值。"
    )
    input_schema: ClassVar[type[BaseModel]] = ThicknessParams
    output_schema: ClassVar[type[BaseModel]] = ThicknessOutput
    layer_inputs: ClassVar[dict[str, str]] = {"top": "horizon", "bottom": "horizon"}

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        top_layer = ctx.input_layers.get("top")
        bottom_layer = ctx.input_layers.get("bottom")
        if not isinstance(top_layer, HorizonLayer) or top_layer.sample is None:
            return AlgorithmResult.failure("上层位缺少样点数据")
        if not isinstance(bottom_layer, HorizonLayer) or bottom_layer.sample is None:
            return AlgorithmResult.failure("下层位缺少样点数据")
        if top_layer.sample.shape != bottom_layer.sample.shape:
            return AlgorithmResult.failure(
                f"层位形状不一致：上层 {top_layer.sample.shape}，"
                f"下层 {bottom_layer.sample.shape}"
            )

        ctx.report_progress(0.1, "读取层位")
        top = np.asarray(top_layer.sample, dtype=np.float32)
        bot = np.asarray(bottom_layer.sample, dtype=np.float32)

        valid = np.isfinite(top) & np.isfinite(bot)
        if top_layer.mask is not None:
            valid &= np.asarray(top_layer.mask, dtype=bool)
        if bottom_layer.mask is not None:
            valid &= np.asarray(bottom_layer.mask, dtype=bool)

        # Thickness is bottom - top in sample units (positive when bottom is deeper).
        thickness_samples = np.where(valid, bot - top, np.nan)
        thickness_m = thickness_samples * float(ctx.params.depth_step_m)

        ctx.report_progress(0.5, "汇总统计")
        valid_cells = int(np.sum(valid))
        coverage = float(valid_cells) / float(valid.size) if valid.size else 0.0
        if valid_cells == 0:
            return AlgorithmResult.failure("两层位之间没有重叠的有效点")

        finite_values = thickness_m[valid]
        stats = ThicknessOutput(
            mean_m=float(np.mean(finite_values)),
            min_m=float(np.min(finite_values)),
            max_m=float(np.max(finite_values)),
            std_m=float(np.std(finite_values)),
            coverage=coverage,
            valid_cells=valid_cells,
        )

        ctx.report_progress(0.8, "构建几何")
        rows, cols = np.where(valid)
        geometry = np.column_stack(
            [
                rows.astype(np.float32),
                cols.astype(np.float32),
                top[rows, cols].astype(np.float32),
                bot[rows, cols].astype(np.float32),
                thickness_m[rows, cols].astype(np.float32),
            ]
        )

        measurement = MeasurementLayer(
            name=ctx.params.name or "层间厚度",
            geometry=geometry,
            values={
                "mean_m": stats.mean_m,
                "min_m": stats.min_m,
                "max_m": stats.max_m,
                "std_m": stats.std_m,
                "coverage": stats.coverage,
                "valid_cells": float(stats.valid_cells),
            },
            units={
                "mean_m": "m",
                "min_m": "m",
                "max_m": "m",
                "std_m": "m",
                "coverage": "ratio",
                "valid_cells": "cells",
            },
            color=(0.95, 0.55, 0.2, 0.6),
            opacity=0.6,
            metadata={
                "algorithm": ThicknessAlgorithm.id,
                "top_horizon": top_layer.name,
                "bottom_horizon": bottom_layer.name,
                "shape": list(top_layer.sample.shape),
                "depth_step_m": float(ctx.params.depth_step_m),
            },
            provenance={
                "source": "algorithm.horizon.thickness",
                "top_horizon_id": top_layer.id,
                "bottom_horizon_id": bottom_layer.id,
            },
        )

        ctx.report_progress(1.0, "完成")
        summary = (
            f"厚度 {ctx.params.name}: 平均={stats.mean_m:.1f} m，"
            f"范围=[{stats.min_m:.1f}, {stats.max_m:.1f}] m，"
            f"覆盖率={stats.coverage * 100:.1f}%"
        )
        return AlgorithmResult.success(output_layers=[measurement], summary=summary)
