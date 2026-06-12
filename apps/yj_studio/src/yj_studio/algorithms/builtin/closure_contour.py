"""Structural closure contour extraction on horizon grids."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import label, minimum_filter
from skimage.measure import find_contours

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import HorizonLayer, TrapLayer


class ClosureContourParams(BaseModel):
    z_step_m: float = Field(default=1.0, gt=0.0, description="每采样对应米数，用于报告闭合高度。")
    level_step: float = Field(default=1.0, gt=0.0, description="水位抬升步长（采样单位）。")
    min_relief_samples: float = Field(default=2.0, ge=0.0, description="最小闭合高度（采样）。")
    min_area_cells: int = Field(default=8, ge=1, description="最小闭合面积（网格点数）。")
    max_highs: int = Field(default=50, ge=1, description="候选构造高点数量上限。")
    shallower_is_smaller: bool = Field(default=True, description="True 表示采样值越小越浅。")


@dataclass(frozen=True, slots=True)
class ClosureResult:
    high_ij: tuple[int, int]
    spill_level: float
    relief_samples: float
    area_cells: int
    boundary_ij: np.ndarray
    edge_limited: bool = False


def _structural_highs(z: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Return one representative point for every local structural high."""
    depth = np.asarray(z, dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(depth)
    if not valid_mask.any():
        return []
    safe = np.where(valid_mask, depth, np.inf)
    filt = minimum_filter(safe, size=3, mode="nearest")
    is_min = valid_mask & (safe <= filt)
    labels, n_labels = label(is_min)
    highs: list[tuple[int, int]] = []
    for label_id in range(1, int(n_labels) + 1):
        rows, cols = np.nonzero(labels == label_id)
        if rows.size == 0:
            continue
        if rows.min() == 0 or rows.max() == depth.shape[0] - 1 or cols.min() == 0 or cols.max() == depth.shape[1] - 1:
            continue
        values = safe[rows, cols]
        best = int(np.argmin(values))
        highs.append((int(rows[best]), int(cols[best])))
    highs.sort(key=lambda ij: (float(safe[ij]), int(ij[0]), int(ij[1])))
    return highs


def detect_closures(
    z: np.ndarray,
    valid: np.ndarray | None,
    params: ClosureContourParams | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> list[ClosureResult]:
    """Detect four-way structural closures by raising a virtual water level.

    Internally all calculations use an oriented "depth" where smaller values
    are shallower. Returned ``spill_level`` values are converted back to the
    input horizon's original sample convention.
    """
    p = params or ClosureContourParams()
    raw = np.asarray(z, dtype=np.float32)
    valid_mask = np.isfinite(raw)
    if valid is not None:
        valid_mask &= np.asarray(valid, dtype=bool)
    if not valid_mask.any():
        return []

    oriented = raw if p.shallower_is_smaller else -raw
    highs = _structural_highs(oriented, valid_mask)[: int(p.max_highs)]
    if not highs:
        return []

    z_min = float(np.nanmin(oriented[valid_mask]))
    z_max = float(np.nanmax(oriented[valid_mask]))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
        return []

    levels = np.arange(z_min + float(p.level_step), z_max, float(p.level_step), dtype=np.float32)
    closure: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    edge_limited: set[tuple[int, int]] = set()
    spilled: set[tuple[int, int]] = set()

    n_levels = int(len(levels))
    for level_index, level in enumerate(levels):
        if check_cancel is not None and (level_index == 0 or level_index % 10 == 0):
            check_cancel()
        if progress is not None and (level_index == 0 or level_index % 10 == 0 or level_index == n_levels - 1):
            progress(level_index, n_levels)
        region = valid_mask & (oriented <= float(level))
        labels, n_labels = label(region)
        for label_id in range(1, int(n_labels) + 1):
            comp = labels == label_id
            inside = [high for high in highs if high not in spilled and comp[high]]
            if not inside:
                continue
            touches_edge = bool(comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())
            if touches_edge or len(inside) > 1:
                for high in inside:
                    if touches_edge and high in closure:
                        edge_limited.add(high)
                    spilled.add(high)
                continue
            closure[inside[0]] = (float(level), comp.copy())

    results: list[ClosureResult] = []
    for high, (oriented_level, comp) in closure.items():
        high_value = float(oriented[high])
        relief = float(oriented_level - high_value)
        area = int(comp.sum())
        if relief < float(p.min_relief_samples) or area < int(p.min_area_cells):
            continue
        contours = find_contours(comp.astype(np.float32), 0.5)
        if not contours:
            continue
        ring = np.asarray(max(contours, key=len), dtype=np.float32)
        if ring.shape[0] < 3:
            continue
        if float(np.linalg.norm(ring[0] - ring[-1])) > 1e-5:
            ring = np.vstack([ring, ring[0]])
        spill_level = oriented_level if p.shallower_is_smaller else -oriented_level
        results.append(
            ClosureResult(
                high_ij=(int(high[0]), int(high[1])),
                spill_level=float(spill_level),
                relief_samples=relief,
                area_cells=area,
                boundary_ij=ring.astype(np.float32, copy=False),
                edge_limited=high in edge_limited,
            )
        )

    results.sort(key=lambda item: item.relief_samples, reverse=True)
    return results


@register_algorithm
class ClosureContourAlgorithm(Algorithm):
    id: ClassVar[str] = "horizon.closure_contour"
    category: ClassVar[str] = "horizon"
    label: ClassVar[str] = "闭合等值线"
    description: ClassVar[str] = "在层位上用涨水法寻找四面闭合的构造高，输出闭合多边形 TrapLayer。"
    input_schema: ClassVar[type[BaseModel]] = ClosureContourParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}
    runs_in_subprocess: ClassVar[bool] = False

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        layer = ctx.input_layers.get("horizon")
        if not isinstance(layer, HorizonLayer) or layer.sample is None:
            return AlgorithmResult.failure("缺少层位样点数据")

        params = ctx.params if isinstance(ctx.params, ClosureContourParams) else ClosureContourParams.model_validate(ctx.params or {})
        ctx.report_progress(0.1, "读取层位")
        z = np.asarray(layer.sample, dtype=np.float32)
        valid = np.isfinite(z)
        if layer.mask is not None:
            valid &= np.asarray(layer.mask, dtype=bool)
        if not valid.any():
            return AlgorithmResult.failure("层位没有有效样点")

        ctx.report_progress(0.2, "检测闭合")
        closures = detect_closures(
            z,
            valid,
            params,
            progress=lambda index, total: ctx.report_progress(
                0.2 + 0.55 * float(index + 1) / max(float(total), 1.0),
                "抬升水位",
            ),
            check_cancel=ctx.check_cancel,
        )
        if not closures:
            return AlgorithmResult.failure("没有满足阈值的闭合")

        z_range = _oriented_range(z, valid, shallower_is_smaller=params.shallower_is_smaller)
        layers: list[TrapLayer] = []
        for idx, closure in enumerate(closures, start=1):
            ctx.check_cancel()
            score = float(min(1.0, closure.relief_samples / max(z_range, 1e-6)))
            boundary = np.column_stack(
                [
                    closure.boundary_ij[:, 0],
                    closure.boundary_ij[:, 1],
                    np.full(closure.boundary_ij.shape[0], closure.spill_level, dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            high_i, high_j = closure.high_ij
            layers.append(
                TrapLayer(
                    name=f"闭合@({high_i},{high_j})",
                    boundary=boundary,
                    score=score,
                    attributes={
                        "high_inline": high_i,
                        "high_xline": high_j,
                        "spill_level": closure.spill_level,
                        "relief_samples": closure.relief_samples,
                        "relief_m": closure.relief_samples * float(params.z_step_m),
                        "area_cells": closure.area_cells,
                        "edge_limited": closure.edge_limited,
                        "source_horizon": layer.name,
                    },
                    color=_trap_color(score),
                    opacity=0.9,
                    metadata={"algorithm": self.id, "source_horizon_id": layer.id},
                    provenance={"source": f"algorithm.{self.id}", "horizon_id": layer.id},
                )
            )
            ctx.report_progress(0.2 + 0.75 * idx / len(closures), "构建闭合图层")

        max_relief = max(float(trap.attributes.get("relief_m", 0.0)) for trap in layers)
        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=layers,
            summary=f"闭合等值线：{len(layers)} 个闭合，最大闭合高度 {max_relief:.1f} m",
        )


def _oriented_range(z: np.ndarray, valid: np.ndarray, *, shallower_is_smaller: bool) -> float:
    values = np.asarray(z, dtype=np.float32)
    oriented = values if shallower_is_smaller else -values
    finite = oriented[np.asarray(valid, dtype=bool) & np.isfinite(oriented)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(finite) - np.nanmin(finite))


def _trap_color(score: float) -> tuple[float, float, float, float]:
    s = float(np.clip(score, 0.0, 1.0))
    return (0.45 + 0.50 * s, 0.45 * (1.0 - s), 0.45 * (1.0 - s), 0.9)
