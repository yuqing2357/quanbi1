"""Volumetric evaluation for structural trap candidates."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field
from skimage.draw import polygon

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import HorizonLayer, MeasurementLayer, TrapLayer, VolumeLayer


class TrapEvaluateParams(BaseModel):
    net_to_gross: float = Field(default=0.6, ge=0.0, le=1.0, description="净毛比 NTG。")
    water_saturation: float = Field(default=0.3, ge=0.0, le=1.0, description="含水饱和度 Sw。")
    cell_area_m2: float = Field(default=625.0, gt=0.0, description="单网格平面面积。")
    z_step_m: float = Field(default=1.0, gt=0.0, description="采样点到米的换算。")
    formation_volume_factor: float = Field(default=1.1, gt=0.0, description="地层体积系数 Bo。")
    default_porosity: float = Field(default=0.2, ge=0.0, le=1.0, description="未提供孔隙度体时使用的孔隙度。")
    name: str = Field(default="圈闭评价", description="输出测量图层名称。")


def rasterize_closure(boundary_ij: np.ndarray, grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.asarray(boundary_ij, dtype=np.float32)
    if boundary.ndim != 2 or boundary.shape[0] < 3 or boundary.shape[1] < 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    rows, cols = polygon(boundary[:, 0], boundary[:, 1], shape=grid_shape)
    return rows.astype(np.int32, copy=False), cols.astype(np.int32, copy=False)


def volumetrics(
    rows: np.ndarray,
    cols: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    porosity: np.ndarray | None,
    params: TrapEvaluateParams,
    *,
    spill_level: float | None = None,
) -> dict[str, float]:
    rr = np.asarray(rows, dtype=np.int64)
    cc = np.asarray(cols, dtype=np.int64)
    top_arr = np.asarray(top, dtype=np.float32)
    bot_arr = np.asarray(bottom, dtype=np.float32)
    valid = (
        (rr >= 0)
        & (rr < top_arr.shape[0])
        & (cc >= 0)
        & (cc < top_arr.shape[1])
        & np.isfinite(top_arr[rr, cc])
        & np.isfinite(bot_arr[rr, cc])
    )
    rr = rr[valid]
    cc = cc[valid]
    if rr.size == 0:
        return _empty_values(float(params.default_porosity))

    top_values = top_arr[rr, cc]
    bottom_values = bot_arr[rr, cc]
    if spill_level is not None and np.isfinite(float(spill_level)):
        bottom_values = np.minimum(bottom_values, float(spill_level))
    gross_samples = np.maximum(bottom_values - top_values, 0.0)
    positive = gross_samples > 0.0
    rr = rr[positive]
    cc = cc[positive]
    top_values = top_values[positive]
    bottom_values = bottom_values[positive]
    gross_samples = gross_samples[positive]
    if rr.size == 0:
        return _empty_values(float(params.default_porosity))

    gross_thickness_m = gross_samples * float(params.z_step_m)
    grv = float(np.sum(gross_thickness_m) * float(params.cell_area_m2))
    mean_phi = (
        _mean_porosity_interval(rr, cc, top_values, bottom_values, porosity, float(params.default_porosity))
        if porosity is not None
        else float(params.default_porosity)
    )
    hcpv = grv * float(params.net_to_gross) * mean_phi * (1.0 - float(params.water_saturation))
    stoiip = hcpv / float(params.formation_volume_factor)
    return {
        "GRV": grv,
        "HCPV": float(hcpv),
        "STOIIP": float(stoiip),
        "area_km2": float(rr.size * float(params.cell_area_m2) / 1_000_000.0),
        "mean_phi": float(mean_phi),
        "gross_thickness_mean_m": float(np.mean(gross_thickness_m)),
        "cell_count": float(rr.size),
    }


@register_algorithm
class TrapEvaluateAlgorithm(Algorithm):
    id: ClassVar[str] = "trap.evaluate"
    category: ClassVar[str] = "trap"
    label: ClassVar[str] = "圈闭评估"
    description: ClassVar[str] = "对圈闭多边形做容积法估算，输出 GRV、HCPV 和 STOIIP。"
    input_schema: ClassVar[type[BaseModel]] = TrapEvaluateParams
    layer_inputs: ClassVar[dict[str, str]] = {"trap": "trap", "top": "horizon", "bottom": "horizon", "porosity": "volume?"}
    runs_in_subprocess: ClassVar[bool] = False

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        params = ctx.params if isinstance(ctx.params, TrapEvaluateParams) else TrapEvaluateParams.model_validate(ctx.params or {})
        trap = ctx.input_layers.get("trap")
        top_layer = ctx.input_layers.get("top")
        bottom_layer = ctx.input_layers.get("bottom")
        if not isinstance(trap, TrapLayer) or trap.boundary is None:
            return AlgorithmResult.failure("缺少圈闭边界")
        if not isinstance(top_layer, HorizonLayer) or top_layer.sample is None:
            return AlgorithmResult.failure("缺少顶部层位")
        if not isinstance(bottom_layer, HorizonLayer) or bottom_layer.sample is None:
            return AlgorithmResult.failure("缺少底部层位")
        if top_layer.sample.shape != bottom_layer.sample.shape:
            return AlgorithmResult.failure(f"顶底层位形状不一致：{top_layer.sample.shape} vs {bottom_layer.sample.shape}")

        ctx.report_progress(0.2, "栅格化圈闭")
        rows, cols = rasterize_closure(np.asarray(trap.boundary)[:, :2], tuple(top_layer.sample.shape))
        if rows.size == 0:
            return AlgorithmResult.failure("圈闭边界没有覆盖有效网格")

        porosity = None
        porosity_layer = ctx.input_layers.get("porosity")
        if isinstance(porosity_layer, VolumeLayer):
            try:
                porosity = _volume_from_layer(porosity_layer, ctx.services)
            except (KeyError, ValueError) as exc:
                return AlgorithmResult.failure(str(exc))
            if porosity.shape[:2] != top_layer.sample.shape:
                return AlgorithmResult.failure(f"孔隙度体平面尺寸与层位不一致：{porosity.shape[:2]} vs {top_layer.sample.shape}")

        ctx.report_progress(0.65, "计算容积参数")
        spill = _maybe_float((trap.attributes or {}).get("spill_level"))
        values = volumetrics(
            rows,
            cols,
            top_layer.sample,
            bottom_layer.sample,
            porosity,
            params,
            spill_level=spill,
        )
        measurement = MeasurementLayer(
            name=params.name or f"{trap.name} 评价",
            geometry=np.asarray(trap.boundary, dtype=np.float32),
            values=values,
            units={
                "GRV": "m3",
                "HCPV": "m3",
                "STOIIP": "m3",
                "area_km2": "km2",
                "mean_phi": "ratio",
                "gross_thickness_mean_m": "m",
                "cell_count": "cells",
            },
            color=(0.90, 0.42, 0.20, 0.75),
            opacity=0.75,
            metadata={
                "algorithm": self.id,
                "trap_id": trap.id,
                "top_horizon_id": top_layer.id,
                "bottom_horizon_id": bottom_layer.id,
                "porosity_layer_id": getattr(porosity_layer, "id", ""),
                "net_to_gross": float(params.net_to_gross),
                "water_saturation": float(params.water_saturation),
                "cell_area_m2": float(params.cell_area_m2),
                "z_step_m": float(params.z_step_m),
                "formation_volume_factor": float(params.formation_volume_factor),
            },
            provenance={"source": f"algorithm.{self.id}", "trap_id": trap.id},
        )

        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=[measurement],
            summary=(
                f"圈闭评价：GRV={values['GRV']:.1f} m3，"
                f"HCPV={values['HCPV']:.1f} m3，STOIIP={values['STOIIP']:.1f} m3"
            ),
        )


def _volume_from_layer(layer: VolumeLayer, services: dict[str, object]) -> np.ndarray:
    volume_store = services.get("volume_store")
    if volume_store is None or not callable(getattr(volume_store, "get_volume", None)):
        raise KeyError("缺少 volume_store 服务，无法读取孔隙度体")
    volume = np.asarray(volume_store.get_volume(layer.volume_id))
    if volume.ndim != 3:
        raise ValueError(f"孔隙度体必须为三维，当前 shape={volume.shape}")
    return volume


def _mean_porosity_interval(
    rows: np.ndarray,
    cols: np.ndarray,
    top_values: np.ndarray,
    bottom_values: np.ndarray,
    porosity: np.ndarray | None,
    default: float,
) -> float:
    if porosity is None:
        return float(default)
    por = np.asarray(porosity, dtype=np.float32)
    total = 0.0
    count = 0
    nz = int(por.shape[2])
    for i, j, top, bottom in zip(rows, cols, top_values, bottom_values, strict=False):
        k0 = int(np.floor(min(float(top), float(bottom))))
        k1 = int(np.ceil(max(float(top), float(bottom))))
        k0 = max(0, min(k0, nz - 1))
        k1 = max(k0, min(k1, nz - 1))
        segment = por[int(i), int(j), k0 : k1 + 1]
        finite = segment[np.isfinite(segment)]
        if finite.size:
            total += float(np.sum(finite))
            count += int(finite.size)
    return float(total / count) if count else float(default)


def _empty_values(mean_phi: float) -> dict[str, float]:
    return {
        "GRV": 0.0,
        "HCPV": 0.0,
        "STOIIP": 0.0,
        "area_km2": 0.0,
        "mean_phi": float(mean_phi),
        "gross_thickness_mean_m": 0.0,
        "cell_count": 0.0,
    }


def _maybe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
