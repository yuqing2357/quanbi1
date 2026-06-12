"""Sandbody extraction from porosity or lithology volumes."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.builtin.connectivity import BodyResult, cuboid_mesh_from_bbox, detect_bodies
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import LithBodyLayer, VolumeLayer


class SandbodyExtractParams(BaseModel):
    porosity_cutoff: float = Field(default=0.1, ge=0.0, le=1.0, description="孔隙度下限。")
    use_lithology: bool = Field(default=False, description="True 时用岩性码提砂。")
    sand_code: int = Field(default=1, description="砂岩岩性码。")
    connectivity: int = Field(default=1, ge=1, le=3, description="1=6 邻，2=18 邻，3=26 邻。")
    min_voxels: int = Field(default=64, ge=1, description="最小砂体体素数。")
    top_k: int = Field(default=20, ge=1, description="最多输出砂体数。")
    cell_volume_m3: float = Field(default=1.0, gt=0.0, description="单体素体积。")
    name_prefix: str = Field(default="Sand", description="输出图层名称前缀。")


def summarize_body(body: BodyResult, porosity: np.ndarray, cell_volume_m3: float) -> dict[str, object]:
    cells = body.cells
    mean_porosity = float("nan")
    if cells is not None and cells.size:
        values = np.asarray(porosity)[tuple(cells.T)]
        values = values[np.isfinite(values)]
        if values.size:
            mean_porosity = float(np.mean(values))
    return {
        "voxel_count": body.voxel_count,
        "volume_m3": float(body.voxel_count) * float(cell_volume_m3),
        "mean_porosity": mean_porosity,
        "bbox": list(body.bbox),
        "centroid": list(body.centroid),
    }


@register_algorithm
class SandbodyExtractAlgorithm(Algorithm):
    id: ClassVar[str] = "reservoir.sandbody_extract"
    category: ClassVar[str] = "reservoir"
    label: ClassVar[str] = "砂体提取"
    description: ClassVar[str] = "用孔隙度阈值或岩性码提取三维砂体，并输出体积与平均孔隙度。"
    input_schema: ClassVar[type[BaseModel]] = SandbodyExtractParams
    layer_inputs: ClassVar[dict[str, str]] = {"porosity": "volume", "lithology": "volume?"}
    runs_in_subprocess: ClassVar[bool] = False

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        params = ctx.params if isinstance(ctx.params, SandbodyExtractParams) else SandbodyExtractParams.model_validate(ctx.params or {})
        porosity_layer = ctx.input_layers.get("porosity")
        if not isinstance(porosity_layer, VolumeLayer):
            return AlgorithmResult.failure("缺少孔隙度体数据")
        try:
            porosity = _volume_from_layer(porosity_layer, ctx.services)
        except (KeyError, ValueError) as exc:
            return AlgorithmResult.failure(str(exc))

        ctx.report_progress(0.15, "构建砂体二值体")
        if params.use_lithology:
            lithology_layer = ctx.input_layers.get("lithology")
            if not isinstance(lithology_layer, VolumeLayer):
                return AlgorithmResult.failure("启用岩性提砂时需要选择岩性体图层")
            try:
                lithology = _volume_from_layer(lithology_layer, ctx.services)
            except (KeyError, ValueError) as exc:
                return AlgorithmResult.failure(str(exc))
            if lithology.shape != porosity.shape:
                return AlgorithmResult.failure(f"孔隙度体与岩性体形状不一致：{porosity.shape} vs {lithology.shape}")
            binary = np.asarray(lithology) == int(params.sand_code)
            source_mode = "lithology"
        else:
            binary = np.asarray(porosity) >= float(params.porosity_cutoff)
            source_mode = "porosity"

        ctx.report_progress(0.35, "连通砂体")
        bodies = detect_bodies(
            binary,
            connectivity=int(params.connectivity),
            min_voxels=int(params.min_voxels),
            top_k=int(params.top_k),
            include_cells=True,
        )
        if not bodies:
            return AlgorithmResult.failure("没有满足阈值的砂体")

        summaries = [summarize_body(body, porosity, float(params.cell_volume_m3)) for body in bodies]
        output_layers: list[LithBodyLayer] = []
        total_volume = 0.0
        for order, (body, summary) in enumerate(zip(bodies, summaries, strict=False), start=1):
            ctx.check_cancel()
            vertices, faces = cuboid_mesh_from_bbox(body.bbox)
            volume_m3 = float(summary["volume_m3"])
            total_volume += volume_m3
            output_layers.append(
                LithBodyLayer(
                    name=f"{params.name_prefix}-{order}",
                    class_value=int(params.sand_code if params.use_lithology else order),
                    class_name="sandbody",
                    vertices=vertices,
                    faces=faces,
                    color=_sand_color(order),
                    opacity=0.42,
                    metadata={
                        "algorithm": self.id,
                        "source_mode": source_mode,
                        "porosity_layer_id": porosity_layer.id,
                        "lithology_layer_id": getattr(ctx.input_layers.get("lithology"), "id", ""),
                        "porosity_cutoff": float(params.porosity_cutoff),
                        "sand_code": int(params.sand_code),
                        **summary,
                    },
                    provenance={"source": f"algorithm.{self.id}"},
                )
            )
            ctx.report_progress(0.35 + 0.6 * order / len(bodies), "构建砂体图层")

        max_phi = max(
            (float(item["mean_porosity"]) for item in summaries if np.isfinite(float(item["mean_porosity"]))),
            default=float("nan"),
        )
        max_phi_text = f"{max_phi:.3f}" if np.isfinite(max_phi) else "n/a"
        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=output_layers,
            summary=f"砂体：{len(output_layers)} 个，总体积 {total_volume:.1f} m3，最大平均孔隙度 {max_phi_text}",
        )


def _volume_from_layer(layer: VolumeLayer, services: dict[str, object]) -> np.ndarray:
    volume_store = services.get("volume_store")
    if volume_store is None or not callable(getattr(volume_store, "get_volume", None)):
        raise KeyError("缺少 volume_store 服务，无法读取体数据")
    volume = np.asarray(volume_store.get_volume(layer.volume_id))
    if volume.ndim != 3:
        raise ValueError(f"体数据必须为三维，当前 shape={volume.shape}")
    return volume


def _sand_color(order: int) -> tuple[float, float, float, float]:
    palette = (
        (0.94, 0.72, 0.24),
        (0.46, 0.72, 0.42),
        (0.82, 0.48, 0.30),
        (0.34, 0.62, 0.78),
    )
    r, g, b = palette[(order - 1) % len(palette)]
    return (r, g, b, 0.62)
