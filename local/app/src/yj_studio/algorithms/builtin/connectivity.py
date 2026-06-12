"""3D connected-component analysis for volume-like data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import center_of_mass, find_objects, generate_binary_structure, label

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import LithBodyLayer, MaskLayer, VolumeLayer

Comparator = Literal[">=", "<=", ">", "<", "=="]


class ConnectivityParams(BaseModel):
    threshold: float = Field(default=0.5, description="二值化阈值。")
    comparator: Comparator = Field(default=">=", description="阈值比较符。")
    connectivity: int = Field(default=1, ge=1, le=3, description="1=6 邻，2=18 邻，3=26 邻。")
    min_voxels: int = Field(default=64, ge=1, description="最小连通体体素数。")
    top_k: int = Field(default=20, ge=1, description="最多输出的连通体数量。")
    name_prefix: str = Field(default="Component", description="输出图层名称前缀。")


@dataclass(frozen=True, slots=True)
class BodyResult:
    label_id: int
    voxel_count: int
    bbox: tuple[int, int, int, int, int, int]
    centroid: tuple[float, float, float]
    cells: np.ndarray | None = None


def threshold_volume(values: np.ndarray, threshold: float, comparator: Comparator = ">=") -> np.ndarray:
    arr = np.asarray(values)
    if comparator == ">=":
        return arr >= threshold
    if comparator == "<=":
        return arr <= threshold
    if comparator == ">":
        return arr > threshold
    if comparator == "<":
        return arr < threshold
    if comparator == "==":
        return arr == threshold
    raise ValueError(f"Unsupported comparator: {comparator}")


def detect_bodies(
    binary: np.ndarray,
    *,
    connectivity: int = 1,
    min_voxels: int = 64,
    top_k: int = 20,
    include_cells: bool = True,
) -> list[BodyResult]:
    mask = np.asarray(binary, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"binary must be a 3D array, got shape {mask.shape}")
    if not mask.any():
        return []

    structure = generate_binary_structure(3, int(connectivity))
    labels, n_labels = label(mask, structure=structure)
    if n_labels == 0:
        return []

    counts = np.bincount(labels.ravel())
    label_ids = [
        int(label_id)
        for label_id in range(1, int(n_labels) + 1)
        if int(counts[label_id]) >= int(min_voxels)
    ]
    label_ids.sort(key=lambda item: (-int(counts[item]), item))
    label_ids = label_ids[: int(top_k)]
    if not label_ids:
        return []

    objects = find_objects(labels)
    centers = center_of_mass(mask, labels, label_ids)
    if len(label_ids) == 1 and isinstance(centers, tuple):
        centers = [centers]

    bodies: list[BodyResult] = []
    for label_id, center in zip(label_ids, centers, strict=False):
        slices = objects[label_id - 1] if 0 <= label_id - 1 < len(objects) else None
        if slices is None:
            continue
        i_slice, j_slice, k_slice = slices
        bbox = (
            int(i_slice.start),
            int(i_slice.stop - 1),
            int(j_slice.start),
            int(j_slice.stop - 1),
            int(k_slice.start),
            int(k_slice.stop - 1),
        )
        cells = None
        if include_cells:
            submask = labels[slices] == label_id
            coords = np.argwhere(submask).astype(np.int32, copy=False)
            if coords.size:
                coords += np.asarray([i_slice.start, j_slice.start, k_slice.start], dtype=np.int32)
            else:
                coords = coords.reshape(0, 3)
            cells = coords
        bodies.append(
            BodyResult(
                label_id=label_id,
                voxel_count=int(counts[label_id]),
                bbox=bbox,
                centroid=(float(center[0]), float(center[1]), float(center[2])),
                cells=cells,
            )
        )
    return bodies


def cuboid_mesh_from_bbox(
    bbox: tuple[int, int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    i0, i1, j0, j1, k0, k1 = bbox
    # Use cell extents rather than centre coordinates so one-voxel bodies
    # still have a visible, non-degenerate box.
    x0, x1 = float(i0), float(i1 + 1)
    y0, y1 = float(j0), float(j1 + 1)
    z0, z1 = float(k0), float(k1 + 1)
    vertices = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, faces


@register_algorithm
class ConnectivityAlgorithm(Algorithm):
    id: ClassVar[str] = "volume.connectivity"
    category: ClassVar[str] = "volume"
    label: ClassVar[str] = "连通性分析"
    description: ClassVar[str] = "对三维体或三维掩膜做连通域标注，输出按体素数排序的体对象。"
    input_schema: ClassVar[type[BaseModel]] = ConnectivityParams
    layer_inputs: ClassVar[dict[str, str]] = {"volume": "volume|mask"}
    runs_in_subprocess: ClassVar[bool] = False

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        params = ctx.params if isinstance(ctx.params, ConnectivityParams) else ConnectivityParams.model_validate(ctx.params or {})
        source_layer = ctx.input_layers.get("volume")
        try:
            volume = _array_from_input(source_layer, ctx.services)
        except (KeyError, TypeError, ValueError) as exc:
            return AlgorithmResult.failure(str(exc))

        ctx.report_progress(0.15, "二值化体数据")
        binary = threshold_volume(volume, float(params.threshold), params.comparator)
        ctx.report_progress(0.35, "标注连通域")
        bodies = detect_bodies(
            binary,
            connectivity=int(params.connectivity),
            min_voxels=int(params.min_voxels),
            top_k=int(params.top_k),
            include_cells=False,
        )
        if not bodies:
            return AlgorithmResult.failure("没有满足阈值的连通体")

        output_layers: list[LithBodyLayer] = []
        max_voxels = max(body.voxel_count for body in bodies)
        for order, body in enumerate(bodies, start=1):
            ctx.check_cancel()
            vertices, faces = cuboid_mesh_from_bbox(body.bbox)
            score = float(body.voxel_count / max(max_voxels, 1))
            output_layers.append(
                LithBodyLayer(
                    name=f"{params.name_prefix}-{order}",
                    class_value=int(body.label_id),
                    class_name="connected_component",
                    vertices=vertices,
                    faces=faces,
                    color=_body_color(order, score),
                    opacity=0.35,
                    metadata={
                        "algorithm": self.id,
                        "source_layer_id": getattr(source_layer, "id", ""),
                        "voxel_count": body.voxel_count,
                        "bbox": list(body.bbox),
                        "centroid": list(body.centroid),
                        "threshold": float(params.threshold),
                        "comparator": params.comparator,
                        "connectivity": int(params.connectivity),
                    },
                    provenance={"source": f"algorithm.{self.id}"},
                )
            )
            ctx.report_progress(0.35 + 0.6 * order / len(bodies), "构建连通体图层")

        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=output_layers,
            summary=f"连通域：{len(output_layers)} 个，最大 {max_voxels} 体素",
        )


def _array_from_input(layer: object, services: dict[str, object]) -> np.ndarray:
    if isinstance(layer, VolumeLayer):
        volume_store = services.get("volume_store")
        if volume_store is None or not callable(getattr(volume_store, "get_volume", None)):
            raise KeyError("缺少 volume_store 服务，无法读取体数据")
        return np.asarray(volume_store.get_volume(layer.volume_id))
    if isinstance(layer, MaskLayer):
        if layer.mask is None:
            raise ValueError("掩膜图层没有 mask 数据")
        mask = np.asarray(layer.mask)
        if mask.ndim != 3:
            raise ValueError(f"连通性分析需要三维掩膜，当前 shape={mask.shape}")
        return mask
    raise TypeError("需要 VolumeLayer 或三维 MaskLayer 作为输入")


def _body_color(order: int, score: float) -> tuple[float, float, float, float]:
    palette = (
        (0.10, 0.55, 0.85),
        (0.85, 0.36, 0.20),
        (0.20, 0.65, 0.42),
        (0.72, 0.42, 0.75),
        (0.78, 0.64, 0.18),
    )
    r, g, b = palette[(order - 1) % len(palette)]
    s = float(np.clip(score, 0.25, 1.0))
    return (r * s, g * s, b * s, 0.55)
