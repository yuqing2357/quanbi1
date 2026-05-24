"""Measure-as-algorithm wrappers.

The interactive ``MeasureTool`` already writes ``MeasurementLayer`` directly
to the LayerStore. These algorithms expose the same computation as
:class:`Algorithm` instances so they appear in the AlgorithmDock listing and
can be replayed from a saved project — for example, "recompute area for the
selected polygon after the user edited its vertices."
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.config.defaults import DEPTH_STEP_TO_SAMPLE
from yj_studio.scene.layers import (
    HorizonStickLayer,
    MeasurementLayer,
    PolygonLayer,
)


class DistanceParams(BaseModel):
    depth_step_m: float = Field(default=DEPTH_STEP_TO_SAMPLE, gt=0.0)
    name: str = Field(default="Polyline length")


class DistanceOutput(BaseModel):
    total_xy: float
    total_3d_m: float
    segment_count: int


@register_algorithm
class MeasureDistanceAlgorithm(Algorithm):
    id: ClassVar[str] = "measure.distance"
    category: ClassVar[str] = "measure"
    label: ClassVar[str] = "Polyline Distance"
    description: ClassVar[str] = (
        "Measure the total length of a polyline / horizon-stick. Reports both"
        " the planar (inline/xline) distance and the 3D distance with z"
        " expressed in metres via depth_step_m."
    )
    input_schema: ClassVar[type[BaseModel]] = DistanceParams
    output_schema: ClassVar[type[BaseModel]] = DistanceOutput
    layer_inputs: ClassVar[dict[str, str]] = {"path": "polygon|horizon_stick"}

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        layer = ctx.input_layers.get("path")
        points = _points_for(layer)
        if points is None or points.shape[0] < 2:
            return AlgorithmResult.failure("Path layer needs at least 2 points")

        ctx.report_progress(0.3, "Computing segments")
        deltas = np.diff(points, axis=0)
        xy_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        z_lengths_m = np.abs(deltas[:, 2]) * float(ctx.params.depth_step_m)
        seg_3d = np.sqrt(xy_lengths ** 2 + z_lengths_m ** 2)

        stats = DistanceOutput(
            total_xy=float(np.sum(xy_lengths)),
            total_3d_m=float(np.sum(seg_3d)),
            segment_count=int(deltas.shape[0]),
        )

        measurement = MeasurementLayer(
            name=ctx.params.name or "Polyline length",
            geometry=points.astype(np.float32),
            values={"total_xy": stats.total_xy, "total_3d_m": stats.total_3d_m},
            units={"total_xy": "cells", "total_3d_m": "m"},
            color=(0.2, 0.85, 0.4, 0.9),
            opacity=0.9,
            metadata={"algorithm": MeasureDistanceAlgorithm.id, "source_layer": layer.name},
            provenance={"source": "algorithm.measure.distance", "input_layer_id": layer.id},
        )
        ctx.report_progress(1.0, "Done")
        return AlgorithmResult.success(
            output_layers=[measurement],
            summary=f"Distance: {stats.total_3d_m:.1f} m over {stats.segment_count} segments",
        )


class AreaParams(BaseModel):
    depth_step_m: float = Field(default=DEPTH_STEP_TO_SAMPLE, gt=0.0)
    name: str = Field(default="Polygon area")


class AreaOutput(BaseModel):
    area_xy: float
    perimeter_xy: float
    vertex_count: int


@register_algorithm
class MeasureAreaAlgorithm(Algorithm):
    id: ClassVar[str] = "measure.area"
    category: ClassVar[str] = "measure"
    label: ClassVar[str] = "Polygon Area"
    description: ClassVar[str] = (
        "Compute the planar (inline/xline) area and perimeter of a closed"
        " polygon using the shoelace formula. Z is ignored."
    )
    input_schema: ClassVar[type[BaseModel]] = AreaParams
    output_schema: ClassVar[type[BaseModel]] = AreaOutput
    layer_inputs: ClassVar[dict[str, str]] = {"polygon": "polygon"}

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        layer = ctx.input_layers.get("polygon")
        if not isinstance(layer, PolygonLayer) or layer.vertices is None:
            return AlgorithmResult.failure("Need a PolygonLayer with vertices")
        verts = np.asarray(layer.vertices, dtype=np.float32)
        if verts.shape[0] < 3:
            return AlgorithmResult.failure("Polygon needs at least 3 vertices")

        ctx.report_progress(0.4, "Computing area")
        x = verts[:, 0]
        y = verts[:, 1]
        area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        closed = np.vstack([verts, verts[:1]])
        deltas = np.diff(closed, axis=0)
        perimeter = float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))

        stats = AreaOutput(area_xy=area, perimeter_xy=perimeter, vertex_count=int(verts.shape[0]))

        measurement = MeasurementLayer(
            name=ctx.params.name or "Polygon area",
            geometry=verts,
            values={"area_xy": stats.area_xy, "perimeter_xy": stats.perimeter_xy},
            units={"area_xy": "cells^2", "perimeter_xy": "cells"},
            color=(0.95, 0.85, 0.2, 0.85),
            opacity=0.85,
            metadata={"algorithm": MeasureAreaAlgorithm.id, "source_layer": layer.name},
            provenance={"source": "algorithm.measure.area", "input_layer_id": layer.id},
        )
        ctx.report_progress(1.0, "Done")
        return AlgorithmResult.success(
            output_layers=[measurement],
            summary=f"Area: {stats.area_xy:.1f} cells² (perimeter {stats.perimeter_xy:.1f})",
        )


def _points_for(layer) -> np.ndarray | None:
    if isinstance(layer, PolygonLayer) and layer.vertices is not None:
        return np.asarray(layer.vertices, dtype=np.float32)
    if isinstance(layer, HorizonStickLayer) and layer.points is not None:
        return np.asarray(layer.points, dtype=np.float32)
    return None
