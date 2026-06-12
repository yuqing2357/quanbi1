from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraState:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0)

