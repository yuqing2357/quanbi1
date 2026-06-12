from __future__ import annotations

import numpy as np

from .model import normalise_target_type


TARGET_TYPE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "trap": (0.95, 0.22, 0.18, 0.58),
    "turbidite": (0.15, 0.65, 0.95, 0.58),
    "fault": (0.10, 0.78, 0.48, 0.58),
    "sandbody": (1.00, 0.76, 0.18, 0.58),
    "unknown": (0.85, 0.35, 0.92, 0.55),
}


def target_type_color(target_type: str | None, *, alpha: float | None = None) -> tuple[float, float, float, float]:
    rgba = TARGET_TYPE_COLORS.get(normalise_target_type(target_type), TARGET_TYPE_COLORS["unknown"])
    if alpha is None:
        return rgba
    return (rgba[0], rgba[1], rgba[2], float(alpha))


def mask_summary(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return {"area_px": 0, "bbox": None, "centroid": None}
    return {
        "area_px": int(xs.size),
        "bbox": (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        "centroid": (float(xs.mean()), float(ys.mean())),
    }
