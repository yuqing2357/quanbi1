from __future__ import annotations

import numpy as np


def bbox_corners(bbox: tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    return np.asarray(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=np.float32,
    )


def distance_point_to_bbox(point: tuple[float, float, float], bbox: tuple[float, float, float, float, float, float]) -> float:
    x, y, z = point
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    dz = max(zmin - z, 0.0, z - zmax)
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def distance_point_to_points(point: tuple[float, float, float], points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    if pts.shape[1] < 3:
        pts = np.column_stack([pts, np.zeros((pts.shape[0], 3 - pts.shape[1]), dtype=np.float32)])
    delta = pts[:, :3] - np.asarray(point, dtype=np.float32)
    return float(np.sqrt(np.sum(delta * delta, axis=1)).min())


def distance_point_to_polyline(point: tuple[float, float, float], points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    if pts.shape[0] == 1:
        return distance_point_to_points(point, pts)
    best = float("inf")
    p = np.asarray(point, dtype=np.float32)
    for left, right in zip(pts[:-1], pts[1:]):
        segment = right - left
        denom = float(np.dot(segment, segment))
        if denom <= 1.0e-12:
            best = min(best, float(np.linalg.norm(p - left)))
            continue
        t = float(np.dot(p - left, segment) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        candidate = left + segment * t
        best = min(best, float(np.linalg.norm(p - candidate)))
    return best


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0
    xy = pts[:, :2]
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_rect(point: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def points_in_rect(points: np.ndarray, rect: tuple[float, float, float, float]) -> bool:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return False
    x0, y0, x1, y1 = rect
    xs = pts[:, 0]
    ys = pts[:, 1]
    return bool(np.any((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)))
