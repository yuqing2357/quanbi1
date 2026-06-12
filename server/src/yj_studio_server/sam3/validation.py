from __future__ import annotations

from typing import Any


def validate_sam3_payload(
    payload: dict[str, Any],
    *,
    kind: str,
    max_boxes: int = 50,
    max_points: int = 200,
    max_keep_top_k: int = 50,
    max_track_frames: int = 5000,
    max_batch_frames: int = 5000,
) -> None:
    """Validate user-facing SAM3 requests before they enter the job queue."""
    if kind not in {"segment", "track", "batch", "infer_volume"}:
        raise ValueError("SAM3 job kind must be 'segment', 'track', 'batch', or 'infer_volume'")

    keep_top_k = _optional_int(payload, "keep_top_k", default=3)
    if keep_top_k < 1 or keep_top_k > max_keep_top_k:
        raise ValueError(f"keep_top_k must be between 1 and {max_keep_top_k}")

    for key in ("confidence", "confidence_threshold"):
        if key in payload:
            value = _float(payload[key], key)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1")

    prompts = payload.get("prompts")
    if prompts is None:
        prompts = {}
    if not isinstance(prompts, dict):
        raise ValueError("prompts must be an object")

    boxes = prompts.get("boxes", [])
    points = prompts.get("points", [])
    _validate_prompt_list(boxes, "prompts.boxes", width=4, max_items=max_boxes)
    _validate_prompt_list(points, "prompts.points", width=2, max_items=max_points)

    if kind == "track":
        seed, back, fwd = _parse_track_window(payload)
        if seed < 0:
            raise ValueError("track seed/index must be non-negative")
        if back < 0 or fwd < 0:
            raise ValueError("track back/fwd must be non-negative")
        if back + fwd + 1 > max_track_frames:
            raise ValueError(f"track frame window must be <= {max_track_frames}")

    if kind in {"batch", "infer_volume"}:
        n_frames = _batch_frame_count(payload)
        if n_frames < 1:
            raise ValueError("batch job requires frames, indices, or start/end")
        if n_frames > max_batch_frames:
            raise ValueError(f"batch frame count must be <= {max_batch_frames}")


def _optional_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    if key not in payload:
        return default
    try:
        return int(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _validate_prompt_list(value: Any, name: str, *, width: int, max_items: int) -> None:
    if value in (None, ""):
        return
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if len(value) > max_items:
        raise ValueError(f"{name} may contain at most {max_items} items")
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) < width:
            raise ValueError(f"{name}[{index}] must contain at least {width} numbers")
        for coord in item[:width]:
            _float(coord, f"{name}[{index}]")


def _parse_track_window(payload: dict[str, Any]) -> tuple[int, int, int]:
    idx = payload.get("index")
    if isinstance(idx, dict):
        seed = _int_value(idx.get("seed", idx.get("index", 0)), "index.seed")
        back = _int_value(idx.get("back", 0), "index.back")
        fwd = _int_value(idx.get("fwd", 0), "index.fwd")
    else:
        seed = _int_value(payload.get("seed", payload.get("index", 0)), "index")
        back = _int_value(payload.get("back", payload.get("n_back", 0)), "back")
        fwd = _int_value(payload.get("fwd", payload.get("n_fwd", 0)), "fwd")

    start = payload.get("start_index")
    end = payload.get("end_index")
    if back == 0 and fwd == 0 and start is not None and end is not None:
        lo, hi = sorted((_int_value(start, "start_index"), _int_value(end, "end_index")))
        if not lo <= seed <= hi:
            seed = (lo + hi) // 2
        back = seed - lo
        fwd = hi - seed
    return seed, back, fwd


def _int_value(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _batch_frame_count(payload: dict[str, Any]) -> int:
    frames = payload.get("frames")
    if isinstance(frames, list):
        return len(frames)
    indices = payload.get("indices")
    if isinstance(indices, list):
        return len(indices)
    start = payload.get("start_index", payload.get("start"))
    end = payload.get("end_index", payload.get("end"))
    if start is None or end is None:
        return 0
    step = _int_value(payload.get("step", 1), "step")
    if step == 0:
        raise ValueError("step cannot be zero")
    start_i = _int_value(start, "start_index")
    end_i = _int_value(end, "end_index")
    distance = end_i - start_i
    if distance == 0:
        return 1
    if (distance > 0 and step < 0) or (distance < 0 and step > 0):
        raise ValueError("step direction does not reach end_index")
    return abs(distance) // abs(step) + 1
