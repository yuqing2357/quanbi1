"""Dataset export helpers for confirmed targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .model import BUILTIN_TARGET_TYPES, GeoTarget, TargetSet, TargetStatus
from .store import TargetStore


def _bbox_xywh(bbox: tuple[float, float, float, float] | None) -> list[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = bbox
    return [float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0))]


def export_stage_to_coco(
    store: TargetStore,
    target_set: TargetSet,
    output_dir: str | Path,
    *,
    split_strategy: str = "spatial",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, Any]:
    """Export every (non-deleted) target in the set to a COCO-style dataset.

    Used for the dedicated training stage, where membership itself means "in the
    training set" — there is no per-target confirmed/edited gate.
    """
    return export_confirmed_to_coco(
        store,
        target_set,
        output_dir,
        split_strategy=split_strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        include=lambda target: target.status != TargetStatus.DELETED,
    )


def export_confirmed_to_coco(
    store: TargetStore,
    target_set: TargetSet,
    output_dir: str | Path,
    *,
    split_strategy: str = "spatial",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    include: Callable[[GeoTarget], bool] | None = None,
) -> dict[str, Any]:
    """Export confirmed targets to a compact COCO-style dataset.

    The internal source of truth remains ``targets.json`` plus ``.npy`` masks.
    This exporter creates PNG masks and a COCO JSON file for training and
    interchange. ``include`` overrides the default "confirmed or edited" filter
    (see :func:`export_stage_to_coco`).
    """

    output = Path(output_dir)
    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    target_types = list(dict.fromkeys([*BUILTIN_TARGET_TYPES, *target_set.target_types]))
    category_id = {name: idx + 1 for idx, name in enumerate(target_types)}
    categories = [{"id": idx, "name": name} for name, idx in category_id.items()]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 1
    annotation_id = 1
    records: list[dict[str, Any]] = []

    default_include = lambda target: target.status == TargetStatus.CONFIRMED or bool(target.edits)
    accept = include or default_include
    for target in target_set.targets.values():
        if not accept(target):
            continue
        cat_id = category_id.setdefault(target.type, len(category_id) + 1)
        if not any(category["name"] == target.type for category in categories):
            categories.append({"id": cat_id, "name": target.type})
        for frame_key, frame in sorted(target.frames.items()):
            if not frame.mask_ref:
                continue
            mask = np.asarray(store.read_mask(frame.mask_ref))
            if mask.ndim != 2:
                continue
            height, width = mask.shape
            safe_key = frame_key.replace(":", "_")
            file_name = f"masks/{target.id}_{safe_key}.png"
            Image.fromarray(((mask > 0).astype(np.uint8) * 255), mode="L").save(output / file_name)
            records.append(
                {
                    "target": target,
                    "frame": frame,
                    "cat_id": cat_id,
                    "file_name": file_name,
                    "width": int(width),
                    "height": int(height),
                }
            )

    splits = split_frames(
        [(str(record["frame"].axis), int(record["frame"].index)) for record in records],
        strategy=split_strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    for record, split in zip(records, splits, strict=False):
        target = record["target"]
        frame = record["frame"]
        if split_strategy == "round_robin":
            split = _split_for_image(image_id)
        images.append(
            {
                "id": image_id,
                "file_name": str(record["file_name"]),
                "width": int(record["width"]),
                "height": int(record["height"]),
                "split": split,
                "target_id": target.id,
                "axis": frame.axis,
                "index": frame.index,
                "source_image": frame.image_ref,
            }
        )
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(record["cat_id"]),
                "target_id": target.id,
                "bbox": _bbox_xywh(frame.bbox),
                "area": int(frame.area_px),
                "split": split,
                "iscrowd": 0,
                "segmentation": [],
                "mask_file": str(record["file_name"]),
            }
        )
        image_id += 1
        annotation_id += 1

    payload = {
        "info": {
            "schema_version": 1,
            "project": target_set.project,
            "volume_id": target_set.volume_id,
            "format": "yj_studio_coco_masks_v1",
            "split_strategy": split_strategy,
            "splits": {
                "train": max(0.0, 1.0 - float(val_fraction) - float(test_fraction)),
                "val": float(val_fraction),
                "test": float(test_fraction),
            },
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (output / "annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def split_frames(
    frame_locations: list[tuple[str, int]],
    *,
    strategy: str = "spatial",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> list[str]:
    if strategy == "round_robin":
        return [_split_for_image(index + 1) for index in range(len(frame_locations))]
    if strategy != "spatial":
        raise ValueError(f"Unsupported split strategy: {strategy}")
    val_fraction = max(0.0, min(float(val_fraction), 1.0))
    test_fraction = max(0.0, min(float(test_fraction), 1.0))
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    splits = ["train"] * len(frame_locations)
    by_axis: dict[str, list[tuple[int, int]]] = {}
    for position, (axis, frame_index) in enumerate(frame_locations):
        by_axis.setdefault(str(axis), []).append((int(frame_index), position))

    for items in by_axis.values():
        unique_indices = sorted({index for index, _ in items})
        n_indices = len(unique_indices)
        if n_indices < 3:
            continue
        test_count = _block_count(n_indices, test_fraction)
        val_count = _block_count(n_indices - test_count, val_fraction)
        train_count = max(1, n_indices - val_count - test_count)
        val_start = train_count
        test_start = train_count + val_count
        index_to_split: dict[int, str] = {}
        for order, frame_index in enumerate(unique_indices):
            if order >= test_start:
                index_to_split[frame_index] = "test"
            elif order >= val_start:
                index_to_split[frame_index] = "val"
            else:
                index_to_split[frame_index] = "train"
        for frame_index, position in items:
            splits[position] = index_to_split[frame_index]
    return splits


def _split_for_image(image_id: int) -> str:
    bucket = (int(image_id) - 1) % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"


def _block_count(n_items: int, fraction: float) -> int:
    if fraction <= 0.0 or n_items <= 0:
        return 0
    return max(1, int(round(float(n_items) * float(fraction))))
