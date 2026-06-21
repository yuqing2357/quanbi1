"""Pydantic models for geological SAM3 targets.

The server is the authoritative store for target metadata and masks.  The
models here deliberately keep large arrays out of JSON; frames only contain
small metadata plus references to `.npy` files managed by ``TargetStore``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BUILTIN_TARGET_TYPES: tuple[str, ...] = (
    "trap",
    "turbidite",
    "fault",
    "sandbody",
    "unknown",
)

TargetType = str


class TargetStatus(str, Enum):
    ACTIVE = "active"
    TO_REVIEW = "to_review"
    LOST = "lost"
    MERGED = "merged"
    SPLIT = "split"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DELETED = "deleted"


class TargetStage(str, Enum):
    """Lifecycle stage of a target, mapped to a physically separate store.

    ``temporary`` holds AI-assistant results awaiting review (only multi-frame
    tracks). ``saved`` holds user-confirmed targets (long-term pool).
    ``training`` holds classified targets assembled into the training dataset.
    Each stage lives in its own subdirectory with its own ``next_seq`` and id
    prefix so numbering never collides across stages.
    """

    TEMPORARY = "temporary"
    SAVED = "saved"
    TRAINING = "training"


# Stage -> on-disk subdirectory under ``<root>/<project>/``.
STAGE_SUBDIR: dict[TargetStage, str] = {
    TargetStage.TEMPORARY: "temp",
    TargetStage.SAVED: "saved",
    TargetStage.TRAINING: "training",
}

# Stage -> id prefix (e.g. TMP1, SAV1, TRN1).
STAGE_PREFIX: dict[TargetStage, str] = {
    TargetStage.TEMPORARY: "TMP",
    TargetStage.SAVED: "SAV",
    TargetStage.TRAINING: "TRN",
}


def coerce_stage(value: "str | TargetStage | None", default: TargetStage = TargetStage.SAVED) -> TargetStage:
    """Best-effort parse of a stage from a string/enum, falling back to ``default``."""
    if isinstance(value, TargetStage):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    for stage in TargetStage:
        if text in (stage.value, STAGE_SUBDIR[stage], STAGE_PREFIX[stage].lower()):
            return stage
    return default


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_target_type(value: str | None) -> str:
    target_type = (value or "unknown").strip().lower()
    return target_type or "unknown"


def frame_key(axis: Literal["inline", "crossline", "timeslice"], index: int) -> str:
    return f"{axis}:{int(index)}"


class TargetFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")

    axis: Literal["inline", "crossline", "timeslice"]
    index: int
    mask_ref: str | None = None
    cell_ids_ref: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    centroid: tuple[float, float] | None = None
    area_px: int = 0
    score: float | None = None
    status: TargetStatus = TargetStatus.ACTIVE
    origin: str = "sam3"
    image_ref: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @property
    def key(self) -> str:
        return frame_key(self.axis, self.index)


class GeoTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    type: TargetType = "unknown"
    name: str | None = None
    volume_id: str | None = None
    status: TargetStatus = TargetStatus.ACTIVE
    source: str = "sam3"
    frames: dict[str, TargetFrame] = Field(default_factory=dict)
    trajectory: list[str] = Field(default_factory=list)
    score: float | None = None
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    merged_into: str | None = None
    edits: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("type")
    @classmethod
    def _normalise_type(cls, value: str) -> str:
        return normalise_target_type(value)

    def add_frame(self, frame: TargetFrame) -> None:
        key = frame.key
        self.frames[key] = frame
        if key not in self.trajectory:
            self.trajectory.append(key)
        self.updated_at = utc_now_iso()

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def area_px(self) -> int:
        return int(sum(max(0, frame.area_px) for frame in self.frames.values()))

    @property
    def frame_range(self) -> str:
        if not self.frames:
            return ""
        axes = {frame.axis for frame in self.frames.values()}
        if len(axes) != 1:
            return "mixed"
        axis = next(iter(axes))
        indices = sorted(frame.index for frame in self.frames.values())
        return f"{axis}:{indices[0]}" if len(indices) == 1 else f"{axis}:{indices[0]}-{indices[-1]}"


class TargetSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project: str = "default"
    volume_id: str | None = None
    schema_version: int = 1
    version: int = 1
    next_seq: int = 1
    # Id prefix for this set's stage (e.g. "TMP"/"SAV"/"TRN"; legacy "T").
    # ``new_id`` and the sequence-advance validator both key off this so each
    # stage's numbering is independent and never collides.
    id_prefix: str = "T"
    targets: dict[str, GeoTarget] = Field(default_factory=dict)
    target_types: list[str] = Field(default_factory=lambda: list(BUILTIN_TARGET_TYPES))
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _advance_sequence_past_existing_ids(self) -> "TargetSet":
        """Keep migrated/stale stores from reusing historical numeric ids."""

        prefix = self.id_prefix or "T"
        numeric_ids = []
        for target_id in self.targets:
            if target_id.startswith(prefix):
                suffix = target_id[len(prefix):]
                if suffix.isdigit():
                    numeric_ids.append(int(suffix))
        if numeric_ids:
            self.next_seq = max(int(self.next_seq), max(numeric_ids) + 1)
        return self

    def new_id(self, prefix: str | None = None) -> str:
        prefix = prefix or self.id_prefix or "T"
        while True:
            target_id = f"{prefix}{self.next_seq}"
            self.next_seq += 1
            if target_id not in self.targets:
                self.updated_at = utc_now_iso()
                return target_id

    def add_target(self, target: GeoTarget) -> GeoTarget:
        self.targets[target.id] = target
        if target.type not in self.target_types:
            self.target_types.append(target.type)
        self.updated_at = utc_now_iso()
        return target

    def remove_target(self, target_id: str) -> GeoTarget | None:
        target = self.targets.pop(target_id, None)
        if target is not None:
            self.updated_at = utc_now_iso()
        return target

    def get_required(self, target_id: str) -> GeoTarget:
        try:
            return self.targets[target_id]
        except KeyError as exc:
            raise KeyError(f"Unknown target id: {target_id}") from exc

    def summaries(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for target in self.targets.values():
            if target.status == TargetStatus.DELETED and not include_deleted:
                continue
            rows.append(
                {
                    "id": target.id,
                    "name": target.name,
                    "type": target.type,
                    "status": target.status.value,
                    "frame_range": target.frame_range,
                    "frame_count": target.frame_count,
                    "area_px": target.area_px,
                    "score": target.score,
                    "volume_id": target.volume_id,
                    "updated_at": target.updated_at,
                }
            )
        return sorted(rows, key=lambda row: _target_id_sort_key(str(row["id"])))


def _target_id_sort_key(target_id: str) -> tuple[str, int, str]:
    prefix = target_id.rstrip("0123456789")
    suffix = target_id[len(prefix) :]
    number = int(suffix) if suffix else -1
    return prefix, number, target_id
