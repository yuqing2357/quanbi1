from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from .cache import enforce_slice_cache_budget
from .config import ServerConfig, load_config
from .sam3 import JobQueue, JobState, JobStore, SAM3Engine
from .sam3.image import slice_to_rgb_image
from .sam3.models import ModelRegistry
from .sam3.reassociate import annotate_gaps, detect_merge_split, link_targets_by_iou
from .sam3.tracking import collect_object_frames, persist_tracked_targets
from .sam3.training import run_training_backend
from .sam3.validation import validate_sam3_payload
from .targets import (
    GeoTarget,
    TargetSet,
    TargetStatus,
    TargetStore,
    export_confirmed_to_coco,
    frame_key,
    normalise_target_type,
)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="YJ Studio Server", version="0.1.0")
    app.state.config = cfg
    app.state.jobs = JobStore(persist_dir=cfg.runtime_root / "jobs")
    app.state.queue = JobQueue(worker_count=int(cfg.sam3.get("worker_count", 1)))
    app.state.sam3 = _make_sam3_engine(cfg)
    app.state.models = _make_model_registry(cfg)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "project_root": str(cfg.project_root),
            "data_root": str(cfg.data_root),
            "runtime_root": str(cfg.runtime_root),
            "data_root_exists": cfg.data_root.exists(),
            "project_id": cfg.project_id,
            "slice_cache_max_gb": cfg.slice_cache_max_gb,
        }

    @app.get("/volumes")
    def volumes() -> list[dict[str, Any]]:
        return [_volume_payload(volume_id, spec, cfg.data_root) for volume_id, spec in cfg.volumes.items()]

    @app.get("/slice")
    def volume_slice(
        volume_id: str = Query(...),
        axis: Literal["inline", "xline", "z"] = Query(...),
        index: int = Query(...),
    ) -> Response:
        spec, path = _volume_spec_and_path(cfg, volume_id)
        cache_path = _slice_cache_path(cfg, volume_id, axis, index, path)
        if cache_path.exists():
            return FileResponse(
                cache_path,
                media_type="application/x-npy",
                headers={
                    "X-Volume-Id": volume_id,
                    "X-Slice-Axis": axis,
                    "X-Slice-Index": str(index),
                    "X-Slice-Cache": "hit",
                },
            )
        data, shape = _load_slice(path, axis, index)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".npy.partial")
        with tmp_path.open("wb") as handle:
            np.save(handle, data, allow_pickle=False)
        tmp_path.replace(cache_path)
        _enforce_slice_cache_budget(cfg)
        return FileResponse(
            cache_path,
            media_type="application/x-npy",
            headers={
                "X-Volume-Id": volume_id,
                "X-Slice-Axis": axis,
                "X-Slice-Index": str(index),
                "X-Volume-Shape": ",".join(str(int(v)) for v in shape),
                "X-Slice-Dtype": str(data.dtype),
                "X-Slice-Cache": "miss",
            },
        )

    @app.post("/sam3/jobs")
    def submit_sam3_job(payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind", "")).strip().lower()
        if kind not in {"segment", "track", "infer_volume"}:
            raise HTTPException(status_code=400, detail="SAM3 job kind must be 'segment', 'track', or 'infer_volume'")
        request = dict(payload)
        if kind == "infer_volume":
            request.setdefault("target_status", TargetStatus.TO_REVIEW.value)
            request.setdefault("result_kind", "infer_volume")
            _validate_sam3_request(cfg, request, kind="infer_volume")
        else:
            _validate_sam3_request(cfg, request, kind=kind)
        job = app.state.jobs.create(kind, request)
        runner = _run_track_job if kind == "track" else _run_sam3_job
        if kind == "infer_volume":
            runner = _run_sam3_batch_job
        app.state.queue.submit(runner, app, job.id)
        return {"job_id": job.id, "state": job.state.value}

    @app.get("/sam3/jobs/{job_id}")
    def sam3_job_status(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(app, job_id)
        return job.status_payload()

    @app.get("/sam3/jobs/{job_id}/result")
    def sam3_job_result(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(app, job_id)
        if job.state != JobState.done:
            raise HTTPException(status_code=409, detail=f"SAM3 job is not done: {job.state.value}")
        if job.result is None:
            raise HTTPException(status_code=404, detail="SAM3 job has no result payload")
        return job.result

    @app.get("/sam3/jobs/{job_id}/mask/{candidate_index}")
    def sam3_job_mask(job_id: str, candidate_index: int) -> Response:
        job = _get_job_or_404(app, job_id)
        if candidate_index < 0 or candidate_index >= len(job.mask_paths):
            raise HTTPException(status_code=404, detail=f"Unknown mask candidate: {candidate_index}")
        path = Path(job.mask_paths[candidate_index])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Mask file not found: {path}")
        return FileResponse(
            path,
            media_type="application/x-npy",
            headers={
                "X-SAM3-Job-Id": job.id,
                "X-SAM3-Candidate": str(candidate_index),
            },
        )

    @app.post("/sam3/jobs/{job_id}/cancel")
    def cancel_sam3_job(job_id: str) -> dict[str, Any]:
        job = app.state.jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown SAM3 job: {job_id}")
        return job.status_payload()

    @app.post("/sam3/jobs/batch")
    def submit_sam3_batch_job(payload: dict[str, Any]) -> dict[str, Any]:
        _validate_sam3_request(cfg, payload, kind="batch")
        job = app.state.jobs.create("batch", payload)
        app.state.queue.submit(_run_sam3_batch_job, app, job.id)
        return {"job_id": job.id, "state": job.state.value}

    @app.get("/sam3/gpus")
    def sam3_gpus() -> dict[str, Any]:
        return _gpu_payload(cfg)

    @app.post("/sam3/extract")
    def sam3_extract(payload: dict[str, Any]) -> dict[str, Any]:
        scope = str(payload.get("scope", "page")).strip().lower()
        target_type = normalise_target_type(str(payload.get("type", payload.get("target_type", "unknown"))))
        request = dict(payload)
        request["target_type"] = target_type
        request.setdefault("prompts", {})
        if isinstance(request["prompts"], dict):
            request["prompts"].setdefault("text", target_type)
        mode = str(payload.get("mode", "prompt")).strip().lower()
        if scope == "page":
            request["kind"] = "segment"
            _validate_sam3_request(cfg, request, kind="segment")
            job = app.state.jobs.create("segment", request)
            app.state.queue.submit(_run_sam3_job, app, job.id)
            return {"job_id": job.id, "state": job.state.value, "scope": scope}
        if scope == "volume":
            if mode == "track":
                # Whole-volume extraction with cross-frame identity: one seeded
                # multi-object track over the requested axis range.
                request["kind"] = "track"
                _validate_sam3_request(cfg, request, kind="track")
                job = app.state.jobs.create("track", request)
                app.state.queue.submit(_run_track_job, app, job.id)
                return {"job_id": job.id, "state": job.state.value, "scope": scope, "mode": mode}
            if mode in {"infer", "infer_volume", "batch_infer"}:
                request["kind"] = "infer_volume"
                request.setdefault("target_status", TargetStatus.TO_REVIEW.value)
                request.setdefault("result_kind", "infer_volume")
                _validate_sam3_request(cfg, request, kind="infer_volume")
                job = app.state.jobs.create("infer_volume", request)
                app.state.queue.submit(_run_sam3_batch_job, app, job.id)
                return {"job_id": job.id, "state": job.state.value, "scope": scope, "mode": mode}
            _validate_sam3_request(cfg, request, kind="batch")
            job = app.state.jobs.create("batch", request)
            app.state.queue.submit(_run_sam3_batch_job, app, job.id)
            return {"job_id": job.id, "state": job.state.value, "scope": scope}
        raise HTTPException(status_code=400, detail="scope must be 'page' or 'volume'")

    @app.get("/sam3/targets")
    def sam3_targets(
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
        include_deleted: bool = Query(False),
    ) -> dict[str, Any]:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target_set = store.load()
        payload = target_set.model_dump(mode="json")
        payload["summaries"] = target_set.summaries(include_deleted=include_deleted)
        return payload

    @app.get("/sam3/targets/{target_id}")
    def sam3_target(
        target_id: str,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> dict[str, Any]:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target = _get_target_or_404(store, target_id)
        return target.model_dump(mode="json")

    @app.patch("/sam3/targets/{target_id}")
    def update_sam3_target(
        target_id: str,
        payload: dict[str, Any],
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> dict[str, Any]:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        with store.mutate() as target_set:
            target = _get_target_or_404(store, target_id, target_set=target_set)
            if "name" in payload:
                target.name = str(payload.get("name") or "").strip() or None
            if "type" in payload:
                target.type = normalise_target_type(str(payload.get("type", "unknown")))
                if target.type not in target_set.target_types:
                    target_set.target_types.append(target.type)
            if "status" in payload:
                try:
                    target.status = TargetStatus(str(payload["status"]))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid target status: {payload['status']}") from exc
            if "notes" in payload:
                target.notes = str(payload.get("notes") or "").strip() or None
            target.updated_at = _utc_now_iso()
            result = target.model_dump(mode="json")
        return result

    @app.delete("/sam3/targets/{target_id}")
    def delete_sam3_target(
        target_id: str,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> dict[str, Any]:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        with store.mutate() as target_set:
            target = _get_target_or_404(store, target_id, target_set=target_set)
            target.status = TargetStatus.DELETED
            target.updated_at = _utc_now_iso()
            result = target.model_dump(mode="json")
        return result

    @app.post("/sam3/targets/merge")
    def merge_sam3_targets(
        payload: dict[str, Any],
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> dict[str, Any]:
        ids = [str(item) for item in payload.get("target_ids", []) if str(item).strip()]
        if len(ids) < 2:
            raise HTTPException(status_code=400, detail="At least two target_ids are required")
        store = _target_store(cfg, project=project, volume_id=volume_id)
        with store.mutate() as target_set:
            base = _get_target_or_404(store, ids[0], target_set=target_set)
            for target_id in ids[1:]:
                other = _get_target_or_404(store, target_id, target_set=target_set)
                for key, frame in other.frames.items():
                    base.frames.setdefault(key, frame)
                    if key not in base.trajectory:
                        base.trajectory.append(key)
                other.status = TargetStatus.MERGED
                other.merged_into = base.id
                other.updated_at = _utc_now_iso()
                if other.id not in base.child_ids:
                    base.child_ids.append(other.id)
            base.updated_at = _utc_now_iso()
            result = base.model_dump(mode="json")
        return result

    @app.post("/sam3/targets/{target_id}/split")
    def split_sam3_target(
        target_id: str,
        payload: dict[str, Any],
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> dict[str, Any]:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        with store.mutate() as target_set:
            source = _get_target_or_404(store, target_id, target_set=target_set)
            groups = payload.get("groups")
            if not groups:
                groups = [[key] for key in source.trajectory or sorted(source.frames)]
            new_targets: list[dict[str, Any]] = []
            for group in groups:
                frame_keys = [str(key) for key in group if str(key) in source.frames]
                if not frame_keys:
                    continue
                child_id = target_set.new_id()
                child = GeoTarget(
                    id=child_id,
                    type=normalise_target_type(str(payload.get("type", source.type))),
                    volume_id=source.volume_id,
                    status=TargetStatus.ACTIVE,
                    source=f"split:{source.id}",
                    parent_ids=[source.id],
                )
                for key in frame_keys:
                    frame = source.frames[key].model_copy(deep=True)
                    child.add_frame(frame)
                target_set.add_target(child)
                if child.id not in source.child_ids:
                    source.child_ids.append(child.id)
                new_targets.append(child.model_dump(mode="json"))
            if not new_targets:
                raise HTTPException(status_code=400, detail="No valid frame groups to split")
            source.status = TargetStatus.SPLIT
            source.updated_at = _utc_now_iso()
            result = {"source": source.model_dump(mode="json"), "targets": new_targets}
        return result

    @app.get("/sam3/targets/{target_id}/mask/{axis}/{index}")
    def sam3_target_mask(
        target_id: str,
        axis: str,
        index: int,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> Response:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target = _get_target_or_404(store, target_id)
        frame = target.frames.get(_target_frame_key(axis, index))
        if frame is None or not frame.mask_ref:
            raise HTTPException(status_code=404, detail=f"Target mask not found: {target_id}/{axis}/{index}")
        path = store.resolve_ref(frame.mask_ref)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Target mask file not found: {path}")
        return FileResponse(
            path,
            media_type="application/x-npy",
            headers={"X-Target-Id": target_id, "X-Target-Axis": frame.axis, "X-Target-Index": str(frame.index)},
        )

    @app.put("/sam3/targets/{target_id}/mask/{axis}/{index}")
    async def put_sam3_target_mask(
        target_id: str,
        axis: str,
        index: int,
        request: Request,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> dict[str, Any]:
        raw = await request.body()
        try:
            mask = np.load(BytesIO(raw), allow_pickle=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid .npy mask payload: {exc}") from exc
        if np.asarray(mask).ndim != 2:
            raise HTTPException(status_code=400, detail=f"Target mask must be 2D, got shape {np.asarray(mask).shape}")
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target_axis = _target_axis(axis)
        with store.mutate() as target_set:
            target = _get_target_or_404(store, target_id, target_set=target_set)
            frame = store.frame_from_mask(
                target_id=target.id,
                axis=target_axis,
                index=int(index),
                mask=np.asarray(mask),
                origin="edited",
                image_ref=f"{target.volume_id or volume_id}:{axis}:{int(index)}",
            )
            target.add_frame(frame)
            target.edits.append(
                {
                    "at": _utc_now_iso(),
                    "kind": "mask_put",
                    "axis": target_axis,
                    "index": int(index),
                    "mask_ref": frame.mask_ref,
                }
            )
            target.updated_at = _utc_now_iso()
            result = target.model_dump(mode="json")
        return result

    @app.post("/sam3/targets/cells")
    async def create_sam3_cell_target(
        request: Request,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
        axis: str = Query(...),
        index: int = Query(...),
        index_hi: int | None = Query(None),
        target_type: str = Query("unknown"),
        name: str | None = Query(None),
        source: str = Query("sam3_reservoir"),
        grid_id: str | None = Query(None),
        grid_layer_id: str | None = Query(None),
    ) -> dict[str, Any]:
        raw = await request.body()
        try:
            cells = np.asarray(np.load(BytesIO(raw), allow_pickle=False), dtype=np.int32)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid .npy cell payload: {exc}") from exc
        if cells.ndim == 1 and cells.size == 0:
            cells = cells.reshape(0, 3)
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise HTTPException(status_code=400, detail=f"Cell ids must have shape (N, 3), got {cells.shape}")
        if cells.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Cell target requires at least one cell")

        store = _target_store(cfg, project=project, volume_id=volume_id)
        try:
            target_axis = _target_axis(axis)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with store.mutate() as target_set:
            target_id = target_set.new_id()
            target = GeoTarget(
                id=target_id,
                type=normalise_target_type(target_type),
                name=str(name).strip() if name else None,
                volume_id=volume_id or target_set.volume_id,
                source=str(source or "sam3_reservoir"),
                metadata={
                    "reservoir_axis": str(axis),
                    "index_lo": int(index),
                    "index_hi": int(index_hi if index_hi is not None else index),
                    "grid_id": grid_id,
                    "grid_layer_id": grid_layer_id,
                    "cell_count": int(cells.shape[0]),
                },
            )
            frame = store.frame_from_cells(
                target_id=target_id,
                axis=target_axis,
                index=int(index),
                cells=cells,
                origin=str(source or "sam3_reservoir"),
                image_ref=f"{volume_id or target_set.volume_id or 'reservoir'}:{axis}:{int(index)}",
            )
            target.add_frame(frame)
            target_set.add_target(target)
            result = target.model_dump(mode="json")
        return result

    @app.get("/sam3/targets/{target_id}/cells")
    def sam3_target_cells(
        target_id: str,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> Response:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target = _get_target_or_404(store, target_id)
        refs = [frame.cell_ids_ref for frame in target.frames.values() if frame.cell_ids_ref]
        path = store.write_cells_union_cache(target_id, [str(ref) for ref in refs])
        return FileResponse(path, media_type="application/x-npy", headers={"X-Target-Id": target_id})

    @app.get("/sam3/targets/{target_id}/mask3d")
    def sam3_target_mask3d(
        target_id: str,
        project: str | None = Query(None),
        volume_id: str | None = Query(None),
    ) -> Response:
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target = _get_target_or_404(store, target_id)
        try:
            path, index_lo, index_hi = store.write_target_mask3d_cache(target)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        headers = {"X-Target-Id": target_id}
        if index_lo is not None and index_hi is not None:
            headers["X-Mask3D-Index-Lo"] = str(index_lo)
            headers["X-Mask3D-Index-Hi"] = str(index_hi)
        return FileResponse(path, media_type="application/x-npy", headers=headers)

    @app.post("/sam3/train/jobs")
    def submit_train_job(payload: dict[str, Any]) -> dict[str, Any]:
        job = app.state.jobs.create("train", payload)
        app.state.queue.submit(_run_train_job, app, job.id)
        return {"job_id": job.id, "state": job.state.value}

    @app.get("/sam3/train/jobs/{job_id}")
    def train_job_status(job_id: str) -> dict[str, Any]:
        job = _get_job_or_404(app, job_id)
        payload = job.status_payload()
        if job.result is not None:
            payload["result"] = job.result
        return payload

    @app.get("/sam3/models")
    def sam3_models() -> dict[str, Any]:
        registry: ModelRegistry = app.state.models
        return registry.load()

    @app.post("/sam3/models/{model_id}/activate")
    def activate_sam3_model(model_id: str) -> dict[str, Any]:
        registry: ModelRegistry = app.state.models
        try:
            payload = registry.activate(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc
        checkpoint = _checkpoint_for_model(payload, model_id)
        if checkpoint:
            engine = app.state.sam3
            if hasattr(engine, "reload_checkpoint"):
                engine.reload_checkpoint(_resolve_under_project(cfg.project_root, checkpoint))
        return payload

    return app


def _volume_payload(volume_id: str, spec: dict[str, Any], data_root: Path) -> dict[str, Any]:
    path = data_root / str(spec.get("path", ""))
    payload: dict[str, Any] = {
        "id": volume_id,
        "label": spec.get("label", volume_id),
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "cmap": spec.get("cmap"),
        "clim": spec.get("clim"),
        "mask_volume": spec.get("mask_volume"),
    }
    if path.exists() and path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        payload.update({"shape": list(arr.shape), "dtype": str(arr.dtype)})
    return payload


def _slice_cache_path(
    cfg: ServerConfig,
    volume_id: str,
    axis: str,
    index: int,
    source_path: Path,
) -> Path:
    stat = source_path.stat()
    safe_volume = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in volume_id)
    key = f"{safe_volume}_{axis}_{index}_{stat.st_size}_{stat.st_mtime_ns}.npy"
    return cfg.runtime_root / "cache" / "slices" / key


def _enforce_slice_cache_budget(cfg: ServerConfig) -> dict[str, int]:
    budget_bytes = int(max(0.0, float(cfg.slice_cache_max_gb)) * 1024**3)
    return enforce_slice_cache_budget(cfg.runtime_root / "cache" / "slices", budget_bytes)


def _validate_sam3_request(cfg: ServerConfig, payload: dict[str, Any], *, kind: str) -> None:
    sam3_cfg = dict(cfg.sam3)
    try:
        validate_sam3_payload(
            payload,
            kind=kind,
            max_boxes=int(sam3_cfg.get("max_boxes", 50)),
            max_points=int(sam3_cfg.get("max_points", 200)),
            max_keep_top_k=int(sam3_cfg.get("max_keep_top_k", 50)),
            max_track_frames=int(sam3_cfg.get("max_track_frames", 5000)),
            max_batch_frames=int(sam3_cfg.get("max_batch_frames", 5000)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _make_sam3_engine(cfg: ServerConfig) -> SAM3Engine:
    sam3_cfg = dict(cfg.sam3)
    checkpoint = _resolve_under_project(
        cfg.project_root,
        sam3_cfg.get("checkpoint", "weights/sam3.pt"),
    )
    source_root = _resolve_under_project(
        cfg.project_root,
        sam3_cfg.get("source_root", "libs"),
    )
    return SAM3Engine(
        checkpoint,
        device=str(sam3_cfg.get("device", "cuda")),
        resolution=int(sam3_cfg.get("resolution", 1008)),
        source_root=source_root,
        load_video=bool(sam3_cfg.get("load_video", True)),
    )


def _make_model_registry(cfg: ServerConfig) -> ModelRegistry:
    training_cfg = dict(cfg.training)
    subdir = str(training_cfg.get("models_subdir", "sam3/models"))
    return ModelRegistry(cfg.results_root / subdir)


def _checkpoint_for_model(registry_payload: dict[str, Any], model_id: str) -> str | None:
    for model in registry_payload.get("models", []):
        if isinstance(model, dict) and model.get("id") == model_id and model.get("checkpoint"):
            return str(model["checkpoint"])
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_id(cfg: ServerConfig, payload: dict[str, Any] | None = None, project: str | None = None) -> str:
    if project:
        return str(project)
    if payload:
        value = payload.get("project") or payload.get("project_id")
        if value:
            return str(value)
    return cfg.project_id or "default"


def _target_store(
    cfg: ServerConfig,
    *,
    project: str | None = None,
    volume_id: str | None = None,
) -> TargetStore:
    subdir = str(cfg.sam3.get("results_subdir", "sam3"))
    return TargetStore(cfg.results_root / subdir, project=_project_id(cfg, project=project), volume_id=volume_id)


def _payload_target_type(payload: dict[str, Any]) -> str:
    prompts = payload.get("prompts") if isinstance(payload.get("prompts"), dict) else {}
    value = payload.get("target_type") or payload.get("type") or prompts.get("target_type") or prompts.get("text")
    return normalise_target_type(str(value or "unknown"))


def _volume_axis(axis: str) -> Literal["inline", "xline", "z"]:
    mapping = {
        "inline": "inline",
        "xline": "xline",
        "crossline": "xline",
        "z": "z",
        "timeslice": "z",
    }
    try:
        return mapping[str(axis)]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"Unsupported axis: {axis}") from exc


def _target_axis(axis: str) -> Literal["inline", "crossline", "timeslice"]:
    mapping = {
        "inline": "inline",
        "i": "inline",
        "xline": "crossline",
        "crossline": "crossline",
        "j": "crossline",
        "z": "timeslice",
        "timeslice": "timeslice",
        "k": "timeslice",
    }
    try:
        return mapping[str(axis)]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"Unsupported target axis: {axis}") from exc


def _target_frame_key(axis: str, index: int) -> str:
    return frame_key(_target_axis(axis), int(index))


def _get_target_or_404(
    store: TargetStore,
    target_id: str,
    *,
    target_set: TargetSet | None = None,
) -> GeoTarget:
    current_set = target_set or store.load()
    target = current_set.targets.get(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Unknown target: {target_id}")
    return target


def _existing_target_rows(
    store: TargetStore,
    target_set: TargetSet,
    volume_id: str | None,
    target_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_id in sorted(target_ids):
        target = target_set.targets.get(target_id)
        if target is None:
            continue
        if target.status in {TargetStatus.DELETED, TargetStatus.MERGED}:
            continue
        if volume_id and target.volume_id and target.volume_id != volume_id:
            continue
        frames: dict[int, np.ndarray] = {}
        for frame in target.frames.values():
            if not frame.mask_ref:
                continue
            path = store.resolve_ref(frame.mask_ref)
            if not path.exists():
                continue
            frames[int(frame.index)] = np.asarray(store.read_mask(frame.mask_ref), dtype=bool)
        if frames:
            rows.append({"target_id": target.id, "frames": frames})
    return rows


def _target_suggestions(
    suggestions: list[dict[str, Any]],
    obj_to_target_id: dict[str, str] | dict[int, str],
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for suggestion in suggestions:
        kind = str(suggestion.get("type", ""))
        if kind == "merge":
            obj_ids = [int(obj) for obj in suggestion.get("obj_ids", [])]
            target_ids = [
                str(obj_to_target_id.get(str(obj)) or obj_to_target_id.get(obj) or "")
                for obj in obj_ids
            ]
            target_ids = [target_id for target_id in target_ids if target_id]
            if len(set(target_ids)) < 2:
                continue
            row = dict(suggestion)
            row["target_ids"] = target_ids
            mapped.append(row)
        elif kind == "split":
            obj_id = int(suggestion.get("obj_id", 0))
            target_id = str(obj_to_target_id.get(str(obj_id)) or obj_to_target_id.get(obj_id) or "")
            if not target_id:
                continue
            row = dict(suggestion)
            row["target_id"] = target_id
            mapped.append(row)
    return mapped


def _gpu_payload(cfg: ServerConfig) -> dict[str, Any]:
    gpu_ids = list(cfg.sam3.get("gpu_ids", [0, 1, 2, 3]))
    payload: dict[str, Any] = {
        "worker_count": int(cfg.sam3.get("worker_count", len(gpu_ids) or 1)),
        "gpu_ids": gpu_ids,
        "workers": [{"worker_id": idx, "cuda_visible_devices": str(gpu)} for idx, gpu in enumerate(gpu_ids)],
        "torch_cuda_available": None,
        "devices": [],
    }
    try:
        import torch  # type: ignore

        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["devices"] = [
                {
                    "id": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "active_worker": idx in gpu_ids,
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001 - optional diagnostics endpoint
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _resolve_under_project(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def _get_job_or_404(app: FastAPI, job_id: str):
    job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown SAM3 job: {job_id}")
    return job


def _run_sam3_job(app: FastAPI, job_id: str) -> None:
    jobs: JobStore = app.state.jobs
    job = jobs.get(job_id)
    if job is None or job.state == JobState.cancelled:
        return
    if job.kind != "segment":
        jobs.update(
            job_id,
            state=JobState.error,
            progress=1.0,
            message="failed",
            error=f"_run_sam3_job only handles segment jobs, got kind={job.kind!r}"
            " (track jobs are dispatched to _run_track_job)",
        )
        return

    cfg: ServerConfig = app.state.config
    payload = dict(job.params)
    try:
        jobs.update(job_id, state=JobState.running, progress=0.05, message="loading slice")
        volume_id = str(payload.get("volume_id", ""))
        axis = str(payload.get("axis", ""))
        index = int(payload.get("index", payload.get("slice_index", 0)))
        if axis not in {"inline", "xline", "crossline", "z", "timeslice"}:
            raise ValueError(f"Unsupported axis: {axis}")
        volume_axis = _volume_axis(axis)
        target_axis = _target_axis(axis)

        spec, path = _volume_spec_and_path(cfg, volume_id)
        data, shape = _load_slice(path, volume_axis, index)
        # Keep the same orientation as the desktop SAM3 path: image rows are samples/depth.
        slice2d = np.asarray(data, dtype=np.float32).T
        clim = _parse_clim(spec.get("clim"))
        rgb = slice_to_rgb_image(slice2d, clim=clim)

        prompts = payload.get("prompts") or {}
        if not isinstance(prompts, dict):
            prompts = {}
        jobs.update(job_id, progress=0.25, message="running SAM3")
        detections = app.state.sam3.segment(
            rgb,
            text=str(prompts.get("text", payload.get("text", ""))),
            boxes=_list_of_float_lists(prompts.get("boxes", [])),
            points=_list_of_float_lists(prompts.get("points", [])),
            point_box_radius_px=float(payload.get("point_box_radius_px", 8.0)),
            confidence=float(payload.get("confidence", payload.get("confidence_threshold", 0.4))),
        )
        detections.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        keep_top_k = int(payload.get("keep_top_k", 3))
        detections = detections[: max(1, keep_top_k)]

        jobs.update(job_id, progress=0.85, message="writing masks")
        output_dir = _sam3_job_output_dir(cfg, job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        project = _project_id(cfg, payload)
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target_type = _payload_target_type(payload)
        candidates: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        mask_paths: list[str] = []
        # SAM3 inference is already done above; only the read-modify-write of
        # targets.json runs under the per-project lock so concurrent jobs can't
        # clobber each other or reuse ids. See docs/project_review_and_remediation §1.1.
        with store.mutate() as target_set:
            if not target_set.volume_id:
                target_set.volume_id = volume_id
            for candidate_index, det in enumerate(detections):
                mask = np.asarray(det.get("mask"), dtype=bool)
                score = float(det.get("score", 0.0))
                target = store.add_single_frame_target(
                    target_set,
                    axis=target_axis,
                    index=index,
                    mask=mask,
                    target_type=target_type,
                    score=score,
                    source="sam3_interactive",
                    volume_id=volume_id,
                    image_ref=f"{volume_id}:{axis}:{index}",
                )
                if payload.get("target_status") is not None:
                    target.status = TargetStatus(str(payload["target_status"]))
                frame = next(iter(target.frames.values()))
                mask_path = store.resolve_ref(frame.mask_ref) if frame.mask_ref else output_dir / f"cand{candidate_index}.npy"
                mask_paths.append(str(mask_path))
                targets.append(target.model_dump(mode="json"))
                candidates.append(
                    {
                        "index": candidate_index,
                        "target_id": target.id,
                        "target_type": target.type,
                        "score": score,
                        "box": [float(v) for v in det.get("box", [])],
                        "mask_path": _relative_result_path(cfg, mask_path),
                        "mask_url": f"/sam3/jobs/{job_id}/mask/{candidate_index}",
                        "shape": [int(v) for v in mask.shape],
                        "dtype": "uint8",
                    }
                )

        result = {
            "job_id": job_id,
            "kind": "segment",
            "project": project,
            "volume_id": volume_id,
            "axis": axis,
            "index": index,
            "volume_shape": [int(v) for v in shape],
            "candidates": candidates,
            "targets": targets,
            "target_set_url": f"/sam3/targets?project={project}&volume_id={volume_id}",
        }
        jobs.update(
            job_id,
            state=JobState.done,
            progress=1.0,
            message="done",
            result=result,
            mask_paths=mask_paths,
        )
    except Exception as exc:  # noqa: BLE001 - job boundary returns status instead of crashing uvicorn
        jobs.update(
            job_id,
            state=JobState.error,
            progress=1.0,
            message="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _parse_track_range(payload: dict[str, Any]) -> tuple[int, int, int]:
    """Resolve (seed, back, fwd) from a track payload.

    Accepts either ``index={"seed","back","fwd"}``, flat ``seed/back/fwd``,
    or a ``start_index/end_index`` range (seed defaults to the midpoint).
    """
    idx = payload.get("index")
    if isinstance(idx, dict):
        seed = int(idx.get("seed", idx.get("index", 0)))
        back = int(idx.get("back", 0))
        fwd = int(idx.get("fwd", 0))
    else:
        seed = int(payload.get("seed", payload.get("index", 0)))
        back = int(payload.get("back", payload.get("n_back", 0)))
        fwd = int(payload.get("fwd", payload.get("n_fwd", 0)))
    start = payload.get("start_index")
    end = payload.get("end_index")
    if back == 0 and fwd == 0 and start is not None and end is not None:
        lo, hi = sorted((int(start), int(end)))
        if not lo <= seed <= hi:
            seed = (lo + hi) // 2
        back = seed - lo
        fwd = hi - seed
    return seed, max(0, back), max(0, fwd)


def _box_to_norm_xywh(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    x0 = max(0.0, min(x0, width - 1.0))
    x1 = max(0.0, min(x1, width - 1.0))
    y0 = max(0.0, min(y0, height - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    cx = (lo_x + hi_x) / 2.0 / float(width)
    cy = (lo_y + hi_y) / 2.0 / float(height)
    bw = max(1.0, hi_x - lo_x) / float(width)
    bh = max(1.0, hi_y - lo_y) / float(height)
    return [cx, cy, bw, bh]


def _job_cancelled(jobs: JobStore, job_id: str) -> bool:
    job = jobs.get(job_id)
    return job is None or job.state == JobState.cancelled


def _run_track_job(app: FastAPI, job_id: str) -> None:
    """Multi-object cross-frame tracking on seismic axial slices.

    Renders the [seed-back, seed+fwd] window to JPEG (server-side, PIL only —
    no Qt/matplotlib), seeds one obj_id per geological target, propagates
    forward+backward, and writes each object's frames into ONE GeoTarget so the
    numbering stays consistent across the whole sweep. See
    docs/project_review_and_remediation.md §2.1.
    """
    jobs: JobStore = app.state.jobs
    job = jobs.get(job_id)
    if job is None or job.state == JobState.cancelled:
        return
    cfg: ServerConfig = app.state.config
    payload = dict(job.params)
    tempdir: Path | None = None
    try:
        import tempfile
        from PIL import Image

        jobs.update(job_id, state=JobState.running, progress=0.02, message="planning track")
        volume_id = str(payload.get("volume_id", ""))
        axis = str(payload.get("axis", ""))
        if axis not in {"inline", "xline", "crossline", "z", "timeslice"}:
            raise ValueError(f"Unsupported axis: {axis}")
        volume_axis = _volume_axis(axis)
        target_axis = _target_axis(axis)
        axis_index = {"inline": 0, "xline": 1, "z": 2}[volume_axis]

        seed, back, fwd = _parse_track_range(payload)
        prompts = payload.get("prompts") if isinstance(payload.get("prompts"), dict) else {}
        boxes = _list_of_float_lists(prompts.get("boxes", []))
        text = str(prompts.get("text", payload.get("text", "")))
        target_type = _payload_target_type(payload)
        project = _project_id(cfg, payload)
        confidence = float(payload.get("confidence", payload.get("confidence_threshold", 0.4)))
        keep_top_k = int(payload.get("keep_top_k", 3))

        spec, path = _volume_spec_and_path(cfg, volume_id)
        clim = _parse_clim(spec.get("clim"))

        # Window resolved against the real axis length.
        seed_data, shape = _load_slice(path, volume_axis, seed)
        axis_len = int(shape[axis_index])
        seed = max(0, min(seed, axis_len - 1))
        idx_lo = max(0, seed - back)
        idx_hi = min(axis_len - 1, seed + fwd)
        indices = list(range(idx_lo, idx_hi + 1))
        seed_local = seed - idx_lo
        fwd_budget = idx_hi - seed
        back_budget = seed - idx_lo

        engine = app.state.sam3
        seed_rgb = slice_to_rgb_image(np.asarray(seed_data, dtype=np.float32).T, clim=clim)
        seed_h, seed_w = int(seed_rgb.shape[0]), int(seed_rgb.shape[1])

        # Seed boxes: explicit prompt boxes, or derive them by text segmentation
        # of the seed frame (powers "extract all <type>" + track, direction 5/6).
        if not boxes:
            if not text:
                raise ValueError("track job requires prompts.boxes or prompts.text")
            detections = engine.segment(seed_rgb, text=text, confidence=confidence)
            detections.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            boxes = [list(det["box"]) for det in detections[: max(1, keep_top_k)] if det.get("box")]
            if not boxes:
                raise ValueError("text seeding produced no detections to track")

        seeds = [
            {"obj_id": k, "box_xywh": _box_to_norm_xywh(box, seed_w, seed_h), "text": text}
            for k, box in enumerate(boxes, start=1)
        ]

        # Render frames to JPEG (image order: rows=samples, cols=trace).
        tempdir = Path(tempfile.mkdtemp(prefix=f"yj_track_{job_id}_"))
        for offset, idx in enumerate(indices):
            if _job_cancelled(jobs, job_id):
                return
            data, _shape = _load_slice(path, volume_axis, idx)
            rgb = slice_to_rgb_image(np.asarray(data, dtype=np.float32).T, clim=clim)
            Image.fromarray(rgb).save(tempdir / f"{offset:05d}.jpg", quality=92)
            jobs.update(
                job_id,
                progress=0.05 + 0.25 * (offset + 1) / len(indices),
                message=f"rendered {offset + 1}/{len(indices)} frames",
            )

        jobs.update(job_id, progress=0.32, message="tracking")
        n_obj_frames = max(1, len(indices) * len(seeds))

        def _on_progress(done: int) -> None:
            jobs.update(
                job_id,
                progress=min(0.92, 0.32 + 0.6 * done / n_obj_frames),
                message=f"tracked {done} object-frames",
            )

        collected = collect_object_frames(
            engine,
            tempdir,
            seeds=seeds,
            seed_local=seed_local,
            fwd_budget=fwd_budget,
            back_budget=back_budget,
            indices=indices,
            cancelled=lambda: _job_cancelled(jobs, job_id),
            progress=_on_progress,
        )
        if _job_cancelled(jobs, job_id):
            return

        # Atomic write under the per-project lock: allocate ids + persist frames.
        jobs.update(job_id, progress=0.94, message="writing targets")
        store = _target_store(cfg, project=project, volume_id=volume_id)
        gap_metadata = annotate_gaps(
            collected,
            indices,
            gap_limit=int(payload.get("gap_limit", 5)),
        )
        object_suggestions = detect_merge_split(
            collected,
            indices,
            iou_merge=float(payload.get("merge_iou", 0.5)),
            persist_frames=int(payload.get("suggestion_frames", 3)),
        )

        def _resolve_link(
            obj_id: int,
            frames: dict[int, np.ndarray],
            target_set: TargetSet,
            linkable_target_ids: set[str],
        ) -> str | None:
            existing = _existing_target_rows(store, target_set, volume_id, linkable_target_ids)
            return link_targets_by_iou(
                existing,
                frames,
                iou_thresh=float(payload.get("link_iou", 0.3)),
                min_overlap_frames=int(payload.get("link_overlap_frames", 1)),
            )

        summary = persist_tracked_targets(
            store,
            collected,
            seeds=seeds,
            target_axis=target_axis,
            target_type=target_type,
            volume_id=volume_id,
            image_axis_label=axis,
            gap_metadata=gap_metadata,
            link_resolver=_resolve_link,
        )
        suggestions = _target_suggestions(object_suggestions, summary.get("obj_to_target_id", {}))

        result = {
            "job_id": job_id,
            "kind": "track",
            "project": project,
            "volume_id": volume_id,
            "axis": axis,
            "seed": seed,
            "frame_range": [idx_lo, idx_hi],
            "target_set_url": f"/sam3/targets?project={project}&volume_id={volume_id}",
            "gaps": {str(key): value for key, value in gap_metadata.items()},
            "suggestions": suggestions,
            **summary,
        }
        jobs.update(job_id, state=JobState.done, progress=1.0, message="done", result=result)
    except Exception as exc:  # noqa: BLE001 - job boundary returns status instead of crashing uvicorn
        jobs.update(
            job_id,
            state=JobState.error,
            progress=1.0,
            message="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if tempdir is not None:
            import shutil

            shutil.rmtree(tempdir, ignore_errors=True)


def _run_sam3_batch_job(app: FastAPI, job_id: str) -> None:
    jobs: JobStore = app.state.jobs
    job = jobs.get(job_id)
    if job is None or job.state == JobState.cancelled:
        return
    cfg: ServerConfig = app.state.config
    payload = dict(job.params)
    try:
        frames = _batch_frame_payloads(payload)
        if not frames:
            raise ValueError("Batch job requires frames or range")
        jobs.update(job_id, state=JobState.running, progress=0.02, message="batch queued")
        child_job_ids: list[str] = []
        target_ids: list[str] = []
        errors: list[dict[str, Any]] = []
        for pos, frame_payload in enumerate(frames):
            if jobs.get(job_id) and jobs.get(job_id).state == JobState.cancelled:
                return
            request = dict(payload)
            request.update(frame_payload)
            request["kind"] = "segment"
            child = jobs.create("segment", request)
            child_job_ids.append(child.id)
            _run_sam3_job(app, child.id)
            child_done = jobs.get(child.id)
            if child_done and child_done.result:
                target_ids.extend(
                    str(target.get("id"))
                    for target in child_done.result.get("targets", [])
                    if target.get("id")
                )
            elif child_done and child_done.error:
                errors.append({"job_id": child.id, "error": child_done.error})
            jobs.update(
                job_id,
                progress=0.02 + 0.96 * float(pos + 1) / float(len(frames)),
                message=f"processed {pos + 1}/{len(frames)} frames",
            )
        result = {
            "job_id": job_id,
            "kind": str(payload.get("result_kind") or job.kind or "batch"),
            "project": _project_id(cfg, payload),
            "child_job_ids": child_job_ids,
            "target_ids": target_ids,
            "errors": errors,
        }
        jobs.update(
            job_id,
            state=JobState.done if not errors else JobState.error,
            progress=1.0,
            message="done" if not errors else "completed with errors",
            result=result,
            error=None if not errors else f"{len(errors)} child jobs failed",
        )
    except Exception as exc:  # noqa: BLE001 - job boundary
        jobs.update(
            job_id,
            state=JobState.error,
            progress=1.0,
            message="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _batch_frame_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if isinstance(frames, list):
        result: list[dict[str, Any]] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            result.append(
                {
                    "volume_id": frame.get("volume_id", payload.get("volume_id")),
                    "axis": frame.get("axis", payload.get("axis", "inline")),
                    "index": int(frame.get("index", frame.get("slice_index", 0))),
                }
            )
        return result
    indices = payload.get("indices")
    if isinstance(indices, list):
        return [
            {
                "volume_id": payload.get("volume_id"),
                "axis": payload.get("axis", "inline"),
                "index": int(index),
            }
            for index in indices
        ]
    start = payload.get("start_index", payload.get("start"))
    end = payload.get("end_index", payload.get("end"))
    if start is None or end is None:
        return []
    step = int(payload.get("step", 1))
    if step == 0:
        raise ValueError("step cannot be zero")
    start_i = int(start)
    end_i = int(end)
    inclusive_end = end_i + (1 if step > 0 else -1)
    return [
        {
            "volume_id": payload.get("volume_id"),
            "axis": payload.get("axis", "inline"),
            "index": index,
        }
        for index in range(start_i, inclusive_end, step)
    ]


def _run_train_job(app: FastAPI, job_id: str) -> None:
    jobs: JobStore = app.state.jobs
    job = jobs.get(job_id)
    if job is None or job.state == JobState.cancelled:
        return
    cfg: ServerConfig = app.state.config
    payload = dict(job.params)
    try:
        jobs.update(job_id, state=JobState.running, progress=0.1, message="exporting dataset")
        project = _project_id(cfg, payload)
        volume_id = str(payload.get("volume_id") or "") or None
        store = _target_store(cfg, project=project, volume_id=volume_id)
        target_set = store.load()
        dataset_version = str(payload.get("dataset_version") or f"D{uuid.uuid4().hex[:8]}")
        dataset_subdir = str(cfg.training.get("dataset_subdir", "sam3/datasets"))
        output_dir = cfg.results_root / dataset_subdir / project / dataset_version
        export_payload = export_confirmed_to_coco(store, target_set, output_dir)
        metrics: dict[str, Any] = {
            "exported_images": len(export_payload.get("images", [])),
            "exported_annotations": len(export_payload.get("annotations", [])),
            "training_status": "not_run",
        }
        checkpoint = str(payload.get("checkpoint")) if payload.get("checkpoint") else None
        training_result: dict[str, Any] | None = None
        training_command = payload.get("training_command") or payload.get("command") or cfg.training.get("command")
        message = "Dataset exported; no training.command configured."
        status = "ready" if checkpoint else "dataset_exported"
        if training_command:
            jobs.update(job_id, progress=0.78, message="running training backend")
            train_subdir = str(cfg.training.get("output_subdir", "sam3/training_runs"))
            train_output_dir = cfg.results_root / train_subdir / project / dataset_version
            timeout_raw = payload.get("timeout_s", cfg.training.get("timeout_s"))
            timeout_s = float(timeout_raw) if timeout_raw is not None else None
            env_payload = payload.get("env") if isinstance(payload.get("env"), dict) else {}
            env_config = cfg.training.get("env") if isinstance(cfg.training.get("env"), dict) else {}
            training_result = run_training_backend(
                training_command,
                dataset_dir=output_dir,
                output_dir=train_output_dir,
                timeout_s=timeout_s,
                extra_env={**env_config, **env_payload},
            )
            checkpoint = training_result.get("checkpoint") or checkpoint
            metrics.update(dict(training_result.get("metrics") or {}))
            metrics.update(
                {
                    "training_status": "completed",
                    "training_output_dir": training_result.get("output_dir"),
                    "training_command": training_result.get("command"),
                }
            )
            status = "ready" if checkpoint else "trained_no_checkpoint"
            message = "Training backend completed." if checkpoint else "Training completed without a checkpoint."
        jobs.update(job_id, progress=0.95, message="recording model version")
        registry: ModelRegistry = app.state.models
        model = registry.add_model(
            checkpoint=checkpoint,
            dataset_version=dataset_version,
            metrics=metrics,
            status=status,
        )
        result = {
            "job_id": job_id,
            "project": project,
            "dataset_version": dataset_version,
            "dataset_path": str(output_dir),
            "training_result": training_result,
            "model": model,
            "message": message,
        }
        jobs.update(job_id, state=JobState.done, progress=1.0, message="done", result=result)
    except Exception as exc:  # noqa: BLE001 - job boundary
        jobs.update(
            job_id,
            state=JobState.error,
            progress=1.0,
            message="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _volume_spec_and_path(cfg: ServerConfig, volume_id: str) -> tuple[dict[str, Any], Path]:
    spec = cfg.volumes.get(volume_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown volume: {volume_id}")
    path = cfg.data_root / str(spec.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Volume file not found: {path}")
    return spec, path


def _load_slice(
    path: Path,
    axis: Literal["inline", "xline", "z"] | str,
    index: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to open volume: {exc}") from exc
    if arr.ndim != 3:
        raise HTTPException(status_code=400, detail=f"Volume must be 3D, got shape {arr.shape}")

    shape = tuple(int(v) for v in arr.shape)
    axis_index = {"inline": 0, "xline": 1, "z": 2}[str(axis)]
    if not 0 <= int(index) < shape[axis_index]:
        raise HTTPException(
            status_code=416,
            detail=f"{axis} index {index} outside shape {shape}",
        )
    if axis == "inline":
        data = np.asarray(arr[int(index), :, :])
    elif axis == "xline":
        data = np.asarray(arr[:, int(index), :])
    else:
        data = np.asarray(arr[:, :, int(index)])
    return data, shape


def _parse_clim(value: object) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first, second = value[0], value[1]
        if first is None or second is None:
            return None
        return (float(first), float(second))
    return None


def _list_of_float_lists(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    result: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)):
            continue
        try:
            result.append([float(v) for v in item])
        except (TypeError, ValueError):
            continue
    return result


def _sam3_job_output_dir(cfg: ServerConfig, job_id: str) -> Path:
    subdir = str(cfg.sam3.get("results_subdir", "sam3"))
    return cfg.results_root / subdir / "jobs" / job_id


def _relative_result_path(cfg: ServerConfig, path: Path) -> str:
    try:
        return str(path.relative_to(cfg.data_root)).replace("\\", "/")
    except ValueError:
        return str(path)


app = create_app()
