# 下一步详细实现指南（可直接照此写代码）

> **【已归档·历史参考】** 本文件覆盖 SAM3/目标侧的步骤 4–9，大半已实现。它**不再是**「接下来要做什么」的入口——当前要做的事看 [`implementation_runbook_and_feature_backlog.md`](implementation_runbook_and_feature_backlog.md)，仍需实现的完整目录看 [`next_steps_geophysics_and_training.md`](next_steps_geophysics_and_training.md)，现状看 [`project_review_and_remediation.md`](project_review_and_remediation.md)。保留本文仅供回顾 SAM3/目标侧的细节路线。

本文件承接 [`project_review_and_remediation.md`](project_review_and_remediation.md) 的「已实现快照」，把**剩余路线**展开到文件、函数签名、代码骨架、API 契约、数据流、测试与完成定义。已完成的 §1.1 / §1.2 / §2.1 / §2.2 不再赘述。

> 约定：服务器启动/重启/训练/验证由用户手动执行；本机 py312 环境**无 fastapi**，凡 FastAPI-free 的核心（如 `sam3/tracking.py`、`sam3/reassociate.py`）都要保证能脱离 FastAPI 单测。
>
> 现成接缝（全部已存在，直接挂）：
> - 追踪核心：[`sam3/tracking.py`](../server/src/yj_studio_server/sam3/tracking.py) `collect_object_frames` / `persist_tracked_targets`
> - 并发安全写：[`targets/store.py`](../apps/yj_studio/src/yj_studio/targets/store.py) `TargetStore.mutate()`
> - 远程 SAM3 任务客户端：[`ai/remote_client.py`](../apps/yj_studio/src/yj_studio/ai/remote_client.py) `RemoteSAM3Client`（`submit_segment/poll/result/cancel`）
> - 远程目标库客户端：[`data/remote_target_store.py`](../apps/yj_studio/src/yj_studio/data/remote_target_store.py) `RemoteTargetStore`
> - 目标管理面板：[`ui/docks/target_dock.py`](../apps/yj_studio/src/yj_studio/ui/docks/target_dock.py) `TargetDock.refresh()`

---

## 当前推进状态（2026-06-12）

步骤 5 的 AI 面板端到端入口已完成：

- `RemoteSAM3Client.submit_track()` 已提交 `kind=track` job；
- `RemoteSAM3TrackTask` 已实现提交/轮询/取消；
- `AIDock` 已新增目标类型、种子前/后帧数、「追踪」按钮；
- `MainWindow` 已在追踪完成后自动刷新 `TargetDock`；
- 新增测试 `apps/yj_studio/tests/test_remote_sam3_track.py`，本机通过。

新增完成：

- 步骤 4 重关联/合并分裂建议已完成：`sam3/reassociate.py`、`persist_tracked_targets(link_resolver=...)`、`_run_track_job` suggestions/gaps、`TargetDock.show_track_result()`。
- 步骤 7.2 3D mask 按真实 index 重建已完成：`write_target_mask3d_cache()` + `/mask3d` index header。
- 步骤 8.1 mask 回写接口已完成：服务器 PUT `.npy` mask + 客户端 `RemoteTargetStore.put_mask()` + `GeoTarget.edits`。
- 步骤 8.2 的导出格式部分已完成：COCO/PNG 导出包含 `schema_version` 与 train/val/test split，edited 目标可进入导出。
- 步骤 9 基础健壮性已完成：job 终态持久化、切片缓存 LRU、`TargetSet.schema_version` 与未知字段兼容、SAM3 入队前输入校验、`metadata_is_lightweight` 修正。
- 储层路径 B 的基础接口已完成：`POST /sam3/targets/cells` 接收 `.npy` cell IJK，`RemoteTargetStore.create_cell_target()` 上传二进制 cell，`SAM3Workbench(target_store=...)` 可在生成 `ReservoirSelectionLayer` 时同步写 `GeoTarget`。
- 步骤 7.1 的基础 2D 叠加已完成：目标类型配色、`MaskLayer` 空间摘要、2D 剖面 mask 轮廓与 `Tn` 编号。
- 步骤 8.2 的训练后端基础已完成：`training.command` 可配置外部训练脚本，服务端采集 `metrics.json`/checkpoint 并登记模型；activate 会 reload checkpoint。

仍未完成：储层工作台路径 B 的主窗口入口接线/恢复（当前没有实际构造 `SAM3Workbench` 的入口）、步骤 6 真多卡、步骤 7.1 的轨迹/面积曲线子面板、真实 SAM3 微调脚本/评估/回流；步骤 9 后续只剩随真多卡/训练补更完整的集成测试。

---

## 步骤 4 ·（§2.3）丢失/重关联 +（§2.4）合并/分裂建议

### 4.0 先纠一个认知（很重要）

SAM3 video 是**种子式**：我们在种子帧一次 `add_prompt` 固定 obj_id=1..N，传播时这些 obj_id **保持不变**，被遮挡的对象只是某些帧 mask 为空，重现时**仍是同一 obj_id**。所以「同一 track 内重新拿到新 ID」基本不会发生——`collect_object_frames` 已经天然保持单 track 内 ID 一致。

因此 §2.3 真正要做的是两件**不同**的事：
- **(a) track 内的 gap/lost 标注**：记录某对象在哪些帧缺失、是否长期消失。轻量。
- **(b) 跨 track / 跨子区间的目标关联**：体级文本提取会拆成多段 track（或 batch 多帧），同一物理圈闭可能在两段里各生成一个 `GeoTarget`，需要在**重叠帧用 IoU 判定为同一目标并合并**。这才是「重关联」的实质价值点。

### 4.1 新增 `server/src/yj_studio_server/sam3/reassociate.py`（FastAPI-free）

```python
from __future__ import annotations
import numpy as np

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(union) if union else 0.0

def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, bool))
    return (float(xs.mean()), float(ys.mean())) if xs.size else None

def annotate_gaps(
    collected: dict[int, dict[int, np.ndarray]],
    indices: list[int],
    *,
    gap_limit: int = 5,
) -> dict[int, dict]:
    """Per object: which absolute indices are missing, and whether the trailing
    gap exceeds gap_limit (→ status hint 'lost'). Pure metadata, no array work."""
    out: dict[int, dict] = {}
    span = set(indices)
    for obj_id, frames in collected.items():
        present = set(frames)
        missing = sorted(span - present)
        # trailing gap = consecutive missing at the high-index end
        trailing = 0
        for idx in reversed(indices):
            if idx in present:
                break
            trailing += 1
        out[obj_id] = {"missing": missing, "status_hint": "lost" if trailing > gap_limit else "active"}
    return out

def link_targets_by_iou(
    existing: list[dict],      # [{"target_id":..., "frames":{abs_index: mask}}]
    candidate_frames: dict[int, np.ndarray],   # one new object's abs_index -> mask
    *,
    iou_thresh: float = 0.3,
    min_overlap_frames: int = 1,
) -> str | None:
    """Return the target_id of an existing target that overlaps this candidate
    on shared frames (mean IoU over shared frames > thresh), else None.
    Used to fold a new track's object into an existing target across jobs."""
    best_id, best_iou = None, iou_thresh
    for ex in existing:
        shared = set(ex["frames"]) & set(candidate_frames)
        if len(shared) < min_overlap_frames:
            continue
        mean_iou = np.mean([mask_iou(ex["frames"][i], candidate_frames[i]) for i in shared])
        if mean_iou > best_iou:
            best_id, best_iou = ex["target_id"], float(mean_iou)
    return best_id
```

### 4.2 合并/分裂建议（同一文件）

```python
from scipy import ndimage   # 已在依赖里（reservoir 用过）；若无则用 cv2/自写两遍扫描

def detect_merge_split(
    collected: dict[int, dict[int, np.ndarray]],
    indices: list[int],
    *,
    iou_merge: float = 0.5,
    persist_frames: int = 3,
) -> list[dict]:
    """返回 suggestions: 
       {"type":"merge","obj_ids":[a,b],"frames":[...]}  两对象持续高 IoU
       {"type":"split","obj_id":a,"frames":[...]}        单对象持续多连通域
    只检测、不改动；交给 UI 让用户确认。"""
    suggestions: list[dict] = []
    obj_ids = sorted(collected)
    # merge: pairwise IoU on shared frames
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            a, b = obj_ids[i], obj_ids[j]
            hot = [idx for idx in indices
                   if idx in collected[a] and idx in collected[b]
                   and mask_iou(collected[a][idx], collected[b][idx]) > iou_merge]
            if len(hot) >= persist_frames:
                suggestions.append({"type": "merge", "obj_ids": [a, b], "frames": hot})
    # split: connected-component count >= 2 persistently
    for a in obj_ids:
        multi = [idx for idx, m in collected[a].items()
                 if ndimage.label(np.asarray(m, bool))[1] >= 2]
        if len(multi) >= persist_frames:
            suggestions.append({"type": "split", "obj_id": a, "frames": sorted(multi)})
    return suggestions
```

### 4.3 接入 `_run_track_job`（[app.py](../server/src/yj_studio_server/app.py)）

`collect_object_frames` 返回 `collected` 后、`persist_tracked_targets` 之前插入：

```python
from .sam3.reassociate import annotate_gaps, detect_merge_split, link_targets_by_iou

gaps = annotate_gaps(collected, indices)
suggestions = detect_merge_split(collected, indices)
# 跨 job 关联：把本次每个对象先尝试并入已有目标
store = _target_store(cfg, project=project, volume_id=volume_id)
with store.mutate() as target_set:
    existing = [{"target_id": t.id,
                 "frames": {f.index: store.read_mask(f.mask_ref) > 0
                            for f in t.frames.values() if f.mask_ref}}
                for t in target_set.targets.values()
                if t.volume_id == volume_id and t.status not in (TargetStatus.DELETED, TargetStatus.MERGED)]
    # ... 对 collected 的每个 obj：link_targets_by_iou 命中则写进该 target_id，否则 new_id
    #     gaps[obj]/suggestions 写进 target.metadata
```

> 注意：跨 job 关联会让 `persist_tracked_targets` 的「无脑 new_id」逻辑不再够用——给它加一个可选参数 `link_resolver: Callable[[int, dict], str | None]`，命中已有 target_id 时复用、否则 new_id。保持 §2.2 的锁内分配语义不变。

`result["suggestions"] = suggestions`，随 job 返回；`TargetDock` 读后用黄条提示（4.4）。

### 4.4 UI：建议提示（[target_dock.py](../apps/yj_studio/src/yj_studio/ui/docks/target_dock.py)）
- `refresh()` 后若 job result 带 `suggestions`，在表格上方加一行可点的提示：「检测到 T2/T5 可能应合并 — [合并] [忽略]」。点击调用既有 `self._target_store.merge_targets([...])` / `split_target(...)`。

### 4.5 测试（`server/tests/test_reassociate.py`，FastAPI-free）
- `mask_iou`：相同 mask=1.0、不相交=0.0、半重叠≈0.33。
- `annotate_gaps`：缺帧列表正确；尾部连续缺 > gap_limit → `status_hint="lost"`。
- `link_targets_by_iou`：重叠帧高 IoU → 返回该 target_id；无重叠/低 IoU → None。
- `detect_merge_split`：两对象持续重叠 → merge 建议；单对象哑铃形（两连通域）持续 → split 建议。

**DoD**：体级提取产出的多段 track 能按 IoU 合并到同一 `GeoTarget`；遮挡帧标 lost；dock 显示并可一键确认合并/分裂。

---

## 步骤 5 ·（§3）端到端贯通：交互框选 → 服务器 track → 目标库（最高价值）

目标：用户在**地震体剖面**上框 1~N 个目标 → 点「追踪」→ 服务器多目标 track → 目标库出现 T1/T2…，本机 GPU=0。这是把 §2.1 的服务端能力真正交到用户手上。

### 5.1 客户端：`RemoteSAM3Client.submit_track`（[ai/remote_client.py](../apps/yj_studio/src/yj_studio/ai/remote_client.py)）

仿现有 `submit_segment`，新增：

```python
def submit_track(
    self, *, volume_id: str, axis: str, seed: int, back: int, fwd: int,
    boxes: list[tuple[float, float, float, float]] | None = None,
    text: str = "", confidence: float = 0.4, keep_top_k: int = 3,
    target_type: str = "unknown",
) -> str:
    body = {
        "kind": "track",
        "project": self.project_id,
        "volume_id": volume_id,
        "axis": axis,
        "index": {"seed": int(seed), "back": int(back), "fwd": int(fwd)},
        "target_type": target_type or "unknown",
        "prompts": {"text": text, "boxes": [list(b) for b in (boxes or [])]},
        "confidence": float(confidence),
        "keep_top_k": int(keep_top_k),
    }
    payload = self._post_json("/sam3/jobs", body)
    job_id = str(payload.get("job_id", ""))
    if not job_id:
        raise RuntimeError("server did not return job_id for track")
    return job_id
```

> 服务端 `_parse_track_range` 已支持 `index={"seed","back","fwd"}` 和 `prompts.boxes`，契约对齐，无需改服务端。

### 5.2 客户端：`RemoteSAM3TrackTask`（[algorithms/runner.py](../apps/yj_studio/src/yj_studio/algorithms/runner.py)）

仿 `RemoteSAM3Task` 的四信号 + QTimer 轮询，但**完成时不建 MaskLayer**，而是把 result（含 `target_ids`）交回去触发目标库刷新：

```python
class RemoteSAM3TrackTask(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(dict, str)     # result dict, summary  ← 注意签名不同
    errored  = pyqtSignal(str, str)
    cancelled = pyqtSignal()

    def __init__(self, client, track_params: dict, *, parent=None):
        super().__init__(parent)
        self._client = client; self._params = dict(track_params)
        self._job_id = None; self._finished = False
        self._timer = QTimer(self); self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll)

    def start(self):
        if not self._client.is_ready():
            self._fail("远程 SAM3 未就绪"); return
        try:
            self._client.mark_busy("远程 SAM3 追踪中")
            self._job_id = self._client.submit_track(**self._params)
        except Exception as e:
            self._client.mark_ready(); self._fail(f"{type(e).__name__}: {e}"); return
        self.progress.emit(0.02, "已提交追踪任务"); self._timer.start()

    def _poll(self):
        st = self._client.poll(self._job_id)
        self.progress.emit(float(st.get("progress",0) or 0), str(st.get("message","")))
        state = str(st.get("state",""))
        if state in ("queued","running"): return
        self._timer.stop(); self._client.mark_ready()
        if state == "done":
            result = self._client.result(self._job_id)
            self._finished = True
            n = len(result.get("target_ids", []))
            self.finished.emit(result, f"追踪完成：{n} 个目标")
        elif state == "cancelled":
            self._finished = True; self.cancelled.emit()
        else:
            self._fail(str(st.get("error") or "追踪失败"))
    # cancel()/_fail() 同 RemoteSAM3Task
```

### 5.3 UI 入口：AI 面板加「追踪」（[ui/docks/ai_dock.py](../apps/yj_studio/src/yj_studio/ui/docks/ai_dock.py)）

AI 面板已收集 `axis / slice_index / boxes / points / text / confidence / keep_top_k`。新增：
- 两个 `QSpinBox`：`向前帧数 fwd`、`向后帧数 back`（默认各 20）。
- 一个「追踪」按钮（旁边即「运行 SAM3 分割」）。

`_on_track_clicked`：
```python
def _on_track_clicked(self):
    vol = self._active_volume_layer()
    if vol is None: ...return
    if not isinstance(self._ai_service, RemoteSAM3Client):
        QMessageBox.information(self,"追踪","追踪需要远程模式（mode=remote）。"); return
    if not self._boxes:
        QMessageBox.information(self,"追踪","请先在剖面上框选至少一个目标。"); return
    params = dict(volume_id=vol.volume_id, axis=axis, seed=slice_index,
                  back=self._back_spin.value(), fwd=self._fwd_spin.value(),
                  boxes=list(self._boxes), text=self._text_edit.toPlainText().strip(),
                  confidence=..., keep_top_k=..., target_type=...)
    task = RemoteSAM3TrackTask(self._ai_service, params, parent=self)
    task.progress.connect(self._on_progress)
    task.finished.connect(self._on_track_finished)   # 见 5.4
    task.errored.connect(self._on_errored)
    task.cancelled.connect(self._on_cancelled)
    self._current_task = task; task.start()
```

新增信号 `track_finished = pyqtSignal(dict)`，在 `_on_track_finished(result, summary)` 里 `self.track_finished.emit(result)` + 显示 summary。

> 框坐标系一致性：AI 面板的框已经是「剖面 RGB 像素坐标」（与 `sam3_segment` 同源），服务端 `_run_track_job` 的 `_box_to_norm_xywh` 也按种子帧 RGB 的 W/H 归一化——两端同序，无需额外转换。

### 5.4 串起目标库刷新（[ui/main_window.py](../apps/yj_studio/src/yj_studio/ui/main_window.py)）

```python
self.ai_dock.track_finished.connect(lambda result: self._target_dock.refresh())
```
追踪完成 → 目标库自动出现新的 T1/T2…；用户在 TargetDock 选中即可加载 2D/3D。

### 5.5 储层工作台（路径 B，次要）
储层角点剖面渲染依赖 matplotlib，**暂留本地**。把工作台 `_save_selection` / `_propagate_*` 的输出从「emit `ReservoirSelectionLayer`」改为「写 `GeoTarget`（cell 进 `TargetFrame.cell_ids_ref`，经 `RemoteTargetStore`）」：
- 服务端已新增 `POST /sam3/targets/cells`：body 为 `.npy` 的 `(N,3)` cell IJK；服务器分配目标 ID 并写 `cells/<target_id>/...npy`。
- 客户端已新增 `RemoteTargetStore.create_cell_target()`：以二进制 `.npy` 上传，避免大 JSON。
- `SAM3Workbench` 已新增可选 `target_store`：若传入远程目标库，单帧保存/沿轴追踪/视频追踪都会同步写 `GeoTarget`，并把 `target_id`/`external_cells_ref` 写入本地 layer metadata。
- 仍待接线：当前主程序没有实际构造 `SAM3Workbench` 的入口；恢复该入口时，构造函数传入 `target_store=MainWindow.target_store`，并连接 `target_committed` → `TargetDock.refresh()`。
- `ReservoirSelectionLayer` 改为「从 GeoTarget 渲染 cell」的视图层（`target_dock._load_selected_cells` 已是此形态）。

### 5.6 测试
- 客户端：mock `urlopen`，断言 `submit_track` POST body 结构 = §2.1 契约。
- `RemoteSAM3TrackTask`：注入假 client（submit_track 返回 id；poll 依次返回 running→done；result 含 target_ids），断言 `finished` 带正确 result、`mark_ready` 被调。
- 端到端（服务器，手动）：远程模式框 2 个目标→追踪→TargetDock 出现 T1/T2，本机 `nvidia-smi` 无占用、服务器有。

**DoD**：远程模式下，剖面框选→追踪→目标库出现多目标，全程本机不跑 GPU。

---

## 步骤 6 ·（§4）真多卡（前置：步骤 5 端到端已通）

### 6.1 worker 进程池（[server/.../sam3/jobs.py](../server/src/yj_studio_server/sam3/jobs.py)）

把共享单引擎的 `ThreadPoolExecutor` 换成「每卡一进程」的 `ProcessPoolExecutor`：

```python
import os
from concurrent.futures import ProcessPoolExecutor

_ENGINE = None   # 每个 worker 进程内的全局引擎

def _init_worker(gpu_id: int, engine_kwargs: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global _ENGINE
    from .engine import SAM3Engine
    _ENGINE = SAM3Engine(**engine_kwargs)   # 绑定到这张卡

def _worker_segment(rgb, **kw):   # 在 worker 进程跑
    return _ENGINE.segment(rgb, **kw)       # 返回可 pickle 的 [{"mask":ndarray,...}]

def _worker_collect(frames_dir, **kw):
    from .tracking import collect_object_frames
    return collect_object_frames(_ENGINE, frames_dir, **kw)   # 返回 {obj_id:{idx:ndarray}}

class GpuPool:
    def __init__(self, gpu_ids, engine_kwargs):
        # 每卡一个单进程 executor，便于按卡路由；或一个池+initializer 轮询
        self._pools = {g: ProcessPoolExecutor(max_workers=1,
                          initializer=_init_worker, initargs=(g, engine_kwargs))
                       for g in gpu_ids}
    def submit_segment(self, gpu_id, rgb, **kw): return self._pools[gpu_id].submit(_worker_segment, rgb, **kw)
    def submit_collect(self, gpu_id, frames_dir, **kw): return self._pools[gpu_id].submit(_worker_collect, frames_dir, **kw)
```

### 6.2 关键：写库收敛到主进程（§1.1 的进程内锁对多进程失效）

worker **只计算、不碰 `targets.json`**。已有的 `sam3/tracking.py` 正好支持：
- worker 跑 `collect_object_frames` → 回传 `collected`（纯 numpy，可 pickle）。
- **主进程**调 `persist_tracked_targets(store, collected, ...)`，在 `store.mutate()` 锁内统一落库 → 无跨进程写冲突，ID 唯一。

`_run_sam3_batch_job` 改为：把帧 round-robin `submit_segment` 到各卡 → 收集所有 future 的 mask+meta → 主进程持锁统一 `add_single_frame_target`。

### 6.3 `/sam3/gpus` 返回真实负载
worker 心跳上报 `torch.cuda.mem_get_info()`；主进程聚合。

### 6.4 测试
- 假 `GpuPool`（用 ThreadPool 模拟，worker 函数 sleep+返回固定 mask），断言 N 帧分发到 len(gpu_ids) 个执行器、结果数=输入、主进程落库后目标数正确、无 ID 冲突。
- 真多卡（服务器手动）：N=100 帧，断言 4 卡均占用、总时长≈1/4。

**DoD**：四卡并行、提速≈4×、目标库无冲突。

---

## 步骤 7 ·（§5）展示

### 7.1 2D 叠加编号/轮廓/类别色（[view/renderers/mask_renderer.py](../apps/yj_studio/src/yj_studio/view/renderers/mask_renderer.py) + 2D section overlay）
- ✅ 类别配色表：已新增 `targets/style.py`，提供 `target_type_color()`。
- ✅ 轮廓与编号：`view_2d_section.py` 对带 `target_id` 的 `MaskLayer` 画 `contour(level=0.5)` 与 `Tn` 标签。
- ✅ `MaskLayer` metadata：`build_mask_layer()` 已自动写入 `area_px/bbox/centroid`，并按 `target_type` 应用颜色。
- 待做：dock 子面板画 `area_px` / `centroid` 随 index 的折线（matplotlib），数据来自 `GeoTarget.frames`。

### 7.2 3D 体按真实 index 重建（修 #4，[targets/store.py](../apps/yj_studio/src/yj_studio/targets/store.py) `write_mask3d_cache`）

现状 `np.stack(masks, axis=0)` 按 trajectory 顺序堆叠，忽略真实 index 间隔。改为按真实 index 定位：

```python
def write_mask3d_cache(self, target: GeoTarget) -> Path:   # 改成收 target，拿得到 index
    frames = [f for f in target.frames.values() if f.mask_ref]
    if not frames:
        np.save(path, np.zeros((0,0,0), np.uint8)); return path
    idxs = [f.index for f in frames]
    lo, hi = min(idxs), max(idxs)
    sample = self.read_mask(frames[0].mask_ref)
    H, W = sample.shape
    vol = np.zeros((hi - lo + 1, H, W), np.uint8)
    for f in frames:
        m = self.read_mask(f.mask_ref)
        if m.shape == (H, W):
            vol[f.index - lo] = (m > 0).astype(np.uint8)   # 缺帧保持 0
    np.save(path, vol)
    return path
```
`/sam3/targets/{id}/mask3d` 路由对应改为传 target（已能拿到）。返回 header 带 `lo` 便于本机定位。

**DoD**：非连续帧的 3D 体在深度方向位置正确（缺帧留空）。

---

## 步骤 8 ·（§6）标注回流 + 真实训练

### 8.1 修正 mask 回写（闭环关键）
- `GeoTarget` 加 `edits: list[dict]`（model.py）。
- 服务端 `PUT /sam3/targets/{id}/mask/{axis}/{index}`（body=.npy 字节流）：
  ```python
  @app.put("/sam3/targets/{target_id}/mask/{axis}/{index}")
  def put_target_mask(target_id, axis, index, request: Request, project=..., volume_id=...):
      raw = await request.body()              # .npy bytes
      mask = np.load(BytesIO(raw))
      store = _target_store(...)
      with store.mutate() as ts:
          t = _get_target_or_404(store, target_id, target_set=ts)
          frame = store.frame_from_mask(target_id=target_id, axis=_target_axis(axis),
                      index=int(index), mask=mask, origin="edited")
          t.add_frame(frame)                  # 覆盖同 key
          t.edits.append({"at": _utc_now_iso(), "axis": axis, "index": int(index)})
      return {...}
  ```
- 客户端 `RemoteTargetStore.put_mask(target_id, axis, index, mask)`；本机笔刷/橡皮修过后调用。

### 8.2 训练集格式先定死（即使训练后端后做）
- ✅ `export_confirmed_to_coco` 已包含 **train/val/test 划分**字段 + `schema_version`，且 `edited` 帧也纳入（不只 `confirmed`）。
- ✅ `_run_train_job` 已支持配置式训练后端：导出后运行 `training.command`，采集 `metrics.json` 与 checkpoint，写入 `ModelRegistry.add_model(metrics=...)`。
- ✅ `activate` 已调用 `SAM3Engine.reload_checkpoint(path)`，下一次推理懒加载激活权重。
- 待做：真实 SAM3 训练脚本、评估集指标、失败样本回流、回滚 UI。

**DoD**：修正 mask 能回写并进训练集；训练集带划分；模型可激活/回滚。

---

## 步骤 9 ·（§7）工程健壮性（穿插做）

| 项 | 状态 | 落点/做法 |
|---|---|---|
| 切片缓存 LRU | ✅ | `server/cache.py` + `app._enforce_slice_cache_budget(cfg)`；写 `cache/slices/*.npy` 后按 `slice_cache_max_gb` 的 mtime/LRU 清理 |
| Job 持久化 | ✅ | `sam3/jobs.py`；`done/error/cancelled` 终态落 `runtime/server/jobs/<id>.json`，启动/get miss 时回读 |
| schema_version | ✅ | `targets/model.py`；`TargetSet.schema_version=1`，`TargetFrame/GeoTarget/TargetSet` 忽略未知字段 |
| 输入校验 | ✅ | `sam3/validation.py`；`submit_sam3_job` / batch / extract 入队前校验 boxes、points、track/batch 帧数、`keep_top_k`、`confidence` |
| 小 bug #10 | ✅ | `store.py`；`metadata_is_lightweight` 改为检测大内联数组，不再被 `mask_ref` 误判 |
| 服务端测试 | 部分 ✅ | 已补 job/cache/validation/reassociate/tracking 等 FastAPI-free 测试；FastAPI 装上后再补 TestClient 完整路由契约与多卡假池测试 |

---

## 总览：依赖与建议顺序

```text
步骤4 §2.3/2.4  reassociate.py + detect_merge_split + 接入 _run_track_job   [核心闭环]
步骤5 §3        submit_track → RemoteSAM3TrackTask → AI 面板「追踪」→ 刷新目标库   [最高价值·端到端]
步骤6 §4        ProcessPool 每卡一进程 + 主进程 persist 统一落库               [前置=步骤5]
步骤7 §5        2D 编号/轮廓/类别叠加 ; 3D 按 index 重建
步骤8 §6        mask 回写 PUT + edits ; 训练集划分格式 ; 真实训练后端
步骤9 §7        缓存 LRU / job 持久化 / schema_version / 输入校验 / 小bug / 基础测试  [基础项已完成]
```

> 建议先做**步骤 5（端到端贯通）**：它把已经建好的服务端 track 能力真正接到用户手上，价值最高、且不依赖步骤 4。步骤 4 的跨 job 关联在「体级文本提取」频繁使用时再做收益最大。
```
