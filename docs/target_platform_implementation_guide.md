# 详细实现指南：SAM3 目标管理平台（可直接照此写代码）

本文件是 [`target_platform_implementation_roadmap.md`](target_platform_implementation_roadmap.md) 的**落地版**。路线图回答「做什么、为什么、什么顺序」；本文件回答「**每个文件怎么改、函数签名长什么样、API 收发什么 JSON、怎么测**」。

深度优先级：P0 的 M1/M2/M3 给到代码骨架级；M4–M8 给到接口与步骤级。所有代码骨架都标注了归属文件路径，风格对齐项目现有约定（urllib 远程调用、`AlgorithmRunner` 四信号、FastAPI + pydantic、dataclass 图层）。

> 所有「在服务器上跑」的步骤（启动/重启/训练/验证）都由你手动执行；本指南只给代码与命令。

---

## 当前代码落点（2026-06-11）

本指南中的 M3-M8 已落地为第一版可运行骨架：

- 共享目标模型：`apps/yj_studio/src/yj_studio/targets/`
- 本地目标客户端：`apps/yj_studio/src/yj_studio/data/remote_target_store.py`
- 本地目标管理 Dock：`apps/yj_studio/src/yj_studio/ui/docks/target_dock.py`
- 服务器目标桥接：`server/src/yj_studio_server/targets.py`
- 服务器模型 registry：`server/src/yj_studio_server/sam3/models.py`
- 服务器 API 扩展：`server/src/yj_studio_server/app.py`
- 配置项：`project_id`、`target_backend`、`slice_cache_max_gb`、`sam3.gpu_ids`、`sam3.worker_count`、`training.*`

注意：多 GPU worker、训练后端和 SAM3 真实推理仍需要在服务器环境手动验证；本机只做轻量导入、模型层和假引擎契约检查。

**第二轮加固（2026-06-11）已落地**：
- 并发安全：`TargetStore.mutate()`（per-project 锁 + 原子读改写），服务端写路径全部改走它。
- 方向约定：唯一 `ai/adapters/mask_to_layer.sam3_mask_to_layer()`，三处手写转置已收敛。
- 服务端多目标追踪：`sam3/engine.track_video` + `init_track_state`、FastAPI-free 核心 `sam3/tracking.py`（`collect_object_frames` / `persist_tracked_targets`）、`app._run_track_job`；路由 `POST /sam3/jobs kind=track`、`/sam3/extract mode=track`。

> **实时进度、完成度与下一步精细路线以 [`project_review_and_remediation.md`](project_review_and_remediation.md) 为准。** 本指南的 M1–M8 骨架保留作设计参考。

---

## 第 0 章：先理解三个已有接缝（改造全部挂在这上面）

写代码前必须认清项目已经留好的三个挂载点，**不要另起炉灶**：

### 接缝 A：后端按环境变量切换（本机）

[`ui/main_window.py:1286`](../apps/yj_studio/src/yj_studio/ui/main_window.py) 的 `_make_volume_store()`：

```python
backend = os.environ.get("YJ_STUDIO_VOLUME_BACKEND", "local")
if backend == "remote" and server_url:
    return RemoteVolumeStore(server_url, timeout_s=...)
return VolumeStore()
```

→ **M1 照抄这个模式**做 `_make_sam3_backend()`：`remote` 时返回 `RemoteSAM3Client`，否则返回本机 `AIService`。

### 接缝 B：服务注入算法上下文（本机）

[`ui/main_window.py:86`](../apps/yj_studio/src/yj_studio/ui/main_window.py)：

```python
self.algorithm_runner.register_service("ai_service", self.ai_service)
self.algorithm_runner.register_service("volume_store", self.volume_store)
```

算法里通过 `ctx.services["ai_service"]` 取用（见 `sam3_segment.py:109`）。
→ **M1 让 `ai_service` 既可能是本机 `AIService`、也可能是 `RemoteSAM3Client`，两者实现同一个接口（鸭子类型）**，算法代码几乎不用改。

### 接缝 C：任务统一四信号（本机）

[`algorithms/runner.py`](../apps/yj_studio/src/yj_studio/algorithms/runner.py) 的 `AlgorithmTask`（子进程）/`InProcessAlgorithmTask`（线程）都暴露：

```python
progress = pyqtSignal(float, str)
finished = pyqtSignal(list, str)   # output layers, summary
errored  = pyqtSignal(str, str)
cancelled = pyqtSignal()
def start(self): ...
def cancel(self): ...
```

→ **M1 新增 `RemoteAlgorithmTask`**，对外暴露同样四信号，内部用 `QTimer` 轮询服务器 `/jobs/{id}`。UI（`ai_dock.py`）零改动。

### 接缝 D：服务器配置 + 接口（服务器）

[`server/src/yj_studio_server/config.py`](../server/src/yj_studio_server/config.py) 的 `ServerConfig` 已有 `results_root`、`runtime_root`、`volumes`。
[`server/src/yj_studio_server/app.py`](../server/src/yj_studio_server/app.py) 已有 `/slice` 用 `np.load(path, mmap_mode="r")` 取片。
→ **M1 在 app.py 加 `/sam3/jobs` 路由，复用 `_slice` 的取片逻辑喂模型。**

---

## 第 1 章：M1 — SAM3 迁移到服务器（P0）

### 1.1 目标与判据

- 提交一次 SAM3 分割，**本机 GPU 占用为 0，服务器 GPU 上升**。
- AI 面板（`ai_dock.py`）和工作台都能用，UI 不改逻辑。
- 服务器不可用时能降级回本机（保留 `mode=local`）。

### 1.2 服务器侧改造

#### 1.2.1 新增模型持有者 `server/src/yj_studio_server/sam3/engine.py`

进程内常驻一份 SAM3 模型（M5 再扩成多卡多 worker）。骨架：

```python
# server/src/yj_studio_server/sam3/engine.py
from __future__ import annotations
import numpy as np

class SAM3Engine:
    """Holds the SAM3 image processor (+ optional video predictor) on GPU.
    One instance per GPU worker. Built lazily on first job."""

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 resolution: int = 1008, load_video: bool = True) -> None:
        self._cfg = (checkpoint_path, device, resolution, load_video)
        self._processor = None
        self._video = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
        from sam3.model.sam3_image_processor import Sam3Processor
        ckpt, device, res, load_video = self._cfg
        image_model = build_sam3_image_model(device=device, checkpoint_path=ckpt)
        self._processor = Sam3Processor(image_model, resolution=res, device=device)
        if load_video:
            self._video = build_sam3_video_model(
                checkpoint_path=ckpt, device=device, strict_state_dict_loading=False)

    def segment(self, rgb: np.ndarray, *, text: str = "",
                boxes_norm: list[list[float]] | None = None,
                confidence: float = 0.4) -> list[dict]:
        """rgb: (H,W,3) uint8. boxes_norm: [[cx,cy,w,h], ...] normalised.
        Returns [{"mask": (H,W) bool, "score": float, "box": (x0,y0,x1,y1)}]."""
        self._ensure_loaded()
        from PIL import Image
        self._processor.set_confidence_threshold(confidence)
        state = self._processor.set_image(Image.fromarray(rgb))
        if text:
            state = self._processor.set_text_prompt(prompt=text, state=state)
        for b in (boxes_norm or []):
            state = self._processor.add_geometric_prompt(box=b, label=True, state=state)
        return _decode_state(state)   # 复用本机 decode_sam3_masks 的逻辑，搬一份到服务器
```

> 复用现有代码：`_decode_state` 直接照搬本机 [`ai/adapters/mask_to_layer.py`](../apps/yj_studio/src/yj_studio/ai/adapters/mask_to_layer.py) 的 `decode_sam3_masks`（它已不依赖 Qt）。`_apply_box_prompt` 的归一化逻辑照搬 [`sam3_segment.py:210`](../apps/yj_studio/src/yj_studio/algorithms/builtin/ai/sam3_segment.py)。

#### 1.2.2 新增任务存储 `server/src/yj_studio_server/sam3/jobs.py`

初版用「进程内字典 + 后台线程」即可（M5 换队列）。骨架：

```python
# server/src/yj_studio_server/sam3/jobs.py
from __future__ import annotations
import uuid, threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class JobState(str, Enum):
    queued = "queued"; running = "running"; done = "done"
    error = "error"; cancelled = "cancelled"

@dataclass
class Job:
    id: str
    kind: str                      # "segment" | "track"
    params: dict[str, Any]
    state: JobState = JobState.queued
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None   # 见 1.2.4 result schema
    error: str | None = None

class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
    def create(self, kind: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, params=params)
        with self._lock: self._jobs[job.id] = job
        return job
    def get(self, job_id: str) -> Job | None:
        with self._lock: return self._jobs.get(job_id)
    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in fields.items(): setattr(job, k, v)
```

#### 1.2.3 在 `app.py` 加路由

把 SAM3 引擎与 JobStore 挂到 `app.state`，新增四个路由。请求/响应契约（**这是你写本机 client 的依据**）：

```text
POST /sam3/jobs
  请求 JSON:
  {
    "kind": "segment",                # 或 "track"
    "volume_id": "seismic",
    "axis": "inline",                 # inline|xline|z
    "index": 600,                     # segment 用单值；track 用 {"lo":..,"hi":..,"seed":..}
    "prompts": {
      "text": "盐丘",
      "boxes": [[120,80,300,260]],    # 像素坐标 x0,y0,x1,y1（服务器自己归一化）
      "points": [[210,170]]           # 像素坐标；服务器包成小框
    },
    "confidence": 0.4,
    "keep_top_k": 3
  }
  响应: {"job_id": "ab12...", "state": "queued"}

GET /sam3/jobs/{job_id}
  响应: {"job_id","state","progress","message","error"}

GET /sam3/jobs/{job_id}/result        # state==done 才有
  响应: 见 1.2.4

POST /sam3/jobs/{job_id}/cancel
  响应: {"job_id","state":"cancelled"}
```

路由骨架（接 `app.py` 现有 `create_app`）：

```python
# server/src/yj_studio_server/app.py (新增片段)
from .sam3.engine import SAM3Engine
from .sam3.jobs import JobStore, JobState

def create_app(config=None):
    cfg = config or load_config()
    app = FastAPI(...)
    app.state.config = cfg
    app.state.jobs = JobStore()
    app.state.sam3 = SAM3Engine(checkpoint_path=str(cfg... / "weights" / "sam3.pt"))

    @app.post("/sam3/jobs")
    def submit_job(payload: dict, background: BackgroundTasks):
        job = app.state.jobs.create(payload["kind"], payload)
        background.add_task(_run_job, app, job.id)   # 初版用 BackgroundTasks；M5 换队列
        return {"job_id": job.id, "state": job.state}

    @app.get("/sam3/jobs/{job_id}")
    def job_status(job_id: str):
        job = app.state.jobs.get(job_id) or _404()
        return {"job_id": job.id, "state": job.state, "progress": job.progress,
                "message": job.message, "error": job.error}

    @app.get("/sam3/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = app.state.jobs.get(job_id) or _404()
        if job.state != JobState.done: raise HTTPException(409, "job not done")
        return job.result
    # + cancel
```

`_run_job` 做的事：取片（复用 `/slice` 的 `np.load(mmap)` + 取轴逻辑）→ `slice_to_rgb_image`（把这个 adapter 也搬一份到服务器，或放进共享包）→ `engine.segment(...)` → 写 mask 落盘 → 填 `job.result`。

#### 1.2.4 result schema（M1 先返回 mask 引用，M3 升级为 GeoTarget）

```text
GET /sam3/jobs/{job_id}/result 响应:
{
  "job_id": "ab12...",
  "axis": "inline", "index": 600,
  "volume_shape": [1684,1451,1201],
  "candidates": [
    {
      "score": 0.87,
      "box": [120,80,300,260],
      "mask_path": "results/sam3/<job>/cand0.npy",   # 服务器相对 data_root
      "mask_url": "/sam3/jobs/ab12/mask/0"            # 直接拉 .npy 字节流
    }
  ]
}
```

mask 用单独的 `GET /sam3/jobs/{id}/mask/{k}` 返回 `.npy` 字节流（复用 `/slice` 的 `FileResponse` 写法），**避免把 bool 大数组塞进 JSON**。

#### 1.2.5 服务器配置追加（`server/config/server.yaml`）

```yaml
sam3:
  checkpoint: weights/sam3.pt
  device: cuda
  resolution: 1008
  load_video: true
results_subdir: results/sam3
```

`config.py` 的 `ServerConfig` 加一个 `sam3: dict = field(default_factory=dict)` 字段并在 `load_config` 里读出。

### 1.3 本机侧改造

#### 1.3.1 新增 `apps/yj_studio/src/yj_studio/ai/remote_client.py`

仿 `RemoteVolumeStore` 的 urllib 风格，**对外暴露和本机 `AIService` 相同的语义**（`is_ready()`、提交任务）。但 SAM3 提交走任务模型，所以核心是「提交 + 返回 job_id」：

```python
# apps/yj_studio/src/yj_studio/ai/remote_client.py
from __future__ import annotations
import json
from urllib.request import urlopen, Request

class RemoteSAM3Client:
    def __init__(self, server_url: str, timeout_s: float = 180.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s

    def is_ready(self) -> bool:
        try:
            with urlopen(f"{self.server_url}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def submit_segment(self, *, volume_id, axis, index, text="",
                       boxes=None, points=None, confidence=0.4, keep_top_k=3) -> str:
        body = json.dumps({
            "kind": "segment", "volume_id": volume_id, "axis": axis, "index": index,
            "prompts": {"text": text, "boxes": boxes or [], "points": points or []},
            "confidence": confidence, "keep_top_k": keep_top_k,
        }).encode()
        req = Request(f"{self.server_url}/sam3/jobs", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())["job_id"]

    def poll(self, job_id: str) -> dict:
        with urlopen(f"{self.server_url}/sam3/jobs/{job_id}", timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def result(self, job_id: str) -> dict:
        with urlopen(f"{self.server_url}/sam3/jobs/{job_id}/result", timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def fetch_mask(self, job_id: str, k: int):
        import numpy as np
        from io import BytesIO
        with urlopen(f"{self.server_url}/sam3/jobs/{job_id}/mask/{k}", timeout=self.timeout_s) as r:
            return np.load(BytesIO(r.read()), allow_pickle=False)
```

#### 1.3.2 新增 `RemoteAlgorithmTask`（在 `algorithms/runner.py` 或新文件）

对外四信号一致，内部 `QTimer` 轮询：

```python
class RemoteSAM3Task(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list, str)
    errored  = pyqtSignal(str, str)
    cancelled = pyqtSignal()

    def __init__(self, client, params, *, volume_layer, parent=None):
        super().__init__(parent)
        self._client = client; self._params = params; self._vl = volume_layer
        self._job_id = None
        self._timer = QTimer(self); self._timer.setInterval(400)
        self._timer.timeout.connect(self._poll)

    def start(self):
        try:
            self._job_id = self._client.submit_segment(**self._params)
            self._timer.start()
        except Exception as e:
            self.errored.emit(str(e), "")

    def _poll(self):
        st = self._client.poll(self._job_id)
        self.progress.emit(st["progress"], st["message"])
        if st["state"] == "done":
            self._timer.stop()
            layers = self._build_layers(self._client.result(self._job_id))
            self.finished.emit(layers, "SAM3 完成")
        elif st["state"] in ("error", "cancelled"):
            self._timer.stop()
            (self.cancelled if st["state"]=="cancelled" else
             lambda *_: self.errored.emit(st.get("error",""), ""))()

    def _build_layers(self, result):
        # 把 candidates 的 mask 拉回来，转置对齐（同 sam3_segment.py:186），build_mask_layer
        ...
    def cancel(self):
        if self._job_id: self._client.cancel(self._job_id)
```

#### 1.3.3 在 `main_window.py` 接线

```python
# 仿 _make_volume_store()
def _make_sam3_backend():
    backend = os.environ.get("YJ_STUDIO_VOLUME_BACKEND", "local")
    url = os.environ.get("YJ_STUDIO_SERVER_URL", "").strip()
    if backend == "remote" and url:
        return RemoteSAM3Client(url, timeout_s=...)
    return AIService(SAM3Config(), parent=self)
```

`ai_dock.py` 的 `_on_run_clicked` 已经走 `runner.submit(...)`。让 `submit` 在检测到 `ai_service` 是 `RemoteSAM3Client` 时返回 `RemoteSAM3Task`（或在 dock 里分流）。**因为四信号一致，`_on_progress/_on_finished` 全部复用。**

### 1.4 M1 测试清单

- **服务器单测**（`server/scripts/run_tests.sh` 体系）：构造 8×8×8 假体 + 假 prompt，`engine.segment` 用 monkeypatch 假模型，断言 job 走到 `done`、`mask_path` 存在、shape 对。
- **契约测试**：用 `urllib` 打本机起的测试 app（`TestClient`），断言四个路由的 JSON 结构匹配 1.2.3。
- **端到端**（你手动）：`mode=remote` 起软件 → AI 面板放框 → 运行 → 出 mask；同时 `nvidia-smi`（服务器）看到占用、本机看不到。
- **降级**：服务器停掉时 `is_ready()` False，UI 提示而非崩溃。

### 1.5 M1 分步 checklist

1. [ ] 服务器：`sam3/engine.py`（搬 decode + 归一化）
2. [ ] 服务器：`sam3/jobs.py`
3. [ ] 服务器：`app.py` 加 4+1 路由 + `_run_job`
4. [ ] 服务器：搬 `volume_to_image.py` 到服务器可用位置
5. [ ] 服务器：`config.py` + `server.yaml` 加 `sam3` 段
6. [ ] 本机：`ai/remote_client.py`
7. [ ] 本机：`RemoteSAM3Task`
8. [ ] 本机：`_make_sam3_backend()` + dock 分流
9. [ ] 测试：服务器单测 + 契约 + 端到端

---

## 第 2 章：M2 — 框改放大、追踪回全局坐标（P0）

### 2.1 现状问题定位

[`view/view_sam3_workbench.py`](../apps/yj_studio/src/yj_studio/view/view_sam3_workbench.py) 把 ROI 当**裁切边界**：
- 构造时绑定 `self._roi`，`render_roi_section(grid, axis, index, self._roi, ...)` 只渲染 ROI 子图；
- `_propagation_range()` 用 ROI 的 `il,ih/jl,jh` 限定帧范围；
- mask 反查只在 ROI 子图的 `cell_id_grid` 内。

[`reservoir/sam3_render.py`](../apps/yj_studio/src/yj_studio/reservoir/sam3_render.py) 的 `SAM3Frame` = `image(H,W,3)` + `cell_id_grid(H,W,3)`，尺寸由 ROI 长宽比驱动。**ROI 一变，像素网格就变 → 跨帧像素不对齐 → 追踪漂移。**

### 2.2 改造方案

**核心**：把「检测坐标系」与「显示窗口」解耦。

1. **检测坐标系 = 完整剖面**。新增 `render_full_section(grid, axis, index, transform=...)`（或给 `render_roi_section` 传「全剖面 bbox」），让每帧的 `image`/`cell_id_grid` 尺寸**只由完整剖面决定，与缩放框无关** → 跨帧像素稳定对齐。
2. **缩放框 = 仅显示**。`_on_box_drawn` 不再改变 ROI/裁切，改为 `self._axes.set_xlim/set_ylim`（matplotlib 视图缩放），用一个「重置缩放」按钮还原。
3. **追踪基于全剖面**。`_propagate_along_axis` / `_propagate_with_video_predictor` 用全剖面帧；`_propagation_range` 用完整轴范围（受性能影响可加「最大帧数」上限，但坐标系是全局的）。
4. ROI 概念可降级为「初始定位/感兴趣提示」，不再参与像素裁切。

### 2.3 数据流对比

```text
之前: ROI裁切图(尺寸随ROI变) → SAM3 → mask(局部) → 拼回(易错位)
之后: 全剖面图(尺寸恒定) → SAM3 → mask(全局) ; set_xlim/ylim 只改显示
```

### 2.4 测试

- 取斜穿 ≥10 帧的目标，记录每帧 `np.argwhere(mask).mean(0)` 质心 → 改造后质心序列应平滑、无因 ROI 变化导致的跳变。
- 缩放框操作后，`_render` 出的 mask.shape 不随缩放改变（恒等于全剖面 shape）。
- 同一目标在 segment 与 track 两条路径下 mask 落在同一全局坐标。

> 注意 M2 会改 `render_roi_section` 的调用点和 workbench 较多方法。建议先加 `render_full_section` 与现有并存，灰度切换，跑通后再删 ROI 裁切路径。

---

## 第 3 章：M3 — GeoTarget 数据模型 + 编号一致（P0，中枢）

### 3.1 数据结构（新模块 `apps/yj_studio/src/yj_studio/targets/`）

```python
# targets/model.py  —— 本机与服务器共用同一套字段定义（pydantic 便于 JSON 往返）
from __future__ import annotations
from pydantic import BaseModel, Field

class TargetFrame(BaseModel):
    axis: str                       # inline|xline|z|i|j
    index: int
    mask_ref: str                   # 相对路径: masks/<target_id>/<axis>_<index>.npy
    bbox: tuple[float,float,float,float]      # x0,y0,x1,y1 全局像素
    centroid: tuple[float,float]
    area_px: int
    score: float | None = None
    cell_ids_ref: str | None = None # 储层反查结果（可选，另存 .npy）
    origin: str = "detected"        # detected|propagated|edited

class GeoTarget(BaseModel):
    id: str                         # "T1" / "Trap-001"
    type: str = "unknown"           # trap|turbidite|fault|sandbody|...
    volume_id: str = ""
    status: str = "active"          # active|lost|merged|split|confirmed
    source: str = "sam3_interactive"
    frames: dict[str, TargetFrame] = Field(default_factory=dict)  # key=f"{axis}:{index}"
    trajectory: list[dict] = Field(default_factory=list)          # [{index,centroid,area,score}]
    edits: list[dict] = Field(default_factory=list)
    score: float | None = None
    notes: str = ""

class TargetSet(BaseModel):
    project: str
    volume_id: str
    next_seq: int = 1               # 下一个编号序号，保证 ID 唯一
    targets: dict[str, GeoTarget] = Field(default_factory=dict)
    def new_id(self, prefix="T") -> str:
        tid = f"{prefix}{self.next_seq}"; self.next_seq += 1; return tid
```

### 3.2 持久化布局

```text
data/results/sam3/<project>/
  targets.json                 # TargetSet.model_dump_json()，无大数组
  masks/<target_id>/<axis>_<index>.npy     # 2D bool
  cells/<target_id>/<axis>_<index>.npy     # 可选 (N,3) int32
  volumes/<target_id>_mask3d.npy           # M4 按需生成
  previews/<target_id>.png
```

读写函数 `targets/store.py`：`load_target_set(project) -> TargetSet`、`save_target_set(ts)`、`write_mask(target_id, axis, index, mask) -> ref`、`read_mask(ref) -> np.ndarray`。**元数据与大数组分开存**，别学 `ReservoirSelectionLayer` 把 cell_ids 内联进场景 JSON。

### 3.3 多目标 video predictor 改造（关键）

现状 [`view_sam3_workbench.py:64`](../apps/yj_studio/src/yj_studio/view/view_sam3_workbench.py) 的 `_extract_video_mask` 只挑 `obj_id=1`。改为多对象：

```python
# 1) seed 多个对象：每个候选 GeoTarget 一个 obj_id，建立固定映射
obj_to_target = {}        # obj_id(int) -> target_id(str)
for k, det in enumerate(seed_detections, start=1):
    predictor.add_prompt(inference_state=state, frame_idx=seed_local,
                         boxes_xywh=[det_box_xywh], box_labels=[1], obj_id=k, ...)
    obj_to_target[k] = target_set.new_id()      # T1, T2, ...

# 2) 传播时按 out_obj_ids 拆分收集，obj_id ↔ target_id 跨帧恒定
def _extract_all(outputs):
    masks = outputs["out_binary_masks"]; ids = outputs["out_obj_ids"]
    return {int(oid): _to_bool(masks[i]) for i, oid in enumerate(ids)}

for frame_idx_local, outputs in predictor.propagate_in_video(...):
    for oid, mask in _extract_all(outputs).items():
        tid = obj_to_target[oid]
        # 写 TargetFrame(mask_ref=..., origin="propagated") 到 target_set.targets[tid]
```

→ 同一物理目标在所有帧保持同一 `T{n}`，落地你的方向 2/3。

### 3.4 丢失重关联（方向 3 第 3 点）

某帧 `out_obj_ids` 缺某 oid → 该 target 该帧标缺；重现时按 **IoU + 质心距离**与历史活跃/丢失目标匹配：

```python
# targets/reassociate.py
def match(new_masks: dict[str, np.ndarray],         # 候选(临时 key)->mask
          active: dict[str, np.ndarray],             # target_id->上一次 mask
          iou_thresh=0.3) -> dict[str, str]:         # 临时 key -> 复用的 target_id(或新建)
    ...
```

### 3.5 目标管理 dock（新 `ui/docks/target_dock.py`）

列表列：ID / 类型(下拉) / 状态 / 帧范围 / 面积 / score。操作：改名、改类型、删、合并(并 cell+mask、保留小号 ID)、拆分、确认(status=confirmed)。所有修改写回 `targets.json` 并追加 `edits` 记录。选中目标 → 高亮其在当前剖面的 mask + 在 3D 视图重建体（复用 `ReservoirSelectionLayer` 渲染）。

### 3.6 统一两条路径

让 M1 的服务器 result schema（1.2.4 的 `candidates`）直接升级为返回 `GeoTarget` 列表；本机 segment 路径与 workbench track 路径都消费同一套 `TargetSet`。消除 P6。

### 3.7 测试

- 两目标同帧：`add_prompt` 两 obj_id，断言传播后 `targets` 有两条、帧间 ID 不串。
- 隔帧遮挡：人为删中间帧某 oid，断言重关联回原 ID。
- `TargetSet` JSON 往返：`model_validate(ts.model_dump())` 相等。
- mask 落盘/读回一致；targets.json 不含大数组。

---

## 第 4 章：M4–M8 实现要点（接口/步骤级）

### M4 结果落盘 + 二三维展示（P1）
- 服务器读接口：`GET /sam3/targets?project=&volume_id=`、`GET /sam3/targets/{id}`、`GET /sam3/targets/{id}/mask3d`（按帧 cell 并集堆叠成 3D，或返回各帧 cell）。
- 二维：扩展 `MaskLayer` 渲染，叠加轮廓 + 编号标签 + 类别色（按 `type` 配色表）。
- 三维：`GeoTarget.frames` 的 cell 并集 → `ReservoirSelectionLayer`（已有渲染器），改为从 target 喂数据。
- 轨迹/面积曲线、修正前后对比：target_dock 子面板（matplotlib）。
- 导出：mask(.npy)、轮廓(GeoJSON/shp)、编号表(csv)、属性表(csv)、3D 体(.npy)。
- 测试：重开软件能从 `targets.json` 恢复目标（验证 P4）。

### M5 多卡调度 + 批量（P1）
- 把 1.2.2 的 `BackgroundTasks` 换成真正队列：每张卡一个 worker 进程，`CUDA_VISIBLE_DEVICES=0..3`，各持一份 `SAM3Engine`。
- 调度：`POST /sam3/jobs/batch`（多帧/多剖面），按帧 round-robin 分卡；`GET /sam3/gpus` 返回每卡负载。
- 训练/推理分离：训练任务标 `queue=train` 走单独卡或时分。
- 失败重试 + 长任务进度（沿用 job.progress）。
- 测试：N=100 帧批量，4 卡均占用、总时长≈1/4、结果数=输入数。

### M6 语言/语义批量提取（P2）
- 封装 `extract_all(volume_id, type, scope, mode)`：`scope=page` 对当前剖面跑文本提示；`scope=volume` 在 M5 批量上对一组剖面跑文本提示 + video 串联编号。
- 指令解析初版用下拉/规则（type×scope×mode），不上 LLM。
- 测试：给定文本提示，多帧产出稳定编号的同类目标集合。

### M7 标注管理 + 微调/重训（P2，闭环关键）
- 导出训练标签：`confirmed` 的 `GeoTarget` → (image, mask, class, bbox)，COCO 或自定义 JSON。
- 数据集划分 train/val/test + 版本号；每次标注/训练留痕。
- 服务器训练脚本：基于标注微调 SAM3 或下游头；产出新 checkpoint。
- 模型版本管理 + 评估(IoU/Dice/P/R) + 对比；最优模型热切换给 `SAM3Engine`（改 checkpoint 路径重载）。
- 失败样本回流：追踪失败/错检帧 → 回训练集。
- 测试：固定测试集上微调后 IoU/Dice 不降；版本可回滚。

### M8 协同 + 项目管理（P3）
- 主动学习（挑高不确定帧）、质量检查、多人协同、按工区/剖面/体/类型/模型版本组织。不阻塞主线。

---

## 第 5 章：实现总顺序与每步「完成定义(DoD)」

| 步 | 里程碑 | 完成定义 |
|---|---|---|
| 1 | M1 | remote 模式下 AI 面板分割成功，本机 GPU=0、服务器 GPU 升 |
| 2 | M2 | 跨帧质心平滑无跳变，mask.shape 不随缩放变 |
| 3 | M3 | 多目标稳定编号、跨帧 ID 一致、targets.json 往返一致 |
| 4 | M4 | 重开软件目标可恢复、二维编号叠加、三维体重建 |
| 5 | M5 | 批量任务四卡并行、提速≈4× |
| 6 | M6 | 「提取所有 X」产出多目标 |
| 7 | M7 | 微调后指标不降、模型可热切换/回滚 |
| 8 | M8 | 按需扩展 |

**先做 M1→M2→M3（地基三件套），每件都先写最小测试再扩功能；M1 之后所有新能力直接在服务器侧实现，不再在本机重复。**

---

## 附：关键文件索引（改造落点）

| 功能 | 现有文件 | 动作 |
|---|---|---|
| 后端切换 | `ui/main_window.py:1286` `_make_volume_store` | 仿写 `_make_sam3_backend` |
| 服务注入 | `ui/main_window.py:86` | `ai_service` 可为 RemoteSAM3Client |
| 任务信号 | `algorithms/runner.py` | 新增 `RemoteSAM3Task` |
| 单剖面分割 | `algorithms/builtin/ai/sam3_segment.py` | decode/归一化逻辑搬服务器 |
| 工作台追踪 | `view/view_sam3_workbench.py` | M2 解耦坐标、M3 多 obj_id |
| 切片渲染 | `reservoir/sam3_render.py` | M2 加 `render_full_section` |
| mask 适配 | `ai/adapters/mask_to_layer.py` | 复用 decode；M3 产 GeoTarget |
| 图像化 | `ai/adapters/volume_to_image.py` | 搬一份到服务器 |
| 选择图层 | `scene/layers/reservoir_selection_layer.py` | M3/M4 由 GeoTarget 喂数据 |
| 服务器配置 | `server/src/yj_studio_server/config.py` | 加 `sam3` 段 |
| 服务器接口 | `server/src/yj_studio_server/app.py` | 加 `/sam3/*` 路由 |
| 新增模块 | — | `targets/`、`ai/remote_client.py`、`server/.../sam3/` |
```
