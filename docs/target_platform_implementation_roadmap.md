# 实施路线与开发步骤文档：从 SAM3 分割工具到地质目标管理平台

本文档把你提出的十个发展方向（辅助标注、目标编号、视频追踪实例管理、目标对象化、语言交互、框选改放大、模型训练、二三维展示、多卡调度、闭环扩展）整理为一条**可落地、可分步实现、可逐步验证**的开发路线。

它要回答的核心问题：

1. 当前系统已经具备什么功能；
2. 当前系统存在什么问题；
3. 每一个后期功能应该怎么实现；
4. 功能之间的数据如何流动；
5. 每一步开发完成后如何测试；
6. 哪些功能优先级最高、哪些可后期扩展；
7. 如何利用服务器四卡显卡提升效率；
8. 如何把本地交互环境和服务器计算环境连接起来；
9. 如何最终形成「标注—训练—推理—追踪—展示—修正—再训练」闭环。

> 配套文档：项目整体状态见 [`docs/current_project_status_and_roadmap.md`](current_project_status_and_roadmap.md)。本文件聚焦 SAM3 / 目标管理 / 训练这条主线，与那份文档的「阶段 3/4/5」衔接。

---

## 当前实现状态（2026-06-11）

M3-M8 已完成第一轮代码落地，重点是把目标、mask、任务、训练集和模型版本变成可持久化的工程对象：

- M3：新增 `yj_studio.targets`，包含 `GeoTarget / TargetFrame / TargetSet / TargetStatus`，目标 ID 默认 `T1/T2/...`；`targets.json` 只存元数据，二维 mask 与 cells 分别写入 `.npy`。
- M4：服务器新增目标读取/编辑/合并/拆分、`mask`、`cells`、`mask3d` API；本地新增 `RemoteTargetStore` 与 `TargetDock`，可刷新目标、确认/删除/合并/拆分，并把目标加载为 2D mask 或 3D selection。
- M5-M6：服务器新增 `/sam3/jobs/batch`、`/sam3/gpus`、`/sam3/extract`；当前是进程内批量任务骨架，GPU worker 信息从配置暴露，真实多 GPU 占用仍需要服务器手动验证。
- M7-M8：新增 confirmed 目标 COCO/PNG/mask 导出、训练 job、模型版本 registry、模型激活 API；暂不做账号权限，符合当前无 token 的决定。
- 本地运行入口 `local/run_viewer.py` 已支持 `project_id` 与 `target_backend`，VSCode 直接运行时继续读取 `local/config/local.yaml`。
- **后续加固（2026-06-11 第二轮）**：并发写 `targets.json` 加锁 + ID 集中分配（`TargetStore.mutate()`）；mask 方向约定统一为单一 `sam3_mask_to_layer()`；**服务端多目标视频追踪 + 跨帧 ID 一致**（`sam3/engine.track_video` + `sam3/tracking.py` 核心 + `app._run_track_job`，仅地震体轴向切片）。均有本机测试。

本轮没有在远程服务器上启动/停止/验证服务；真实 GPU、长任务和服务端依赖检查仍按约定由用户手动执行。

> **实时进度与整改路线以 [`project_review_and_remediation.md`](project_review_and_remediation.md) 为准**（含已实现快照、完成度表、下一步精细路线）。

---

## 第一部分：当前系统已经具备什么（现状盘点）

下面是代码里**已经真实存在**的能力，按数据流顺序列出，给出对应文件，作为后续改造的起点。

### 1.1 SAM3 模型加载

- [`ai/service.py`](../apps/yj_studio/src/yj_studio/ai/service.py) — `AIService`：懒加载，工作线程里 `build_sam3_image_model` + 可选 `build_sam3_video_model`，状态机 `IDLE/LOADING/READY/BUSY/ERROR`。
- [`ai/config.py`](../apps/yj_studio/src/yj_studio/ai/config.py) — `SAM3Config`：`checkpoint_path=weights/sam3.pt`、`device="cuda"`、`load_video_model`。
- **关键事实：模型当前完全跑在本机 GPU 上。** 视频模型依赖 triton，Windows 上靠 `triton-windows`，失败时自动降级为「仅图像模式」。

### 1.2 两条 SAM3 使用路径（重要）

系统现在有**两个并存、互不相通**的 SAM3 入口，后续要统一：

**路径 A：AI 面板 → 单剖面分割（面向地震体）**
- [`ui/docks/ai_dock.py`](../apps/yj_studio/src/yj_studio/ui/docks/ai_dock.py)：文本 + 几何提示（框/点），通过 `ai_box_prompt`/`ai_point_prompt` 工具收集。
- [`algorithms/builtin/ai/sam3_segment.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/ai/sam3_segment.py) — `SAM3SegmentAlgorithm`：从 `volume_store.get_slice` 取一张切片 → `slice_to_rgb_image` → SAM3 → `decode_sam3_masks`。
- 输出：一个或多个 `MaskLayer`（2D），经撤销栈加入图层。
- 点提示没有原生接口，被包成小框；框是 `[cx,cy,w,h]` 归一化送入 `add_geometric_prompt`。

**路径 B：SAM3 工作台 → ROI 剖面分割 + 追踪（面向储层体）**
- [`view/view_sam3_workbench.py`](../apps/yj_studio/src/yj_studio/view/view_sam3_workbench.py) — `SAM3Workbench`：绑定一个固定 ROI，沿 i/j 轴逐帧。
- 三种追踪：
  - `_run_sam3`：当前帧图像模式分割；
  - `_propagate_along_axis`：「框追踪」，用上一帧 mask 的 bbox 当下一帧框提示，逐帧重调图像模型；遇低分（<0.25）停。
  - `_propagate_with_video_predictor`：「视频追踪」，把 ROI 帧导出成临时 JPEG 序列 → `init_state` → `add_prompt(obj_id=1)` → `propagate_in_video` 正反向。
- mask 通过 `frame.cell_id_grid` 反查成储层 cell（i,j,k）。
- 输出：`ReservoirSelectionLayer`（3D，cell-IJK 集合）。见 [`scene/layers/reservoir_selection_layer.py`](../apps/yj_studio/src/yj_studio/scene/layers/reservoir_selection_layer.py)。

### 1.3 任务调度框架（已为远程/子进程预留接缝）

- [`algorithms/runner.py`](../apps/yj_studio/src/yj_studio/algorithms/runner.py) — `AlgorithmRunner.submit` 按 `runs_in_subprocess` 分流：
  - `True` → `AlgorithmTask`（`multiprocessing.Process` + `Queue` 协议，消息含 `module:Class` 路径、params、序列化图层）；
  - `False` → `InProcessAlgorithmTask`（`QThread`，SAM3 走这条，因为模型在 GPU）。
- 四个统一信号：`progress / finished / errored / cancelled`。
- **runner.py 的注释已经写明**：这个进程协议「打算在阶段 9 承载 SAM3 子进程，worker 跑在另一个 conda 环境」。**这正是把 SAM3 迁到服务器的天然接缝。**

### 1.4 服务器侧（已具备）

- [`server/src/yj_studio_server/app.py`](../server/src/yj_studio_server/app.py)：FastAPI，`/health`、`/volumes`、`/slice`（返回 2D `.npy` 字节流，带磁盘切片缓存）。
- 本地远程取片：[`data/remote_volume_store.py`](../apps/yj_studio/src/yj_studio/data/remote_volume_store.py) — `RemoteVolumeStore.get_slice`，内存 LRU。
- **目前服务器只做切片，没有任何 SAM3 / 任务队列 / 结果存储。**

---

## 第二部分：当前系统存在什么问题（改造动机）

把这些问题列清楚，因为后续每一个功能本质上都在解决其中一条。

| 编号 | 问题 | 影响 | 对应你的方向 |
|---|---|---|---|
| P1 | **SAM3 全在本机 GPU 跑** | 本机重、四卡服务器闲置、与远程架构矛盾 | 方向 9 |
| P2 | **只支持单目标**（video 永远 `obj_id=1`） | 无法多目标、无法编号 | 方向 2/3/5 |
| P3 | **结果是匿名 mask/cell 集合，无身份** | 无 T1/T2、无类型、无属性、无轨迹 | 方向 2/3/4 |
| P4 | **追踪结果不落盘为 mask 体** | video 临时 JPEG 用完即删，selection 只存进场景 JSON | 方向 1/8、状态文档阶段 5 |
| P5 | **框 = 裁切/ROI 绑定** | 局部坐标，跨帧偏移、易追丢 | 方向 6 |
| P6 | **两条 SAM3 路径割裂** | 维护成本高，数据模型不统一 | 全部 |
| P7 | **无标注数据格式/训练流程** | 无法微调、无法形成闭环 | 方向 1/7/10 |
| P8 | **无语义批量提取** | 不能「提取本页所有浊积体」 | 方向 5 |

**最关键的两条**：P1（不迁服务器，多卡和大规模都谈不上）和 P3（不把目标对象化，编号/追踪/统计/训练全都没有载体）。这两条是整条路线的地基。

---

## 第三部分：核心数据模型（贯穿所有功能的「目标对象」）

你的方向 2/3/4 的本质是同一件事：**把匿名 mask 升级为有身份的地质目标实例（GeoTarget）。** 这是整个平台的中枢数据结构，必须先定义，后面所有功能都挂在它上面。

建议新增数据结构（落在 `apps/yj_studio/src/yj_studio/targets/` 新模块 + 服务器对应 schema）：

```text
GeoTarget（目标实例）
  id            : "T1" / "Trap-001"        # 全局唯一、跨帧一致
  type          : "trap" | "turbidite" | "fault" | "sandbody" | ...
  volume_id     : 所属体数据
  status        : active | lost | merged | split | confirmed
  source        : sam3_interactive | sam3_text | sam3_video | manual
  frames        : { axis, index → TargetFrame }   # 每帧的存在形态
  trajectory    : [ (index, centroid_ij, area, score) ... ]
  edits         : [ 人工修正记录 ]
  created/updated, score, notes

TargetFrame（目标在某一帧的形态）
  axis, index          # 全局坐标系下的剖面位置
  mask_ref             # 指向落盘的 2D mask（不内联大数组）
  bbox, centroid, area
  cell_ids (可选)      # 储层体反查结果
  origin               # detected | propagated | edited
```

**落盘格式**（对齐状态文档「SAM3 结果以 mask 为核心」）：

```text
data/results/sam3/
  <project>/
    targets.json                 # 所有 GeoTarget 的元数据（无大数组）
    masks/
      <target_id>/<axis>_<index>.npy   # 2D bool mask，按帧
    volumes/
      <target_id>_mask3d.npy           # 可选：堆叠成的 3D mask 体
    previews/
      <target_id>.png
```

> 设计原则：**元数据（targets.json）与大数组（masks/）分离**。这样 UI 加载目标列表很快，3D 重建时才按需读 mask。和现在 `ReservoirSelectionLayer` 把 cell_ids 内联进场景 JSON 的做法不同——后者目标多了会让工程文件爆炸。

---

## 第四部分：分阶段实施路线（含优先级、实现、数据流、测试）

下面按**依赖顺序**分 8 个里程碑。每个里程碑标注优先级（P0 最高）、做什么、数据怎么流、怎么测。

---

### 里程碑 M1：SAM3 迁移到服务器（P0，地基）

> 解决 P1。对应方向 9 的前半、状态文档阶段 4。**所有后续功能都建议直接在服务器侧实现**，避免「先本机做一遍、再迁一遍」。

**做什么**
1. 服务器新增 SAM3 推理服务：在 `server/` 下加 `yj_studio_server/sam3/`，加载 SAM3 image + video 模型（服务器有 GPU，无 triton 困扰）。
2. 新增异步任务接口（对齐状态文档建议）：
   ```text
   POST /sam3/jobs            # 提交：volume_id, axis, index(或区间), prompts(text/box/point), 模式(segment/track)
   GET  /sam3/jobs/{job_id}   # 查询状态/进度
   GET  /sam3/jobs/{job_id}/result   # 取 mask（.npy）/ targets 元数据
   POST /sam3/jobs/{job_id}/cancel
   ```
3. **提示坐标统一用归一化或全局像素坐标**，服务器侧自己取切片（复用 `/slice` 的 `np.load(mmap)`），避免本机传大图。
4. 本机改造：新增 `RemoteSAM3Client`（仿照 `RemoteVolumeStore` 的 urllib 风格），并在 `AlgorithmRunner` 增加一种「远程任务」handle，复用现有 `progress/finished/errored/cancelled` 四信号——UI 完全不用改。

**数据流**
```text
本机：用户提示(axis/index/box/point/text)
  → RemoteSAM3Client.submit  → POST /sam3/jobs
服务器：取切片 → SAM3 推理 → 写 masks/*.npy + 更新 targets.json
本机：轮询 GET /jobs/{id} → 完成后 GET result → 渲染成图层
```

**测试**
- 服务器侧 `pytest`：构造小体 + 假 prompt，断言 job 完成、mask shape 正确、落盘路径存在。
- 端到端：本机点一个框 → 看 AI 面板出 mask，且**本机 GPU 占用为 0、服务器 GPU 上升**（这是 P1 修复的判据）。
- 回归：保留本机 `device="cpu"` 离线模式作为服务器不可用时的备选。

**注意**：服务器是生产环境，启动/重启/验证都由你手动控制（见状态文档第 9 节）；我只提供代码与命令。

---

### 里程碑 M2：框选改为「放大」，追踪回到全局坐标（P0）

> 解决 P5。对应方向 6。**这一步要在 M1 之后、多目标之前做**，因为坐标系是后面编号一致性的前提。

**做什么**
1. `SAM3Workbench` 当前把 ROI 当「裁切边界」（ROI 外的 cell 永不绘制，渲染 ROI 子图）。改为：
   - 框 = **视图缩放/局部精修**（只影响显示与点选精度）；
   - 检测/分割/追踪 = 在**完整剖面（完整 numpy 坐标）**上进行。
2. 渲染管线 [`reservoir/sam3_render.py`](../apps/yj_studio/src/yj_studio/reservoir/sam3_render.py) 的 `render_roi_section` 拆成「全剖面渲染」+「显示窗口（缩放框）」两层；mask 与 `cell_id_grid` 始终在全剖面坐标下对齐。
3. 追踪时不再以「ROI 裁切图」为单位喂 video predictor，而是全剖面（或足够大的稳定窗口），保证前后帧像素位置一致。

**数据流**
```text
之前：ROI裁切图 → SAM3 → mask(局部坐标) → 拼回去（易错位）
之后：全剖面图 → SAM3 → mask(全局坐标) ；缩放框只改 matplotlib 显示范围
```

**测试**
- 取一个明显斜穿多帧的目标，对比改造前后**跨帧质心漂移**：改造后质心应连续、不跳变。
- 单帧分割结果与全剖面坐标叠加无偏移（mask.shape == 全剖面 shape）。

---

### 里程碑 M3：目标对象化 + 编号一致性（P0，中枢）

> 解决 P2/P3。对应方向 2/3/4。落地第三部分的 `GeoTarget` 数据模型。

**做什么**
1. 服务器与本机共用 `GeoTarget` / `TargetFrame` 定义（pydantic 模型）。
2. 多目标分割：SAM3 一次返回多个候选 → 每个候选成为一个 `GeoTarget`，分配 `T1/T2/...`。
3. **视频追踪支持多对象**：现在 `_extract_video_mask` 只挑 `obj_id=1`；改为 `add_prompt` 多个 `obj_id`，传播时按 `out_obj_ids` 分别收集，**obj_id ↔ GeoTarget.id 固定映射**，保证跨帧一致。
4. 目标管理面板（新 dock）：列表显示 T1/T2…、类型、状态、帧范围、面积；支持改名、改类型、删除、合并、拆分、确认。
5. 丢失重关联：某帧没检出时标 `lost`；重新出现时按 IoU/质心距离与历史目标匹配，恢复同一 ID。

**数据流**
```text
SAM3 多候选/多obj_id → 实例化 GeoTarget(分配ID)
  → 每帧写 TargetFrame(mask_ref, bbox, centroid, area, cell_ids)
  → targets.json 持久化
目标管理面板 ← 读 targets.json ；人工修正 → 写回 + 记 edits
```

**测试**
- 两个目标同帧：断言得到两个稳定 ID，传播后帧间 ID 不串。
- 人为隔帧遮挡：丢失后重现，断言重关联回原 ID（而非新 ID）。
- targets.json 往返序列化一致。

---

### 里程碑 M4：结果落盘 + 二维/三维展示（P1）

> 解决 P4。对应方向 8、状态文档阶段 5。

**做什么**
1. 按第三部分目录把 mask 落盘；提供 `GET /sam3/targets`、`GET /targets/{id}/mask3d` 等读接口。
2. 二维：在剖面视图叠加 mask + 轮廓 + 编号 + 类别（复用 `MaskLayer` 渲染，扩展显示编号标签）。
3. 三维：把某目标各帧 mask → cell-IJK → 现有 `ReservoirSelectionLayer`/`ReservoirBodyLayer` 渲染（这条已跑通，只是改为从 `GeoTarget` 喂数据，而非临时变量）。
4. 轨迹/面积变化曲线、修正前后对比，作为目标管理面板的子视图。
5. 导出：mask、矢量轮廓、编号表、属性表、3D 体。

**数据流**
```text
targets.json + masks/  → 二维叠加 / 三维体重建 / 统计曲线 / 导出文件
```

**测试**
- 关闭软件重开，目标与 mask 能从磁盘恢复（验证 P4 真正修复）。
- 3D 重建 cell 数与各帧 cell 并集一致。

---

### 里程碑 M5：多卡调度与批量推理（P1）

> 对应方向 9 后半。前提是 M1 已把推理放到服务器。

**做什么**
1. 服务器侧任务队列（轻量即可：进程内 asyncio 队列 + 每卡一个 worker；或 Redis/RQ 视规模）。
2. 多 GPU 分配：`CUDA_VISIBLE_DEVICES` 绑定 4 个 worker，每 worker 持一份模型；批量任务（多帧/多页）按帧切分到不同卡。
3. 训练任务与推理任务分离（不同卡或时分），失败自动重试，长任务进度上报（复用 `/jobs/{id}` 进度字段）。
4. 结果自动合并回单一 `GeoTarget` 集合。

**数据流**
```text
批量请求(N帧) → 队列 → 分发到 GPU0..3 并行 → 各自写 masks/ → 汇总 targets.json → 本机展示
```

**测试**
- 提交 N=100 帧批量任务，断言 4 卡均被占用、总时长≈单卡的 1/4、结果数=输入数。
- GPU 监控接口返回每卡负载。

---

### 里程碑 M6：语言交互与语义批量提取（P2）

> 对应方向 5。依赖 M3（要能产出多个带类型的目标）。

**做什么**
1. SAM3 已支持文本提示（`set_text_prompt`）。封装「提取本页所有 X」：对当前剖面用文本提示跑 SAM3，所有候选→多个 `GeoTarget`，类型=X。
2. 「整个数据体提取 X」：在 M5 批量调度上，对一组剖面跑文本提示 + 视频追踪，自动编号串联。
3. 语义指令解析层：把「提取所有可能的圈闭体」映射为 (type=trap, scope=current_page|whole_volume, mode=segment+track)。初期用规则/下拉即可，不必上 LLM。

**数据流**
```text
语言指令 → {type, scope, mode} → 批量 SAM3(文本提示) → 多 GeoTarget(编号+类型) → 展示
```

**测试**：给定文本提示，断言在多帧上产出稳定编号的同类目标集合。

---

### 里程碑 M7：标注数据管理 + 模型微调/重训（P2，闭环关键）

> 对应方向 1/7/10。这是把「用工具」变成「养模型」的一步。

**做什么**
1. 标注数据管理：把「人工确认/修正后的 GeoTarget」导出为训练标签（图像 + mask + 类别 + bbox），统一格式（COCO/自定义 JSON 皆可）。
2. 数据集划分：train/val/test；版本记录（每次标注/训练/模型更新留痕）。
3. 微调/重训：服务器侧训练脚本，基于已有标注微调 SAM3（或下游检测头）；模型版本管理 + 评估（IoU/Dice/Precision/Recall）+ 模型对比。
4. 最优模型部署回推理模块（M1 的服务可热切换 checkpoint）。
5. 失败样本回流：追踪失败/错检的帧重新进训练集。

**数据流（闭环）**
```text
原始数据 → 交互辅助标注(M1-M3) → 目标提取/编号/追踪 → 人工修正(M3/M4)
  → 标签积累(M7) → 微调/重训(M7) → 新模型部署回推理(M1) → 提升标注效率 …↺
```

**测试**：固定测试集上，微调后模型的 IoU/Dice 较基线不下降；模型版本可回滚。

---

### 里程碑 M8：协同与项目管理（P3，可后期扩展）

> 对应方向 10 的剩余项。

- 主动学习（优先挑高不确定区域让你标）、标注质量检查、多人协同标注、按工区/剖面/体/类型/模型版本组织的项目管理。
- 这些不阻塞主线闭环，等 M1–M7 跑顺后再做。

---

## 第五部分：优先级总览

| 优先级 | 里程碑 | 一句话 |
|---|---|---|
| **P0** | M1 SAM3 上服务器 | 地基；不做则多卡/批量/大规模无从谈起 |
| **P0** | M2 框改放大、全局坐标 | 编号一致性的坐标前提 |
| **P0** | M3 目标对象化 + 编号 | 中枢数据模型；编号/追踪/统计/训练的载体 |
| **P1** | M4 结果落盘 + 二三维展示 | 让结果可持久、可复用、可看 |
| **P1** | M5 多卡调度 + 批量 | 把四卡用起来 |
| **P2** | M6 语言语义批量提取 | 「提取所有浊积体」 |
| **P2** | M7 标注管理 + 微调重训 | 闭合「数据↔模型」回路 |
| **P3** | M8 协同 + 项目管理 | 团队化、规模化扩展 |

**建议实施顺序：M1 → M2 → M3 → M4 →（M5/M6 并行）→ M7 → M8。**
理由：M1/M2/M3 是地基且互相依赖；M4 让前三者产出可见可存；M5 与 M6 都建立在「服务器多目标」之上、可并行；M7 需要前面积累的修正数据才有意义。

---

## 第六部分：本地 ↔ 服务器如何连接（贯穿全程）

统一原则：**本机只发意图、收结果；服务器算一切重活。** 复用项目已有的两个接缝：

1. **数据接缝**：`RemoteVolumeStore` ←→ `/slice`（已有）。
2. **计算接缝**：`AlgorithmRunner` 的任务协议 ←→ `/sam3/jobs`（M1 新增）。runner.py 注释已预留此设计——新增一个「远程任务」handle，对外仍是 `progress/finished/errored/cancelled` 四信号，UI 零改动。

连接形态：

```text
本机 (local/config/local.yaml: mode=remote, server_url)
  - 体切片：GET /slice          （已通）
  - SAM3 任务：POST /sam3/jobs + 轮询 /jobs/{id}   （M1）
  - 目标读取：GET /sam3/targets, /targets/{id}/mask3d （M4）
服务器 (/root/quanbi, 4×GPU)
  - 模型常驻、任务队列、多卡分发、mask 落盘、训练
```

UI 始终显示「远程/本地模式 + 服务器连接状态 + server_url」（状态文档第 13 节已确认）。

---

## 第七部分：闭环总图

```text
原始地球物理数据
      │
      ▼
[M1/M2] 服务器 SAM3 + 全局坐标交互辅助标注
      │
      ▼
[M3] 目标对象化：编号一致 + 多目标 + 实例管理（GeoTarget）
      │
      ▼
[M4] mask 落盘 + 二维/三维展示 + 统计 + 导出
      │
      ├──[M5] 多卡批量推理 ─┐
      ├──[M6] 语言语义提取 ─┤  规模化产出目标
      │                     │
      ▼                     ▼
[M3/M4] 人工修正、确认、合并/拆分
      │
      ▼
[M7] 标注积累 → 微调/重训 → 模型版本管理 → 部署回 M1
      │
      └──────────────► 提升标注与提取效率（回到顶部）↺
```

---

## 第八部分：给实现者的关键提醒

1. **不要再在本机加新 SAM3 功能。** 新能力直接在服务器侧落（M1 之后），否则会重复迁移。
2. **先定 `GeoTarget` 数据模型再写功能**（M3）。它是后面一切的载体，结构定错会牵连所有里程碑。
3. **统一两条 SAM3 路径**（路径 A 单剖面 / 路径 B 工作台）到同一套服务器接口 + 同一套 `GeoTarget` 输出，消除 P6。
4. **元数据与大数组分离落盘**，别把 mask/cell 内联进场景 JSON。
5. **服务器操作（启动/重启/训练/验证）一律等你手动确认**，代码与命令我可以给，但不主动执行（状态文档第 9 节）。
6. 每个里程碑都先写最小测试（服务器 pytest + 一次端到端），再扩功能。
```
