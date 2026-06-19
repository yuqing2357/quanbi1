# 项目体检与改进实施文档（2026-06-11）

本文件基于对当前代码的全量核对，针对「SAM3 → 地质目标管理平台」主线，逐项给出**问题 + 具体实现路径 + 数据流 + 测试 + 完成定义(DoD)**。

配套：
- **下一步代码级实现指南（剩余步骤 4–9，函数签名+骨架+契约+测试）：[`next_steps_detailed_implementation.md`](next_steps_detailed_implementation.md)** ← 照此写代码
- 愿景与里程碑：[`target_platform_implementation_roadmap.md`](target_platform_implementation_roadmap.md)
- 文件级实现骨架：[`target_platform_implementation_guide.md`](target_platform_implementation_guide.md)
- 项目整体状态：[`current_project_status_and_roadmap.md`](current_project_status_and_roadmap.md)

> 约定：服务器启动/重启/训练/验证都由用户手动执行；本文件只给代码方向与命令。
>
> **进度状态（最近更新 2026-06-12）**：第一优先级两项必修（§1.1 并发锁、§1.2 方向约定）+ 第二优先级核心（§2.1 服务端 track、§2.2 多目标 ID 一致）+ 步骤 4 重关联/建议 + 第三优先级端到端入口（AI 面板框选 → 远程 track → TargetDock 刷新）+ 3D mask 按真实 index 重建 + mask 回写接口/训练导出格式 + 基础工程健壮性（job 持久化、schema 兼容、切片缓存 LRU、SAM3 请求校验、metadata 轻量检测）**已实现并通过本机轻量测试**。详见下方「已实现快照」。

---

## 0. 当前完成度总览

| 方向 | 主题 | 完成度 | 主要缺口 |
|---|---|---|---|
| 1 | SAM3 辅助标注 / 训练数据 | ~75% | 回写接口已通；缺本机笔刷 UI 接入 |
| 2 | 目标编号一致性 | **~80%** | 服务端多目标 ID + 工作台储层 cell 目标入口已通；地震体工作台旧本地路径仍需逐步收敛 |
| 3 | 视频追踪 + 实例管理 | **~65%** | 服务端 track、重关联/建议、储层工作台入口已通；真多卡与逐帧复核仍待做 |
| 4 | 目标对象化 | ~55% | 缺逐帧属性曲线、edits 记录 |
| 5 | 语言 / 语义提取 | **~45%** | track 体级串联已通（文本种子）；缺语义层 |
| 6 | 框选改放大 | ~90% | 基本完成 |
| 7 | 模型训练 / 微调 | ~55% | 可配置训练命令已通；真实 SAM3 训练脚本/评估策略待接 |
| 8 | 二三维展示 | ~65% | 2D 基础编号/轮廓/类别色、mask3d 与储层 selection 3D 接线已通；缺轨迹/面积曲线 |
| 9 | 多卡调度 | ~20% | 非真并行、共享单引擎 |
| 10 | 闭环扩展 | ~5% | 后期 |

**跨领域隐患**：并发写 `targets.json` ✅已修(§1.1)、方向 transpose 翻转 ✅已修(§1.2)、3D 按顺序堆叠 ✅已修(§5.2)、job 仅内存 ✅已修、缓存无清理 ✅已修、schema 迁移起点 ✅已建、输入无上限 ✅已加基础校验。储层 cell 目标落库基础接口与主窗口入口 ✅已建。仍待：真多卡、多维展示、真实训练后端、地震体工作台旧本地路径收敛。

---

## 已实现快照（截至 2026-06-11）

本节描述代码里**已经落地并测试**的内容，是后续所有开发的新起点。新增/改造按数据流列出。

### A. 并发安全的目标存储（§1.1）
- [`targets/store.py`](../local/app/src/yj_studio/targets/store.py)：进程级 per-project `RLock`（按 `project_root` 绝对路径注册）+ `TargetStore.mutate()` 上下文管理器，做**原子读改写**（加锁→load→改→save；body 抛异常则跳过保存）。
- [`server/.../app.py`](../server/src/yj_studio_server/app.py)：`PATCH/DELETE/merge/split` 四个写路由 + `_run_sam3_job`/`_run_track_job` 落库段**全部走 `store.mutate()`**；SAM3 推理留在锁外，只把 `targets.json` 读改写放进临界区；`new_id()` 在锁内分配，杜绝重号。
- **局限**：进程内锁，多进程 GPU worker（§4）需升级为文件锁或单写进程。
- 测试：`local/app/tests/test_targets_store.py::test_target_store_mutate_is_concurrency_safe`（24 线程并发无丢失、ID 唯一）、`_skips_save_on_error`。

### B. 唯一的 mask 方向约定（§1.2）
- **约定**：服务器侧 mask 一律以 **image 序**存储（行=samples/depth，列=trace，与喂 SAM3 的 RGB 同序）；本机只在进入 scene/view 层时转置**一次**。
- [`ai/adapters/mask_to_layer.py`](../local/app/src/yj_studio/ai/adapters/mask_to_layer.py)：唯一转换函数 `sam3_mask_to_layer()`。旧本地 SAM3 算法、`RemoteSAM3Task._build_layers`、`target_dock` 三处手写 `.T` 已全部收敛到它。
- 测试：`test_ai_adapters.py::test_sam3_mask_to_layer_is_a_single_canonical_transpose`、`test_targets_store.py::test_target_store_mask_roundtrip_preserves_orientation`。

### C. 服务端单剖面分割 → GeoTarget（M1，已有基础上接入 §1.1）
- [`sam3/engine.py`](../server/src/yj_studio_server/sam3/engine.py) `SAM3Engine.segment`；`app._run_sam3_job`：取片→`slice_to_rgb_image`→SAM3→多候选各成一个 `GeoTarget`（`source=sam3_interactive`），经 `mutate()` 落库。
- 本机 `RemoteSAM3Client` + `RemoteSAM3Task`（runner 按算法 id `ai.sam3.segment` 路由），UI 零改动。

### D. 服务端多目标追踪 + 跨帧 ID 一致（§2.1 / §2.2，本轮核心）
- [`sam3/engine.py`](../server/src/yj_studio_server/sam3/engine.py)：`init_track_state()` + `track_video()`（多对象 `add_prompt`，正反向 `propagate_in_video`，逐帧产出 `{obj_id: mask}`；autocast bf16 仅 cuda，CPU/无 torch 时 `nullcontext`）；`_extract_objects()` 按 `out_obj_ids` 拆每个对象。
- [`sam3/tracking.py`](../server/src/yj_studio_server/sam3/tracking.py)（**新建，不依赖 FastAPI，可单测**）：
  - `collect_object_frames()`：驱动引擎，帧本地索引→绝对体索引，按对象分桶，丢空 mask。
  - `persist_tracked_targets()`：**一对象=一 `GeoTarget`**（`source=sam3_video`，帧 `origin=propagated`），ID 在 `store.mutate()` 锁内集中分配。
- [`app.py`](../server/src/yj_studio_server/app.py) `_run_track_job`：取片→**PIL 渲染 JPEG 序列（无 Qt/matplotlib）**→调用核心。窗口按真实轴长 clamp。
- 路由：`POST /sam3/jobs {kind:"track"}` 与 `/sam3/extract {mode:"track",scope:"volume"}`。
- 种子两种：`prompts.boxes`（交互框）或 `prompts.text`（先在种子帧文本分割得到框再追踪 → 对接方向 5/6 的体级提取）。
- 测试：`server/tests/test_tracking.py`（3 项：多对象 T1/T2 一致、空 mask/越界帧丢弃、无帧对象跳过）。

### E. AI 面板端到端追踪入口（步骤 5）
- [`ai/remote_client.py`](../local/app/src/yj_studio/ai/remote_client.py)：新增 `RemoteSAM3Client.submit_track()`，POST `kind=track`，契约对齐服务器 `_parse_track_range`。
- [`algorithms/runner.py`](../local/app/src/yj_studio/algorithms/runner.py)：新增 `RemoteSAM3TrackTask`，负责提交、轮询、取消；完成时返回 job result，不创建本地图层。
- [`ui/docks/ai_dock.py`](../local/app/src/yj_studio/ui/docks/ai_dock.py)：新增目标类型、种子前/后帧数与「追踪」按钮；要求远程后端和至少一个框选提示。
- [`ui/main_window.py`](../local/app/src/yj_studio/ui/main_window.py)：`AIDock.track_finished` 自动触发 `TargetDock.refresh()`。
- 测试：`local/app/tests/test_remote_sam3_track.py` 覆盖 `submit_track` POST body 与 `RemoteSAM3TrackTask` running→done。

### F. 重关联、gap 标注与 merge/split 建议（步骤 4）
- [`sam3/reassociate.py`](../server/src/yj_studio_server/sam3/reassociate.py)：新增 `mask_iou`、`centroid`、`annotate_gaps`、`link_targets_by_iou`、`detect_merge_split`，纯 numpy/可选 scipy，不依赖 FastAPI。
- [`sam3/tracking.py`](../server/src/yj_studio_server/sam3/tracking.py)：`persist_tracked_targets()` 支持 `link_resolver`，可把新 track 对象按 IoU 合并进已有 `GeoTarget`；同时写入 `metadata["tracking"]["last_gap"]` 并按 trailing gap 标记 `lost/active`。
- [`app.py`](../server/src/yj_studio_server/app.py)：`_run_track_job` 现在返回 `gaps` 与 `suggestions`，并在落库前尝试跨 job IoU 重关联。
- [`target_dock.py`](../local/app/src/yj_studio/ui/docks/target_dock.py)：追踪结果若带 merge/split 建议，Dock 顶部显示提示，可确认或忽略。
- 测试：`server/tests/test_reassociate.py`、`server/tests/test_tracking.py::test_persist_can_link_track_to_existing_target_and_mark_gap`。

### G. 3D mask 按真实 index 重建（§5.2）
- [`targets/store.py`](../local/app/src/yj_studio/targets/store.py)：新增 `write_target_mask3d_cache(target)`，按 `[index_lo..index_hi]` 创建子体，缺帧留空，不再按 trajectory 顺序压缩堆叠。
- [`app.py`](../server/src/yj_studio_server/app.py)：`GET /sam3/targets/{id}/mask3d` 改用真实 index 重建，并返回 `X-Mask3D-Index-Lo/Hi` header。
- 测试：`local/app/tests/test_targets_store.py::test_target_store_mask3d_uses_real_frame_indices`。

### H. Mask 回写与训练导出格式（§6.1 / §6.2 部分）
- [`targets/model.py`](../local/app/src/yj_studio/targets/model.py)：`GeoTarget` 新增 `edits` 列表。
- [`app.py`](../server/src/yj_studio_server/app.py)：新增 `PUT /sam3/targets/{id}/mask/{axis}/{index}`，接收 `.npy` mask，覆盖该帧 mask 文件，`TargetFrame.origin="edited"`，并追加 edit 记录。
- [`remote_target_store.py`](../local/app/src/yj_studio/data/remote_target_store.py)：新增 `put_mask()`，本机可把修正后的 `.npy` mask 回写服务器。
- [`targets/export.py`](../local/app/src/yj_studio/targets/export.py)：COCO/PNG 导出新增 `schema_version` 与 `train/val/test` split；confirmed 或 edited 目标均可导出。
- 测试：`local/app/tests/test_remote_target_store.py`、`test_export_includes_edited_targets`。

### I. 基础工程健壮性（§7）
- [`sam3/jobs.py`](../server/src/yj_studio_server/sam3/jobs.py)：`JobStore(persist_dir=...)` 会把 `done/error/cancelled` 终态 job 写入 `runtime/server/jobs/<id>.json`，启动或 `get()` miss 时可回读。
- [`cache.py`](../server/src/yj_studio_server/cache.py)：新增切片缓存 LRU 规划/清理；`/slice` 写入缓存后按 `slice_cache_max_gb`（当前默认 100GB）删除最旧 `.npy`。
- [`targets/model.py`](../shared/src/yj_studio_core/targets/model.py)：`TargetSet.schema_version=1`，`TargetFrame/GeoTarget/TargetSet` 均允许忽略未知字段，便于旧项目/未来字段兼容。
- [`sam3/validation.py`](../server/src/yj_studio_server/sam3/validation.py)：`/sam3/jobs`、`/sam3/jobs/batch`、`/sam3/extract` 入队前校验 `keep_top_k`、`confidence`、boxes/points 数量、track/batch 帧数上限。
- [`targets/store.py`](../shared/src/yj_studio_core/targets/store.py)：`metadata_is_lightweight()` 改为检测大内联数组，不再被 `mask_ref`/`cell_ids_ref` 字符串误伤。
- 测试：`server/tests/test_jobs_persistence.py`、`server/tests/test_cache_budget.py`、`server/tests/test_sam3_validation.py`、`local/app/tests/test_targets_store.py` 新增覆盖；本机用直接单元调用通过，未启动服务器。

### J. 储层 cell 选择 → GeoTarget + 主窗口入口（已由规整删除）
- 2026-06-17 规整后，储层 grid/SAM3Workbench 路径和「打开储层剖面」入口已删除；后续不再通过储层 ROI 或 `ReservoirSelectionLayer` 创建 SAM3 目标。
- 保留井/层位/断层/储层模型浏览能力；SAM3 分割/追踪只走普通 2D 剖面 AI 面板与服务器 `/sam3/jobs`。
- 目标的 cell/mask3d 数据仍通过服务器目标 API 和 `RemoteTargetStore` 消费，不恢复旧 workbench 入口。

### K. 2D 目标基础叠加（§5.1 部分）
- [`targets/style.py`](../shared/src/yj_studio_core/targets/style.py)：新增目标类型配色 `trap/turbidite/fault/sandbody/unknown` 与 `mask_summary()`。
- [`ai/adapters/mask_to_layer.py`](../local/app/src/yj_studio/ai/adapters/mask_to_layer.py)：`build_mask_layer()` 若 metadata 带 `target_type`，自动应用类别色；同时补 `area_px/bbox/centroid` 摘要。
- [`view_2d_section.py`](../local/app/src/yj_studio/view/view_2d_section.py)：带 `target_id` 的 `MaskLayer` 在 2D 剖面上绘制 `contour(level=0.5)` 边界线与 `Tn` 编号。
- 测试：`test_build_mask_layer_uses_target_style_and_summary`。
- **剩余**：TargetDock 里的轨迹/面积曲线子面板尚未实现；3D 目标体类别色也可复用同一套 style。

### L. 可配置训练后端与模型激活 reload（§6.2 部分）
- [`sam3/training.py`](../server/src/yj_studio_server/sam3/training.py)：新增 `run_training_backend()`，支持 `training.command` 或请求 payload 的 `training_command/command`；命令可用 `{dataset_dir}`、`{output_dir}` 占位符，并接收 `YJ_DATASET_DIR`、`YJ_TRAIN_OUTPUT_DIR` 环境变量。
- [`app.py`](../server/src/yj_studio_server/app.py)：`_run_train_job` 现在先导出 COCO/PNG 数据集；若配置训练命令，则运行训练脚本、读取 `metrics.json` 和 checkpoint，登记为模型版本；未配置时仍保持“只导出数据集”的轻量行为。
- [`sam3/engine.py`](../server/src/yj_studio_server/sam3/engine.py)：新增 `reload_checkpoint()`；`POST /sam3/models/{id}/activate` 若模型有 checkpoint，会重置服务端 SAM3 引擎，下次推理懒加载新权重。
- [`server/config/server.example.yaml`](../server/config/server.example.yaml)：补充 `training.output_subdir` 与注释掉的 `training.command` 示例。
- 测试：`server/tests/test_training_backend.py` 用 fake 训练脚本验证 metrics/checkpoint 采集。
- **剩余**：真实 SAM3 微调脚本、评估指标计算、失败样本回流策略仍需在服务器环境实现/验证。

### M. 审校队列、全体推理入口与报告导出收口（G2/G3/G6）
- [`targets/active_learning.py`](../local/app/src/yj_studio/targets/active_learning.py)：`review_queue()` / `target_uncertainty()` 已作为审校排序核心。
- [`ui/docks/target_dock.py`](../local/app/src/yj_studio/ui/docks/target_dock.py)：新增「审校」对话框，按不确定度列出待审目标，支持批量确认/打回；提取模式新增 `infer_volume`；模型对话框仍保留激活/回滚。
- [`server/app.py`](../server/src/yj_studio_server/app.py)：`POST /sam3/jobs` 支持 `kind="infer_volume"`，复用 batch 推理并将结果目标默认写为 `status=to_review`，供审校队列消费。
- [`report/trap_report.py`](../local/app/src/yj_studio/report/trap_report.py)：圈闭报告新增 PDF 摘要表导出；CSV/XLSX 仍保存完整字段。
- 测试：`test_target_dock_review.py`、`test_sam3_validation.py::test_validate_infer_volume_uses_batch_frame_limits`、`test_trap_report.py::test_write_trap_report_pdf`。FastAPI API 级 `infer_volume` 测试需在服务器环境运行（本机 py312 未装 fastapi）。

### 本机测试运行方式
```text
cd local/app ; E:\miniconda\envs\py312\python.exe -m pytest tests/test_targets_store.py tests/test_ai_adapters.py -q
cd <repo>         ; E:\miniconda\envs\py312\python.exe -m pytest server/tests/test_tracking.py -q
```
> 注意：本机 py312 环境**未装 fastapi**，故 `_run_track_job` 整条 HTTP 链路只能在服务器验证；FastAPI-free 的 `sam3/tracking.py` 核心已在本机覆盖。已知无关失败：`test_decode_sam3_masks_from_numpy_state`（float32 `0.87` 精度，既有 bug）、两个 `test_imports` layer-tree 用例（用户在改 `layer_tree_dock.py`）。

---

## 第一优先级 · 必修项 ✅ 已完成（2026-06-11）

> 两项会「静默毁数据」的隐患已修复，是后续多卡与大规模追踪的前提。实现细节见上方「已实现快照 A / B」。

- **§1.1 并发写锁 + ID 集中分配** ✅ — `TargetStore.mutate()` 原子读改写 + per-project 锁；服务端所有写路径已改走它；ID 锁内分配。并发测试通过。**遗留**：进程内锁，多进程 worker 需升级（见 §4）。
- **§1.2 方向约定统一 + golden 测试** ✅ — 唯一 `sam3_mask_to_layer()` 收敛三处转置；约定「服务器存 image 序、本机只转一次」；golden 往返测试通过。

---

## 第二优先级 · 平台核心：服务端多目标追踪 + ID 一致（方向 2/3）

### 2.1 / 2.2 服务端 track job + 多目标编号一致 ✅ 已完成（2026-06-11）

> 实现细节见「已实现快照 D」。要点回顾：
> - 引擎 `track_video()` 多对象传播；`sam3/tracking.py` 的 `collect_object_frames` + `persist_tracked_targets` 为 FastAPI-free 核心；`app._run_track_job` 负责取片+PIL 渲染 JPEG（无 Qt/matplotlib）。
> - **一对象=一 GeoTarget**，obj_id↔target_id 在种子阶段固定、传播全程不变；ID 锁内分配跨帧/并发唯一。→ 落地方向 2「T1 到后续帧仍是 T1」。
> - 仅做**地震体轴向切片**：储层角点剖面渲染依赖 matplotlib，不能上服务器，单独立项。
> - 种子支持 `prompts.boxes`（交互）与 `prompts.text`（文本种子帧→框→追踪，对接体级提取）。
> - 测试 `server/tests/test_tracking.py`（3 项）通过。

**仍未做（本节剩余）**：§2.3 丢失重关联、§2.4 合并/分裂自动检测。下面给精细路线——两者都**直接挂在已有的 `sam3/tracking.py` 核心上**，无需改引擎。

---

### 2.3 丢失重关联（精细路线）

**问题**：SAM3 video 在某帧丢失某对象（遮挡/低分），其 `obj_id` 不再出现在 `out_obj_ids`；该 obj 后续帧没有 mask。当前 `collect_object_frames` 只是「该对象该帧没数据」，不会尝试找回。

**挂载点**：已有 [`sam3/tracking.py`](../server/src/yj_studio_server/sam3/tracking.py) `collect_object_frames` 的循环里，已拿到逐帧 `{obj_id: mask}` 和上一帧每对象 mask——重关联就在这里做，**不碰引擎**。

**实现路径**
1. 新增 `sam3/reassociate.py`：
   ```python
   def match_by_iou(prev: dict[int, np.ndarray],   # obj_id -> 上一帧 mask
                    curr: dict[int, np.ndarray],    # obj_id -> 本帧 mask
                    *, iou_thresh=0.3) -> dict[int, int]:
       """返回 本帧obj_id -> 复用的历史obj_id（贪心，IoU>thresh 视为同一目标）。"""
   ```
2. 在 `collect_object_frames` 内维护 `last_seen[obj_id] = (frame_local, mask)`。每帧：
   - 对本帧 `out_obj_ids` 里**新出现的** obj_id，与 `lost`（最近 N 帧消失）的历史对象做 `match_by_iou` + 质心距离；命中则把本帧 mask **并入历史 obj_id 的桶**（而非新桶），ID 得以续接。
   - 连续 `gap_limit` 帧未出现的对象，结果里标注 `lost`（写进 `GeoTarget.metadata["lost_at"]`，由 `persist_tracked_targets` 透传）。
3. `persist_tracked_targets` 不变——它已按 obj_id 分桶落库，重关联只是让桶在收集阶段就合并好。

**测试**（`server/tests/test_tracking.py` 扩展）：构造「对象 A 在中间几帧空缺再出现」的假引擎序列，断言重现帧并入原 obj 桶 → 最终仍是同一 `GeoTarget`，而非两个。

**DoD**：隔帧遮挡后同一物理目标保持单一 ID。

---

### 2.4 合并 / 分裂的自动检测建议（在人工按钮之上）

**问题**：merge/split 接口与 dock 按钮已存在，但全靠人工判断；系统不会**发现**「两对象 mask 某帧起持续重叠（建议合并）」或「一对象 mask 裂成多连通域（建议分裂）」。

**挂载点**：同样在 `collect_object_frames` 收集阶段（已有逐帧多对象 mask），产出一份 `suggestions` 随 job result 返回。

**实现路径**（轻量，只检测+建议，不自动改）
- 合并：每帧算两两对象 IoU，连续 `k` 帧 > 阈值 → `{"type":"merge","ids":[oid_a,oid_b],"frames":[...]}`。
- 分裂：单对象 mask 的连通域数（`scipy.ndimage.label`）连续 `k` 帧 ≥2 → `{"type":"split","id":oid,"frames":[...]}`。
- `_run_track_job` 把 `suggestions`（用 target_id 重映射后）写入 job result；`TargetDock` 读后用黄条提示，用户点确认才调用既有 `/merge`、`/{id}/split`。

**DoD**：dock 显示合并/分裂建议，确认后走既有接口生效。

---

## 第三优先级 · 两条 SAM3 路径统一（#5，闭合 P6）

**问题**：分割路径产 `GeoTarget`(mask)，**工作台追踪仍走本地单目标旧路径**（`_propagate_with_video_predictor` / `_propagate_along_axis`，本机 GPU、`obj_id=1`），产 `ReservoirSelectionLayer`(cell-IJK)，不进目标库。§2.1 已把多目标追踪做进服务器，现在要让工作台用它。

**实现路径**（现在大半是「改调用」而非「新实现」）
- **地震体工作台追踪**：直接复用 §2.1。工作台框选 → 调 `RemoteSAM3Client` 提交 `POST /sam3/jobs {kind:track, prompts.boxes}` → 轮询 → 刷新目标库。删除本地 `propagate_in_video` 路径（或保留为 `mode=local` 离线备选）。
- **储层角点剖面追踪**（暂留本地，渲染依赖 matplotlib）：把结果从「直接 emit `ReservoirSelectionLayer`」改为「reverse-lookup 的 cell 写入 `TargetFrame.cell_ids_ref`，整体写成 `GeoTarget`（经 `RemoteTargetStore`）」。
- `ReservoirSelectionLayer` 退化为「从某个 `GeoTarget` 渲染 cell」的视图层（`target_dock._load_selected_cells` 已是此形态，复用）。
- 同一个 `GeoTarget`：有 `mask_ref` 走 mask 渲染、有 `cell_ids_ref` 走 cell 渲染。

**DoD**：工作台所有追踪结果都落进目标库、可被 dock 管理；scene 里不再有「脱离目标库的孤立 selection」；地震体追踪在服务器跑（本机 GPU=0）。

---

## 第四优先级 · 真多卡调度（方向 9，前置=1.1 已修）

**问题**：[`jobs.py` `JobQueue`](../server/src/yj_studio_server/sam3/jobs.py) 是共享**同一个** `app.state.sam3` 引擎的 `ThreadPoolExecutor`；batch 串行。四卡跑不起来。

**实现路径**
1. 每卡一个 worker **进程**（不是线程），各自 `os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)`，进程内建独立 `SAM3Engine`。
2. 用 `multiprocessing` 队列或轻量任务中间件（初期 `concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids), initializer=_bind_gpu)`）。
3. `_run_sam3_batch_job` 把帧 round-robin 提交到进程池，而非串行 `_run_sam3_job`。
4. 目标写入：因为是多进程，**§1.1 的进程内锁失效**，必须升级为**文件锁**或把 `targets.json` 写入收敛到主进程（worker 只返回 mask + 元数据，主进程持锁统一落库）。**推荐后者**——而且现成：`sam3/tracking.py` 的 `persist_tracked_targets(store, collected, ...)` 已经把「计算结果（collected）」与「落库（持锁）」分离，worker 只需回传 `collected`，主进程调 `persist_tracked_targets` 统一写。`collect_object_frames` 同理可在 worker 端跑。
5. `/sam3/gpus` 返回真实每卡负载（worker 心跳 + `torch.cuda.mem_get_info`）。

> 架构红利：§2.1 把核心拆成 FastAPI-free 的 `tracking.py`，正好让 worker 进程能直接 import 并调用，不必拖入 FastAPI/HTTP。多卡改造主要是「换执行器 + 主进程统一落库」，核心逻辑不动。

**数据流**
```text
batch(N帧) → 主进程拆帧 → ProcessPool(GPU0..3) 各自分割 → 返回 mask+meta
            → 主进程持锁统一写 GeoTarget → 汇总
```

**测试**：N=100 帧批量，断言 4 进程均被占用、总时长≈单卡 1/4、目标数=输入产出数、无 ID 冲突。

**DoD**：四卡并行、提速≈4×、目标库无冲突。

---

## 第五优先级 · 展示与 3D 重建（方向 8，#4）

### 5.1 二维叠加编号/轮廓/类别色
**现状**：dock 只把 mask 作为 `MaskLayer` 加载，剖面上无 ID 标签、无轮廓线、无类别配色。
**实现路径**：扩展 mask 渲染（`view/renderers/mask_renderer.py` 或 2D section overlay），按 `GeoTarget.type` 取配色表、画轮廓（`skimage.measure.find_contours` 或 marching squares）、在质心标 `T{n}`。轨迹/面积曲线用 `TargetFrame.area_px/centroid` 在 dock 子面板画 matplotlib 折线。

### 5.2 3D mask 体图层规范化 + 体积统计（修 #4）
**问题**：[`store.py` `write_mask3d_cache`](../shared/src/yj_studio_core/targets/store.py) 若只按 trajectory 顺序堆叠，会**忽略真实 index 间隔**；跳帧/丢帧重关联后深度方向错位。规整计划第 5 步还要求输出标准 `(D,H,W)` mask 体图层，并按采样间隔计算体积。
**实现路径**：按帧真实 `index` 放进 `[index_lo..index_hi]` 的子体：
```python
depth = index_hi - index_lo + 1
vol = np.zeros((depth, H, W), np.uint8)
for key, frame in target.frames.items():
    vol[frame.index - index_lo] = read_mask(frame.mask_ref)   # 缺帧保持 0 或后续插值
```
**DoD**：非连续帧的 3D 体在深度方向位置正确。
**新增 DoD**：体积 = 有效体素数 × dx·dy·dz；采样间隔从配置读取，岩性体 `numpy_3x` 按相对地震体 3x 上采样后的有效间隔修正。

---

## 第六优先级 · 标注回流与真实训练（方向 1/7）

### 6.1 修正 mask 回写（方向 1 闭环关键）
**问题**：人工只能改名/类型/状态，**改不了 mask 本身**；标注资产无法迭代。
**实现路径**：新增 `PUT /sam3/targets/{id}/mask/{axis}/{index}`（body=.npy 字节流），覆盖该帧 mask、`origin="edited"`、追加 `GeoTarget.edits` 记录。本机笔刷/橡皮修过后调用上传。

### 6.2 真实训练后端（P2，可后做但格式先定死）
**现状**：[`_run_train_job`](../server/src/yj_studio_server/app.py) 只导 COCO + 登记模型行，自注「attach a real training backend here」。
**实现路径**：
- 现在就把导出格式定死，含 **train/val/test 划分字段** + `schema_version`，避免将来重导历史标注。
- 训练脚本服务端独立运行（用户手动触发），产出 checkpoint → `ModelRegistry.add_model(metrics=IoU/Dice/P/R)` → `activate` 热切换给 `SAM3Engine`（改 checkpoint 重载）。
- 失败样本回流：追踪失败/低分帧打标 → 进下一轮训练集。

**DoD**：confirmed+edited 目标可导出带划分的训练集；模型版本可评估、可激活、可回滚。

---

## 第七优先级 · 工程健壮性 ✅ 基础项已完成

| 项 | 状态 | 实现 |
|---|---|---|
| Job 持久化(#7) | ✅ | 终态 job 落 `runtime/server/jobs/<id>.json`，`JobStore` 启动/get miss 时回读 |
| Schema 版本(#6) | ✅ | `TargetSet.schema_version=1`；Pydantic `extra="ignore"`，缺字段默认 |
| 切片缓存 LRU | ✅ | 写缓存后按 `slice_cache_max_gb` 对 `runtime/server/cache/slices/*.npy` 做 mtime/LRU 清理 |
| 输入校验(#8) | ✅ | 入队前校验 boxes/points 数量、track/batch 帧数、`keep_top_k`、`confidence` |
| 小 bug(#10) | ✅ | `metadata_is_lightweight` 改为检测大内联数组，不再用 `"mask"` 子串 |
| 服务端测试(#9) | 部分 ✅ | 已补 FastAPI-free job/cache/validation/reassociate/tracking 测试；完整 TestClient 契约仍建议在服务器环境补跑 |

---

## 实施顺序（强烈建议按此）

```text
1. [必修] 1.1 并发写锁 + ID 集中分配        ✅ 已完成
2. [必修] 1.2 方向约定统一 + golden 测试     ✅ 已完成
3. [核心] 2.1/2.2 服务端 track(地震体)+多目标编号  ✅ 已完成
———————————————— 以下为下一步（建议顺序）————————————————
4. [核心] 2.3 丢失重关联 ; 2.4 合并/分裂建议   ✅ 已完成
5. [统一] 第三优先级 AI 面板端到端：本机框选→服务器 track→目标库刷新  ✅ 已完成
6. [扩展] 第四优先级 真多卡（worker 回传 collected，主进程 persist_tracked_targets 统一落库）
7. [展示] 5.1 2D 编号/轮廓/类别叠加 ; 5.2 3D 按 index 重建（5.2 ✅）
8. [闭环] 6.1 mask 回写（✅接口/客户端） → 6.2 真训练后端（导出格式 ✅）
9. [健壮] 第七优先级 基础项 ✅；后续随真多卡/训练继续补集成测试
```

**当前态判断**：会静默毁数据的两个隐患（并发写、方向翻转）已修；平台主干（服务端多目标追踪 + 跨帧 ID 一致）已立起来并测过；AI 面板追踪、储层工作台入口、审校队列、`infer_volume → to_review`、报告 CSV/XLSX/PDF 都已完成初版闭环。下一步重心转向真多卡、真实 SAM3 微调/评估、本机笔刷 mask 编辑，以及地震体工作台旧本地路径的逐步收敛。

---

## 附：完成度 checklist

- [x] 1.1 并发写锁 + ID 集中分配（+并发测试） —— 已实现 2026-06-11
      `TargetStore.mutate()` 进程级 per-project 锁 + 原子读改写；服务器 patch/delete/merge/split 与 `_run_sam3_job` 全部改走 `mutate()`；
      测试 `test_target_store_mutate_is_concurrency_safe` / `_skips_save_on_error` 通过。
- [x] 1.2 方向约定 + golden 往返测试 + 删多余转置 —— 已实现 2026-06-11
      唯一转换函数 `ai/adapters/mask_to_layer.sam3_mask_to_layer`；旧本地 SAM3 算法 / `RemoteSAM3Task._build_layers` / `target_dock` 三处手写 `.T` 已收敛到它；
      测试 `test_sam3_mask_to_layer_is_a_single_canonical_transpose` / `test_target_store_mask_roundtrip_preserves_orientation` 通过。
- [x] 2.1 服务端 track job（地震体轴向切片） —— 已实现 2026-06-11
      `SAM3Engine.track_video`/`init_track_state`（多对象 add_prompt + propagate 正反向，CPU/无 torch 时降级为 nullcontext）；
      `sam3/tracking.py` 的 `collect_object_frames` + `persist_tracked_targets`（不依赖 FastAPI，可单测）；
      `app._run_track_job` 负责取片→PIL 渲染 JPEG（无 Qt/matplotlib）→调用核心；`POST /sam3/jobs kind=track` 与 `/sam3/extract mode=track scope=volume` 已路由到它。
      支持两种种子：prompts.boxes（交互框）或 prompts.text（先文本分割种子帧得到框再追踪）。
      测试 `server/tests/test_tracking.py`（3 项）通过。
- [x] 2.2 obj_id↔target_id 固定映射（多目标编号一致） —— 已实现 2026-06-11
      每个种子对象分配一个 obj_id，传播全程不变；落库时一对象=一 GeoTarget（source=sam3_video），帧 origin=propagated；
      ID 在 `store.mutate()` 锁内集中分配，跨帧/并发均唯一。测试断言 2 对象→T1/T2、各帧齐全、ID 不串。
      待办：地震体工作台旧本地路径仍需逐步收敛到服务器 track。
- [x] 2.3 丢失重关联（IoU/gap） —— 已实现 2026-06-12
      `sam3/reassociate.py` + `persist_tracked_targets(link_resolver=...)`；跨 job 可按重叠帧 IoU 合并到已有 GeoTarget；gap 写入 target metadata。
- [x] 2.4 合并/分裂自动检测建议 —— 已实现 2026-06-12
      `detect_merge_split()` 产出 suggestions；`_run_track_job` 映射为 target_id；TargetDock 顶部提示并可确认/忽略。
- [x] 3.x AI 面板端到端追踪：框选→服务器 track→目标库刷新 —— 已实现 2026-06-11
      `RemoteSAM3Client.submit_track()` + `RemoteSAM3TrackTask` + `AIDock` 追踪按钮 + `TargetDock.refresh()` 自动刷新；
      测试 `test_remote_sam3_track.py` 通过。
- [x] 3.x 储层工作台追踪产出 GeoTarget，两路径统一（储层路径） —— 已实现 2026-06-12
      2026-06-17 规整后，储层 grid/SAM3Workbench hook 已删除，不再恢复；
      主窗口「打开储层剖面」入口可从已注册 ReservoirGridLayer 打开工作台，selection 加入本地图层，target 提交后刷新 TargetDock。
- [ ] 4.x 真多卡（每卡独立进程 + 主进程统一落库）
- [ ] 5.1 2D 编号/轮廓/类别叠加 + 轨迹/面积曲线
      基础 2D 叠加已完成 2026-06-12：带 `target_id` 的 MaskLayer 会显示类别色、边界线、Tn 编号。
      仍待：TargetDock 轨迹/面积曲线子面板。
- [x] 5.2 3D 体按真实 index 重建 —— 已实现 2026-06-12
      `write_target_mask3d_cache()` 按 `[index_lo..index_hi]` 放置 mask，缺帧留 0；`/mask3d` header 返回 index 范围。
- [x] 6.1 修正 mask 回写接口 —— 已实现 2026-06-12
      服务器 `PUT /sam3/targets/{id}/mask/{axis}/{index}` + 客户端 `RemoteTargetStore.put_mask()`；edit 记录写入 `GeoTarget.edits`。
- [ ] 6.2 真实训练后端 + 评估 + 激活/回滚
      可配置训练命令已完成 2026-06-12：导出后可运行 `training.command`，采集 `metrics.json`/checkpoint 并登记模型；activate 会 reload checkpoint。
      仍待：真实 SAM3 微调脚本、评估指标、失败样本回流与回滚 UI。
- [x] 7.x job 持久化 / schema 版本 / 缓存 LRU / 输入校验 / 小 bug / 基础服务端测试 —— 已实现 2026-06-12
      `JobStore(persist_dir)` 终态落盘；`TargetSet.schema_version=1` + 未知字段忽略；`/slice` 写缓存后按 100GB 配置 LRU 清理；
      SAM3 请求入队前校验；`metadata_is_lightweight` 改为检测大内联数组。
      本机直接单元调用通过：`test_jobs_persistence.py`、`test_cache_budget.py`、`test_sam3_validation.py`、`test_targets_store.py`。
```
