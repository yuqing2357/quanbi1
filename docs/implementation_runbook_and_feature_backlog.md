# 施工单 · 接下来就做什么、怎么做

本文件 = **你下一步要动手的事**。当前聚焦第一件事：实现 **A1 闭合等值线**（圈闭算法链的地基），给出「打开就能照做」的逐步 runbook（含你代码里**真实的集成点**与坑）。

- 现状（已实现 vs 缺失）看：[`project_review_and_remediation.md`](project_review_and_remediation.md)
- 仍需实现的**完整目录**（算法链 A1–A5 + 训练闭环 + 平台功能 G1–G8 + 里程碑）看：[`next_steps_geophysics_and_training.md`](next_steps_geophysics_and_training.md)

> 已用实际代码核实的关键事实（写代码前先记住）：
> - 算法在 [`AlgorithmRunner`](../local/app/src/yj_studio/algorithms/runner.py) 里跑。`runs_in_subprocess = False` 的算法走 `InProcessAlgorithmTask`（**QThread，不卡 UI**）并能拿到 `ctx.services`；`= True` 的走子进程、**拿不到 services** 且输入输出要能 `to_dict/from_dict` 序列化。
> - 已注册的服务（[main_window.py:94](../local/app/src/yj_studio/ui/main_window.py:94)）：`ctx.services["volume_store"]`、`ctx.services["ai_service"]`。取整卷用 `volume_store.get_volume(volume_id)`（memmap），取切片用 `get_slice(volume_id, axis, index)`。
> - `AlgorithmDock` 默认 `auto_attach_outputs=False`：它监听 `task.finished(layers, summary)`，再通过 undo command 把 `output_layers` 加进 `LayerStore`。算法只管返回，不要自己塞 store。
> - **`TrapLayer` 目前没有任何渲染器**（2D/3D 都没有），算法跑完结果不可见——A1 必须连带补渲染（见步骤 4）。

---

# A1「闭合等值线」逐步实现 Runbook

目标产物：在层位上自动圈出四面闭合构造高，输出可在平面/3D 看见的 `TrapLayer`，带 relief/area/score，纯 CPU 可离线单测。`run()` 的更长代码骨架在 [`next_steps_geophysics_and_training.md`](next_steps_geophysics_and_training.md) 的 A1 段；**本文下面这节是权威的数据契约**（含骨架里缺的去重等修正），照本文为准。

**当前状态（2026-06-12）**：A1 已按本施工单完成代码与本地单元测试；真实层位 UI 手测仍待用户执行。下一步按步骤 8 的模板推进 A2。

## 关键方法与数据契约（输入 / 输出）

> 实现 A1 = 写 3 个函数（1 个纯函数 + 1 个辅助 + 1 个算法 `run`）+ 接 1 处渲染。下面给每个的签名、输入、输出、数据形状。

### ① Params（算法输入参数）
`ClosureContourParams(BaseModel)`，字段即算法面板上可调项：

| 字段 | 类型 | 默认 | 单位/含义 |
|---|---|---|---|
| `z_step_m` | float>0 | 1.0 | 每采样对应米数，仅用于把 relief 报告成米 |
| `level_step` | float>0 | 1.0 | 水位抬升步长（采样单位），越小越精细越慢 |
| `min_relief_samples` | float≥0 | 2.0 | 最小闭合高度（采样），低于此丢弃 |
| `min_area_cells` | int≥1 | 8 | 最小闭合面积（网格点数） |
| `max_highs` | int≥1 | 50 | 候选高点数上限（防爆） |
| `shallower_is_smaller` | bool | True | 你的层位约定：True=采样越小越浅（默认）。若是高程（越大越高）置 False，内部比较方向取反 |

### ② `_structural_highs(z, valid) -> list[tuple[int,int]]`（辅助）
- **输入**：`z: np.ndarray (ni,nx) float32`（采样索引，nan 表无效）；`valid: np.ndarray (ni,nx) bool`。
- **输出**：构造高点的 `(inline, xline)` 列表。
- **关键步骤**：`minimum_filter(np.where(valid,z,inf), size=3)` 取局部极小 → 得 `is_min` 布尔图 → **对 `is_min` 连通域标注，每个连通块只保留一个代表点**（质心或最浅点），避免平台区产生大量并列极小把同一高点重复计算（这是骨架里缺的，必补）。

### ③ `detect_closures(z, valid, params) -> list[ClosureResult]`（核心纯函数，脱机可测）
- **输入**：同上的 `z` / `valid` + `params`。
- **输出**：`list[ClosureResult]`。`ClosureResult` 建议用 dataclass：

  | 字段 | 类型 / 形状 | 含义 |
  |---|---|---|
  | `high_ij` | `tuple[int,int]` | 高点 (inline,xline) |
  | `spill_level` | `float` | 溢出水位（采样），即最大闭合对应的等值线深度 |
  | `relief_samples` | `float` | 闭合高度 = `spill_level - z[high]` |
  | `area_cells` | `int` | 闭合内网格点数 |
  | `boundary_ij` | `np.ndarray (M,2) float` | 闭合多边形顶点 `(row=inline, col=xline)`，**map 平面坐标，不转置** |
  | `edge_limited` | `bool` | 闭合是否因触网格边而被裁断（保守提示） |

- **关键步骤**：①`_structural_highs` 取高点；②水位 `level` 从 `nanmin` 向 `nanmax` 每 `level_step` 抬升，`region = valid & (z<=level)`（`shallower_is_smaller=False` 时反号）；③对 `region` 连通域标注，某连通块**只含 1 个高点且不触边**→更新该高点的当前闭合；④一旦触边或含 ≥2 高点→该高点定格在上一刻闭合（标 `edge_limited`/已溢出）；⑤按 `min_relief_samples`/`min_area_cells` 过滤；⑥`skimage.measure.find_contours(comp,0.5)` 取 `max(key=len)` 环作 `boundary_ij`。

### ④ `ClosureContourAlgorithm.run(ctx) -> AlgorithmResult`（算法外壳）
- **输入**：`ctx.input_layers["horizon"]`（`HorizonLayer`，用 `.sample`(ni,nx) 与可选 `.mask`）；`ctx.params`（即 ①）。`layer_inputs = {"horizon":"horizon"}`，`runs_in_subprocess = False`。
- **处理**：`valid = isfinite(sample) & mask` → 调 `detect_closures` → 每个 `ClosureResult` 映射成一个 `TrapLayer`。
- **输出**：`AlgorithmResult.success(output_layers=[TrapLayer...], summary=...)`；无结果时 `.failure("没有满足阈值的闭合")`。每个 `TrapLayer`：
  - `boundary: np.ndarray (M,3)` = `[inline, xline, spill_level]`（把 `boundary_ij` 补上常数 Z 列）
  - `score: float∈[0,1]` = `min(1, relief_samples / (zmax-zmin))`
  - `attributes`: `{high_inline, high_xline, spill_level, relief_samples, relief_m, area_cells, edge_limited, source_horizon}`
  - `metadata`: `{algorithm: id, source_horizon_id}`

### ⑤ 渲染契约（**重要：闭合是平面多边形，不是剖面线**）
A1 的闭合多边形在 **map 平面（inline×xline）**上、Z=`spill_level` 常数。这决定了它在不同视图的形态，按此接渲染：
- **Z 切片 / 平面图视图**：画**完整闭合多边形**（`boundary[:, :2]`）+ 质心标 `score`。这是最自然、最该优先做的显示。
- **inline / xline 剖面**：多边形与该剖面只相交于少数点/短段——可不画或只画交点标记，别期望看到完整圈。
- **3D 场景**（[`scene_controller.py:71/100/135`](../local/app/src/yj_studio/view/scene_controller.py:71)）：把 `TrapLayer` 加进 `isinstance(...)` 元组，用 `boundary (M,3)` 画一条**水平闭合 polyline**（Z=spill_level），参照 PolygonLayer 的 3D 画法。
- 落点：2D 走 [`view_2d_section.py`](../local/app/src/yj_studio/view/view_2d_section.py) 的图层派发分支（仿 PolygonLayer），3D 走 scene_controller。配色用 `score`（低分灰→高分红）。

### ⑥ 测试契约（断言点）
直接测 `detect_closures`（不经 Qt/runner）：输入合成 `z` → 断言返回的 `ClosureResult` **数量**与 **字段值**（relief>0、boundary 首尾闭合、area≥阈值、触边样例 `edge_limited=True`）。用例清单见步骤 5。

---

## 步骤 0 · 准备（5 分钟）
1. 新建分支：`git checkout -b feat/closure-contour`。
2. 确认依赖（已实测可用，无需安装）：`numpy`、`scipy.ndimage`、`skimage.measure`。
3. 读一遍既有范例 [`builtin/thickness.py`](../local/app/src/yj_studio/algorithms/builtin/thickness.py)——A1 完全照它的结构（ClassVar 元信息 + `run(ctx)` + 返回 `AlgorithmResult.success`）。

## 步骤 1 · 写算法本体
新建 `local/app/src/yj_studio/algorithms/builtin/closure_contour.py`，代码骨架见 geophysics 文档 A1（已给完整 `run()`）。在此基础上补这些**易漏细节**：

- **`runs_in_subprocess`**：显式设 `runs_in_subprocess = False`（无需 services，但要 QThread 不卡 UI；TrapLayer 输出也省去序列化）。
- **nan 安全**：`z` 里 nan 要先 `np.where(valid, z, np.inf)` 再做 `minimum_filter`，否则极小点会落在 nan 边缘。`levels` 用 `np.nanmin/np.nanmax`。
- **局部极小去重/去噪**：`minimum_filter(size=3)` 会在平台区产生大量并列极小点。落地时对 `is_min` 做一次连通域标注，每个连通块只取一个代表点（质心或最浅点），避免同一高点被算很多次。
- **`find_contours` 坐标**：`skimage.measure.find_contours(comp, 0.5)` 返回 `(row, col)` 浮点序列，`row=inline`、`col=xline`，与 `HorizonLayer.sample` 同序——**不要转置**。把它和 `level`(Z) 拼成 `(M,3)` 存进 `TrapLayer.boundary`。
- **闭环判定**：marching squares 对内部连通块给出的环天然首尾相接；若取到多条，用 `max(contours, key=len)` 取外环即可。
- **进度与取消**：水位循环里每若干步 `ctx.report_progress(...)` 且 `ctx.check_cancel()`（长层位扫描要能取消）。
- **空结果**：无满足阈值的闭合时 `return AlgorithmResult.failure("没有满足阈值的闭合")`，不要返回空 success（UI 才会给出可读提示）。

## 步骤 2 · 注册到算法目录
1. 升级旧占位：把 [`builtin/stubs/closure_contour.py`](../local/app/src/yj_studio/algorithms/builtin/stubs/closure_contour.py) 删除（或清空其 `@register_algorithm`），避免**同 id 重复注册**。
2. 确认 `builtin/__init__.py` 会 import 新模块（注册靠 import 的 side-effect）。若它是显式列举 import，加上 `from . import closure_contour`；若是按目录自动发现，确认新文件在扫描路径内。
3. 跑一次 `python -c "from yj_studio.algorithms import registry; print('horizon.closure_contour' in registry...)"`（按 registry 实际 API 调整）确认已注册且无重复 id 报错。

## 步骤 3 · 让结果能落到图层
`AlgorithmDock` 已通用处理 `finished(layers, summary)` → undo command → `LayerStore.add`。A1 不用改 dock。**但**要确认 `payload_to_layer`（子进程路径用）认识 TrapLayer——因为 A1 走 in-process，这步可跳过；将来若改子进程再补 TrapLayer 的 `to_dict/from_dict`（已存在）注册到 `payload_to_layer` 映射。

## 步骤 4 · 补 TrapLayer 渲染（关键，否则看不见）
照 `PolygonLayer` 的现成接法加 TrapLayer 分支，**严格按上面 ⑤ 渲染契约**（闭合是 map 平面多边形、Z 常数，剖面里只见交点、平面/3D 才见完整圈）：

- **优先：Z 切片 / 平面视图**画完整闭合多边形（`boundary[:, :2]`）+ 质心标 `score`。这是最该先做、最直观的显示。
- **inline/xline 剖面**（[`view_2d_section.py:335`](../local/app/src/yj_studio/view/view_2d_section.py:335) 附近，仿 `PolygonLayer and layer.closed` 那段）：多边形与该剖面只相交于少数点/短段，可只画交点标记或先不画——**别期望在剖面上看到完整圈**（这是新手最易踩的"看不到结果"）。
- **3D 场景**（[`scene_controller.py:71/100/135`](../local/app/src/yj_studio/view/scene_controller.py:71)）：把 `TrapLayer` 加进那几处 `isinstance(layer, (...))` 元组，用 `boundary (M,3)` 画水平闭合 polyline（参照 PolygonLayer 的 3D 绘制）。
- **manual_geometry_renderer.py**：若 2D 走的是这个渲染器，把 TrapLayer 也纳入其联合类型。
- **配色**：用 `score` 映射颜色（低分灰、高分红），或复用 `targets/style.py` 的配色思路。

> 验收点：层位上跑 A1 → **平面图/3D** 能看到完整闭合圈、质心有 score；图层树里可勾选/隐藏。

## 步骤 5 · 单元测试（`local/app/tests/test_closure_contour.py`，纯 CPU、无 Qt）
把核心抽成纯函数 `detect_closures(z, valid, params) -> list[ClosureResult]` 后直接测它（不经 runner，最快）：
1. **单高斯洼地**：`z = r²`（中心最浅）→ 恰 1 个闭合，relief>0，boundary 首尾点距 < 1 格。
2. **双高被深鞍隔开**：两个高斯坑 + 高鞍 → 2 个闭合；把鞍部抬浅到连通 → 两高点都进入 `spilled`，闭合数减少。
3. **单调斜坡**：`z = x` → 无闭合 → `detect_closures` 返回空 / 算法 `.failure`。
4. **阈值过滤**：`min_relief_samples`/`min_area_cells` 调高 → 小闭合被剔除。
5. **nan 鲁棒**：在 `valid=False` 区填 nan，断言不崩、不把 nan 边当高点。

跑：`cd local/app; $env:PYTHONPATH="src"; E:\miniconda\envs\py312\python.exe -m pytest tests/test_closure_contour.py -q`。

## 步骤 6 · UI 手测路径
1. 本地模式启动 `python run_yj_studio.py`，「文件→打开体数据」加载有层位的工程，或加载一个 HorizonLayer。
2. 右侧「算法」标签 → 选「闭合等值线」→ 选层位输入 → 调 `min_relief_m` → 运行。
3. 在**平面图/Z 切片或 3D**看到闭合圈、summary 报告闭合数与最大闭合高度；图层树出现 `闭合@(i,j)` 条目。（剖面视图里只见交点是正常的，见步骤 4）

## 步骤 7 · 常见坑清单
- **inline/xline 顺序**：全程 `(row=inline, col=xline)`，与 `sample` 同序；只有渲染投影到具体剖面时才按 axis 取列。别在算法里转置。
- **Z 方向**：sample 越小越浅；闭合是「浅区被深区包围」，水位 `level` 从浅(`zmin`)往深(`zmax`)涨。若你的层位是「正深度（越大越深）」这套就对；若是「高程（越大越高）」要把比较方向反过来——做成 `Params.shallower_is_smaller: bool` 显式可配。
- **边界接触**：靠网格边的高点没法判定四面闭合（数据被裁断），按当前逻辑会被判 `spilled`——这是对的（保守），但要在 attributes 标 `edge_limited=True` 提示用户。
- **性能**：层位很大 + `level_step` 很小 → 水位循环次数爆炸。`level_step` 默认别太小（如 1~2 采样）；或先粗扫定位高点、再在每个高点局部细扫。
- **重复 id**：忘了删 stub 里的同名注册 → 启动即报重复 id。

## 步骤 8 · 推广到 A2–A5 的固定套路
每个算法都按同一七步走：①照 thickness.py 写 `run`；②`runs_in_subprocess=False`（要 services 的尤其如此）；③删同名 stub 注册；④确认输出层有渲染器（没有就补，TrapLayer 这次补完后 A2 直接复用）；⑤抽纯函数写脱机单测；⑥UI 手测；⑦填 attributes/metadata 供下游（A2 吃 A1、A5 吃 A2、A4 吃 A3）。

---

# A1 之后的立即顺序

A1 跑通后，按这个顺序继续（详细 Params/`run()`/测试/DoD 全在 [`next_steps_geophysics_and_training.md`](next_steps_geophysics_and_training.md)）：

```text
1. A1 闭合等值线        ← 代码/测试完成，真实 UI 手测待做
2. A2 圈闭检测          ← 代码/测试完成，真实 UI 手测待做
3. G5.1 3D 目标渲染     ← TrapLayer surface + GeoTarget mask3d marching-cubes 核心完成
4. G6.1 圈闭清单报告    ← CSV/XLSX 核心已合并 A2/A5 字段；PDF/缩略图待做
   —— 至此完成里程碑 M-α：圈出带置信度的圈闭 + 出报告 ——
5. A3 连通性            ← 代码/测试完成，真实体数据参数验收待做
6. A4 砂体提取          ← 代码/测试完成，真实孔隙度/岩性体验收待做
7. A5 圈闭评价          ← 代码/测试完成，测网面积/时深/孔隙度校核待做
并行: Phase B 训练闭环（G3.2 空间分块划分核心已完成；模型 parent 版本链已完成；模型管理 UI 已有激活/回滚；G2.1/G2.2 审校队列/主动学习核心已完成；微调脚本 + G3.4 全体批量推理 + 审校 Dock UI 待做）
   —— 完成里程碑 M-β：标注↔训练闭环跑通 ——
```

完整里程碑 M-α…M-ε 与平台功能全集见 [`next_steps_geophysics_and_training.md`](next_steps_geophysics_and_training.md) 末尾。

## 当前验证命令（2026-06-12）

```text
cd local/app
$env:PYTHONPATH="src"; $env:PYTHONDONTWRITEBYTECODE="1"
E:\miniconda\envs\py312\python.exe -m pytest tests/test_closure_contour.py tests/test_trap_detect.py tests/test_trap_report.py tests/test_connectivity.py tests/test_sandbody_extract.py tests/test_trap_evaluate.py tests/test_algorithms_registry.py tests/test_schema_form.py -q --basetemp pytest_tmp_phase_a3
E:\miniconda\envs\py312\python.exe -m pytest tests/test_mask_volume_renderer.py tests/test_trap_report.py -q
E:\miniconda\envs\py312\python.exe -m pytest tests/test_targets_store.py::test_split_frames_spatial_uses_contiguous_index_blocks tests/test_targets_store.py::test_export_confirmed_targets_uses_spatial_split -q
E:\miniconda\envs\py312\python.exe -m pytest tests/test_active_learning.py -q
E:\miniconda\envs\py312\python.exe -m pytest tests/test_target_dock_models.py -q
E:\miniconda\envs\py312\python.exe -m pytest tests/test_target_dock_review.py tests/test_reservoir_workbench_entry.py -q
cd <repo>
E:\miniconda\envs\py312\python.exe -m pytest server\tests\test_training_backend.py::test_model_registry_records_parent_model_id -q
E:\miniconda\envs\py312\python.exe -m pytest server\tests\test_sam3_validation.py -q
```

本机轻量结果：Phase A 组合测试 30 passed；G5.1/G6.1 新增测试含 PDF 通过；G3.2 新增导出测试 2 passed；G2.1/G2.2 核心 + 审校 UI 测试通过；储层工作台入口测试 2 passed；模型 UI 解析测试 1 passed；模型版本链测试 1 passed；SAM3 请求验证测试 6 passed。服务器未启动、未重启、未做真实数据验证。`server/tests/test_sam3_batch_train_api.py` 的 FastAPI API 级 `infer_volume` 覆盖需要在服务器环境或装有 fastapi 的环境运行。
