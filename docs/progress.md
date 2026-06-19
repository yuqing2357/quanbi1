# YJ Studio 实施进度核对（历史基线见 archive/implementation_plan.md）

> 基准时间:2026-05 当前工作区 `f:\圈闭软件`。
> 旧项目 `D:\商书记项目\` 保持只读,仅作素材库。
> Python 环境:`E:\miniconda\envs\py312\python.exe`(SAM3 暂同进程使用其中的 torch)。

图例:✅ 完成 / 🟡 部分完成 / ⬜ 未开始 / ⏭ 二期或远期(本期不做)。

---

## 总览

| Phase | 状态 | 说明 |
|---|---|---|
| 0 脚手架 | ✅ | 包结构、PyQt6、logging、libs vendored、legacy 归档、pyproject 全部到位 |
| 1 数据 + 三正交切片 | ✅ | VolumeStore/CoordTransform/AttributeCache + SliceControlsDock + 自动加载 YJ 工区 |
| 2 解释对象骨架 + LayerTree + Property + Undo | ✅ | QUndoStack + 9 个 Command 子类 + PropertyDock + LayerTree 右键菜单(颜色/透明度/重命名/删除/合并/拆分)+ VolumeLayer ROI clipping(数据通路 + SliceControlsDock UI + volume_slice_renderer 截切) |
| 3 井 + 层位 + 断层 + 岩性体 | ✅ | reader/repository/Layer/Renderer/Dock 完整 |
| 4 2D 剖面 + 视图联动 | ✅ | views_area + view_sync_service + 井旁/连井剖面、SectionNavigatorDock |
| 5 交互工具 | ✅ | 9 个核心工具 + 7 个二期 stub + ToolPalette + ToolManager |
| 6 任意剖面 + 沿层/构造高点 | ✅ | arbitrary_section + horizon_service + view_horizon_map + ArbitrarySectionDialog |
| 7 Project + Export | ⬜ | io/writers 空;无 .yjproj save/load;无截图/视频/geojson/mask 导出 |
| 8 算法插件框架 | ✅ | 子进程 Runner + IPC + Layer payload 序列化 + ThicknessAlgorithm + Measure 包装 + 9 个二期 stub + SchemaForm + AlgorithmDock + AddLayerCommand 联动 |
| 9 SAM3 集成 | ✅ | 2026-06-17 规整后：SAM3 只走 AI Dock + RemoteSAM3Client + 服务器 `/sam3/jobs`；本机 AIService/本地 SAM3 算法已删除，通用算法面板不显示 `ai.sam3.*` |
| 10 打包 | ⬜ | packaging/ 不存在 |

---

## 详细盘点

### Phase 0 — 脚手架 ✅

- `local/app/` 包 + `pyproject.toml`(锁定 PyQt6/pyvista/pyvistaqt/vtk/numpy/scipy/pydantic/pyzmq)
- `__main__.py` / `app.py`:`python -m yj_studio` 可启动,Microsoft YaHei 字体
- `logging_config.py` 配置 logging
- `libs/cigvis/` 与 `libs/well_section/` vendored
- `legacy/` 保留两份原型脚本
- 测试基础:`pytest` 配置 + `tests/test_imports.py`

### Phase 1 — 数据 + 三正交切片 ✅

- `data/volume_store.py`:VolumeStore + LRU/mmap
- `data/coord_transform.py`:depth↔sample, ijk↔inline/xline
- `data/attribute_cache.py`:`estimate_volume_clim` 等
- `config/paths.py`:固定工区路径常量(`DEFAULT_SEISMIC_NPY`, `DEFAULT_LITH_POR_MODEL_ROOT`, `existing_processed_root`)
- `config/styles.py`:PALETTE / LITH_BODY_STYLE 等已搬迁
- `io/readers/volume_npy.py`:`load_available_volume_specs` + `VolumeSpec`
- `view/view_3d.py` + `view/renderers/volume_slice_renderer.py`:VTK 切片渲染
- `view/scene_controller.py`:Layer→Actor 派发
- `ui/docks/slice_controls_dock.py`:axis slider + volume 切换 + clim + cmap
- 启动自动加载工区数据(MainWindow.\_discover_default_volumes + load_default_volume)

### Phase 2 — 解释对象骨架 ✅

- 13 个 Layer 子类全部存在(`scene/layers/*.py`)
- `scene/layer_store.py` + selection/camera_state/object_registry/project
- `scene/undo_commands.py`:**QUndoCommand 体系**(`SetLayerFieldCommand` 含 mergeWith / `RenameLayerCommand` / `SetColorCommand` / `SetOpacityCommand` / `SetVisibleCommand` / `AddLayerCommand` / `RemoveLayerCommand` / `MergeLayersCommand` / `SplitLayerCommand`),保留原 `LayerFieldChange` 数据类做兼容
- `MainWindow.undo_stack`(`QUndoStack`,limit=100)+ Edit 菜单(Ctrl+Z / Ctrl+Shift+Z 用 `QKeySequence.StandardKey`)
- `ui/docks/property_dock.py`:通用字段(name/visible/color/opacity)+ Volume 专属(cmap/clim,带 Reset 按钮调 estimate_volume_clim)
- `ui/docks/layer_tree_dock.py`:右键菜单(重命名/颜色/透明度/合并/拆分/删除),所有改动走 `_push(command)`;扩展选中模式;`_on_item_changed` 改名+显隐同时变更时用 macro 包成一条 undo
- 合并/拆分支持 horizon_stick / fault_stick / polygon / annotation,按各自字段(`points` / `sticks` / `vertices` / `items`)拼接或拆开
- `VolumeLayer` 增 `roi: ROIBox` 字段 + `effective_roi()` 自动 clamp/收敛全覆盖→None;`build_slice_image` 接受 ROI 把切片图与 3D quad 同时裁剪;`_slice_within_roi` 在切片落在 ROI 外时直接移除 actor
- `SliceControlsDock` 增 ROI 区(Enable + 6 个 SpinBox + Reset);MainWindow `_set_roi` 通过 `SetLayerFieldCommand` push 到 undo stack
- 新测试:`tests/test_undo_commands.py`(merge/split/add/remove/rename/color/opacity/roi roundtrip)、`tests/test_property_dock.py`、`tests/test_volume_slice_roi.py`
- `tests/conftest.py`:session 级 QApplication fixture(`QT_QPA_PLATFORM=offscreen`)

### Phase 3 — 井 + 层位 + 断层 + 岩性体 ✅

- 读取层:`io/readers/{layers_npz, fault_mesh, lith_body, well_coordinates, well_logs}.py`
- 数据层:`data/well_repository.py`(已实现 from_coordinates_csv)
- Layer:`HorizonLayer / FaultSurfaceLayer / WellLayer / WellLogLayer / LithBodyLayer` 完整
- Renderer:`view/renderers/{horizon, fault, well, well_log, lith_body}_renderer.py` 全部存在
- Dock:`ui/docks/{horizon_dock, fault_dock, wells_dock}.py`
- MainWindow 启动 `load_default_{horizons, faults, lith_bodies, wells}` 一键加载工区
- 测试:`test_well_renderer / test_horizon_renderer / test_fault_renderer / test_lith_body_renderer / test_well_log_renderer / test_well_repository / test_well_coordinates / test_fault_mesh_reader / test_lith_body_reader / test_well_logs.py`

### Phase 4 — 2D 剖面 + 视图联动 ✅

- `view/view_2d_section.py` + `view/views_area.py`:中央多视图(3D + N 个 2D)
- `view/view_well_section.py`:连井剖面视图
- `services/view_sync_service.py`:SyncTopic 发布订阅
- `services/section_service.py`:正交剖面 + 层位/井/断层在剖面上的相交
- `services/well_section_service.py`:连井剖面数据组装(复用 libs/well_section/)
- `ui/docks/{section_navigator_dock, well_section_dock}.py`
- 井旁剖面 = WellsDock 双击 → MainWindow.\_open_well_adjacent_section → inline/xline 自动开
- 高亮联动:`view/highlight.py` + Renderer set_highlight 接口
- 测试:`test_section_service / test_well_section_service.py`

### Phase 5 — 交互工具 ✅

- `tools/tool.py` + `tools/tool_manager.py`:基类 + 活动工具切换 + cursor + 状态栏
- 9 个一期工具:`navigation_tool / point_pick_tool / box_pick_tool / polygon_tool / brush_tool / eraser_tool / horizon_stick_tool / fault_stick_tool / measure_tool`
- 7 个二期 stub(`tools/stubs/__init__.py`):Fill / ConnectedComponent / Threshold / RegionGrow / Snap / Contour / HorizonAutotrack(全部点击提示"Phase 2 stub")
- `tools/catalog.py` 通过 `build_default_tools()` 一次性注册全 16 个
- `ui/docks/tool_palette_dock.py`
- 测试:`test_tools.py`、`test_manual_geometry.py`

### Phase 6 — 任意剖面 + 沿层 + 构造高点 ✅

- `data/arbitrary_section.py`:`sample_arbitrary_section` reslice 引擎
- `scene/layers/arbitrary_section_layer.py`
- `view/view_arbitrary_section.py`
- `ui/dialogs/arbitrary_section_dialog.py`(支持俯视底图 + 井点 + 已选 polyline 输入)
- `services/horizon_service.py`:`build_structure_map / find_horizon_high_point / sample_volume_along_horizon`
- `view/view_horizon_map.py`(构造图 / 沿层切片视图)
- HorizonDock 三个动作:构造图、跳高点、沿层取样
- 菜单 View → New Inline / Xline / Z / Arbitrary Section
- 测试:`test_arbitrary_section.py`

### Phase 7 — Project + Export ⬜

**全部未开始**
- ❌ `io/writers/` 内仅有空 `__init__.py`
- ❌ 无 `io/project_file.py` / `services/project_service.py`
- ❌ MainWindow File 菜单只有 Open Volume + Exit,没有 New/Open/Save/Save As/Recent
- ❌ 无 screenshot / video / geojson / mask 导出
- ❌ `scene/project.py` 仅有数据类,无序列化

### Phase 8 — 算法插件框架 ✅

- `algorithms/algorithm.py`:Algorithm ABC + `input_schema/output_schema/layer_inputs/runs_in_subprocess/supports_cancel` + `import_path()` classmethod
- `algorithms/context.py`:`AlgorithmContext` with progress callback + cancel checker(软取消通过 `report_progress` 自动 `check_cancel`)
- `algorithms/result.py`、`algorithms/registry.py`:`@register_algorithm` 装饰器 + 全局 `registry`
- `algorithms/protocol.py`:`RunMessage` / `ProgressMessage` / `DoneMessage` / `ErrorMessage` / `CancelledMessage` + `CancellationError`
- `algorithms/serialization.py`:`layer_to_payload` / `payload_to_layer`,完整跨进程往返(含 numpy ndarray pickle)
- `algorithms/worker.py`:`run_worker(inbox, outbox)` 子进程入口,Windows-spawn 安全;非阻塞 `cancel` 排空
- `algorithms/runner.py`:`AlgorithmTask`(QObject + QTimer 轮询 outbox,信号 `progress/finished/errored/cancelled`)+ `AlgorithmRunner.submit(...)`(`multiprocessing.Process`)+ `run_sync(...)`(测试/in-proc 用)
- `algorithms/builtin/thickness.py`:`ThicknessAlgorithm`(顶/底 horizon → MeasurementLayer + 5 列 geometry + 统计 dict)
- `algorithms/builtin/measure.py`:`MeasureDistanceAlgorithm`、`MeasureAreaAlgorithm`(包装 MeasureTool 计算)
- 9 个二期 stub(`algorithms/builtin/stubs/_base.py` + 9 个子文件):horizon_autotrack / auto_track_horizon_3d / fault_autopick / sandbody_extract / connectivity / closure_contour / trap_detect / trap_evaluate / region_grow,每个都有完整 pydantic schema,`run` 返回 friendly failure
- `ui/widgets/schema_form.py`:pydantic v2 → Qt 表单(int/float/bool/str/Literal),支持 `ge/le/gt/lt` 数值范围;独立处理 LayerRef(从 `Algorithm.layer_inputs` `{role: "kind1|kind2"}` 自动列出 LayerStore 中匹配的 Layer);`refresh_layer_choices` 接 LayerStore 信号
- `ui/docks/algorithm_dock.py`:分类树(horizon/fault/reservoir/trap/measure)+ 描述 + SchemaForm + Run/Cancel 按钮 + QProgressBar + summary;输出 Layer 走 UndoStack macro,Ctrl+Z 撤销一整次算法运行
- MainWindow:`algorithm_runner = AlgorithmRunner(...)`,新 dock 与 PropertyDock tab 在一起;builtin 模块导入触发自动注册
- 测试:`test_algorithms_thickness.py`、`test_algorithms_measure.py`、`test_algorithms_serialization.py`、`test_algorithms_registry.py`、`test_schema_form.py`、`test_algorithm_runner_subprocess.py`(真子进程端到端)

**Phase 8 暂留尾巴**:厚度 3D 半透明体渲染(`ThicknessRenderer`)未做,当前 MeasurementLayer.geometry 用 manual_geometry_renderer 渲为散点。MeasurementDock 显示数字。**何时补**:Phase 9 完成后,或用户实操时反馈强需求时

### Phase 9 — SAM3 集成 ✅（2026-06-17 规整后口径）

**AI 基础设施**
- 本机不再保留 `ai/config.py`、`ai/service.py` 或本地 SAM3 模型加载 fallback。
- `ai/state.py`:`AIServiceState` 仅作为远程 client/UI 状态枚举。
- `ai/remote_client.py`:`RemoteSAM3Client` 负责连接服务器、提交 `/sam3/jobs`、轮询、取回 mask/GeoTarget 结果。
- `ai/adapters/volume_to_image.py`、`ai/adapters/mask_to_layer.py`、`ai/adapters/frames_export.py` 继续作为切片图像化、mask 图层适配和导出辅助。

**算法与入口**
- 本机已删除旧本地 SAM3 算法文件（原 `algorithms/builtin/ai/sam3_*`）。
- `algorithms/remote_sam3.py` 只保留 AI 面板用的远程描述类；通用 Algorithm Dock 不注册、不显示 `ai.sam3.*`。
- 分割/追踪唯一入口为「AI 面板 + 普通 2D 剖面 → 服务器 `/sam3/jobs` → GeoTarget」。

**算法基础设施扩展**
- `algorithms/context.py`:`AlgorithmContext.services` 字段
- `algorithms/runner.py`:`InProcessAlgorithmTask`(QThread 包装),`AlgorithmRunner` 根据 `runs_in_subprocess` 路由 + `register_service` 注入 ai_service / volume_store
- `algorithms/algorithm.py`:默认 `runs_in_subprocess=True`,AI 算法显式 False

**UI**
- `ui/docks/ai_dock.py`:服务状态横幅 + Start/Unload 按钮 + 轴/切片选择(可同步 3D 视图当前切片)+ text prompt + prompt 列表(box/point)+ Pick Box/Pick Point/Clear + Run/Cancel + 进度条;输出走远程 `/sam3/jobs` 和目标刷新
- `tools/ai_prompt_tools.py`:`AIPointPromptTool` / `AIBoxPromptTool`,2D 剖面视图上拾点/画框 → 通过 ToolManager 的远程 AI 后端服务广播给 AI Dock
- `tools/tool_manager.py`:`register_service` / `service(name)` 让工具拿远程 AI 后端
- `tools/catalog.py`:新工具加进 default 列表
- `view/renderers/mask_renderer.py`:confidence 双通道(高置信不透明、低置信红色半透);同时修正 Z 翻转(`display_z`)与 horizons 对齐
- `ui/main_window.py`:`ai_service` + dock,服务注入 algorithm_runner 和 tool_manager

**测试**
- `test_ai_adapters.py`、`test_remote_sam3_track.py`、`test_sam3_unified_exit.py` 等覆盖远程 client、AI 面板出口和算法面板隐藏；已删除本地 `test_ai_service.py` / `test_sam3_*_algorithm.py`

**已知限制**
- 真实 SAM3 模型、GPU 占用和长任务只在服务器环境验证。
- 第 5 步继续规范化 3D mask 体图层与体积统计。

### Phase 10 — 打包 ⬜

- ❌ `packaging/` 目录不存在
- ❌ 无 yj_studio.spec / installer.nsi
- ❌ docs/user_guide.md / architecture.md / ai_integration.md 未写

---

## 实施推进顺序(本期剩余工作)

| 序号 | 工作块 | 估时 | 依赖 |
|---|---|---|---|
| A | Phase 2 缺口补齐(QUndoStack 接入 + 7 个 Command + PropertyDock + ROI clipping + LayerTree 右键菜单) | 0.5 周 | 当前 |
| B | Phase 7 Project + Export(.yjproj save/load + screenshot/video/geojson/mask) | 1.5 周 | A |
| C | Phase 8 算法插件框架(ThicknessAlgorithm + Measure 包装 + 9 个 stub + schema_form + algorithm_dock) | 2 周 | B |
| D | Phase 9 SAM3 集成(同进程版,torch 直接 import;ai_dock + 单切片→传播→3D + 置信度+修正重推) | 3 周 | C |
| E | Phase 10 打包 + 文档 + 演示视频 | 2 周 | D |

总剩余约 **9 周**(plan 原估剩余 9 周,一致)。

---

## 重要约束(本期不变)

1. 所有新增代码仅落在 `f:\圈闭软件\` 内,**不修改** `D:\商书记项目\`。
2. Python 环境:`E:\miniconda\envs\py312\python.exe`(无界面 smoke test 用 `-B -m pytest`)。
3. SAM3 同进程接入:本期接受,部署时再评估是否拆子进程。
4. 仍遵守 implementation_plan 的五条原则(单向数据流、Layer ≠ Actor、真实数据接入、工具/算法插件化、AI 进程隔离 — 此项暂放宽)。
