# 地震解释与圈闭识别系统 — 软件化实施方案

> 本文档是基于 [需求功能](../需求功能) 全部 196 项功能点，以及现有 [可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py)、[可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py) 两个原型脚本，定稿的桌面软件化实施说明。
>
> **范围**：一期完整实施 + 二期/远期接口预留。
> **技术栈**：Python 3.12 + PyQt6 + PyVista (VTK) + 子进程内 PyTorch (SAM3)。
> **交付形态**：课题组内部工具 + 对外演示软件。
> **数据**：固定单工区（YJ 区块），主体 `F:\YJ-ALL-SEISMIC_depth_0_653.npy` + `处理后文件/` 全套。
>
> 本文档只描述"怎么做"，不附实际代码实现。代码由开发者本人完成。

---

## 目录

1. [整体设计思路](#1-整体设计思路)
2. [核心抽象与契约](#2-核心抽象与契约)
3. [目录结构详解](#3-目录结构详解)
4. [需求功能 → 实现模块映射](#4-需求功能--实现模块映射)
5. [分 Phase 实施步骤](#5-分-phase-实施步骤)
6. [关键逻辑详细说明](#6-关键逻辑详细说明)
7. [SAM3 / AI 接入设计](#7-sam3--ai-接入设计)
8. [二期与远期模块的接口预留](#8-二期与远期模块的接口预留)
9. [打包与分发](#9-打包与分发)
10. [风险清单与替代方案](#10-风险清单与替代方案)
11. [代码风格与工程规约](#11-代码风格与工程规约)
12. [遗留资产复用清单](#12-遗留资产复用清单)
13. [附录](#13-附录)

---

## 1. 整体设计思路

### 1.1 软件定位

这是一个**面向单一工区的桌面地震解释系统**，不是通用可视化框架。它要在一台带 GPU 的工作站上：

- 加载已经处理好的地震体、属性体、层位、断层、井数据
- 提供完整的 2D/3D 浏览、解释、井震对比、岩性储层理解能力
- 提供工业软件级别的交互工具（点选、框选、多边形、画笔、橡皮、测量）
- 预留 AI 自动解释（SAM3）、圈闭识别、圈闭评价、成果表达的扩展点
- 可以通过 PyInstaller 打包给课题组之外的同行/导师演示

### 1.2 与现有原型的关系

现有两份脚本是宝贵的**数据加载/坐标转换/渲染配色**经验来源，但其结构（argparse + 全局函数 + 单 Canvas）**无法承载需求清单**。新软件必须重新组织代码，但要把这两份脚本里所有**业务逻辑**（不是结构）原封不动地搬过去。

具体复用率约 **20–25%**：

| 类别 | 复用 | 重写 |
|---|---|---|
| 数据读取（npy/npz/csv/mesh）| **100% 复用** | – |
| 坐标转换（depth↔sample、ijk↔inline/xline）| **100% 复用** | – |
| 配色/样式常量（PALETTE / LITH_STYLE / VOLUME_DISPLAY_STYLE）| **100% 复用** | – |
| cigvis 自定义 colormap（Petrel / stratum）| **100% 复用** | – |
| `well_section` 模块（沿井 / 连井剖面 HTML 生成）| **原样保留** | – |
| 屏幕投影井点拾取 | 思路参考 | VTK picker 重写 |
| viser/vispy 节点装配 | – | **完全弃用**，改 PyVista Actor |
| `DesktopControlWindow` 单类 Qt 面板 | – | **拆分到 ~10 个 dock** |
| viser server 主循环 | – | 改 Qt 事件循环 |

### 1.3 五条不可动摇的设计原则

1. **L1 (I/O) → L2 (Data) → L3 (Scene) ←→ L4 (View) ↔ L5 (UI) 单向数据流**。任何组件不许越层调用；UI 不直接读数据，必须经过 Scene 层。
2. **场景对象（Layer）≠ 渲染对象（Actor）**。Layer 是纯数据 + 元信息；VTK Actor 是 View 层根据 Layer 的内容生成的渲染产物。两者通过 Qt Signal 同步。
3. **真实数据从第一天接入**。不写"等以后接真实数据"的占位逻辑；不在合成 cube 上验证架构。合成数据只用于 pytest fixture 和 AI mask 占位。
4. **交互工具与算法都是一等公民、可插拔**。新增工具/算法不修改 MainWindow，只注册到 ToolManager / AlgorithmRegistry。
5. **AI 推理与 GUI 进程隔离**。SAM3、auto-track、trap detector 这些重型模型走子进程，通过 ZMQ 与 UI 通信，UI 进程不 import torch。

### 1.4 七层架构总览

```
┌────────────────────────────────────────────────────────────────┐
│ L7 Distribution        PyInstaller spec / NSIS installer       │
├────────────────────────────────────────────────────────────────┤
│ L6 Plugin Layer        entry_points / pluggy registry          │
├────────────────────────────────────────────────────────────────┤
│ L5 UI Layer (PyQt6)                                            │
│    MainWindow │ Docks │ Tool Palette │ Property Editor │ Menus │
├────────────────────────────────────────────────────────────────┤
│ L4 View Layer (PyVista QtInteractor + VTK)                     │
│    View3D │ View2DSection │ SceneController │ Renderers │      │
│    Picker │ ColormapRegistry │ ViewSync                        │
├────────────────────────────────────────────────────────────────┤
│ L3 Scene / Domain Layer (Qt-signal-driven, no rendering)       │
│    LayerStore │ 13×Layer 子类 │ Selection │ Project │          │
│    ObjectRegistry │ UndoStack │ CameraState                    │
├────────────────────────────────────────────────────────────────┤
│ L2 Data Layer (no Qt dependency)                               │
│    VolumeStore │ WellRepository │ HorizonRepository │          │
│    FaultRepository │ AttributeCache │ CoordTransform │         │
│    ArbitrarySectionEngine                                      │
├────────────────────────────────────────────────────────────────┤
│ L1 I/O Layer                                                   │
│    readers: npy/npz/segy/las/csv/grdecl/mesh.npz               │
│    writers: png/mp4/yjproj/mask.npy/geojson/report.pdf         │
├────────────────────────────────────────────────────────────────┤
│ L0 AI Inference (separate process)                             │
│    sam3_worker (subprocess) ←→ AIService (UI side)             │
└────────────────────────────────────────────────────────────────┘
```

L2、L3、L4 三层是软件的**核心**，决定了功能能不能扩、扩得快不快。L1 和 L5 是相对易换的外壳。L0 通过子进程隔离，与主软件解耦。

### 1.5 一期 vs 二期 vs 远期

按需求清单 196 个功能点，分三档：

- **一期完整实施**（36 个核心功能点 + 完整软件骨架）：地震视图、层位/断层/井显示与手动解释、岩性体显示、12 个核心交互工具、SAM3 接入。
- **二期填充算法**（约 80 个功能点）：自动层位追踪、自动断层、砂体识别、连通体、阈值、区域增长、边界吸附、轮廓提取、圈闭识别全套、圈闭评价全套。所有这些**一期已经在算法插件框架里预留了 stub**。
- **远期独立子项目**（约 30 个功能点）：圈闭综合评价表、解释报告生成、版本对比等需要模板引擎和版本管理基建的功能。

---

## 2. 核心抽象与契约

下面定义六个核心抽象的**接口契约**。开发时这六个抽象的方法签名要先定下来再写实现，避免后期修改影响所有 Layer/Tool/Algorithm。

### 2.1 `Layer` 抽象

**职责**：表示一个可显示、可操作、可保存的解释对象。

**公共字段**：
- `id: UUID` — 全局唯一
- `name: str` — 用户可见名（可编辑，对应需求六-14）
- `color: tuple[float,...]` — RGBA，对应需求六-15
- `opacity: float` — 0–1，对应需求六-15
- `visible: bool` — 显隐，对应需求六-17 和 LayerTree 复选框
- `locked: bool` — 锁定后不可被工具修改
- `provenance: dict` — `{"source": "manual" | "auto" | "ai.sam3", "created_at": ..., "params": {...}}`
- `metadata: dict` — 任意附加信息（来源文件、统计、注释）

**公共方法**：
- `bounding_box() -> (xmin, xmax, ymin, ymax, zmin, zmax)` — 渲染层用来做 frustum culling 与相机重置
- `to_dict() -> dict` / `from_dict(d) -> Layer` — 用于 `.yjproj` 序列化
- `accept(visitor)` — Visitor 模式给导出/统计/算法用

**Layer 13 个子类**：

| 子类 | 数据形态 | 对应需求 |
|---|---|---|
| `VolumeLayer` | `np.ndarray` (X,Y,Z) + clim + cmap | 一-1,2,3,12 |
| `ArbitrarySectionLayer` | polyline + 重采样图像 | 一-5,6,7,8 |
| `HorizonLayer` | (X,Y) → Z 网格 + mask | 二-1,4 |
| `HorizonStickLayer` | 用户拾取点集 | 二-2 |
| `FaultSurfaceLayer` | 三角网格 vertices + faces | 三-1 |
| `FaultStickLayer` | 用户拾取断层杆 | 三-2 |
| `WellLayer` | 井轨迹 polyline + 井头位置 + 井名 | 四-1,2,3 |
| `WellLogLayer` | 沿井轨迹的值序列（POR/PERM/LITH）| 四-6,7 |
| `LithBodyLayer` | 三角网格（透明岩性体）| 五-1,2 |
| `MaskLayer` | 2D 或 3D 0/1 或多类 mask | 六-1~7、七-1~16 |
| `PolygonLayer` | 3D polygon vertices | 六-3、八-13 |
| `AnnotationLayer` | 文本/线/点的混合集合 | 六-4 |
| `MeasurementLayer` | 距离/面积/厚度计算结果 | 二-5、九-1,2,3 |

> **重要约定**：Layer 不持有 VTK 对象。Layer 修改后通过 `LayerStore.signals.layer_changed.emit(layer_id, field)` 通知 View 层重建/更新对应 Actor。

### 2.2 `LayerStore`

继承 `QObject`，是 Scene 层的中央容器。

**信号**：
- `layer_added(layer_id: str)`
- `layer_removed(layer_id: str)`
- `layer_changed(layer_id: str, field: str)` — field ∈ {"color", "opacity", "visible", "data", "name", ...}
- `selection_changed(layer_ids: list[str])`

**方法**：
- `add(layer: Layer) -> str`
- `remove(layer_id: str)`
- `get(layer_id: str) -> Layer`
- `iter_by_type(layer_cls) -> Iterator[Layer]`
- `update(layer_id, **fields)` — 统一入口，内部发 signal

**LayerStore 是 LayerTree dock、SceneController、Algorithm Runner 三方的共享真相源**。

### 2.3 `InteractionTool` 抽象

**职责**：响应 View 的鼠标/键盘事件，转化为对 Layer 的修改。

**接口（实现这几个钩子即可）**：

- `id: str`、`label: str`、`icon: str`、`cursor: str`
- `activate(view)` — 切换为当前工具时调用
- `deactivate(view)` — 退出工具时调用
- `on_mouse_press(view, event)` / `on_mouse_move` / `on_mouse_release`
- `on_key_press(view, event)`
- `on_pick_result(world_xyz, picked_layer_id, picked_cell_id)` — Picker 命中后回调

**ToolManager** 持有 `active_tool`，把 View3D / View2DSection 的事件转发给它，并在切换工具时调用 activate/deactivate。

**一期实现的工具**：Navigation、PointPick、BoxPick、Polygon、Brush、Eraser、HorizonStick、FaultStick、Measure。
**二期填充的工具**：Fill、ConnectedComponent、Threshold、RegionGrow、Snap、Contour、HorizonAutoTrack。stub 类一期就要建好（带禁用状态 + tooltip 提示"二期功能"）。

### 2.4 `Algorithm` 抽象

**职责**：把"输入 Layer + Prompt 参数 → 输出 Layer"的处理过程封装成可插拔单元。

**接口**：
- `id: str` — 如 `"horizon.thickness"`、`"ai.sam3.box_prompt"`
- `category: str` — `"horizon" | "fault" | "reservoir" | "trap" | "ai" | "measure"`
- `label: str`、`description: str`
- `input_schema: pydantic.BaseModel` — 描述输入 layer 类型、参数、prompt
- `output_schema: pydantic.BaseModel`
- `runs_in_subprocess: bool` — True 表示需要 AI worker / 算力较重
- `supports_cancel: bool`
- `run(ctx: AlgorithmContext) -> AlgorithmResult`

**关键设计**：`input_schema` 是 pydantic 模型，**UI 自动根据 schema 生成参数面板**（用 `qtpydantic` 风格的代码或自写一个）。这样新增算法不需要写 UI 代码，只需要写 schema + run。

**AlgorithmRunner** 负责：
- 在主线程或子进程里启动 algorithm
- 监听进度（`AlgorithmContext.report_progress`）
- 取消支持
- 异常 → 状态栏提示
- 输出 layer 自动加入 LayerStore

### 2.5 `Picker`

VTK 自带 `vtkCellPicker` / `vtkPointPicker` / `vtkPropPicker`。封装统一接口：

- `pick(view, screen_xy, modes=["cell", "point", "prop"]) -> PickResult`
- `PickResult`: `world_xyz`、`layer_id`、`picked_type`（cell/point/prop）、`extra`（如井名、层位名）

避免 [run_cigvis_web_with_por_perm_lith_well_desktop.py:759](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py) 那种"屏幕投影+距离阈值"的近似拾取——VTK 原生 picker 在透明 actor 上做不准时，再增加 invisible pickable proxy actor 兜底。

### 2.6 `ViewSyncService`

需求一-4 三向正交联动、一-12 剖面与解释对象联动、四-4 井旁剖面联动——所有联动通过这一个服务实现。

**核心抽象**：`SyncTopic`，例如：
- `slice.inline_position` (int)
- `slice.xline_position` (int)
- `slice.z_position` (int)
- `selection.current_layer` (str)
- `camera.azimuth_elevation`

任何 View 都可以 `subscribe(topic, callback)` 与 `publish(topic, value)`。这样新加一个视图（比如 2D matplotlib 剖面）只需要订阅几个 topic，不用改其他视图代码。

---

## 3. 目录结构详解

```
商书记项目/
├─ apps/
│   └─ yj_studio/                        # 主软件
│       ├─ pyproject.toml
│       ├─ src/yj_studio/
│       │   ├─ __init__.py
│       │   ├─ __main__.py               # python -m yj_studio
│       │   ├─ app.py                    # QApplication 启动、主题、字体、i18n
│       │   │
│       │   ├─ config/                   # 全局常量
│       │   │   ├─ defaults.py           # Z_WINDOW_START / DEPTH_STEP_TO_SAMPLE 等
│       │   │   ├─ paths.py              # 固定工区路径
│       │   │   ├─ styles.py             # PALETTE / LITH_STYLE / VOLUME_DISPLAY_STYLE / LITH_BODY_STYLE
│       │   │   └─ shortcuts.py
│       │   │
│       │   ├─ io/                       # L1
│       │   │   ├─ readers/
│       │   │   │   ├─ volume_npy.py
│       │   │   │   ├─ layers_npz.py
│       │   │   │   ├─ fault_mesh.py
│       │   │   │   ├─ well_logs.py      # POR / PERM / LITH csv
│       │   │   │   ├─ grdecl_model.py
│       │   │   │   ├─ lith_body.py
│       │   │   │   ├─ segy.py           # 用 segyio，二期接入
│       │   │   │   └─ las.py            # 复用 cigvis/io/las.py
│       │   │   ├─ writers/
│       │   │   │   ├─ screenshot.py
│       │   │   │   ├─ video.py
│       │   │   │   ├─ mask_npy.py
│       │   │   │   ├─ project.py        # .yjproj 写
│       │   │   │   ├─ report.py         # 远期，HTML/PDF 报告
│       │   │   │   ├─ geojson.py
│       │   │   │   └─ segy_horizon.py
│       │   │   └─ project_file.py       # .yjproj 解析（TOML + 引用外部 .npy/.npz）
│       │   │
│       │   ├─ data/                     # L2
│       │   │   ├─ volume_store.py       # mmap LRU、属性体注册表
│       │   │   ├─ well_repository.py
│       │   │   ├─ horizon_repository.py
│       │   │   ├─ fault_repository.py
│       │   │   ├─ attribute_cache.py    # estimate_clim 等
│       │   │   ├─ coord_transform.py    # depth↔sample, ijk↔inline/xline
│       │   │   └─ arbitrary_section.py  # 任意折线 reslice 引擎
│       │   │
│       │   ├─ scene/                    # L3
│       │   │   ├─ layer.py              # Layer 基类
│       │   │   ├─ layers/               # 13 个子类，每个文件一个
│       │   │   ├─ layer_store.py
│       │   │   ├─ selection.py
│       │   │   ├─ camera_state.py
│       │   │   ├─ project.py            # Project / Session 数据类
│       │   │   ├─ object_registry.py    # UUID-based 解释对象池
│       │   │   └─ undo_commands.py      # QUndoCommand 子类集
│       │   │
│       │   ├─ view/                     # L4
│       │   │   ├─ qt_vtk_view.py        # QtInteractor 子类基础
│       │   │   ├─ view_3d.py            # 主 3D 视图
│       │   │   ├─ view_2d_section.py    # 2D 剖面视图
│       │   │   ├─ view_sync.py          # 信号桥
│       │   │   ├─ scene_controller.py   # Layer → Actor 派发器
│       │   │   ├─ renderers/            # 每种 Layer 一个 renderer
│       │   │   ├─ picker.py
│       │   │   ├─ axis_overlay.py
│       │   │   └─ colormap_registry.py  # 复用 cigvis.colormap
│       │   │
│       │   ├─ tools/                    # 交互工具
│       │   │   ├─ tool.py               # InteractionTool 基类
│       │   │   ├─ tool_manager.py
│       │   │   ├─ navigation_tool.py
│       │   │   ├─ point_pick_tool.py
│       │   │   ├─ box_pick_tool.py
│       │   │   ├─ polygon_tool.py
│       │   │   ├─ brush_tool.py
│       │   │   ├─ eraser_tool.py
│       │   │   ├─ horizon_pick_tool.py
│       │   │   ├─ fault_stick_tool.py
│       │   │   ├─ measure_tool.py
│       │   │   └─ stubs/                # 二期工具占位
│       │   │       ├─ fill_tool.py
│       │   │       ├─ connected_component_tool.py
│       │   │       ├─ threshold_tool.py
│       │   │       ├─ region_grow_tool.py
│       │   │       ├─ snap_tool.py
│       │   │       ├─ contour_tool.py
│       │   │       └─ horizon_autotrack_tool.py
│       │   │
│       │   ├─ algorithms/               # 算法插件
│       │   │   ├─ algorithm.py          # Algorithm 基类
│       │   │   ├─ registry.py
│       │   │   ├─ context.py            # AlgorithmContext (input layers + params + progress)
│       │   │   ├─ result.py
│       │   │   ├─ runner.py             # 本地/子进程统一调度
│       │   │   └─ builtin/
│       │   │       ├─ thickness.py      # 二-5 层间厚度
│       │   │       ├─ measure_distance.py
│       │   │       ├─ measure_area.py
│       │   │       └─ stubs/            # 二期算法占位
│       │   │           ├─ horizon_autotrack.py
│       │   │           ├─ fault_autopick.py
│       │   │           ├─ sandbody_extract.py
│       │   │           ├─ connectivity.py
│       │   │           ├─ trap_detect.py
│       │   │           ├─ trap_evaluate.py
│       │   │           ├─ closure_contour.py
│       │   │           └─ region_grow.py
│       │   │
│       │   ├─ ai/                       # L0 客户端
│       │   │   ├─ ai_service.py         # UI 端门面
│       │   │   ├─ sam3_protocol.py      # IPC 消息格式
│       │   │   ├─ sam3_client.py        # 主进程客户端 (ZMQ)
│       │   │   ├─ workers/
│       │   │   │   └─ sam3_worker.py    # 子进程 entry point
│       │   │   └─ adapters/
│       │   │       ├─ volume_to_image.py    # 抽切片 → SAM3 输入
│       │   │       ├─ mask_propagation.py   # 七-13 跨切片传播
│       │   │       ├─ mask_to_3d.py         # 七-14 2D → 3D
│       │   │       └─ mask_to_layer.py
│       │   │
│       │   ├─ ui/                       # L5
│       │   │   ├─ main_window.py
│       │   │   ├─ views_area.py         # 中央多视图 Tab/Split
│       │   │   ├─ docks/
│       │   │   │   ├─ layer_tree_dock.py
│       │   │   │   ├─ property_dock.py
│       │   │   │   ├─ tool_palette_dock.py
│       │   │   │   ├─ slice_controls_dock.py
│       │   │   │   ├─ section_navigator_dock.py
│       │   │   │   ├─ well_section_dock.py     # 复用 well_section 模块
│       │   │   │   ├─ horizon_dock.py
│       │   │   │   ├─ fault_dock.py
│       │   │   │   ├─ wells_dock.py
│       │   │   │   ├─ algorithm_dock.py
│       │   │   │   ├─ ai_dock.py
│       │   │   │   ├─ log_dock.py
│       │   │   │   └─ measurement_dock.py
│       │   │   ├─ widgets/
│       │   │   │   ├─ color_button.py
│       │   │   │   ├─ opacity_slider.py
│       │   │   │   ├─ layer_tree_widget.py
│       │   │   │   ├─ colormap_picker.py
│       │   │   │   └─ schema_form.py    # pydantic schema → Qt 表单
│       │   │   ├─ dialogs/
│       │   │   │   ├─ project_dialog.py
│       │   │   │   ├─ export_dialog.py
│       │   │   │   ├─ arbitrary_section_dialog.py
│       │   │   │   └─ about_dialog.py
│       │   │   ├─ menus.py
│       │   │   └─ status_bar.py
│       │   │
│       │   ├─ services/                 # 跨层服务
│       │   │   ├─ project_service.py
│       │   │   ├─ export_service.py
│       │   │   ├─ task_runner.py        # QThreadPool 包装
│       │   │   ├─ view_sync_service.py
│       │   │   ├─ navigation_service.py # 一-10 构造高定位
│       │   │   ├─ section_service.py    # 任意/沿井/沿层剖面
│       │   │   └─ telemetry.py
│       │   │
│       │   ├─ plugins/registry.py
│       │   └─ resources/
│       │       ├─ icons/
│       │       ├─ qss/
│       │       └─ i18n/
│       │
│       └─ tests/
│           ├─ data/                     # 真实数据子集 + 极少量合成
│           ├─ test_volume_store.py
│           ├─ test_layer_store.py
│           ├─ test_tools.py
│           └─ test_algorithms.py
│
├─ libs/
│   ├─ cigvis/                           # 现有副本 → 这里
│   └─ well_section/                     # 现有 well_section 模块
│
├─ legacy/                               # 老脚本归档（不删，做参考）
│   ├─ run_cigvis_web_with_por_perm_lith_wells.py
│   └─ run_cigvis_web_with_por_perm_lith_well_desktop.py
│
├─ sam3/                                 # 模型源码（已存在）
├─ tools/                                # 数据预处理脚本（已存在）
├─ processed/                            # 软链接到 F:\YJ-..._processed
├─ docs/
│   ├─ implementation_plan.md            # 本文档
│   ├─ architecture.md                   # 后续：架构总览
│   ├─ user_guide.md                     # 后续：用户手册
│   └─ ai_integration.md                 # 后续：AI 接入细节
└─ packaging/
    ├─ yj_studio.spec                    # PyInstaller
    └─ installer.nsi                     # NSIS
```

### 3.1 命名约定

- 模块名全部 snake_case，类名 PascalCase。
- Layer 子类一律 `XxxLayer`、Renderer 子类一律 `XxxRenderer`、Tool 一律 `XxxTool`、Algorithm 一律 `XxxAlgorithm`。
- 信号统一命名 `xxx_changed` / `xxx_added` / `xxx_removed`。
- 文件名与类名一致，每文件一个主类。

### 3.2 包依赖关系（不可逆向）

```
ui    → services, scene, view, tools, algorithms, ai
view  → scene, data, config
tools → scene, view (只允许读 view，不允许直接改 actor)
algorithms → scene, data, ai (走 ai_service)
scene → data, config
data  → io, config
io    → (无内部依赖)
config → (无内部依赖)
ai    → (无内部依赖，与主软件解耦)
```

**严格禁止反向依赖**。任何"data 想用 ui 的状态"的需求都说明设计错了，应改为"ui 把状态写进 scene/services"。

---

## 4. 需求功能 → 实现模块映射

完整映射表，按需求清单顺序。带 **★** 的为一期完整实现，带 **□** 的为一期 stub 预留 + 二期填充，带 **○** 的为远期。

### 一、地震解释视图

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 Inline 切片 | `view_3d.py` + `volume_slice_renderer.py` + `slice_controls_dock.py` | ★ Phase 1 |
| 2 Crossline 切片 | 同上 | ★ Phase 1 |
| 3 Time/Depth 切片 | 同上 | ★ Phase 1 |
| 4 三向正交联动 | `view_sync_service.py` | ★ Phase 1 |
| 5 任意方向剖面 | `arbitrary_section.py` + `arbitrary_section_layer.py` + `arbitrary_section_renderer.py` + `view_2d_section.py` | ★ Phase 6 |
| 6 任意折线剖面 | 同上，UI 入口在 `section_navigator_dock.py` 的"绘制折线" | ★ Phase 6 |
| 7 沿井剖面 | 复用 [libs/well_section/](../libs/well_section/) → `well_section_dock.py` + `view_2d_section.py` | ★ Phase 6 |
| 8 沿层剖面 | `section_service.along_horizon()` → 同 view_2d_section | ★ Phase 6 |
| 9 局部区域裁剪 | `volume_layer` 增加 ROI clipping box（VTK `vtkBoxClipDataSet` 或 `vtkPlane` 多平面切割）| ★ Phase 2 |
| 10 构造高部位快速定位 | `navigation_service.py` + `horizon_dock.py` 的"跳转最高点"按钮 | ★ Phase 6 |
| 11 同相轴辅助追踪 | `algorithms/builtin/stubs/horizon_autotrack.py` | □ 二期 |
| 12 剖面与解释对象联动 | `view_sync_service.py` 订阅 `selection_changed` + 各 Renderer 实现 highlight | ★ Phase 4 |

### 二、层位解释

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 层位显示 | `horizon_layer.py` + `horizon_renderer.py` | ★ Phase 3 |
| 2 层位手动拾取 | `horizon_pick_tool.py` + `horizon_stick_layer.py` | ★ Phase 5 |
| 3 层位自动追踪 | `algorithms/builtin/stubs/horizon_autotrack.py` | □ 二期 |
| 4 层位构造图 | `horizon_dock.py` 调 matplotlib 等高线，导出 PNG | ★ Phase 6 |
| 5 层间厚度计算 | `algorithms/builtin/thickness.py` → 生成 `MeasurementLayer` | ★ Phase 8 |

### 三、断层解释

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 断层显示 | `fault_surface_layer.py` + `fault_renderer.py` | ★ Phase 3 |
| 2 断层手动拾取 | `fault_stick_tool.py` + `fault_stick_layer.py` | ★ Phase 5 |

### 四、井与剖面对比

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 井位显示 | `well_layer.py` + `well_renderer.py`（井头点 + 名签）| ★ Phase 3 |
| 2 井轨迹显示 | 同上，polyline | ★ Phase 3 |
| 3 井名显示 | 同上，VTK `vtkBillboardTextActor3D` 或 `vtkCaptionActor2D` | ★ Phase 3 |
| 4 井旁地震剖面 | `section_service.through_well()` + `view_2d_section.py` | ★ Phase 4 |
| 5 连井剖面 | 复用 `libs/well_section/` → `well_section_dock.py` | ★ Phase 4 |
| 6 井上岩性柱 | `welllog_layer.py` 模式="lith" + `welllog_renderer.py` | ★ Phase 3 |
| 7 井上孔隙度曲线 | `welllog_layer.py` 模式="por"/"perm" | ★ Phase 3 |

### 五、岩性与储层理解

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 岩性体三维显示 | `lithbody_layer.py` + `lithbody_renderer.py`（复用 LITH_BODY_STYLE）| ★ Phase 3 |
| 2 岩性分类显示 | 同上，按 class_value 上色 | ★ Phase 3 |
| 3 砂体空间展布识别 | `algorithms/builtin/stubs/sandbody_extract.py` | □ 二期 |
| 4 砂体边界提取 | 同上 + `algorithms/builtin/stubs/closure_contour.py` | □ 二期 |
| 5 砂体厚度估计 | `algorithms/builtin/stubs/sandbody_extract.py` 输出 thickness map | □ 二期 |
| 6 砂体连通性 | `algorithms/builtin/stubs/connectivity.py`（基于 6/26 邻接）| □ 二期 |
| 7 砂体与井关系 | `algorithms/builtin/stubs/sandbody_extract.py` + WellRepository 查询 | □ 二期 |
| 8 有效储层范围 | 二期 | □ |
| 9 储层顶底界面 | 二期 | □ |
| 10,11 岩性圈闭 / 复合圈闭 | 二期 | □ |

### 六、交互解释工具（一期核心）

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 点选解释 | `point_pick_tool.py` + `picker.py` | ★ Phase 5 |
| 2 框选解释 | `box_pick_tool.py` | ★ Phase 5 |
| 3 多边形圈选 | `polygon_tool.py` + `polygon_layer.py` | ★ Phase 5 |
| 4 画笔标注 | `brush_tool.py`（写 `MaskLayer` / `AnnotationLayer`）| ★ Phase 5 |
| 5 橡皮擦修改 | `eraser_tool.py` | ★ Phase 5 |
| 6 区域填充 | `tools/stubs/fill_tool.py` | □ 二期 |
| 7 连通体选择 | `tools/stubs/connected_component_tool.py` | □ 二期 |
| 8 阈值筛选 | `tools/stubs/threshold_tool.py` | □ 二期 |
| 9 区域增长 | `tools/stubs/region_grow_tool.py` | □ 二期 |
| 10 边界吸附 | `tools/stubs/snap_tool.py` | □ 二期 |
| 11 轮廓线提取 | `tools/stubs/contour_tool.py` | □ 二期 |
| 12 对象合并 | `object_registry.py` + `undo_commands.MergeCommand` | ★ Phase 2 |
| 13 对象拆分 | `object_registry.py` + `undo_commands.SplitCommand` | ★ Phase 2 |
| 14 对象重命名 | `layer_tree_dock.py` 双击 + `undo_commands.RenameCommand` | ★ Phase 2 |
| 15 颜色/透明度调整 | `property_dock.py` + `layer_tree_dock.py` 右键 | ★ Phase 2 |
| 16 人工修正 | 工具直接写 Layer，Undo 自动支持 | ★ Phase 5 |
| 17 三维高亮 | `selection.py` + 各 Renderer 实现 `set_highlight(bool)` | ★ Phase 2 |
| 18 与剖面同步 | `view_sync_service.py` | ★ Phase 4 |

### 七、AI 辅助解释

| 需求 | 实现位置 | 阶段 |
|---|---|---|
| 1 SAM3 切片分割 | `ai/workers/sam3_worker.py` | ★ Phase 9 |
| 2 点提示 | `ai_dock.py` + `point_pick_tool.py`（mode=ai_prompt）| ★ Phase 9 |
| 3 框提示 | `ai_dock.py` + `box_pick_tool.py`（mode=ai_prompt）| ★ Phase 9 |
| 4 文本提示 | `ai_dock.py` 文本输入 | ★ Phase 9 |
| 5 多提示联合 | `ai_service.py` 聚合多个 prompt | ★ Phase 9 |
| 6–11 地震事件/断层候选/砂体/河道/扇体/异常振幅分割 | 都走同一 SAM3 入口，不同 prompt 模板 | ★ Phase 9（基础）/ □ 二期（模板化）|
| 12 分割结果人工修正 | brush/eraser 直接编辑 `MaskLayer` | ★ Phase 9 |
| 13 跨切片传播 | `ai/adapters/mask_propagation.py` | ★ Phase 9 |
| 14 2D mask → 3D 体 | `ai/adapters/mask_to_3d.py` | ★ Phase 9 |
| 15 3D 连通体提取 | `algorithms/builtin/stubs/connectivity.py` | □ 二期 |
| 16 置信度显示 | `MaskLayer.confidence` 字段 + `mask_renderer.py` 双通道 | ★ Phase 9 |
| 17 不确定区域提示 | 同上，confidence < threshold 高亮 | ★ Phase 9 |
| 18 候选结果排序 | `ai_dock.py` 列表 | ★ Phase 9 |
| 19 AI vs 人工对比 | 远期（需要版本管理）| ○ |
| 20 人工修正后重推 | `ai_service.py` 接受 mask layer 作为 prompt | ★ Phase 9 |

### 八–十、圈闭识别/评价/成果表达

全部 □ 二期 或 ○ 远期。**一期只做一件事：在 `algorithms/builtin/stubs/` 下建好每个算法的 stub 类**（带 schema、不带实现），让 UI 能列出来并提示"二期功能"。

详见 [§8 二期与远期模块的接口预留](#8-二期与远期模块的接口预留)。

---

## 5. 分 Phase 实施步骤

总长度约 **22 周**（一期）。每个 Phase 给出：目标、任务清单、验收标准、依赖前提。

### Phase 0：脚手架（1 周）

**目标**：把工程骨架立起来，能跑空白主窗口；定义六大核心抽象的方法签名。

**任务清单**：
1. 创建 `apps/yj_studio/` 目录、`pyproject.toml`、`__main__.py`。
2. 配 Python 3.12 虚拟环境，锁定关键依赖版本：
   - `PyQt6>=6.6`
   - `pyvista>=0.43`、`pyvistaqt>=0.11`
   - `vtk>=9.3`
   - `numpy`、`scipy`、`pydantic>=2`、`pyzmq`
   - 暂不要装 torch（留给 AI worker 子进程的独立环境）
3. 写 `app.py`：QApplication 启动、设置中文字体（思源黑体/微软雅黑）、加载 qss 主题。
4. 写 `main_window.py` 雏形：菜单栏 + 状态栏 + 空白中央区。
5. **写六个核心抽象的接口文件，只定义签名不写实现**：
   - `scene/layer.py`
   - `scene/layer_store.py`
   - `tools/tool.py`、`tools/tool_manager.py`
   - `algorithms/algorithm.py`、`algorithms/registry.py`、`algorithms/runner.py`
   - `view/picker.py`
   - `services/view_sync_service.py`
6. 配置 logging（输出到 console + `~/.yj_studio/logs/`）。
7. 配 pre-commit（black / isort / ruff）+ pytest 框架。
8. 把 [可视化文件/cigvis/](../可视化文件/cigvis/) 整个移到 `libs/cigvis/`，把 [可视化文件/代码/well_section/](../可视化文件/代码/well_section/) 移到 `libs/well_section/`，让 `apps/yj_studio` 通过 editable install 引用。

**验收**：
- `python -m yj_studio` 能弹出窗口，标题"YJ Studio v0.1.0"。
- 六个抽象的接口文件 import 成功，pytest 空跑通过。
- pre-commit 跑通。

**依赖**：无。

---

### Phase 1：数据 + 场景 + 三正交切片（2 周）

**目标**：把真实地震体 `F:\YJ-ALL-SEISMIC_depth_0_653.npy` 装进软件，3D 视图里能看到三正交切片，可切换属性体、调 clim、调 cmap。

**任务清单**：
1. **L1 readers 抽取**：
   - 把 [run_cigvis_web_with_por_perm_lith_wells.py:709](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) 的 `load_volume_by_key` 和 `load_available_volume_specs` 抽到 `io/readers/volume_npy.py`。
   - 把 `load_layers` 抽到 `io/readers/layers_npz.py`（即便此 Phase 不用，也先抽出来）。
2. **L2 数据**：
   - 实现 `VolumeStore`：mmap 加载 + 多体注册表 + LRU（默认缓存 3 个体）+ `get_slice(axis, index, volume_id)` 方法。
   - 实现 `AttributeCache`：把 `estimate_clim` / `estimate_volume_clim` 搬过来。
   - 实现 `CoordTransform`：从 [defaults.py](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py:57) 的 `Z_WINDOW_START` / `DEPTH_STEP_TO_SAMPLE` 出发，封装 `depth_m_to_sample(depth)` / `sample_to_depth(sample)` / `ijk_to_inline_xline(i,j,k)`。
3. **L3 场景**：
   - 实现 `VolumeLayer` + 完整字段。
   - 实现 `LayerStore` + 信号。
   - 实现最简 `Project`（只存"当前打开的 volume id"和 camera state）。
4. **L4 视图**：
   - 实现 `qt_vtk_view.py`：`QtInteractor` 子类，挂到中央区。
   - 实现 `volume_slice_renderer.py`：用 `vtkImageData` + `vtkImageReslice`（不是 `vtkVolume`，体太大不要做体渲染）渲染三个 axis-aligned 切片。
   - 实现 `scene_controller.py`：监听 `LayerStore.layer_added/changed`，分发到对应 renderer。
5. **L5 UI**：
   - `slice_controls_dock.py`：三个 slider 控制 x/y/z 切片位置 + 一个 dropdown 切体 + 两个 spinbox 调 clim + cmap 下拉。
   - 菜单栏 "File → Open Volume..." 调 `volume_npy.py` 读 .npy。
6. **第一刀：默认行为**——启动时如果 `paths.py` 中的固定工区路径存在，自动加载 `seismic.npy` 并显示。让你打开软件即看到工区数据。

**验收**：
- 启动软件后看到真实 YJ 地震体的三正交切片。
- 切片位置滑块工作，clim/cmap 工作。
- 切换到属性体（coherence、dip_angle、curvature 等）工作，shape mismatch 友好报错。
- VolumeStore 内存占用合理（mmap，不是 full load）。

**依赖**：Phase 0。

**注意事项**：
- `vtkImageData` 在 GB 级数据上不要 SetScalars 后 Modified() 整体重传，要用 reslice 提取切片后 SetInputData 给独立的 `vtkImageActor`。
- 切片更新时只更新对应 axis 的一个 actor，不要重建整个 scene。
- `VolumeStore` 切换 volume 时必须保证旧 actor 解绑后才释放旧 mmap，否则 VTK 还在访问已 munmap 的内存会段错误。

---

### Phase 2：解释对象骨架 + LayerTree + Property + Undo（2 周）

**目标**：搭起完整的解释对象数据模型与 UI 编辑能力，覆盖需求六-12,13,14,15,17。

**任务清单**：
1. **L3 完善**：
   - 13 个 Layer 子类全部写好**字段定义 + to_dict/from_dict**（不需要 renderer 全部就绪，但数据契约要稳）。
   - `ObjectRegistry`：UUID 池 + 解释对象关系图（如"horizon_top_T3 是 by_user 创建的"）。
   - `selection.py`：当前选中 Layer 列表 + `selection_changed` 信号。
   - `undo_commands.py`：实现 SetVisibleCommand / SetColorCommand / SetOpacityCommand / RenameCommand / RemoveLayerCommand / MergeCommand / SplitCommand。
2. **L5 dock**：
   - `layer_tree_dock.py`：QTreeView + 树节点显示所有 Layer，支持显隐复选框、双击改名、右键菜单（颜色、透明度、删除、合并、拆分、跳转中心）。
   - `property_dock.py`：选中 Layer 时显示其字段（name/color/opacity/metadata），编辑触发 Undo Command。
3. **L4 渲染**：
   - 给所有 Renderer 实现 `set_highlight(layer_id, bool)`（高亮当前 selection）。
   - VolumeLayer 增加 ROI clipping（需求一-9）：用 `vtkBoxWidget2` 或代码生成 `vtkPlane` 多平面切割。
4. **集成 QUndoStack**：菜单"编辑 → 撤销/重做"，快捷键 Ctrl+Z / Ctrl+Shift+Z。

**验收**：
- LayerTree 里能看到当前 Project 的所有 Layer。
- 显隐复选框可撤销重做。
- 改颜色/透明度可撤销重做。
- 重命名、合并（多选 + 右键合并）、拆分能工作。
- 选中某个 Layer，3D 视图里它对应的 actor 视觉高亮（如 outline）。

**依赖**：Phase 1。

**注意事项**：
- 合并/拆分仅在同类型 Layer 之间允许（不允许 Horizon 与 Fault 合并）。
- Undo 栈的容量限制要在 settings 里配，默认 100 步。
- LayerTree 的多选行为：Ctrl+点击多选、Shift+点击范围选；右键菜单根据选中数量动态显示项。

---

### Phase 3：井 + 层位 + 断层显示（2 周）

**目标**：把现有原型脚本里的全部静态显示能力（井位、井轨迹、井名、井柱、层位、断层、岩性透明体）以 Layer 形式恢复。

**任务清单**：
1. **L1 readers 复用**：把 [run_cigvis_web_with_por_perm_lith_wells.py:336](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) 的 `load_layers` / `load_fault_meshes` / `load_attribute_logs` / `load_lithology_body_meshes` 全部抽到 `io/readers/`，保持函数签名与行为完全一致。
2. **L2 repository**：
   - `WellRepository`：从 `combined_well_coordinates_inside_*.csv` 构建 in-memory well 表，提供按名查询、按 inline/xline 范围查询。
   - `HorizonRepository`：从 `层位/*.npz` 集合构建。
   - `FaultRepository`：从 `断层/*_mesh.npz`。
3. **L3 Layer 实现**：HorizonLayer / FaultSurfaceLayer / WellLayer / WellLogLayer / LithBodyLayer。
4. **L4 Renderer**：
   - `horizon_renderer.py`：`vtkPlaneSource` 或 `vtkStructuredGrid` + 颜色按层位名 PALETTE 取，复用现有 [build_surface](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py:356) 的 NaN 处理逻辑。
   - `fault_renderer.py`：`vtkPolyData` + 法线计算 + smooth shading。
   - `well_renderer.py`：井轨迹 `vtkTubeFilter` + 井头球 + 井名 billboard text。**保留 [run_cigvis_web_with_por_perm_lith_wells.py:598](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) `create_well_name_labels` 的 z_offset / font_screen_scale 概念**。
   - `welllog_renderer.py`：沿井轨迹的点或柱（按 cmap 上色）。
   - `lithbody_renderer.py`：从 `lithology_body_class_*_mesh.npz` 加载 + 透明 mesh。
5. **L5 docks**：
   - `wells_dock.py`：列出所有井，可选中、可定位（跳转到井位）。
   - `horizon_dock.py`：列出层位，复选框控制显隐。
   - `fault_dock.py`：列出断层。
6. **样式常量统一**：把 PALETTE / LITH_STYLE / VOLUME_DISPLAY_STYLE / LITH_BODY_STYLE 搬到 `config/styles.py`。

**验收**：
- 启动 → 自动加载 → 视图里能看到地震切片 + 所有层位 + 所有断层 + 所有井（井轨迹 + 井头 + 井名 + 井上 POR/PERM/LITH 曲线/柱）+ 透明岩性体。
- 等效于原始 [run_cigvis_web_with_por_perm_lith_wells.py](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) 的初始视觉效果。

**依赖**：Phase 2。

**注意事项**：
- 井名 label 在 VTK 里推荐 `vtkBillboardTextActor3D`，永远面向相机。需求四-3 的"井名显示"在视觉上比原型的"屏幕投影 text"要稳定。
- 井轨迹深度单位是 m，sample index 是 0–653。`load_attribute_logs` 里 `sample = depth_m / DEPTH_STEP_TO_SAMPLE - Z_WINDOW_START` 这一行**必须**封装到 `CoordTransform`，否则未来切换深度窗口会到处改。
- 岩性透明体（LithBodyLayer）的 alpha 滑块要复用现有 [run_cigvis_web_with_por_perm_lith_wells.py:1230](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) 的逻辑。

---

### Phase 4：2D 剖面视图 + 视图联动（2 周）

**目标**：实现工业软件必备的"3D + 任意数量 2D 剖面"多视图布局。这是后续井剖面、井旁剖面、沿层剖面、AI 单切片标注的前提。

**任务清单**：
1. **L4 视图**：
   - `view_2d_section.py`：基于 matplotlib + Qt（`FigureCanvasQTAgg`）。一个 2D section view 显示：底图地震切片 + 该切片上经过的层位线 + 经过的断层线 + 经过的井柱叠加。
   - `views_area.py`：中央区改成 `QSplitter` 或 `QTabWidget` 嵌套，支持 1 个 3D view + N 个 2D view。
2. **L4 同步**：`view_sync.py` + `services/view_sync_service.py`。
   - SyncTopic 至少包括：`slice.inline_position`、`slice.xline_position`、`slice.z_position`、`selection.current_layer`、`camera.azimuth_elevation`。
   - 3D 视图改 inline 位置 → 已打开的 2D Inline View 自动刷新；反之亦然。
3. **L5 UI**：
   - `section_navigator_dock.py`：列出当前打开的所有 2D 视图（Inline=320、Xline=180、Well=W12 等），可关闭、可重命名、可重新激活。
   - 菜单"视图 → 新建 Inline 剖面"等入口。
4. **井旁剖面（需求四-4）**：
   - `section_service.through_well(well_name)`：根据井位 ijk 计算 inline/xline 切片或一段沿井 polyline 剖面，返回 reslice 出来的 2D 数组。
   - `wells_dock.py` 双击井名 → 新建该井的井旁剖面 view。
5. **连井剖面（需求四-5）**：复用 `libs/well_section/`。`well_section_dock.py` 选两口以上井 → 调 `build_well_section_html` 生成 HTML → 用 `QWebEngineView` 嵌入 2D view，或浏览器打开（与原型一致）。
6. **解释对象与剖面同步（需求一-12, 六-18）**：
   - 选中某层位时，所有 2D 视图里该层位的线条加粗高亮。
   - 选中某井时，3D 视图相机平移到井头，井柱外圈出现高亮 outline。

**验收**：
- 能同时打开 3D 视图 + Inline 280 剖面 + Xline 150 剖面 + 井 W12 井旁剖面。
- 移动 3D 视图的 inline 切片 → Inline 280 视图同步更新到新位置（或新建另一个 Inline）。
- 双击层位 T3 → 所有视图里 T3 高亮加粗。
- 选两口井生成连井 HTML 剖面，能展示。

**依赖**：Phase 3。

**注意事项**：
- 2D 视图不要试图用 PyVista，matplotlib 在 2D 切片 + 层位线 + 文本标注上更灵活。
- ViewSync 用 Qt signal 实现，避免循环触发：每个 publish 加一个 `_origin` 参数，订阅者发现自己是 origin 就忽略。
- 沿层剖面（一-8）实质是"沿层位深度面做 reslice"，在 `section_service.along_horizon()` 里实现：给定 horizon layer，沿其深度面采样地震体（用 `scipy.ndimage.map_coordinates`）。

---

### Phase 5：交互工具调色板 + 基础工具（2.5 周）

**目标**：建立 ToolPalette + 9 个一期核心工具 + 工具状态机闭环。需求六-1,2,3,4,5,16,17 + 二-2 + 三-2 + 二-5。

**任务清单**：
1. **L3+L4 工具基建**：
   - `tools/tool.py`：InteractionTool 基类。
   - `tools/tool_manager.py`：active_tool 切换、事件转发、cursor 切换。
   - View3D / View2DSection 把鼠标/键盘事件桥接到 ToolManager。
2. **9 个工具**：
   - **NavigationTool**（默认）：照常旋转/平移/缩放，不拦截事件。
   - **PointPickTool**：单击 → Picker → 选中对应 Layer 或 cell。需求六-1。
   - **BoxPickTool**：drag rectangle → 选中框内 Layer / cell。需求六-2。
   - **PolygonTool**：依次单击落点 → 双击完成 → 生成 PolygonLayer。需求六-3。
   - **BrushTool**：在 MaskLayer 上画 0/1。半径可调。需求六-4。
   - **EraserTool**：BrushTool 反向。需求六-5。
   - **HorizonStickTool**：在 2D 剖面上单击落点 → 生成/扩展 HorizonStickLayer。需求二-2。
   - **FaultStickTool**：在 2D 剖面上画一段断层杆。需求三-2。
   - **MeasureTool**：单击两点测距离 / 多边形测面积 / 沿剖面测厚度。需求二-5、九-1,2,3。
3. **二期 stub 工具**：在 `tools/stubs/` 下建好 fill / connected_component / threshold / region_grow / snap / contour / horizon_autotrack 七个类，全部继承 InteractionTool 但 `on_mouse_press` 弹"该功能将在二期实现"的状态栏提示。
4. **L5 ToolPalette**：
   - `tool_palette_dock.py`：垂直按钮组，每个工具一个图标按钮。
   - 切换工具时光标变化、状态栏显示当前工具名。
5. **人工修正（需求六-16）**：BrushTool/EraserTool 在 MaskLayer 上的每次笔划生成一个 `EditMaskCommand`，可撤销。

**验收**：
- ToolPalette 里能看到 16 个工具按钮，9 个可用、7 个禁用提示二期。
- 切换工具 → cursor 变化 → 在 3D 或 2D 视图操作 → 生成对应 Layer 或修改对应 Layer。
- 所有工具操作都可 Ctrl+Z。
- 测距/测面积/测厚度结果显示在 `measurement_dock.py`。

**依赖**：Phase 4。

**注意事项**：
- BrushTool 在 3D 视图中是"投射到当前活动切片"，不是空中画。需要 ToolManager 维护"当前活动切片轴 + 位置"。
- PolygonTool 落点判定：3D 中按"投射到当前切片"，2D 中按平面坐标。
- HorizonStick / FaultStick 在 3D 视图也允许操作（在当前 inline/xline 上落点），但**主要工作流是在 2D 剖面上**。

---

### Phase 6：任意剖面 + 沿层/沿井剖面 + 构造高点定位（2 周）

**目标**：需求一-5,6,7,8,10、二-4。

**任务清单**：
1. **任意方向剖面（一-5）**：
   - `arbitrary_section.py`：给定起点终点 + 沿 z 范围，调 `scipy.ndimage.map_coordinates` 在地震体内重采样出 (N, Z) 2D 图。
   - UI：`arbitrary_section_dialog.py` 让用户输入起终点或在 3D 视图上拾两点。
2. **任意折线剖面（一-6）**：
   - 多段 polyline 重采样，把每段拼成一个长 2D 图。
3. **沿井剖面（一-7）**：已在 Phase 4 用 `well_section` 模块实现，此处只补 UI 入口（从 wells_dock 右键"沿井剖面"）。
4. **沿层剖面（一-8）**：
   - `section_service.along_horizon(horizon_layer, depth_offset)`：沿 horizon 深度面做 reslice。
   - 用途：展示砂体沿某层的展布。
5. **构造高点定位（一-10）**：
   - `navigation_service.locate_structure_high(horizon_layer)`：找 horizon 最深值（深度小=构造高）位置。
   - 跳转：3D 相机平移到该 ijk + 三正交切片定位到该位置。
6. **层位构造图（二-4）**：
   - `horizon_dock.py` 新增"生成构造图"按钮：用 matplotlib `contour` + `contourf` 出等高线图，导出 PNG 到工程目录。

**验收**：
- 能在 3D 视图上拾两点 → 生成任意剖面 2D view。
- 能在 3D 视图上画折线 → 生成折线剖面。
- 选某层位 → 一键跳转构造高点，3D 与所有 2D 视图同步定位。
- 选某层位 → 一键生成构造等高线图 PNG。

**依赖**：Phase 5。

**注意事项**：
- `map_coordinates` 在 mmap 数组上工作正常但很慢，必要时把目标 ROI 一次性读到内存。
- 折线剖面要标注每个折点的水平位置，否则用户看不出走向。

---

### Phase 7：Project 文件 + Export（1.5 周）

**目标**：让软件具备工程化基础——能保存当前会话、能输出截图视频。

**任务清单**：
1. **`.yjproj` 文件设计**：
   - 顶层 TOML 描述：工区根路径、当前 Volume、所有打开的 Layer 元信息、视图布局、camera 状态、selection。
   - Layer 的数据本身（如 HorizonStick 的点集、PolygonLayer 的顶点）：小数据写入 TOML 内嵌；大数据（MaskLayer）写到 `<projname>_data/` 旁边的 `.npy`/`.npz`。
2. **`project_service.py`**：save/load/save_as/recent_files。
3. **菜单与对话框**：File → New / Open / Save / Save As / Export...
4. **Export**：
   - `screenshot.py`：当前 3D view → PNG（高分辨率渲染：`vtkRenderLargeImage`）。
   - `video.py`：相机绕轴旋转录屏，输出 MP4（用 `imageio-ffmpeg`）。
   - `geojson.py`：把 HorizonLayer/FaultStickLayer/PolygonLayer 输出 GeoJSON（用于与 GIS 软件交换）。
   - `mask_npy.py`：MaskLayer 输出 `.npy`。
5. **固定工区默认值**：File → New Project 默认从 `paths.py` 拉路径，无需用户输入。

**验收**：
- 软件状态可保存到 `.yjproj`，重启软件 → Open → 完全恢复。
- 截图/视频/GeoJSON/Mask 导出工作。

**依赖**：Phase 6。

**注意事项**：
- `.yjproj` 永远不要把巨型 numpy 数组塞进 TOML，那样会让文件几百 MB 不可读。
- 视频导出要在独立 QThread，不阻塞 UI；要有进度条与取消按钮。

---

### Phase 8：算法插件框架 + 端到端验证（2 周）

**目标**：把 Algorithm 抽象彻底落地。让"层间厚度"和"测距/测面积"端到端跑通，并在 `algorithm_dock` 里自动生成参数面板。

**任务清单**：
1. **Algorithm 框架**：
   - `algorithms/algorithm.py`：基类 + Protocol。
   - `algorithms/context.py`：`AlgorithmContext`(input_layers, params, report_progress, request_cancel, layer_store)。
   - `algorithms/result.py`：`AlgorithmResult`(output_layers, summary, ok, error)。
   - `algorithms/runner.py`：本地直跑 vs subprocess（先实现本地，subprocess 在 Phase 9 与 AI 共用）。
   - `algorithms/registry.py`：装饰器 `@register_algorithm` 自动注册。
2. **`schema_form.py`**：pydantic schema → Qt 表单。支持以下字段类型：
   - 数字 (int/float) → QSpinBox/QDoubleSpinBox
   - 字符串 → QLineEdit
   - bool → QCheckBox
   - enum (Literal) → QComboBox
   - Layer 引用 (LayerRef) → QComboBox 列出 LayerStore 中匹配类型的 Layer
3. **`algorithm_dock.py`**：左侧分类树（horizon / fault / reservoir / trap / ai / measure），右侧选中算法后显示其 schema_form + "运行"按钮。
4. **一期落地算法**：
   - `ThicknessAlgorithm`：输入两个 HorizonLayer，输出 MeasurementLayer（厚度网格 + 等值线 + 平均厚度）。
   - `MeasureDistanceAlgorithm`、`MeasureAreaAlgorithm`：从 MeasurementTool 的产出包装成 Algorithm，保持 UI 一致。
5. **二期 stub 注册**：把所有二期 stub 算法（horizon_autotrack / sandbody_extract / connectivity / trap_detect / closure_contour / region_grow / fault_autopick / trap_evaluate）注册到 registry，schema 完整、run 方法 raise NotImplementedError 并提示二期。这样 `algorithm_dock` 里能看到完整的功能图谱，让用户/演示对象感受到软件的完整规划。

**验收**：
- `algorithm_dock` 里能看到所有一期+二期算法分类列表。
- 选 ThicknessAlgorithm → 选两个层位 → 运行 → 生成 MeasurementLayer 显示在 3D 视图与 LayerTree。
- 二期算法点运行 → 提示"二期功能，敬请期待"。

**依赖**：Phase 7。

**注意事项**：
- schema_form 是整个软件长期受益的基础组件。投入时间做扎实。
- ThicknessAlgorithm 的输出 layer 要带 provenance 标记 "auto.thickness"，便于后续 AI vs 人工对比。

---

### Phase 9：SAM3 集成（3 周）

**目标**：需求七-1,2,3,4,5,12,13,14,16,17,18,20。

**任务清单**：
1. **AI 子进程框架**：
   - `ai/sam3_protocol.py`：定义 IPC 消息（pydantic 模型）：`LoadModelRequest`、`SegmentRequest`、`ProgressTick`、`SegmentResponse`、`CancelRequest`、`ShutdownRequest`。
   - `ai/workers/sam3_worker.py`：子进程 entry。启动后 import 你的 [sam3/sam3/sam3/agent/agent_core.py](../sam3/sam3/sam3/agent/agent_core.py)，加载模型到 GPU，进入 ZMQ REP 循环。
   - `ai/sam3_client.py`：主进程客户端，封装"发请求→等响应+进度"的异步 API。
   - `ai/ai_service.py`：UI 端门面，与 LayerStore 集成。
   - **进程隔离原则**：主软件 `pip install` 不依赖 torch。AI worker 用独立的 conda env（`yj_studio_ai`），由 `ai_service` 启动子进程时指定 python 路径。
2. **数据适配器**：
   - `volume_to_image.py`：把 VolumeStore 的切片转成 SAM3 期望的 RGB image（dynamic range 拉伸到 0–255）。
   - `mask_to_layer.py`：SAM3 输出的 mask（H,W）回写为 MaskLayer。
   - `mask_propagation.py`：用户在 inline=N 上获得 mask 后，对相邻 inline=N±1, N±2, ... 自动重推（用前帧 mask 作为 mask prompt）。需求七-13。
   - `mask_to_3d.py`：多个 2D mask 在 inline 方向堆叠成 3D MaskLayer。需求七-14。
3. **UI**：
   - `ai_dock.py`：选 prompt 类型（point/box/text/multi）→ 选 prompt 输入工具 → 运行 → 显示结果列表（按 confidence 排序，对应七-18）→ 接受/拒绝 → mask 进入 LayerStore。
   - `point_pick_tool` / `box_pick_tool` 增加 mode="ai_prompt"：拾取后不进入 selection，而是 publish 给 ai_service。
4. **置信度可视化（七-16,17）**：MaskRenderer 用双通道：mask=1 高不透明、confidence<阈值时半透明红色 outline。
5. **人工修正后重推（七-20）**：用户用 brush/eraser 改完 mask → ai_dock"以当前 mask 作为提示重新分割"按钮 → 把当前 mask 作为 mask prompt 喂回 SAM3。

**验收**：
- 启动 AI worker（菜单"AI → 启动 SAM3 服务"），状态栏显示"AI ready"。
- 在 Inline 切片上画框 → 几秒后看到分割 mask 叠加显示。
- 一键传播到相邻 20 个 inline → 形成一个 3D mask。
- 用 brush 修改后重推，结果变化。

**依赖**：Phase 8。

**注意事项**：
- SAM3 加载需要 30 秒以上，主进程不要 block。`ai_service.start()` 异步启动 + 状态机（loading → ready → busy → error）。
- ZMQ 用 REQ/REP 简单稳妥；如果想要服务器主动 push 进度，用 PUSH/PULL 加一个独立的 progress 端口。
- Windows 下 multiprocessing 必须 `if __name__ == "__main__":` 守卫；用 spawn 启动方式。
- 子进程 crash 时主进程要捕获、状态栏提示、允许重启。

---

### Phase 10：打包、文档、演示（2 周）

**目标**：可分发的安装包 + 用户手册 + 演示视频。

**任务清单**：
1. **PyInstaller**：
   - `packaging/yj_studio.spec`：onedir 模式（不要 onefile，VTK 大）。
   - 处理 PyVista/VTK 的数据文件（shaders、icons）。
   - 排除 torch（AI 子进程独立 env 单独打包）。
2. **NSIS 安装包**：
   - 主软件 + 可选 AI 包（用户勾选才安装 ~5GB AI env）。
   - 创建桌面快捷方式 + 开始菜单项 + 关联 .yjproj 文件。
3. **`docs/user_guide.md`**：截图 + 步骤的用户手册，覆盖一期所有功能。
4. **演示视频**：录一段 10–15 分钟的完整工作流（打开 → 浏览 → 解释 → AI 辅助 → 导出报告）。
5. **`docs/architecture.md`** 与 **`docs/ai_integration.md`**：开发者文档。

**验收**：
- 干净 Windows 机器双击安装 → 启动 → 加载 → 演示完整功能。
- 演示视频可发课题组、可放报告。

**依赖**：Phase 9。

**注意事项**：
- 打包前用 `pyinstaller --debug imports` 检查丢失模块。
- 中文字体在 packaged exe 里要显式包含 ttf 文件并 `QFontDatabase.addApplicationFont`。
- VTK 9.x 在 PyInstaller 下偶尔会缺少 vtkmodules，spec 文件需要 `hiddenimports` 显式列出。

---

## 6. 关键逻辑详细说明

### 6.1 启动序列

```
1. QApplication 启动
2. 加载 qss 主题 + 中文字体
3. 检查 ~/.yj_studio/settings.json，读取 last_project 路径
4. 实例化 MainWindow（空状态）
5. 实例化所有 docks（隐藏状态）
6. 实例化 LayerStore、SelectionService、UndoStack、ViewSyncService、TaskRunner
7. 实例化 View3D、把它放进 views_area
8. 注册所有 InteractionTool 到 ToolManager
9. 注册所有 Algorithm 到 AlgorithmRegistry
10. 如果 last_project 存在 → 加载它
    否则如果 config.paths 中固定工区路径存在 → 自动 new Project 并加载默认 layers
    否则 → 显示欢迎对话框
11. 显示主窗口
```

### 6.2 Layer 增删改的完整数据流

**新增 Layer**（以加载层位为例）：

```
[User] 菜单 File → Import → Horizon
  ↓
[UI] horizon_dock 弹文件对话框，选 *.npz
  ↓
[L1] io/readers/layers_npz.py: load_layers(path) → dict{sample, mask, meta}
  ↓
[L2] HorizonRepository: 转成内存对象
  ↓
[L3] HorizonLayer(name=path.stem, sample=..., mask=..., color=PALETTE[i])
[L3] LayerStore.add(layer) → 发 signal layer_added(uuid)
  ↓
[L4] SceneController 接 signal → 找到 horizon_renderer → renderer.add(layer) → 创建 VTK actor
[L5] LayerTreeDock 接 signal → 在树里追加一行
[L5] LayerTreeDock 复选框默认勾选 → set_visible 触发 layer_changed("visible")
[L4] horizon_renderer 接 layer_changed → actor.SetVisibility(True)
  ↓
[L5] StatusBar: "Added horizon: T3"
```

**修改 Layer**（改颜色为例）：

```
[User] 在 PropertyDock 改颜色
  ↓
[L5] property_dock.color_button → undo_stack.push(SetColorCommand(layer_id, old, new))
  ↓
[L3] SetColorCommand.redo() → LayerStore.update(layer_id, color=new) → 发 layer_changed("color")
  ↓
[L4] 对应 renderer 接 signal → actor.GetProperty().SetColor(...) + render
[L5] LayerTreeDock 接 signal → 树节点图标颜色更新
```

**关键约定**：**任何对 Layer 字段的修改都必须经过 `LayerStore.update()`**。绝对禁止 `layer.color = new_color` 直接赋值——那会绕过 signal 与 undo。建议把 Layer 实现为 `@dataclass(frozen=True)` 或 `pydantic.BaseModel`，强制走 update 路径。

### 6.3 工具状态机

```
ToolManager.active_tool = NavigationTool (默认)

[User] 在 ToolPalette 点 PolygonTool
  ↓
ToolManager.set_active("polygon")
  ├─ active_tool.deactivate(view)  # NavigationTool 退出
  ├─ active_tool = PolygonTool 实例
  └─ active_tool.activate(view) → cursor = "crosshair"

[User] 在 3D view 单击
  ↓
View3D.mousePressEvent → ToolManager.dispatch_press(view, event)
  ↓
PolygonTool.on_mouse_press(view, event):
  - Picker.pick(view, event.pos()) → world_xyz
  - 把 world_xyz 加到当前未完成的 polygon points 中
  - 临时画一段红线表示在编辑
  - 不调 LayerStore.add（还没完成）

[User] 双击完成
  ↓
PolygonTool.on_mouse_double_click:
  - 创建 PolygonLayer(points=current_points)
  - undo_stack.push(AddLayerCommand(layer))
  - 清空 current_points
  - 视图刷新
```

### 6.4 算法运行的统一流程

```
[User] 在 algorithm_dock 选 ThicknessAlgorithm
  ↓
schema_form 根据 ThicknessAlgorithm.input_schema 渲染：
  - Layer ref: "顶层" (筛选 HorizonLayer)
  - Layer ref: "底层"
  - float: "深度步长" 默认 10.0

[User] 选完点"运行"
  ↓
algorithm_dock.on_run:
  ctx = AlgorithmContext(
      params=schema_form.collect(),
      input_layers={"top": ..., "bottom": ...},
      layer_store=layer_store,
      progress_callback=lambda v: progress_bar.setValue(v),
      cancel_check=lambda: cancel_button.is_clicked()
  )
  AlgorithmRunner.run(ThicknessAlgorithm, ctx) → AlgorithmResult

[Runner]:
  - 如果 algorithm.runs_in_subprocess → 派子进程
  - 否则 → QThreadPool 跑（不阻塞 UI）
  - 监听 progress / cancel

[Algorithm.run]:
  - 读输入 layer 数据
  - 计算 thickness map
  - 返回 AlgorithmResult(output_layers=[MeasurementLayer(name="T2-T3 厚度")])

[Runner 收到结果]:
  - 把输出 layer 加到 layer_store
  - 触发 layer_added signal
  - 状态栏 "算法完成: 平均厚度 25.3 m"
```

### 6.5 撤销/重做模型

QUndoStack + QUndoCommand 子类。**所有的"状态改变"都要走 Command**。

每个 Command 实现 `redo()` 和 `undo()`，**必须是对称、可重复的**。

例如 `SetVisibleCommand(layer_id, old_value, new_value)`：
- `redo()`: `layer_store.update(layer_id, visible=new_value)`
- `undo()`: `layer_store.update(layer_id, visible=old_value)`

`EditMaskCommand`（BrushTool 用）：
- 存 patch 修改前的小区域 mask 副本 + 修改后的副本
- redo/undo 互换 patch

**绝对不要存"整个 mask 的副本"**——大 mask 撤销栈会爆内存。只存被改动的最小 bounding box。

### 6.6 视图联动的循环触发防护

ViewSyncService 用 publisher pattern。**防循环关键**：

```
publish(topic, value, origin=self):
    for subscriber in subscribers[topic]:
        if subscriber is origin:
            continue
        subscriber.on_sync(topic, value)
```

每个 View 在接收到 sync 时**不要再 publish**。如果某些情况下必须级联（如 inline 变化触发整个 camera reset），也必须显式带 origin 让链终止。

### 6.7 大文件加载策略

| 文件 | 大小级 | 策略 |
|---|---|---|
| seismic.npy | 几 GB | `np.load(mmap_mode="r")`，永不 full load |
| 属性体 | 同上 | 同上，LRU 缓存最多 3 个 |
| 层位 .npz | MB 级 | 全加载 |
| 断层 mesh .npz | MB 级 | 全加载，VTK 拷贝 vertices/faces |
| 测井 csv | KB 级 | 全加载到 dict |
| 岩性体 mesh | MB–几十 MB | 全加载 |
| MaskLayer | 看尺寸 | 优先 in-memory，超过 100MB 时 backing file |

VolumeStore 切换属性体时**不要立刻丢旧体**——保持 LRU，因为用户经常来回切。

---

## 7. SAM3 / AI 接入设计

### 7.1 进程拓扑

```
                  UI 进程 (主)
┌──────────────────────────────────────────────────┐
│ PyQt6 + VTK + numpy                              │
│                                                  │
│  ai_service ── sam3_client ──┐                   │
│                              │ ZMQ REQ/REP       │
│                              │ ZMQ PUSH/PULL     │
└──────────────────────────────┼───────────────────┘
                               │
┌──────────────────────────────┼───────────────────┐
│ AI 进程 (子进程, conda env: yj_studio_ai)        │
│ python + torch + cuda + sam3                     │
│                                                  │
│  sam3_worker ── load SAM3 once, hold on GPU      │
│  ZMQ REP listening                               │
│  reads VolumeStore via shared mmap path          │
└──────────────────────────────────────────────────┘
```

**为什么子进程**：
- SAM3 是 PyTorch + CUDA，初始化 ~1.5GB 显存
- 模型加载 ~30 秒，不能阻塞 UI
- 主软件不依赖 torch，打包体积可控
- 子进程 crash 不带挂 UI

### 7.2 IPC 消息格式

所有消息用 pydantic 序列化为 JSON。msgpack 也行但 JSON 调试方便。

```
# 主 → 子
LoadModelRequest:
  model_path: str
  device: "cuda:0" | "cpu"

SegmentRequest:
  request_id: str          # UUID
  volume_path: str         # 子进程 mmap 直接读
  axis: "inline"|"xline"|"z"
  slice_index: int
  roi: [i_min, i_max, j_min, j_max] | null
  prompts:
    - kind: "point"|"box"|"text"|"mask"
      value: 详见各 kind
  options:
    multi_mask: bool
    return_confidence: bool

CancelRequest:
  request_id: str

ShutdownRequest

# 子 → 主
ModelReady:
  device: str
  model_id: str

ProgressTick:
  request_id: str
  fraction: float (0-1)
  message: str

SegmentResponse:
  request_id: str
  masks: list[MaskPayload]
    MaskPayload:
      class_id: int
      class_name: str | null
      confidence: float | null
      shape: [H, W]
      data_path: str    # 落到临时文件，主进程读取
      rle: str | null   # 可选 RLE 编码

ErrorResponse:
  request_id: str | null
  code: str
  message: str
```

### 7.3 SAM3 调用细节

参考 [sam3/sam3/sam3/agent/agent_core.py](../sam3/sam3/sam3/agent/agent_core.py) 与 [sam3/sam3/sam3/model/sam3_image_processor.py](../sam3/sam3/sam3/model/sam3_image_processor.py) 的 API。`sam3_worker.py` 的工作：

1. 启动时 `import sam3.agent.agent_core` 等。
2. 加载权重。
3. 在 REP 循环里：
   - 收到 `SegmentRequest` → 从 `volume_path` mmap 取出 `slice_index` 切片 → dynamic range 拉伸到 0–255 RGB → 把 prompts 翻译为 SAM3 期望的格式 → 推理 → mask 落地 → 返回 `SegmentResponse`。
4. 进度通过 PUSH 端口主动推。

### 7.4 跨切片传播（七-13）

`ai/adapters/mask_propagation.py`：

```
输入: seed_mask (在 inline=N0 上), 方向: ±10 inline
循环 i in [N0+1, N0+2, ..., N0+10]:
    上一帧 mask 作为 mask prompt
    可选: 加上一些点 prompt（从上一帧 mask 内随机/中心采）
    调 sam3_client.segment(inline=i, prompt=mask_prompt)
    保存结果
    if confidence < 阈值 → 停止传播
反向同理
最终聚合成 (N_propagated, H, W) → MaskLayer (3D)
```

### 7.5 2D Mask → 3D Volume（七-14）

`ai/adapters/mask_to_3d.py`：把 propagation 输出的多个 2D mask 在第三轴上堆叠，并可选用 `scipy.ndimage.morphology` 做一次 3D 闭运算填平 inline 之间的小缺口。

### 7.6 置信度与不确定（七-16,17）

SAM3 输出 mask 时同时输出 confidence map（每像素或每 mask 一个标量）。MaskLayer 增加 `confidence: np.ndarray | float | None` 字段。MaskRenderer 渲染时：
- mask=1 区域：按 layer 颜色不透明
- confidence < threshold 区域：叠加红色 outline（用 `vtkContourFilter` 取 confidence 等值线）

### 7.7 人工修正后重推（七-20）

用户用 brush/eraser 修改 mask 后，ai_dock 显示"基于当前 mask 重新分割"按钮：把当前 MaskLayer 作为 `kind="mask"` prompt 发回 SAM3。SAM3 会基于 mask 提示精修边界。

---

## 8. 二期与远期模块的接口预留

### 8.1 二期算法 stub 清单

一期在 `algorithms/builtin/stubs/` 下建好以下文件，**每个文件**都要做到：

1. 继承 `Algorithm`
2. `id` / `category` / `label` / `description` 写完整
3. `input_schema` / `output_schema` 写完整（pydantic 模型）
4. `run()` 抛 `NotImplementedError("二期实现")`
5. `runs_in_subprocess` 标对

| 文件 | id | 输入 | 输出 |
|---|---|---|---|
| `horizon_autotrack.py` | `horizon.autotrack` | VolumeLayer + 种子点 | HorizonLayer |
| `fault_autopick.py` | `fault.autopick` | VolumeLayer + 属性体 | FaultSurfaceLayer 列表 |
| `sandbody_extract.py` | `reservoir.sandbody_extract` | VolumeLayer + HorizonLayer(顶) + HorizonLayer(底) | MaskLayer + AnnotationLayer(统计) |
| `connectivity.py` | `reservoir.connectivity` | MaskLayer + WellLayer 列表 | AnnotationLayer(连通图) |
| `closure_contour.py` | `trap.closure_contour` | HorizonLayer | PolygonLayer 列表 |
| `trap_detect.py` | `trap.detect_structural` | HorizonLayer + FaultLayer 集 | 多 PolygonLayer + 评分 |
| `trap_evaluate.py` | `trap.evaluate` | TrapLayer + 多种输入 | AnnotationLayer(评价表) |
| `region_grow.py` | `mask.region_grow` | MaskLayer + 种子点 + 阈值 | MaskLayer |
| `auto_track_horizon_3d.py` | `horizon.autotrack_3d` | VolumeLayer + HorizonStickLayer | HorizonLayer |

**为什么一期就做 stub**：

- `algorithm_dock` 一开始就能展示完整功能矩阵，给课题组演示有底气。
- 二期实现某个算法时，UI 一行不动，schema 已经定好，只需要把 `run` 写出来。
- schema 定下来的过程会暴露很多"二期到底需要什么输入"的问题，提前发现胜过晚改。

### 8.2 圈闭识别（八）与圈闭评价（九）

需求八/九共 45 项，全部走算法插件路径。一期只做两件事：

1. 建 8.1 表里的 `trap_detect.py` 与 `trap_evaluate.py` 两个 stub。
2. 在 `scene/layers/` 下建 `TrapLayer`（类似 PolygonLayer 但带评价属性字段）。

二期把这两个 stub 的 `run` 写出来即可。

### 8.3 成果表达（十）

17 项，全部远期。骨架已经在 `io/writers/report.py` 预留。二期不动，远期独立子项目处理。

---

## 9. 打包与分发

### 9.1 两个分发产物

| 产物 | 体积 | 内容 | 受众 |
|---|---|---|---|
| `YJStudio-1.0-Setup.exe` | ~300 MB | UI 主软件 + VTK + numpy | 课题组演示 |
| `YJStudio-AI-1.0-Setup.exe` | ~5 GB | 主软件 + AI conda env + SAM3 权重 | 完整工作站 |

### 9.2 PyInstaller spec 要点

- `onedir` 模式
- `hiddenimports`: `vtkmodules.all`、`pyvista._plot`、`pydantic`、`zmq.backend.cython`
- `datas`: 中文字体 ttf、qss、icons、cigvis 自带的 cmap 数据
- `excludes`: torch、torchvision、PIL（如不需要）

### 9.3 NSIS 安装脚本

- 主软件强制安装
- AI 包用 Section /o（默认不选）
- 安装目录权限要可写（用户数据在 `%APPDATA%/YJStudio/`）
- 关联 `.yjproj` 文件双击打开

### 9.4 版本管理

- `pyproject.toml` 里维护 `version`
- `app.py` 启动时把版本号写状态栏与 About 对话框
- 自动化：`bumpver` 或手动改

---

## 10. 风险清单与替代方案

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| **VTK 体渲染 GB 级卡顿** | 高 | Phase 1 推进受阻 | 不用 `vtkVolume`，只用 `vtkImageReslice` 做切片；mmap 不 full load；切片缓存（同一 slice 不重 reslice）|
| **PyQt6 + pyvistaqt 兼容性** | 中 | 启动失败 | 锁版本组合（pyvista 0.43.x + pyvistaqt 0.11.x + PyQt 6.6.x）；备选退到 PyQt5 |
| **PyInstaller 打包 VTK/PyTorch 失败** | 高 | 不能交付演示 | 分两个包；VTK 用 onedir；torch 进 AI 子环境，主软件不打 |
| **SAM3 是 2D 模型，3D 效果差** | 高 | 七-1 体验不佳 | 三种 prompt 策略并行实验：①per-slice + propagation；②inline+xline+z 三视投票；③SAM3 做检测，3D region grow 扩展。Phase 9 留时间对比 |
| **multiprocess + CUDA 在 Windows 起不来** | 中 | AI 不可用 | spawn 启动；`if __name__ == "__main__"` 守卫；ZMQ 替代 pipe；明确报错指导 |
| **VTK pick 在透明 mesh 上失准** | 中 | 井点击失败 | 加 invisible pickable proxy actor；保留屏幕投影距离阈值作为 fallback |
| **中文字体在 packaged 后乱码** | 中 | 演示效果差 | 显式打包 ttf 文件 + `QFontDatabase.addApplicationFont`；VTK text 用 `vtkTextProperty.SetFontFile` |
| **`.yjproj` 序列化大数据撑爆 TOML** | 高 | 工程文件不可读 | TOML 仅存元信息和小数据；大数据（MaskLayer）外置 .npy；定 100KB 阈值 |
| **撤销栈吃光内存（大 Mask 编辑）** | 高 | 软件崩溃 | EditMaskCommand 只存被改动的最小 bbox；栈容量限制；定期清理 |
| **View 间循环触发死循环** | 中 | 软件挂起 | publish 必带 origin；订阅者不 republish；加日志监控 sync 调用次数 |
| **大 mmap 在 VTK actor 上仍持有时被 munmap** | 中 | 段错误 | VolumeStore 切换 volume 时严格按"先解绑 actor → render once → 才释放 mmap"顺序；用 weakref 跟踪 |
| **PyVista 高级 API 隐藏 VTK 底层** | 低 | 性能调优受限 | 关键路径直接 vtkmodules，PyVista 只用作 scene/actor 组装 |
| **cigvis 上游变动导致接口断裂** | 低 | 渲染参数不一致 | libs/cigvis 当作 vendored 代码，禁止从 PyPI 升级；课题组 fork 固定 commit |
| **真实数据 inline/xline 索引偏移与原型不一致** | 中 | 井位错位 | CoordTransform 单测覆盖原型行为；新加任何数据前先单测对齐 |
| **AI 进程模型权重路径错** | 中 | AI 一直 loading | 启动时验证权重存在；权重路径写入 settings.json 可改 |
| **未来要支持多工区** | 中 | 重构 | Project 已经是一等公民，固定工区只是默认值，多工区只需要改 paths.py + 工区选择对话框 |

---

## 11. 代码风格与工程规约

### 11.1 风格

- 4 空格缩进，全 type hint（pyright strict 模式）。
- 不允许出现 wildcard import（`from foo import *`）。
- 不允许循环 import；如果两个模块互相依赖，引入第三个模块或用 `TYPE_CHECKING`。
- docstring 用 Google 风格，公开类必填。
- 注释只写 WHY，不写 WHAT。

### 11.2 测试

- pytest，最小覆盖：
  - `data/` 100%（coord_transform / volume_store 必测）
  - `scene/layer_store.py` 100%
  - `algorithms/builtin/` 中每个一期算法都要有单测
  - `tools/` 至少 smoke test
- UI 不强求单测，但 main_window 启动要有 smoke test。

### 11.3 日志

- 用标准 `logging`，不要 print。
- 每个模块 `logger = logging.getLogger(__name__)`。
- 日志级别：DEBUG（开发）、INFO（默认）、WARNING、ERROR。
- 日志文件：`~/.yj_studio/logs/yj_studio.log`，rotate 每 10MB。

### 11.4 错误处理

- I/O 层抛具体异常类（`VolumeFileMissingError`、`IncompatibleShapeError` 等）。
- Service 层捕获 I/O 异常，转译为用户消息（i18n），通过状态栏或 dialog 提示。
- UI 永远不要 raw exception 弹给用户。

### 11.5 国际化

- 一期只做中文，但所有用户可见字符串走 `tr("...")`。
- 未来如需英文版，加 Qt Linguist `.ts` 文件即可。

### 11.6 Git

- 主分支 `main` 永远可发布。
- 功能分支 `feat/phase-x-yyy`，PR 合并。
- commit 信息中文/英文均可，但同一项目内统一。
- 不在 commit 中 dump 大文件。

---

## 12. 遗留资产复用清单

### 12.1 完全复用（拷过去即用，不要改）

| 来源 | 目的地 | 原因 |
|---|---|---|
| [可视化文件/cigvis/](../可视化文件/cigvis/) 全部 | `libs/cigvis/` | 自定义 colormap、出图能力 |
| [可视化文件/代码/well_section/](../可视化文件/代码/well_section/) | `libs/well_section/` | 连井剖面 HTML 生成 |
| [可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py:54-124](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) PALETTE / LITH_STYLE / LITH_BODY_STYLE / VOLUME_DISPLAY_STYLE / MODEL_VOLUME_DISPLAY_STYLE | `config/styles.py` | 颜色与样式约定 |
| [run_cigvis_web_with_por_perm_lith_wells.py:336-353](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) load_layers | `io/readers/layers_npz.py` | 层位加载 |
| [run_cigvis_web_with_por_perm_lith_wells.py:366-426](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) load_fault_meshes | `io/readers/fault_mesh.py` | 断层加载（移除 SurfaceNode 拼接，改为返回纯 numpy）|
| [run_cigvis_web_with_por_perm_lith_wells.py:429-505](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) load_attribute_logs | `io/readers/well_logs.py` | POR/LITH/PERM csv 加载 |
| [run_cigvis_web_with_por_perm_lith_wells.py:508-558](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) load_model_point_clouds | `io/readers/grdecl_model.py` | GRDECL 模型点云 |
| [run_cigvis_web_with_por_perm_lith_wells.py:561-595](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) load_lithology_body_meshes | `io/readers/lith_body.py` | 透明岩性体 |
| [run_cigvis_web_with_por_perm_lith_wells.py:296-322](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) estimate_clim / estimate_volume_clim | `data/attribute_cache.py` | clim 估计逻辑 |
| [run_cigvis_web_with_por_perm_lith_wells.py:681-713](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_wells.py) load_available_volume_specs / load_volume_by_key | `data/volume_store.py` | 体数据注册表 |
| [run_cigvis_web_with_por_perm_lith_well_desktop.py:57-58](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py) Z_WINDOW_START / DEPTH_STEP_TO_SAMPLE | `config/defaults.py` | 深度坐标关键常数 |

### 12.2 借鉴思路重写（结构换，逻辑保留）

| 原型代码 | 重写位置 | 主要变化 |
|---|---|---|
| `DesktopVisCanvas` ([desktop:659](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py)) | `view/qt_vtk_view.py` + `view_3d.py` | VisPy → VTK；KeyPress 改 ToolManager |
| `DesktopControlWindow` ([desktop:790](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py)) | 拆 ~10 个 docks | 单类 → 多 dock |
| `_pick_well_from_screen_projection` ([desktop:759](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py)) | `view/picker.py` | 屏幕投影 → VTK Picker，距离 fallback 保留 |
| `update_slice_volume_desktop` 等更新函数 ([desktop:628-651](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py)) | `view/renderers/volume_slice_renderer.py` | 函数 → 方法 |
| `create_desktop_well_name_labels` ([desktop:563](../可视化文件/代码/run_cigvis_web_with_por_perm_lith_well_desktop.py)) | `view/renderers/well_renderer.py` | VisPy Text → VTK BillboardTextActor3D |
| `set_node_group_visible` / `set_node_group_name` 等组管理 | LayerStore.update + Renderer 自动响应 | 函数 → 信号驱动 |
| `swap_slice_volume` (viserplot.py 与 desktop) | `volume_slice_renderer.swap_volume` | 同 |
| viser GUI add_folder / on_update 回调 | Qt dock + signal/slot | viser → PyQt6 |

### 12.3 弃用（不再使用）

- `argparse` 大量参数（~700 行）→ defaults.py + Project + 设置面板
- viser server + `while True: sleep(0.1)` 主循环 → Qt 事件循环
- `_prefer_local_cigvis()` 的 sys.path hack → `pip install -e libs/cigvis`
- `print(...)` 1000+ 行 → logging + 状态栏 + log_dock
- `SurfaceNode` viser 节点装配 → VTK Actor

---

## 13. 附录

### 13.1 关键术语

| 术语 | 含义 |
|---|---|
| **Layer** | 解释对象的数据模型（不含 VTK 渲染对象）|
| **Renderer** | 把 Layer 转成 VTK Actor 的适配器 |
| **Tool** | 响应鼠标/键盘的交互工具 |
| **Algorithm** | 输入 Layer + 参数 → 输出 Layer 的可插拔单元 |
| **Project** | 工程文件，保存一次完整解释会话 |
| **Session** | 当前 Project 的运行时状态 |
| **Picker** | 把屏幕点转换为 world coord 或 layer/cell id |
| **SyncTopic** | ViewSyncService 中的一个联动话题 |
| **Provenance** | Layer 的来源标签（手动/自动/AI） |

### 13.2 关键常量（来自原型，必须搬迁）

| 常量 | 值 | 来源 | 用途 |
|---|---|---|---|
| `Z_WINDOW_START` | 0.0 / 150.0 | 两份脚本不同 | 坐标变换基准 |
| `DEPTH_STEP_TO_SAMPLE` | 10.0 | 两份 | depth_m → sample 转换 |
| `POR_COLUMN` | `"POR_shalizhuojiyanxiangkong-20221217-fupinbi-chouxi"` | 两份 | POR csv 列名 |
| `PALETTE` | 15 色 RGBA 元组 | 两份 | 层位/断层默认颜色 |
| `LITH_STYLE` | coarse/fine/raw 三套岩性 cmap | 两份 | 岩性显示 |
| `LITH_BODY_STYLE` | 砾/砂/泥三类 | web 脚本 | 透明岩性体 |
| `VOLUME_DISPLAY_STYLE` | seismic/coherence/dip/azimuth/curvature 五属性 | 两份 | 体数据显示样式 |
| `MODEL_VOLUME_DISPLAY_STYLE` | model_lithology/model_porosity 两体 | web 脚本 | 模型体显示 |

### 13.3 关键路径（固定工区）

| 路径 | 内容 |
|---|---|
| `F:\YJ-ALL-SEISMIC_depth_0_653.npy` | 主地震体 0–653 全深度 |
| `F:\YJ-ALL-SEISMIC_depth_0_653_processed/地震属性/` | coherence / dip / azimuth / curvature 属性体 |
| `F:\YJ-ALL-SEISMIC_depth_0_653_processed/层位/` | `*.npz` 层位 |
| `F:\YJ-ALL-SEISMIC_depth_0_653_processed/断层/` | `*_mesh.npz` |
| `F:\YJ-ALL-SEISMIC_depth_0_653_processed/测井坐标/` | `combined_well_coordinates_*.csv` |
| `F:\YJ-ALL-SEISMIC_depth_0_653_processed/{por,lith,perm}/` | 测井 csv |
| `F:\YJ-LITH-POR_model_numpy/` | GRDECL 模型 .npy + 岩性体 mesh |

**做法**：用 NTFS junction link 把这些路径软链到 `processed/`，使代码用相对路径即可工作，便于开发环境与演示环境切换。

### 13.4 时间预算汇总

| Phase | 周数 | 累计 |
|---|---|---|
| 0 脚手架 | 1 | 1 |
| 1 数据 + 三正交 | 2 | 3 |
| 2 解释对象骨架 | 2 | 5 |
| 3 井 + 层位 + 断层 | 2 | 7 |
| 4 2D 剖面 + 联动 | 2 | 9 |
| 5 交互工具 | 2.5 | 11.5 |
| 6 任意剖面 + 构造高定位 | 2 | 13.5 |
| 7 Project + Export | 1.5 | 15 |
| 8 算法插件框架 | 2 | 17 |
| 9 SAM3 集成 | 3 | 20 |
| 10 打包 + 文档 + 演示 | 2 | **22** |

二期与远期另计，估约 6 个月 + 6 个月。

### 13.5 单元测试关键路径

最先要覆盖的单测：

1. `data/coord_transform.py`：所有 ijk↔inline/xline↔depth 转换的边界情况
2. `data/volume_store.py`：load / get_slice / swap / LRU eviction
3. `scene/layer_store.py`：add/remove/update + 信号发射次数
4. `scene/undo_commands.py`：每个 Command 的 redo/undo 对称性
5. `algorithms/builtin/thickness.py`：用真实两个层位 + 已知正确答案
6. `io/readers/*`：能 load 现有真实文件，不出错

### 13.6 一期完成后的能力图谱

完成一期 22 周开发后，YJ Studio 应具备：

- 工区数据一键加载（一-1,2,3,4 完整）
- 任意方向/折线/沿井/沿层剖面（一-5,6,7,8）
- 构造高点快速定位（一-10）
- 剖面与解释对象同步联动（一-12, 六-18）
- 层位手动拾取与显示，构造图，厚度（二-1,2,4,5）
- 断层手动拾取与显示（三-1,2）
- 井位/轨迹/井名/井旁剖面/连井剖面/岩性柱/孔隙度曲线（四-1~7 完整）
- 岩性体三维显示与分类（五-1,2）
- 点选/框选/多边形/画笔/橡皮/合并/拆分/重命名/颜色/高亮/同步（六-1~5,12~18，9 个核心工具完整）
- SAM3 切片分割、点/框/文本/多提示、跨切片传播、2D→3D、置信度、人工修正、重推（七-1~5,12~14,16,17,18,20）
- 工程文件保存/加载（`.yjproj`）
- 截图/视频/GeoJSON/Mask 导出
- 完整的算法插件框架，二期填算法不动主软件
- 完整的工具调色板，二期填工具不动主软件
- PyInstaller 打包的安装包 + NSIS 安装程序 + 中英文用户手册 + 演示视频

剩余预留接口（二期/远期），按算法插件 + 工具 stub 填充即可，**不需要重构主架构**。

---

**文档结束。**
