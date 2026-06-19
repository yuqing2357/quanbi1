# 旧项目高信号地图

本文件记录对 `D:\商书记项目` 的实施前侦察结果。旧项目保持只读，新实现只写入
`F:\圈闭软件`。

## 顶层结构

- `处理后文件/`：当前窗口版地震、属性、层位、断层、井坐标、井曲线和岩性数据。
- `原始文件/`：LAS、原始层位、断层、井坐标等输入资料。
- `可视化文件/`：旧版 cigvis 可视化代码、本地 `cigvis` 副本、`well_section` 连井剖面模块。
- `docs/implementation_plan.md`：新桌面软件实施方案，按 Phase 0 到 Phase 10 推进。
- `tools/`：GRDECL/岩性体/全深度处理相关数据预处理脚本。
- `sam3/`、`dinov3-main/`、`EUPE/`：AI/视觉模型源码或第三方研究代码，当前阶段不直接导入主软件。

## 已迁移到当前工作区的只读资产副本

- `libs/cigvis/`：来自 `可视化文件/cigvis/`，作为 vendored 可视化/colormap 资产。
- `libs/well_section/`：来自 `可视化文件/代码/well_section/`，后续用于连井剖面。
- `legacy/run_cigvis_web_with_por_perm_lith_wells.py`：Web 原型，提取样式、读取器、坐标逻辑。
- `legacy/run_cigvis_web_with_por_perm_lith_well_desktop.py`：桌面原型，提取交互思路和窗口常量差异。
- `docs/implementation_plan.md`：实施方案副本。

## Phase 0 实现边界

- 建立 `apps/yj_studio` Python 包与空白 PyQt6 主窗口。
- 定义 Layer、LayerStore、InteractionTool、Algorithm、Picker、ViewSyncService 等核心契约。
- 搬入样式常量、路径常量和最小数据读取逻辑。
- 使用 `py312` 验证 import、单测和无界面 smoke test。
