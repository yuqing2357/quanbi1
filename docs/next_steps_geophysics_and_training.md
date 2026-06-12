# 下一步详细计划 · 圈闭算法链 + 训练闭环（可直接照此写代码）

本文件承接 [`next_steps_detailed_implementation.md`](next_steps_detailed_implementation.md)（步骤 4–9，SAM3/目标侧大半已完成），补上**两条尚未被任何文档覆盖、且是科研主线（圈闭检测 + 标注↔训练闭环）最核心的缺口**：

- **Phase A**：地球物理圈闭算法链落地（当前全是空壳 stub）。
- **Phase B**：训练闭环收尾（导出/激活已有，缺真实微调脚本与评估/回滚）。
- **Phase C**：储层工作台主窗口入口接线（已有类，缺入口）。
- **Phase D**：真多卡（详见旧文档步骤 6，本文不重复）。

> 环境与约束（沿用旧文档）：服务器启动/重启/训练/验证由用户手动执行；本机 py312 **无 fastapi**，凡纯算法核心都要能脱离 FastAPI/GPU 单测。
>
> 已确认可用依赖（py312）：`numpy`、`scipy`、`scipy.ndimage`、`skimage.measure`、`shapely`。`cv2` 不可用，方案中不使用。
>
> 算法框架接缝（已存在，直接用）：
> - 基类与契约：[`algorithms/algorithm.py`](../apps/yj_studio/src/yj_studio/algorithms/algorithm.py) `Algorithm.run(ctx)`；返回 [`result.py`](../apps/yj_studio/src/yj_studio/algorithms/result.py) `AlgorithmResult.success(output_layers=[...], summary=...)` / `.failure(msg)`。
> - 上下文：[`context.py`](../apps/yj_studio/src/yj_studio/algorithms/context.py) `ctx.input_layers` / `ctx.params` / `ctx.report_progress(frac, msg)` / `ctx.check_cancel()`。
> - 注册：`@register_algorithm` 装饰器（side-effect 注册到 registry）。
> - 既有完整范例：[`builtin/thickness.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/thickness.py)（读 HorizonLayer.sample → 出 MeasurementLayer）。
> - 现有占位：[`builtin/stubs/`](../apps/yj_studio/src/yj_studio/algorithms/builtin/stubs)，本 Phase 把它们从 `PhaseTwoStub` 升级为真实现并迁出 `stubs/`。
> - 可用输出层：`TrapLayer`(boundary/score/attributes)、`PolygonLayer`、`LithBodyLayer`、`MeasurementLayer`、`MaskLayer`，见 [`scene/layers/`](../apps/yj_studio/src/yj_studio/scene/layers)。

数据约定（来自 thickness.py 实证）：`HorizonLayer.sample` 是 `(ni, nx)` 的二维数组，值=Z 采样索引，**索引越小越浅**；`nan` 表示无数据。`HorizonLayer.mask` 可选有效域。

---

## Phase A · 地球物理圈闭算法链（P0，最高优先）

依赖链：**A1 闭合等值线** 是地基；A2 圈闭检测复用 A1；A3 连通性是体级地基；A4 砂体提取复用 A3；A5 圈闭评价消费 A1/A2 的结果。建议严格按 A1→A2→A3→A4→A5 推进。每个都纯 CPU、可脱机单测。

### A1 · 闭合等值线 `horizon.closure_contour`

**状态（2026-06-12）**：代码与单元测试已完成。新增真实算法
`apps/yj_studio/src/yj_studio/algorithms/builtin/closure_contour.py`，旧
`stubs/closure_contour.py` 已移除注册；`TrapLayer` 已接入 2D/3D manual
geometry 渲染链。已通过：
`cd apps/yj_studio; PYTHONPATH=src; E:\miniconda\envs\py312\python.exe -m pytest tests/test_closure_contour.py tests/test_algorithms_registry.py -q`。
仍待用户在界面中用真实层位手动验收。

把 [`stubs/closure_contour.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/stubs/closure_contour.py) 升级为真实现，迁到 `builtin/closure_contour.py`。

**算法（涨水/溢出点法，纯 numpy + scipy.ndimage + skimage）**：自浅向深逐层抬升「水位」`level`，`region = valid & (z <= level)` 取比该深度浅的区域；对 region 做连通域标注。某连通域若 **不接触网格边界** 且 **只含一个构造高点**，它就是该高点当前的闭合域；继续加深直到它「接触边界」或「与另一个高点的域合并」——上一刻的域即该高点的**最大闭合**（溢出点对应水位）。

```python
# apps/yj_studio/src/yj_studio/algorithms/builtin/closure_contour.py
from __future__ import annotations
from typing import ClassVar
import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import label, minimum_filter
from skimage.measure import find_contours

from yj_studio.algorithms.algorithm import Algorithm
from yj_studio.algorithms.context import AlgorithmContext
from yj_studio.algorithms.registry import register_algorithm
from yj_studio.algorithms.result import AlgorithmResult
from yj_studio.scene.layers import HorizonLayer, TrapLayer


class ClosureContourParams(BaseModel):
    z_step_m: float = Field(default=1.0, gt=0.0, description="每采样对应米数，用于报告闭合高度。")
    level_step: float = Field(default=1.0, gt=0.0, description="水位抬升步长（采样单位）。")
    min_relief_samples: float = Field(default=2.0, ge=0.0, description="最小闭合高度（采样），低于此丢弃。")
    min_area_cells: int = Field(default=8, ge=1, description="最小闭合面积（网格点数）。")
    max_highs: int = Field(default=50, ge=1)


def _structural_highs(z: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """局部极小（最浅点）作为候选构造高点。"""
    filt = minimum_filter(np.where(valid, z, np.inf), size=3, mode="nearest")
    is_min = valid & (z <= filt)
    rows, cols = np.nonzero(is_min)
    return list(zip(rows.tolist(), cols.tolist()))


@register_algorithm
class ClosureContourAlgorithm(Algorithm):
    id: ClassVar[str] = "horizon.closure_contour"
    category: ClassVar[str] = "horizon"
    label: ClassVar[str] = "闭合等值线"
    description: ClassVar[str] = "在层位上用涨水法寻找四面闭合的构造高，输出闭合多边形 TrapLayer。"
    input_schema: ClassVar[type[BaseModel]] = ClosureContourParams
    layer_inputs: ClassVar[dict[str, str]] = {"horizon": "horizon"}

    def run(self, ctx: AlgorithmContext) -> AlgorithmResult:
        layer = ctx.input_layers.get("horizon")
        if not isinstance(layer, HorizonLayer) or layer.sample is None:
            return AlgorithmResult.failure("缺少层位样点数据")
        z = np.asarray(layer.sample, dtype=np.float32)
        valid = np.isfinite(z)
        if layer.mask is not None:
            valid &= np.asarray(layer.mask, dtype=bool)
        if not valid.any():
            return AlgorithmResult.failure("层位没有有效样点")

        ctx.report_progress(0.1, "定位构造高点")
        highs = _structural_highs(z, valid)[: ctx.params.max_highs]
        if not highs:
            return AlgorithmResult.failure("未找到构造高点")

        zmin = float(np.nanmin(z[valid]))
        zmax = float(np.nanmax(z[valid]))
        levels = np.arange(zmin + ctx.params.level_step, zmax, ctx.params.level_step)
        # high -> (closure_level, component_mask)，随水位加深不断更新到溢出前一刻
        closure: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
        spilled: set[tuple[int, int]] = set()

        for li, level in enumerate(levels):
            ctx.check_cancel()
            ctx.report_progress(0.1 + 0.6 * li / max(len(levels), 1), "抬升水位")
            region = valid & (z <= float(level))
            lab, n = label(region)
            for comp_id in range(1, n + 1):
                comp = lab == comp_id
                inside = [h for h in highs if comp[h] and h not in spilled]
                if not inside:
                    continue
                touches = comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any()
                if touches or len(inside) > 1:
                    # 溢出：这些高点定格在上一刻的闭合（若有）
                    spilled.update(inside)
                    continue
                closure[inside[0]] = (float(level), comp)

        ctx.report_progress(0.8, "构建闭合多边形")
        traps: list[TrapLayer] = []
        for (hi, hj), (level, comp) in closure.items():
            area = int(comp.sum())
            relief = float(level) - float(z[hi, hj])
            if area < ctx.params.min_area_cells or relief < ctx.params.min_relief_samples:
                continue
            contours = find_contours(comp.astype(np.float32), 0.5)
            if not contours:
                continue
            ring = max(contours, key=len)  # (M,2)=(row,col)=(inline,xline)
            boundary = np.column_stack([ring[:, 0], ring[:, 1], np.full(len(ring), level)]).astype(np.float32)
            traps.append(TrapLayer(
                name=f"闭合@({hi},{hj})",
                boundary=boundary,
                score=float(min(1.0, relief / max(zmax - zmin, 1e-6))),
                attributes={
                    "high_inline": hi, "high_xline": hj,
                    "spill_level": level, "relief_samples": relief,
                    "relief_m": relief * float(ctx.params.z_step_m),
                    "area_cells": area, "source_horizon": layer.name,
                },
                metadata={"algorithm": self.id, "source_horizon_id": layer.id},
            ))
        if not traps:
            return AlgorithmResult.failure("没有满足阈值的闭合")
        ctx.report_progress(1.0, "完成")
        return AlgorithmResult.success(
            output_layers=traps,
            summary=f"闭合等值线：{len(traps)} 个闭合，最大闭合高度 "
                    f"{max(t.attributes['relief_m'] for t in traps):.1f} m",
        )
```

**测试**（`apps/yj_studio/tests/test_closure_contour.py`，纯 CPU）：
- 合成单高斯洼地（一个明显高点）→ 恰好 1 个 TrapLayer，relief>0，boundary 闭环（首尾接近）。
- 两个被深鞍部隔开的高点 → 2 个闭合；把鞍部填浅到能连通 → 合并后两高点都 `spilled`，闭合数减少。
- 单调斜坡（无闭合）→ `.failure("没有满足阈值的闭合")`。
- `min_relief_samples`/`min_area_cells` 提高 → 小闭合被过滤。

**DoD**：给定层位能稳定圈出四面闭合构造高，输出带 relief/area/score 的 TrapLayer，可在 3D/2D 叠加显示。

---

### A2 · 圈闭检测 `trap.detect_structural`

**状态（2026-06-12）**：代码与单元测试已完成。新增真实算法
`apps/yj_studio/src/yj_studio/algorithms/builtin/trap_detect.py`，复用 A1
`detect_closures()`，输出带 `rank/candidate_score/relief/area` 的 `TrapLayer`；
旧 `stubs/trap_detect.py` 已移除注册。已通过：
`cd apps/yj_studio; PYTHONPATH=src; E:\miniconda\envs\py312\python.exe -m pytest tests/test_closure_contour.py tests/test_trap_detect.py tests/test_algorithms_registry.py -q`。
仍待用户在真实层位上 UI 手动验收。

把 [`stubs/trap_detect.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/stubs/trap_detect.py) 升级。**v1 = 复用 A1 的 `detect_closures`** + 综合评分排序 + 阈值过滤；断层封堵的三面闭合留 v2。

**Params（输入）**

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `structural_only` | bool | True | v1 只做四面构造闭合；False 触发断层封堵（v2，先 NotImplemented 提示） |
| `score_threshold` | float∈[0,1] | 0.4 | 综合分阈值，低于此不输出 |
| `z_step_m` | float>0 | 1.0 | relief 换算米 |
| `min_relief_m` | float≥0 | 10.0 | 最小闭合高度（米），硬淘汰 |
| `area_weight` | float∈[0,1] | 0.3 | 综合分里面积权重（其余给 relief） |

`layer_inputs = {"horizon":"horizon", "faults":"fault?"}`（faults 可选，v2 用）；`runs_in_subprocess = False`。

**关键方法与输入输出**
- 复用 A1：`closures = detect_closures(z, valid, closure_params)`（把 A2 的 z_step/relief 阈值传入；A1/A2 共用同一纯函数，避免重复实现）。
- `rank_closures(closures, *, area_weight) -> list[tuple[ClosureResult, float, int]]`
  - **输入**：A1 的 `ClosureResult` 列表 + 面积权重。
  - **输出**：`(closure, score, rank)` 列表，按 score 降序、rank 从 1 起。
  - **算法**：`norm_relief = relief / max_relief`、`norm_area = area / max_area`（在本批内归一化）；`score = (1-area_weight)*norm_relief + area_weight*norm_area`。
- `_fault_bounded_closures(horizon, faults, params) -> list[...]`（v2 占位）：`structural_only=False` 时调用，当前 `raise NotImplementedError("断层封堵圈闭为 v2 功能")` 并在 run 里转成 `.failure` 友好提示。
- `run(ctx)`：
  - **输入**：`ctx.input_layers["horizon"]`（HorizonLayer），可选 `["faults"]`；`ctx.params`。
  - **输出**：`AlgorithmResult.success(output_layers=[TrapLayer...], summary)`。每个 TrapLayer 在 A1 的基础上 `attributes["rank"]`、名字 `Trap-1/2/...`；过滤 `relief_m >= min_relief_m and score >= score_threshold`。

**测试**：同一合成数据，A2 输出数 == A1 中 `score>=threshold` 的数；调 `area_weight` 改变排名顺序但不改数量；`min_relief_m` 提高 → 数量下降；rank 严格单调 1..N。

**DoD**：一键从层位得到带排名/置信度的候选圈闭列表，参数可调。

---

### A3 · 连通性 `volume.connectivity`（体级地基）

**状态（2026-06-12）**：代码与单元测试已完成。新增真实算法
`apps/yj_studio/src/yj_studio/algorithms/builtin/connectivity.py`，旧
`stubs/connectivity.py` 已移除注册；算法 id 采用本文约定的
`volume.connectivity`。核心函数 `detect_bodies()` 可被 A4 复用，输出
`BodyResult(label_id, voxel_count, bbox, centroid, cells?)`；算法面板输出
`LithBodyLayer` 的可见 bbox 体对象，避免把大体素列表内联进工程 JSON。
已通过：
`cd apps/yj_studio; PYTHONPATH=src; E:\miniconda\envs\py312\python.exe -m pytest tests/test_connectivity.py tests/test_algorithms_registry.py -q`。
仍待用户在真实体数据上手动验收阈值与连通性参数。

把 [`stubs/connectivity.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/stubs/connectivity.py) 升级。输入 `VolumeLayer`（或 `MaskLayer`）+ 阈值，做 3D 连通域标注。

**Params（输入）**

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `threshold` | float | 0.5 | 二值化阈值 |
| `comparator` | str | `">="` | 比较符（`>=,<=,>,<,==`） |
| `connectivity` | int∈[1,3] | 1 | scipy 邻接：1=面(6邻)，3=全(26邻) |
| `min_voxels` | int≥1 | 64 | 最小体素数，过滤碎块 |
| `top_k` | int≥1 | 20 | 最多输出几个体 |

`layer_inputs = {"volume":"volume"}`；`runs_in_subprocess = False`（要 `ctx.services["volume_store"]`）。

**关键方法与输入输出**
- `detect_bodies(binary, *, connectivity, min_voxels, top_k) -> list[BodyResult]`（核心纯函数，A4 也复用）
  - **输入**：`binary: np.ndarray (ni,nx,nz) bool`。
  - **输出**：`list[BodyResult]`，按体素数降序、截断 top_k。`BodyResult` 建议 dataclass：

    | 字段 | 类型/形状 | 含义 |
    |---|---|---|
    | `label_id` | int | 连通域标号 |
    | `voxel_count` | int | 体素数 |
    | `bbox` | `tuple[int,int,int,int,int,int]` | `(i0,i1,j0,j1,k0,k1)` |
    | `centroid` | `tuple[float,float,float]` | 质心 (i,j,k) |
    | `cells` | `np.ndarray (N,3) int` 或 `mask: np.ndarray bool` | 体素索引或布尔体（择一，量大用 bbox+mask） |

  - **算法**：`scipy.ndimage.label(binary, structure=generate_binary_structure(3, connectivity))` → `np.bincount(lab.ravel())` 取尺寸 → 过滤 `>=min_voxels` → 取前 top_k → 每个算 bbox/centroid。
- `run(ctx)`：
  - **输入**：`vol = ctx.services["volume_store"].get_volume(volume_id)`（整卷 memmap，取 volume_id 自 input layer）；`binary = comparator(vol, threshold)`。
  - **输出**：每个 `BodyResult` → 一个 `LithBodyLayer`（或 `MaskLayer` 摘要），cells/bbox 写缓存 `.npy` 避免大 JSON；`summary="连通域 N 个，最大 M 体素"`。
  - **内存**：大体注意 memmap，必要时按 z 分块标注后合并（先不优化，能跑为先）。

**测试**（脱机测 `detect_bodies`）：合成体放 2 个分离立方块 → `len==2`、voxel_count 正确；`min_voxels` 过滤小块；`connectivity=1 vs 3` 改变对角是否连通。

**DoD**：体级二值连通分析可用，`detect_bodies` 可被 A4 直接复用。

---

### A4 · 砂体提取 `reservoir.sandbody_extract`

**状态（2026-06-12）**：代码与单元测试已完成。新增真实算法
`apps/yj_studio/src/yj_studio/algorithms/builtin/sandbody_extract.py`，旧
`stubs/sandbody_extract.py` 已移除注册。实现了孔隙度阈值路径与岩性码路径，
复用 A3 `detect_bodies()`，输出 `LithBodyLayer`，并在 metadata 中记录
`voxel_count / volume_m3 / mean_porosity / bbox / centroid`。已通过：
`cd apps/yj_studio; PYTHONPATH=src; E:\miniconda\envs\py312\python.exe -m pytest tests/test_sandbody_extract.py tests/test_connectivity.py -q`。
仍待用户在真实孔隙度/岩性体上手动验收 cutoff 与最小体素数。

把 [`stubs/sandbody_extract.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/stubs/sandbody_extract.py) 升级。本质 = **属性阈值 + A3 的 `detect_bodies`**，针对孔隙度/岩性体。

**Params（输入）**

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `porosity_cutoff` | float∈[0,1] | 0.1 | 孔隙度下限（`use_lithology=False` 时用） |
| `use_lithology` | bool | False | True 改用 `lithology==sand_code` 判砂 |
| `sand_code` | int | 1 | 砂岩岩性码 |
| `min_voxels` | int≥1 | 64 | 最小砂体体素数 |
| `top_k` | int≥1 | 20 | 最多输出几个砂体 |
| `cell_volume_m3` | float>0 | 1.0 | 单体素体积（=dx·dy·dz），算砂体体积用 |

`layer_inputs = {"porosity":"volume", "lithology":"volume?"}`；`runs_in_subprocess = False`。

**关键方法与输入输出**
- 二值化：`binary = (lith==sand_code)` 若 `use_lithology` 否则 `(por >= porosity_cutoff)`。`por`/`lith` 来自 `ctx.services["volume_store"].get_volume(...)`。
- 复用 A3：`bodies = detect_bodies(binary, connectivity=1, min_voxels=..., top_k=...)`。
- `summarize_body(body, por, cell_volume_m3) -> dict`
  - **输入**：一个 `BodyResult` + 孔隙度体 + 单体素体积。
  - **输出**：`{volume_m3 = voxel_count*cell_volume_m3, mean_porosity = por[body.cells].mean(), bbox, centroid}`。
- `run(ctx)`：每个砂体 → 一个 `LithBodyLayer`（带上面 dict 入 `attributes`），按 `volume_m3` 降序命名 `Sand-1..`；`summary="砂体 N 个，总有效体积 V m³，最大平均孔隙度 0.xx"`。

**测试**：合成孔隙度体（高孔块+背景）→ 提取出高孔块、`volume_m3` 与体素数×cell_volume 吻合；切 `use_lithology` 走岩性码路径；`porosity_cutoff` 改变体数。

**DoD**：从孔隙度/岩性体一键得到排序砂体列表，带体积与平均孔隙度。

---

### A5 · 圈闭评价 `trap.evaluate`（储量雏形）

**状态（2026-06-12）**：代码与单元测试已完成。新增真实算法
`apps/yj_studio/src/yj_studio/algorithms/builtin/trap_evaluate.py`，旧
`stubs/trap_evaluate.py` 已移除注册。实现 `rasterize_closure()` 与
`volumetrics()`，输出 `MeasurementLayer`，包含
`GRV / HCPV / STOIIP / area_km2 / mean_phi / gross_thickness_mean_m / cell_count`。
孔隙度体为可选输入；未提供时使用参数 `default_porosity`。已通过：
`cd apps/yj_studio; PYTHONPATH=src; E:\miniconda\envs\py312\python.exe -m pytest tests/test_trap_evaluate.py tests/test_algorithms_registry.py -q`。
仍待用户基于真实测网面积、时深关系与孔隙度体做地质参数校核。

把 [`stubs/trap_evaluate.py`](../apps/yj_studio/src/yj_studio/algorithms/builtin/stubs/trap_evaluate.py) 升级。输入 **TrapLayer + 顶/底层位 + 孔隙度体**，算容积法储量。

**Params（输入）**

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `net_to_gross` | float∈[0,1] | 0.6 | 净毛比 NTG |
| `water_saturation` | float∈[0,1] | 0.3 | 含水饱和度 Sw |
| `cell_area_m2` | float>0 | 625.0 | 单网格平面面积（如 25m×25m），见 G1.5 测网几何 |
| `z_step_m` | float>0 | 1.0 | 采样→米 |
| `formation_volume_factor` | float>0 | 1.1 | 地层体积系数 Bo |

`layer_inputs = {"trap":"trap", "top":"horizon", "bottom":"horizon", "porosity":"volume?"}`；`runs_in_subprocess = False`。

**关键方法与输入输出**
- `rasterize_closure(boundary_ij, grid_shape) -> tuple[np.ndarray, np.ndarray]`
  - **输入**：TrapLayer 的 `boundary[:, :2]`（map 平面 inline,xline）+ 层位网格形状 `(ni,nx)`。
  - **输出**：闭合内的 `(rows, cols)` 索引（`skimage.draw.polygon(boundary_ij[:,0], boundary_ij[:,1], shape)`，无需 shapely，更快）。
- `volumetrics(rows, cols, top, bottom, por, params) -> dict`
  - **输入**：闭合内格点 + 顶/底层位 sample + 孔隙度体 + Params。
  - **输出**：`{GRV, HCPV, STOIIP, area_km2, mean_phi, gross_thickness_mean_m, cell_count}`。
  - **公式**：`gross_thk = (bottom.sample - top.sample)[rows,cols] * z_step_m`（取 spill 以上）；`GRV = sum(cell_area_m2 * gross_thk)`；`mean_phi = por 在闭合内高度区间的均值`；`HCPV = GRV * NTG * mean_phi * (1-Sw)`；`STOIIP = HCPV / Bo`；`area_km2 = cell_count*cell_area_m2/1e6`。
- `run(ctx)`：对输入 TrapLayer 调上面两步 → 出一个 `MeasurementLayer`（values=上面 dict，units 标 m³/km²/ratio）；`summary="圈闭储量：GRV=… HCPV=… 含油面积=… km²"`。

**测试**（脱机测 `rasterize_closure`+`volumetrics`）：矩形闭合 + 常数厚度 + 常数孔隙度 → 手算 GRV/HCPV 与算法吻合（误差<1%）；`Sw`/`NTG` 对 HCPV 线性；`rasterize_closure` 对矩形返回正确格点数。

**DoD**：选中一个圈闭即可给出容积法储量估算，参数透明可调。

---

### Phase A 落地清单（迁移与接线）

1. 每个算法从 `builtin/stubs/` 迁到 `builtin/`（或保留文件位置但去掉 `PhaseTwoStub` 继承，改继承 `Algorithm`）。`builtin/__init__.py` / `stubs/__init__.py` 的导入相应调整。A1/A2/A3/A4/A5 已完成。
2. 抽纯函数：`detect_closures()`(A1)、`detect_bodies()`(A3)、`rasterize_closure()`/`volumetrics()`(A5) 供复用与脱机单测。已完成。
3. `test_tools.py` 之类的「算法数量」类断言若存在，需同步（本次新增的是真实算法，不改 tool 目录）。
4. UI 无需改：算法面板 [`AlgorithmDock`](../apps/yj_studio/src/yj_studio/ui/docks/algorithm_dock.py) 自动列出注册算法；输出 TrapLayer/LithBodyLayer 已有渲染器即可显示（若 TrapLayer 还没 3D/2D 渲染器，补一个轮廓渲染，参照 PolygonLayer 渲染）。

---

## Phase B · 训练闭环收尾（P0）

现状（旧文档 §8.2）：`export_confirmed_to_coco` 带 train/val/test 划分；`_run_train_job` 能跑配置式 `training.command` 并登记 checkpoint；`activate` 会 `reload_checkpoint`。**缺真实微调脚本、评估、回滚 UI。**

### B1 · 微调脚本契约（先把接口定死）
约定 `training.command` 调用一个脚本，**输入/输出走目录约定**，与 server 解耦：
```text
输入: --dataset <results_root>/sam3/datasets/<project>/<dataset_version>/   (COCO + masks/)
      --base-checkpoint <weights/...>            # 起点权重
      --output <results_root>/sam3/training_runs/<project>/<dataset_version>/
输出: <output>/checkpoint.pt                     # 微调后权重
      <output>/metrics.json                      # {"mAP":..,"mask_iou":..,"epochs":..}
```
`_run_train_job` 已采集 `metrics.json`/checkpoint → 写 `ModelRegistry.add_model(metrics=...)`。脚本本体放 `libs/sam3/train/`（已有 `transforms/` 等），写一个最小 LoRA/decoder-only 微调入口即可起步。

### B2 · 评估集与指标
**状态（2026-06-12）**：G3.2 空间分块划分核心已完成。`export_confirmed_to_coco()`
默认使用 `split_strategy="spatial"`，按 `axis + index` 的连续空间块划分
train/val/test，避免相邻剖面随机泄漏；保留 `split_strategy="round_robin"`
用于旧行为兼容。新增 `split_frames()` 纯函数，已通过：
`cd apps/yj_studio; PYTHONPATH=src; E:\miniconda\envs\py312\python.exe -m pytest tests/test_targets_store.py::test_split_frames_spatial_uses_contiguous_index_blocks tests/test_targets_store.py::test_export_confirmed_targets_uses_spatial_split -q`。
仍待：真实 SAM3 微调脚本在 val/test 上计算 mask IoU/mAP 并写入 `metrics.json`。

- 导出时按 split 写 `val`/`test`；微调脚本在 `val` 上算 mask IoU/mAP 写进 `metrics.json`。
- `ModelRegistry` 每个模型存 `metrics` + `parent_model_id`（形成版本链）。**状态（2026-06-12）**：
  已完成，`add_model()` 默认把当前 active model 记录为新模型的 parent；已通过
  `E:\miniconda\envs\py312\python.exe -m pytest server\tests\test_training_backend.py::test_model_registry_records_parent_model_id -q`。

### B3 · 模型管理 UI（回滚）
- 新增（或在 AI 面板加）一个「模型」小面板：列 `GET /sam3/models`（id、metrics、激活态、parent），按钮「激活」→ `POST /sam3/models/{id}/activate`。**状态（2026-06-12）**：
  TargetDock 的「模型」按钮已升级为模型管理对话框，支持列表、激活选中模型、回滚到父模型。
- 回滚 = 激活 `parent_model_id`。服务端版本链字段与本机 UI 已具备；真实服务器手动验证待做。

### B4 · 测试
- 脚本契约：FastAPI-free，喂一个 fake `training.command`（写假 checkpoint+metrics.json 的 shell/py），断言 `_run_train_job` 登记模型且 `activate` 触发 `reload_checkpoint`（已有部分覆盖，补 metrics 链路）。

**DoD**：确认目标导出 → 微调产出带指标的新权重 → 列表可见可激活 → 推理用新权重 → 可一键回滚旧权重。这条通了即「标注—训练—推理—修正—再训练」闭环成立。

---

## Phase C · 储层工作台主窗口入口接线（P1）

状态（2026-06-12）：主窗口入口已接线。`MainWindow` 持有 `ReservoirRegistry`，视图菜单新增「打开储层剖面」；当当前会话已有 `ReservoirGridLayer`/`ReservoirPropertyLayer` 时，可打开 `ViewReservoirSection`，在储层剖面画 ROI 后构造 `SAM3Workbench(target_store=self.target_store, ...)`。工作台 `selection_committed` 会通过 undo 加入 `ReservoirSelectionLayer`，`target_committed` 会刷新 TargetDock。`SceneController` 也已接入 `ReservoirGridRenderer` / `ReservoirSelectionRenderer`，让储层 grid/property/selection 层进入 3D 渲染链。

边界：此入口**不重新引入旧 GRDECL 自动加载主流程**。它只使用当前会话已注册的 live `ReservoirGrid`，符合当前“储层大数据走 numpy/远程，本机轻展示”的方向；旧 Petrel/GRDECL 加载仍保留在离线转换/兼容工具中。

落点 [`ui/main_window.py`](../apps/yj_studio/src/yj_studio/ui/main_window.py)：
- ✅ 在储层剖面上画 ROI 的回调里构造 `SAM3Workbench(grid=..., roi=..., axis=..., transform=..., ai_service=self.ai_service, target_store=self.target_store, ...)`，作为新视图加入中央视图区。
- ✅ 连接 `selection_committed` → `AddLayerCommand(ReservoirSelectionLayer)`；`target_committed` → `TargetDock.refresh()`。
- ✅ `RemoteSAM3TrackTask` 完成已联动 `_on_ai_track_finished`（refresh + show_track_result），工作台路径也汇入 TargetDock。

**DoD**：储层模型上框 ROI → 工作台分割/传播 → GeoTarget 出现在 TargetDock，本地图层与远程目标同步。

---

## Phase D · 真多卡（P2）

不重复，直接照 [`next_steps_detailed_implementation.md` 步骤 6](next_steps_detailed_implementation.md)：`ProcessPoolExecutor` 每卡一进程、worker 只算不写、主进程 `persist_tracked_targets` 持锁统一落库。前置是 Phase A/B/C 价值已交付且服务器吞吐成为瓶颈时再做。

---

## 建议推进顺序与理由

```text
A1 闭合等值线   ← 地基，纯 CPU 可测，直接产出"圈闭"可视成果，最快见效
A2 圈闭检测     ← 复用 A1，给出排名+置信度，科研主线的核心交付物
B1–B4 训练闭环  ← 与 A 并行可做；打通"标注↔训练"是平台立项目标
A3 连通性       ← 体级地基
A4 砂体提取     ← 复用 A3
A5 圈闭评价     ← 消费 A1/A2，给储量数字，汇报有说服力
C  工作台入口   ← 小工作量，把已建能力接到用户手上
D  真多卡       ← 仅当吞吐成瓶颈
```

> 最快见效路径：**先 A1→A2**（一两天就能在界面上圈出带置信度的候选圈闭，直接对上你的研究方向），训练闭环 B 与之并行推进。A3–A5 紧随其后形成"构造圈闭 + 储层砂体 + 储量评价"的完整地质解释链。

---

# 全平台功能清单（G1–G8，平台级尚缺项）

上面 Phase A–D 是「圈闭算法链 + 训练/工作台/多卡」这条主线的待实现项，**已展开到「关键方法 + 输入输出契约」深度，可直接照着写**。本节是站在「能给真实地震工区用的圈闭检测 + 标注↔训练平台」完整形态下，**平台级**仍缺的功能全集，每条标注：**位置 / 数据模型 / 集成点 / 优先级(P0最急) / 量级**。

> **深度策略（重要，关于"为什么 G 项只有一行"）**：G1–G8 是横跨里程碑 M-γ…M-ε 的平台功能，时间跨度大。我**故意只写到规划级（一行：位置/数据模型/集成点/优先级/量级）**，因为：① 现在就把 P2 的运维/工业 IO 类（如 G4.2/4.3 ML 断层/层位、G7.x、G1.4/1.6）写成方法签名是**过早设计**——等你做到时接口一定会变，白写；② 真正近期要建的是上面 Phase A–D。**建议**：M-α/M-β 关键路径上的 7 项（G1.1 工程存档、G2.1 审校队列、G2.2 主动学习、G3.2 空间分块划分、G3.4 全体推理、G5.1 3D 渲染、G6.1 圈闭报告）值得现在就深挖到 A1 那种契约深度——需要我展开哪几项，告诉我；其余等你做到再深挖即可。

## G1 · 数据 I/O 与工程持久化（地基，多为 P0）

| # | 功能 | 说明 / 数据模型 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G1.1 | **工程/会话存档 `.yjproj`** | 当前**完全没有**：关掉程序，加载的体、图层、层位、断层、目标、ROI、视图布局全丢。做 JSON manifest（引用 data/ 下 npy 与 server project_id）+ `save/open/save as/最近工程`。落点新建 `apps/yj_studio/src/yj_studio/project/session.py`，菜单「文件」加项；序列化复用各 Layer 的 `to_dict`。 | **P0** | 中 |
| G1.2 | **SEG-Y 导入** | 现仅 legacy/cigvis 用 `cigsegy`。新 app 做 `data/segy_import.py`：`cigsegy.SegyNP` 读体→落 `data/.../*.npy`+metadata（含道头映射 inline/xline/CDP/X/Y）→注册 VolumeLayer。导入对话框选道头字节位置。 | **P0** | 中 |
| G1.3 | **测井 LAS 导入** | `libs/cigvis/io/las.py:load_las` 已存在但未接入 app。`WellRepository` 加 `import_las(path)`：曲线→WellLogLayer，井口坐标+井斜；再支持井分层(tops)文本。 | P1 | 小-中 |
| G1.4 | **层位/断层互操作格式** | 与 Petrel/OpendTect 往返：导入/导出 ZMAP+、CPS-3、IESX/Charisma 点集；面可导出 GeoTiff。落点 `data/horizon_io.py`、`data/fault_io.py`。让 A1/A2 的圈闭也能导出多边形。 | P1 | 中 |
| G1.5 | **测网几何 + 坐标系(CRS/EPSG)** | inline/xline ↔ 真实 X/Y，存 affine + EPSG。面积/储量要真实 m²、报告要真实坐标。落点扩展 `data/coord_transform.py` + 工程级 survey geometry。 | **P0**（A5 依赖） | 中 |
| G1.6 | **时深转换（速度模型）** | TWT↔深度：常速/层速度/速度体。圈闭高度与储量要在深度域算。落点 `data/time_depth.py` + VelocityModel 数据类。 | P1 | 中 |

## G2 · 标注工作流（"annotate" 半环，当前很薄）

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G2.1 | **标注状态机 + 审校队列** | ✅ 核心 + UI 已完成：`TargetStatus` 增加 `to_review/rejected`，`review_queue()` 生成待审列表；TargetDock 新增「审校」对话框，可按主动学习优先级批量确认/打回。 | **P0** | 中 |
| G2.2 | **主动学习（先标最不确定）** | ✅ 核心 + 初版 UI 已完成：`targets/active_learning.py` 提供 `target_uncertainty()` 与 `review_queue()`，按模型分数、面积稳定性、lost/edit 状态排序；TargetDock「审校」入口直接使用该排序。 | **P0** | 中 |
| G2.3 | **标注审计/版本** | GeoTarget.edits[] 已记录；做 diff 视图（谁/何时/改了哪帧），支持回退单次编辑。落点审校 dock 子面板。 | P2 | 小 |
| G2.4 | **快捷键标注模式** | 键盘驱动：接受/打回/下一个、笔刷增减、类别热键。决定标注吞吐量。落点 `tools/` + 全局快捷键。 | P1 | 小-中 |
| G2.5 | **类别/本体管理** | 目标类型/地质标签现偏硬编码。做可编辑 taxonomy（名称/颜色/父类），存工程。落点 `targets/taxonomy.py` + 设置面板。 | P1 | 小 |
| G2.6 | **传播结果逐帧复核** | 沿传播轴逐帧步进、标坏帧、就地重新种子重跑该段。工作台已有帧步进雏形，补「标坏帧/重种子」。落点 `view_sam3_workbench.py`。 | P1 | 中 |

## G3 · 训练 / 模型（接 Phase B）

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G3.1 | **数据集版本与快照** | 导出即固化不可变快照（已有 dataset_version 目录），补「列出/对比两版差异/标记用于某次训练」。落点 server 加 `DatasetRegistry`。 | P1 | 小-中 |
| G3.2 | **防泄漏的空间分块划分** | ✅ 已完成核心：`export_confirmed_to_coco(split_strategy="spatial")` 按连续 index 块划分 train/val/test，保留 `round_robin` 兼容旧行为。 | **P0** | 小-中 |
| G3.3 | **实验跟踪** | 部分完成：模型管理对话框可展示版本、metrics、active 与 parent；更完整的训练曲线/对比视图待做。 | P1 | 中 |
| G3.4 | **全体批量推理** | ✅ 初版契约已完成：server `/sam3/jobs` 支持 `kind="infer_volume"`，复用 batch worker，生成目标默认 `status=to_review`；TargetDock 提取模式新增 `infer_volume`，结果进入审校队列。真多卡仍归 Phase D。 | **P0** | 中 |
| G3.5 | **不确定度/置信度校准** | 模型分数校准到可信概率（温度标定），输出置信度图，喂 G2.2。落点 `sam3/calibration.py`。 | P2 | 中 |
| G3.6 | **地震专用数据增强** | 沿道/沿剖面翻转、振幅缩放、加噪、随机 crop。落点 `libs/sam3/train/transforms/`。 | P1 | 小-中 |

## G4 · 地震属性计算（解释驱动，当前是预烘焙 .npy）

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G4.1 | **应用内属性计算** | 现在相干/曲率是外部预生成 .npy（styles.py）。做成算法：相干/半相似、曲率、倾角/方位、RMS 振幅、瞬时包络、谱分解。每个=产出 attribute VolumeLayer 的算法。圈闭/断层识别核心输入。落点 `algorithms/builtin/attributes/`。 | P1 | 中-大 |
| G4.2 | **断层自动拾取** | 升级 `stubs/fault_autopick.py`：基于相干/ML 提断层面。 | P2 | 大 |
| G4.3 | **层位自动追踪** | 升级 `stubs/horizon_autotrack.py`/`auto_track_horizon_3d.py`：种子 + 波形相似度传播。 | P2 | 大 |

## G5 · 解释 / 可视化

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G5.1 | **3D 目标体渲染** | ✅ 核心已完成：TrapLayer 平面/3D surface 已接入；GeoTarget `mask3d` 现在可通过 `MaskRenderer.build_mask_volume_mesh()` 做 marching-cubes 成面，TargetDock「3D」按钮在无 cell 时会回退加载 `mask3d`。多体透明度/传递函数待 G5.2。 | **P0** | 中 |
| G5.2 | **多体共渲染 + 透明度/传递函数** | 地震+属性+模型叠加，opacity/colorbar 编辑器、传递函数。落点 PropertyDock + 渲染器。 | P1 | 中 |
| G5.3 | **交会图 / 直方图** | 属性交会图、直方图，用于选 cutoff（喂 A4 砂体阈值）。落点新 `ui/docks/crossplot_dock.py`。 | P1 | 中 |
| G5.4 | **目标轨迹/面积曲线子面板** | 每个目标 area_px/relief 随 index 折线（旧文档 7.1 待做）。落点 TargetDock 子面板。 | P2 | 小 |
| G5.5 | **栅栏图 / 多剖面联动** | 多连井/任意剖面拼图。已有任意剖面，补多面板联动。 | P2 | 中 |

## G6 · 结果输出 / 报告

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G6.1 | **圈闭清单报告** | ✅ CSV/XLSX/PDF 核心已完成：A2 圈闭字段 + A5 `MeasurementLayer` 的 GRV/HCPV/STOIIP/area/phi/thickness 可合并导出；PDF 为高信号摘要表。缩略图待做。落点 `report/trap_report.py`。 | **P0** | 小-中 |
| G6.2 | **多边形/面导出工业格式** | 圈闭边界、砂体、层位导出 ZMAP/shp/csv 回流 Petrel。接 G1.4。 | P1 | 小 |
| G6.3 | **图件导出** | 高分辨率截图/带标注成图（2D/3D）。落点视图右键「导出图像」。 | P1 | 小 |

## G7 · 工程化 / 运维 / 健壮性

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G7.1 | **服务端鉴权** | 你已决定暂缓；列此：token/API key 中间件 + 客户端带 header。落点 server `app.py` 依赖注入。 | P2（你定） | 小 |
| G7.2 | **大体外存/分块** | 超内存体 out-of-core + 推理 tiling。`VolumeStore` 已 memmap；补按需 chunk 与切片缓存协同。 | P1 | 中 |
| G7.3 | **作业恢复/断点续传** | job 终态已持久化；补「重启后恢复 running 作业/续传」。落点 server jobs.py。 | P2 | 中 |
| G7.4 | **日志/遥测/错误上报** | 统一结构化日志 + 客户端异常落盘。 | P2 | 小 |
| G7.5 | **配置/偏好面板** | 路径、默认参数、快捷键、配色集中可视化配置。落点 settings dialog + 写 local.yaml。 | P2 | 小-中 |
| G7.6 | **示例工程 / onboarding** | 内置小工区样例 + 引导。 | P2 | 小 |

## G8 · 质量控制 / 地质合理性

| # | 功能 | 说明 / 集成点 | 优先级 | 量级 |
|---|---|---|---|---|
| G8.1 | **地质合理性校验** | 圈闭须在盖层之下、跨层位闭合一致、与井分层吻合（井震标定）。落点 `qc/geo_consistency.py`，对 A2 输出后置校验打标。 | P1 | 中 |
| G8.2 | **掩膜质量 / 帧间一致性指标** | 相邻帧 IoU、面积突变检测，自动标可疑帧喂审校。落点 `targets/qc.py`，复用 reassociate.py 的 mask_iou。 | P1 | 小 |

---

# 里程碑与推进顺序（总）

```text
里程碑 M-α（出"圈闭"成果）：A1→A2 + G5.1 3D目标渲染 + G6.1 圈闭报告
里程碑 M-β（标注↔训练闭环跑通）：Phase B 微调脚本 + G3.2 空间分块划分 + G3.4 全体批量推理 + G2.1/G2.2 审校+主动学习
里程碑 M-γ（能接真实工区数据）：G1.1 工程存档 + G1.2 SEG-Y导入 + G1.5 测网坐标 + G1.3 LAS
里程碑 M-δ（解释能力补全）：A3→A4→A5（核心代码已完成，真实数据手动验收待做） + G4.1 属性计算 + G5.2/G5.3 可视化 + G8 QC
里程碑 M-ε（工程化）：G7 系列 + G1.4/G6.2 互操作 + G1.6 时深
```

> 强建议：**M-α 与 M-β 并行起步**。M-α 一两周内就有「圈出带置信度的圈闭 + 一键出报告」可演示成果；M-β 打通标注↔训练闭环（平台核心价值）。一旦要用真实工区数据，G1.1 工程存档和 G1.2 SEG-Y 从 P1 跳成阻塞项。具体先做哪一步、怎么做，见施工单 [`implementation_runbook_and_feature_backlog.md`](implementation_runbook_and_feature_backlog.md)。
