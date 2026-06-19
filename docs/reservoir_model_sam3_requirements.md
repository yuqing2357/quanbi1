# 储层模型与 SAM3 分割需求说明

## 最终运行格式

GRDECL corner-point grid 只作为离线源数据。项目运行时使用裁剪后的规则
numpy 体：

```text
data/reservoir/npy_625x625x2_v3/lithology_binary_uint8.npy
data/reservoir/npy_625x625x2_v3/porosity_float16.npy
data/reservoir/npy_625x625x2_v3/metadata.json
```

最终空间采样间隔为：

```text
axis0:  6.25 m
axis1:  6.25 m
sample: 2.00 m
```

地震体横向间隔为 `12.5 m x 12.5 m`。储层规则体横向加密 2 倍，因此每个
地震面以及相邻地震面中点都有对应的储层剖面。地震纵向采样为 `10 m`，
储层规则体纵向加密 5 倍。

## 空间裁剪

转换范围取有效 GRDECL 储层覆盖范围与地震体空间范围的交集，并向外对齐到
地震节点。输出轴包含两端节点，索引关系为：

```text
seismic_axis = seismic_index_origin + output_index / scale
scale = (2, 2, 5)
```

这样可以避免横向 `2N` 与 `2N-1` 混用造成的半格偏移，并确保裁剪范围内的
整数地震面和中间位置都能精确落到输出节点。

## 属性采样

- 横向属性：在最终支持域内选择最近的有效原生储层列。
- 岩性：对每个 2 m 深度带做区间聚合；原生类 1/2 任一与深度带相交即记为 1。
- 孔隙度：选择最近的有效原生 `z_center`，输出为 `float16`，无数据为 `NaN`。
- 几何边界：使用 ACTNUM 和 ZCORN 确定真实有效深度范围。
- 横向支持域：先在原生逻辑 `(i,j)` 拓扑中只填充被数据包围的内部洞，再按相邻
  有效列的平均活动深度插值共享 COORD pillar，并栅格化原生列四边形。真实外边界
  保持为空，不再用“到最近有效列的距离阈值”切出支持域。

岩性体存储为 `uint8` 二值：

```text
0 -> 0
1 -> 1
2 -> 1
null/inactive -> 0
```

## 历史数据

以下数据均为历史中间结果，不作为最终运行时模型：

```text
data/reservoir/numpy/lithology_volume_seismic.npy
data/reservoir/numpy/porosity_volume_seismic.npy
data/reservoir/numpy_3x/
data/reservoir/numpy_3x_direct/
```

`numpy_3x` 来自已有 1x 规则体的 repeat upsample；`numpy_3x_direct` 虽然直接
读取 native 属性，但采用 `4.1667 m x 4.1667 m x 3.3333 m` 网格。二者都不再
是最终规格。

## 转换与运行时

最终转换工具为：

```text
tools/bake_reservoir_npy.py
```

转换发生在服务器离线阶段。运行时只读取最终 `.npy` 文件，不解析 GRDECL，
也不重新引入客户端 corner-point grid UI。

2026-06-19 的服务器最终产物为 shape `(2959, 2201, 2826)`，岩性 `uint8`、
孔隙度 `float16`，总计约 52 GB。全量校验确认岩性范围 `[0, 1]`、孔隙度范围
`[0, 0.36279296875]`，且目标岩性体素均处于有效孔隙度支持域内。

SAM3 数据流为：

```text
6.25 m x 6.25 m x 2 m numpy 储层体
  -> inline / xline / z / 任意剖面
  -> SAM3 图像或视频帧
  -> mask 映射回规则体 voxel index
  -> 三维 mask / 连通体 / 体积统计 / 训练样本
```
