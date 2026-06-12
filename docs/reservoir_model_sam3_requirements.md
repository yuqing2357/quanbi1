# 储层模型与 SAM3 分割需求说明

## 当前决策

储层模型在项目运行时统一使用 numpy 规则体，不再把 Petrel/GRDECL
corner-point grid 作为主格式。GRDECL 文件只作为离线源数据，用于重新生成
numpy 体。

当前主储层数据位于：

```text
data/reservoir/numpy_3x/lithology_binary_3x_uint8.npy
data/reservoir/numpy_3x/porosity_3x_float16.npy
data/reservoir/numpy_3x/metadata.json
```

这两个体已经裁剪到储层最小三维范围，并做 3x 上采样：

```text
1x bbox: i=200:1684, j=43:975, k=132:613
1x shape: (1484, 932, 481)
3x shape: (4452, 2796, 1443)
```

岩性体存储为 `uint8` 二值：

```text
0 -> 0
1 -> 1
2 -> 1
null/NaN -> 0
```

孔隙度体存储为 `float16`，保留 `NaN`。

## 为什么改为 numpy

SAM3 的图像和视频分割流程天然需要规则像素/体素网格。把储层模型统一成
numpy 后，可以直接复用项目已有的 `VolumeStore`、`VolumeLayer`、正交剖面、
ROI 裁剪和后续 mask 体素回写流程。

旧 GRDECL 的 corner-point cell 几何适合 Petrel 风格精细网格显示，但会让
SAM3 的像素 mask 反查变成 cell-id 映射问题。当前项目优先服务 SAM3 识别、
规则体训练和后续 numpy 数据流，因此以 3x numpy 为主。

## 运行时约定

应用启动时，`model_lithology` 和 `model_porosity` 应从
`data/reservoir/numpy_3x` 发现：

```text
model_lithology -> lithology_binary_3x_uint8.npy
model_porosity  -> porosity_3x_float16.npy
```

旧的 `lithology_volume_seismic.npy` 和 `porosity_volume_seismic.npy` 是 1x
中间结果，只作为重新生成 3x numpy 的源数据，不作为默认运行时模型。

旧的 GRDECL 解析、corner-point grid、cell section、ReservoirGridLayer 等代码
只保留为离线转换/历史调试路径；新的交互和 SAM3 工作流应基于 numpy 体。

## 后续 SAM3 路线

后续 SAM3 储层识别应直接在规则体剖面上工作：

```text
3x numpy 储层体
  -> 取 inline / xline / z / 任意剖面
  -> 生成 SAM3 图像帧
  -> SAM3 输出 mask
  -> mask 像素直接映射回 3x voxel index
  -> 生成三维 mask / 连通体 / 属性统计
```

这样输出结果可以直接保存为 numpy mask，并与岩性、孔隙度、地震体或后续训练
数据保持一致。
