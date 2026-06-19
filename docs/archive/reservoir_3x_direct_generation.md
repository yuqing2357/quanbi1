# 储层 3x 直接重采样运行说明

> **历史方案，已停止作为最终运行规格。** 最终方案为
> `6.25 m x 6.25 m x 2 m` 的裁剪规则体，见
> [`reservoir_model_sam3_requirements.md`](reservoir_model_sam3_requirements.md)
> 和 `tools/bake_reservoir_npy.py`。本文仅保留旧
> `numpy_3x` / `numpy_3x_direct` 结果的追溯信息。

## 目的

当前 `data/reservoir/numpy_3x` 是从已有 1x 地震对齐储层体
`lithology_volume_seismic.npy` / `porosity_volume_seismic.npy` 做 repeat
upsample 得到的。它的体素间隔是 3x 精度，但信息来源仍是 1x 粗体。

`tools/create_reservoir_3x_direct_numpy.py` 用于生成一份新的 3x 体：

- 输入：GRDECL-derived native arrays + GRDECL COORD 几何；
- 输出：`data/reservoir/numpy_3x_direct/`；
- 不读取、不 repeat 现有 1x 地震对齐体；
- shape 与现有运行时 3x 体保持一致：`(4452, 2796, 1443)`；
- 体素间隔：`4.1666667m x 4.1666667m x 3.3333333m`。

## 本地轻量校验

只检查参数、bbox、metadata，不生成大文件：

```powershell
E:\miniconda\envs\py312\python.exe tools\create_reservoir_3x_direct_numpy.py --dry-run
```

## 服务器实际生成

在服务器项目根目录执行。该任务会生成约 16.7 GiB 岩性体和 33.5 GiB 孔隙度体，
建议放在服务器上跑，不在本地运行。

```bash
cd /root/quanbi
python tools/create_reservoir_3x_direct_numpy.py \
  --source-numpy-dir data/reservoir/numpy \
  --grdecl-dir data/reservoir/grdecl \
  --reference-3x-metadata data/reservoir/numpy_3x/metadata.json \
  --out-dir data/reservoir/numpy_3x_direct \
  --chunk-axis0 8 \
  --query-workers -1 \
  --overwrite
```

快速试跑一个 axis0 chunk：

```bash
cd /root/quanbi
python tools/create_reservoir_3x_direct_numpy.py \
  --out-dir data/reservoir/numpy_3x_direct_smoke \
  --chunk-axis0 2 \
  --max-axis0-chunks 1 \
  --overwrite
```

## 结果核验

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("data/reservoir/numpy_3x_direct")
print(json.dumps(json.loads((root / "metadata.json").read_text()), indent=2, ensure_ascii=False)[:2000])
for name in ["lithology_binary_3x_uint8.npy", "porosity_3x_float16.npy"]:
    arr = np.load(root / name, mmap_mode="r")
    print(name, arr.shape, arr.dtype)
PY
```

确认通过后，再把服务器配置中的路径从 `reservoir/numpy_3x/...` 切换到
`reservoir/numpy_3x_direct/...`。

## 新旧结果切片对比

生成不同方向和位置的二维切片对比图：

```bash
cd /root/quanbi
python tools/visualize_reservoir_3x_comparison.py \
  --old-dir data/reservoir/numpy_3x \
  --new-dir data/reservoir/numpy_3x_direct \
  --out-dir data/results/reservoir_3x_direct_comparison \
  --axis0-positions 25% 50% 75% \
  --axis1-positions 25% 50% 75% \
  --sample-positions 25% 50% 75%
```

输出目录会包含岩性、孔隙度的新旧并排对比图，以及 `summary.json`。
