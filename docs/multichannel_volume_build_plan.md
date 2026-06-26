# 多通道体构建方案（储层模型 + 地震结构特征融合）

> 目标：为后续 SAM3 微调 / 圈闭识别，构建一个与储层模型**精确共配准**的多通道数据，
> 让模型在识别储层体里的特殊形态目标时，除了看 0/1 储层分割，还能参考地震的
> 同相轴趋势、构造边界、断层不连续。本文是**可直接照做的实施计划**，每一步都给出
> 命令、产物、检查标准。所有重活在服务器上执行（本机无 fastapi/大数据）。

---

## 0. 已确认的几何事实（构建的基石，务必先认同）

储层模型 `npy_625x625x2_v3` 是地震某子立方体做 **2×2×5 各向异性 node-aligned 细化**
得到的，`metadata.json` 已闭式记录映射，**无时深歧义、无需重配准**：

| 量 | 值 |
|---|---|
| 模型 shape (axis0,axis1,sample) | `(2959, 2201, 2826)` |
| 模型体素间距 (m) | `6.25 / 6.25 / 2.0` |
| 地震全尺寸 | `(1684, 1451, 654)` |
| 地震体素间距 (m) | `12.5 / 12.5 / 10.0` |
| 细化倍数 scale | `(2, 2, 5)` |
| 地震起点 origin | `(204, 0, 88)` |
| 模型覆盖的地震裁剪切片 | `seismic[204:1684, 0:1101, 88:654]` → `(1480, 1101, 566)` |

**映射公式（两向都是精确的，node 对齐、含端点）：**

```
model_idx  = (seismic_idx - origin) * scale          # 地震 → 模型
seismic_idx_in_crop = model_idx / scale              # 模型 → 裁剪后地震（整除处即 node）
```

恒等校验（已验证通过）：`(span-1)*scale + 1 == model_shape`，三轴全中。

> ⚠️ 这意味着所有“配准”工作其实**只是整数倍上/下采样**，不需要任何空间变换/插值矩阵。
> 这是本方案能简单可靠的根本原因。

---

## 1. 核心决策：构建“逻辑多通道体”，不要物化整块大融合体

先把存储账算清楚（单通道、float32、模型分辨率）：

| 通道 | 网格 | 单通道大小 |
|---|---|---|
| 岩性 lithology | 模型 `(2959,2201,2826)` | 18.4 GB (uint8) |
| 孔隙度 porosity | 模型 | 36.8 GB (float16) |
| 地震属性（每个，模型分辨率 float32） | 模型 | ~33 GB |

把 4 通道物化成一个 `(C, 2959,2201,2826)` 的体 → **>120 GB**，而且地震属性是从 ×2×5
上采样来的、**没有新增信息**，纯属冗余存储。

**因此采用如下结构（推荐，且是本方案默认路线）：**

> **“多通道体” = 一组共配准的单通道源体 + 一份 `channel_spec.json` 通道契约 +
> 一个按需在取切片时把各通道堆叠起来的 provider。**
> 地震属性只在**地震原生分辨率（裁剪网格）**存一份（每个 ~1.7 GB，float16），
> 取 2D 切片时再按整数倍上采样对齐到模型切片。

这样：存储省 ~50×；通道可随时增删换；与现有“服务器按 volume 取切片”的架构一致；
训练时 dataloader 直接拿到 `(C,H,W)` 的多通道切片。

> 如果将来确实要喂给某个只能吃“一个文件一个体”的工具，再用第 6 步**只对标注 ROI**
> 物化成多通道小块（chips），而不是整块物化。

---

## 2. 通道清单（建议先做 4 通道，可裁可加）

| 通道 | 含义 | 来源 | 物理意义 / 为何要 |
|---|---|---|---|
| C0 `lithology` | 储层 0/1 | `model_lithology`（模型网格） | 目标的空间分布（你已有） |
| C1 `porosity` | 孔隙度 | `model_porosity`（模型网格） | 储层“质量”差异 |
| C2 `cosphase` | cos(瞬时相位) | 地震裁剪体 → 属性 | 同相轴/层理趋势，与振幅无关，**背斜褶皱**的关键 |
| C3 `coherence` | 相干/不连续 | 地震裁剪体 → 属性 | 断层/错断，**断鼻圈闭**的关键 |
| (可选 C4 `envelope`) | 瞬时振幅(包络) | 地震裁剪体 → 属性 | 反射强度，兜底外观信息 |

> 起步只做 C0–C3。C2 给“地层怎么走”，C3 给“哪里断了”，正好覆盖背斜/断鼻两类判别信号。

---

## 3. 实施步骤

> 环境（服务器 yjstudio-server env）：`numpy`、`scipy`（`scipy.signal.hilbert`、
> `scipy.ndimage`）。无需 GPU。所有脚本放 `server/scripts/`。
> 约定数据根 `--root /root/quanbi`。

### Step 1 · 裁剪地震到储层范围 ✅脚本已就绪

脚本：[`server/scripts/crop_seismic_to_reservoir.py`](../server/scripts/crop_seismic_to_reservoir.py)

```bash
python server/scripts/crop_seismic_to_reservoir.py --dry-run   # 先只校验几何
python server/scripts/crop_seismic_to_reservoir.py             # 正式裁剪
```

**产物**：`data/seismic/YJ-SEISMIC-RESERVOIR-CROP.npy`  `(1480,1101,566)`，dtype 同地震。

**检查**：
- [ ] dry-run 打印的三轴恒等式 `match=True`；
- [ ] 输出 shape == `(1480,1101,566)`；
- [ ] `np.load(...).dtype` 与原地震一致；非全 0、无全 NaN。

---

### Step 2 · 配准目检（必做，1 次，省后患）

写脚本 `server/scripts/verify_alignment.py`：取**同一条 inline**，分别从
模型岩性切片 和 “裁剪地震切片上采样到模型分辨率” 出图叠加，确认结构对得上。

**核心逻辑：**
```python
# 选一条模型 inline a0_model（0..2958），对应裁剪地震行 a0_seis = a0_model // 2
litho = lith_vol[a0_model]                     # (2201, 2826)  模型切片
seis  = crop_vol[a0_model // 2]                # (1101, 566)   地震切片
# 上采样地震切片到模型切片分辨率（×2 横向, ×5 纵向, node 对齐、含端点）
seis_up = resample_nodes(seis, (2201, 2826))   # 见下方 node 对齐函数
# 出图：左=岩性, 中=地震(灰度), 右=岩性轮廓叠加在地震上
```

**node 对齐上采样函数（通用，后续 provider 也用）：**
```python
import numpy as np
from scipy.ndimage import map_coordinates

def resample_nodes(src2d, out_hw, order=1):
    """把 src2d 按 node 对齐重采样到 out_hw（含端点）。整数倍时节点处精确命中。"""
    oh, ow = out_hw
    sh, sw = src2d.shape
    yy = np.linspace(0, sh - 1, oh)
    xx = np.linspace(0, sw - 1, ow)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(src2d.astype(np.float32), [gy, gx], order=order, mode="nearest")
```

**产物**：`data/seismic/attrs/qc/align_inline_<idx>.png`（3~5 条不同位置的剖面）+
`data/seismic/attrs/qc/alignment_report.json`（记录每条的岩性边缘 vs 地震梯度的相关性）。

**检查（验收门槛）**：
- [ ] 目视：储层体顶/底界与地震强同相轴**走向一致、不整体偏移**；
- [ ] 数值：岩性边界处地震梯度显著高于体内部（report 里相关性 > 0 且明显）；
- [ ] 若发现整体错位 → 停，回查 metadata 与地震文件是否匹配（**绝不可带病往下走**）。

---

### Step 3 · 计算地震属性（在裁剪体上）

写脚本 `server/scripts/compute_seismic_attributes.py`，对 `YJ-SEISMIC-RESERVOIR-CROP.npy`
逐属性计算，**沿 axis0 分块**处理以控内存（整块 922M 样本，hilbert 会翻倍）。

**C2 cos(瞬时相位)** —— 沿 sample（纵向）轴做解析信号：
```python
from scipy.signal import hilbert
# trace 在 axis=2（sample）。对一个 axis0 slab：
analytic = hilbert(slab.astype(np.float32), axis=2)   # slab: (n0_chunk, 1101, 566)
cosphase = np.cos(np.angle(analytic))                 # [-1,1]
cosphase = ((cosphase + 1.0) * 0.5).astype(np.float16)  # → [0,1]
```
> **必须先 cos 再上采样**：直接插值相位角会因 ±π 卷绕出错；cos(phase) 已连续，线性插值安全。

**C3 相干/不连续**（baseline：局部横向方差归一化，简单稳健，可后续升级为 semblance）：
```python
from scipy.ndimage import uniform_filter
a = slab.astype(np.float32)
win = (1, 5, 9)                      # (axis0, axis1, sample) 小窗
mean  = uniform_filter(a, win, mode="nearest")
meansq= uniform_filter(a*a, win, mode="nearest")
var   = np.clip(meansq - mean*mean, 0, None)
# 不连续 = 局部方差大；归一到 [0,1]（用全局稳健分位，见 stats）
```
> 分块时 axis0 方向要留 **halo**（窗口半径），算完丢弃 halo 行，避免块边接缝。

**C4 包络(可选)**：`envelope = np.abs(analytic)`，再做稳健分位裁剪归一。

**归一化**：振幅类（envelope）用 0.5%/99.5% 分位裁剪后线性缩放到 [0,1]；
cosphase、coherence 已在 [0,1]。所有缩放参数写进 `stats.json`，保证训练/推理一致。

**产物**（`data/seismic/attrs/`，均 `(1480,1101,566)` float16）：
- `cosphase_f16.npy`、`coherence_f16.npy`、(可选 `envelope_f16.npy`)
- `stats.json`：每通道的 `min/max/p005/p995/scale/offset`、窗口参数、生成时间、源文件名+大小

**检查**：
- [ ] 每个属性 shape == `(1480,1101,566)`、dtype float16、范围 ⊂ [0,1]；
- [ ] 抽 1~2 条剖面出 PNG：cosphase 应呈清晰层状条纹；coherence 在断层/边界处亮线；
- [ ] 无全 0 / 全 NaN；NaN（若地震有空道）统一填 0 并在 stats 记录占比。

---

### Step 4 · 写通道契约 `channel_spec.json`（多通道体的“定义文件”）

这份文件就是“多通道体”的正式定义，provider 和训练代码都读它。放
`data/seismic/attrs/channel_spec.json`：

```json
{
  "grid_model_shape": [2959, 2201, 2826],
  "grid_model_spacing_m": [6.25, 6.25, 2.0],
  "scale": [2, 2, 5],
  "origin_in_full_seismic": [204, 0, 88],
  "channels": [
    {"name": "lithology", "source": "model_lithology", "grid": "model",
     "path": "data/reservoir/npy_625x625x2_v3/lithology_binary_uint8.npy",
     "norm": "as_is", "dtype": "uint8"},
    {"name": "porosity",  "source": "model_porosity",  "grid": "model",
     "path": "data/reservoir/npy_625x625x2_v3/porosity_float16.npy",
     "norm": "clip01_porosity", "dtype": "float16"},
    {"name": "cosphase",  "source": "seismic_crop",    "grid": "seismic_crop",
     "path": "data/seismic/attrs/cosphase_f16.npy",  "norm": "as_is"},
    {"name": "coherence", "source": "seismic_crop",    "grid": "seismic_crop",
     "path": "data/seismic/attrs/coherence_f16.npy", "norm": "as_is"}
  ],
  "stats_path": "data/seismic/attrs/stats.json"
}
```

**检查**：
- [ ] 每个 path 存在、shape 与声明网格匹配（model 网格 vs seismic_crop 网格）；
- [ ] 写个 1 行加载校验：逐通道 `np.load(mmap)` 成功、shape 对。

---

### Step 5 · 多通道切片 provider（“按需堆叠”的核心，推荐主路线）

写一个小模块 `shared/src/yj_studio_core/multichannel.py`（放 shared，前后端可共用、本机可单测），
对外只暴露一个函数：给定 `axis`（inline/xline/depth）和模型索引 `index`，返回
`(C, H, W)` 的多通道切片，已对齐到模型分辨率、已归一化。

**契约：**
```python
def extract_multichannel_slice(spec, axis, index) -> np.ndarray:
    """
    返回 float32 (C, H, W)：
      - model 网格通道：直接取模型切片；
      - seismic_crop 网格通道：取对应裁剪地震切片(index//scale)，
        node 对齐上采样到模型切片分辨率(resample_nodes)；
      - 按 stats 做归一化。
    通道顺序 == channel_spec.channels 顺序。
    """
```

**关键索引换算（inline 为例，axis0）：** 模型 `index`(0..2958) → 地震裁剪行 `index // 2`；
得到的 `(1101,566)` 地震属性切片，`resample_nodes` 到 `(2201,2826)`。xline、depth 同理用各自 scale。

**单测（本机可跑，无需大数据）**：构造 `scale=(2,2,5)` 的迷你合成体，断言
- node 处上采样值与源精确相等；
- 输出通道数/顺序/形状正确；
- 已知错位时能被 alignment 检查捕捉。

**检查**：
- [ ] `extract_multichannel_slice` 三个 axis 都能出 `(C,H,W)`，无越界；
- [ ] 各通道范围合理；岩性通道仍是 0/1。

---

### Step 6 · （可选）只对标注 ROI 物化多通道 chips

当你已用 SAM3/标注流程圈出若干圈闭目标后，再写 `server/scripts/export_training_chips.py`：
对每个标注目标，按其 bbox（含 padding）调用 `extract_multichannel_slice` 截 patch，
连同 mask、类别、3D 实例 id、空间坐标，存成训练样本。**只物化标注区域，不物化整体。**

**产物**：`data/training/chips/<class>/<inst_id>_<slice>.npz`（`image (C,h,w)`, `mask`, `label`, `coords`）
+ `index.jsonl`（样本清单，供 dataloader）。

---

### Step 7 · （可选）把属性体登记进 config 做可视化 QC

为了能在前端把地震属性当图层叠加目检，在 `config/server.yaml` 的 `volumes:` 下加：
```yaml
  seismic_crop:
    label: 地震(储层范围)
    path: seismic/YJ-SEISMIC-RESERVOIR-CROP.npy
    cmap: gray
    clim: null
    voxel_spacing_m: {axis0: 12.5, axis1: 12.5, sample: 10.0}
  seis_cosphase:
    label: 地震-同相轴(cos相位)
    path: seismic/attrs/cosphase_f16.npy
    cmap: gray
    clim: [0.0, 1.0]
    voxel_spacing_m: {axis0: 12.5, axis1: 12.5, sample: 10.0}
```
> 改 config 需**重启服务器**。这步纯为肉眼对照，不是训练必需。

---

## 4. 最终产物清单

| 文件 | 网格 | 用途 |
|---|---|---|
| `data/seismic/YJ-SEISMIC-RESERVOIR-CROP.npy` | 地震裁剪 (1480,1101,566) | 属性计算源 + 目检 |
| `data/seismic/attrs/cosphase_f16.npy` | 地震裁剪 | C2 同相轴趋势 |
| `data/seismic/attrs/coherence_f16.npy` | 地震裁剪 | C3 断层不连续 |
| `data/seismic/attrs/envelope_f16.npy`（可选） | 地震裁剪 | C4 反射强度 |
| `data/seismic/attrs/stats.json` | — | 各通道归一化参数 |
| `data/seismic/attrs/channel_spec.json` | — | **多通道体定义**（provider 读它） |
| `data/seismic/attrs/qc/*.png` + `alignment_report.json` | — | 配准/属性目检证据 |
| `data/training/chips/*`（第 6 步，可选） | 模型 | 训练样本 |

## 5. 需要编写的脚本/模块

| 文件 | 状态 | 部署面 |
|---|---|---|
| `server/scripts/crop_seismic_to_reservoir.py` | ✅ 已就绪 | 服务器(server/) |
| `server/scripts/verify_alignment.py` | 待写 | 服务器 |
| `server/scripts/compute_seismic_attributes.py` | 待写 | 服务器 |
| `shared/src/yj_studio_core/multichannel.py` | 待写（含单测） | **shared，两边都要** |
| `server/scripts/export_training_chips.py`（可选） | 待写 | 服务器 |

## 6. 总验收标准（全绿才算“体构建完成”）

1. **几何**：crop 脚本三轴恒等式全 `match=True`，shape `(1480,1101,566)`。
2. **配准**：Step 2 目检剖面结构对齐、无整体偏移，report 相关性为正且显著。
3. **属性**：cosphase/coherence 范围 ⊂[0,1]、无全 0/NaN，剖面呈现预期的层状条纹 / 断层亮线。
4. **契约**：`channel_spec.json` 中每个 path 存在、shape 与网格声明一致。
5. **provider**：`extract_multichannel_slice` 三 axis 出 `(C,H,W)`，单测通过（node 处精确、岩性仍 0/1）。

---

## 7. 风险与注意

- **配准是唯一硬门槛**：Step 2 不过关，后面全废。务必先做、留 PNG 证据。
- **地震只给“背景”不给“细节”**：纵向 10m/横向 12.5m 比模型粗，融合是补构造上下文，
  不要指望它提升边界精度。
- **分块处理留 halo**：相干/滤波类沿 axis0 分块时必须留窗口半径的重叠，否则块边有接缝。
- **先 cos 再插值**：相位类属性严禁先插值原始相位角（卷绕出错）。
- **归一化一致性**：训练与推理必须用同一份 `stats.json`；任何重算属性都要更新它。
- **dtype 用 float16 存属性**：[0,1] 范围 float16 足够，体积减半（每个 ~1.7GB）。

---

## 8. 建议执行顺序

```
Step1 裁剪 → Step2 配准目检(卡点) → Step3 属性 → Step4 契约 → Step5 provider+单测
                                                          → (Step7 可视化QC，按需)
（攒到标注后）→ Step6 物化训练 chips → 进 SAM3 微调
```
```
