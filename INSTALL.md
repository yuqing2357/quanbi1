# YJ Studio 部署指南（Windows + NVIDIA GPU）

这份指南帮你把整个项目从压缩包恢复到能跑的状态。预计 30–60 分钟（取决于网速，下载 PyTorch + SAM3 权重大约 8 GB）。

## 0. 前置条件

| 项 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 / 11 64 位 |
| GPU | NVIDIA，至少 12 GB 显存（推荐 16 GB+，SAM3 视频追踪吃显存） |
| 显卡驱动 | 支持 CUDA 13.0（驱动版本 ≥ 565.x） |
| 磁盘 | 项目根 + 权重 ≈ 8 GB，conda 环境 ≈ 6 GB |
| Conda | Miniconda 或 Anaconda，任意版本（≥ 4.12 即可） |

> 检查驱动：`nvidia-smi`，CUDA Version 一栏要 ≥ 13.0。如果 < 13.0，先升驱动。

## 1. 解压项目

把压缩包解压到一个**完全 ASCII** 的路径（**不要**有中文 / 空格），例如：

```
D:\yj_studio\
```

> 如果一定要放中文路径，整体能跑，但 Triton 编译路径必须改 —— 见末尾「常见问题：中文路径」。

解压后结构应该是：

```
D:\yj_studio\
├── local\                   ← 本地运行：桌面端 app(local\app\) + 启动脚本
├── server\                  ← 服务器端代码
├── shared\                  ← 桌面端与服务器共用核心包 yj_studio_core
├── config\                   ← 统一配置：config\*.yaml + config\env\（environment/requirements）
│   ├── env\environment.yml   ← 本指南要用
│   └── env\requirements.lock.txt
├── libs\sam3\                ← SAM3 源码（已含权重之外的所有文件）
├── weights\
│   └── sam3.pt               ← 5 GB，必需
├── docs\
├── run_yj_studio.py
└── INSTALL.md                ← 本文件
```

如果 `libs\sam3\model_builder.py` 不存在，说明压缩包没含 SAM3 源码 —— 联系发包方补。
如果 `weights\sam3.pt` 不存在或不到 4 GB，也是少文件，必须补。

## 2. 创建 conda 环境

打开 **Anaconda Prompt**（不是普通 PowerShell，conda 命令默认只在 Anaconda Prompt 里可用），cd 到项目根：

```cmd
cd /d D:\yj_studio
conda env create -f config/env/environment.yml
```

第一次会从 pytorch 官方源下载 `torch==2.11.0+cu130`（~3 GB）和 `triton-windows`。耐心等，10–20 分钟。

完成后激活：

```cmd
conda activate py312
```

> 想用别的环境名？把 `config/env/environment.yml` 第一行 `name: py312` 改成你想要的名字再创建。
>
> 注：所有依赖/配置清单已统一到 `config/`（`config/env/` 放环境与 requirements，`config/*.yaml` 放运行配置）。

## 3. 验证安装

```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望输出：`2.11.0+cu130 True`

如果 `is_available()` 是 `False`，几乎一定是驱动版本不够，回到第 0 步检查 `nvidia-smi`。

```cmd
python -c "import sys; sys.path.insert(0, r'libs'); import sam3; print('SAM3 OK', sam3.__file__)"
```

期望输出：`SAM3 OK D:\yj_studio\libs\sam3\__init__.py`

## 4. 启动 YJ Studio

```cmd
python run_yj_studio.py
```

`run_yj_studio.py` 已经处理好三个 Windows 特有的问题（OpenMP libiomp 冲突、Triton 长路径 UNC 前缀、Triton 缓存路径需要 ASCII）。你不用做任何额外配置。

第一次启动会因为 Triton JIT 编译 SAM3 的若干 kernel，慢 1–2 分钟，之后会缓存到 `C:\yj_triton_cache\`，再启动就快了。

## 5. 跑一遍冒烟测试（可选）

```cmd
python tools\smoke_sam3_video.py
```

如果输出最后是 `[OK]`，说明 SAM3 视频追踪整条链路通畅。

---

## 常见问题

### 中文路径

如果项目根含中文（比如压缩包名字带中文，解压时没改名），Triton 编译会失败。两种修复：

1. **推荐**：移到 ASCII 路径，重新解压。
2. **不推荐但可行**：保持中文路径不动 —— `run_yj_studio.py` 已经把 Triton 缓存重定向到 `C:\yj_triton_cache\`，主程序能跑；但 `tools\` 下的独立脚本需要先 `set TRITON_CACHE_DIR=C:\yj_triton_cache` 才能跑。

### `OMP: Error #15: libiomp5md.dll already initialized`

`run_yj_studio.py` 已设置 `KMP_DUPLICATE_LIB_OK=TRUE`。如果你不通过它启动而是直接 `import torch`，先在 cmd 里 `set KMP_DUPLICATE_LIB_OK=TRUE`。

### `cc1.exe: fatal error: \\cuda_utils.c: No such file or directory`

Triton 在 Windows 长路径下的 bug。`run_yj_studio.py` 里已经 monkey-patch 了 `subprocess.check_call` 来剥 `\\?\` 前缀。如果你跑独立脚本遇到这个错，把同样的 patch 抄过去，或者直接在脚本里 `import run_yj_studio`。

### `ModuleNotFoundError: No module named 'sam3'`

`libs\` 没在 `sys.path` 里。`run_yj_studio.py` 会自动加，独立脚本需要 `sys.path.insert(0, r'libs')`。

### GPU 显存不够（SAM3 视频追踪卡死/超慢）

SAM3 视频追踪每多一帧 KV 缓存涨一截。16 GB 显存大约能撑 10 帧 1024×3200 的画面。如果你的卡 < 16 GB，把 `local\app\src\yj_studio\reservoir\sam3_render.py` 里的 `_DPI = 200` 改成 `100`，分辨率减半、显存压力减到 1/4。

### opencv (cv2) 找不到

主程序不依赖 cv2。`libs\sam3\` 里有几个文件 import cv2，但只在评测 / 训练路径上才会触发。如果你确实要跑它们：

```cmd
pip install opencv-python
```

---

## 卸载

```cmd
conda env remove -n py312
rmdir /s /q C:\yj_triton_cache
```

项目目录本身可以直接删。
