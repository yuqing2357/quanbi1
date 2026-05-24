# SAM3 vendored source

把 `D:\商书记项目\sam3\sam3\sam3\` 整个目录(三层嵌套里最内层的那一个,里面有
`agent/ assets/ eval/ model/ perflib/ sam/ ...` 和 `model_builder.py`、`__init__.py`)
**完整复制**到本目录,目标结构如下:

```
f:\圈闭软件\libs\sam3\
├── README.md                ← 本文件
├── __init__.py              ← from .model_builder import build_sam3_image_model
├── model_builder.py
├── logger.py
├── agent/
├── assets/
│   └── bpe_simple_vocab_16e6.txt.gz
├── eval/
├── model/
│   ├── sam3_image_processor.py
│   ├── sam3_video_inference.py
│   ├── sam3_video_predictor.py
│   └── ...
├── perflib/
├── sam/                     ← 注意:这是 sam3 源码里 `sam/` 不是 `sam3/`
├── solver/
├── train/
└── ...
```

## 不要复制什么

- ❌ `D:\商书记项目\sam3\sam3\` 这一层及其同级的 `weights/`、`.git/`、`README.md`、`MANIFEST.in` 等(那是 git 仓库顶层,不是 Python 包)
- ❌ `D:\商书记项目\sam3\sam3\sam3\eval\hota_eval_toolkit\` / `teta_eval_toolkit\` 这两个评测工具子目录可选不要(它们只在评测时用,推理用不到,而且文件多)— 但全复制也无害,只是慢些

## 复制完后验证

打开 PowerShell:

```powershell
& 'E:\miniconda\envs\py312\python.exe' -c "import sys; sys.path.insert(0, r'f:\圈闭软件\libs'); import sam3; print('OK', sam3.__file__)"
```

应该输出 `OK f:\圈闭软件\libs\sam3\__init__.py`。如果报 `ModuleNotFoundError`,通常是目录层数不对,检查 `f:\圈闭软件\libs\sam3\__init__.py` 是否存在。

## 推荐复制命令

在 Windows PowerShell 里(robocopy 跳 `.git` / 跳到测试工具):

```powershell
robocopy 'D:\商书记项目\sam3\sam3\sam3' 'f:\圈闭软件\libs\sam3' /E /XD .git __pycache__ hota_eval_toolkit teta_eval_toolkit
```

`/E` 递归子目录,`/XD` 跳过指定目录。复制 ~30 秒。
