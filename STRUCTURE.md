# 项目结构说明（STRUCTURE）

本文件说明仓库的目录组织、架构约束，以及结构重构的进度与后续步骤。

## 顶层目录职责

| 目录 | 职责 | 是否进 git |
|---|---|---|
| `local/app/` | 桌面端应用（UI / 视图 / 算法 / 工具）。包 `yj_studio`，src 布局 | 是 |
| `local/` | 本地运行：桌面端 app（`local/app/`）+ 启动脚本（`run_viewer.py`、`scripts/`） | 是 |
| `server/` | 服务器端：FastAPI 应用、SAM3 引擎/作业/追踪、部署脚本、服务端测试 | 是 |
| `shared/` | 桌面端与服务器共用的核心包 `yj_studio_core`（含 `targets` 数据模型） | 是 |
| `config/` | **统一配置**：`config/*.yaml`（运行配置）+ `config/env/`（environment / requirements） | 例子进 git，live 配置忽略 |
| `data/` | 数据（地震体、油藏 numpy、结果）。大文件 | 否（gitignore） |
| `weights/` | 模型权重（`sam3.pt` 等） | 否（gitignore） |
| `libs/` | 第三方 vendored 源码（sam3、cigvis、well_section） | 是 |
| `tools/` | 一次性转换 / 校验 / 冒烟脚本 | 是 |
| `docs/` | 文档、图、计划 | 是 |
| `packaging/` | 打包配置 | 是 |
| `legacy/` | 历史材料（建议后续清理或外移） | 是 |
| `runtime/` `cache/` | 运行时生成物（日志、缓存、jobs） | 否（gitignore） |

## 配置统一约定（Phase 1 已完成）

所有配置集中在 `config/`：

```text
config/
├── local.yaml            # 本地运行配置（live，gitignore）
├── local.example.yaml    # 模板（进 git）
├── server.yaml           # 服务器配置（live，gitignore）
├── server.example.yaml   # 模板（进 git）
└── env/
    ├── environment.yml            # 完整 conda 环境（桌面端）
    ├── environment-local.yml
    ├── environment-server.yml
    ├── requirements.txt
    ├── requirements-server.txt
    └── requirements.lock.txt
```

读取方为**向后兼容**：新位置 `config/` 优先，找不到再回退旧位置（`server/config/`、`local/config/`），保证未及时重新部署的服务器照常启动。涉及的读取点：
- `local/run_viewer.py`（DEFAULT_CONFIG / FALLBACK_CONFIG）
- `server/src/yj_studio_server/config.py` `default_config_path()`
- `server/scripts/start_server.sh`（CONFIG 探测顺序）
- `server/scripts/install_env.sh`、`INSTALL.md`（environment / requirements 路径）

## 架构约束（重要）

- **根目录定位用标记，不用层级数**：`local/app/src/yj_studio/config/paths.py` 的 `_find_workspace_root()` 向上找 `.git`（或同时含 `libs/` 与 `config/`）的目录。**新增任何需要根路径的代码都应复用它，不要再写 `parents[N]`。**（仓库仍有约 30 处自算 `ROOT`/`parents[]`；因 `apps/yj_studio`→`local/app` 深度一致它们仍正确，后续可择机收敛，非必须。）
- **服务器对桌面包的唯一依赖是 `yj_studio.targets`**。这是 Phase 2 把它抽成 `shared/` 的依据——抽出后 server 与桌面端都依赖 `shared/`，解除 server→apps 的耦合。
- **`server/` 历史上是 git 未跟踪的**，靠手动拷贝部署，导致服务器代码会悄悄落后本地。Phase 1 已把它纳入 git。部署方式见 `docs/`（tar+scp+备份+重启）。

## 重构进度

### Phase 1 — 已完成（低风险，未移动大包）
- [x] 所有配置集中到 `config/`（+ `config/env/`），读取点改为新位置优先、旧位置回退。
- [x] 修复 `config/paths.py` 的 `parents[5]` 脆弱根路径 → `_find_workspace_root()`（标记式）。
- [x] `server/`、`local/`、`config/` 纳入 git。
- [x] 更新 `.gitignore`（live 配置忽略、example 进 git；保留旧位置忽略项）。
- [x] 本文件 STRUCTURE.md。
- [x] 重测：桌面端仅余 2 个 layer_tree WIP 失败；服务端 FastAPI-free 19/19。

### Phase 2 — 已完成（2026-06-12）
- [x] 抽 `shared/src/yj_studio_core/`，把 `targets` 移入（`yj_studio_core.targets`）；新增 `shared/pyproject.toml`（可 `pip install -e`）。
- [x] 全部 14 处 `yj_studio.targets` 引用改为 `yj_studio_core.targets`；server `targets.py` 垫片改为指向 `shared/src` + `yj_studio_core.targets`。
- [x] `apps/yj_studio` → `local/app`（`git mv`），删除空的 `apps/`。
- [x] 更新路径：`run_yj_studio.py`（PROJECT_SRC + SHARED_SRC）、`pyproject.toml` pytest `pythonpath`、`server/scripts/{start_server,install_env,run_tests}.sh`、`.gitignore`。
- [x] PYTHONPATH/安装：桌面经 `run_yj_studio` 注入 `shared/src`+`local/app/src`；服务端 `start_server.sh` PYTHONPATH 含 `shared/src`+`local/app/src`；`install_env.sh` 改为 `pip install -e shared` + `pip install -e local/app`。
- [x] 重测：桌面端 174/176（仅 2 个 layer_tree WIP）、服务端 FastAPI-free 19/19、导入冒烟（yj_studio + yj_studio_core + 15 算法）通过。
- [x] 重新部署服务器并外网验证。

> 备注：`apps/yj_studio`→`local/app` 深度一致（都 2 段），所以包内 `parents[N]` 计算自动仍正确；约 30 处自算根路径**未**强制收敛，可后续择机统一到 `_find_workspace_root()`。`legacy/` 清理、`tools/` 细分留作后续可选项。
