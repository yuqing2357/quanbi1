# 项目结构说明（STRUCTURE）

本文件说明仓库的目录组织、架构约束，以及结构重构的进度与后续步骤。

## 顶层目录职责

| 目录 | 职责 | 是否进 git |
|---|---|---|
| `apps/yj_studio/` | 桌面端应用（UI / 视图 / 算法 / 工具）。**Phase 2 计划迁到 `local/`** | 是 |
| `server/` | 服务器端：FastAPI 应用、SAM3 引擎/作业/追踪、部署脚本、服务端测试 | 是 |
| `local/` | 本地运行：启动脚本、远程连接探测等本地辅助 | 是 |
| `shared/` | **（Phase 2 引入）** 桌面端与服务器共用的核心包（先收纳 `targets` 数据模型） | 是 |
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

- **根目录定位用标记，不用层级数**：`apps/yj_studio/src/yj_studio/config/paths.py` 的 `_find_workspace_root()` 向上找 `.git`（或同时含 `libs/` 与 `config/`）的目录。Phase 2 移动包后仍正确。**新增任何需要根路径的代码都应复用它，不要再写 `parents[N]`。**（当前仓库仍有约 30 处自算 `ROOT`/`parents[]`，Phase 2 统一收敛。）
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

### Phase 2 — 待执行（需确认后做，改动面大）
目标：`apps/yj_studio` → `local/`，抽出 `shared/`，彻底解耦。建议顺序（每步 `git mv` + 重测）：

1. **抽 `shared/` 共享包**
   - 新建 `shared/src/yj_studio_core/`，把 `apps/yj_studio/src/yj_studio/targets/` 移入为 `yj_studio_core/targets/`（或保留名 `yj_studio.targets` 但置于 shared，桌面端与 server 都加 `shared/src` 到 path）。
   - 改 server `from yj_studio.targets import ...` → 新包名；改桌面端对 `targets` 的引用。
   - 更新 3 处 PYTHONPATH（`run_yj_studio.py`、`server/scripts/start_server.sh`、pytest 配置）加入 `shared/src`。
2. **桌面端 `apps/yj_studio` → `local/app`（或 `local/yj_studio`）**
   - `git mv apps/yj_studio local/app`；更新 `run_yj_studio.py` 的 `PROJECT_SRC`、`pyproject.toml` 的 `where`/`testpaths`/`pythonpath`、所有相对根路径计算（收敛到 `_find_workspace_root()`）。
   - 全局搜 `apps/yj_studio` 字符串（脚本、文档、部署）逐一替换。
3. **收敛根路径**：把约 30 处自算 `ROOT`/`parents[N]` 改为统一的 `_find_workspace_root()`（或各包内等价 helper）。
4. **部署脚本与 conda**：更新 `start_server.sh` 的 PYTHONPATH、`install_env.sh` 的 `pip install -e` 目标路径。
5. **全量重测**（桌面 + 服务端 FastAPI-free）→ **重新部署服务器** → 外网探测 `/health`、`/sam3/gpus` 确认全绿。
6. 评估 `legacy/` 是否删除或外移；`tools/` 是否细分。

> Phase 2 会同时打断 import / PYTHONPATH / 部署 / conda / 测试，必须分步小跑、每步重测，最后才重新部署服务器。
