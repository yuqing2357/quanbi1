# 项目当前状态与后续路线图

本文档记录当前 YJ Studio Portable 项目的真实状态、已经形成的架构边界，以及后续希望完成的目标。它的作用是给后续开发提供统一判断标准：哪些事情应该发生在服务器，哪些事情只能发生在本机，哪些历史内容需要逐步清理。

## 1. 总体目标

项目最终要形成的工作方式是：

```text
本机：
  只负责桌面 UI、交互、可视化、用户操作、少量临时状态。

服务器：
  负责大体数据读取、储层/地震切片、缓存、SAM3 推理、长任务、结果生成。
```

也就是说，本机不应该再承担几十 GB 级别 `.npy` 的持续读写和转换。用户在本机点击、拖动、选择剖面时，本机应该向服务器请求需要的当前信息；服务器返回当前切片、结果或任务状态，本机只显示。

## 1.1 已确认决策摘要

以下决策来自当前阶段的项目约束，后续设计和实现默认按这些原则执行：

1. 本机保留一份大体数据备份，但远程模式下不把这份备份作为默认读取来源。
2. 服务器切片缓存最大允许占用 100 GB。
3. 服务器 API 暂时不需要账号、token 或鉴权。
4. SAM3 正式结果以 mask 为核心保存，目标是后续能作为三维空间体展示和分析。
5. 暂时不把 `.npy` 转换为 Zarr 或其他 chunked 格式。
6. 暂时不需要 systemd 开机自启；服务器是否启动、停止、验证由用户控制。
7. 本机 UI 可以保留并明确显示“远程模式 / 本地模式”和服务器连接状态。

## 2. 当前目录结构

当前项目已经整理为以下结构：

```text
YJ_Studio_Portable/
  apps/        主程序与桌面端 UI 代码
  data/        地震体、储层模型、处理结果等大数据
  weights/     SAM3 等模型权重
  server/      服务器端 API、配置、启动脚本、部署说明
  local/       本地运行、远程连接、本地调试入口
  tools/       历史转换脚本、检查脚本、烟雾测试脚本
  docs/        项目文档、架构说明、可视化图片
  runtime/     运行时日志、缓存、任务状态
  cache/       临时缓存、传输缓存
  libs/        第三方或项目内置依赖
  legacy/      历史项目内容
```

当前边界原则：

- `data/` 只放数据，不放服务代码。
- `server/` 只放服务器端运行相关内容。
- `local/` 只放本地启动、连接和调试相关内容。
- `runtime/` 放运行时产物，默认不提交。
- `tools/` 暂时保留原路径，因为文档、脚本、注释里仍有引用。

## 3. 当前核心数据状态

### 3.1 地震体数据

当前服务器上的主要地震体：

```text
data/seismic/YJ-ALL-SEISMIC.npy
shape: (1684, 1451, 1201)
dtype: float32
size: 约 10.93 GB
```

地震体目前已经可以被服务器 `/volumes` 接口识别，并可以通过 `/slice` 返回正交切片。

### 3.2 储层岩性模型

当前储层岩性模型已经从原始 GRDECL 方向转为 numpy 运行格式：

```text
data/reservoir/numpy_3x/lithology_binary_3x_uint8.npy
shape: (4452, 2796, 1443)
dtype: uint8
size: 约 16.73 GB
```

当前岩性值设计：

```text
0: 非目标或背景
1: 目标岩性
```

原先不再需要单独保留的 `2` 已经按你的想法并入 `1`，所以后续 SAM3 或识别流程可以把它当作二值岩性体使用。

### 3.3 储层孔隙度模型

当前储层孔隙度模型：

```text
data/reservoir/numpy_3x/porosity_3x_float16.npy
shape: (4452, 2796, 1443)
dtype: float16
size: 约 33.46 GB
```

孔隙度保留为 `float16`，比 `float32` 节省一半空间，同时比 `float8` 更稳妥，适合后续显示、阈值、识别和模型输入。

### 3.4 GRDECL 的角色

GRDECL 当前不再作为桌面端运行时的主要格式。它的角色是：

```text
历史源数据 / 可再生来源 / 离线转换输入
```

运行时应该尽量使用 numpy 或未来的 chunked 格式，而不是让 UI 直接解析 GRDECL。

## 4. 当前桌面端状态

### 4.1 左侧图层入口

当前桌面端左侧图层应该以以下三个主要体数据入口为核心：

```text
地震体数据
岩性模型
孔隙度模型
```

设计原则：

- 默认都是关闭状态。
- 用户勾选后才加载显示。
- 储层模型不应该显示成完整长方体。
- 岩性和孔隙度显示应使用储层有效区域 mask，避免出现无意义矩形块。

### 4.2 储层显示原则

储层显示应该接近 Petrel 中的储层有效体，而不是粗暴显示完整三维数组外包矩形。

当前重要设计：

- 岩性模型显示时使用孔隙度有限值区域作为透明 mask。
- 孔隙度模型本身也只应在有效储层区域可见。
- 背景无效区域应透明。

## 5. 当前本地/服务器运行边界

### 5.1 本地配置

本地运行配置集中在：

```text
local/config/local.yaml
```

核心内容：

```yaml
mode: remote
server_url: http://114.214.170.109:8765
request_timeout_s: 180
volume_backend: remote
```

含义：

- `mode: remote` 表示本机以远程服务为主要后端。
- `server_url` 是服务器 API 地址。
- `request_timeout_s` 当前设置较长，是因为储层体很大，某些方向第一次切片可能较慢。
- `volume_backend: remote` 表示体数据读取应走远程切片接口，而不是本地 `.npy`。

### 5.2 本地入口

本地 VSCode 运行入口：

```text
local/run_viewer.py
```

它的目标是：

1. 读取 `local/config/local.yaml`。
2. 检查远程服务器连接。
3. 设置运行时环境变量。
4. 启动桌面端。

注意：如果本机已经打开了旧版本 YJ Studio 窗口，旧进程仍可能按旧逻辑读取本机数据。需要关闭旧窗口后重新运行新的入口。

### 5.3 服务器配置

服务器配置集中在：

```text
server/config/server.yaml
```

它定义：

- 监听地址和端口。
- 项目根目录。
- 数据根目录。
- runtime/cache/log 路径。
- 可用体数据列表。

当前体数据列表包括：

```text
seismic
model_lithology
model_porosity
```

### 5.4 服务器已有内容清单

以下内容记录当前服务器侧已经具备的项目状态。后续若服务器目录有人工调整，应同步更新本节。

服务器连接与项目位置：

```text
host: 114.214.170.109
ssh_port: 2401
user: root
project_root: /root/quanbi
os: Linux
conda_env: /root/anaconda3/envs/yjstudio-server
env_name: yjstudio-server
```

服务器上的项目主体目录：

```text
/root/quanbi/
  apps/        桌面端和共享源码
  data/        服务器实际读取的大体数据
  docs/        项目文档
  libs/        依赖源码或第三方库
  local/       本地端配置和连接辅助脚本的同步副本
  server/      服务器 API、配置、启动脚本
  tools/       转换、检查、烟雾测试脚本
  weights/     SAM3 等模型权重
  runtime/     服务器运行时日志、缓存、任务状态
```

服务器上已经存在并用于远程服务的数据：

```text
/root/quanbi/data/seismic/YJ-ALL-SEISMIC.npy
  shape: (1684, 1451, 1201)
  dtype: float32
  size: 约 10.93 GB

/root/quanbi/data/reservoir/numpy_3x/lithology_binary_3x_uint8.npy
  shape: (4452, 2796, 1443)
  dtype: uint8
  size: 约 16.73 GB

/root/quanbi/data/reservoir/numpy_3x/porosity_3x_float16.npy
  shape: (4452, 2796, 1443)
  dtype: float16
  size: 约 33.46 GB
```

服务器端已同步的核心服务代码：

```text
/root/quanbi/server/run_server.py
/root/quanbi/server/src/yj_studio_server/app.py
/root/quanbi/server/src/yj_studio_server/config.py
/root/quanbi/server/config/server.yaml
/root/quanbi/server/config/server.example.yaml
```

服务器端已有脚本：

```text
/root/quanbi/server/scripts/start_server.sh
/root/quanbi/server/scripts/start_background.sh
/root/quanbi/server/scripts/stop_server.sh
/root/quanbi/server/scripts/healthcheck.sh
/root/quanbi/server/scripts/validate_data.py
/root/quanbi/server/scripts/run_tests.sh
/root/quanbi/server/scripts/install_env.sh
```

其中：

- `start_server.sh`：前台启动服务器 API。
- `start_background.sh`：后台启动服务器 API。
- `stop_server.sh`：停止后台服务。
- `healthcheck.sh`：检查服务健康状态。
- `validate_data.py`：检查服务器侧关键数据文件是否存在、shape 和 dtype 是否符合预期。
- `run_tests.sh`：服务器侧轻量测试入口。
- `install_env.sh`：服务器环境安装辅助脚本。

服务器配置文件：

```text
/root/quanbi/server/config/server.yaml
```

当前配置约定：

```text
host: 0.0.0.0
port: 8765
project_root: /root/quanbi
data_root: /root/quanbi/data
runtime_root: /root/quanbi/runtime/server
results_root: /root/quanbi/data/results
auth.enabled: false
```

服务器当前 API 设计已经覆盖：

```text
GET /health
GET /volumes
GET /slice
```

服务器运行时目录：

```text
/root/quanbi/runtime/server/logs/
/root/quanbi/runtime/server/cache/
/root/quanbi/runtime/server/cache/slices/
/root/quanbi/runtime/server/jobs/
/root/quanbi/runtime/server/tmp/
```

当前切片缓存已经放在：

```text
/root/quanbi/runtime/server/cache/slices/
```

注意：缓存上限已经确定为 100 GB，但自动清理机制仍属于待实现内容。目前已有的是“切片缓存写入位置和命名规则”，还需要补缓存大小统计、LRU 或按时间清理。

服务器端还保留了 systemd 模板：

```text
/root/quanbi/server/systemd/yj-studio-server.service
```

但当前决策是暂不启用 systemd 开机自启。服务器是否启动、停止、重启、验证，都由用户控制。

## 6. 当前服务器 API 状态

当前服务器端已经形成的核心接口：

```text
GET /health
GET /volumes
GET /slice
```

### 6.1 `/health`

用于检查服务器是否活着，以及服务器是否能看到数据目录。

### 6.2 `/volumes`

用于返回服务器上可用体数据的信息，包括：

- id
- label
- path
- shape
- dtype
- size
- cmap
- clim
- mask_volume

### 6.3 `/slice`

用于按需返回某一个体数据的一张正交切片。

示例：

```text
/slice?volume_id=seismic&axis=z&index=600
/slice?volume_id=model_lithology&axis=inline&index=2226
/slice?volume_id=model_porosity&axis=z&index=721
```

返回格式是 `.npy` 字节流，本机读取后得到一个 2D numpy 数组。

## 7. 当前服务器缓存策略

服务器端切片缓存位置：

```text
/root/quanbi/runtime/server/cache/slices/
```

设计目的：

- 第一次请求某个切片时，从大体数据中读取并生成切片。
- 后续请求同一个切片时，直接从服务器缓存返回。
- 缓存增长发生在服务器端，而不是本机端。

当前缓存属于初步版本，还没有完善：

- 目标最大占用为 100 GB。
- 尚未实现 LRU 清理，需要后续补上。
- 暂无按日期/体数据自动清理。

后续需要补上缓存管理，否则长期使用后服务器缓存也会持续增长。

## 8. 当前重要限制

虽然已经开始远程化，但当前还不是完整的“全功能远程桌面端”。

当前已经远程化的主要部分：

```text
体数据发现
正交切片读取
基础远程连接
服务器端切片缓存
```

仍需继续远程化的部分：

```text
任意剖面采样
沿层采样
井剖面相关体数据采样
SAM3 推理
SAM3 视频传播
大规模属性计算
结果保存与任务队列
```

因此，在后续功能没有迁移前，如果使用某些旧功能，仍可能触发本机读取本地数据。这是必须继续修的重点。

## 9. 当前验证原则

从现在开始，服务器端运行和验证由用户控制。

原则：

- Codex 不主动运行服务器命令。
- Codex 不主动重启服务器服务。
- Codex 不主动做服务器验证。
- 只有用户明确说“现在可以验证”或“你来运行验证”，Codex 才执行验证操作。
- Codex 可以提供命令、解释、检查清单和修改代码。

这条原则很重要，因为服务器是实际运行环境，任何重启、停止或长任务都可能影响当前工作。

## 10. 当前本机增长问题的解释

你提出的问题是正确的：

```text
如果本机只是可视化，本机不应该大量增长。
```

之前出现增长的原因主要有三类：

1. 历史本机转换留下的大文件。
2. 旧版本桌面端仍在本机 `np.load` 大体数据。
3. Python/Qt/Triton 的小型运行缓存。

其中真正危险的是第 2 点。当前已经开始修复，目标是让 `volume_backend: remote` 时体数据切片全部来自服务器。

本机历史残留需要单独清理，尤其是：

```text
data/reservoir/numpy_3x/*.partial
```

这些 `.partial` 是历史转换中间产物，不应该作为当前远程查看流程的一部分继续增长。

## 11. 后续目标路线图

### 阶段 1：确认远程查看闭环

目标：

- 本机打开软件。
- 左侧显示三类体数据。
- 勾选体数据后只请求服务器切片。
- 本机不再打开本地几十 GB `.npy`。
- 服务器缓存增长可解释、可控。

需要完成：

- 用户手动验证 `local/run_viewer.py`。
- 检查本机 `data/`、`cache/`、`runtime/` 是否还持续增长。
- 检查服务器 `runtime/server/cache/slices/` 是否按切片请求增长。
- 修复任何仍触发本地读大体数据的入口。

### 阶段 2：完善远程切片性能

目标：

- 切片响应稳定。
- 同一切片重复请求快速返回。
- 不同方向切片性能可接受。

需要完成：

- 服务器端切片缓存上限：100 GB。
- LRU 或按时间清理缓存，超过 100 GB 后自动删除最旧或最少使用的切片缓存。
- 切片请求日志。
- 可选预取当前切片附近的切片。
- 评估 `.npy` 是否需要转为更适合远程切片的 chunked 格式。

候选格式：

```text
Zarr
N5
按 axis 预生成切片 cache
多方向 chunked numpy/zarr
```

### 阶段 3：所有体数据采样迁移到服务器

目标：

所有需要体数据值的功能都不再直接读本机 `.npy`。

需要迁移：

- 任意剖面。
- 沿层采样。
- 井剖面体数据采样。
- 地震属性读取。
- 储层属性剖面。
- 可能的体数据 ROI 截取。

建议接口：

```text
GET  /slice
POST /section/arbitrary
POST /section/well
POST /sample/horizon
POST /volume/roi
```

### 阶段 4：SAM3 迁移到服务器

目标：

本机只提交识别请求和交互点/框，服务器负责 SAM3 推理。

SAM3 结果应以 mask 为核心保存。原因是后续不仅要看单帧或 PNG 序列，还需要把视频传播追出来的结果作为一个三维空间体来展示。因此结果格式应优先支持：

```text
三维 mask 体
按切片/时间传播得到的 mask 序列
可被桌面端作为三维图层加载和显示的 mask volume
```

建议接口：

```text
POST /sam3/jobs
GET  /sam3/jobs/{job_id}
GET  /sam3/jobs/{job_id}/result
POST /sam3/jobs/{job_id}/cancel
```

服务器端负责：

- GPU 推理。
- 视频传播。
- 三维 mask 保存。
- 任务状态。
- 中间结果缓存。
- 失败日志。

本机负责：

- 点选/框选。
- 显示任务进度。
- 显示返回 mask。
- 允许用户接受、撤销、编辑结果。

### 阶段 5：结果管理

目标：

所有识别结果和解释结果都有统一管理方式。

建议服务器结果目录：

```text
data/results/
  sam3/
  masks/       # SAM3 传播/分割得到的三维 mask 体
  sections/
  exports/
```

建议 runtime 目录：

```text
runtime/server/jobs/
runtime/server/cache/
runtime/server/logs/
```

原则：

- `data/results/` 是有价值结果。
- `runtime/server/` 是可再生运行状态。
- 不把临时缓存当作正式成果。

### 阶段 6：部署和后台服务

目标：

服务器 API 可以在用户明确启动后后台运行，用户不需要手动保持 SSH 窗口，但暂不设置开机自启。

已有基础：

```text
server/scripts/start_server.sh
server/scripts/start_background.sh
server/scripts/stop_server.sh
server/systemd/yj-studio-server.service
```

后续建议：

- 用户确认后台脚本行为。
- 暂不启用 systemd 开机自启。
- 增加日志轮转。
- 增加服务健康检查。
- 增加失败自动重启。

### 阶段 7：清理本机大数据

目标：

当远程流程稳定后，本机可以逐渐减少运行时对大数据的依赖，但本机仍保留一份大体数据备份。

可以考虑：

- 本机保留大体数据备份。
- 本机运行远程模式时不把本机备份作为默认读取来源。
- 本机保留轻量 metadata，辅助 UI 显示和状态判断。
- 本机保留小样例数据。
- 本机删除或移动历史 `.partial`。
- 本机大体数据仅作为备份，不参与运行。

需要用户确认后才能删除：

```text
data/reservoir/numpy_3x/*.partial
旧版 data/reservoir/numpy/*.npy
旧的转换日志和中间缓存
cache/remote_transfer/
```

## 12. 需要用户后续决定的问题

以下问题已经有当前决策，后续实现应按这些约束执行：

1. 本机保留一份大体数据备份。
2. 服务器切片缓存最大允许 100 GB。
3. 暂时不需要账号或 token 保护服务器 API。
4. SAM3 结果以 mask 为核心保存，后续需要支持三维空间展示。
5. 暂时不把 `.npy` 转换成 Zarr。
6. 暂时不需要 systemd 开机自启。
7. 本机 UI 可以保留“远程模式 / 本地模式”的明确状态显示。

这些决策不是永久锁死的，但在下一阶段开发中先作为默认约束。

## 13. 已确认的实现约束

### 本机数据策略

本机保留大体数据备份，但远程模式下不应默认读取本机大体数据。后续需要在 UI 和配置中明确区分：

```text
remote mode:
  从服务器请求体数据切片、SAM3 结果和任务状态。

local mode:
  允许使用本机 data/ 下的大体数据，主要用于离线调试或服务器不可用时的备选。
```

### 服务器缓存策略

服务器切片缓存上限为 100 GB。后续需要实现缓存管理器：

```text
runtime/server/cache/slices/
  最大 100 GB
  超过上限后清理最旧或最少使用的切片
  缓存只作为可再生运行时产物，不作为正式结果
```

### 服务器安全策略

当前暂不做账号、token 或鉴权。后续如果服务器暴露到更开放网络，再重新评估。

### SAM3 结果策略

SAM3 的结果不应只保存为 PNG 序列。PNG 可以作为预览或导出，但正式结果应该是可被三维展示和后续分析使用的 mask 数据。

建议优先设计：

```text
data/results/sam3/
  jobs/
  masks/
    <job_id>_mask.npy
    <job_id>_metadata.json
  previews/
```

其中 mask 应支持：

- 单剖面分割结果。
- 视频传播后的连续 mask。
- 后续转成三维图层显示。
- 后续用于统计、编辑、导出。

### 数据格式策略

暂时继续使用 `.npy`。Zarr 或其他 chunked 格式先不进入当前阶段，除非 `.npy` 切片性能成为明确瓶颈。

### 服务部署策略

暂时不做 systemd 开机自启。服务器服务可以通过手动脚本或后台脚本启动，但是否启动、停止、重启、验证，都由用户控制。Codex 不主动执行这些操作。

### UI 状态策略

桌面端应显示当前运行模式：

```text
远程模式
本地模式
服务器连接状态
当前 server_url
```

这样可以避免误以为在远程运行，实际却在读取本机大体数据。

## 14. 推荐下一步

推荐下一步只做一件事：

```text
用户手动运行 local/run_viewer.py，观察本机是否还大量增长。
```

如果仍然增长，需要记录：

- 增长的是哪个目录。
- 增长的文件名。
- 增长发生在勾选哪个图层之后。
- 是否打开了任意剖面、SAM3、沿层采样等旧功能。

只有确认“基础远程切片查看”稳定后，再继续迁移任意剖面和 SAM3。这样风险最小，也最符合项目现在的真实状态。
