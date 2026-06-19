# YJ Studio Remote Server Architecture

## 目标

本项目拆成四个清晰边界：

- `data/`: 大数据、权重、结果数据。
- `server/`: 服务器端服务、配置、部署、启动、日志入口。
- `local/`: 本地开发、调试、连接测试、桌面端启动辅助。
- `runtime/`: 自动生成的运行时缓存、日志、任务状态。

`local/app/`、`shared/` 和 `libs/` 是共享源码层，不属于某一台机器的运行状态。

## 边界

### data/

只保存数据：

```text
data/seismic/
data/reservoir/npy_625x625x2_v3/
data/reservoir/grdecl/
data/results/
```

不要放服务代码、启动脚本、机器私有配置。

### server/

服务器端只关心远程服务：

```text
server/src/yj_studio_server/
server/config/
server/scripts/
server/systemd/
runtime/server/logs/
```

服务从 `server/config/server.yaml` 读取数据路径，但不把数据复制进 `server/`。

### local/

本地端只关心本地开发和远程连接：

```text
local/config/
local/scripts/
local/tests/
runtime/local/cache/
runtime/local/logs/
```

本地不直接承担大规模转换、SAM3 推理或全量体数据处理。

### runtime/

运行时产物统一放这里，并默认不提交：

```text
runtime/server/logs/
runtime/server/cache/
runtime/server/jobs/
runtime/local/logs/
runtime/local/cache/
```

## 当前服务器路径

```text
/root/quanbi
```

建议服务器真实配置为：

```text
/root/quanbi/server/config/server.yaml
```

它可以从样例复制：

```bash
cp /root/quanbi/server/config/server.example.yaml /root/quanbi/server/config/server.yaml
```

## 接口路线

第一阶段：

- `GET /health`
- `GET /volumes`
- `GET /slice`

第二阶段：

- 本地异步请求与缓存。
- 服务器端 RGBA PNG 切片。
- 服务器端透明 mask。

第三阶段：

- `POST /sam3/jobs`
- `GET /sam3/jobs/{job_id}`
- `GET /sam3/jobs/{job_id}/result`

## 同步策略

- 只改远程服务：同步 `server/`。
- 只改本地调试：同步 `local/`。
- 改桌面端逻辑：同步 `local/app/`；改共享核心：同步 `shared/`。
- 改 SAM3 源码：同步 `libs/`。
- 更新大数据：同步 `data/`。

不要把 `runtime/` 作为同步目标。
