# SAM3 地质切片多帧追踪问题修复记录

> 记录日期：2026-06-20
> 适用范围：YJ Studio 服务器端 SAM3 视频追踪、Inline/Xline 相邻切片传播、GeoTarget 多帧结果落库。
> 结论性质：本记录描述已经验证有效的最终修复方法，用于后续维护、部署和回归检查。

## 1. 问题现象

在二维剖面上框选浊积体等目标后，单帧 SAM3 分割能够得到合理 mask；人工切换前后 Inline 或 Xline 切片，也能确认目标在多个相邻切片中连续存在。

但是执行追踪后，目标库经常只保存初始种子帧，并显示为“单帧分割”或“追踪仅 1 帧”。这说明问题不在目标是否真实存在，而在 SAM3 视频模型的播种、传播或运行依赖链。

## 2. 最终确认的根本原因

### 2.1 SAM3 视频模型是 detector-first 架构

SAM3 视频模型并不是一个可以在全新状态下直接用 point 或 mask 启动的普通 Tracker。

其正确工作顺序是：

```text
Visual Grounding / Detector 在种子帧建立目标
→ 生成对象及传播缓存
→ Tracker 基于已有缓存、记忆和 keep-alive 传播
→ 正向和反向输出连续帧 mask
```

Tracker 的 point/mask 路径主要用于已有视频传播结果上的交互细化。若在没有传播缓存的新状态中直接用该路径播种，会缺少 `cached_frame_outputs` 等必要状态，不能形成可靠的完整传播。

### 2.2 多个框不能逐个调用 detector `add_prompt`

Visual Grounding 检测器的 `add_prompt` 会建立或重置本次检测状态。

如果对多个种子框逐个调用：

```text
add_prompt(box_1)
add_prompt(box_2)
...
```

后一次调用可能覆盖前一次建立的目标状态，造成：

- 多目标种子最终只保留一个对象；
- 项目分配的 seed ID 与模型内部对象 ID 失配；
- 种子帧能够显示，但相邻传播帧无法归属到目标；
- 最终目标库只记录一帧。

正确方法是把所有种子框一次性提交：

```text
add_prompt(
    boxes_xywh=[box_1, box_2, ...],
    box_labels=[1, 1, ...],
    obj_id=None,
)
```

这使检测器在同一个种子帧状态中同时建立全部对象，然后进行一次完整视频传播。

### 2.3 模型内部对象 ID 不能直接等同于项目目标 ID

Visual Grounding 检测器会自行生成内部对象 ID，不能假设传入的 `obj_id` 会原样保留。

修复后，系统在种子帧执行以下映射：

1. 读取每个模型内部对象的 seed-frame mask；
2. 计算 mask 的归一化包围框；
3. 将模型包围框与用户种子框计算 IoU；
4. 按最大 IoU 把模型内部 ID 映射回应用 seed ID；
5. 后续所有传播帧沿用该映射；
6. 最终再由 seed ID 映射为 `GeoTarget` 的 `Txxx` 编号。

完整关系为：

```text
用户种子框
→ 应用 seed ID
→ SAM3 内部 model object ID
→ 正反向传播 mask
→ GeoTarget Txxx
```

### 2.4 自然视频时间消歧不适合短地质切片序列

SAM3 默认自然视频配置带有时间消歧和约 15 帧的 hot-start 规则，要求目标在视频早期满足自然视频检测与确认条件。

本项目通常只追踪前后各 5 帧，即总共约 11 个地质切片。地质剖面不是自然视频，目标也不一定会被通用检测器在每个切片重新检测。因此，默认 hot-start 可能在传播后处理阶段压掉目标。

服务器现在默认使用：

```yaml
sam3:
  video_temporal_disambiguation: false
```

对应模型构建参数：

```python
apply_temporal_disambiguation=False
```

这样保留 Visual Grounding 种子，并让 Tracker 通过内部记忆和 keep-alive 在短切片序列中传播。

### 2.5 连通域 Triton 箐子存在环境兼容风险

Tracker 的 mask 填洞过程会调用连通域运算。在没有可用 `cc_torch` 扩展的环境中，默认实现可能进入 Triton CUDA 路径，并出现：

```text
Triton Error [CUDA]: invalid argument
```

修复方案为：

- 如果 `cc_torch` 可用，保留原生快速实现；
- 否则优先使用 OpenCV CPU 连通域；
- OpenCV 不可用时再回退到 scikit-image；
- 保持与 SAM3 所需 `(labels, counts)` 张量契约一致；
- 将结果送回原始 tensor device。

该回退只处理较小的目标 mask，CPU 开销相对有限，但能避免追踪流程因 GPU/Triton 环境差异中断。

## 3. 最终有效流程

```text
用户在种子剖面框选一个或多个目标
→ 本地端提交 boxes、轴向、种子索引和前后帧数
→ 服务器渲染连续切片 JPEG 序列
→ 所有种子框一次性进入 Visual Grounding add_prompt
→ 根据种子框与 seed-frame mask IoU 建立 model ID ↔ seed ID
→ propagate_in_video 正向传播
→ propagate_in_video 反向传播
→ 将每帧 model ID 转换回 seed ID
→ collect_object_frames 按真实切片索引收集 mask
→ persist_tracked_targets 将同一对象写入一个 GeoTarget
→ 生成多帧 mask3d、帧统计和追踪诊断
→ 本地 2D 播放器与独立 3D 窗口展示结果
```

## 4. 关键实现位置

### 4.1 视频追踪与单次多框播种

文件：

```text
server/src/yj_studio_server/sam3/engine.py
```

关键逻辑：

- `SAM3Engine.track_video`
- `SAM3Engine._seed_frame_via_detector`
- `_match_models_to_seeds`
- `_mask_norm_box`
- `_iou_xywh`

### 4.2 短序列模型配置

文件：

```text
server/src/yj_studio_server/sam3/engine.py
server/src/yj_studio_server/app.py
server/src/yj_studio_server/sam3/gpu_pool.py
config/server.example.yaml
```

关键配置：

```yaml
video_temporal_disambiguation: false
```

### 4.3 连通域 CPU 回退

文件：

```text
server/src/yj_studio_server/sam3/engine.py
```

关键逻辑：

- `_install_cpu_connected_components_fallback`
- `_connected_components_cv2`

### 4.4 传播结果收集和落库

文件：

```text
server/src/yj_studio_server/sam3/tracking.py
server/src/yj_studio_server/app.py
```

结果中记录：

- 请求帧数；
- 请求切片范围；
- 每个对象实际收集帧数；
- 每个目标最终保存帧数；
- 缺失帧；
- 当前追踪模式 `detector_vg`。

## 5. 为什么此前方案没有达到预期

此前排查主要集中在：

- 目标 ID 映射；
- hot-start 抑制；
- mask 保存与统计；
- 2D/3D 可视化；
- 目标列表刷新。

这些问题确实会影响结果表达，但没有完全命中最核心的架构约束：

> 新视频状态必须先由 Visual Grounding 检测器建立对象和传播缓存，而且多个框必须在同一次 detector prompt 中提交。

仅修复后续 ID、保存或可视化，无法补回在播种阶段没有正确建立的视频对象状态。

## 6. 回归测试要求

后续修改追踪代码时，至少应保持以下测试：

1. 单框只调用一次 Visual Grounding prompt；
2. 单框结果覆盖种子帧、前向帧和反向帧；
3. 多框在一次 prompt 中提交；
4. 多个模型内部 ID 能按框 IoU 映射到不同 seed ID；
5. 正反向传播中对象 ID 保持稳定；
6. 缺少 `cc_torch` 时启用 CPU 连通域；
7. 有 `cc_torch` 时不替换原生实现；
8. 连通域回退输出 shape、device、labels 和 counts 契约正确；
9. 收集帧数与最终 GeoTarget 保存帧数一致。

当前服务器相关回归结果：

```text
18 passed, 1 skipped
```

跳过项由本地可选运行依赖决定，不影响核心播种、传播、ID 映射和落库测试。

## 7. 部署注意事项

本修复涉及服务器端视频模型初始化和 GPU worker 初始化，因此部署后必须重启服务器及多 GPU worker。

旧的单帧追踪结果不会自动补齐，必须重新执行追踪。

建议部署后首先进行一个固定样例验证：

1. 选择人工确认跨 5 个以上切片连续存在的浊积体；
2. 设置种子前后各 5 帧；
3. 执行追踪；
4. 检查结果中的请求帧数、收集帧数和保存帧数；
5. 使用 2D 播放器逐帧检查；
6. 使用 3D 窗口检查目标体连续性；
7. 保存该样例作为后续模型或环境升级的冒烟测试。
