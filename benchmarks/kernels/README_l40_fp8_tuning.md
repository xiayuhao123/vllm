# L40 block-FP8 调优准备

本工具用于现有 `vllm/vllm-openai:v0.31.0-l40-mtpgraph1` 镜像。
无需构建镜像，也不加载 Qwen 模型。它不会修改安装目录或服务配置。

## 当前状态

- #58763 的 GDN metadata 切片回移已在源码分支完成，尚未加入服务镜像。
- 调优工具和 CPU 回归已准备；没有在本地执行 CUDA 数值检查或 L40 计时。
- #58005 的 L40S JSON 仅作为候选，保留原名称和来源。
- 等 L40 结果明确后，再合并 metadata、有效配置和必要的 Split-K 分派调整。
  暂时继续使用已经验证的 `mtpgraph1`，不需要重做此前四组测试。

## 在内网手动运行

解压工具包后，进入 `vllm-l40-v031-fp8-tuning` 目录。
先确保所选 GPU 空闲。只有两张卡且 TP2 模型正在运行时，需停止模型一次；
也可以选另一张空闲 L40。以下选择物理 GPU 2，容器内部对应设备 0。

```bash
mkdir -p results
docker run --rm \
  --gpus '"device=2"' \
  --ipc=host \
  --network=none \
  --entrypoint python3 \
  -v "$PWD:/work:ro" \
  -v "$PWD/results:/results" \
  vllm/vllm-openai:v0.31.0-l40-mtpgraph1 \
  /work/tune_l40_block_fp8.py \
  --m 1 4 8 16 32 \
  --output /results
```

这个离线任务会编译多种内核，耗时取决于 GPU 和编译缓存；不是五轮服务测试。
完成后按原命令启动原模型即可，原 benchmark 和部署参数均可继续沿用。
这一步可等维护窗口再做。

仅查看搜索计划、不使用 GPU：

```bash
docker run --rm --network=none --entrypoint python3 \
  -v "$PWD:/work:ro" \
  vllm/vllm-openai:v0.31.0-l40-mtpgraph1 \
  /work/tune_l40_block_fp8.py --plan
```

可用 `--shapes gdn mlp` 限定两个已启用 Split-K 的形状，或用 `--m 4`
缩小一次运行的范围。默认覆盖日志中的五个矩阵形状和五个小 M。
`--exhaustive` 扩大搜索空间，首轮无需添加。

## 测量与筛选

- 保留 M64/N128/K128 默认 tile 作为基线，搜索 M16/32、N32/64/128 等参数。
  量化 block 和 K tile 固定为 128，不改变模型 FP8 格式。
- 对支持的形状联合比较普通 GEMM 和 Split-K=4，后者计入 FP32 中间结果
  写入与归约，避免只比较分区 GEMM 的耗时。
- 每个候选先以独立解量化后的 FP32 矩阵乘检查数值，再核对 CUDA Graph
  replay 与 eager 的输出一致性。失败候选不会进入推荐结果。
- 计时使用 CUDA Graph 完整内核链。分别记录热缓存与每次 replay 前清理
  L2 的样本；清理操作在开始事件之前，不计入 GEMM 耗时。
- 基线表示当前全开部署：两个 primary 形状使用默认 tile + Split-K=4，
  其余形状使用默认 tile 的普通 GEMM。N8192 的可选 Split-K 同时作为候选。

## 返回什么结果

请保留并返回 `results/report.json`，无需提供模型权重。它包含环境版本、
内核源码和种子文件 SHA-256、数值检查、所有计时样本、失败候选和每个形状的
优胜参数。每完成一个 M/N/K case 会保存一次，方便检查已完成的进度。

`results/candidate_configs/` 仅导出普通 GEMM 的候选 JSON。
未测小 M 和较大 M 保留默认配置锚点，避免把小 tile 自动外推到大 prefill。
Split-K 推荐在报告中单独保存；**不要直接将这些 JSON 复制到服务镜像**，
因为现有全局 Split-K 开关可能仍套用不适合该 tile 的分区策略。

这些是内核微基准结果，不能直接换算为 tokens/s 或端到端加速。
我们会结合各形状收益、方差及当前分派规则选取保守改动，再统一打包。

## 来源

- [#58763](https://github.com/vllm-project/vllm/pull/58763)：纯投机 GDN
  metadata 使用切片，保留混合 prefill 路径。
- [#58005](https://github.com/vllm-project/vllm/pull/58005)：L40S 调优种子，
  详见 `l40_fp8_seeds/README.md`。种子不是 L40 实测配置。
