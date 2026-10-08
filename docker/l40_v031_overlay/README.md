# vLLM 0.31.0 L40 离线覆盖包

这是基于官方 **v0.31.0** 的候选版本。保留 Split-K、独立 MTP decode graph，
新增可选的采样和卷积 update 预热。保留官方的新采样接口、初始化和缓存修复。
PCIe IPC 使用 0.31.0 原有的 FlashInfer 后端。无需重新编译 CUDA 扩展，也无需
在这个包里携带 FlashInfer wheels；不要叠加在旧版 0.30.0 镜像上。

## 1. 构建

将整个压缩包传入内网并解压。下面在解压后的目录执行：

```bash
sha256sum -c SHA256SUMS
docker build --progress=plain --pull=false --network=none \
  --build-arg BASE_IMAGE=vllm/vllm-openai:v0.31.0 \
  -t vllm/vllm-openai:v0.31.0-l40-mtpgraph1 .
```

前提是本机已有官方 `vllm/vllm-openai:v0.31.0` 镜像。安装器检查 vLLM 版本、
FlashInfer `0.7.0.post1` 和七个 Python 文件的官方源码哈希；其中两个新增模块
应当尚不存在。全部检查通过后才复制。若源码不匹配，请保留错误并检查镜像来源，
不要删掉校验强行覆盖。构建通常很快，因为只复制 Python 文件。

## 2. 手动部署

停止旧服务后，沿用你的部署方式：

```bash
docker run --rm \
  --name qwen38-fp8-31-l40-mtpgraph1 \
  --gpus '"device=2,3"' \
  --ipc=host \
  -p 18013:8000 \
  -v /model:/model \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 \
  -e VLLM_ENABLE_SM89_FP8_SPLIT_K=1 \
  -e VLLM_ENABLE_SM89_MTP_DECODE_GRAPH=1 \
  -e VLLM_ENABLE_SM89_EXTRA_WARMUP=1 \
  vllm/vllm-openai:v0.31.0-l40-mtpgraph1 \
  /model/Qwen/Qwen3.8-27B-FP8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --enable-per-request-metrics \
  --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kernel-config '{"linear_backend_per_quant":{"fp8_block_w8a8":"triton"}}' \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192
```

唯一新增的开关是 `VLLM_ENABLE_SM89_EXTRA_WARMUP=1`，控制 L40 采样 fallback 和
Qwen 卷积 update 的启动预热；默认关闭。它主要针对首次请求 JIT 尖峰，可能增加
启动时间，不能预先计作稳态加速。原 Split-K 和 graph 开关继续独立生效。
`VLLM_ENABLE_SM89_FP8_SPLIT_K_N8192_K5120` 继续默认关闭，维持之前的范围。
这里没有新增 scheduler、Mamba 缓存或 KV scale 参数。

## 3. 沿用你的五轮 benchmark

你的 benchmark 脚本只改结果名称，例如：

```bash
TEST_NAME="mtp3-burst-c4-16k-v031-l40-mtpgraph1"
```

保持原来的五轮、seed 12345–12349、并发 4、四条请求、16K 输入、1024 输出、
`--ignore-eos` 和每轮 reset prefix cache。**客户端继续使用原来的 0.30.0 镜像**，
避免同时改变 benchmark 工具。无需改温度或另跑四组开关矩阵。

## 4. 同一容器补三条共享前缀请求

完成原 benchmark 后，在这个包的目录执行。第一条开始前清空一次缓存；
第二、三条之间不要清缓存。文件 1、2 完全相同，文件 3 在相同历史上追加一轮。
共享文本足够长，可以跨越此前日志中 1600 token 的混合缓存块边界。
如果部署使用不同模型名，先改三个 JSON 里的 `model`。

```bash
curl -fsS -X POST http://127.0.0.1:18013/reset_prefix_cache
curl -fsS http://127.0.0.1:18013/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @prefix-check-1.json \
  -o prefix-response-1.json
curl -fsS http://127.0.0.1:18013/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @prefix-check-2.json \
  -o prefix-response-2.json
curl -fsS http://127.0.0.1:18013/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @prefix-check-3.json \
  -o prefix-response-3.json
```

检查三份响应的 `choices[0].message.content` 应为 `L40-031-CACHE-OK`，且没有
重复/乱码/报错；第 2、3 条的 `usage.prompt_tokens_details.cached_tokens` 应大于 0。
若安装了 jq，可以这样查看：

```bash
jq '{content:.choices[0].message.content,usage:.usage}' prefix-response-*.json
```

第三条使用预期的 assistant 回答构造。如果前两条没有按要求回答，先记录异常，
不要把第三条当作真实回答续接的验证。这只是一次小范围缓存/输出检查，不能替代
模型质量评估。若缓存字段为 0 或缺失，保存响应和日志后分析，不要直接判定补丁
正确或缓存损坏。这个检查的 `temperature=0` 不用于你的性能 benchmark。

## 5. 保存日志

```bash
bash collect_logs.sh qwen38-fp8-31-l40-mtpgraph1 l40-v031-logs
```

重点看 `Selected TritonFp8BlockScaledMMKernel`、PCIe IPC 初始化、Split-K 选择、
`Independent MTP ... captured/replay`、采样 provider 注册、
`Warmed Qwen causal-conv1d update`，以及首次请求是否仍出现
`_causal_conv1d_update_kernel` 的 JIT 告警。暖机/捕获中的选择日志不代表真实调用
次数，也不能单凭初始化日志证明性能提升。

本机只有 CPU，未构建/启动 Docker 或验证 L40 性能。FP8 KV scale 告警仍未修复。
`#58763`、L40 本机 FP8 tuning、DFlash 留到此次迁移稳定后再做。

## 开发环境重新打包

源码 checkout 内使用 uv 管理的 Python：

```bash
.venv/bin/python docker/l40_v031_overlay/prepare_overlay.py \
  ../vllm-l40-v031-mtp-graph-offline
```

脚本生成源码覆盖、官方 base/payload 哈希、可复核的补丁、上述三个请求文件、
完整文件校验清单和 tar.gz，并重新读取压缩包核对文件内容。
