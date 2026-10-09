# vLLM 0.31.0 L40 DFlash2 前缀缓存候选

从你已测试的 tuned1 开始，保留原 FP8 tuning、PCIe IPC、GDN metadata、MTP
graph 和预热。新增可关闭的 DFlash2 草稿缓存组识别；没有混入 tuned2/3。
只覆盖 Python 文件，不重编 CUDA，不下载 FlashInfer 包。

## 构建

在解压目录执行，本机需已有官方原始 0.31.0 镜像：

```bash
sha256sum -c SHA256SUMS
docker build --progress=plain --pull=false --network=none \
  --build-arg BASE_IMAGE=vllm/vllm-openai:v0.31.0 \
  -t vllm/vllm-openai:v0.31.0-l40-dflashcache1 .
```

安装器核对版本、FlashInfer 和全部覆盖文件的官方哈希。不在已有 tuned1 镜像上
直接叠加；新包已经包含 tuned1。旧镜像保留用于回退。

## 手动部署

沿用此前 DFlash probabilistic 命令，只换镜像及容器名，并加入新开关、两个
上游前缀缓存参数。完整示例如下：

```bash
docker run --rm \
  --name qwen38-fp8-31-l40-dflashcache1 \
  --gpus '"device=2,3"' \
  --ipc=host \
  -p 18013:8000 \
  -v /model:/model \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 \
  -e VLLM_ENABLE_SM89_FP8_SPLIT_K=1 \
  -e VLLM_ENABLE_SM89_MTP_DECODE_GRAPH=0 \
  -e VLLM_ENABLE_SM89_EXTRA_WARMUP=1 \
  -e VLLM_ENABLE_DFLASH2_PREFIX_CACHE=1 \
  vllm/vllm-openai:v0.31.0-l40-dflashcache1 \
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
  --speculative-config '{"method":"dflash","model":"/model/Qwen/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"attention_backend":"FLASH_ATTN","kv_cache_dtype":"bfloat16","draft_sample_method":"probabilistic"}' \
  --kernel-config '{"linear_backend_per_quant":{"fp8_block_w8a8":"triton"}}' \
  --mamba-cache-mode align \
  --prefix-match-unit 16 \
  --enable-mamba-shared-prefix-checkpoint \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192
```

保持 `disable_eagle_block_drop` 为默认 false。本候选保留草稿尾部重算保护。
新开关只识别已知 DFlash2 完整层范围；不支持的架构、层名字或分组保留上游回退。
`prefix-match-unit` 只改变匹配粒度，不能凭空生成 Mamba 状态。

启动应出现 `DFlash2 prefix-cache draft groups identified`。若仍出现无法识别
draft group 的告警，保存日志，此时新识别没有覆盖实际布局。新配置把代码和两个
上游参数一起验证，若有提升，不能单独归因给组识别。

## 先做六个缓存检查

空闲服务上执行一次 reset，之后六次请求之间不要清缓存：

```bash
BASE_URL=http://127.0.0.1:18013
curl -fsS -X POST "$BASE_URL/reset_prefix_cache"
for N in 1 2 3 4 5 6; do
  curl -fsS "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@prefix-check-${N}.json" \
    -o "prefix-response-${N}.json" || break
done
bash collect_logs.sh qwen38-fp8-31-l40-dflashcache1 dflashcache1-logs
```

期望正文依次是：`L40-DOC-ALPHA`、`L40-DOC-ALPHA`、`BEACON-GREEN`、
`L40-DOC-ALPHA`、`BEACON-RED`、`L40-DOC-BETA`，详见 `PREFIX_EXPECTED.json`。
都应正常 stop，无截断或空正文。第二次应有缓存命中；后续分支必须读到各自文档
及历史的正确值。第五次修改尾部仍有较长共享前缀，第六次修改开头应减少命中。
检查 `cached_tokens`、TTFT、输出内容及启动日志。

这些输入略有变化，不能把 TTFT 与旧固定标记的三次检查直接算提升百分比。可先
用原来的三份请求复测，得到同输入对照，再执行新六份检查。无需重做旧四组矩阵。

确认缓存输出正确后，再按原方式跑童话或业务任务。不用我们的自动部署脚本。
童话脚本里将 `SERVER_CONTAINER` 改成新容器名，防止抓到其他引擎日志。
此前 4096-token 限制出现空正文，完整任务比较需要保存停止原因和思考用量，或
使用足够大的输出上限；保留旧结果，不把截断样本计作完整故事。

## 回退和验证边界

设新环境变量为 0 可关闭新增识别，但细粒度匹配参数仍然存在。要完全恢复此前
基线，使用旧 tuned1 镜像和原部署命令，去掉本次三个新增开关/参数。
本机只验证 CPU 控制流与覆盖包；尚未在 L40 上证明命中提升、状态复制正确性
或真实业务输出质量。先在测试服务验证，不能仅凭命中率升高就投入生产。
