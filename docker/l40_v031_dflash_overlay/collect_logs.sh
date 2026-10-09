#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${1:-qwen38-fp8-31-l40-dflashcache1}"
OUT_DIR="${2:-l40-logs-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

docker inspect --format 'image={{.Config.Image}} image_id={{.Image}}' "$CONTAINER" \
  > "$OUT_DIR/runtime.txt"
docker inspect --format '{{json .Config.Cmd}}' "$CONTAINER" >> "$OUT_DIR/runtime.txt"
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" |
  grep -E '^(VLLM_ENABLE_SM89_FP8_SPLIT_K(_N8192_K5120)?|VLLM_ENABLE_SM89_FP8_TUNING|VLLM_ENABLE_DFLASH2_PREFIX_CACHE|VLLM_ENABLE_SM89_MTP_DECODE_GRAPH|VLLM_ENABLE_SM89_EXTRA_WARMUP|VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC|VLLM_ENABLE_CUDA_COMPATIBILITY|VLLM_USE_V2_MODEL_RUNNER|VLLM_CACHE_ROOT|TRITON_CACHE_DIR|FLASHINFER_WORKSPACE_BASE)=' \
  >> "$OUT_DIR/runtime.txt" || true
docker exec "$CONTAINER" python3 -c 'import importlib.metadata as m; print("vLLM:",m.version("vllm")); print("FlashInfer:",m.version("flashinfer-python")); print("Torch:",m.version("torch"))' \
  >> "$OUT_DIR/runtime.txt"

docker logs --timestamps "$CONTAINER" > "$OUT_DIR/server.log" 2>&1
grep -Ei 'non-default args|DFlash2 prefix-cache draft groups|no KV cache group|kv lcm block sizes|kv cache group sizes|Using V2|Registered.*V2 sampler|Independent MTP|Warmed Qwen causal-conv1d update|Skipping speculative Qwen conv-update|_causal_conv1d_update_kernel|Selected.*Fp8|SM89 FP8 split-K selected|L40 FP8 measured config selected|all-reduce backends|PCIe IPC|JIT kernel warmup|_topk_topp|_topp_sb|JIT compilation during inference|CUDAGraphMode|Capturing.*graphs|Fused multi-step|GDN.*kernel|Enabled custom fusions|FP8 kernel config|scaling factor|uncalibrated|autotun|compile.*cache|compilation.*took|engine.*took|Application startup complete' \
  "$OUT_DIR/server.log" > "$OUT_DIR/key.log" || true
cat "$OUT_DIR/runtime.txt"
cat "$OUT_DIR/key.log"
printf '\nSaved runtime.txt, key.log and full server.log in %s\n' "$OUT_DIR"
