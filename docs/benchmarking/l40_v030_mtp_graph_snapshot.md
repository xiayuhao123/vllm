# L40 v0.30.0 MTP graph snapshot

This is a local experiment checkpoint, dated 2026-10-08. It preserves the
candidate deployed by the user, not a claim of general performance or numerical
equivalence. No changes to FP8 KV calibration are included.

## Source and deployment

- Upstream base: vLLM v0.30.0, commit `ced6857afa`.
- Earlier L40 A checkpoint: `784f5ee36d` (small-M SM89 block-FP8 split-K,
  benchmark, tests and Docker overlay).
- Runtime: two L40 GPUs, TP2, Qwen3.8-27B-FP8 (the resolved draft architecture
  is dense `Qwen3_5MTP`), BF16 model dtype, FP8 KV, MTP3.
- Image: `vllm/vllm-openai:v0.30.0-l40-a-fi071-mtpgraph1`.
- Dependency base: `vllm/vllm-openai:v0.30.0-l40-a-fi071`, FlashInfer
  `0.7.0.post1`, Torch `2.13.0+cu130`, Python 3.12.
- Enabled: `VLLM_ENABLE_SM89_FP8_SPLIT_K=1`,
  `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, and
  `VLLM_ENABLE_SM89_MTP_DECODE_GRAPH=1`.
- Linear override:
  `--kernel-config '{"linear_backend_per_quant":{"fp8_block_w8a8":"triton"}}'`.
- Capacity: max model length 262144, token budget 8192, GPU memory utilization
  0.90, prefix caching and chunked prefill enabled.

## Changes in this checkpoint

- Register native top-k/top-p JIT warmups for the V2 sampler in the active
  model-runner warmup registry, including FlashInfer fallback coverage.
- Add shape-specific split-K selection logs, explicitly distinct from runtime
  invocation counts.
- Add a default-off independent draft decode graph experiment. The target and
  draft first step keep their original PIECEWISE execution. Later draft steps
  use a private FULL_DECODE_ONLY config and draft-only attention builders.
- Reuse native FlashInfer graph-enabled fixed-buffer wrappers; rebuild metadata
  outside every step. Capture exact request counts 1..8, subject to configured
  limits. Do not replay padded batches or enable fused multi-step.
- Limit eligibility to CUDA SM89, dense Qwen3_5MTP, FlashInfer, TP1/2 and
  PP/DP/DCP/PCP=1. Exclude LoRA, DBO, KV transfer, offloading, watermarks and
  adaptive/dynamic speculation. Log skipped configurations and eager fallback.
- Correct Python executable names in the original Docker overlay instructions.

## User-reported benchmark

The existing script runs five bursts of four concurrent random requests with
16384 input tokens and 1024 output tokens, seeds 12345..12349, no EOS stopping,
and a successful prefix-cache reset before every burst. The client dataset seed
does not fix server sampling; model defaults remain temperature=1, top_k=20,
top_p=0.95. The meaning of the reported plus/minus values was not independently
confirmed from raw result JSONs.

| Metric | MTP graph | L40 A, same switches | Original 0.30.0, switches off |
| --- | --- | --- | --- |
| Mean TTFT (ms) | 9717 ±40 | 9883 ±290 | 13084 ±297 |
| Output throughput (tok/s) | 129.7 ±6.9 | 125.3 ±7.4 | 114.5 ±4.2 |
| Total-token throughput (tok/s) | 2211.7 | 2136.9 | 1952.3 |
| Mean TPOT (ms) | 17.9 ±1.0 | 20.0 ±2.2 | 19.2 ±1.9 |
| Mean ITL (ms) | 49.4 ±0.6 | 52.1 ±3.0 | 51.1 ±2.8 |
| MTP acceptance rate (%) | 59.1 ±6.2 | 53.7 ±5.1 | 56.0 ±5.0 |
| MTP acceptance length | 2.77 ±0.19 | 2.61 ±0.15 | 2.68 ±0.15 |
| Duration (s) | 31.7 ±1.7 | 32.8 ±2.0 | 35.8 ±1.3 |

Against the same-switch L40 A baseline, output throughput increased 3.5%, TPOT
decreased 10.5%, and ITL decreased 5.2%. Acceptance length also increased 6.1%,
and sampler warmup changed concurrently. These measurements do not isolate
graph-only gains or establish statistical significance. Whole-burst throughput
includes prefill and request completion effects.

## Runtime evidence and remaining checks

The supplied service log showed final capture buckets 1..8, followed by actual
request replay for counts 1/2/3/4 and steps 1/2. It also showed successful PCIe
IPC initialization and Triton selection. Startup eager fallbacks occurred before
final capture. The supplied excerpt covers the first two bursts, not all five
raw benchmark runs. No graph capture/replay exception appeared in that excerpt.

The first burst still triggered `_causal_conv1d_update_kernel` JIT. Extending
warmup coverage for real speculative/varlen/cache configurations remains future
work; it is not included in this checkpoint. Missing L40 tuning JSONs,
uncalibrated FP8 KV scales and conservative draft cache-group detection also
remain. KV utilization was about 11% with no queued requests in the excerpt.

Local verification previously covered 55 isolated CPU tests, syntax/Ruff,
installer/source hashes, patch application, shell quoting and package checksums.
CUDA dependencies were isolated with test doubles. Runtime graph execution is
now supported by the user's L40 log; full-model numerical equivalence and broad
quality evaluation are still unverified.

## Saved offline artifacts

The exact small delivered overlay and a FlashInfer upgrade recipe without wheel
binaries are preserved in [docker/l40_snapshots](../../docker/l40_snapshots/README.md).
Source URLs and SHA256 values for the required wheels are in the dependency
recipe. The existing 1.6 GB wheel bundle and Docker images remain external.

To disable only the independent graph, set
`VLLM_ENABLE_SM89_MTP_DECODE_GRAPH=0` and restart the same image. The sampler
warmup and split-K logs remain. A complete rollback uses the previous FI071/A
image. No additional benchmark reruns are required to preserve this checkpoint.
