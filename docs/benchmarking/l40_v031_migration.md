# L40 v0.31.0 migration candidate

The source base is official v0.31.0 (`db9527a46873454610df6dbedf79a36d6bf1a7f6`).
The validated v0.30.0 checkpoint remains `l40-v0.30.0-mtpgraph1`.

This candidate retains the existing opt-in small-M block-FP8 Split-K and
independent MTP decode graph experiments. Split-K still requires Triton,
SM89, 128x128 quantization blocks, M <= 32 and the supported N/K shapes.
The 8192x5120 shape remains separately disabled by default. MTP graph eligibility
remains limited to dense Qwen3_5MTP, FlashInfer, target PIECEWISE, TP1/2, and
exact request counts 1..8; padded or larger batches use the original path.

The overlay leaves the upstream sampler unchanged, including custom logits
processors and the new random-number generator. It also leaves draft-layer
annotations, cache allocation/reuse, sparse-cache checkpoint retention,
dummy-run slot mappings and decode graph eligibility unchanged. In particular,
the upstream speculative FlashInfer D2H metadata path is not modified.
PCIe IPC all-reduce remains an upstream FlashInfer backend, enabled through
`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, not a custom kernel in this overlay.

`VLLM_ENABLE_SM89_EXTRA_WARMUP=1` enables additional startup work on CUDA SM89:

- Register native V2 sampling fallback providers using the runner's registry.
  Upstream ROCm registration still works with this flag disabled.
- Compile Qwen convolution update using scratch states with the loaded cache's
  exact strides. Cover ordinary decode and speculative varlen updates, request
  counts 1/4/16, packed and contiguous projection rows, and full/mixed query
  lengths. MTP=3 uses maximum query length 4 and speculative index stride 4.
  The convolution has no bias in the Qwen GDN implementation. Its update calls
  do not pass convolution APC block-index arguments; no hypothetical APC update
  path is introduced. Upstream prefill and prefix-cache behavior is retained.

The extra-warmup flag is off by default and does not enable Split-K or graphs.
Pooling skips update warmup. Insufficient speculative state length skips the
speculative warmup with a log, without altering cache allocation. Scratch state
allocations use at most 17 cache rows. Startup logs report warmup launches,
not an inference invocation counter or a guarantee that every possible JIT key
has been compiled.

The warmup targets first-request JIT spikes; it does not establish a steady
decode improvement. This candidate has not been tested on local CUDA hardware.
FP8 KV scaling remains uncalibrated if the checkpoint lacks scales.

Build/deployment instructions are in `docker/l40_v031_overlay/README.md`.
Use the same original five-round benchmark and v0.30.0 benchmark client image,
changing only server image and result name. The optional three shared-prefix
requests should run after that benchmark, without resetting between requests.
They are a small cache/output smoke check, not a model-quality evaluation.
Cross-version sampled outputs need not match exactly because v0.31.0 updates
the sampler RNG; acceptance variation still limits performance attribution.

Deferred until this migration is validated: #58763, local L40 FP8 tuning,
DFlash, KV-scale calibration and speculative metadata synchronization changes.
