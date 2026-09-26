# L40 block-FP8 split-K experiment

This experiment is based on vLLM v0.30.0. Build it as a thin Python overlay on
the matching prebuilt `vllm/vllm-openai:v0.30.0` image (or its `-cu129` variant
if that is the CUDA runtime your cluster uses). The Dockerfile rejects a base
image with a different vLLM version. From this source checkout, for example:

```bash
docker build --build-arg BASE_IMAGE=vllm/vllm-openai:v0.30.0-cu129 \
  -f docker/Dockerfile.l40_sm89_split_k \
  -t registry/vllm-openai:l40-sm89-v030 .
docker push registry/vllm-openai:l40-sm89-v030
```

`VLLM_ENABLE_SM89_FP8_SPLIT_K=1` enables an experimental Triton path for the
small-M block-FP8 GEMMs in the 27B FP8 TP2 workload:

- `(M, N, K) = (<=32, 5120, 8704)` for the MLP down projection;
- `(M, N, K) = (<=32, 5120, 3072)` for the GDN output projection.
- `(M, N, K) = (<=32, 8192, 5120)` is an additional candidate from the
  supplied logs. It requires `VLLM_ENABLE_SM89_FP8_SPLIT_K_N8192_K5120=1`.

The path is restricted to CUDA SM89 and 128x128 FP8 blocks. It runs four
K-aligned partial GEMMs, stores FP32 partial outputs, then performs a fixed
order reduction and casts once to the requested output type. The original
kernel remains the default for every other GPU, shape, quantization layout,
and for `VLLM_BATCH_INVARIANT=1`.

Use it only after comparing throughput and output correctness against the
default on the target L40 deployment:

```bash
VLLM_ENABLE_SM89_FP8_SPLIT_K=1 vllm serve ...
```

Set `VLLM_ENABLE_SM89_FP8_SPLIT_K_N8192_K5120=1` as well to enable the
additional 8192x5120 candidate. The image build checks that the base vLLM
distribution is exactly 0.30.0 before copying the patched Python files.

The image includes a standalone benchmark. It prints the loaded vLLM version
and module path, checks output against the original kernel, then compares the
full kernel chain, including the FP32 partial buffer and reduction:

```bash
python /opt/vllm-l40/benchmark_l40_sm89_block_fp8_split_k.py --m 4 --shape mlp --check-only
python /opt/vllm-l40/benchmark_l40_sm89_block_fp8_split_k.py --m 4 --shape mlp
python /opt/vllm-l40/benchmark_l40_sm89_block_fp8_split_k.py --m 4 --shape gdn
python /opt/vllm-l40/benchmark_l40_sm89_block_fp8_split_k.py --m 4 --shape n8192
```

It does not predict end-to-end decode gains: all-reduce, attention/GDN kernels,
and request scheduling can dominate at a given concurrency.

For end-to-end evaluation, keep the model, TP=2 placement, request set,
sampling parameters, concurrency, context lengths, and speculative settings
identical. First run without speculative decoding, then repeat with DFlash2.
First inspect the startup line `Selected ... for Fp8LinearMethod`. v0.30.0 can
select another block-FP8 linear backend before Triton. If the model does not
use `TritonFp8BlockScaledMMKernel`, the A flag cannot affect that layer. Add
the following option to both Triton comparison runs:

```bash
--kernel-config '{"linear_backend_per_quant":{"fp8_block_w8a8":"triton"}}'
```

Restart the Pod for each row so the environment and captured graphs are fixed.
Use the same image for every v0.30.0 run:

| Run | block-FP8 linear backend | A flag | PCIe all-reduce flag |
| --- | --- | --- | --- |
| v0.30.0 baseline | auto | unset | unset |
| Triton baseline | Triton override above | unset | unset |
| Triton + A | Triton override above | `1` | unset |
| PCIe all-reduce only | auto | unset | `1` |
| Both, if each helps | Triton override above | `1` | `1` |

Compare Triton + A with both the Triton baseline (A's incremental gain) and
the auto baseline (whether the change is useful for production). Also record
the existing v0.28.0 deployment as a reference before replacing it.

The PCIe all-reduce backend is separately opt-in in v0.30.0. Confirm from
startup logs that it initialized and was included in the selected all-reduce
backends; if it falls back, the run is not a valid test of that backend. For A,
confirm that an SM89 split-K kernel appears in a GPU trace during model decode.
The standalone benchmark proves only that the function can select the path;
it cannot prove the model routes through it.

Record Pod output tokens/s, per-request TPOT and TTFT (including p50/p95),
GPU memory, and speculative acceptance length. Reject any speedup that comes
with unacceptable output differences or a regression in the long-context
workload. The default kernel remains available by removing the A environment
variable, so a rollback does not require another image build.
