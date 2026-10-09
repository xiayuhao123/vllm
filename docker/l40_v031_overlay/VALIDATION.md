# L40 tuned1 verification

Base image: official `vllm/vllm-openai:v0.31.0`, source tag
`db9527a46873454610df6dbedf79a36d6bf1a7f6`. Validated previous checkpoint:
`38dfd1d138`; metadata/tuner checkpoint: `f26acbffb6`.

## User L40 measurement evidence

Report SHA-256:
`c4f299a48a97e5882a3d37877c54fbfa7e8e2030742fdcc7e75a1d5cb16d8d7e`.
Original kernel source SHA-256:
`ba23d416e2ca047ed26639a8a639d0b4cfe82b20dfcaf47f9df57a85bd085a41`.
Matched against this worktree before modifying dispatch.

The report identifies NVIDIA L40, Python 3.12.3, Torch 2.13.0+cu130, CUDA 13.0,
Triton 3.7.1, vLLM 0.31.0 and FlashInfer 0.7.0.post1.
All 1,997 candidates across 25 M/N/K cases have status `ok`, pass independent
FP32/BF16 reference thresholds and exact eager/graph replay checks. Each has
45 finite positive hot and 45 flushed-L2 timing samples. There are no rejected
candidate records. This is the tool's numerical criterion, not a model eval.

All cold-L2 winners use ordinary GEMM and also improve hot timing vs the deployed
baseline. Some other tuned Split-K candidates win hot timing, so selecting
cold-L2 winners is a deployment hypothesis for streamed model weights, not a
measurement of the model's actual cache behavior. Split-K remains available for
unmeasured batches and when tuning is disabled.

Representative M=4 medians, microseconds:

| N / K | Baseline cold | Selected cold | Baseline hot | Selected hot |
| --- | --- | --- | --- | --- |
| 5120 / 3072 | 42.85 | 37.31 | 22.18 | 11.78 |
| 5120 / 8704 | 104.32 | 92.99 | 47.46 | 33.06 |
| 8192 / 5120 | 115.62 | 88.10 | 61.70 | 22.14 |
| 17408 / 5120 | 196.99 | 178.53 | 86.62 | 44.93 |
| 7168 / 5120 | 110.30 | 76.80 | 61.09 | 27.97 |

The exact selected parameters and provenance are saved in `FP8_SELECTION.json`.
Only M=1/4/8/16/32 and these five N/K shapes are installed. No nearest-M lookup
or prefill extrapolation is used; only BF16 output, E4M3 inputs and FP32 scales
on L40 SM89 with 128×128 quantization blocks are eligible. Hardware checks use
A.device's index and cached queries, including nonzero GPU indices. Source flag
defaults off; the candidate image explicitly sets it on. Batch-invariant is excluded.

## Local verification

Mac CPU, uv-managed Python: GDN 64, private MTP graph 42, legacy graph 4,
warmup 14, existing plus measured FP8 policy/dispatch 23, tuner 10 and installer 7
tests pass: **164 passed, 6 CUDA skips**. GPU-dependent vLLM imports are isolated
by local harnesses; these are CPU tensor/control-flow tests, not a CUDA serving
test. New tests first failed for missing measured lookup and dispatch, then passed.
The BF16 exact-M path, opt-out, unmeasured M, FP16 fallback and another GPU type
on a nonzero index use the real wrapper with launches intercepted for dispatch.

Packaging checks all nine payload modules against official tag source,
installs the overlay twice on that source fixture, checks hash manifests and
Python syntax, compares all 25 selected configs to the report, and verifies tar
contents. No sampler source changes are included. #58762 group metadata reuse,
upstream hybrid-cache fixes, existing graph and conv-update warmup are preserved.

All applicable pre-commit hooks passed. Independent review found and resolved a
nonzero-device guard issue; the final review has no unresolved actionable findings.
The original `mtpgraph1` archive is retained. No Docker daemon/build/run was used
locally. No new CUDA extension or FlashInfer dependency package is needed.

## Remaining serving validation

Build the candidate image once, keep the user's deployment parameters and original
five-run benchmark, changing only image and result name. Check measured-config
selection in startup/capture logs, PCIe initialization and MTP replay. Runtime
speed, speculative acceptance and model quality cannot be inferred from this
microbenchmark alone. The three supplied chat-completions prefix checks remain
a small output/cache smoke check. FP8 KV scaling is still uncalibrated.
