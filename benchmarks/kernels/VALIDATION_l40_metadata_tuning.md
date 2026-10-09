# Metadata backport and tuning preparation validation

Base: `38dfd1d138c4483a5fe80084a239d388e43f8016`
(`l40-v0.31.0-mtpgraph1`). Date: 2026-10-09.
The validated image/overlay archive is unchanged.

## Scope

The only runtime source modification is the three guarded slicing changes in
`vllm/v1/attention/backends/gdn_attn.py`, backported from
[PR #58763](https://github.com/vllm-project/vllm/pull/58763).
Pure speculative batches use active-prefix views; mixed prefill paths retain
mask indexing. Per-KV-group ownership, upstream cache fixes and private MTP graph
staging are preserved. No FlashInfer sequence-length synchronization change is
included, and synchronization-count reductions are not claimed.

The FP8 tuner is an offline benchmark, not a runtime dispatch change. It uses the
existing ordinary and Split-K kernels and preserves the current default candidate.
Its report and candidate JSON require L40 measurement and dispatch review before
they can be installed. The #58005 seeds retain their L40S names and provenance.

## Local verification

Platform: macOS ARM64, uv-managed Python 3.12, CPU Torch. No CUDA GPU or Docker
build/run was used. GPU-dependent imports for vLLM regression suites were isolated
by local CPU harnesses; actual CPU tensor slicing, builder methods and utilities
were exercised. These runs are not a complete CUDA vLLM installation test.

| Check | Result |
| --- | --- |
| GDN metadata builder, padding, ragged batches and updated groups | 64 passed |
| Private MTP graph configuration/staging regression | 42 passed |
| Legacy draft graph checks selected for migration | 4 passed |
| Qwen warmup branches | 14 passed, 3 CUDA skips |
| Existing Split-K selection/reduction checks | 11 passed, 3 CUDA skips |
| New tuner plan, candidate rejection, timing, reference and export policy | 10 passed |
| Existing overlay installer compatibility | 7 passed |

Total: 152 passed, 6 CUDA-dependent skips. The GDN view tests failed against the
original gather implementation before the backport and passed after it. The
tuner policy tests likewise went through failing tests before implementation.

Local harnesses are in `/private/tmp/vllm-l40-validation/`:
`run_gdn_tests.py`, `run_graph_tests_v031.py` (also `--legacy`),
`run_warmup_tests_v031.py` and `run_splitk_tests_v031.py`.
Run them with that directory's uv-managed `.venv/bin/python`.
Standalone tuner and installer checks:

```bash
.venv/bin/python -m pytest --noconftest -q \
  tests/kernels/quantization/test_l40_fp8_tuning.py \
  docker/l40_v031_overlay/tests/test_apply_overlay.py
```

An independent read-only code review found no actionable algorithm or integration
defects. All applicable pre-commit hooks passed, including mypy, test tethering,
Ruff, SPDX and Markdown checks. The default GPU-free plan produced 25 cases and
1,248 ordinary candidates before eligible Split-K expansion, with all five seed
shapes present.

## Hardware verification still required

- Ordinary and Split-K candidate correctness and graph replay on actual L40.
- Hot and flushed-L2 timing samples on an idle card in the deployed CUDA image.
- Review of per-shape tile/partition winners, variance and numerical failures.
- One consolidated image and the user's original serving benchmark after useful
  candidates have been selected. No new four-way matrix is requested now.

The tool checks independently dequantized FP32 reference results, eager vs graph
replay output and timing validity. It saves intermediate reports after completed
cases. A kernel timing ratio alone is not evidence of end-to-end throughput gains.
