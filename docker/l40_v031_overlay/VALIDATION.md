# Validation scope

Base: official v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6`.

CPU validation on macOS with a uv-managed Python 3.12 environment:

| Check | Result | Scope |
| --- | --- | --- |
| Independent MTP graph tests | 42 passed | Real policy/speculator source; GPU imports and allocations isolated |
| Existing autoregressive graph regression cases | 4 passed | Selected upstream replay cases, 17 other cases deselected |
| Additional sampling and Qwen update warmup tests | 14 passed, 3 skipped | Real registry/warmup source, convolution launch replaced for CPU tests |
| Split-K selection tests | 11 passed, 3 skipped | Real selector, GPU arithmetic not executed |
| Overlay installer tests | 7 passed | Source/payload/syntax validation, repeat installation, version rejection |

Total: **78 CPU cases passed**. Six GPU cases were skipped. CPU import harnesses
isolate unavailable CUDA dependencies and are not a full installed-vLLM test
run. In particular, mocked convolution calls prove scratch-cache construction
and call arguments, not actual kernel compilation or arithmetic correctness.

All applicable staged pre-commit hooks passed, including Ruff, markdownlint,
shellcheck, SPDX, test tethering and mypy for Python 3.10. Non-applicable hooks
were skipped normally. No hooks were bypassed.

The upstream base speculator, autoregressive graph manager, attention metadata
builder, KV-cache managers and V2 sampler were checked byte-for-byte against
v0.31.0. Original FP8 helper definitions were preserved except the explicit
Split-K dispatch addition. The modified autoregressive speculator retains
`dummy_run=dummy_run` and `input_batch.decode_graph_eligible`.

The generated patch is checked/applied to exact official source files, then
compared with all seven payload files. The installer is applied twice to that
official-source fixture. Package entries, payload hashes and tar contents are
verified before delivery.

No Docker build, Docker daemon startup, L40 execution, model-quality evaluation,
prefix-cache service smoke check or end-to-end benchmark was run locally.
These hardware-dependent checks remain for deployment. Reuse the user's
original five-round workload and perform the three optional shared-prefix
requests in the same container; no four-way experiment is required.
