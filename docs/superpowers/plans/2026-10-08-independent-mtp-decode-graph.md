# Independent Qwen MTP decode graph

**Goal:** An opt-in SM89 path capturing MTP steps 2+ with a private FULL_DECODE_ONLY config while the target remains PIECEWISE.

**Design:** Reuse FlashInfer's existing graph-enabled fixed-address decode wrappers and `fast_plan_decode`, rebuilding metadata outside each graph replay. Scope to dense Qwen3_5MTP, CUDA SM89, TP1/2, single-node DP/PP/DCP/PCP=1, no LoRA, adaptive/dynamic speculation, DBO or KV transfer. Capture exact request counts 1..8 (respecting scheduler and configured capture limits); never replay a padded graph. All other cases retain the original execution path. Do not swallow capture/runtime CUDA errors.

- [x] Add CPU tests for private config identity/isolation, disabled/ineligible paths, capture limits, exact-batch fallback, native graph wrapper invariants, and metadata-before-each-replay order.
- [x] Confirm tests fail with missing policy/graph behavior.
- [x] Add `VLLM_ENABLE_SM89_MTP_DECODE_GRAPH` (default false) and a small policy helper in the autoregressive speculator package.
- [x] Rebuild only the drafter's metadata builders with the private config after inspecting their real backend/capture support. Use that same config for the draft decode graph manager. Leave target and draft prefill managers unchanged.
- [x] Keep fused multi-step disabled on the experimental path. Reject non-graph metadata before replay; log enablement, actual capture buckets, replay and fallback.
- [x] Extend CPU regression for ordinary graph execution unchanged when the flag is off. Run syntax/Ruff/diff checks and clearly separate mocked-control-flow tests from actual GPU validation.
- [x] Produce one combined offline overlay on the user's FI071/A image, including the previous sampler warmup and split-K logs; validate original/patched hashes and allow absent new helper modules.
- [x] Include launch/log scripts, flag-on/off verification instructions using the existing benchmark workload, and startup/correctness caveats. Verify package application, shell quoting, manifests and archive.

**GPU validation boundary:** No local CUDA/L40 is available. Include a short smoke check of seeded greedy generation with the flag on/off on the same image, and then the user's existing stochastic benchmark. The CPU tests do not establish Triton execution, graph pointer correctness, numerical equivalence or throughput. Mark the candidate experimental and leave the runtime flag off by default.
