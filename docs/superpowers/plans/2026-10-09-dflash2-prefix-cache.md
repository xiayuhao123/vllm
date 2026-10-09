# DFlash2 prefix-cache experiment

**Goal:** Improve multi-turn prefix reuse without disabling draft-tail protection.

**Base:** Deployed `85c1c3c281` / `l40-v0.31.0-tuned1`; the MTP history is
saved separately at `l40-v0.31.0-mtp-checkpoint-20261009`.

**Architecture:** An opt-in annotator identifies all five Qwen DFlash2 attention
layers from the exact range registered by `DFlashQwen3ForCausalLM`. It requires
the resolved `DFlash2DraftModel` architecture, complete layer coverage and an
exact group partition. Unsupported or incomplete layouts retain the upstream
conservative fallback. No KV layout, kernel, sampler or checkpoint algorithm
changes. Fine matching and shared-prefix checkpoints use upstream options.

## Contract and test design

The annotator supplies group ownership, not cached tensor contents. Its input is
the global config, per-layer specs and groups; its output is `is_eagle_group`
flags. A false positive could allow stale draft KV or an incompatible Mamba
state to be reused, so ambiguous inputs must preserve the old fallback. CPU
annotation tests using actual cache spec types are the cheapest first check;
L40 repeated/extended chat requests remain necessary for serving correctness.

## Steps

- [x] Extend `tests/v1/core/test_kv_cache_utils.py` with a five-layer DFlash2
  case split across groups, a mixed target/draft attention group, opt-out,
  incomplete/duplicated partitions, missing draft layers and unsupported
  architecture/method cases. Watch the positive annotation assertion fail.
- [x] Add `VLLM_ENABLE_DFLASH2_PREFIX_CACHE` (default off) in `vllm/envs.py`.
  In `vllm/v1/core/kv_cache_utils.py`, validate the complete registered range
  `model.layers.<target_count .. target_count + draft_count>.self_attn.attn`
  against attention specs and annotate every group containing those layers.
  Never annotate a group containing Mamba; do not alter existing MTP rules.
- [x] Test the full annotation function and source-loading guard on CPU. Re-run
  tuned1 graph and FP8 dispatch checks. Check all relevant pre-commit hooks.
- [x] Package a separate official-v0.31.0 Python overlay named
  `vllm-l40-v031-dflashcache1-offline`; verify hashes and idempotent installation
  into an official-tag fixture. No local Docker startup or CUDA rebuild.
- [x] Document manual deployment with the new opt-in flag, upstream
  `--prefix-match-unit 16` and `--enable-mamba-shared-prefix-checkpoint`, while
  keeping `disable_eagle_block_drop=false`. These are one combined experimental
  configuration; improvements cannot be attributed to annotation alone.
- [x] Supply repeated/extended chat requests and a small shared-prefix branch
  check. The user runs them on L40, checking expected output, cached tokens and
  TTFT before a fairytale benchmark. No old four-way matrix is required.

- [ ] User L40 serving validation: outputs, branch correctness, cache hits and
  TTFT. Local CPU checks and review do not complete this step.

## Limits

Identifying groups alone need not improve the measured 6592-token coarse hit:
the physical 3296-token alignment remains. Fine matching does not create Mamba
states at arbitrary positions. This candidate targets reuse correctness and
prefill/TTFT; no steady decode speedup or production readiness is claimed from
CPU tests.
