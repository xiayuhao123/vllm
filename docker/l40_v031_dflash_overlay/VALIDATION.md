# DFlash2 prefix-cache candidate verification

Base: official v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
Existing L40 changes: tuned1, `85c1c3c281`. Tuned2/tuned3 are not included.

The new code changes group ownership only, behind
`VLLM_ENABLE_DFLASH2_PREFIX_CACHE` (default off). It requires the resolved
`DFlash2DraftModel` architecture, method `dflash`, the complete attention-layer
range registered after the target's layers, an exact group partition, and no
Mamba spec in any identified draft group. Every matching draft group retains
tail-drop protection. Other methods and incomplete layouts retain upstream
behavior. The original 3296-token physical alignment is not changed.

CPU regression checks use real cache spec dataclasses and the actual annotation
functions, with GPU imports isolated by a source-loading harness. The positive
test failed before the implementation; all 15 checks passed after it. Checks
cover split and mixed attention groups, missing and duplicated layers,
unsupported architecture/method, a Mamba draft spec/group, an unknown namespace,
opt-out and the existing MTP trailing-layer rule. Two checks run the real
`get_kv_cache_groups()` through packed and uniform-page-size hybrid grouping.

Unchanged tuned1 regressions: 42 independent MTP graph checks and 23 FP8 dispatch
checks passed; 3 CUDA tests were skipped. Total: 80 CPU checks passed, 3 skipped.
All overlay hashes were checked, installation into an official-tag source
fixture succeeded twice, and a modified base was rejected. This verifies Python
source packaging, not a built container's CUDA runtime.

The six JSON requests check a cold request, identical repetition, chat extension,
an alternative extension, a changed document tail and a changed document head.
Expected answers depend on document contents rather than repeating a literal
marker in the instruction. Temperature 0 and disabled thinking are confined to
these cache checks; the existing fairytale benchmark sampling is unchanged.

Fine matching and Mamba shared-prefix checkpoints use official upstream code.
CPU checks cannot prove their GPU state copies or model outputs correct.
Acceptance, cache hits, TTFT and numerical serving behavior require user L40
validation before production use. Identifying groups alone is not evidence of a
performance improvement.
