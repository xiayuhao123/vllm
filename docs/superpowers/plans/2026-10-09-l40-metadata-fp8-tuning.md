# GDN slice backport and L40 FP8 tuning preparation

Base: validated `l40-v0.31.0-mtpgraph1` / `38dfd1d138`.
Keep its source tag, archive and deployment parameters unchanged.

1. Extend `tests/v1/attention/test_gdn_metadata_builder.py` with trailing-padding
   and ragged pure-spec cases. Require non-FULL pure batches to return state and
   accepted-token views rather than gathers; check update_block_table too.
   Preserve mixed batches, per-group ownership and FULL graph staging.
2. Port only #58763's three guarded slice changes to `gdn_attn.py`.
   Keep #58762 metadata reuse and all hybrid-cache/graph fixes. No sampler,
   attention backend, quantization or speculative length changes.
3. Prepare a standalone small-M L40 block-FP8 tuner under `benchmarks/kernels/`.
   Use the existing ordinary and Split-K=4 kernels, five observed N/K shapes,
   actual GPU name and existing 128x128 quantization blocks. Compare full chains
   under CUDA Graph replay, hot and flushed L2 conditions, and check numerics
   against a separate dequantized FP32 reference before timing.
4. Provide a bounded default search plus `--plan` without GPU dependencies.
   Include the current default config and #58005 L40S candidates as explicitly
   labelled seeds. Never rename seed files into installed L40 configurations.
   Record all candidates/failures, environment, cache mode and timing samples.
   Export candidate ordinary-kernel configs with large-M default fallback
   anchors; write Split-K recommendations separately for later dispatch review.
5. Run isolated CPU metadata regression tests, tuner policy/format/timing-order
   tests and pre-commit. State that actual CUDA numerics/timing remain untested
   locally. Deliver a small tool bundle runnable in the existing image, without
   a Docker build or model reload per candidate. Consolidate the final serving
   overlay only after L40 tuning results are available.

Success: slice behavior passes regression checks and the tuner can generate a
validated plan/report/candidate package without altering the serving image.
No performance gain or L40 tuning completion is claimed from CPU tests.
