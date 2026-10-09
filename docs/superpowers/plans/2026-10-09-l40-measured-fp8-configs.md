# Apply measured L40 FP8 configs

Base: `f26acbffb6`. User report SHA-256:
`c4f299a48a97e5882a3d37877c54fbfa7e8e2030742fdcc7e75a1d5cb16d8d7e`.

1. Verify 25 unique cases, all 1,997 candidate records, replay/numeric status,
   45 samples per cache mode, kernel SHA against the deployed source and winners.
   Save a compact report with medians and candidate provenance.
2. Write failing CPU tests for exact M/N/K lookup, unmeasured/large-M fallback,
   opt-out, L40S/other-platform exclusion, batch-invariant exclusion and BF16-only
   output. Add a wrapper dispatch test proving tuned tiles skip Split-K and
   opt-out retains previous dispatch. Use the existing kernel test file.
3. Add one pure configuration module with exactly 25 measured tuples and an
   opt-in environment flag. Gate the ordinary Triton wrapper before its original
   JSON/default config selection. Preserve existing kernels and all fallbacks.
   Cache device-name lookup to avoid repeated NVML calls in the launch path.
4. Extend the official-v0.31 overlay module list with GDN and the configuration
   module; set the new flag to 1 in the candidate image's Dockerfile. Preserve the
   original mtpgraph1 archive. Document one build and the unchanged five-run
   benchmark, plus how to turn off measured tuning without disabling old Split-K.
5. Run CPU regression, installer checks, source/tag fixture installation and
   all pre-commit hooks; review; save a Git checkpoint and a new overlay archive.
   Do not run Docker on this Mac. User CUDA microbenchmark evidence does not
   establish end-to-end model accuracy or serving speed; retain that limitation.

Cold-L2 winners are prioritized for weights streamed through a full model. This
is a deployment hypothesis, not a measured cache-hit rate. Some tuned Split-K
variants win hot-cache microbenchmarks. All selected ordinary variants improve
both modes vs the current baseline, so no global Split-K removal is warranted.
