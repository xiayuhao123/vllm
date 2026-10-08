# L40 offline snapshot artifacts

This directory preserves small artifacts used by the local vLLM v0.30.0 L40
experiment. It does not include Docker images or the 1.6 GB wheel bundle.

- `vllm-l40-v030-mtp-graph-offline.tar.gz`: the exact delivered Python overlay,
  including its Dockerfile, installer, payload/source hashes, launch/log helpers,
  optional GPU primitive check and validation notes. The archive also contains
  the six Python modules, so it can be used independently of a Git checkout.
- `vllm-l40-pcie-fi071-recipe.tar.gz`: FlashInfer offline upgrade Dockerfiles,
  installation/check scripts, source-wheel URLs/hashes and original instructions.
  The archive deliberately excludes wheels; restore them from the manifest or
  the existing external payload image before using a wheel-based build.
- `SHA256SUMS`: checksums for both archives.

Verify and unpack outside the repository if desired:

```bash
sha256sum -c SHA256SUMS
tar -xzf vllm-l40-v030-mtp-graph-offline.tar.gz
tar -xzf vllm-l40-pcie-fi071-recipe.tar.gz
```

For the deployment and user-reported measurements at this checkpoint, see the
[experiment snapshot](../../docs/benchmarking/l40_v030_mtp_graph_snapshot.md).
The model benchmark remains the user's own script. No helper script is required
to enable the graph: add `-e VLLM_ENABLE_SM89_MTP_DECODE_GRAPH=1` to the existing
Docker command before its image argument.
