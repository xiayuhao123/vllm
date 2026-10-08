# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def make_independent_mtp_decode_config(
    config: VllmConfig,
    *,
    backend_names: set[str],
    min_cg_support: AttentionCGSupport,
) -> tuple[VllmConfig | None, str]:
    """Resolve the limited SM89 experiment without mutating target config."""
    if not envs.VLLM_ENABLE_SM89_MTP_DECODE_GRAPH:
        return None, "disabled"
    if not current_platform.is_cuda() or current_platform.get_device_capability() != (
        8,
        9,
    ):
        return None, "requires CUDA SM89"
    spec = config.speculative_config
    if spec is None or spec.method != "mtp" or spec.num_speculative_tokens < 2:
        return None, "requires autoregressive MTP with at least two draft tokens"
    if spec.draft_model_config.hf_config.architectures != ["Qwen3_5MTP"]:
        return None, "only dense Qwen3_5MTP is covered by this experiment"
    if config.model_config.enforce_eager or spec.enforce_eager:
        return None, "enforce_eager is enabled"
    compilation = config.compilation_config
    if compilation.cudagraph_mode != CUDAGraphMode.PIECEWISE:
        return None, "target mode must be PIECEWISE"
    if not compilation.cudagraph_capture_sizes:
        return None, "no capture sizes configured"
    if backend_names != {"FLASHINFER"}:
        return None, "draft attention must exclusively use FLASHINFER"
    if min_cg_support.value < AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE.value:
        return None, "draft attention does not support single-token decode graphs"
    parallel = config.parallel_config
    if parallel.tensor_parallel_size not in (1, 2) or any(
        getattr(parallel, name) != 1
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        return None, "requires TP1/2 and PP/DP/DCP/PCP=1"
    if parallel.enable_dbo or config.lora_config or config.kv_transfer_config:
        return None, "DBO, LoRA and KV transfer are outside the experiment"
    if getattr(config, "watermark_config", None) is not None:
        return None, "watermarked sampling is outside the experiment"
    offload = getattr(config, "offload_config", None)
    if offload and (
        offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0
    ):
        return None, "weight offloading is outside the experiment"
    if spec.enable_adaptive_verification or spec.uses_dynamic_speculative_decoding():
        return None, "adaptive/dynamic speculation is outside the experiment"

    limit = min(
        8,
        config.scheduler_config.max_num_seqs,
        config.scheduler_config.max_num_batched_tokens,
        max(compilation.cudagraph_capture_sizes),
        compilation.max_cudagraph_capture_size
        if compilation.max_cudagraph_capture_size is not None
        else 8,
    )
    if limit < 1:
        return None, "no draft capture capacity"
    draft = copy.copy(config)
    draft.compilation_config = copy.copy(compilation)
    draft.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    # Exact small request counts avoid advancing the state of graph padding rows.
    draft.compilation_config.cudagraph_capture_sizes = list(range(1, limit + 1))
    draft.compilation_config.max_cudagraph_capture_size = limit
    return draft, "enabled"


def exact_draft_graph_batch(
    desc: BatchExecutionDescriptor, num_reqs: int
) -> BatchExecutionDescriptor:
    """Fall back before replay if a draft graph would execute padding rows."""
    if desc.cg_mode == CUDAGraphMode.FULL and (
        desc.num_reqs != num_reqs or desc.num_tokens != num_reqs
    ):
        return replace(
            desc, cg_mode=CUDAGraphMode.NONE, num_tokens=num_reqs, num_reqs=num_reqs
        )
    return desc


def check_independent_decode_metadata(metadata: dict[str, Any] | None) -> None:
    """Do not replay a graph against dynamic/prefill FlashInfer metadata."""
    if not metadata or any(
        getattr(value, "num_prefills", None) != 0
        or not getattr(
            getattr(getattr(value, "decode", None), "wrapper", None),
            "is_cuda_graph_enabled",
            False,
        )
        for value in metadata.values()
    ):
        raise RuntimeError(
            "Independent MTP replay requires graph-enabled FlashInfer decode "
            "metadata. Disable VLLM_ENABLE_SM89_MTP_DECODE_GRAPH to use the "
            "original path."
        )
