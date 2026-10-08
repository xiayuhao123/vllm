# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in startup warmups for the SM89 serving experiment."""

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms import current_platform

logger = init_logger(__name__)


def sm89_extra_warmup_enabled() -> bool:
    return (
        envs.VLLM_ENABLE_SM89_EXTRA_WARMUP
        and current_platform.is_cuda()
        and current_platform.get_device_capability() == (8, 9)
    )


def register_v2_sampler_warmups(registry: JitWarmupRegistry) -> None:
    """Keep upstream ROCm registration and opt in only on CUDA SM89."""
    if not (current_platform.is_rocm() or sm89_extra_warmup_enabled()):
        return
    from vllm.v1.sample.ops.topk_topp_sampler import register_top_k_top_p_warmups

    before = len(registry)
    with registry.activate():
        register_top_k_top_p_warmups()
    logger.info(
        "Registered %d V2 sampler JIT warmup providers "
        "(including native top-k/top-p fallbacks).",
        len(registry) - before,
    )
