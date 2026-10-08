# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.model_executor.warmup.sm89_warmup import register_v2_sampler_warmups
from vllm.platforms import current_platform
from vllm.v1.sample.ops import topk_topp_sampler


@pytest.mark.parametrize(
    "enabled,cuda,rocm,capability,expected",
    [
        (False, True, False, (8, 9), 0),
        (True, True, False, (8, 9), 1),
        (True, True, False, (9, 0), 0),
        (True, False, False, None, 0),
        (False, False, True, None, 1),
        (True, False, True, None, 1),
    ],
)
def test_registration_keeps_rocm_and_limits_cuda_to_opt_in_sm89(
    monkeypatch, enabled, cuda, rocm, capability, expected
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_SM89_EXTRA_WARMUP", enabled)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setattr(current_platform, "get_device_capability", lambda: capability)
    registered: list[object] = []

    def register():
        # Registration must happen inside this runner's registry context.
        JitWarmupRegistry.register(provider)
        registered.append(provider)

    provider = object()
    monkeypatch.setattr(topk_topp_sampler, "register_top_k_top_p_warmups", register)
    registry = JitWarmupRegistry(SimpleNamespace())
    register_v2_sampler_warmups(registry)
    assert len(registry) == expected
    assert len(registered) == expected
