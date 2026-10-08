# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_causal_conv1d_update_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
)
from vllm.platforms import current_platform


def _cuda_gdn_config(device=None, state_tokens=3) -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    if device is None:
        device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, state_tokens),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


@pytest.mark.parametrize("num_speculative_tokens", [0, 3])
@pytest.mark.parametrize("transpose_state", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_update_warmup_uses_scratch_cache_and_real_decode_layouts(
    monkeypatch, num_speculative_tokens, transpose_state, state_dtype
):
    from dataclasses import replace

    from vllm.model_executor.layers.mamba.ops import causal_conv1d

    config = _cuda_gdn_config(torch.device("cpu"), 3 + num_speculative_tokens)
    config = replace(
        config,
        conv_state=config.conv_state.to(state_dtype),
        conv_dtype=state_dtype,
        conv_weight_dtype=torch.bfloat16,
        norm_weight_dtype=torch.float32,
    )
    if transpose_state:
        config = replace(
            config,
            conv_state=config.conv_state.transpose(1, 2).contiguous().transpose(1, 2),
        )
    config.conv_state.fill_(42)
    before = config.conv_state.clone()
    calls = []

    def update(x, state, weight, bias, activation, **kwargs):
        assert state.data_ptr() != config.conv_state.data_ptr()
        assert state.stride() == config.conv_state.stride()
        assert weight.shape == (config.conv_dim, config.conv_kernel_size)
        assert weight.dtype == torch.bfloat16
        assert state.dtype == state_dtype
        assert bias is None
        assert activation == "silu"
        assert x.stride(1) == 1
        indices = kwargs["conv_state_indices"]
        assert indices.min() > 0
        assert indices.max() < state.size(0)
        state.fill_(-1)
        calls.append((x, kwargs))
        return x

    monkeypatch.setattr(causal_conv1d, "causal_conv1d_update", update)
    _warm_causal_conv1d_update_kernel(
        torch.device("cpu"), config, num_speculative_tokens
    )
    torch.testing.assert_close(config.conv_state, before)
    ordinary = [(x, kw) for x, kw in calls if "query_start_loc" not in kw]
    assert {x.size(0) for x, _ in ordinary} == {1, 4, 16}
    assert {x.stride(0) for x, _ in ordinary} == {
        config.conv_dim,
        config.conv_dim + config.hv * config.v,
    }
    speculative = [(x, kw) for x, kw in calls if "query_start_loc" in kw]
    if not num_speculative_tokens:
        assert not speculative
    else:
        assert speculative
        assert {kw["conv_state_indices"].stride(0) for _, kw in speculative} == {4}
        lengths = [kw["query_start_loc"].diff().tolist() for _, kw in speculative]
        assert [4, 4, 4, 4] in lengths
        assert [1, 2, 3, 4] in lengths
        assert all(kw["max_query_len"] == 4 for _, kw in speculative)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA is required")
@pytest.mark.parametrize("transpose_state", [False, True])
def test_update_warmup_compiles_without_mutating_live_cache(transpose_state):
    from dataclasses import replace

    config = _cuda_gdn_config(state_tokens=6)
    if transpose_state:
        config = replace(
            config,
            conv_state=config.conv_state.transpose(1, 2).contiguous().transpose(1, 2),
        )
    config.conv_state.fill_(42)
    before = config.conv_state.clone()
    _warm_causal_conv1d_update_kernel(torch.device("cuda"), config, 3)
    torch.accelerator.synchronize()
    torch.testing.assert_close(config.conv_state, before)
