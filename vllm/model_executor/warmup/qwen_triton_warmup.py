# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen Triton kernels from the loaded model's compile keys."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.sm89_warmup import sm89_extra_warmup_enabled

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)

# Covers L=1 constexpr, non-divisible runtime L, and divisible runtime L.
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)


@dataclass(frozen=True)
class _QwenGDNWarmupConfig:
    h: int
    hv: int
    k: int
    v: int
    conv_kernel_size: int
    conv_state: torch.Tensor
    conv_dtype: torch.dtype
    norm_weight_dtype: torch.dtype
    norm_before_gate: bool
    norm_activation: str
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_stride_token: int
    state_dtype: torch.dtype
    conv_weight_dtype: torch.dtype | None = None

    @property
    def conv_dim(self) -> int:
        return 2 * self.h * self.k + self.hv * self.v


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _is_qwen_gdn_layer(module: object) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel_size",
            "tp_size",
            "kv_cache",
            "norm",
            "A_log",
            "dt_bias",
        )
    )


def _iter_qwen_gdn_layers(static_forward_context: object):
    if not isinstance(static_forward_context, dict):
        return

    for module in static_forward_context.values():
        if _is_qwen_gdn_layer(module):
            yield module


def _split_qwen_gdn_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        conv_cache, ssm_state = kv_cache[:2]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state

    if isinstance(kv_cache, torch.Tensor) and kv_cache.size(0) >= 2:
        conv_cache = kv_cache[0]
        ssm_state = kv_cache[1]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state
    return None


def _qwen_gdn_warmup_config(
    static_forward_context: object,
) -> _QwenGDNWarmupConfig | None:
    found_layer = False
    for layer in _iter_qwen_gdn_layers(static_forward_context):
        found_layer = True
        cache_tensors = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache_tensors is None:
            continue

        conv_cache, ssm_state = cache_tensors
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        tp_size = int(layer.tp_size)
        h = int(layer.num_k_heads) // tp_size
        hv = int(layer.num_v_heads) // tp_size
        norm = layer.norm
        conv_weight = getattr(getattr(layer, "conv1d", None), "weight", None)

        return _QwenGDNWarmupConfig(
            h=h,
            hv=hv,
            k=int(layer.head_k_dim),
            v=int(layer.head_v_dim),
            conv_kernel_size=int(layer.conv_kernel_size),
            conv_state=conv_state,
            conv_dtype=conv_state.dtype,
            norm_weight_dtype=norm.weight.dtype,
            norm_before_gate=bool(norm.norm_before_gate),
            norm_activation=str(norm.activation),
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_stride_token=int(ssm_state.stride(0)),
            state_dtype=ssm_state.dtype,
            conv_weight_dtype=(
                conv_weight.dtype if isinstance(conv_weight, torch.Tensor) else None
            ),
        )

    if found_layer:
        logger.info("Skipping Qwen GDN Triton warmup: no bound Qwen GDN cache found.")
    else:
        logger.info("Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.")
    return None


def _warm_gated_rms_norm_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
    max_num_tokens: int,
    x_dtype: torch.dtype,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        warmup_layer_norm_fwd,
    )

    warmup_layer_norm_fwd(
        max_num_tokens=max_num_tokens,
        rows_per_token=config.hv,
        group_size=config.v,
        x_dtype=x_dtype,
        weight_dtype=config.norm_weight_dtype,
        device=device,
        norm_before_gate=config.norm_before_gate,
        is_rms_norm=True,
        activation=config.norm_activation,
    )


def _warm_causal_conv1d_fwd_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    x_storage = torch.empty(
        (1, config.conv_dim), dtype=config.conv_dtype, device=device
    )
    x = x_storage.t()
    weight = torch.empty(
        (config.conv_dim, config.conv_kernel_size),
        dtype=config.conv_dtype,
        device=device,
    )
    cache_indices = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    causal_conv1d_fn(
        x,
        weight,
        None,
        config.conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        metadata=None,
        validate_data=False,
    )


def _warm_causal_conv1d_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
    num_speculative_tokens: int,
) -> None:
    """Compile ordinary and varlen Qwen updates without touching live states."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    # The kernels specialize all state strides. Clone/contiguous would lose the
    # transposed cache layout or page padding used by the loaded model.
    state = torch.empty_strided(
        (17, *config.conv_state.shape[1:]),
        config.conv_state.stride(),
        dtype=config.conv_dtype,
        device=device,
    )
    state.zero_()
    weight_dtype = config.conv_weight_dtype or config.norm_weight_dtype
    weight = torch.zeros(
        (config.conv_dim, config.conv_kernel_size),
        dtype=weight_dtype,
        device=device,
    )
    row_strides = (config.conv_dim, config.conv_dim + config.hv * config.v)
    launches = 0
    for num_reqs in (1, 4, 16):
        indices = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
        for row_stride in row_strides:
            x = torch.empty_strided(
                (num_reqs, config.conv_dim),
                (row_stride, 1),
                dtype=weight_dtype,
                device=device,
            )
            x.zero_()
            causal_conv1d_update(
                x,
                state,
                weight,
                None,
                "silu",
                conv_state_indices=indices,
                validate_data=False,
            )
            launches += 1

    max_query_len = num_speculative_tokens + 1
    if num_speculative_tokens > 0:
        required_state_len = config.conv_kernel_size - 1 + num_speculative_tokens
        if state.size(2) < required_state_len:
            logger.info(
                "Skipping speculative Qwen conv-update warmup: state length "
                "%d is below required %d.",
                state.size(2),
                required_state_len,
            )
        else:
            for num_reqs in (1, 4, 16):
                # Qwen uses the first column of the speculative state table.
                table = torch.ones(
                    (num_reqs, max_query_len), dtype=torch.int32, device=device
                )
                table[:, 0] = torch.arange(
                    1, num_reqs + 1, dtype=torch.int32, device=device
                )
                indices = table[:, 0]
                accepted = torch.ones(num_reqs, dtype=torch.int32, device=device)
                for variable_lengths in (False, True):
                    lengths = [
                        (i % max_query_len + 1) if variable_lengths else max_query_len
                        for i in range(num_reqs)
                    ]
                    offsets = [0]
                    for length in lengths:
                        offsets.append(offsets[-1] + length)
                    query_start_loc = torch.tensor(
                        offsets, dtype=torch.int32, device=device
                    )
                    for row_stride in row_strides:
                        x = torch.empty_strided(
                            (offsets[-1], config.conv_dim),
                            (row_stride, 1),
                            dtype=weight_dtype,
                            device=device,
                        )
                        x.zero_()
                        causal_conv1d_update(
                            x,
                            state,
                            weight,
                            None,
                            "silu",
                            conv_state_indices=indices,
                            num_accepted_tokens=accepted,
                            query_start_loc=query_start_loc,
                            max_query_len=max_query_len,
                            validate_data=False,
                        )
                        launches += 1
    logger.info(
        "Warmed Qwen causal-conv1d update: "
        "configured query length=%d, state strides=%s, launches=%d; "
        "scratch states only.",
        max_query_len,
        config.conv_state.stride(),
        launches,
    )


def _warm_fused_post_conv_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E501
        fused_post_conv_prep,
    )

    qkv_dim = 2 * config.h * config.k + config.hv * config.v
    for length in _FLA_POST_CONV_WARMUP_LENGTHS:
        conv_output = torch.empty(
            (length, qkv_dim), dtype=config.conv_dtype, device=device
        )
        a = torch.empty((length, config.hv), dtype=config.conv_dtype, device=device)
        b = torch.empty_like(a)

        fused_post_conv_prep(
            conv_output,
            a,
            b,
            config.a_log,
            config.dt_bias,
            config.h,
            config.k,
            config.v,
            apply_l2norm=True,
            output_g_exp=False,
        )


def _warm_fused_sigmoid_gating_delta_rule_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device)
    a = torch.empty((1, 1, config.hv), dtype=config.conv_dtype, device=device)
    b = torch.empty_like(a)
    state = torch.empty(
        (1, config.state_stride_token),
        dtype=config.state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
    ssm_state_indices.zero_()

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=a,
        b=b,
        dt_bias=config.dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_triton_warmup(
    runner: "GPUModelRunner",
    model_config: "ModelConfig",
) -> None:
    """Warm Qwen GDN Triton kernels reported by the JIT monitor."""
    model_type = getattr(model_config.hf_text_config, "model_type", "") or getattr(
        model_config.hf_config, "model_type", ""
    )
    if model_type not in _QWEN_MODEL_TYPES:
        return

    device = runner.device
    logger.info("Warming up Qwen GDN Triton kernels for model_type=%s.", model_type)

    gdn_config = _qwen_gdn_warmup_config(
        runner.compilation_config.static_forward_context
    )
    if gdn_config is None:
        return

    max_num_tokens = max(1, int(runner.max_num_tokens))
    _warm_gated_rms_norm_kernel(device, gdn_config, max_num_tokens, model_config.dtype)
    _warm_causal_conv1d_fwd_kernel(device, gdn_config)
    _warm_fused_post_conv_kernel(device, gdn_config)
    # Pooling only runs full prefills; the decode update kernel is unused.
    if not runner.is_pooling_model:
        if sm89_extra_warmup_enabled():
            spec = runner.speculative_config
            _warm_causal_conv1d_update_kernel(
                device,
                gdn_config,
                spec.num_speculative_tokens if spec is not None else 0,
            )
        _warm_fused_sigmoid_gating_delta_rule_update_kernel(device, gdn_config)
    _synchronize_device(device)
