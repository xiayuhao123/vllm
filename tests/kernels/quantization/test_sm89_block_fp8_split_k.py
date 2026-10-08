# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _get_sm89_split_k_factor,
    _run_w8a8_triton_block_scaled_mm_split_k,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "M,N,K,block_size,include_8192,expected",
    [
        (1, 5120, 3072, [128, 128], False, 4),
        (4, 5120, 8704, [128, 128], False, 4),
        (4, 8192, 5120, [128, 128], False, None),
        (4, 8192, 5120, [128, 128], True, 4),
        (0, 5120, 3072, [128, 128], False, None),
        (32, 5120, 8704, [128, 128], False, 4),
        (33, 5120, 8704, [128, 128], False, None),
        (4, 7168, 8704, [128, 128], False, None),
        (4, 5120, 8704, [128, 64], False, None),
    ],
)
def test_sm89_split_k_selection(M, N, K, block_size, include_8192, expected):
    assert (
        _get_sm89_split_k_factor(
            M,
            N,
            K,
            block_size,
            device_capability=(8, 9),
            enabled=True,
            include_8192=include_8192,
            is_cuda=True,
            batch_invariant=False,
        )
        == expected
    )


def test_sm89_split_k_selection_preserves_batch_invariant_mode():
    assert (
        _get_sm89_split_k_factor(
            4,
            5120,
            8704,
            [128, 128],
            device_capability=(8, 9),
            enabled=True,
            is_cuda=True,
            batch_invariant=True,
        )
        is None
    )


def test_sm89_split_k_selection_rejects_non_matching_kernel_config():
    assert (
        _get_sm89_split_k_factor(
            4,
            5120,
            8704,
            [128, 128],
            config={"BLOCK_SIZE_K": 64, "BLOCK_SIZE_N": 128},
            device_capability=(8, 9),
            enabled=True,
            is_cuda=True,
            batch_invariant=False,
        )
        is None
    )


SM89_CUDA = current_platform.is_cuda() and current_platform.get_device_capability() == (
    8,
    9,
)


@pytest.mark.skipif(
    not SM89_CUDA, reason="The split-K kernel is implemented for NVIDIA SM89."
)
@pytest.mark.parametrize("M,N,K", [(1, 5120, 3072), (4, 5120, 8704), (4, 8192, 5120)])
@torch.inference_mode()
def test_sm89_split_k_matches_block_fp8_reference(M, N, K):
    torch.manual_seed(0)
    block_size = [128, 128]
    block_n, block_k = block_size
    fp8_dtype = current_platform.fp8_dtype()
    fp8_info = torch.finfo(fp8_dtype)
    k_tiles = K // block_k
    n_tiles = N // block_n

    A = ((torch.rand(M, K, device="cuda") - 0.5) * fp8_info.max).to(fp8_dtype)
    B = ((torch.rand(N, K, device="cuda") - 0.5) * fp8_info.max).to(fp8_dtype)
    As = torch.rand(M, k_tiles, device="cuda", dtype=torch.float32) * 1e-2
    Bs = torch.rand(n_tiles, k_tiles, device="cuda", dtype=torch.float32) * 1e-2
    output = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }

    _run_w8a8_triton_block_scaled_mm_split_k(
        A, B, output, As, Bs, M, N, block_n, block_k, config, split_k=4
    )
    reference = native_w8a8_block_matmul(A, B, As, Bs, block_size, torch.bfloat16)

    relative_error = torch.mean(
        torch.abs(output.float() - reference.float())
    ) / torch.mean(torch.abs(reference.float()))
    assert relative_error < 0.001
