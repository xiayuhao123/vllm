# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the opt-in SM89 block-FP8 split-K path with the baseline kernel.

Run this on an L40 with the vLLM build that contains the change:

    python benchmarks/kernels/benchmark_l40_sm89_block_fp8_split_k.py --m 4
"""

import argparse
import os
from importlib.metadata import version

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _get_sm89_split_k_factor,
    get_w8a8_block_fp8_configs,
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform


def _make_inputs(M: int, N: int, K: int):
    block_n = block_k = 128
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    A = ((torch.rand(M, K, device="cuda") - 0.5) * fp8_max).to(fp8_dtype)
    B = ((torch.rand(N, K, device="cuda") - 0.5) * fp8_max).to(fp8_dtype)
    As = torch.rand(M, K // block_k, device="cuda", dtype=torch.float32) * 1e-2
    Bs = (
        torch.rand(N // block_n, K // block_k, device="cuda", dtype=torch.float32)
        * 1e-2
    )
    return A, B, As, Bs


def _measure(run, repeats: int) -> float:
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--shape", choices=("mlp", "gdn", "n8192"), default="mlp")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == (8, 9)
    ):
        raise RuntimeError("This benchmark requires an NVIDIA SM89 GPU such as L40.")
    if not 1 <= args.m <= 32:
        raise ValueError("--m must be in [1, 32] to exercise the split-K path.")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")

    shapes = {"mlp": (5120, 8704), "gdn": (5120, 3072), "n8192": (8192, 5120)}
    N, K = shapes[args.shape]
    A, B, As, Bs = _make_inputs(args.m, N, K)

    def run():
        return w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128], torch.bfloat16)

    master_key = "VLLM_ENABLE_SM89_FP8_SPLIT_K"
    extra_key = "VLLM_ENABLE_SM89_FP8_SPLIT_K_N8192_K5120"
    old_master = os.environ.get(master_key)
    old_extra = os.environ.get(extra_key)

    def select_split(enabled: bool):
        os.environ[master_key] = "1" if enabled else "0"
        os.environ[extra_key] = "1" if enabled and args.shape == "n8192" else "0"
        envs.disable_envs_cache()

    try:
        select_split(False)
        baseline = run()

        select_split(True)
        configs = get_w8a8_block_fp8_configs(N, K, 128, 128)
        config = (
            configs[min(configs, key=lambda m: abs(m - args.m))]
            if configs
            else {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}
        )
        factor = _get_sm89_split_k_factor(args.m, N, K, [128, 128], config=config)
        if factor != 4:
            raise RuntimeError("The selected config does not activate split-K.")
        split_output = run()
        relative_error = (
            (baseline.float() - split_output.float()).abs().mean()
            / baseline.float().abs().mean()
        ).item()
        print(f"vLLM {version('vllm')} from {fp8_utils.__file__}")
        print(f"shape=M{args.m}xK{K}, N={N}")
        print(f"relative mean output difference: {relative_error:.6f}")
        if relative_error >= 0.001:
            raise RuntimeError("Split-K output differs from the baseline.")

        if args.check_only:
            return
        select_split(False)
        baseline_ms = _measure(run, args.repeats)
        select_split(True)
        split_k_ms = _measure(run, args.repeats)
    finally:
        for key, old_value in ((master_key, old_master), (extra_key, old_extra)):
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        envs.disable_envs_cache()

    print(f"baseline: {baseline_ms:.3f} ms")
    print(f"split-K:  {split_k_ms:.3f} ms")
    print(f"speedup:  {baseline_ms / split_k_ms:.3f}x")


if __name__ == "__main__":
    main()
