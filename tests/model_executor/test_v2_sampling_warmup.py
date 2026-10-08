# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.platforms import current_platform
from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.worker.gpu.sample import sampler as gpu_sampler


class RecordingKernel:
    def __init__(self, name: str, compiled: list[str]):
        self.name = name
        self.compiled = compiled

    def register_warmup(self):
        JitWarmupRegistry.register(self)

    def get_warmup_keys(self):
        return [self.name]

    def compile(self, key):
        self.compiled.append(key)

    def compile_many(self, keys):
        for key in keys:
            self.compile(key)


@pytest.mark.parametrize("use_flashinfer", [True, False])
@pytest.mark.parametrize("cuda_alike", [True, False])
def test_v2_sampler_warms_native_fallbacks(
    monkeypatch: pytest.MonkeyPatch, use_flashinfer: bool, cuda_alike: bool
):
    """GPU buffers are irrelevant to registering the native fallback kernels."""
    for name in (
        "SamplingStates",
        "PenaltiesState",
        "LogitBiasState",
        "BadWordsState",
        "LogprobTokenIdsState",
        "ThinkingBudgetState",
    ):
        monkeypatch.setattr(gpu_sampler, name, lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        gpu_sampler, "flashinfer_sampler_supported", lambda: use_flashinfer
    )
    monkeypatch.setattr(topk_topp_sampler, "HAS_TRITON", True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: not cuda_alike)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: cuda_alike)

    compiled: list[str] = []
    names = ("_topk_topp", "_topp_split_stats", "_topp_split_step", "_topp_split_mask")
    for name in names:
        monkeypatch.setattr(
            topk_topp_sampler, name, RecordingKernel(name, compiled), raising=False
        )

    registry = JitWarmupRegistry(SimpleNamespace())
    with registry.activate():
        gpu_sampler.Sampler(4, 128, torch.device("cpu"), SimpleNamespace())
        gpu_sampler.Sampler(4, 128, torch.device("cpu"), SimpleNamespace())
    registry.warmup()

    assert compiled == (list(names) if cuda_alike else [])
