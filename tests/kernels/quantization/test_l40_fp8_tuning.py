# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tuner():
    path = (
        Path(__file__).resolve().parents[3] / "benchmarks/kernels/tune_l40_block_fp8.py"
    )
    spec = importlib.util.spec_from_file_location("l40_tuner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_includes_deployed_default_and_preserves_quantization_blocks(tuner):
    configs = tuner.candidate_configs(4, {}, exhaustive=False)
    assert tuner.DEFAULT_CONFIG in configs
    assert any(c["BLOCK_SIZE_M"] == 16 and c["BLOCK_SIZE_N"] == 32 for c in configs)
    assert all(c["BLOCK_SIZE_K"] == 128 for c in configs)
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == len(configs)


def test_l40s_seed_is_a_candidate_without_relabelling_or_duplicating(tuner):
    seed = dict(tuner.DEFAULT_CONFIG, BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, num_stages=5)
    assert tuner.candidate_configs(4, {"4": seed}, False).count(seed) == 1


@pytest.mark.parametrize("bad_samples", [[float("nan")], [0], [-1], []])
def test_invalid_timing_or_wrong_results_cannot_win(tuner, bad_samples):
    valid = dict(
        config=tuner.DEFAULT_CONFIG,
        split_k=1,
        status="ok",
        timings_us={"cold": [10.0, 12.0], "hot": [8.0, 8.0]},
    )
    invalid = dict(
        valid, status="numerical_failure", timings_us={"cold": [1.0], "hot": [1.0]}
    )
    assert tuner.best_candidate([invalid, valid], 1) == valid
    bad_timing = dict(valid, timings_us={"cold": bad_samples, "hot": [1.0]})
    assert tuner.best_candidate([bad_timing, valid], 1) == valid
    with pytest.raises(ValueError):
        tuner.best_candidate([invalid], 1)


def test_config_export_leaves_untuned_and_prefill_sizes_on_default(tuner, tmp_path):
    config = dict(tuner.DEFAULT_CONFIG, BLOCK_SIZE_M=16, BLOCK_SIZE_N=32)
    rows = [
        dict(
            M=4,
            N=5120,
            K=8704,
            candidates=[
                dict(
                    config=config,
                    split_k=1,
                    status="ok",
                    timings_us={"cold": [5.0], "hot": [6.0]},
                )
            ],
        )
    ]
    tuner.export_normal_configs(rows, tmp_path, "NVIDIA_L40")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["4"] == config
    assert data["3"] == tuner.DEFAULT_CONFIG
    assert all(data[str(m)] == tuner.DEFAULT_CONFIG for m in (33, 64, 8192))
    assert all("split_k" not in item for item in data.values())


def test_cache_flush_happens_before_timed_region(tuner):
    trace = []

    class Event:
        def record(self):
            trace.append("event")

        def synchronize(self):
            trace.append("sync")

        def elapsed_time(self, other):
            return 0.005

    result = tuner.time_replays(
        lambda: trace.append("run"), lambda: trace.append("flush"), 2, Event
    )
    assert result == [5.0, 5.0]
    assert trace == ["flush", "event", "run", "event", "sync"] * 2


def test_plan_runs_without_cuda_or_importing_vllm(tuner):
    result = subprocess.run(
        [sys.executable, tuner.__file__, "--plan", "--m", "4", "--shapes", "gdn"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["candidate_only"] is True
    assert len(data["cases"]) == 1
    assert data["cases"][0]["M"] == 4
    assert tuner.DEFAULT_CONFIG in data["cases"][0]["configs"]


def test_fp32_reference_uses_both_quantization_scale_axes(tuner):
    import torch

    A = torch.ones((1, 256))
    B = torch.ones((256, 256))
    As = torch.tensor([[1.0, 2.0]])
    Bs = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    reference = tuner.fp32_reference(A, B, As, Bs)
    expected = torch.cat(
        [torch.full((1, 128), 1408.0), torch.full((1, 128), 2176.0)], dim=1
    ).to(torch.bfloat16)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
