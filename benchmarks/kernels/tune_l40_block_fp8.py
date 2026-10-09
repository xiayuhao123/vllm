# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and measure small-M block-FP8 candidates on an idle L40."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
from importlib.metadata import version
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}
SHAPES = {
    "gdn": (5120, 3072),
    "mlp": (5120, 8704),
    "n8192": (8192, 5120),
    "n17408": (17408, 5120),
    "n7168": (7168, 5120),
}
SEEDS = Path(__file__).with_name("l40_fp8_seeds")


def candidate_configs(M: int, seeds: dict, exhaustive: bool = False) -> list[dict]:
    configs = [DEFAULT_CONFIG.copy()]
    if seeds:
        closest = min(seeds, key=lambda key: abs(int(key) - M))
        configs.append(seeds[closest].copy())
    space = itertools.product(
        [16, 32, 64] if exhaustive else [16, 32],
        [32, 64, 128],
        [1, 16, 32, 64] if exhaustive else [1, 32],
        [4, 8],
        [2, 3, 4, 5] if exhaustive else [2, 3],
    )
    configs.extend(
        dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=gm,
            num_warps=nw,
            num_stages=ns,
        )
        for bm, bn, gm, nw, ns in space
    )
    unique = {}
    for config in configs:
        if set(config) != set(DEFAULT_CONFIG) or config["BLOCK_SIZE_K"] != 128:
            raise ValueError("Candidate must preserve the 128-wide quantization block")
        unique[json.dumps(config, sort_keys=True)] = config
    return list(unique.values())


def best_candidate(rows: list[dict], split_k: int) -> dict:
    valid = [
        row
        for row in rows
        if row["split_k"] == split_k
        and row["status"] == "ok"
        and row["timings_us"]["cold"]
        and all(math.isfinite(x) and x > 0 for x in row["timings_us"]["cold"])
    ]
    if not valid:
        raise ValueError(f"No valid split_k={split_k} candidate")
    return min(valid, key=lambda row: statistics.median(row["timings_us"]["cold"]))


def export_normal_configs(
    rows: list[dict], destination: Path, device_name: str
) -> None:
    grouped: dict[tuple[int, int], dict] = {}
    anchors = [*range(1, 34), 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for row in rows:
        configs = grouped.setdefault(
            (row["N"], row["K"]), {str(m): DEFAULT_CONFIG.copy() for m in anchors}
        )
        configs[str(row["M"])] = best_candidate(row["candidates"], 1)["config"]
    destination.mkdir(parents=True, exist_ok=True)
    for (N, K), configs in grouped.items():
        name = (
            f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
            "block_shape=[128,128].json"
        )
        (destination / name).write_text(json.dumps(configs, indent=2) + "\n")


def time_replays(run, flush, repeats: int, event_factory) -> list[float]:
    """Flush on the same stream before recording; exclude flush latency."""
    start, end = event_factory(), event_factory()
    samples = []
    for _ in range(repeats):
        if flush is not None:
            flush()
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return samples


def fp32_reference(A, B, As, Bs):
    """Dequantize independently, then multiply in FP32 and round to BF16."""
    import torch

    af = A.float() * As.repeat_interleave(128, dim=1)
    bf = B.float() * Bs.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (af @ bf.t()).to(torch.bfloat16)


def measure_case(M, N, K, configs, args, torch, fp8, triton) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    dtype = torch.float8_e4m3fn
    A = ((torch.rand(M, K, device="cuda", generator=generator) - 0.5) * 448).to(dtype)
    B = ((torch.rand(N, K, device="cuda", generator=generator) - 0.5) * 448).to(dtype)
    As = torch.rand(M, K // 128, device="cuda", generator=generator) * 0.01 + 0.001
    Bs = (
        torch.rand(N // 128, K // 128, device="cuda", generator=generator) * 0.01
        + 0.001
    )
    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    reference = fp32_reference(A, B, As, Bs).float()
    trials = []
    for config in configs:
        trials.append((config, 1))
        factor = fp8._get_sm89_split_k_factor(
            M,
            N,
            K,
            [128, 128],
            config=config,
            enabled=True,
            include_8192=True,
            device_capability=(8, 9),
            is_cuda=True,
            batch_invariant=False,
        )
        if factor == 4:
            trials.append((config, factor))
    random.Random(args.seed + M + N + K).shuffle(trials)
    flush_buffer = torch.empty(
        args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8
    )
    records = []
    for config, split_k in trials:
        row: dict[str, Any] = {"config": config, "split_k": split_k}
        try:

            def run(config=config, split_k=split_k):
                if split_k == 4:
                    fp8._run_w8a8_triton_block_scaled_mm_split_k(
                        A, B, C, As, Bs, M, N, 128, 128, config, 4
                    )
                else:
                    grid = (
                        triton.cdiv(M, config["BLOCK_SIZE_M"])
                        * triton.cdiv(N, config["BLOCK_SIZE_N"]),
                    )
                    fp8._w8a8_triton_block_scaled_mm[grid](
                        A,
                        B,
                        C,
                        As,
                        Bs,
                        M,
                        N,
                        K,
                        128,
                        128,
                        A.stride(0),
                        A.stride(1),
                        B.stride(1),
                        B.stride(0),
                        C.stride(0),
                        C.stride(1),
                        As.stride(0),
                        As.stride(1),
                        Bs.stride(1),
                        Bs.stride(0),
                        **config,
                    )
                return C

            run()
            torch.accelerator.synchronize()
            error = (C.float() - reference).abs()
            mean_error = (error.mean() / reference.abs().mean().clamp_min(1e-9)).item()
            max_error = (error.max() / reference.abs().max().clamp_min(1e-9)).item()
            row.update(relative_mean_error=mean_error, normalized_max_error=max_error)
            if (
                not torch.isfinite(C).all().item()
                or mean_error >= 0.001
                or max_error >= 0.01
            ):
                row["status"] = "numerical_failure"
                records.append(row)
                continue
            for _ in range(5):
                run()
            torch.accelerator.synchronize()
            eager_output = C.clone()
            graph = torch.cuda.CUDAGraph()
            # Capture the complete chain, including Split-K scratch allocation
            # and reduction. Only replay kernels are timed below.
            with torch.cuda.graph(graph):
                run()
            C.fill_(float("nan"))
            for _ in range(5):
                graph.replay()
            torch.accelerator.synchronize()
            if not torch.equal(C, eager_output):
                row.update(status="graph_replay_failure")
                records.append(row)
                del graph
                continue

            def factory():
                return torch.Event(enable_timing=True)

            samples: dict[str, list[float]] = {"hot": [], "cold": []}
            for round_idx in range(args.rounds):
                order = ("hot", "cold") if round_idx % 2 == 0 else ("cold", "hot")
                for mode in order:
                    if mode == "hot":
                        # A preceding cold round may have evicted the weights.
                        for _ in range(5):
                            graph.replay()
                    samples[mode].extend(
                        time_replays(
                            graph.replay,
                            flush_buffer.zero_ if mode == "cold" else None,
                            args.repeats,
                            factory,
                        )
                    )
            row.update(status="ok", timings_us=samples)
            del graph
        except (
            triton.runtime.OutOfResources,
            triton.compiler.errors.CompilationError,
        ) as exc:
            row.update(status="compile_failure", error=str(exc))
        records.append(row)
    normal = best_candidate(records, 1)
    best = normal
    if any(r["split_k"] == 4 and r["status"] == "ok" for r in records):
        split = best_candidate(records, 4)
        best = min(
            (normal, split),
            key=lambda row: statistics.median(row["timings_us"]["cold"]),
        )
    deployed_factor = 4 if (N, K) in (SHAPES["gdn"], SHAPES["mlp"]) else 1
    baseline = next(
        (
            r
            for r in records
            if r["config"] == DEFAULT_CONFIG
            and r["split_k"] == deployed_factor
            and r["status"] == "ok"
        ),
        None,
    )
    if baseline is None:
        raise RuntimeError("Deployed baseline failed correctness or compilation")
    baseline_us = statistics.median(baseline["timings_us"]["cold"])
    best_us = statistics.median(best["timings_us"]["cold"])
    print(
        f"M={M} N={N} K={K}: baseline={baseline_us:.2f} us, "
        f"best={best_us:.2f} us, split_k={best['split_k']}, "
        f"ratio={baseline_us / best_us:.3f}x",
        flush=True,
    )
    return {
        "M": M,
        "N": N,
        "K": K,
        "candidates": records,
        "deployed_baseline": baseline,
        "cold_winner": best,
        "cold_ratio_vs_deployed": baseline_us / best_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--flush-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("l40-fp8-results"))
    args = parser.parse_args()
    if any(not 1 <= m <= 32 for m in args.m) or min(args.repeats, args.rounds) < 1:
        parser.error("M must be 1..32, and repeat/round counts must be positive")
    if args.flush_mib < 128:
        parser.error("Use at least 128 MiB for the L40 L2 flush buffer")
    plan = []
    for name, M in itertools.product(dict.fromkeys(args.shapes), sorted(set(args.m))):
        N, K = SHAPES[name]
        seed_path = SEEDS / (
            f"N={N},K={K},device_name=NVIDIA_L40S,dtype=fp8_w8a8,"
            "block_shape=[128,128].json"
        )
        seeds = json.loads(seed_path.read_text()) if seed_path.exists() else {}
        plan.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "configs": candidate_configs(M, seeds, args.exhaustive),
                "seed_file": seed_path.name if seeds else None,
                "seed_sha256": (
                    hashlib.sha256(seed_path.read_bytes()).hexdigest()
                    if seeds
                    else None
                ),
            }
        )
    if args.plan:
        print(json.dumps({"candidate_only": True, "cases": plan}, indent=2))
        return
    import torch
    import triton

    from vllm.model_executor.layers.quantization.utils import fp8_utils as fp8
    from vllm.platforms import current_platform

    torch.accelerator.set_device_index(args.device)
    if (
        not current_platform.is_cuda()
        or current_platform.get_device_capability(args.device) != (8, 9)
        or current_platform.get_device_name(args.device) != "NVIDIA L40"
    ):
        raise RuntimeError("This tuning run requires the actual NVIDIA L40, SM89")
    if version("vllm") != "0.31.0" or not hasattr(fp8, "_get_sm89_split_k_factor"):
        raise RuntimeError("Use the existing v0.31.0-l40-mtpgraph1 image")
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "candidate_only": True,
        "device_name": "NVIDIA_L40",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "vllm": version("vllm"),
        "flashinfer": version("flashinfer-python"),
        "kernel_source_sha256": hashlib.sha256(
            Path(fp8.__file__).read_bytes()
        ).hexdigest(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "timing": (
            "CUDA graph full-chain replay; flush before start event; microseconds"
        ),
        "baseline": "existing deployment: default tile with primary-shape Split-K=4",
        "cases": [],
    }
    for case in plan:
        measured = measure_case(
            case["M"],
            case["N"],
            case["K"],
            case["configs"],
            args,
            torch,
            fp8,
            triton,
        )
        measured.update(seed_file=case["seed_file"], seed_sha256=case["seed_sha256"])
        report["cases"].append(measured)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        export_normal_configs(
            report["cases"], args.output / "candidate_configs", "NVIDIA_L40"
        )
    print(f"Saved {args.output}/report.json. Candidate configs are not installed.")


if __name__ == "__main__":
    main()
