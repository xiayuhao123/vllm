# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact small-M candidates measured on L40; no nearest-M extrapolation."""

# Cold-L2 winners from the 2026-10-09 report; all also improve the deployed
# hot-cache baseline. Values: M tile, N tile, M group, warps, stages.
# Report SHA256: c4f299a48a97e5882a3d37877c54fbfa7e8e2030742fdcc7e75a1d5cb16d8d7e
_CONFIGS = {
    (1, 5120, 3072): (16, 64, 64, 4, 5),
    (4, 5120, 3072): (16, 64, 1, 4, 5),
    (8, 5120, 3072): (16, 32, 1, 4, 3),
    (16, 5120, 3072): (16, 32, 1, 4, 5),
    (32, 5120, 3072): (16, 64, 1, 8, 4),
    (1, 5120, 8704): (16, 64, 1, 4, 3),
    (4, 5120, 8704): (16, 32, 1, 8, 5),
    (8, 5120, 8704): (16, 32, 1, 4, 5),
    (16, 5120, 8704): (16, 64, 64, 4, 5),
    (32, 5120, 8704): (32, 32, 32, 4, 5),
    (1, 8192, 5120): (16, 64, 64, 4, 5),
    (4, 8192, 5120): (16, 32, 64, 8, 5),
    (8, 8192, 5120): (16, 32, 1, 4, 3),
    (16, 8192, 5120): (16, 32, 64, 4, 5),
    (32, 8192, 5120): (16, 32, 16, 4, 5),
    (1, 17408, 5120): (16, 64, 64, 4, 3),
    (4, 17408, 5120): (16, 64, 64, 4, 3),
    (8, 17408, 5120): (16, 64, 16, 4, 3),
    (16, 17408, 5120): (16, 32, 32, 4, 3),
    (32, 17408, 5120): (32, 64, 1, 4, 3),
    (1, 7168, 5120): (16, 64, 32, 4, 5),
    (4, 7168, 5120): (16, 32, 64, 4, 5),
    (8, 7168, 5120): (16, 32, 64, 4, 5),
    (16, 7168, 5120): (16, 32, 1, 4, 5),
    (32, 7168, 5120): (16, 64, 32, 4, 5),
}


def get_l40_fp8_tuned_config(
    M: int,
    N: int,
    K: int,
    block_size: list[int],
    *,
    enabled: bool,
    is_cuda: bool,
    capability: tuple[int, int] | None,
    device_name: str,
    batch_invariant: bool,
    compatible_dtype: bool,
) -> dict[str, int] | None:
    if (
        not enabled
        or not is_cuda
        or capability != (8, 9)
        or device_name != "NVIDIA_L40"
        or batch_invariant
        or not compatible_dtype
        or block_size != [128, 128]
    ):
        return None
    values = _CONFIGS.get((M, N, K))
    if values is None:
        return None
    bm, bn, gm, warps, stages = values
    return dict(
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=gm,
        num_warps=warps,
        num_stages=stages,
    )
