# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install the Python overlay only on matching official v0.31.0 source files."""

import hashlib
import importlib.util
import py_compile
import shutil
from importlib.metadata import version
from pathlib import Path

MODULES = (
    "envs.py",
    "model_executor/layers/quantization/utils/fp8_utils.py",
    "v1/worker/gpu/model_runner.py",
    "model_executor/warmup/qwen_triton_warmup.py",
    "model_executor/warmup/sm89_warmup.py",
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
    "v1/worker/gpu/spec_decode/autoregressive/independent_cudagraph.py",
)


def _checksums(path: Path) -> dict[str, str]:
    rows = (line.split() for line in path.read_text().splitlines() if line.strip())
    result = {relative: digest for digest, relative in rows}
    if set(result) != set(MODULES):
        raise ValueError(f"Unexpected module list in {path.name}")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_overlay(installed: Path, package: Path) -> None:
    base = _checksums(package / "BASE_SHA256SUMS")
    patched = _checksums(package / "PAYLOAD_SHA256SUMS")
    # Validate every module before copying any of them.
    for relative in MODULES:
        payload = package / "vllm" / relative
        target = installed / relative
        if _sha256(payload) != patched[relative]:
            raise ValueError(f"Payload checksum mismatch: {relative}")
        compile(payload.read_bytes(), str(payload), "exec")
        current = _sha256(target) if target.exists() else "MISSING"
        if current not in (base[relative], patched[relative]):
            raise ValueError(
                f"Base source mismatch: {relative}. Expected the existing "
                "official v0.31.0 image; merge custom changes before applying."
            )
    for relative in MODULES:
        target = installed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / "vllm" / relative, target)
        py_compile.compile(str(target), doraise=True)
        print(f"Installed {relative}: {patched[relative]}")


def main() -> None:
    actual_version = version("vllm")
    if actual_version != "0.31.0":
        raise SystemExit(f"Expected vLLM 0.31.0, got {actual_version}")
    flashinfer_version = version("flashinfer-python")
    if flashinfer_version != "0.7.0.post1":
        raise SystemExit(
            "Expected official v0.31.0 / FlashInfer 0.7.0.post1, "
            f"got {flashinfer_version}"
        )
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Cannot locate installed vllm package")
    installed = Path(next(iter(spec.submodule_search_locations)))
    install_overlay(installed, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
