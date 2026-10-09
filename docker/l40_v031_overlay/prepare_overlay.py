# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Package a hash-checked Python overlay against official v0.31.0."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from apply_overlay import MODULES

BASE = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
RECIPE = Path(__file__).resolve().parent
REPO = RECIPE.parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(destination: Path) -> Path:
    if git("rev-parse", "v0.31.0").decode().strip() != BASE:
        raise ValueError("Unexpected official v0.31.0 base")
    destination.mkdir(parents=True, exist_ok=True)
    base_rows, payload_rows, tracked = [], [], []
    patch = b""
    for relative in MODULES:
        source = REPO / "vllm" / relative
        payload = destination / "vllm" / relative
        payload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, payload)
        original = subprocess.run(
            ["git", "show", f"{BASE}:vllm/{relative}"],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        if original.returncode == 0:
            base_rows.append(f"{sha(original.stdout)}  {relative}\n")
            tracked.append(f"vllm/{relative}")
        else:
            base_rows.append(f"MISSING  {relative}\n")
            new = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", f"vllm/{relative}"],
                cwd=REPO,
                capture_output=True,
                check=False,
            )
            if new.returncode != 1 or new.stderr:
                raise RuntimeError(f"Cannot create patch for {relative}")
            patch += new.stdout
        payload_rows.append(f"{sha(payload.read_bytes())}  {relative}\n")
    patch = git("diff", BASE, "--", *tracked) + patch
    (destination / "vllm-v031-l40.patch").write_bytes(patch)
    (destination / "BASE_SHA256SUMS").write_text("".join(base_rows))
    (destination / "PAYLOAD_SHA256SUMS").write_text("".join(payload_rows))
    recipe_files = (
        "Dockerfile",
        "apply_overlay.py",
        "README.md",
        "collect_logs.sh",
        "VALIDATION.md",
    )
    for name in recipe_files:
        shutil.copyfile(RECIPE / name, destination / name)
    shutil.copyfile(
        REPO / "benchmarks/kernels/l40_fp8_selection_20261009.json",
        destination / "FP8_SELECTION.json",
    )
    (destination / "SOURCE.txt").write_text(
        f"Official base: {BASE}\n"
        f"Checkout HEAD: {git('rev-parse', 'HEAD').decode().strip()}\n"
        "PAYLOAD_SHA256SUMS identifies the actual overlaid source.\n"
    )
    # Identical first/second prompts test reuse; the third extends the same chat.
    paragraph = "A shared document describes a quiet town with a library and a park.\n"
    prompt = paragraph * 800 + "\nReply with exactly L40-031-CACHE-OK and nothing else."
    messages = [{"role": "user", "content": prompt}]
    for number in (1, 2, 3):
        request = {
            "model": "/model/Qwen/Qwen3.8-27B-FP8",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        (destination / f"prefix-check-{number}.json").write_text(
            json.dumps(request, ensure_ascii=False) + "\n"
        )
        if number == 2:
            messages = [
                *messages,
                {"role": "assistant", "content": "L40-031-CACHE-OK"},
                {
                    "role": "user",
                    "content": "Repeat the same exact marker, nothing else.",
                },
            ]
    # Ship only intended payloads, even if this directory has old test outputs.
    files = sorted(
        [
            *(destination / "vllm" / relative for relative in MODULES),
            *(destination / name for name in recipe_files),
            *(destination / f"prefix-check-{number}.json" for number in (1, 2, 3)),
            *(
                destination / name
                for name in (
                    "BASE_SHA256SUMS",
                    "PAYLOAD_SHA256SUMS",
                    "SOURCE.txt",
                    "vllm-v031-l40.patch",
                    "FP8_SELECTION.json",
                )
            ),
        ]
    )
    manifest = destination / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{sha(path.read_bytes())}  {path.relative_to(destination)}\n"
            for path in files
        )
    )
    files.append(manifest)
    archive = destination.with_name(destination.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"{destination.name}/{path.relative_to(destination)}")
    with tarfile.open(archive, "r:gz") as tar:
        for path in files:
            item = tar.extractfile(
                f"{destination.name}/{path.relative_to(destination)}"
            )
            if item is None or item.read() != path.read_bytes():
                raise ValueError(f"Archive verification failed: {path}")
    archive.with_name(archive.name + ".sha256").write_text(
        f"{sha(archive.read_bytes())}  {archive.name}\n"
    )
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(prepare(args.destination.resolve()))
