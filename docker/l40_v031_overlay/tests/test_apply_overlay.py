# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def overlay_module():
    path = Path(__file__).resolve().parents[1] / "apply_overlay.py"
    spec = importlib.util.spec_from_file_location("apply_overlay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deployment(tmp_path, overlay_module):
    installed = tmp_path / "installed"
    package = tmp_path / "package"
    base_rows = []
    payload_rows = []
    for relative in overlay_module.MODULES:
        original = b"VALUE = 'original'\n"
        patched = b"VALUE = 'patched'\n"
        target = installed / relative
        payload = package / "vllm" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        payload.write_bytes(patched)
        base_rows.append(f"{hashlib.sha256(original).hexdigest()}  {relative}\n")
        payload_rows.append(f"{hashlib.sha256(patched).hexdigest()}  {relative}\n")
    (package / "BASE_SHA256SUMS").write_text("".join(base_rows))
    (package / "PAYLOAD_SHA256SUMS").write_text("".join(payload_rows))
    return installed, package


def test_matching_base_is_patched_and_reapplication_is_safe(overlay_module, deployment):
    installed, package = deployment
    overlay_module.install_overlay(installed, package)
    overlay_module.install_overlay(installed, package)
    for relative in overlay_module.MODULES:
        assert (installed / relative).read_bytes() == b"VALUE = 'patched'\n"


def test_new_helper_can_be_installed_and_reapplied(overlay_module, deployment):
    installed, package = deployment
    relative = overlay_module.MODULES[-1]
    (installed / relative).unlink()
    rows = (package / "BASE_SHA256SUMS").read_text().splitlines()
    rows[-1] = f"MISSING  {relative}"
    (package / "BASE_SHA256SUMS").write_text("\n".join(rows) + "\n")
    overlay_module.install_overlay(installed, package)
    overlay_module.install_overlay(installed, package)
    assert (installed / relative).read_bytes() == b"VALUE = 'patched'\n"


@pytest.mark.parametrize("failure", ["base", "payload", "syntax"])
def test_invalid_last_module_does_not_modify_earlier_files(
    overlay_module, deployment, failure
):
    installed, package = deployment
    relative = overlay_module.MODULES[-1]
    if failure == "base":
        (installed / relative).write_bytes(b"VALUE = 'custom'\n")
    else:
        payload = package / "vllm" / relative
        payload.write_bytes(b"this is invalid Python!\n")
        if failure == "syntax":
            rows = (package / "PAYLOAD_SHA256SUMS").read_text().splitlines()
            rows[-1] = f"{hashlib.sha256(payload.read_bytes()).hexdigest()}  {relative}"
            (package / "PAYLOAD_SHA256SUMS").write_text("\n".join(rows) + "\n")
    with pytest.raises((ValueError, SyntaxError)):
        overlay_module.install_overlay(installed, package)
    for untouched in overlay_module.MODULES[:-1]:
        assert (installed / untouched).read_bytes() == b"VALUE = 'original'\n"


@pytest.mark.parametrize(
    "vllm_version,flashinfer_version",
    [("0.30.0", "0.7.0.post1"), ("0.31.0", "0.6.18.post1")],
)
def test_incompatible_binary_distribution_is_rejected_before_copying(
    overlay_module, monkeypatch, vllm_version, flashinfer_version
):
    versions = {"vllm": vllm_version, "flashinfer-python": flashinfer_version}
    monkeypatch.setattr(overlay_module, "version", versions.__getitem__)

    def unexpected_copy(*args):
        raise AssertionError("Must reject distribution mismatch before copying")

    monkeypatch.setattr(overlay_module, "install_overlay", unexpected_copy)
    with pytest.raises(SystemExit, match="Expected"):
        overlay_module.main()
