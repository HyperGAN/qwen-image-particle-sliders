"""Origin and co-install admission checks for the distinct modern prerelease."""

import os
import shutil
import subprocess
import sys

import pytest
import torch

if os.environ.get("QWEN_TEST_CORE012") != "1":
    pytest.skip("separate modern prerelease environment required", allow_module_level=True)

from qwen_image_sliders.hypergan_hndl import QwenHNDLBank, _shared, build_game
from qwen_image_sliders.tiny import TinyRuntime


def test_loaded_core_module_origin_is_checked_not_only_distribution_hashes(tmp_path, monkeypatch):
    from particle_sliders import pg09

    monkeypatch.setattr(pg09, "__file__", str(tmp_path / "shadow.py"))
    with pytest.raises(RuntimeError, match="loaded module origin differs"):
        build_game(None, [])


@pytest.mark.parametrize("shadow", ["core", "pg", "stable_distribution"])
def test_actual_subprocess_shadow_or_coinstall_refuses_before_use(tmp_path, shadow):
    from importlib.metadata import distribution

    if shadow in ("core", "pg"):
        owner, name = (
            ("particle-sliders-core", "particle_sliders")
            if shadow == "core"
            else ("particlegan", "particlegan")
        )
        shutil.copytree(
            distribution(owner).locate_file(name),
            tmp_path / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    else:
        metadata = tmp_path / "qwen_image_particle_sliders-0.2.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: qwen-image-particle-sliders\nVersion: 0.2.0\n"
        )
    result = subprocess.run(
        [sys.executable, "-c", "from qwen_image_sliders.hypergan_hndl import _shared; _shared()"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert (
        "origin differs" if shadow in ("core", "pg") else "without the stable distribution"
    ) in result.stderr


def test_partial_attachment_failure_unwinds_owned_forwards(monkeypatch):
    from qwen_image_sliders import hypergan_hndl

    runtime = TinyRuntime()
    forwards = {name: module.forward for name, module in runtime.model.named_modules()}
    field, projection, builder, identity = _shared()
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected second projection setup failure")
        return builder(*args, **kwargs)

    monkeypatch.setattr(hypergan_hndl, "_shared", lambda: (field, projection, fail_second, identity))
    with pytest.raises(RuntimeError, match="second projection setup failure"):
        QwenHNDLBank(runtime.model, torch.nn.Parameter(torch.zeros(128, 4)))
    assert len(calls) == 2
    assert all(module.forward == forwards[name] for name, module in runtime.model.named_modules())
