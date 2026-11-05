"""Installed source/profile contracts for the canonical PG095 environment."""

import importlib
from importlib.metadata import distribution, version
from pathlib import Path

import pytest

from qwen_image_sliders.contracts import file_digest, lock
from qwen_image_sliders.core_profile import require_current_core


def test_installed_shared_sources_match_the_entire_locked_cohort():
    pins = lock()
    assert version("particle-sliders-core") == pins["core"]["version"] == "0.13.0"
    assert version("particlegan") == pins["particlegan"]["version"] == "0.9.5"
    core = distribution("particle-sliders-core")
    for relative, expected in pins["core"]["source_sha256"].items():
        assert file_digest(core.locate_file(relative)) == expected, relative
    assert len(pins["core"]["source_sha256"]) == 577
    assert pins["particlegan"]["source_profile"] == "release-0.9.5"
    require_current_core()


def test_source_guard_refuses_changed_and_extra_core_data_before_native_use(tmp_path, monkeypatch):
    import shutil

    from qwen_image_sliders import core_profile

    core = distribution("particle-sliders-core")
    for package in ("particle_sliders", "concept_slider_core"):
        shutil.copytree(
            core.locate_file(package), tmp_path / package, ignore=shutil.ignore_patterns("__pycache__")
        )

    class CopiedCore:
        def __getattr__(self, name):
            return getattr(core, name)

        def locate_file(self, relative):
            return tmp_path / relative

    original_distribution = core_profile.distribution
    monkeypatch.setattr(
        core_profile,
        "distribution",
        lambda name: CopiedCore() if name == "particle-sliders-core" else original_distribution(name),
    )
    path = tmp_path / "particle_sliders/pg095_release.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(RuntimeError, match="shared source differs"):
        require_current_core()
    path.write_bytes(original)
    extra = path.with_name("_qwen_unqualified_resource.json")
    extra.write_text("{}\n")
    with pytest.raises(RuntimeError, match="source inventory differs"):
        require_current_core()
    extra.unlink()
    require_current_core()


def test_installed_public_entrypoints_resolve_to_product_backends():
    package = distribution("qwen-image-particle-sliders")
    expected = {
        "particle_sliders.image_backends": ("qwen_image_sliders.lifecycle", "BACKEND"),
        "particle_sliders.inference_backends": ("qwen_image_sliders.inference", "INFERENCE_BACKEND"),
    }
    for group, (module, attribute) in expected.items():
        (entry,) = [entry for entry in package.entry_points if entry.group == group and entry.name == "qwen"]
        assert entry.load() is getattr(importlib.import_module(module), attribute)
    assert Path(importlib.import_module("qwen_image_sliders").__file__).is_file()


def test_modern_binding_refuses_supported_environment_before_model_access():
    # A stable wheel contains no modern runtime; editable development checkouts
    # may expose source files, whose admission still refuses the stable lane.
    if importlib.util.find_spec("qwen_image_sliders.hypergan_hndl") is not None:
        from qwen_image_sliders.hypergan_hndl import build_game

        with pytest.raises(RuntimeError, match="separate environment without the stable distribution"):
            build_game(None, [])
