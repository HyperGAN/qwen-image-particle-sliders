"""Installed source/profile contracts for the supported S081 environment."""

import importlib
from importlib.metadata import distribution, version
from pathlib import Path

import pytest

from qwen_image_sliders.contracts import file_digest, lock


def test_installed_shared_sources_match_the_entire_locked_cohort():
    pins = lock()
    assert version("particle-sliders-core") == pins["core"]["version"] == "0.8.1"
    assert version("particlegan") == pins["particlegan"]["version"] == "0.9.1"
    core = distribution("particle-sliders-core")
    for relative, expected in pins["core"]["source_sha256"].items():
        assert file_digest(core.locate_file(relative)) == expected, relative


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
